"""Match orchestration: QThread ownership, debounce, cancel, and stale-drop.

:class:`MatchController` is the single seam between the GUI (viewport / main
window, which speak ``QRect`` in image px) and the headless engine (which speaks
plain numpy + :class:`fastmatch.types.MatchParams`). It owns exactly one
long-lived :class:`~PySide6.QtCore.QThread` running one
:class:`fastmatch.worker.MatchWorker`, and enforces the resolved threading model
from DESIGN.md §F.1: **single in-flight job, latest-wins, 120 ms debounce.**

Request lifecycle (DESIGN.md §F.2):

1. :meth:`request` validates + clips the rect (rejecting too-small / whole-image
   / extreme-aspect selections), cancels any in-flight job, and (re)arms the
   debounce timer.
2. On debounce fire, :meth:`_dispatch` crops the template from the (read-only)
   document, bumps ``job_id``, announces busy, and emits the queued run signal.
3. The worker runs ``match`` off-thread and emits ``finished``/``error`` back.
4. :meth:`_on_finished` / :meth:`_on_error` drop stale results (``job_id !=``
   current) and otherwise surface them via :attr:`matches_ready` / :attr:`failed`.

Coordinate contract (DESIGN.md §F.3): all boxes here are half-open integer image
px, origin top-left, +x right / +y down. The viewport has already applied the
floor-TL / ceil-BR rounding rule when producing the ``QRect``; the controller
only clips to image bounds and crops.
"""

from __future__ import annotations

import numpy as np
import torch
from PySide6.QtCore import QObject, QRect, Qt, QThread, QTimer, Signal, Slot

from .device import resolve_device
from .document import ImageDocument
from .engine import Matcher
from .types import Match, MatchParams
from .worker import MatchWorker

# Selection-validation thresholds (DESIGN.md §H / §H3). Kept module-level so the
# rejection rules are visible and testable in one place.
_MIN_SIDE_PX = 8          # reject templates with any side < this
_MAX_ASPECT = 20.0        # reject extreme slivers (aspect ratio > this)
_DEBOUNCE_MS = 120        # collapse a burst of selection drags into one search
_THREAD_WAIT_MS = 8000    # bound shutdown's join so a wedged worker can't hang exit
                          # (headroom for one in-flight CPU compute tile to finish
                          #  and observe the stop flag between tiles)


class MatchController(QObject):
    """Owns the match thread; turns viewport rects into engine jobs.

    Signals:
        matches_ready: ``object`` — ``list[Match]`` for the latest accepted
            job. Wired to the viewport's ``set_matches``.
        busy_changed: ``bool`` — True when a job is dispatched, False when it
            finishes / fails / is superseded.
        progress: ``int`` in ``0..100`` — forwarded from the worker.
        failed: ``str`` — a human-readable reason for rejection or engine error.
    """

    matches_ready = Signal(object)  # list[Match]
    busy_changed = Signal(bool)
    progress = Signal(int)          # 0..100, forwarded from worker
    failed = Signal(str)

    # Private queued bridge to the worker's run() slot. Declared as a signal
    # (not a direct call) so the cross-thread invocation is marshalled onto the
    # worker thread's event loop. Payload is plain CPU data only.
    _run_requested = Signal(object, object, object, object, int)  # (template, params, exclude_box, mask, job_id)

    # Private queued bridge to staging. Emitted from the GUI thread in __init__;
    # the connected slot (_prepare_engine) runs on the *worker* thread so the
    # CUDA context init + the multi-GB set_image happen off the GUI thread
    # (DESIGN.md §C.14). Payload is the initial params (plain CPU data).
    _prepare_requested = Signal(object)  # (params0,)

    # Private readiness signal, emitted from _prepare_engine on the worker
    # thread. Auto-connection marshals _on_ready back onto the GUI thread so the
    # _staged flag / debounce timer are only ever touched on the GUI thread.
    _ready = Signal()

    def __init__(
        self,
        doc: ImageDocument,
        params0: MatchParams,
        *,
        engine: Matcher | None = None,
    ) -> None:
        """Build the engine, stage the image, and start the worker thread.

        Args:
            doc: The loaded image document. ``doc.full`` is treated as
                read-only and is the source the engine stages and templates are
                cropped from.
            params0: Initial parameters; only ``params0.device`` is consulted
                here (to map the UI device preference onto ``Matcher``
                construction). Per-request params arrive via :meth:`request`.
            engine: Testing / dependency-injection hook. When ``None`` (the
                normal path) a :class:`Matcher` is built from ``params0`` *on the
                worker thread* and ``doc.full`` is staged there (so the CUDA
                context and the multi-GB upload never touch the GUI thread —
                DESIGN.md §C.14). When provided, it is used as-is and assumed to
                already be image-staged — this lets tests inject a fake engine
                without a real GPU or large upload (treated as already staged).
        """
        super().__init__()
        self._doc = doc

        # --- job bookkeeping --------------------------------------------------
        self._current_job_id = 0          # monotonic; latest accepted job
        self._busy = False
        # Pending dispatch parameters, captured by request() and consumed when
        # the debounce timer fires (or when staging completes, if a request
        # raced ahead of it). None means "nothing armed".
        self._pending: tuple[QRect, MatchParams, object] | None = None
        # Staging gate: the worker thread builds the engine + stages the image
        # asynchronously, so _dispatch must not emit _run_requested until that
        # has completed. A request arriving first stays armed in _pending and is
        # dispatched on _on_ready. An injected engine is treated as already
        # staged (the DI/test path skips deferred construction).
        self._staged = False
        # Teardown guard: shutdown() flips this True so stale progress/result
        # deliveries to a being-torn-down controller are dropped (§C.15), and so
        # shutdown() itself is idempotent.
        self._stopped = False

        # --- threading: one persistent QThread owning one worker -------------
        self._thread = QThread()
        self._thread.setObjectName("fastmatch-worker")
        if engine is None:
            # Deferred construction: the worker is created without an engine and
            # _prepare_engine (queued onto the worker thread) builds + stages it.
            self._engine = None
            self._worker = MatchWorker(None)  # engine assigned by _prepare_engine
        else:
            # Injected engine: already constructed + image-staged. Treat as ready
            # so the first request dispatches immediately (DI/test path).
            self._engine = engine
            self._worker = MatchWorker(self._engine)
            self._staged = True
        self._worker.moveToThread(self._thread)

        # Worker results come back on the worker thread; Qt auto-connection
        # marshals them to this (GUI) thread's event loop because sender and
        # receiver live on different threads.
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.progress)  # straight forward 0..100

        # When the thread's event loop exits (after shutdown's quit()/wait()),
        # reclaim the worker and the thread on the GUI thread. This also reclaims
        # a *parked* (wedged) controller once its in-flight job finally returns
        # and the loop quits — see app.py's orphaned-controller parking.
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        # Dispatch bridge: queued so run() executes on the worker thread even
        # though we emit from the GUI thread inside _dispatch.
        self._run_requested.connect(self._worker.run, Qt.ConnectionType.QueuedConnection)
        # Staging bridge: queued so _prepare_engine runs on the worker thread.
        self._prepare_requested.connect(
            self._prepare_engine, Qt.ConnectionType.QueuedConnection
        )
        # Readiness comes back from the worker thread; auto-connection marshals
        # _on_ready onto this (GUI) thread.
        self._ready.connect(self._on_ready)

        self._thread.start()

        # Single-shot debounce timer. Re-arming (start()) while already running
        # restarts the countdown, collapsing a burst of drags into one search.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._dispatch)

        # Kick off deferred staging on the worker thread (no-op when an engine
        # was injected — that path is already marked staged above).
        if engine is None:
            self._prepare_requested.emit(params0)

    # ------------------------------------------------------------------ staging
    @Slot(object)
    def _prepare_engine(self, params0: MatchParams) -> None:
        """Build the engine and stage the image — runs on the worker thread.

        Invoked via a queued connection from :meth:`__init__`, so the expensive
        CUDA-context init (inside ``resolve_device``'s canary) and the multi-GB
        ``set_image`` upload happen off the GUI thread (DESIGN.md §C.14). On
        completion it hands the engine to the worker and signals readiness via
        :meth:`_on_ready` so any request that raced ahead can dispatch.
        """
        if self._stopped:
            return  # torn down before staging even started; nothing to build
        # Map the UI's device preference ("auto"/"cuda"/"cpu") onto the engine.
        # resolve_device is canary-gated and falls back to CPU, so this never
        # hard-fails on a GPU-less / mismatched-wheel machine. compute_dtype is
        # mapped like device (§C.11); per-call compute_dtype stays advisory.
        dev = resolve_device(params0.device)
        engine = Matcher(device=dev.type, compute_dtype=params0.compute_dtype)
        engine.set_image(self._doc.full)
        # Hand the staged engine to the worker (this slot already runs on the
        # worker thread, so the worker owns it on its own thread) and announce
        # readiness; the _ready signal marshals _on_ready onto the GUI thread.
        self._engine = engine
        self._worker._engine = engine
        self._ready.emit()

    @Slot()
    def _on_ready(self) -> None:
        """Mark staging complete and dispatch any request that arrived first.

        Auto-connection marshals this onto the GUI thread (it is emitted from
        :meth:`_prepare_engine` on the worker thread), so the ``_staged`` flag
        and the debounce timer are touched only on the GUI thread.
        """
        if self._stopped:
            return
        self._staged = True
        # A request may have armed _pending before staging finished. Dispatch it
        # now (directly, not via the debounce) so the first search is not lost.
        # Stop the debounce first: it may still be counting down, and a stray
        # later fire would otherwise leave isActive() True, which _is_superseded
        # treats as "a newer request is pending" and would drop this result.
        if self._pending is not None:
            self._debounce.stop()
            self._dispatch()

    # ------------------------------------------------------------------ request
    @Slot(QRect, object)
    def request(
        self, rect_img: QRect, params: MatchParams, mask: object = None
    ) -> None:
        """Validate a selection and (re)arm a debounced search.

        Args:
            rect_img: Selection in half-open image px (the viewport already
                applied floor-TL / ceil-BR rounding).
            params: Parameters for this search.
            mask: Optional bbox-sized boolean template mask (rectilinear select),
                aligned to ``rect_img``; cropped here if the rect clips the image.

        Invalid selections (after clipping to the image) emit :attr:`failed`
        with a reason and dispatch nothing. Valid ones cancel any in-flight job
        and arm the 120 ms debounce; the clipped rect is what gets cropped.
        """
        clipped = self._clip_to_image(rect_img)
        reason = self._rejection_reason(clipped)
        if reason is not None:
            self.failed.emit(reason)
            return

        # Keep the mask aligned to the (possibly edge-clipped) rect: crop it by the
        # same top-left/size shift the clip applied, so it matches the cropped crop.
        if mask is not None and clipped != rect_img:
            dx = clipped.x() - rect_img.x()
            dy = clipped.y() - rect_img.y()
            mask = mask[dy : dy + clipped.height(), dx : dx + clipped.width()]

        # A new valid selection supersedes whatever is running: cancel it so the
        # GPU/CPU stops wasting work, then debounce this one. Latest-wins is
        # ultimately enforced by job_id stale-drop, but cancelling early frees
        # the worker sooner for the request we actually care about.
        if self._busy:
            self._worker.request_cancel()

        self._pending = (clipped, params, mask)
        self._debounce.start()  # (re)arm; restarts countdown if already running

    # ----------------------------------------------------------------- dispatch
    @Slot()
    def _dispatch(self) -> None:
        """Crop the template and hand the job to the worker (debounce target)."""
        if self._stopped:
            return  # torn down; a queued debounce timeout must not dispatch
        if self._pending is None:
            return  # defensive: timer fired with nothing armed
        if not self._staged:
            # Staging (engine build + set_image) is still running on the worker
            # thread. Keep the request armed in _pending; _on_ready will dispatch
            # it once the engine is ready, so the first search is never lost.
            return
        rect, params, mask = self._pending
        self._pending = None

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        # ascontiguousarray: the engine wants a C-contiguous template, and a
        # numpy slice of a (possibly memmap) array is generally a non-owning,
        # non-contiguous view. This produces an owned plain ndarray safe to
        # ship across the (queued) signal to the worker thread.
        template = np.ascontiguousarray(self._doc.full[y : y + h, x : x + w])
        mask_arr = None if mask is None else np.ascontiguousarray(mask)

        self._current_job_id += 1
        self._set_busy(True)
        # exclude_box is the source region in full-image coords so the engine
        # can suppress the template's own location among the hits (§H4).
        self._run_requested.emit(
            template, params, (x, y, w, h), mask_arr, self._current_job_id
        )

    # ------------------------------------------------------------ result slots
    def _is_superseded(self, job_id: int) -> bool:
        """True if a result for ``job_id`` should be dropped as stale.

        Two ways a job is superseded:

        * Its id no longer matches the latest dispatched job — a newer job has
          already been sent to the worker.
        * A newer request is armed but not yet dispatched (``_pending`` set /
          debounce running). This is the subtle case: when ``request`` cancels
          an in-flight job, that job returns early (typically empty) and emits
          ``finished`` while it is *still* the current id, because the
          superseding job won't claim a new id until the debounce fires. We
          must not surface that aborted result, and we must stay busy until the
          pending job completes.
        """
        if job_id != self._current_job_id:
            return True
        return self._pending is not None or self._debounce.isActive()

    @Slot(object, int)
    def _on_finished(self, matches: object, job_id: int) -> None:
        """Surface a finished job's matches, dropping stale ones.

        Args:
            matches: ``list[Match]`` from the worker (plain CPU data).
            job_id: The job that produced ``matches``.
        """
        if self._stopped:
            return  # being torn down; drop late deliveries (§C.15)
        if self._is_superseded(job_id):
            return  # a newer request superseded this one; stay busy for it
        self._set_busy(False)
        self.matches_ready.emit(matches)

    @Slot(str, int)
    def _on_error(self, message: str, job_id: int) -> None:
        """Surface an engine error for the current job, dropping stale ones.

        Args:
            message: Formatted traceback string from the worker.
            job_id: The job that raised.
        """
        if self._stopped:
            return  # being torn down; drop late deliveries (§C.15)
        if self._is_superseded(job_id):
            return  # stale / superseded error; the pending job is what matters
        self._set_busy(False)
        self.failed.emit(message)

    # ---------------------------------------------------------------- shutdown
    def shutdown(self) -> bool:
        """Cancel any in-flight job and join the worker thread cleanly.

        Idempotent and safe to call from ``closeEvent`` (DESIGN.md §M5). The
        join is bounded by ``_THREAD_WAIT_MS`` so a pathologically wedged worker
        cannot hang application exit.

        Returns:
            ``True`` if the worker thread actually stopped (or was never
            running) and the engine refs were reclaimed; ``False`` if the join
            timed out with the thread still running. On ``False`` the caller
            MUST NOT drop or rebuild this controller — the live thread would be
            GC'd while running (SIGABRT). Park it instead (see app.py) and let
            ``thread.finished -> deleteLater`` reclaim it once the job returns.
        """
        # FIRST flip the teardown guard and disconnect the worker's signals so
        # any in-flight result/progress that returns during/after the join is
        # dropped rather than delivered to a being-torn-down controller (§C.15).
        # Disconnects are wrapped because a connection may already be gone (a
        # second shutdown() call) or the worker already deleted.
        self._stopped = True
        self._debounce.stop()
        if self._thread is None:
            # Already fully torn down by a prior successful shutdown(); idempotent.
            return True
        if self._worker is not None:
            # Disconnect first so an in-flight result/progress that returns
            # during/after the join is dropped, not delivered (§C.15). Wrapped
            # because a connection may already be gone (a second shutdown()).
            try:
                self._worker.finished.disconnect(self._on_finished)
                self._worker.error.disconnect(self._on_error)
                self._worker.progress.disconnect(self.progress)
            except (RuntimeError, TypeError):
                pass
            # Sticky stop (so a queued run() behind the stop won't start and
            # clear the cancel) AND a per-job cancel (so the in-flight job aborts
            # at its next tile boundary). request_stop accompanies request_cancel
            # (§C.2). Guarded so a second shutdown() (worker already reclaimed)
            # is a clean no-op.
            self._worker.request_stop()
            self._worker.request_cancel()

        if self._thread.isRunning():
            # quit() returns once the loop processes the request; wait() blocks
            # until the thread actually exits (or the timeout elapses).
            self._thread.quit()
            stopped = self._thread.wait(_THREAD_WAIT_MS)
        else:
            stopped = True

        if stopped:
            # Thread is down: safe to drop engine refs and free CUDA memory. The
            # worker is reclaimed by thread.finished -> deleteLater (wired in
            # __init__); we only schedule the thread object's deletion here.
            self._engine = None
            self._worker = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._thread.deleteLater()
            self._thread = None  # sentinel: shutdown() is now an idempotent no-op
        return stopped

    # ----------------------------------------------------------------- helpers
    def _set_busy(self, busy: bool) -> None:
        """Update busy state, emitting :attr:`busy_changed` only on transitions."""
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def _clip_to_image(self, rect: QRect) -> QRect:
        """Clip ``rect`` to the image bounds in half-open image px.

        Uses QRect's half-open ``[x, x+w)`` convention via ``intersected`` with
        the full-image rect, matching the canonical coordinate contract.
        """
        # QRect(0,0,W,H) under Qt's right()/bottom() are inclusive, but
        # intersected() with another QRect honors the same convention on both
        # sides, so the resulting width/height stay correct half-open extents.
        image_rect = QRect(0, 0, self._doc.width, self._doc.height)
        return rect.intersected(image_rect)

    def _rejection_reason(self, rect: QRect) -> str | None:
        """Return a human reason to reject ``rect``, or ``None`` if acceptable.

        Rejects (DESIGN.md §C.7 / §H3):
            * empty / degenerate after clipping,
            * any side ``< _MIN_SIDE_PX``,
            * the selection covering the whole image (nothing distinct left to
              find that isn't the source itself),
            * aspect ratio (long/short) ``> _MAX_ASPECT``.
        """
        w, h = rect.width(), rect.height()
        if rect.isEmpty() or w <= 0 or h <= 0:
            return "Selection is empty after clipping to the image."
        if w < _MIN_SIDE_PX or h < _MIN_SIDE_PX:
            return f"Selection too small: each side must be at least {_MIN_SIDE_PX}px (got {w}x{h})."
        # Whole-image (or larger, pre-clip) selection: after clipping it can be
        # at most the image size, so equality means it spans everything.
        if w >= self._doc.width and h >= self._doc.height:
            return "Selection covers the whole image; box a smaller region to search for."
        long_side, short_side = (w, h) if w >= h else (h, w)
        if short_side > 0 and long_side / short_side > _MAX_ASPECT:
            return f"Selection aspect ratio too extreme (> {_MAX_ASPECT:g}:1): {w}x{h}."
        return None
