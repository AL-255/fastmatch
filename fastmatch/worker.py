"""The match worker that runs the engine on a dedicated QThread.

``MatchWorker`` is a plain ``QObject`` (not a ``QThread`` subclass) so it can be
``moveToThread``'d onto the single long-lived matching thread owned by
:class:`fastmatch.controller.MatchController`. It wraps a
:class:`fastmatch.engine.Matcher` and exposes the engine's blocking
``match()`` call as the queued :meth:`run` slot.

Three invariants hold this layer together (see DESIGN.md §C.6, §F.1, §F.2):

* **One worker, one in-flight job.** The owning thread runs a single
  ``match()`` at a time; the controller debounces and stale-drops by ``job_id``.
* **Plain CPU payloads only.** Signals carry ``list[Match]`` dataclasses and
  ``str`` — never CUDA tensors, large ndarrays, or QPixmaps. The engine has
  already done ``.cpu()`` before returning, so nothing GPU-resident crosses a
  signal boundary (DESIGN.md §M7).
* **Cooperative cancellation.** :meth:`request_cancel` sets a
  :class:`threading.Event` that the engine polls at every tile boundary via the
  ``cancel`` callback. Because the event is shared mutable state touched from the
  GUI thread (set) and the worker thread (read), it must be a thread-safe
  primitive — hence ``threading.Event`` rather than a bare bool.
"""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from .engine import Matcher
from .types import Match, MatchParams


class MatchWorker(QObject):
    """Runs :class:`Matcher.match` on its owning thread, reporting via signals.

    Signals:
        progress: ``int`` in ``0..100`` — per-tile determinate progress,
            forwarded straight from the engine's progress callback.
        finished: ``(object, int)`` — ``(list[Match], job_id)`` on success.
            The payload is plain CPU data the GUI can consume directly.
        error: ``(str, int)`` — ``(traceback_str, job_id)`` if ``match()``
            raised. The full formatted traceback is sent so the controller /
            UI can surface a useful message instead of swallowing it.
    """

    progress = Signal(int)          # 0..100
    finished = Signal(object, int)  # (list[Match], job_id)
    error = Signal(str, int)        # (traceback_str, job_id)

    def __init__(self, engine: Matcher) -> None:
        """Bind the worker to an engine instance.

        Args:
            engine: A constructed :class:`Matcher` that already has its image
                staged via ``set_image``. The worker does not own the engine's
                lifecycle beyond calling ``match``; the controller wires it up.
        """
        super().__init__()
        self._engine = engine
        # Shared cancel flag. Set from the GUI thread (request_cancel), polled
        # from the worker thread (the cancel callback). threading.Event is the
        # safe cross-thread primitive for this hand-off.
        self._cancel_event = threading.Event()
        # Sticky shutdown flag, separate from the per-job cancel. request_stop()
        # sets it once at teardown and it is *never* cleared by run(); a per-job
        # cancel, by contrast, is cleared at the start of every job. Keeping the
        # two distinct stops a queued run() from wiping a shutdown-intent cancel
        # (which would let a slow match start uncancelled and wedge the join).
        self._stop_event = threading.Event()

    def request_cancel(self) -> None:
        """Signal the in-flight (or next) ``match()`` to abort cooperatively.

        Safe to call from any thread (typically the GUI thread). Setting the
        event makes the engine's ``cancel`` callback return True at its next
        tile-boundary poll; ``match`` then returns early. :meth:`run` clears the
        event at the start of every job so a stale cancel never kills a fresh
        request.
        """
        self._cancel_event.set()

    def request_stop(self) -> None:
        """Sticky shutdown request: abort the in-flight job and refuse new ones.

        Safe to call from any thread. Unlike :meth:`request_cancel`, the stop
        flag is never cleared by :meth:`run`, so a job already queued behind the
        stop will see it (and return immediately without starting) rather than
        clearing the cancel and running a full, uncancellable match.
        """
        self._stop_event.set()

    @Slot(object, object, object, int)
    def run(
        self,
        template: "object",
        params: MatchParams,
        exclude_box: "tuple | None",
        mask: "object" = None,
        job_id: int = 0,
    ) -> None:
        """Execute one match job and emit its result.

        Invoked via a queued connection from the controller, so the body runs
        on this object's owning (worker) thread. ``template`` is a plain numpy
        ndarray (typed loosely as ``object`` to match the ``Signal(object,...)``
        marshalling and keep numpy out of the import surface here).

        Args:
            template: ``(h, w, 3)`` uint8 RGB crop of the source region.
            params: Search parameters for this job.
            exclude_box: ``(x, y, w, h)`` source region to exclude from hits,
                or ``None``.
            job_id: Monotonic id that rides on the result for stale-drop.
        """
        # A sticky stop takes precedence over everything: a job queued behind a
        # shutdown must NOT start and must NOT clear the cancel event (clearing
        # it would let a subsequent match run uncancelled and wedge the join).
        if self._stop_event.is_set():
            return

        # Clear any cancel left over from a previous job *before* the engine can
        # observe it. A cancel arriving after this point (for the current job)
        # still takes effect; one arriving for the previous, already-finished
        # job is harmless because that result was stale-dropped by job_id.
        self._cancel_event.clear()

        def _cancel() -> bool:
            """Cancel callback polled by the engine at every tile boundary.

            Honours both the per-job cancel and the sticky stop, so a stop set
            mid-job (after run() already passed the start-of-job guard) still
            aborts the in-flight match.
            """
            return self._cancel_event.is_set() or self._stop_event.is_set()

        def _progress(pct: int) -> None:
            """Progress callback the engine calls per finished tile (0..100)."""
            self.progress.emit(int(pct))

        try:
            matches: list[Match] = self._engine.match(
                template,
                params,
                exclude_box=exclude_box,
                mask=mask,
                cancel=_cancel,
                progress=_progress,
            )
            # Engine contract: returns a plain list[Match] (CPU dataclasses),
            # already moved off the GPU. Emit as-is — no tensors cross here.
            self.finished.emit(matches, job_id)
        except Exception:
            # Capture the full traceback as a string so the GUI thread gets a
            # diagnosable message rather than a bare repr. The job_id rides
            # along so the controller can ignore errors from stale jobs.
            self.error.emit(traceback.format_exc(), job_id)
