"""Lifecycle tests for :class:`fastmatch.controller.MatchController`.

These pin the controller/worker/app coherence introduced by the C2/C3/C6/C14/C15
fixes:

* A real request still delivers ``matches_ready`` even though engine staging is
  now *deferred* onto the worker thread, and two rapid requests honour
  latest-wins (only the newest result surfaces).
* :meth:`MatchController.shutdown` returns ``True`` and joins cleanly — no
  ``QThread: Destroyed while still running`` abort.
* An invalid (too-small) selection emits ``failed`` and dispatches nothing.

Everything runs offscreen on a CPU device against a tiny synthetic stamped
image, so the suite stays fast (well under a second of real matching).
"""

from __future__ import annotations

import os

# Must be set before PySide6 imports a platform plugin so the tests run headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# Skip cleanly if PySide6 (or the engine) is unavailable in this environment.
PySide6 = pytest.importorskip("PySide6", reason="controller tests require PySide6")
pytest.importorskip("fastmatch.engine", reason="engine not yet implemented")

from PySide6.QtCore import QEventLoop, QRect, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fastmatch.controller import MatchController  # noqa: E402
from fastmatch.document import ImageDocument  # noqa: E402
from fastmatch.types import MatchParams  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def qapp():
    """A single session-wide QApplication (Qt forbids more than one)."""
    app = QApplication.instance() or QApplication([])
    yield app


MOTIF_W, MOTIF_H = 28, 24  # asymmetric -> no accidental rotational symmetry


def _stamped_image() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """A small textured image with a distinctive motif stamped at known spots."""
    rng = np.random.default_rng(0)
    h, w = 240, 320
    img = rng.integers(96, 136, (h, w, 3)).astype(np.uint8)
    motif = np.zeros((MOTIF_H, MOTIF_W, 3), dtype=np.uint8)
    motif[:, :, 0] = 30
    motif[: MOTIF_H // 2, : MOTIF_W // 2, 0] = 240          # top-left red
    motif[: MOTIF_H // 2, MOTIF_W // 2 :, 1] = 220          # top-right green
    motif[MOTIF_H // 2 :, : MOTIF_W // 3, 2] = 200          # bottom-left blue
    motif[MOTIF_H // 2 :, MOTIF_W // 3 :, :] = 255          # bottom-right white
    positions = [(20, 18), (200, 30), (60, 150), (250, 120)]
    for (x, y) in positions:
        img[y : y + MOTIF_H, x : x + MOTIF_W] = motif
    return np.ascontiguousarray(img), positions


def _make_doc() -> tuple[ImageDocument, list[tuple[int, int]]]:
    img, positions = _stamped_image()
    h, w = img.shape[:2]
    return ImageDocument(full=img, path="<synthetic>", height=h, width=w), positions


def _cpu_params() -> MatchParams:
    """CPU-pinned params tuned for the tiny high-SNR synthetic scene."""
    return MatchParams(
        threshold=0.85, threshold_floor=0.50, scales=(1.0,), max_results=50, device="cpu"
    )


class _Collector:
    """Connects to a signal up front and records every emission's args.

    Connecting *before* the triggering action matters because some controller
    signals (e.g. ``failed`` for a rejected rect) fire synchronously inside
    :meth:`MatchController.request`; a connect-after-the-fact spy would miss
    them. :meth:`wait` then spins a bounded nested loop so the queued staging +
    worker ``run()`` get a chance to run on the worker thread.
    """

    def __init__(self, signal) -> None:
        self.events: list = []
        self._signal = signal
        signal.connect(self._on)

    def _on(self, *args) -> None:
        self.events.append(args)

    def wait(self, timeout_ms: int = 8000) -> bool:
        """Return True once at least one emission is recorded (or on timeout)."""
        if self.events:
            return True
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        self._signal.connect(loop.quit)
        timer.start(timeout_ms)
        loop.exec()
        timer.stop()
        try:
            self._signal.disconnect(loop.quit)
        except (RuntimeError, TypeError):
            pass
        return bool(self.events)


def _pump(ms: int = 150) -> None:
    """Spin the event loop for a short fixed interval (let queued slots run)."""
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(ms)
    loop.exec()


# --------------------------------------------------------------------------- #
# (1) deferred staging still delivers; latest-wins on two rapid requests
# --------------------------------------------------------------------------- #
def test_deferred_request_delivers_matches(qapp) -> None:
    """A real request delivers Matches even though staging is deferred."""
    doc, positions = _make_doc()
    ctrl = MatchController(doc, _cpu_params())
    try:
        ready = _Collector(ctrl.matches_ready)
        sx, sy = positions[0]
        ctrl.request(QRect(sx, sy, MOTIF_W, MOTIF_H), _cpu_params())

        assert ready.wait(), "matches_ready never fired (deferred dispatch lost)"
        matches = list(ready.events[-1][0])
        # The *other* stamps are recovered; the source stamp is excluded.
        assert len(matches) >= 1
        srcs = [(m.x, m.y) for m in matches]
        assert (sx, sy) not in srcs, "source region was not excluded"
    finally:
        assert ctrl.shutdown() is True


def test_latest_wins_two_rapid_requests(qapp) -> None:
    """Two rapid requests: only the latest selection's result surfaces."""
    doc, positions = _make_doc()
    ctrl = MatchController(doc, _cpu_params())
    try:
        ready = _Collector(ctrl.matches_ready)
        # First aim at one stamp, then immediately re-aim at another before the
        # debounce/staging dispatch — latest-wins must keep only the second.
        a = positions[0]
        b = positions[1]
        ctrl.request(QRect(a[0], a[1], MOTIF_W, MOTIF_H), _cpu_params())
        ctrl.request(QRect(b[0], b[1], MOTIF_W, MOTIF_H), _cpu_params())

        assert ready.wait(), "no result delivered for the rapid-request pair"
        # Let any straggling (aborted, superseded) emission arrive too.
        _pump(200)
        matches = list(ready.events[-1][0])
        srcs = [(m.x, m.y) for m in matches]
        # The *second* selection (b) is the source and must be excluded; the
        # first (a) was superseded so its aborted result must not surface.
        assert b not in srcs, "latest selection's source not excluded (wrong job won)"
        assert len(matches) >= 1
        # Latest-wins: exactly one result should have surfaced (the superseded
        # job is stale-dropped, not delivered).
        assert len(ready.events) == 1, f"expected 1 delivery, got {len(ready.events)}"
    finally:
        assert ctrl.shutdown() is True


# --------------------------------------------------------------------------- #
# (2) shutdown joins cleanly and is idempotent
# --------------------------------------------------------------------------- #
def test_shutdown_returns_true_and_joins(qapp) -> None:
    """shutdown() returns True and the thread joins (no Destroyed-while-running)."""
    doc, _ = _make_doc()
    ctrl = MatchController(doc, _cpu_params())
    # Let staging complete so the join exercises a fully-started thread.
    _pump(200)
    assert ctrl.shutdown() is True
    # Idempotent: a second call is a clean no-op that still reports stopped.
    assert ctrl.shutdown() is True


# --------------------------------------------------------------------------- #
# (3) invalid tiny rect -> failed, dispatches nothing
# --------------------------------------------------------------------------- #
def test_invalid_tiny_rect_emits_failed(qapp) -> None:
    """A sub-minimum selection emits ``failed`` and never dispatches a job."""
    doc, _ = _make_doc()
    ctrl = MatchController(doc, _cpu_params())
    try:
        failed = _Collector(ctrl.failed)
        busy = _Collector(ctrl.busy_changed)

        # 4px side is below the 8px minimum -> rejected before any dispatch.
        # (failed fires synchronously inside request(); the collector caught it.)
        ctrl.request(QRect(10, 10, 4, 4), _cpu_params())

        assert failed.wait(timeout_ms=2000), "invalid rect did not emit failed"
        msg = failed.events[-1][0]
        assert isinstance(msg, str) and msg
        # Nothing was dispatched: busy never went True.
        _pump(150)
        assert all(args[0] is False for args in busy.events), (
            "an invalid rect dispatched a job (went busy)"
        )
    finally:
        assert ctrl.shutdown() is True
