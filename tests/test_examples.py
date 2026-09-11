"""Multi-example search: aligned mean template + positive/negative verification."""
import numpy as np
import pytest

from fastmatch.engine import Matcher
from fastmatch.examples import match_examples, prepare_examples
from fastmatch.types import MatchParams

S = 20  # motif size


def _scene(seed=0, noise=45.0):
    """Noisy copies of a ringed motif, plus decoys sharing its dark centre."""
    rng = np.random.default_rng(seed)
    img = rng.normal(128, 12, (640, 640, 3))
    yy, xx = np.mgrid[0:S, 0:S] - (S - 1) / 2
    rr = np.hypot(yy, xx)
    motif = np.full((S, S, 3), 150.0)
    motif[(rr > 5) & (rr < 8)] = (90, 60, 200)     # purple ring
    motif[rr <= 3.5] = (20, 20, 30)                # dark core
    decoy = np.full((S, S, 3), 150.0)
    decoy[rr <= 3.5] = (20, 20, 30)                # same core, no ring
    decoy[:, S // 2 - 1:S // 2 + 1] = (200, 120, 60)  # and an orange bar
    targets, decoys = [], []
    for gy in range(8):
        for gx in range(8):
            x, y = 20 + gx * 75 + int(rng.integers(0, 10)), 20 + gy * 75 + int(rng.integers(0, 10))
            which = (gx + gy) % 2
            patch = motif if which == 0 else decoy
            img[y:y + S, x:x + S] = patch + rng.normal(0, noise, patch.shape)
            (targets if which == 0 else decoys).append((x, y))
    return np.clip(img, 0, 255).astype(np.uint8), targets, decoys


def _found(matches, points, tol=3):
    return sum(any(abs(m.x - x) <= tol and abs(m.y - y) <= tol for m in matches) for x, y in points)


def _matcher(img):
    m = Matcher(device="cpu")
    m.set_image(img)
    return m


def test_prepare_snaps_sloppy_boxes_and_averages():
    img, targets, _ = _scene()
    (x0, y0), (x1, y1) = targets[0], targets[1]
    # Second box drawn 3 px off and a bit larger: re-centred then snapped back.
    mean, pos, neg, pos_boxes, _ = prepare_examples(
        img, [(x0, y0, S, S), (x1 + 3, y1 - 2, S + 4, S + 2)], [])
    assert pos_boxes[0] == (x0, y0, S, S)
    assert abs(pos_boxes[1][0] - x1) <= 1 and abs(pos_boxes[1][1] - y1) <= 1
    assert mean.shape == (S, S, 3) and pos.shape == (2, S, S, 3) and neg.shape[0] == 0


def test_examples_find_all_targets_and_reject_decoys():
    img, targets, decoys = _scene()
    m = _matcher(img)
    params = MatchParams(threshold_floor=0.3)
    pos = [(x, y, S, S) for x, y in targets[:5]]
    neg = [(x, y, S, S) for x, y in decoys[:3]]
    res = [r for r in match_examples(m, pos, neg, params) if r.score >= 0.3]
    assert _found(res, targets) == len(targets)          # every target, examples included
    assert _found(res, decoys) == 0                      # no decoy survives verification
    # Examples are part of the result (the count is the total number of instances).
    assert _found(res, targets[:5]) == 5


def test_single_noisy_template_is_worse_than_examples():
    img, targets, decoys = _scene()
    m = _matcher(img)
    params = MatchParams(threshold_floor=0.3)
    single = [r for r in m.match(img[targets[0][1]:targets[0][1] + S, targets[0][0]:targets[0][0] + S],
                                 params, exclude_box=None) if r.score >= 0.3]
    assert _found(single, decoys) > 0  # the one-box search confuses decoys for targets


def test_negative_boxes_are_never_returned_and_cancel_is_honoured():
    img, targets, decoys = _scene(seed=1)
    m = _matcher(img)
    neg = [(x, y, S, S) for x, y in decoys[:4]]
    res = match_examples(m, [(targets[0][0], targets[0][1], S, S)], neg, MatchParams(threshold_floor=0.0))
    assert _found(res, decoys[:4]) == 0
    assert match_examples(m, [(targets[0][0], targets[0][1], S, S)], neg, MatchParams(),
                          cancel=lambda: True) == []


def test_requires_a_positive():
    img, _, _ = _scene()
    with pytest.raises(ValueError):
        prepare_examples(img, [], [])


# --------------------------------------------------------------------------- #
# Controller / GUI wiring (offscreen, CPU)
# --------------------------------------------------------------------------- #
import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, QEventLoop, QPointF, QRect, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from fastmatch.controller import MatchController  # noqa: E402
from fastmatch.document import ImageDocument  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _wait_for(signal, timeout_ms=15000):
    got = []
    signal.connect(lambda *a: got.append(a))
    loop, timer = QEventLoop(), QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    signal.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    return got


def _doc(img):
    return ImageDocument(full=img, path="<scene>", height=img.shape[0], width=img.shape[1])


def test_controller_runs_multi_example_search(qapp):
    img, targets, decoys = _scene()
    params = MatchParams(threshold_floor=0.3, device="cpu")
    ctrl = MatchController(_doc(img), params)
    try:
        (x0, y0) = targets[0]
        ctrl.request(QRect(x0, y0, S, S), params, None,
                     ([(x, y, S, S) for x, y in targets[1:5]], [(x, y, S, S) for x, y in decoys[:3]]))
        got = _wait_for(ctrl.matches_ready)
        assert got, "no result"
        res = [m for m in got[-1][0] if m.score >= 0.3]
        assert _found(res, targets) == len(targets)
        assert _found(res, decoys) == 0
    finally:
        assert ctrl.shutdown() is True


def _drag(view, a, b, mods):
    """Left-drag between two IMAGE-px points (mapped to viewport px)."""
    vp = view.viewport()
    a = view.mapFromScene(QPointF(*a)); a = (a.x(), a.y())
    b = view.mapFromScene(QPointF(*b)); b = (b.x(), b.y())
    for kind, p in ((QEvent.Type.MouseButtonPress, a), (QEvent.Type.MouseMove, b),
                    (QEvent.Type.MouseButtonRelease, b)):
        btns = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else Qt.MouseButton.LeftButton
        ev = QMouseEvent(kind, QPointF(*p), QPointF(vp.mapToGlobal(QPointF(*p).toPoint())),
                         Qt.MouseButton.LeftButton, btns, mods)
        {QEvent.Type.MouseButtonPress: view.mousePressEvent, QEvent.Type.MouseMove: view.mouseMoveEvent,
         QEvent.Type.MouseButtonRelease: view.mouseReleaseEvent}[kind](ev)


def test_shift_and_ctrl_drags_add_examples_and_research(qapp):
    from fastmatch.app import MainWindow
    img, targets, decoys = _scene()
    from PySide6.QtWidgets import QRubberBand
    w = MainWindow(_doc(img), device="cpu")
    w._viewport.setViewport(QWidget())  # raster viewport for offscreen...
    # ...which deletes the rubber band parented to the old GL viewport.
    w._viewport._rubber = QRubberBand(QRubberBand.Shape.Rectangle, w._viewport.viewport())
    w.resize(1600, 1000)
    w.show()
    qapp.processEvents()
    w._viewport.fit_in_view()
    try:
        view = w._viewport
        none = Qt.KeyboardModifier.NoModifier
        # Shift+drag with no selection yet is just a normal selection.
        _drag(view, (100, 100), (140, 140), Qt.KeyboardModifier.ShiftModifier)
        assert view.template_rect() is not None and view.examples() == ([], [])
        _drag(view, (200, 200), (240, 240), Qt.KeyboardModifier.ShiftModifier)
        _drag(view, (300, 300), (330, 330), Qt.KeyboardModifier.ControlModifier)
        pos, neg = view.examples()
        assert len(pos) == 1 and len(neg) == 1
        assert w._examples_label.isVisible() and "2" in w._examples_label.text()
        assert w._act_clear_examples.isEnabled()
        # A plain drag replaces the selection and drops the example set.
        _drag(view, (120, 120), (160, 160), none)
        assert view.examples() == ([], []) and not w._examples_label.isVisible()

        # End to end through the app: selection + examples -> every target found.
        (x0, y0) = targets[0]
        w._on_region_selected(QRect(x0, y0, S, S))
        view._set_examples([QRect(x, y, S, S) for x, y in targets[1:5]],
                           [QRect(x, y, S, S) for x, y in decoys[:3]])
        w._params = MatchParams(threshold=0.3, threshold_floor=0.3, device="cpu")
        view.examplesChanged.emit(4, 3)
        got = _wait_for(w._controller.matches_ready)
        assert got
        res = [m for m in got[-1][0] if m.score >= 0.3]
        assert _found(res, targets) == len(targets) and _found(res, decoys) == 0
        w._act_clear_examples.trigger()
        assert view.examples() == ([], []) and not w._act_clear_examples.isEnabled()
    finally:
        w.close()


def _window(qapp, img):
    from PySide6.QtWidgets import QRubberBand
    from fastmatch.app import MainWindow
    w = MainWindow(_doc(img), device="cpu")
    w._viewport.setViewport(QWidget())
    w._viewport._rubber = QRubberBand(QRubberBand.Shape.Rectangle, w._viewport.viewport())
    w.resize(1600, 1000)
    w.show()
    qapp.processEvents()
    w._viewport.fit_in_view()
    w._auto_run = False  # these tests exercise the example bookkeeping only
    return w


def _centre(r):
    return QPointF(r[0] + r[2] / 2, r[1] + r[3] / 2)


def test_numbered_examples_can_be_deleted_one_by_one(qapp):
    img, targets, decoys = _scene()
    w = _window(qapp, img)
    try:
        view = w._viewport
        sel = QRect(targets[0][0], targets[0][1], S, S)
        view.set_template_rect(sel)
        w._on_region_selected(sel)
        pos = [QRect(x, y, S, S) for x, y in targets[1:4]]   # examples #2, #3, #4
        neg = [QRect(x, y, S, S) for x, y in decoys[:2]]     # ×1, ×2
        view._set_examples(pos, neg)
        # Hit testing follows the on-screen numbering.
        assert view.example_at(_centre(targets[0] + (S, S))) == ("pos", 1)
        assert view.example_at(_centre(targets[2] + (S, S))) == ("pos", 3)
        assert view.example_at(_centre(decoys[1] + (S, S))) == ("neg", 2)
        assert view.example_at(QPointF(1, 1)) is None
        # Delete #3: #4 becomes #3.
        view.remove_example("pos", 3)
        assert view.examples()[0] == [(*targets[1], S, S), (*targets[3], S, S)]
        assert view.example_at(_centre(targets[3] + (S, S))) == ("pos", 3)
        # Delete negative ×1 through the right-click menu (first action).
        w._exec_menu = lambda menu, _pos: menu.actions()[0]
        w._on_example_menu("neg", 1, QPointF(0, 0).toPoint())
        assert view.examples()[1] == [(*decoys[1], S, S)]
        assert "2" in w._examples_label.text() or "3" in w._examples_label.text()
        # Delete #1 (the selection): #2 is promoted to be the selection.
        view.remove_example("pos", 1)
        assert view.template_rect() == QRect(targets[1][0], targets[1][1], S, S)
        assert w._last_rect == view.template_rect()
        assert view.examples() == ([(*targets[3], S, S)], [(*decoys[1], S, S)])
        view.viewport().grab()  # numbered tags paint without error
        # Down to the selection alone, deleting it clears the whole selection.
        view.remove_example("pos", 2)
        view.remove_example("neg", 1)
        assert view.example_at(_centre(targets[1] + (S, S))) is None  # unnumbered now
        view._set_examples([], [QRect(decoys[0][0], decoys[0][1], S, S)])
        view.remove_example("pos", 1)
        assert view.template_rect() is None and w._last_rect is None
        assert view.examples() == ([], [])
    finally:
        w.close()
