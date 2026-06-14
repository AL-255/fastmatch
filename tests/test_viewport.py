"""Viewport zoom/pan regression tests (offscreen, Qt-only — no engine).

Two behaviours that have regressed before and are easy to break:

  * **Cursor-pivot zoom** — the wheel zoom must keep the scene point under the
    cursor fixed. We anchor manually (NoAnchor + an explicit re-centre on the
    wheel event's logical ``position()``) rather than via Qt's ``AnchorUnderMouse``,
    because the latter mis-tracks the cursor on fractional HiDPI displays (e.g.
    GNOME at 150%). Driving a synthesized ``QWheelEvent`` and checking the
    scene-point drift pins this independent of display scaling.
  * **Infinite-canvas scene rect** — the scene is padded beyond the image so the
    view can place any image point (edges included) anywhere in the viewport,
    which is what lets the cursor stay the pivot at the edges.

These run under ``QT_QPA_PLATFORM=offscreen``; we use a plain ``QWidget`` viewport
so geometry/mapToScene are exercised cleanly without an OpenGL surface.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="viewport tests need PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from fastmatch.document import ImageDocument  # noqa: E402
from fastmatch.viewport import ImageViewport  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _viewport(qapp) -> ImageViewport:
    doc = ImageDocument(np.zeros((2000, 2400, 3), np.uint8), "mem", 2000, 2400)
    vp = ImageViewport()
    vp.set_image(doc)
    vp.setViewport(QWidget())  # raster viewport: clean offscreen geometry, no GL
    vp.resize(800, 600)
    vp.show()
    qapp.processEvents()
    return vp


def _wheel(vp: ImageViewport, x: float, y: float, delta: int) -> None:
    pf = QPointF(float(x), float(y))
    ev = QWheelEvent(
        pf, pf, QPoint(0, 0), QPoint(0, delta),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    vp.wheelEvent(ev)


def _drift(vp: ImageViewport, qapp, x: int, y: int, delta: int) -> float:
    p = QPoint(x, y)
    before = vp.mapToScene(p)
    _wheel(vp, x, y, delta)
    qapp.processEvents()
    after = vp.mapToScene(p)
    return ((after.x() - before.x()) ** 2 + (after.y() - before.y()) ** 2) ** 0.5


def test_wheel_zoom_keeps_cursor_as_pivot(qapp) -> None:
    """The scene point under the cursor is fixed across a wheel zoom (in & out)."""
    vp = _viewport(qapp)
    vp.set_zoom(4.0)
    qapp.processEvents()
    # Off-centre cursor: AnchorViewCenter would drift here; the manual anchor must not.
    assert _drift(vp, qapp, 220, 160, 120) < 1.0   # zoom in
    assert _drift(vp, qapp, 600, 420, -120) < 1.0  # zoom out, different point


def test_wheel_zoom_pivot_holds_at_image_edge(qapp) -> None:
    """Cursor stays the pivot even with the image edge under it (infinite canvas)."""
    vp = _viewport(qapp)
    vp.set_zoom(8.0)
    vp.centerOn(QPointF(5, 5))  # image top-left corner near the cursor
    qapp.processEvents()
    assert _drift(vp, qapp, 70, 55, 120) < 1.0
    assert _drift(vp, qapp, 70, 55, -120) < 1.0


def test_scene_rect_padded_for_infinite_canvas(qapp) -> None:
    """The scene extends beyond the image so any edge point can be centred."""
    vp = _viewport(qapp)
    sr = vp._scene.sceneRect()  # noqa: SLF001 - white-box check of the padding
    assert sr.x() < 0 and sr.y() < 0
    assert sr.width() > 2400 and sr.height() > 2000
    vp.set_zoom(8.0)
    ctr = vp.viewport().rect().center()
    for px, py in ((0, 0), (2400, 2000)):  # image corners
        vp.centerOn(QPointF(px, py))
        qapp.processEvents()
        s = vp.mapToScene(ctr)
        assert abs(s.x() - px) < 2 and abs(s.y() - py) < 2


# --------------------------------------------------------------- focus mode
def test_focus_mode_toggle_via_space_and_signal(qapp) -> None:
    """Space toggles focus mode and emits focusModeChanged (no auto-repeat)."""
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent

    vp = _viewport(qapp)
    seen: list[bool] = []
    vp.focusModeChanged.connect(seen.append)
    assert vp.is_focus_mode() is False

    def _space() -> None:
        vp.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        )

    _space()
    assert vp.is_focus_mode() is True
    _space()
    assert vp.is_focus_mode() is False
    assert seen == [True, False]


def test_focus_mode_dims_outside_boxes(qapp) -> None:
    """drawForeground veils the image OUTSIDE the boxes and leaves boxes bright."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter

    from fastmatch.types import Match

    vp = _viewport(qapp)
    vp.set_matches([Match(100, 100, 200, 200, 0.95, 1.0)])  # covers (100..300)
    vp.set_match_threshold(0.5)
    vp.set_focus_mode(True)

    img = QImage(400, 400, QImage.Format.Format_RGB32)
    img.fill(Qt.GlobalColor.white)
    p = QPainter(img)
    vp.drawForeground(p, QRectF(0, 0, 400, 400))  # identity transform: scene == px
    p.end()

    inside = img.pixelColor(200, 200)   # inside the box -> a hole, untouched
    outside = img.pixelColor(20, 20)    # outside the box -> dimmed by the veil
    assert inside.red() == 255
    assert outside.red() < 255
    assert outside.red() < inside.red()


def test_focus_mode_no_dim_without_boxes(qapp) -> None:
    """With nothing to spotlight, focus mode must not dim the whole image."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter

    vp = _viewport(qapp)
    vp.clear_matches()
    vp.clear_template()
    vp.set_focus_mode(True)

    img = QImage(64, 64, QImage.Format.Format_RGB32)
    img.fill(Qt.GlobalColor.white)
    p = QPainter(img)
    vp.drawForeground(p, QRectF(0, 0, 64, 64))
    p.end()
    assert img.pixelColor(32, 32).red() == 255  # untouched
