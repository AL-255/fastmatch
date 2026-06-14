"""Overlay rendering tests (offscreen, Qt-only — no engine).

Pins the box-legibility fix: match/selection box colours are deliberately
theme-independent (a match always reads as the same green), so on the Light
theme's pale canvas the bright outlines would wash out. A dark contrast
*casing* is drawn just under the coloured outline in normal mode to keep the
edges legible on any background; XOR mode composites against the background
instead and skips the casing.

We render the overlay item straight onto a light-grey ``QImage`` and inspect
pixels, so the behaviour is checked independent of any view/GL surface.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="overlay tests need PySide6")

from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtGui import QImage, QPainter, QColor  # noqa: E402
from PySide6.QtWidgets import QApplication, QStyleOptionGraphicsItem  # noqa: E402

from fastmatch.overlay import MatchOverlayItem  # noqa: E402

_BG = (225, 225, 225)  # the Light theme's canvas colour


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _render(xor: bool) -> QImage:
    img = QImage(60, 60, QImage.Format.Format_RGB32)
    img.fill(QColor(*_BG))
    item = MatchOverlayItem(60, 60)
    item.set_matches(np.array([[10, 10, 40, 40]], np.int32), np.array([1.0], np.float32))
    item.set_threshold(0.0)
    item.set_line_width(2)
    item.set_xor(xor)
    painter = QPainter(img)
    opt = QStyleOptionGraphicsItem()
    opt.exposedRect = QRectF(0, 0, 60, 60)
    item.paint(painter, opt, None)
    painter.end()
    return img


def _rgb(img: QImage, x: int, y: int) -> tuple[int, int, int]:
    c = img.pixelColor(x, y)
    return (c.red(), c.green(), c.blue())


def test_normal_mode_draws_dark_casing_and_green_line_on_light_canvas(qapp) -> None:
    """On the pale canvas the box edge has both a dark casing and the green line."""
    img = _render(xor=False)
    edge = [_rgb(img, x, 10) for x in range(7, 15)]  # across the top edge
    has_casing = any(sum(c) < 300 for c in edge)  # clearly darker than bg (675)
    has_green = any(c[1] > c[0] + 40 and c[1] > c[2] + 40 for c in edge)
    assert has_casing, f"no dark contrast casing along the edge: {edge}"
    assert has_green, f"no green outline along the edge: {edge}"


def test_box_interior_is_untouched(qapp) -> None:
    """Only the outline is drawn — the interior keeps the background colour."""
    img = _render(xor=False)
    assert _rgb(img, 30, 30) == _BG


def test_xor_mode_skips_the_plain_casing(qapp) -> None:
    """XOR composites against the background, producing a different edge result."""
    normal = _render(xor=False)
    xored = _render(xor=True)
    assert _rgb(xored, 10, 10) != _rgb(normal, 10, 10)


def test_visible_boxes_respects_threshold(qapp) -> None:
    """visible_boxes() returns exactly the post-threshold match boxes (focus mode)."""
    ov = MatchOverlayItem(400, 300)
    boxes = np.array([[40, 40, 70, 70], [250, 170, 80, 80]], np.int32)
    scores = np.array([0.95, 0.60], np.float32)
    ov.set_matches(boxes, scores)

    ov.set_threshold(0.5)
    vb = ov.visible_boxes()
    assert [(r.x(), r.y(), r.width(), r.height()) for r in vb] == [
        (40.0, 40.0, 70.0, 70.0),
        (250.0, 170.0, 80.0, 80.0),
    ]

    ov.set_threshold(0.8)  # only the 0.95 box survives
    vb = ov.visible_boxes()
    assert len(vb) == 1 and (vb[0].x(), vb[0].y()) == (40.0, 40.0)

    ov.clear()
    assert ov.visible_boxes() == []
