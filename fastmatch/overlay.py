"""Batched match-overlay graphics item.

A single :class:`MatchOverlayItem` draws *all* match boxes (and the source-
template box) for the whole image. We deliberately avoid one ``QGraphicsRectItem``
per match: with up to hundreds of detections that would flood the scene graph
and the OpenGL viewport. Instead boxes live as numpy arrays and ``paint()``
issues one batched ``drawRects`` after culling to the exposed rect.

Coordinates are scene px, which equal full-res image px (the viewport's scene
units), so boxes are drawn with **no conversion** — they stay pixel-locked at
any zoom/pan. The client-side threshold filter is just a numpy mask, so moving
the threshold slider re-filters instantly without re-running the engine.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

# zValue must sit above the pyramid item (which paints at the default 0) so the
# boxes are never hidden behind tile pixmaps.
_OVERLAY_Z = 10.0

# Match boxes: green; source/template box: a distinct cyan so it reads as "the
# thing you selected" versus "the things we found".
_MATCH_COLOR = QColor(40, 220, 70)
_SOURCE_COLOR = QColor(0, 200, 255)
# A semi-transparent dark casing drawn just under the coloured outline (normal
# mode only) so the bright box edges stay legible on light backgrounds too — the
# match/selection colours are theme-independent, the casing keeps them readable.
_CASING_COLOR = QColor(0, 0, 0, 170)


class MatchOverlayItem(QGraphicsItem):
    """Draws every match box plus the source box in one batched item.

    Box storage is a ``(N, 4) int32`` array of ``(x, y, w, h)`` half-open image
    boxes and a parallel ``(N,) float32`` score array. A boolean visibility mask
    (scores >= threshold) is recomputed only when matches or threshold change.
    """

    def __init__(self, full_w: int, full_h: int, parent: QGraphicsItem | None = None) -> None:
        """Create the overlay sized to the full image.

        Args:
            full_w: Full-resolution image width (scene px).
            full_h: Full-resolution image height (scene px).
            parent: Optional parent graphics item.
        """
        super().__init__(parent)
        self._full_w = int(full_w)
        self._full_h = int(full_h)

        self._boxes = np.empty((0, 4), dtype=np.int32)
        self._scores = np.empty((0,), dtype=np.float32)
        self._mask = np.empty((0,), dtype=bool)
        self._threshold = 0.85
        self._source_box: QRect | None = None

        # Configurable box appearance (set from the View menu).
        self._line_width = 1     # cosmetic pen width in DEVICE px (immune to zoom)
        self._xor = False        # XOR the box pixels with the background (always
                                 # visible regardless of underlying colour)

        self.setZValue(_OVERLAY_Z)
        # The overlay is purely decorative; it must never intercept mouse events
        # meant for rubber-band selection / panning in the view.
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    # ------------------------------------------------------------------ data

    def set_matches(self, boxes: np.ndarray, scores: np.ndarray) -> None:
        """Replace the match set.

        Args:
            boxes: ``(N, 4) int32`` half-open ``(x, y, w, h)`` boxes in image px.
            scores: ``(N,) float32`` scores in ``[0, 1]``, parallel to ``boxes``.
        """
        boxes = np.ascontiguousarray(boxes, dtype=np.int32).reshape(-1, 4)
        scores = np.ascontiguousarray(scores, dtype=np.float32).reshape(-1)
        if boxes.shape[0] != scores.shape[0]:
            raise ValueError(
                f"boxes/scores length mismatch: {boxes.shape[0]} vs {scores.shape[0]}"
            )
        self._boxes = boxes
        self._scores = scores
        self._recompute_mask()
        self.update()

    def set_threshold(self, t: float) -> None:
        """Set the client-side score threshold; re-filter via numpy mask only.

        Does not re-run the engine: the engine returned everything at or above
        the floor, so we just hide boxes below ``t``.
        """
        t = float(t)
        if t == self._threshold:
            return
        self._threshold = t
        self._recompute_mask()
        self.update()

    def set_source_box(self, rect: QRect | None) -> None:
        """Set (or clear) the source-template box, drawn in a distinct color.

        Args:
            rect: Half-open image-px rect, or ``None`` to remove it.
        """
        # Copy so a later caller mutation can't alias our stored rect.
        self._source_box = QRect(rect) if rect is not None else None
        self.update()

    def set_line_width(self, px: int) -> None:
        """Set the box outline width in device px (cosmetic; zoom-independent)."""
        w = max(1, int(px))
        if w == self._line_width:
            return
        self._line_width = w
        self.update()

    def set_xor(self, on: bool) -> None:
        """Toggle XOR-with-background compositing for the boxes.

        When on, box pixels are bitwise-XORed with whatever is underneath, so the
        outline stays visible on any background (and inverts again if redrawn over
        itself) — the classic CAD/selection look. When off, boxes are drawn as
        solid coloured lines.
        """
        on = bool(on)
        if on == self._xor:
            return
        self._xor = on
        self.update()

    def visible_count(self) -> int:
        """Number of match boxes currently at/above threshold."""
        return int(self._mask.sum()) if self._mask.size else 0

    def visible_boxes(self) -> "list[QRectF]":
        """Post-threshold match boxes as scene-coord rects (for the focus spotlight)."""
        if not (self._boxes.shape[0] and self._mask.any()):
            return []
        return [
            QRectF(float(x), float(y), float(w), float(h))
            for x, y, w, h in self._boxes[self._mask]
        ]

    def clear(self) -> None:
        """Remove all matches and the source box."""
        self._boxes = np.empty((0, 4), dtype=np.int32)
        self._scores = np.empty((0,), dtype=np.float32)
        self._mask = np.empty((0,), dtype=bool)
        self._source_box = None
        self.update()

    # ------------------------------------------------------------- internals

    def _recompute_mask(self) -> None:
        """Recompute the visibility mask (scores >= threshold)."""
        if self._scores.size:
            self._mask = self._scores >= self._threshold
        else:
            self._mask = np.empty((0,), dtype=bool)

    # ----------------------------------------------------------- QGraphicsItem

    def boundingRect(self) -> QRectF:
        """Full image rect; the overlay may draw a box anywhere in the image."""
        return QRectF(0.0, 0.0, float(self._full_w), float(self._full_h))

    def paint(
        self,
        painter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Draw all visible boxes, culled to the exposed rect, in one batch.

        Culling: ``option.exposedRect`` is in item (== scene == image) px. We
        build a vectorized numpy mask of boxes that intersect it (half-open
        overlap test) and feed only those to ``drawRects``. The pen is cosmetic,
        so its configured width is in device px and immune to view scale; the
        ``XOR`` mode (View menu) composites the lines onto the background with a
        bitwise XOR so they stay visible over any colour.
        """
        exposed = option.exposedRect
        # Expand the cull window by 1px so boxes straddling the edge still draw.
        ex0 = exposed.left() - 1.0
        ey0 = exposed.top() - 1.0
        ex1 = exposed.right() + 1.0
        ey1 = exposed.bottom() + 1.0

        # Save/restore so the composition mode never leaks to other items.
        painter.save()
        if self._xor:
            # "XOR with background": invert the box pixels against whatever is
            # underneath so the outline is visible on ANY background. True bitwise
            # XOR (RasterOp_SourceXorDestination) is NOT supported by Qt's OpenGL
            # paint engine (the viewport is a QOpenGLWidget) — it warns and draws
            # nothing useful. CompositionMode_Difference IS GL-supported and gives
            # the same effect: |dest - colour| per channel, which inverts against
            # the background (e.g. the box vanishes to black over its own colour
            # and pops everywhere else).
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Difference
            )

        def _pen(color: QColor, width: int) -> QPen:
            p = QPen(color)
            p.setCosmetic(True)  # width in device px, immune to view scale
            p.setWidth(width)
            return p

        def _draw_boxes(rects: list[QRectF], color: QColor) -> None:
            """Draw ``rects`` outlined in ``color``, with a dark contrast casing.

            In normal mode a slightly wider semi-transparent dark stroke is laid
            down first, so the bright outline stays legible on *any* background —
            a light canvas/image (where the match green or selection cyan would
            otherwise wash out) as much as a dark one. XOR mode already inverts
            against the background, so it skips the casing.
            """
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if not self._xor:
                painter.setPen(_pen(_CASING_COLOR, self._line_width + 2))
                painter.drawRects(rects)
            painter.setPen(_pen(color, self._line_width))
            painter.drawRects(rects)

        if self._boxes.shape[0] and self._mask.any():
            visible = self._boxes[self._mask]
            bx = visible[:, 0].astype(np.float64)
            by = visible[:, 1].astype(np.float64)
            bw = visible[:, 2].astype(np.float64)
            bh = visible[:, 3].astype(np.float64)
            # Half-open intersection: box [bx, bx+bw) overlaps exposed [ex0, ex1].
            keep = (
                (bx < ex1)
                & ((bx + bw) > ex0)
                & (by < ey1)
                & ((by + bh) > ey0)
            )
            if keep.any():
                culled = visible[keep]
                # Build a flat QRectF list in one pass; drawRects batches the
                # GL calls far better than per-box drawRect.
                rects = [
                    QRectF(float(x), float(y), float(w), float(h))
                    for x, y, w, h in culled
                ]
                _draw_boxes(rects, _MATCH_COLOR)

        # Source box on top, in its own color, if present and exposed.
        sb = self._source_box
        if sb is not None and not sb.isNull():
            if (
                sb.x() < ex1
                and (sb.x() + sb.width()) > ex0
                and sb.y() < ey1
                and (sb.y() + sb.height()) > ey0
            ):
                _draw_boxes([QRectF(sb)], _SOURCE_COLOR)

        painter.restore()


# zValue above the match overlay so ruler lines/labels are never hidden.
_RULER_Z = 11.0
_MEASURE_COLOR = QColor(255, 210, 0)   # ruler line: amber
_CALIB_COLOR = QColor(255, 120, 0)     # calibration reference: orange


class MeasureOverlayItem(QGraphicsItem):
    """Draws the calibration reference line, the ruler line, and the in-progress
    drag line, each labelled with its physical (or pixel) length.

    Like :class:`MatchOverlayItem` it paints in scene px (== image px) with
    cosmetic pens so geometry stays pixel-locked at any zoom. Labels and endpoint
    dots are drawn in *device* space (reset transform) so they stay a constant
    on-screen size and legible regardless of zoom.
    """

    def __init__(self, w: int, h: int, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self._w, self._h = int(w), int(h)
        self.setZValue(_RULER_Z)
        self._line_width = 2
        self._scale: float | None = None    # units/px, for labels (None -> px)
        self._unit = ""
        self._pending: tuple[QPointF, QPointF, str] | None = None  # (p1,p2,kind)
        self._measure: tuple[QPointF, QPointF] | None = None
        self._calib: tuple[QPointF, QPointF] | None = None

    def boundingRect(self) -> QRectF:
        # Generous (matches the viewport's padded scene rect) so device-space
        # labels near the image edges are never clipped by item culling.
        return QRectF(-self._w, -self._h, 3 * self._w, 3 * self._h)

    # -- setters (each repaints) --------------------------------------------
    def set_line_width(self, px: int) -> None:
        self._line_width = max(1, int(px))
        self.update()

    def set_calibration(self, scale: float | None, unit: str) -> None:
        self._scale = scale
        self._unit = unit or ""
        self.update()

    def set_pending(self, p1: QPointF | None, p2: QPointF | None, kind: str) -> None:
        self._pending = (p1, p2, kind) if (p1 is not None and p2 is not None) else None
        self.update()

    def set_measure(self, p1: QPointF | None, p2: QPointF | None) -> None:
        self._measure = (p1, p2) if (p1 is not None and p2 is not None) else None
        self.update()

    def set_calibration_line(self, p1: QPointF | None, p2: QPointF | None) -> None:
        self._calib = (p1, p2) if (p1 is not None and p2 is not None) else None
        self.update()

    def clear_all(self) -> None:
        self._pending = self._measure = self._calib = None
        self.update()

    # -- painting -----------------------------------------------------------
    def _label(self, p1: QPointF, p2: QPointF, kind: str) -> str:
        dx, dy = abs(p2.x() - p1.x()), abs(p2.y() - p1.y())
        # Calibration is defined on the longer axis; the ruler uses true distance.
        span = max(dx, dy) if kind == "calibrate" else math.hypot(dx, dy)
        if self._scale:
            return f"{self._scale * span:.4g} {self._unit}".rstrip()
        return f"{span:.0f} px"

    def _draw_segment(self, painter, p1, p2, color, label) -> None:
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(self._line_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QLineF(p1, p2))
        # Endpoint dots: a fatter cosmetic point => constant on-screen diameter.
        dot = QPen(color)
        dot.setCosmetic(True)
        dot.setWidth(self._line_width + 5)
        dot.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(dot)
        painter.drawPoints([p1, p2])
        if label:
            self._draw_label(painter, QLineF(p1, p2).pointAt(0.5), label, color)

    def _draw_label(self, painter, scene_pt: QPointF, text: str, color: QColor) -> None:
        # Map the midpoint to device px, then draw text/background unscaled.
        dev = painter.worldTransform().map(scene_pt)
        painter.save()
        painter.resetTransform()
        fm = painter.fontMetrics()
        box = fm.boundingRect(text).adjusted(-5, -3, 5, 3)
        box.moveTo(int(dev.x()) + 10, int(dev.y()) - box.height() // 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRect(box)
        painter.setPen(color)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def paint(self, painter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        if self._calib is not None:
            p1, p2 = self._calib
            self._draw_segment(painter, p1, p2, _CALIB_COLOR, self._label(p1, p2, "calibrate"))
        if self._measure is not None:
            p1, p2 = self._measure
            self._draw_segment(painter, p1, p2, _MEASURE_COLOR, self._label(p1, p2, "measure"))
        if self._pending is not None:
            p1, p2, kind = self._pending
            color = _CALIB_COLOR if kind == "calibrate" else _MEASURE_COLOR
            self._draw_segment(painter, p1, p2, color, self._label(p1, p2, kind))
