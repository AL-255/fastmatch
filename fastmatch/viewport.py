"""The tiled, GPU-composited image viewport.

:class:`ImageViewport` is a ``QGraphicsView`` whose scene units equal full-res
image pixels. It composites the image from a lazy mip pyramid (one
``PyramidItem`` painting only the visible tiles at a level chosen from the zoom),
overlays match boxes (one batched :class:`MatchOverlayItem`), and handles hybrid
input: SELECT-mode left-drag rubber-band selection, middle-mouse (or Pan-tool)
pan, Space to toggle focus mode (dim outside the boxes), and zoom-under-cursor
wheel.

Tile pixels are decoded off the GUI thread on a ``QThreadPool``; the worker
returns a plain ndarray and the ``QImage``/``QPixmap`` are created on the GUI
thread (Qt pixmaps are not reliably constructible off-GUI). Every tile request
carries the current ``view_generation`` so stale results (from a since-abandoned
zoom/pan) are discarded on arrival.

Coordinate contract (§F.3): integer image px, origin top-left, +x right/+y down,
half-open boxes; float selections round floor-TL / ceil-BR; cursor readout uses
floor.
"""

from __future__ import annotations

import enum
import math

import numpy as np
from PySide6.QtCore import (
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QRunnable,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygon,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QRubberBand,
    QStyleOptionGraphicsItem,
    QWidget,
)

from .document import ImageDocument
from .overlay import MatchOverlayItem, MeasureOverlayItem
from .pyramid import TILE, ImagePyramid, TileCache
from .types import Match

# Hysteresis band around the LOD boundary (in level units) so a tiny zoom wiggle
# at a level transition does not thrash decoding (§E.2).
_LOD_HYSTERESIS = 0.25

# Background behind tiles (and any region with no decoded tile yet).
_BG_COLOR = QColor(30, 30, 30)

# Focus mode: opacity (0-255) of the dark veil laid over the image OUTSIDE the
# boxes. ~60% keeps dimmed context faintly visible while the boxes pop.
_FOCUS_DIM_ALPHA = 150

# Rectilinear selection: click within this many VIEWPORT px of the first vertex
# to close the polygon. Outline colours for the committed / in-progress shapes.
_POLY_CLOSE_PX = 12
_POLY_COMMIT_COLOR = QColor(60, 150, 235)   # committed selection outline (accent)
_POLY_DRAW_COLOR = QColor(255, 180, 40)     # in-progress polyline (amber)

# BT.601 luminance weights (R, G, B), matching the engine's convention (§D.1)
# so the grayscale display mode shows exactly what the matcher "sees" in
# luminance mode: L = 0.299R + 0.587G + 0.114B.
_BT601 = (0.299, 0.587, 0.114)


# --------------------------------------------------------------------------- #
# Off-GUI tile decode worker
# --------------------------------------------------------------------------- #
class _TileSignals(QObject):
    """Signal holder for :class:`_TileDecodeTask` (QRunnable is not a QObject)."""

    # (level, cx, cy, ndarray, generation). ndarray is a plain CPU uint8 array;
    # QImage/QPixmap creation happens on the GUI thread that receives this.
    done = Signal(int, int, int, object, int)


class _TileDecodeTask(QRunnable):
    """Decodes one display tile (block-mean downsample) off the GUI thread.

    Returns only a numpy array via the ``done`` signal — never a QPixmap — so no
    Qt pixmap is constructed on a pool thread (§E.7).
    """

    def __init__(
        self,
        pyramid: ImagePyramid,
        level: int,
        cx: int,
        cy: int,
        generation: int,
        signals: _TileSignals,
    ) -> None:
        super().__init__()
        self._pyramid = pyramid
        self._level = level
        self._cx = cx
        self._cy = cy
        self._generation = generation
        self._signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:  # noqa: D401 - QRunnable entry point
        """Decode the tile region and emit the resulting ndarray."""
        try:
            tile = self._pyramid.tile
            x = self._cx * tile
            y = self._cy * tile
            arr = self._pyramid.region(self._level, x, y, tile, tile)
        except Exception:
            # A decode failure must not crash the pool thread; just drop the
            # tile (the painter falls back to a coarser cached level).
            return
        try:
            self._signals.done.emit(self._level, self._cx, self._cy, arr, self._generation)
        except RuntimeError:
            # The viewport (and its signals object) may have been destroyed while
            # this decode was in flight (e.g. the window closed). Dropping the
            # result is correct — there is nothing left to paint into.
            return


# --------------------------------------------------------------------------- #
# Pyramid graphics item
# --------------------------------------------------------------------------- #
class PyramidItem(QGraphicsItem):
    """Paints only the visible tiles of the chosen LOD level.

    The item spans the full image (``boundingRect == (0,0,W,H)``) in scene px.
    During ``paint`` it determines which tiles of the active level intersect the
    exposed rect, pulls ready :class:`QPixmap` objects from the
    :class:`TileCache`, and schedules decode tasks for any that are missing. A
    coarser cached level is drawn underneath as a blurry fallback so newly
    exposed regions are never blank.
    """

    def __init__(self, viewport: "ImageViewport", pyramid: ImagePyramid) -> None:
        super().__init__()
        self._vp = viewport
        self._pyramid = pyramid
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        # CRITICAL for large images: ask Qt to set option.exposedRect to the
        # actually-exposed (visible) region in paint(). WITHOUT this flag Qt
        # defaults exposedRect to the WHOLE boundingRect (the entire image), so
        # _paint_level would request EVERY tile of the level on every repaint —
        # e.g. all 675 level-0 tiles of a 6177x6747 image. With a 256-tile cache
        # the working set never fits, so tiles are evicted and re-decoded forever:
        # perpetual churn that flickers and "breaks" zoom on big images. With the
        # flag, a deep-zoom paint exposes ~1 tile and the cache settles.
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

    def boundingRect(self) -> QRectF:
        """Full image rect in scene px."""
        return QRectF(0.0, 0.0, float(self._pyramid.full_w), float(self._pyramid.full_h))

    def paint(
        self,
        painter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Composite visible tiles at the active level, plus a coarse fallback."""
        level = self._vp.current_level
        pyr = self._pyramid
        levels = pyr.levels
        if not levels:
            return

        exposed = option.exposedRect  # scene px == full-res image px

        # Coarse fallback first (one level up), so any not-yet-decoded fine tile
        # still shows something. Pinned top levels make this near-free.
        if level + 1 < len(levels):
            self._paint_level(painter, exposed, level + 1, schedule_missing=False)
        self._paint_level(painter, exposed, level, schedule_missing=True)

    def _paint_level(
        self,
        painter,
        exposed: QRectF,
        level: int,
        schedule_missing: bool,
    ) -> None:
        """Draw every tile of ``level`` intersecting ``exposed`` (scene px).

        Each level tile is ``tile`` level-px wide, which maps to ``tile * 2**level``
        scene px; we draw the pixmap into that scene-px target rect so coarse and
        fine levels align exactly.
        """
        pyr = self._pyramid
        lv = pyr.levels[level]
        tile = pyr.tile
        scene_tile = tile * (1 << level)  # scene px covered by one level tile

        # Tile-column/row range intersecting the exposed scene rect.
        cx0 = max(0, int(math.floor(exposed.left() / scene_tile)))
        cy0 = max(0, int(math.floor(exposed.top() / scene_tile)))
        cx1 = min(lv.cols - 1, int(math.floor((exposed.right()) / scene_tile)))
        cy1 = min(lv.rows - 1, int(math.floor((exposed.bottom()) / scene_tile)))
        if cx1 < cx0 or cy1 < cy0:
            return

        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                key = (level, cx, cy)
                pm = self._vp.tile_cache.get(key)
                if pm is None:
                    if schedule_missing:
                        self._vp.request_tile(level, cx, cy)
                    continue
                # Scene-px target rect for this tile. The pyramid clips edge
                # tiles, so the pixmap may be smaller than a full tile; scale the
                # target to the pixmap's actual level-px size to stay aligned.
                sx = cx * scene_tile
                sy = cy * scene_tile
                tw = pm.width() * (1 << level)
                th = pm.height() * (1 << level)
                painter.drawPixmap(QRectF(sx, sy, tw, th), pm, QRectF(pm.rect()))


# --------------------------------------------------------------------------- #
# The viewport widget
# --------------------------------------------------------------------------- #
class ImageViewport(QGraphicsView):
    """Tiled, OpenGL-composited image view with zoom/pan/select (§C.8, §E)."""

    class Mode(enum.Enum):
        """Interaction mode for the left mouse button."""

        SELECT = enum.auto()
        PAN = enum.auto()
        RECTILINEAR = enum.auto()  # click right-angle vertices -> orthogonal-polygon select
        CALIBRATE = enum.auto()   # one-shot: left-drag picks the calibration span
        MEASURE = enum.auto()     # one-shot: left-drag measures a distance

    # Signals (§C.8) -------------------------------------------------------- #
    regionSelected = Signal(QRect)        # template rect, full-res image px, half-open
    matchesChanged = Signal(int)          # count currently shown (post threshold)
    cursorImagePos = Signal(int, int)     # live image px under cursor (-1,-1 outside)
    calibrationPicked = Signal(QPointF, QPointF)  # two image-px points spanning a length
    measurePicked = Signal(QPointF, QPointF)      # two image-px points to measure
    zoomChanged = Signal(float, int)      # (view_scale, current_pyramid_level)
    viewChanged = Signal(QRect)           # visible image-px rect (prefetch hint)
    focusModeChanged = Signal(bool)       # focus/spotlight mode toggled (Space)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # OpenGL viewport. It MUST use FullViewportUpdate: a QOpenGLWidget is
        # double-buffered and swaps the whole framebuffer each frame, so a partial
        # (Smart/Minimal) update — which repaints only the changed region — leaves
        # the rest of the swapped buffer showing stale/undefined content, i.e. the
        # image FLICKERS (most visibly when zoomed in, where partial repaints and
        # async tile arrivals cover large screen areas). Qt documents
        # FullViewportUpdate as the correct mode for GL viewports; redrawing the
        # whole viewport is free for GL, which composites the full scene per frame
        # regardless. (§E.8)
        self.setViewport(QOpenGLWidget())
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setRenderHints(self.renderHints())  # no MSAA; keep defaults
        self.setBackgroundBrush(_BG_COLOR)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        # We drive everything by hand (rubber-band, panning, anchors) so disable
        # the built-in drag mode and let mapToScene handle DPR-aware mapping.
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # The zoom paths set their own anchor (wheel: NoAnchor + manual cursor
        # re-centre, DPR-safe; set_zoom: AnchorViewCenter), so the default is moot.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # --- authoritative view state (kept as doubles, §E.5/M3) ---
        self._view_scale: float = 1.0       # viewport px per full-res image px
        self._min_scale: float = 1.0 / 256.0
        self._max_scale: float = 64.0        # clamp max zoom 64x (§E.5)
        self._level: int = 0
        self._mode = ImageViewport.Mode.SELECT

        # --- document / pyramid / overlay state ---
        self._doc: ImageDocument | None = None
        self._pyramid: ImagePyramid | None = None
        self._pyramid_item: PyramidItem | None = None
        self._overlay: MatchOverlayItem | None = None
        # Box appearance (View menu); kept on the viewport so it persists across
        # set_image() (which rebuilds the overlay) and is re-applied below.
        self._box_line_width = 1
        self._box_xor = False
        self.tile_cache = TileCache(max_pixmaps=256, pinned_levels=2)

        # When True, base-image tiles render in BT.601 luminance (grayscale) at
        # the QImage step (the match-box overlay stays in colour). Default OFF.
        self._display_grayscale: bool = False

        # --- tile decode plumbing ---
        self._pool = QThreadPool.globalInstance()
        self._tile_signals = _TileSignals()
        self._tile_signals.done.connect(self._on_tile_decoded)
        self._view_generation: int = 0
        self._pending_tiles: set[tuple[int, int, int]] = set()

        # --- input state ---
        self._focus_mode = False   # Space toggles the spotlight (dim outside boxes)
        self._panning = False
        self._pan_last: QPoint | None = None
        self._selecting = False
        self._sel_origin: QPoint | None = None
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
        self._template_rect: QRect | None = None

        # Rectilinear (orthogonal-polygon) selection state. Vertices are in IMAGE
        # px; each committed edge is axis-aligned (snapped H/V). _sel_polygon holds
        # the last COMMITTED selection outline (for drawing); _selection_mask is its
        # rasterized boolean mask (bbox-sized) or None for a plain rectangle.
        self._poly_pts: list[QPoint] = []          # in-progress vertices (image px)
        self._poly_preview: QPoint | None = None   # live snapped cursor (image px)
        self._sel_polygon: list[QPoint] | None = None
        self._selection_mask: np.ndarray | None = None

        # Calibration / measurement (ruler) state.
        self._measure_overlay: MeasureOverlayItem | None = None
        self._measuring = False
        self._measure_p1: QPointF | None = None
        self._measure_kind = "measure"   # "measure" | "calibrate" for the active drag

        self._apply_cursor()

    # ------------------------------------------------------------------ image
    def set_image(self, doc: ImageDocument, *, tile: int = TILE) -> None:
        """Load ``doc`` into the view, rebuilding the pyramid and scene.

        Args:
            doc: The source document (``doc.full`` may be a memmap).
            tile: Display tile edge in level px (default 256).
        """
        self.clear_image()
        self._doc = doc
        self._pyramid = ImagePyramid(doc, tile=tile)

        # Reset and configure the tile cache for the new level count.
        self.tile_cache = TileCache(max_pixmaps=256, pinned_levels=2)
        self.tile_cache.set_num_levels(self._pyramid.num_levels())

        # Scene = the image PLUS a generous margin on every side, so the canvas
        # behaves as if it were unbounded. QGraphicsView clamps the view to the
        # scene rect, so without padding a zoom-under-cursor near an image edge
        # cannot keep the cursor point fixed (the view can't scroll past the
        # image) and the point drifts. A one-image-size margin on each side lets
        # any image point sit anywhere in the viewport at any zoom, so wheel zoom
        # always pivots exactly under the cursor. Item coords (== image px) are
        # unchanged; the margin is just empty scrollable space.
        w, h = float(doc.width), float(doc.height)
        self._scene.setSceneRect(-w, -h, 3.0 * w, 3.0 * h)

        self._pyramid_item = PyramidItem(self, self._pyramid)
        self._scene.addItem(self._pyramid_item)

        self._overlay = MatchOverlayItem(doc.width, doc.height)
        self._overlay.set_line_width(self._box_line_width)
        self._overlay.set_xor(self._box_xor)
        self._scene.addItem(self._overlay)

        # Ruler/calibration overlay sits above the match boxes.
        self._measure_overlay = MeasureOverlayItem(doc.width, doc.height)
        self._measure_overlay.set_line_width(max(2, self._box_line_width + 1))
        self._scene.addItem(self._measure_overlay)

        # Min scale: never zoom out further than the whole image fitting a small
        # window; max stays at the hard 64x cap.
        longest = max(doc.width, doc.height)
        self._min_scale = min(1.0, 64.0 / longest) if longest else 1.0 / 256.0

        self._bump_generation()
        self.fit_in_view()

    def image_size(self) -> tuple[int, int]:
        """Return ``(W, H)`` in full-res image px (``(0, 0)`` if no image)."""
        if self._doc is None:
            return (0, 0)
        return (self._doc.width, self._doc.height)

    def clear_image(self) -> None:
        """Remove the current image, pyramid, overlay, and cached tiles."""
        self._scene.clear()  # drops pyramid + overlay items
        self._doc = None
        self._pyramid = None
        self._pyramid_item = None
        self._overlay = None
        self._measure_overlay = None
        self._measuring = False
        self._measure_p1 = None
        self._template_rect = None
        self._reset_rectilinear_state()
        self._sel_polygon = None
        self._selection_mask = None
        self.tile_cache.clear()
        self._pending_tiles.clear()
        self._bump_generation()

    # --------------------------------------------------------------- matches
    def set_matches(self, matches: list[Match]) -> None:
        """Set match boxes from a ``list[Match]``; convert to numpy internally.

        The viewport's public boundary speaks ``Match`` dataclasses (§C.8); the
        overlay draws from ``(N,4) int32`` / ``(N,) f32`` numpy arrays, so we
        convert here exactly once.
        """
        if self._overlay is None:
            return
        n = len(matches)
        boxes = np.empty((n, 4), dtype=np.int32)
        scores = np.empty((n,), dtype=np.float32)
        for i, m in enumerate(matches):
            boxes[i, 0] = m.x
            boxes[i, 1] = m.y
            boxes[i, 2] = m.w
            boxes[i, 3] = m.h
            scores[i] = m.score
        self._overlay.set_matches(boxes, scores)
        self.matchesChanged.emit(self._overlay.visible_count())

    def clear_matches(self) -> None:
        """Remove all match boxes (keeps the template box)."""
        if self._overlay is None:
            return
        self._overlay.set_matches(
            np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.float32)
        )
        self.matchesChanged.emit(0)

    def set_match_threshold(self, t: float) -> None:
        """Client-side score filter in ``[0, 1]``; no engine re-run (§C.8)."""
        if self._overlay is None:
            return
        self._overlay.set_threshold(t)
        self.matchesChanged.emit(self._overlay.visible_count())

    def visible_match_count(self) -> int:
        """Number of match boxes currently shown (post threshold)."""
        if self._overlay is None:
            return 0
        return self._overlay.visible_count()

    def set_box_line_width(self, px: int) -> None:
        """Set the overlay box outline width (device px; persists across images)."""
        self._box_line_width = max(1, int(px))
        if self._overlay is not None:
            self._overlay.set_line_width(self._box_line_width)

    def set_box_xor(self, on: bool) -> None:
        """Toggle XOR-with-background compositing for the overlay boxes."""
        self._box_xor = bool(on)
        if self._overlay is not None:
            self._overlay.set_xor(self._box_xor)

    def set_background_color(self, color: QColor) -> None:
        """Set the canvas colour behind tiles (follows the application theme)."""
        self.setBackgroundBrush(QColor(color))
        self.viewport().update()

    # --------------------------------------------------- calibration / ruler
    def set_calibration(self, scale: float | None, unit: str) -> None:
        """Set the physical scale (units/px) used to label ruler/calibration lines."""
        if self._measure_overlay is not None:
            self._measure_overlay.set_calibration(scale, unit)

    def set_calibration_line(self, p1: QPointF | None, p2: QPointF | None) -> None:
        """Show (or clear) the persistent calibration reference line."""
        if self._measure_overlay is not None:
            self._measure_overlay.set_calibration_line(p1, p2)

    def clear_measurement(self) -> None:
        """Remove the committed ruler line."""
        if self._measure_overlay is not None:
            self._measure_overlay.set_measure(None, None)

    def clear_calibration(self) -> None:
        """Remove the calibration reference line and forget the physical scale."""
        if self._measure_overlay is not None:
            self._measure_overlay.set_calibration_line(None, None)
            self._measure_overlay.set_calibration(None, "")

    # ------------------------------------------------------ selection/template
    def template_rect(self) -> QRect | None:
        """The current template selection rect (image px), or ``None``."""
        return QRect(self._template_rect) if self._template_rect is not None else None

    def clear_template(self) -> None:
        """Clear the template selection, its highlighted source box, and any mask."""
        self._template_rect = None
        self._reset_rectilinear_state()
        self._sel_polygon = None
        self._selection_mask = None
        if self._overlay is not None:
            self._overlay.set_source_box(None)
        self.viewport().update()

    def set_template_rect(self, rect: QRect | None) -> None:
        """Set the template selection + its highlighted (blue) source box.

        Used to restore the reference box when recalling a saved Memory entry, so
        the blue box returns to the remembered selection location.
        """
        if rect is None:
            self.clear_template()
            return
        self._template_rect = QRect(rect)
        if self._overlay is not None:
            self._overlay.set_source_box(QRect(rect))

    # --------------------------------------------------------------- view ctl
    def set_mode(self, mode: "ImageViewport.Mode") -> None:
        """Switch interaction mode and update the cursor."""
        if mode is not ImageViewport.Mode.RECTILINEAR:
            self._reset_rectilinear_state()  # leaving the tool drops an unfinished poly
        self._mode = mode
        self._apply_cursor()
        self.viewport().update()

    def set_display_grayscale(self, on: bool) -> None:
        """Render base-image tiles in BT.601 luminance (grayscale) when ``on``.

        Shows what the matcher "sees" in luminance mode (``L = 0.299R + 0.587G
        + 0.114B``); the conversion happens at the tile -> QImage/QPixmap step
        (see :meth:`_on_tile_decoded`). The match-box overlay is unaffected and
        stays in colour over the grayscale base. Default OFF.

        Toggling drops every cached tile pixmap (pinned included) so tiles
        re-render in the new mode, bumps the view generation so in-flight decode
        results from the old mode are discarded on arrival, and repaints.
        Idempotent: a no-op if the mode is unchanged.
        """
        on = bool(on)
        if on == self._display_grayscale:
            return  # idempotent: mode unchanged
        self._display_grayscale = on
        # Drop colour pixmaps (incl. pinned tiles) so the painter re-requests
        # tiles and they re-render in the new mode.
        self.tile_cache.clear()
        # Bump the view generation: any decode already in flight finalized its
        # ndarray under the old mode bit; discarding it forces a fresh request.
        self._bump_generation()
        self.viewport().update()
        self._scene.update()

    def fit_in_view(self) -> None:
        """Fit the whole image in the viewport and sync ``view_scale``/LOD."""
        if self._doc is None:
            return
        rect = QRectF(0.0, 0.0, float(self._doc.width), float(self._doc.height))
        # Reset transform first so fitInView computes from identity.
        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        # Recover the resulting uniform scale from the transform (m11 == m22 for
        # KeepAspectRatio without rotation).
        self._view_scale = float(self.transform().m11())
        self._clamp_scale_state()
        self._update_lod(force=True)
        self._emit_view_changed()

    def set_zoom(self, view_scale: float) -> None:
        """Set absolute zoom (viewport px per image px), clamped to limits.

        Programmatic zoom pivots on the view centre. We set the anchor explicitly
        because the wheel path leaves it on ``NoAnchor`` (it re-centres manually).
        """
        if self._doc is None:
            return
        target = self._clamp(view_scale, self._min_scale, self._max_scale)
        if target <= 0:
            return
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        factor = target / self._view_scale if self._view_scale else target
        self.scale(factor, factor)
        self._view_scale = target
        self._update_lod()
        self._emit_view_changed()

    def visible_image_rect(self) -> QRect:
        """Visible region as a half-open image-px ``QRect`` (clipped to image)."""
        poly = self.mapToScene(self.viewport().rect())
        sr = poly.boundingRect()
        if self._doc is None:
            return QRect()
        x0 = max(0, int(math.floor(sr.left())))
        y0 = max(0, int(math.floor(sr.top())))
        x1 = min(self._doc.width, int(math.ceil(sr.right())))
        y1 = min(self._doc.height, int(math.ceil(sr.bottom())))
        return QRect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    @property
    def view_scale(self) -> float:
        """Viewport px per full-res image px (authoritative, kept as double)."""
        return self._view_scale

    @property
    def current_level(self) -> int:
        """Active pyramid level for the current zoom."""
        return self._level

    # ------------------------------------------------------ tile decode bridge
    def request_tile(self, level: int, cx: int, cy: int) -> None:
        """Schedule an off-GUI decode of tile ``(level, cx, cy)`` if not pending.

        Tagged with the current ``view_generation`` so a result that arrives
        after the view moved on is discarded.
        """
        if self._pyramid is None:
            return
        key = (level, cx, cy)
        if key in self._pending_tiles or self.tile_cache.get(key) is not None:
            return
        self._pending_tiles.add(key)
        task = _TileDecodeTask(
            self._pyramid, level, cx, cy, self._view_generation, self._tile_signals
        )
        self._pool.start(task)

    def _on_tile_decoded(
        self, level: int, cx: int, cy: int, arr: object, generation: int
    ) -> None:
        """GUI-thread slot: build the QPixmap and cache it, then repaint.

        QImage/QPixmap construction lives here (GUI thread) per §E.7. The QImage
        must be told the explicit ``strides[0]`` bytesPerLine and the backing
        ndarray kept alive until ``QPixmap.fromImage`` copies it.
        """
        key = (level, cx, cy)
        # Drop stale results from a superseded view generation. The pending
        # flag is cleared AFTER this guard: a stale result must not clear the
        # pending flag of a newer same-tile request still in flight (doing so
        # would let a redundant third decode be scheduled for that tile).
        if generation != self._view_generation:
            return
        self._pending_tiles.discard(key)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return

        if self._display_grayscale:
            # Grayscale display: convert the decoded RGB tile to a single-channel
            # BT.601 luminance image (what the matcher "sees" in luminance mode).
            # A contiguous uint8 (H,W) array feeds Format_Grayscale8 with an
            # explicit bytesPerLine == W. Build directly so the (H,W) buffer is
            # already C-contiguous (so bytesPerLine == strides[0] == W).
            buf = self._tile_to_luminance(arr)
            h, w = buf.shape[0], buf.shape[1]
            bytes_per_line = buf.strides[0]  # == w for a contiguous uint8 (H,W)
            image = QImage(
                buf.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8
            )
        else:
            # Ensure C-contiguity (memmap-derived crops already are, but be safe)
            # so bytesPerLine == strides[0] is valid.
            buf = np.ascontiguousarray(arr)
            h, w = buf.shape[0], buf.shape[1]
            bytes_per_line = buf.strides[0]
            image = QImage(buf.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # buf is referenced by `image` until fromImage copies; keep it alive
        # implicitly by holding the local until after the copy below.
        pm = QPixmap.fromImage(image)  # copies pixels; ndarray/QImage now free
        pm.setDevicePixelRatio(1.0)    # trust scene mapping, not DPR scaling (§E.8)
        self.tile_cache.put(key, pm)
        # Pin coarsest-level tiles as they arrive so the fallback layer persists.
        self.tile_cache.pin_top_levels()
        self.viewport().update()

    @staticmethod
    def _tile_to_luminance(arr: np.ndarray) -> np.ndarray:
        """Decoded RGB tile ``(H,W,3)`` uint8 -> contiguous BT.601 luminance uint8.

        Reuses the engine's BT.601 weights (``L = 0.299R + 0.587G + 0.114B``).
        Returns a C-contiguous ``(H,W)`` uint8 array so the caller can hand it to
        ``QImage.Format_Grayscale8`` with ``bytesPerLine == W``. Rounds to nearest
        (``+ 0.5`` then truncate) and clips to ``[0, 255]`` for fp safety.
        """
        rgb = np.asarray(arr, dtype=np.float32)
        wr, wg, wb = _BT601
        lum = rgb[:, :, 0] * wr + rgb[:, :, 1] * wg + rgb[:, :, 2] * wb
        lum = np.clip(lum + 0.5, 0.0, 255.0)
        return np.ascontiguousarray(lum.astype(np.uint8))

    # ------------------------------------------------------------- LOD / state
    def _update_lod(self, force: bool = False) -> None:
        """Pick the pyramid level for the current zoom (§E.2, with hysteresis)."""
        if self._pyramid is None:
            return
        num = self._pyramid.num_levels()
        # ideal = log2(image px per viewport px) = log2(1/view_scale), >= 0.
        ideal = max(0.0, math.log2(1.0 / self._view_scale)) if self._view_scale > 0 else 0.0
        target = int(round(ideal))
        target = max(0, min(num - 1, target))

        if not force:
            # Hysteresis: only change level if the ideal has crossed beyond the
            # +/-0.25 band around the current level, preventing boundary thrash.
            if abs(ideal - self._level) < (1.0 - _LOD_HYSTERESIS):
                self._emit_zoom_changed()
                return

        if target != self._level or force:
            self._level = target
        self._emit_zoom_changed()

    def _emit_zoom_changed(self) -> None:
        self.zoomChanged.emit(self._view_scale, self._level)

    def _emit_view_changed(self) -> None:
        if self._doc is not None:
            self.viewChanged.emit(self.visible_image_rect())

    def _bump_generation(self) -> None:
        """Invalidate all in-flight tile requests (view changed materially)."""
        self._view_generation += 1
        self._pending_tiles.clear()

    def _clamp_scale_state(self) -> None:
        """Keep ``view_scale`` within limits, re-applying the transform if needed."""
        clamped = self._clamp(self._view_scale, self._min_scale, self._max_scale)
        if clamped != self._view_scale and self._view_scale > 0:
            factor = clamped / self._view_scale
            self.scale(factor, factor)
            self._view_scale = clamped

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    # ----------------------------------------------------------------- cursor
    def _apply_cursor(self) -> None:
        """Set the cursor to reflect mode / pan state."""
        if self._panning:
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._mode is ImageViewport.Mode.PAN:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _is_pan_active(self) -> bool:
        """True if the current state means a left-drag should pan, not select."""
        return self._mode is ImageViewport.Mode.PAN

    # ------------------------------------------------------------- wheel zoom
    def wheelEvent(self, e) -> None:
        """Zoom keeping the point under the cursor fixed (§E.5).

        We anchor MANUALLY rather than via ``AnchorUnderMouse``: Qt's built-in
        anchor mis-tracks the cursor on fractional HiDPI displays (e.g. GNOME at
        150%), zooming toward the view centre instead of the cursor. Reading the
        wheel event's own ``position()`` (device-independent logical px, the same
        space ``mapToScene`` uses) and translating by the scene-point delta is
        DPR-correct on any scale factor.
        """
        if self._doc is None:
            e.ignore()
            return
        # angleDelta is in 1/8-degree units; 1.0015**delta is a smooth exp ramp.
        step = 1.0015 ** e.angleDelta().y()
        target = self._clamp(self._view_scale * step, self._min_scale, self._max_scale)
        if target != self._view_scale and self._view_scale > 0:
            vp_pos = e.position().toPoint()  # cursor in viewport (logical) px
            # NoAnchor + explicit re-centering: pin the scene point under the
            # cursor across the scale (works regardless of display scaling).
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            before = self.mapToScene(vp_pos)
            factor = target / self._view_scale
            self.scale(factor, factor)
            self._view_scale = target
            after = self.mapToScene(vp_pos)
            delta = after - before
            self.translate(delta.x(), delta.y())
            self._bump_generation()
            self._update_lod()
            self._emit_view_changed()
        e.accept()  # do NOT chain to super() — avoids the base scroll behavior

    # ------------------------------------------------------------- key events
    def keyPressEvent(self, e) -> None:
        """Space toggles focus mode; in rectilinear mode Enter closes / Esc cancels
        the in-progress polygon (Backspace removes the last vertex)."""
        key = e.key()
        if self._mode is ImageViewport.Mode.RECTILINEAR and self._poly_pts:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if len(self._poly_pts) >= 3:
                    self._close_rectilinear()
                e.accept()
                return
            if key == Qt.Key.Key_Escape:
                self._cancel_rectilinear()
                e.accept()
                return
            if key == Qt.Key.Key_Backspace:
                self._poly_pts.pop()
                self.viewport().update()
                e.accept()
                return
        if key == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.set_focus_mode(not self._focus_mode)
            e.accept()
            return
        super().keyPressEvent(e)

    # ----------------------------------------------------------- focus mode
    def is_focus_mode(self) -> bool:
        """Whether the focus/spotlight mode is currently on."""
        return self._focus_mode

    def set_focus_mode(self, on: bool) -> None:
        """Turn the focus spotlight on/off (dims the image outside the boxes)."""
        on = bool(on)
        if on == self._focus_mode:
            return
        self._focus_mode = on
        self.viewport().update()       # repaint so drawForeground re-runs
        self.focusModeChanged.emit(on)

    def _focus_boxes(self) -> "list[QRectF]":
        """Scene-coord rects kept bright in focus mode: the visible match boxes
        plus the active template selection (if any)."""
        boxes: list[QRectF] = []
        if self._overlay is not None:
            boxes.extend(self._overlay.visible_boxes())
        if self._template_rect is not None:
            boxes.append(QRectF(self._template_rect))
        return boxes

    def drawForeground(self, painter, rect) -> None:
        """Foreground overlays: the focus-mode veil and the rectilinear-selection
        outline (committed shape + in-progress polyline)."""
        super().drawForeground(painter, rect)
        if self._doc is None:
            return
        self._draw_focus_veil(painter, rect)
        self._draw_rectilinear(painter)

    def _draw_focus_veil(self, painter, rect) -> None:
        """Lay a dark veil over the image everywhere EXCEPT inside the boxes."""
        if not self._focus_mode:
            return
        boxes = self._focus_boxes()
        if not boxes:
            return  # nothing to spotlight -> don't pointlessly dim the whole image
        image = QRectF(0.0, 0.0, float(self._doc.width), float(self._doc.height))
        area = rect.intersected(image)
        if area.isEmpty():
            return
        veil = QPainterPath()
        veil.addRect(area)
        holes = QPainterPath()
        for b in boxes:
            holes.addRect(b.intersected(image))
        veil = veil.subtracted(holes)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.fillPath(veil, QColor(0, 0, 0, _FOCUS_DIM_ALPHA))
        painter.restore()

    def _draw_rectilinear(self, painter) -> None:
        """Draw the committed selection polygon and any in-progress polyline."""
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._sel_polygon and len(self._sel_polygon) >= 2:
            pen = QPen(_POLY_COMMIT_COLOR)
            pen.setCosmetic(True)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawPolygon(QPolygon(self._sel_polygon))
        if self._mode is ImageViewport.Mode.RECTILINEAR and self._poly_pts:
            pen = QPen(_POLY_DRAW_COLOR)
            pen.setCosmetic(True)
            pen.setWidth(2)
            painter.setPen(pen)
            pts = list(self._poly_pts)
            if self._poly_preview is not None:
                pts.append(self._poly_preview)
            painter.drawPolyline(QPolygon(pts))
            # First-vertex marker (~10 device px), so the user can aim to close on it.
            s = max(1.0, 5.0 / max(self._view_scale, 1e-6))
            first = self._poly_pts[0]
            painter.drawRect(QRectF(first.x() - s, first.y() - s, 2 * s, 2 * s))
        painter.restore()

    # ----------------------------------------------------- rectilinear select
    def selection_mask(self) -> "np.ndarray | None":
        """Boolean mask (bbox-sized, ``True`` == inside) for the last rectilinear
        selection, or ``None`` for a plain rectangular selection."""
        return None if self._selection_mask is None else self._selection_mask.copy()

    @staticmethod
    def _snap_rectilinear(last: QPoint, p: "QPointF | QPoint") -> QPoint:
        """Snap ``p`` so the edge ``last -> result`` is axis-aligned (the longer of
        the two deltas wins -> a horizontal or vertical segment)."""
        px, py = int(round(p.x())), int(round(p.y()))
        if abs(px - last.x()) >= abs(py - last.y()):
            return QPoint(px, last.y())   # horizontal edge
        return QPoint(last.x(), py)        # vertical edge

    def _rectilinear_click(self, viewport_pt: QPoint) -> None:
        """Add a (snapped) vertex, or close the polygon if clicking near the first."""
        ip = self._clamped_image_point(viewport_pt)
        if not self._poly_pts:
            v = QPoint(int(round(ip.x())), int(round(ip.y())))
            self._poly_pts.append(v)
            self._poly_preview = v
            self.viewport().update()
            return
        if len(self._poly_pts) >= 3:
            first = self._poly_pts[0]
            vp_first = self.mapFromScene(QPointF(first.x(), first.y()))
            if (abs(viewport_pt.x() - vp_first.x()) <= _POLY_CLOSE_PX
                    and abs(viewport_pt.y() - vp_first.y()) <= _POLY_CLOSE_PX):
                self._close_rectilinear()
                return
        self._poly_pts.append(self._snap_rectilinear(self._poly_pts[-1], ip))
        self.viewport().update()

    def _reset_rectilinear_state(self) -> None:
        self._poly_pts = []
        self._poly_preview = None

    def _cancel_rectilinear(self) -> None:
        """Discard the in-progress polygon (Esc / mode switch)."""
        self._reset_rectilinear_state()
        self.viewport().update()

    def _close_rectilinear(self) -> None:
        """Finalize the orthogonal polygon: rasterize its mask, set it as the
        template selection, and emit ``regionSelected`` with the bounding box."""
        pts = list(self._poly_pts)
        self._reset_rectilinear_state()
        if len(pts) < 3:
            self.viewport().update()
            return
        # Guarantee an orthogonal closure: if the last vertex shares neither axis
        # with the first, insert an L-corner so every edge stays axis-aligned.
        last, first = pts[-1], pts[0]
        if last.x() != first.x() and last.y() != first.y():
            pts.append(QPoint(last.x(), first.y()))
        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        rect = QRect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        if rect.width() < 1 or rect.height() < 1:
            self.viewport().update()
            return
        mask = self._rasterize_polygon(pts, rect)
        if mask is None or not mask.any():
            self.viewport().update()
            return
        self._sel_polygon = pts
        self._selection_mask = mask
        self._template_rect = rect
        if self._overlay is not None:
            self._overlay.set_source_box(rect)
        self.viewport().update()
        self.regionSelected.emit(QRect(rect))

    @staticmethod
    def _rasterize_polygon(pts: "list[QPoint]", rect: QRect) -> "np.ndarray | None":
        """Rasterize the polygon (image-px vertices) into a bbox-sized bool mask."""
        w, h = rect.width(), rect.height()
        if w < 1 or h < 1:
            return None
        img = QImage(w, h, QImage.Format.Format_Grayscale8)
        img.fill(0)
        poly = QPolygon([QPoint(p.x() - rect.x(), p.y() - rect.y()) for p in pts])
        painter = QPainter(img)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(Qt.GlobalColor.white))
        painter.drawPolygon(poly)
        painter.end()
        bpl = img.bytesPerLine()
        arr = np.frombuffer(bytes(img.constBits()), np.uint8).reshape(h, bpl)[:, :w]
        return arr > 127

    # ----------------------------------------------------------- mouse events
    def mousePressEvent(self, e) -> None:
        """Begin pan (middle / space+left) or rubber-band selection (left)."""
        if self._doc is None:
            super().mousePressEvent(e)
            return
        btn = e.button()
        if btn == Qt.MouseButton.MiddleButton or (
            btn == Qt.MouseButton.LeftButton and self._is_pan_active()
        ):
            self._panning = True
            self._pan_last = e.position().toPoint()
            self._apply_cursor()
            e.accept()
            return
        if btn == Qt.MouseButton.LeftButton:
            if self._mode in (
                ImageViewport.Mode.CALIBRATE,
                ImageViewport.Mode.MEASURE,
            ):
                # Begin a two-point span (calibration or ruler) drag.
                self._measuring = True
                self._measure_kind = (
                    "calibrate" if self._mode is ImageViewport.Mode.CALIBRATE else "measure"
                )
                self._measure_p1 = self._clamped_image_point(e.position().toPoint())
                if self._measure_overlay is not None:
                    self._measure_overlay.set_pending(
                        self._measure_p1, self._measure_p1, self._measure_kind
                    )
                e.accept()
                return
            if self._mode is ImageViewport.Mode.RECTILINEAR:
                self._rectilinear_click(e.position().toPoint())
                e.accept()
                return
            # Start a rubber-band selection in viewport coordinates.
            self._selecting = True
            self._sel_origin = e.position().toPoint()
            self._rubber.setGeometry(QRect(self._sel_origin, self._sel_origin))
            self._rubber.show()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        """Double-click closes an in-progress rectilinear polygon."""
        if self._mode is ImageViewport.Mode.RECTILINEAR and len(self._poly_pts) >= 3:
            self._close_rectilinear()
            e.accept()
            return
        super().mouseDoubleClickEvent(e)

    def mouseMoveEvent(self, e) -> None:
        """Pan via scrollbars, extend the rubber-band, and emit cursor pos."""
        pos = e.position().toPoint()
        self._emit_cursor_pos(pos)

        if self._panning and self._pan_last is not None:
            delta = pos - self._pan_last
            self._pan_last = pos
            # Scrollbar-based pan coexists with the OpenGL viewport + rubber-band.
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            self._bump_generation()
            self._emit_view_changed()
            e.accept()
            return

        if self._measuring and self._measure_p1 is not None:
            p2 = self._clamped_image_point(pos)
            if self._measure_overlay is not None:
                self._measure_overlay.set_pending(self._measure_p1, p2, self._measure_kind)
            e.accept()
            return

        if self._selecting and self._sel_origin is not None:
            self._rubber.setGeometry(QRect(self._sel_origin, pos).normalized())
            e.accept()
            return

        if self._mode is ImageViewport.Mode.RECTILINEAR and self._poly_pts:
            # Live preview of the next axis-aligned edge from the last vertex.
            ip = self._clamped_image_point(pos)
            self._poly_preview = self._snap_rectilinear(self._poly_pts[-1], ip)
            self.viewport().update()
            e.accept()
            return

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        """Finish pan, or finalize the selection and emit ``regionSelected``."""
        if self._panning and e.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            self._pan_last = None
            self._apply_cursor()
            e.accept()
            return

        if self._measuring and e.button() == Qt.MouseButton.LeftButton:
            self._measuring = False
            p1, p2 = self._measure_p1, self._clamped_image_point(e.position().toPoint())
            self._measure_p1 = None
            if self._measure_overlay is not None:
                self._measure_overlay.set_pending(None, None, self._measure_kind)
            if p1 is not None and (abs(p2.x() - p1.x()) >= 1 or abs(p2.y() - p1.y()) >= 1):
                if self._measure_kind == "calibrate":
                    if self._measure_overlay is not None:
                        self._measure_overlay.set_calibration_line(p1, p2)
                    self.calibrationPicked.emit(p1, p2)
                else:
                    if self._measure_overlay is not None:
                        self._measure_overlay.set_measure(p1, p2)
                    self.measurePicked.emit(p1, p2)
            e.accept()
            return

        if self._selecting and e.button() == Qt.MouseButton.LeftButton:
            self._selecting = False
            self._rubber.hide()
            rect_img = self._viewport_band_to_image_rect()
            self._sel_origin = None
            if rect_img is not None and rect_img.width() >= 1 and rect_img.height() >= 1:
                # A plain rectangle clears any prior rectilinear polygon/mask.
                self._sel_polygon = None
                self._selection_mask = None
                self._template_rect = rect_img
                if self._overlay is not None:
                    self._overlay.set_source_box(rect_img)
                self.regionSelected.emit(QRect(rect_img))
            e.accept()
            return

        super().mouseReleaseEvent(e)

    # -------------------------------------------------------- coord helpers
    def _clamped_image_point(self, viewport_pt: QPoint) -> QPointF:
        """Map a viewport point to image-px coords (QPointF), clamped to the image."""
        sp = self.mapToScene(viewport_pt)
        w = float(self._doc.width) if self._doc is not None else 0.0
        h = float(self._doc.height) if self._doc is not None else 0.0
        return QPointF(min(max(0.0, sp.x()), w), min(max(0.0, sp.y()), h))
    def _viewport_band_to_image_rect(self) -> QRect | None:
        """Convert the rubber-band (viewport px) to a half-open image-px rect.

        Rounding rule (§F.3): floor the top-left, ceil the bottom-right so the
        box grows to cover the drag; then clip to the image.
        """
        band = self._rubber.geometry()
        if band.isNull() or self._doc is None:
            return None
        # Map the band as an AREA, not just its two inclusive corners. Mapping
        # only topLeft/bottomRight loses up to 1/view_scale image px on the
        # bottom-right when zoomed out (the QRect corners are inclusive integer
        # viewport px, so they under-cover the true dragged area). Mapping the
        # whole QRect returns a QPolygon whose bounding rect spans the real
        # selected scene area; flooring TL / ceiling BR then grows to cover it.
        sr = self.mapToScene(band).boundingRect()
        x0 = int(math.floor(sr.left()))
        y0 = int(math.floor(sr.top()))
        x1 = int(math.ceil(sr.right()))
        y1 = int(math.ceil(sr.bottom()))
        # Clip to image bounds (half-open).
        x0 = max(0, min(self._doc.width, x0))
        y0 = max(0, min(self._doc.height, y0))
        x1 = max(0, min(self._doc.width, x1))
        y1 = max(0, min(self._doc.height, y1))
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            return None
        return QRect(x0, y0, w, h)

    def _emit_cursor_pos(self, viewport_pt: QPoint) -> None:
        """Emit the image-px coordinate under the cursor (floor; -1,-1 outside)."""
        if self._doc is None:
            self.cursorImagePos.emit(-1, -1)
            return
        sp = self.mapToScene(viewport_pt)
        ix = int(math.floor(sp.x()))
        iy = int(math.floor(sp.y()))
        if 0 <= ix < self._doc.width and 0 <= iy < self._doc.height:
            self.cursorImagePos.emit(ix, iy)
        else:
            self.cursorImagePos.emit(-1, -1)

    # --------------------------------------------------------------- resize
    def resizeEvent(self, e) -> None:
        """Re-emit the visible rect on resize (prefetch hint stays current)."""
        super().resizeEvent(e)
        self._emit_view_changed()
