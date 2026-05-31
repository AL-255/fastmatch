"""Coordinate-contract tests for the canonical image-pixel frame (DESIGN §F.3).

These tests are deliberately **self-contained**: they import no Qt and no
sibling FastMatch GUI modules. The float-rect -> integer-box mapping helper is
re-implemented here so the test pins the *contract* (DESIGN §F.3 / §M1) rather
than coupling to the viewport's private implementation. If the viewport ever
drifts from "floor top-left / ceil bottom-right", these tests still describe the
behaviour everything outside the viewport relies on.

Canonical frame (from ``fastmatch.types``):
    integer image px, origin top-left, +x right / +y down, box ``(x, y, w, h)``
    **half-open** so ``full[y:y+h, x:x+w]`` is an ``(h, w, 3)`` crop.
"""

from __future__ import annotations

import math

import numpy as np

from fastmatch.types import Match


# --------------------------------------------------------------------------- #
# Helper under test — mirrors the viewport's float-rect -> integer-box rule.
# DESIGN §F.3: "float->int selection uses floor top-left / ceil bottom-right
# (box grows to cover the drag)". This is intentionally a free function with no
# Qt dependency so the coordinate contract can be verified headlessly.
# --------------------------------------------------------------------------- #
def rect_to_box(
    fx: float,
    fy: float,
    fw: float,
    fh: float,
    *,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Map a float drag rect to a half-open integer image-px box, clipped.

    Applies floor to the top-left and ceil to the bottom-right so the integer
    box always *covers* the dragged float region, then clips to the image so the
    half-open invariants ``x + w <= img_w`` and ``y + h <= img_h`` hold.

    Args:
        fx, fy: Float top-left corner (image px).
        fw, fh: Float width/height (image px); negative extents (drag up/left)
            are normalised to a positive box.
        img_w, img_h: Image bounds used for clipping.

    Returns:
        ``(x, y, w, h)`` integer half-open box, clipped to the image.
    """
    # Normalise so width/height are non-negative regardless of drag direction.
    x0 = min(fx, fx + fw)
    y0 = min(fy, fy + fh)
    x1 = max(fx, fx + fw)
    y1 = max(fy, fy + fh)

    # floor TL / ceil BR -> the integer box grows to cover the float drag.
    ix0 = math.floor(x0)
    iy0 = math.floor(y0)
    ix1 = math.ceil(x1)
    iy1 = math.ceil(y1)

    # Clip to the image (half-open: the right/bottom edge may equal the size).
    ix0 = max(0, min(ix0, img_w))
    iy0 = max(0, min(iy0, img_h))
    ix1 = max(0, min(ix1, img_w))
    iy1 = max(0, min(iy1, img_h))

    return ix0, iy0, ix1 - ix0, iy1 - iy0


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_floor_tl_ceil_br_grows_to_cover() -> None:
    """A sub-pixel float rect expands outward to integer boundaries."""
    x, y, w, h = rect_to_box(10.2, 20.8, 30.1, 40.9, img_w=1000, img_h=1000)
    # TL floored: 10.2 -> 10, 20.8 -> 20.
    assert (x, y) == (10, 20)
    # BR ceiled: (10.2 + 30.1) = 40.3 -> 41 ; (20.8 + 40.9) = 61.7 -> 62.
    assert x + w == 41
    assert y + h == 62
    # Resulting box strictly covers the float region.
    assert x <= 10.2 and y <= 20.8
    assert x + w >= 40.3 and y + h >= 61.7


def test_already_integer_rect_is_unchanged() -> None:
    """Integer-aligned inputs map to themselves (floor/ceil are identities)."""
    x, y, w, h = rect_to_box(5.0, 7.0, 12.0, 9.0, img_w=100, img_h=100)
    assert (x, y, w, h) == (5, 7, 12, 9)


def test_negative_drag_is_normalised() -> None:
    """Dragging up-and-left yields the same box as dragging down-and-right."""
    down_right = rect_to_box(10.0, 10.0, 20.0, 30.0, img_w=200, img_h=200)
    # Same rectangle expressed by starting at the opposite corner.
    up_left = rect_to_box(30.0, 40.0, -20.0, -30.0, img_w=200, img_h=200)
    assert down_right == up_left == (10, 10, 20, 30)


def test_crop_shape_is_h_w_3() -> None:
    """Cropping ``full[y:y+h, x:x+w]`` yields exactly ``(h, w, 3)``."""
    img_h, img_w = 120, 160
    full = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    x, y, w, h = rect_to_box(12.3, 4.7, 40.2, 33.6, img_w=img_w, img_h=img_h)
    crop = full[y : y + h, x : x + w]
    assert crop.shape == (h, w, 3)
    # NumPy slicing must not have silently clamped to a smaller window.
    assert crop.shape[0] == h and crop.shape[1] == w


def test_match_box_round_trips_with_le_1px_drift() -> None:
    """A Match built at (x,y,w,h) maps back to the same box, drift <= 1px.

    Simulates: float drag -> integer box -> crop template -> engine returns a
    Match at the same location -> overlay draws box(x,y,w,h). Every hop stays in
    the one integer frame, so the round-trip is exact (drift 0, well within 1px).
    """
    img_w, img_h = 1000, 800
    fx, fy, fw, fh = 100.4, 200.6, 64.3, 48.9
    x, y, w, h = rect_to_box(fx, fy, fw, fh, img_w=img_w, img_h=img_h)

    # Engine output for a true self-hit at the source location.
    m = Match(x=x, y=y, w=w, h=h, score=1.0, scale=1.0)

    # Overlay consumes (m.x, m.y, m.w, m.h) verbatim — no conversion (DESIGN §E.9).
    back = (m.x, m.y, m.w, m.h)
    assert back == (x, y, w, h)

    # Drift of every corner from the *integer* box is zero; from the float drag
    # it is bounded by 1px (a single floor/ceil step on each side).
    assert abs(m.x - x) <= 1 and abs(m.y - y) <= 1
    assert abs((m.x + m.w) - (x + w)) <= 1 and abs((m.y + m.h) - (y + h)) <= 1
    assert m.x <= fx and m.y <= fy
    assert (m.x + m.w) >= (fx + fw) and (m.y + m.h) >= (fy + fh)


def test_half_open_invariants_after_clipping() -> None:
    """Boxes that overflow the image are clipped to satisfy x+w<=W, y+h<=H."""
    img_w, img_h = 256, 192
    # Drag that starts inside but runs well past the bottom-right corner.
    x, y, w, h = rect_to_box(200.0, 150.0, 999.0, 999.0, img_w=img_w, img_h=img_h)
    assert x >= 0 and y >= 0
    assert w > 0 and h > 0
    # Half-open: the exclusive right/bottom edge may *equal* the size, never exceed.
    assert x + w <= img_w
    assert y + h <= img_h
    assert x + w == img_w  # clipped exactly to the right edge
    assert y + h == img_h  # clipped exactly to the bottom edge

    # And the resulting crop is consistent with the box dimensions.
    full = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    crop = full[y : y + h, x : x + w]
    assert crop.shape == (h, w, 3)


def test_clip_keeps_box_within_image_for_partial_overflow() -> None:
    """A float rect partly off the top-left clips its TL to 0, BR stays in-bounds."""
    img_w, img_h = 300, 300
    x, y, w, h = rect_to_box(-5.7, -3.2, 50.0, 40.0, img_w=img_w, img_h=img_h)
    # Negative TL is clipped up to the origin.
    assert x == 0 and y == 0
    # BR is ceil of the float far corner: ceil(-5.7+50)=45 (>0, in-bounds);
    # ceil(-3.2+40)=37. Width/height shrink because the TL was clipped to 0.
    assert x + w <= img_w and y + h <= img_h
    assert (x + w, y + h) == (45, 37)
