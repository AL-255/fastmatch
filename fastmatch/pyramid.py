"""Display pyramid: level descriptors, lazy block-mean downsampling, tile LRU.

The viewport shows a gigapixel image as a tiled mip pyramid. Level 0 is the
full-resolution source (``ImageDocument.full``); each subsequent level halves
both dimensions via 2x2 block-mean. We never materialize a whole level — a tile
is computed on demand from the appropriate source slice, so this works directly
on a ``np.memmap`` backing array (slice first, downsample the small slice).

Scene/coordinate convention matches the rest of FastMatch: integer image
pixels, origin top-left, ``arr[y, x]`` indexing. ``Level`` geometry is expressed
in that level's *own* resolution (level 0 == full-res image px).

This module is GUI-adjacent: ``TileCache`` stores ``QPixmap`` objects so it
imports Qt, but ``ImagePyramid`` itself is pure numpy and Qt-free.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .document import ImageDocument

# Tile edge length, in each level's own resolution. 256 keeps single textures
# far below the GL_MAX_TEXTURE_SIZE wall (typically >= 16384) and bounds the
# per-tile decode/upload cost. Shared as the public default for ImagePyramid.
TILE = 256


@dataclass(frozen=True)
class Level:
    """Geometry of one pyramid level, in that level's own pixel resolution.

    Attributes:
        index: Level number; 0 == full-resolution source.
        scale: Level px per full-res px (``1 / 2**index``). Multiply a full-res
            coordinate by ``scale`` to get the level-local coordinate.
        w: Level width in level px.
        h: Level height in level px.
        cols: Number of tile columns (``ceil(w / tile)``).
        rows: Number of tile rows (``ceil(h / tile)``).
    """

    index: int
    scale: float
    w: int
    h: int
    cols: int
    rows: int


class ImagePyramid:
    """Lazy mip pyramid over an :class:`ImageDocument`.

    Builds only the cheap per-level *descriptors* up front; pixel data for any
    given tile/region is produced on demand by :meth:`region`. Downsampling is
    2x2 block-mean applied directly from the full-resolution source for the
    requested region only, so a level is never held whole in memory.
    """

    def __init__(self, doc: ImageDocument, tile: int = TILE) -> None:
        """Plan level descriptors for ``doc``.

        Args:
            doc: The source document. ``doc.full`` may be a ``np.memmap``.
            tile: Tile edge in level px (defaults to module-level ``TILE``).
        """
        if tile <= 0:
            raise ValueError(f"tile must be positive, got {tile}")
        self._doc = doc
        self._tile = int(tile)
        self.full_w: int = int(doc.width)
        self.full_h: int = int(doc.height)

        # L = max(1, ceil(log2(max(W,H)/TILE)) + 1): enough levels that the
        # coarsest fits within a few tiles. The +1 accounts for level 0 itself.
        longest = max(self.full_w, self.full_h)
        if longest <= self._tile:
            num = 1
        else:
            num = max(1, math.ceil(math.log2(longest / self._tile)) + 1)

        levels: list[Level] = []
        for i in range(num):
            # Each level halves dims; clamp to >= 1 so we never produce a 0-size
            # level (matters for very tall/thin images where one axis hits 1 fast).
            w = max(1, self.full_w >> i)
            h = max(1, self.full_h >> i)
            cols = max(1, math.ceil(w / self._tile))
            rows = max(1, math.ceil(h / self._tile))
            levels.append(Level(index=i, scale=1.0 / (1 << i), w=w, h=h, cols=cols, rows=rows))
        self.levels: list[Level] = levels

    @property
    def tile(self) -> int:
        """Tile edge length in level px."""
        return self._tile

    def num_levels(self) -> int:
        """Number of pyramid levels (>= 1)."""
        return len(self.levels)

    def region(self, level: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Return a ``(h, w, 3)`` uint8 crop at ``level`` resolution.

        The crop is computed lazily by block-mean downsampling the corresponding
        slice of the full-resolution source. Requested coordinates are in level
        px; the method clips to the level bounds (the returned array therefore
        has at most the requested size — callers must read ``arr.shape``).

        Implementation note (memmap safety): we first slice the source array to
        exactly the source-px window backing this level region, then downsample
        that small slice. We never reshape an entire level into memory.

        Args:
            level: Level index in ``[0, num_levels())``.
            x, y: Top-left of the region, level px.
            w, h: Region size, level px.

        Returns:
            A C-contiguous ``(rh, rw, 3)`` uint8 array (``rw <= w``, ``rh <= h``).
        """
        if not (0 <= level < len(self.levels)):
            raise IndexError(f"level {level} out of range [0, {len(self.levels)})")
        lv = self.levels[level]

        # Clip the requested level-px window to the level's actual extent.
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(lv.w, int(x) + int(w))
        y1 = min(lv.h, int(y) + int(h))
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        factor = 1 << level  # source px per level px along each axis
        if factor == 1:
            # Level 0 is the source itself; just hand back a contiguous copy of
            # the slice (a copy keeps memmap pages from leaking into a QImage).
            crop = self._doc.full[y0:y1, x0:x1, :]
            return np.ascontiguousarray(crop)

        # Source window that maps onto the requested level window. We expand to
        # cover full 2x2 (factor x factor) source blocks per output pixel.
        sx0 = x0 * factor
        sy0 = y0 * factor
        sx1 = min(self.full_w, x1 * factor)
        sy1 = min(self.full_h, y1 * factor)
        src = self._doc.full[sy0:sy1, sx0:sx1, :]
        out = self._block_mean(src, factor)

        # Defense-in-depth: guarantee the result matches the exact level
        # geometry (y1-y0, x1-x0). Independent-axis folding plus integer
        # halving can leave the block-mean output a pixel off the requested
        # window for extreme-aspect levels; cropping/padding here keeps the
        # downstream pixmap-width alignment exact.
        want_h = y1 - y0
        want_w = x1 - x0
        oh, ow = out.shape[0], out.shape[1]
        if oh == want_h and ow == want_w:
            return out
        fixed = np.zeros((want_h, want_w, 3), dtype=np.uint8)
        ch = min(oh, want_h)
        cw = min(ow, want_w)
        fixed[:ch, :cw, :] = out[:ch, :cw, :]
        return fixed

    @staticmethod
    def _block_mean(src: np.ndarray, factor: int) -> np.ndarray:
        """Average non-overlapping ``factor x factor`` blocks of ``src``.

        Performs the reduction in successive 2x2 halving passes. Doing repeated
        2x2 mirrors how each pyramid level derives from the one above it and
        keeps the temporary buffers small. Odd trailing rows/cols are dropped
        (a half-block at the far edge is averaged with its available neighbours,
        which is visually adequate for a display pyramid).

        Each axis folds *independently* per pass: an extreme-aspect slice can
        collapse one axis to a single pixel (e.g. a 40000x130 image at a deep
        level) while the other axis still has dozens of pixels left to halve.
        Aborting both axes when only one can no longer fold would return a
        wrong-sized region, so we fold whichever axis still can.

        Args:
            src: ``(sh, sw, 3)`` uint8 source slice.
            factor: Total downsample factor (a power of two: 2, 4, 8, ...).

        Returns:
            C-contiguous uint8 array, roughly ``(sh//factor, sw//factor, 3)``;
            an axis that exhausts its halving passes early (extreme aspect) is
            reduced as far as it can fold (:meth:`region` then crops/pads the
            result to the exact level geometry).
        """
        # Accumulate in uint16 then average; uint8 sums of 4 would overflow.
        buf = src
        passes = factor.bit_length() - 1  # log2(factor); factor is a power of two
        for _ in range(passes):
            sh, sw = buf.shape[0], buf.shape[1]
            # Drop a trailing odd row/col so we can fold cleanly into 2x2 blocks.
            sh2 = sh - (sh & 1)
            sw2 = sw - (sw & 1)
            fold_h = sh2 >= 2
            fold_w = sw2 >= 2
            if not fold_h and not fold_w:
                # Neither axis can halve further; stop early with what we have.
                break
            if fold_h and fold_w:
                # Normal/square path: 2x2 block-mean. Sum the four members of
                # each block, then divide by 4 (round to nearest with +2 bias)
                # — cheap and avoids per-element float work.
                b = buf[:sh2, :sw2, :].astype(np.uint16)
                summed = (
                    b[0::2, 0::2, :]
                    + b[1::2, 0::2, :]
                    + b[0::2, 1::2, :]
                    + b[1::2, 1::2, :]
                )
                buf = ((summed + 2) >> 2).astype(np.uint8)
            elif fold_h:
                # Only the vertical axis can fold: average vertical pairs.
                b = buf[:sh2, :, :].astype(np.uint16)
                buf = ((b[0::2, :, :] + b[1::2, :, :] + 1) >> 1).astype(np.uint8)
            else:  # fold_w only
                # Only the horizontal axis can fold: average horizontal pairs.
                b = buf[:, :sw2, :].astype(np.uint16)
                buf = ((b[:, 0::2, :] + b[:, 1::2, :] + 1) >> 1).astype(np.uint8)
        return np.ascontiguousarray(buf)


# QPixmap is only needed by TileCache; import Qt lazily-friendly at module top
# since this module is part of the viewport layer (already Qt-coupled in use).
from PySide6.QtGui import QPixmap  # noqa: E402  (kept below numpy/pyramid core)


class TileCache:
    """LRU cache of decoded display ``QPixmap`` tiles, keyed by ``(level, cx, cy)``.

    Bounded by a pixmap count (~64 MB at 256 pixmaps of 256x256 RGB). The two
    coarsest levels are *pinned*: they are never evicted, so the painter always
    has a blurry coarse fallback to draw while finer tiles stream in.
    """

    def __init__(self, max_pixmaps: int = 256, pinned_levels: int = 2) -> None:
        """Create the cache.

        Args:
            max_pixmaps: Soft cap on non-pinned cached pixmaps; LRU evicts above.
            pinned_levels: How many of the coarsest levels to pin once their
                indices are known via :meth:`pin_top_levels`.
        """
        self._max = max(1, int(max_pixmaps))
        self._pinned_levels = max(0, int(pinned_levels))
        # OrderedDict acts as the LRU: most-recently-used at the end.
        self._lru: "OrderedDict[tuple[int, int, int], QPixmap]" = OrderedDict()
        # Pinned tiles live in a separate dict and never count against the cap.
        self._pinned: dict[tuple[int, int, int], QPixmap] = {}
        self._pinned_level_set: set[int] = set()

    def set_num_levels(self, num_levels: int) -> None:
        """Record which level indices count as 'top' (coarsest) for pinning.

        Pinning is by *level index*: the ``pinned_levels`` coarsest levels (the
        highest indices) qualify. Call once when the pyramid is (re)built.
        """
        if num_levels <= 0:
            self._pinned_level_set = set()
            return
        first_pinned = max(0, num_levels - self._pinned_levels)
        self._pinned_level_set = set(range(first_pinned, num_levels))

    def pin_top_levels(self) -> None:
        """Promote every currently-cached coarsest-level tile to pinned status.

        Spec contract name (§C.9). Moves matching tiles out of the LRU into the
        unbounded pinned set so subsequent eviction can never drop them.
        """
        for key in list(self._lru.keys()):
            if key[0] in self._pinned_level_set:
                self._pinned[key] = self._lru.pop(key)

    def get(self, key: tuple[int, int, int]) -> "QPixmap | None":
        """Return the cached pixmap for ``(level, cx, cy)`` or ``None``.

        A hit on a non-pinned tile refreshes its LRU recency.
        """
        pm = self._pinned.get(key)
        if pm is not None:
            return pm
        pm = self._lru.get(key)
        if pm is not None:
            self._lru.move_to_end(key)  # mark most-recently-used
        return pm

    def put(self, key: tuple[int, int, int], pm: QPixmap) -> None:
        """Insert ``pm`` for ``(level, cx, cy)``.

        Coarsest-level tiles go straight to the pinned set; all others enter the
        LRU and may trigger eviction of the least-recently-used tile.
        """
        if key[0] in self._pinned_level_set:
            self._pinned[key] = pm
            return
        if key in self._lru:
            self._lru.move_to_end(key)
        self._lru[key] = pm
        while len(self._lru) > self._max:
            self._lru.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        """Drop every cached pixmap (pinned included)."""
        self._lru.clear()
        self._pinned.clear()

    def __len__(self) -> int:
        """Total cached pixmaps (pinned + LRU)."""
        return len(self._lru) + len(self._pinned)
