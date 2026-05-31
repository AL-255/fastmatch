"""Image loading and synthetic test-image generation for FastMatch.

This module turns a file path into the shared :class:`~fastmatch.document.ImageDocument`
source-of-truth, and can synthesize a ground-truth-labelled test image for the
engine self-test.

Memory discipline is the whole point here. A gigapixel RGB image is many
gigabytes; decoding it into one big in-RAM array can OOM-kill the process. So
:func:`load_image` pre-flights the pixel count and, above a configurable budget,
streams the decode row-strip by row-strip into a ``np.memmap`` (see
``DESIGN.md`` §F.4). Likewise :func:`generate_sample` writes its PNG strip by
strip instead of materializing the whole canvas at once (§F.5).

Every array this module hands back is C-contiguous ``(H, W, 3)`` uint8 — the
invariant ``ImageDocument`` enforces — whether it came from the in-RAM path or
the memmap path, so downstream consumers never special-case the source.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import weakref
from pathlib import Path

import numpy as np
from PIL import Image

from .document import ImageDocument
from .types import Match

# Pillow's decompression-bomb guard is already disabled package-wide in
# ``fastmatch/__init__`` (MAX_IMAGE_PIXELS = None); we still pre-flight the pixel
# count ourselves below, so importing this module re-applies that intent without
# relying on import order.
Image.MAX_IMAGE_PIXELS = None

# Decode strip height (rows) for the memmap path and for sample writing. 1024
# rows of a 12000-wide RGB image is ~35 MB — bounded peak RAM regardless of the
# full image height.
_STRIP_ROWS = 1024


def load_image(
    path: str | Path,
    *,
    max_ram_bytes: int = 6_000_000_000,
    max_image_bytes: int = 200_000_000_000,
) -> ImageDocument:
    """Load an image into an :class:`ImageDocument`, capping peak RAM.

    Opens ``path`` with Pillow and inspects its declared size without decoding
    pixels. If the full RGB buffer (``w * h * 3`` bytes) fits within
    ``max_ram_bytes`` it is decoded in one shot; otherwise the decode is streamed
    into a ``np.memmap`` so peak RAM stays bounded by one strip.

    Args:
        path: Image file on disk.
        max_ram_bytes: Upper bound on the in-RAM RGB buffer before we fall back
            to the memmap path. Defaults to ~6 GB (``DESIGN.md`` H table).
        max_image_bytes: Hard ceiling on the full RGB buffer (``w * h * 3``)
            regardless of path. Guards against a decompression bomb or a corrupt
            header declaring an absurd size; ``load_image`` refuses with a
            ``ValueError`` rather than attempting a multi-hundred-GB decode.

    Returns:
        An ``ImageDocument`` whose ``full`` is C-contiguous ``(H, W, 3)`` uint8
        (an ordinary array on the in-RAM path, a ``np.memmap`` otherwise).

    Raises:
        ValueError: If the declared image needs more than ``max_image_bytes``.
    """
    path = Path(path)
    with Image.open(path) as im:
        # ``im.size`` is (width, height) and is available from the header alone,
        # so this pre-flight happens before any pixel decode.
        w, h = im.size
        # ``w * h * 3`` can overflow a C int on huge images; Python ints are
        # arbitrary precision so this comparison is exact and safe.
        need = w * h * 3
        # Decompression-bomb / corrupt-header guard: refuse outright before we
        # allocate a RAM buffer or a disk-backed memmap of an absurd size.
        if need > max_image_bytes:
            raise ValueError(
                f"image {w}x{h} needs {need} bytes (RGB), exceeding the "
                f"max_image_bytes ceiling of {max_image_bytes}"
            )
        if need <= max_ram_bytes:
            # Small enough: decode fully. ``convert("RGB")`` collapses palette /
            # grayscale / RGBA inputs to the (H, W, 3) uint8 the document needs.
            arr = np.ascontiguousarray(np.asarray(im.convert("RGB")))
            return ImageDocument(full=arr, path=str(path), height=h, width=w)

    # Too large for one buffer: stream-decode into a memmap. We re-open inside
    # ``_load_via_memmap`` rather than holding the image handle open across the
    # whole strip loop.
    mm = _load_via_memmap(path, w, h)
    return ImageDocument(full=mm, path=str(path), height=h, width=w)


def _cleanup(tmp_name: str) -> None:
    """Remove the memmap backing file, tolerating an already-gone file."""
    try:
        os.remove(tmp_name)
    except FileNotFoundError:
        pass


def _load_via_memmap(path: str | Path, w: int, h: int) -> np.memmap:
    """Strip-decode ``path`` into a disk-backed ``np.memmap`` ``(h, w, 3)`` uint8.

    Decodes ``_STRIP_ROWS`` rows at a time via Pillow's ``crop`` (which only
    materializes the cropped region) and writes each strip into the memmap, so
    the in-RAM working set is one strip, not the whole image. Note that only
    TIFF is truly strip-addressable; Pillow decodes a PNG/JPEG whole on first
    pixel access, so for those formats the peak RAM benefit is the bounded write
    buffer rather than a bounded decode.

    The backing file lives under the system temp dir. A
    :func:`weakref.finalize` on the returned (read-only) memmap unlinks the file
    once that memmap is garbage-collected, so a large ``load_image`` no longer
    orphans a full-size temp file for the OS to reclaim later.

    Args:
        path: Image file on disk.
        w: Image width in px (from the pre-flight in :func:`load_image`).
        h: Image height in px.

    Returns:
        A ``(h, w, 3)`` uint8 ``np.memmap`` C-contiguous in row-major order. The
        backing temp file is unlinked when this memmap is GC'd.

    Raises:
        OSError: If the temp filesystem lacks free space for the backing file.
    """
    path = Path(path)
    need = w * h * 3  # backing-file size in bytes
    # A named temp file we can hand to np.memmap. delete=False so it survives the
    # ``with`` block; the memmap below opens it by name.
    fd, tmp_name = tempfile.mkstemp(suffix=".fmraw", prefix="fastmatch_")
    os.close(fd)  # np.memmap reopens by path; we only needed a unique filename.

    # Disk ceiling: don't create a memmap larger than the temp filesystem can
    # hold (a sparse mkstemp succeeds, but writing strips would later ENOSPC).
    free = shutil.disk_usage(tempfile.gettempdir()).free
    if need > free:
        _cleanup(tmp_name)  # remove the just-created (empty) temp file
        raise OSError(
            f"image {w}x{h} needs {need} bytes of temp space but only {free} "
            f"are free under {tempfile.gettempdir()}"
        )

    mm = np.memmap(tmp_name, dtype=np.uint8, mode="w+", shape=(h, w, 3))
    with Image.open(path) as im:
        rgb = im.convert("RGB") if im.mode != "RGB" else im
        for y0 in range(0, h, _STRIP_ROWS):
            y1 = min(y0 + _STRIP_ROWS, h)
            # crop box is (left, upper, right, lower), half-open on the lower/right
            # edges — exactly our row range [y0, y1).
            strip = rgb.crop((0, y0, w, y1))
            mm[y0:y1] = np.asarray(strip, dtype=np.uint8).reshape(y1 - y0, w, 3)

    # Flush dirty pages to disk and drop the writable mapping, then reopen
    # read-mode so consumers treat it as the read-only source-of-truth.
    mm.flush()
    del mm
    ro = np.memmap(tmp_name, dtype=np.uint8, mode="r", shape=(h, w, 3))
    # Unlink the backing file once the read-only memmap is collected; the
    # finalizer holds only the path string, never a reference to ``ro`` itself
    # (which would keep it alive forever).
    weakref.finalize(ro, _cleanup, tmp_name)
    return ro


def _make_motif(tile: int, rng: np.random.Generator) -> np.ndarray:
    """Build the distinctive asymmetric colored motif stamped into samples.

    The motif must be (a) high-contrast against the low-contrast background so
    matching is unambiguous, (b) *asymmetric* so rotation/flip ambiguity does not
    create false ground truth, and (c) *colored* so RGB channel mode is exercised.

    Returns:
        A ``(tile, tile, 3)`` uint8 RGB array.
    """
    m = np.zeros((tile, tile, 3), dtype=np.uint8)
    # Background of the motif tile: a dim teal so the bright shapes pop.
    m[:] = (20, 40, 40)

    t = tile
    # An L-shaped red bar hugging the top and left edges -> strong asymmetry.
    bar = max(2, t // 6)
    m[: bar * 2, :bar] = (230, 30, 30)          # vertical stub of the L (left)
    m[:bar, : t // 2] = (230, 30, 30)            # horizontal stub of the L (top)
    # A green diagonal wedge in the lower-right quadrant (breaks any symmetry).
    yy, xx = np.mgrid[0:t, 0:t]
    wedge = (xx + yy) > int(1.35 * t)
    m[wedge] = (30, 210, 80)
    # A bright blue square offset from center -> a third distinctive landmark.
    cx, cy = int(t * 0.62), int(t * 0.30)
    s = max(3, t // 7)
    m[cy : cy + s, cx : cx + s] = (40, 90, 240)
    # A little salt of yellow pixels so per-channel variance is non-trivial.
    n_dots = max(4, t // 8)
    ys = rng.integers(0, t, size=n_dots)
    xs = rng.integers(0, t, size=n_dots)
    m[ys, xs] = (250, 230, 40)
    return m


def generate_sample(
    path: str | Path,
    w: int = 12000,
    h: int = 12000,
    *,
    tile: int = 200,
    n_targets: int = 40,
    seed: int = 0,
) -> list[Match]:
    """Synthesize a textured PNG with known motif stamps; return ground truth.

    Builds a low-contrast sine-grating + noise background and stamps one
    distinctive asymmetric colored motif (size ``tile``) at ``n_targets``
    non-overlapping random positions. Per ``DESIGN.md`` §F.5: some stamps
    straddle the ``tile``-spaced grid lines (to exercise the engine's seam math)
    and half are brightness-jittered ±5% (to exercise NCC's brightness
    invariance). The PNG is written strip-by-strip to keep peak RAM bounded.

    Args:
        path: Destination PNG path.
        w: Canvas width in px.
        h: Canvas height in px.
        tile: Motif edge length in px (also the grid spacing for seam straddles).
        n_targets: Number of stamps to place.
        seed: Seed for the local numpy ``Generator`` (no global RNG state touched).

    Returns:
        The ground-truth ``list[Match]`` — one per stamp, ``scale=1.0``,
        ``score=1.0``, in image-px half-open ``(x, y, w, h)`` coordinates.
    """
    path = Path(path)
    rng = np.random.default_rng(seed)

    motif = _make_motif(tile, rng)

    # --- Choose non-overlapping stamp positions -----------------------------
    # Reject-sample positions; require a 1px gap between motif boxes so no two
    # stamps fuse into an ambiguous blob. Bias roughly half the candidates onto
    # a tile-grid line so the engine's halo/seam handling is genuinely tested.
    placed: list[tuple[int, int]] = []
    matches: list[Match] = []
    max_x = w - tile
    max_y = h - tile
    if max_x < 0 or max_y < 0:
        raise ValueError(f"tile {tile} larger than canvas {w}x{h}")

    # Whether seam-straddling is even geometrically possible: a stamp straddles a
    # ``tile`` grid line only if a grid line exists strictly inside [0, max_x].
    can_straddle = tile > 1 and max_x >= tile

    attempts = 0
    attempt_cap = 4000 * max(1, n_targets)
    while len(placed) < n_targets and attempts < attempt_cap:
        attempts += 1
        # Bias roughly every other stamp toward a seam straddle, but always start
        # from a *uniformly random* base position so positional entropy stays high
        # (snapping straight to grid multiples on a small canvas would exhaust the
        # handful of grid cells and never converge).
        x = int(rng.integers(0, max_x + 1))
        y = int(rng.integers(0, max_y + 1))
        if can_straddle and len(placed) % 2 == 0:
            # Nudge the top-left so the box crosses the nearest interior grid line
            # (top-left lands just before a multiple of ``tile``), keeping it in range.
            gx = int(round((x + tile / 2.0) / tile)) * tile
            gy = int(round((y + tile / 2.0) / tile)) * tile
            x = int(np.clip(gx - tile // 2, 0, max_x))
            y = int(np.clip(gy - tile // 2, 0, max_y))

        # Overlap rejection: axis-aligned box separation with a 1px guard band so
        # no two stamps fuse into one ambiguous blob (touching is also rejected).
        ok = True
        for (px, py) in placed:
            if abs(px - x) < tile + 1 and abs(py - y) < tile + 1:
                ok = False
                break
        if not ok:
            continue

        placed.append((x, y))
        matches.append(Match(x=x, y=y, w=tile, h=tile, score=1.0, scale=1.0))

    if len(placed) < n_targets:
        raise RuntimeError(
            f"could only place {len(placed)}/{n_targets} non-overlapping motifs "
            f"in {w}x{h}; reduce n_targets or enlarge the canvas"
        )

    # --- Render + write strip-by-strip --------------------------------------
    # The background is a low-contrast sum of two sine gratings plus mild noise.
    # We generate it per strip so the full (h, w, 3) float canvas never exists.
    xs = np.arange(w, dtype=np.float32)
    base_gray = 128.0
    amp = 10.0  # low contrast: keeps background variance well below the motif's

    # Pre-jitter decision per stamp (half get a ±5% brightness scale).
    jitter = {}
    for i, (x, y) in enumerate(placed):
        if i % 2 == 1:
            # Uniform in [0.95, 1.05]; applied multiplicatively to the stamped motif.
            jitter[(x, y)] = float(rng.uniform(0.95, 1.05))

    # Pillow has no public incremental-PNG-row writer, so the destination pixels
    # live in one ``Image`` buffer (the unavoidable single uint8 copy: ~432 MB at
    # 12000^2). What we *do* bound strip-by-strip is the expensive transient
    # working set — the float gratings/noise arrays — which would otherwise be a
    # second full-image float buffer (4x larger) held simultaneously.
    canvas = Image.new("RGB", (w, h))
    try:
        for y0 in range(0, h, _STRIP_ROWS):
            y1 = min(y0 + _STRIP_ROWS, h)
            rows = y1 - y0
            yy = np.arange(y0, y1, dtype=np.float32)[:, None]
            # Two orthogonal gratings -> a faint plaid; noise breaks periodicity.
            grating = (
                np.sin(xs[None, :] / 37.0) + np.cos(yy / 53.0)
            ).astype(np.float32)
            noise = rng.standard_normal((rows, w)).astype(np.float32)
            gray = base_gray + amp * grating + 4.0 * noise
            strip = np.clip(gray, 0, 255).astype(np.uint8)
            strip_rgb = np.repeat(strip[:, :, None], 3, axis=2)

            # Stamp any motif intersecting this strip's row range.
            for (x, y) in placed:
                if y + tile <= y0 or y >= y1:
                    continue
                m = motif
                if (x, y) in jitter:
                    m = np.clip(motif.astype(np.float32) * jitter[(x, y)], 0, 255).astype(
                        np.uint8
                    )
                # Intersect the motif's rows with this strip.
                src_y0 = max(0, y0 - y)
                src_y1 = min(tile, y1 - y)
                dst_y0 = y + src_y0 - y0
                dst_y1 = y + src_y1 - y0
                strip_rgb[dst_y0:dst_y1, x : x + tile] = m[src_y0:src_y1]

            canvas.paste(Image.fromarray(strip_rgb, mode="RGB"), (0, y0))
        canvas.save(path, format="PNG")
    finally:
        canvas.close()

    return matches
