"""Device-agnostic, tiled, multi-scale normalized-cross-correlation engine.

This module implements :class:`Matcher`, the GPU-accelerated (CUDA when present,
CPU otherwise) template-search core of FastMatch. It has **zero Qt imports** —
it speaks only numpy/torch and the Qt-free dataclasses in :mod:`fastmatch.types`
so it can run on a worker thread (or, in principle, out-of-process) without
dragging GUI types along.

Algorithm overview (see ``docs/DESIGN.md`` §D for the authoritative spec):

* Fast NCC via **separable box filters** for the windowed sum ``S1`` and
  sum-of-squares ``S2`` (so the per-window variance costs ``O(th+tw)`` not
  ``O(th*tw)``), plus a single full 2-D cross-correlation for the cross term
  ``sum_w(I * Tz)`` using a zero-mean template ``Tz``. Box filters (not a global
  summed-area table) keep float32 round-off bounded by window size rather than
  image size.
* The cross-correlation uses spatial ``F.conv2d`` for small templates and an
  FFT (``torch.fft.rfft2``) once ``th*tw > 4096`` where the FFT's
  ``O(N log N)`` beats the dense conv.
* The output grid is processed in **non-overlapping core tiles** with a halo of
  ``(th-1, tw-1)`` on the right/bottom so no valid top-left position is ever
  split across a tile boundary. Only detections centered in the tile core are
  kept; the final global NMS dedupes any seam straddlers.
* Multi-scale scales the *template* (the per-image precompute is reused), pools
  candidates from every scale, then runs **one** global NMS so the best scale
  wins at each location.
* On CUDA an OOM degradation ladder (1024 -> 512 tile -> fewer scales -> CPU)
  keeps the search alive instead of crashing.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import replace

from .device import resolve_device
from .types import (
    CONV_METHODS,
    METHODS,
    SHAZAM_METHOD,
    Match,
    MatchParams,
    active_orientations,
    apply_orientation,
    normalize_weights,
)

# torchvision is an *optional* dependency: we use its fused NMS kernel when it
# is importable, but ship a pure-torch greedy NMS fallback so the engine has no
# hard runtime dependency on it.
try:  # pragma: no cover - exercised only when torchvision is installed
    from torchvision.ops import nms as _tv_nms

    _HAVE_TV_NMS = True
except Exception:  # ImportError, or a broken/partial torchvision install
    _tv_nms = None
    _HAVE_TV_NMS = False


# --- module constants -------------------------------------------------------

#: BT.601 luminance weights (R, G, B). Matches the convention stated in §D.1.
_BT601 = (0.299, 0.587, 0.114)

#: Full-range BT.601 (JPEG) RGB[0,255] -> YCbCr[0,255]: rows are (wR, wG, wB, bias)
#: for Y, Cb, Cr. Used to derive the staged ycbcr planes for channel_mode="ycbcr".
_YCBCR_FROM_RGB = (
    (0.299, 0.587, 0.114, 0.0),
    (-0.168736, -0.331264, 0.5, 128.0),
    (0.5, -0.418688, -0.081312, 128.0),
)

#: Below this NCC numerator threshold a template is considered featureless and
#: rejected (variance on the [0,1]-normalized template). Mirrors §H ``var_floor``.
_VAR_FLOOR = 1e-3

#: Minimum template side accepted (px). §H "min template side".
_MIN_SIDE = 8

#: ``th*tw`` above which the cross term switches to FFT (§D.2).
_FFT_THRESHOLD = 4096

#: Conservative starting core-tile size on CUDA before the formula/ladder
#: refine it (§D.7 / §H "compute tile (start)").
_TILE_START = 1024

#: Degrade ladder of tile sizes on CUDA OOM (§D.7).
_TILE_LADDER = (1024, 512)

#: Cap on the local-max prefilter neighborhood (§D.5). The prefilter thins
#: candidates before IoU-NMS; a full template-sized max-pool is O(positions *
#: th * tw), which is pathological on CPU for large templates (it dominates an
#: otherwise-cheap tile and, being a single atomic kernel, makes that tile
#: uninterruptible — wedging cancellation/shutdown). Capping the neighborhood
#: bounds per-tile cost; any extra peaks within one template footprint overlap
#: heavily and are merged by the global IoU-NMS, so detections are unchanged.
_LOCALMAX_MAX_KERNEL = 48

#: 2-3-5-7 smooth radices used to pad FFT lengths to a fast size.
_SMOOTH_RADICES = (2, 3, 5, 7)


# --- coarse-to-fine image-pyramid search constants (§K.1) -------------------

#: Smallest template side (px) we keep at the coarsest pyramid level. The level
#: count is chosen so the down-sampled template never drops below this — below
#: it a blurred template loses the structure that makes a peak detectable.
_MIN_COARSE_SIDE = 16

#: Hard cap on pyramid levels above level 0 (§K.1). Bounds memory and the number
#: of refinement passes regardless of how big the image/template are.
_MAX_PYR_LEVELS = 4

#: Relaxation subtracted from the (clamped) threshold at the coarse level so a
#: true peak that looks weaker once blurred is NOT pruned before refinement
#: (§K.1). Conservative on purpose — recall parity beats a tighter coarse cut.
_COARSE_MARGIN = 0.20

#: Half-width (px, at each refined level's own resolution) of the search window
#: opened around each ×2-mapped coarse candidate. Absorbs the ×2 mapping plus
#: the avg-pool down-sampling rounding uncertainty (§K.1).
_REFINE_RADIUS = 3

#: Below this template×scale pixel count the coarse-to-fine machinery's fixed
#: overhead (pyramid pooling of the template, ROI gather/scatter) outweighs the
#: tiny full-res conv it would replace, so the full path is used instead. Keeps
#: the small-image/test scenes on the unchanged full path and auto-enables the
#: pyramid only when it actually pays (§K.1 "auto-enable only when beneficial").
_PYRAMID_MIN_TEMPLATE_PX = 64 * 64

#: Below this image pixel count the full-res search is already cheap enough that
#: the pyramid's overhead is not worth it (and keeps every existing small-image
#: test on the byte-identical full path).
_PYRAMID_MIN_IMAGE_PX = 1024 * 1024


def _next_smooth(n: int) -> int:
    """Smallest integer ``>= n`` whose only prime factors are 2, 3, 5, 7.

    FFT cost is dominated by the worst prime factor of the transform length;
    padding to a 2-3-5-7-smooth size keeps ``torch.fft.rfft2`` on its fast
    radix paths instead of a slow generic Bluestein transform.
    """
    if n <= 1:
        return 1
    best = 2 * n  # any power-of-two upper bound is smooth, so this terminates
    # Brute-force the small search space of 2^a*3^b*5^c*7^d <= 2n; the ranges
    # are tiny for the tile-sized lengths we ever pad here.
    p2 = 1
    while p2 < best:
        p23 = p2
        while p23 < best:
            p235 = p23
            while p235 < best:
                p2357 = p235
                while p2357 < best:
                    if p2357 >= n:
                        best = p2357
                    p2357 *= _SMOOTH_RADICES[3]
                p235 *= _SMOOTH_RADICES[2]
            p23 *= _SMOOTH_RADICES[1]
        p2 *= _SMOOTH_RADICES[0]
    return best


def _max_filter_2d(x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    """Stride-1 ``(kh x kw)`` sliding-window maximum, computed *separably*.

    A 2-D window max is separable — ``max`` over the window equals a row-wise
    1-D max followed by a column-wise 1-D max — so two 1-D max-pools cost
    ``O(kh + kw)`` per pixel instead of the dense 2-D pool's ``O(kh*kw)``. For the
    local-max prefilter's large (up to 48x48) kernels this is a ~20-50x cut in
    work and was, by the profiler, the single biggest CPU cost; each 1-D pass is
    a contiguous, SIMD-friendly reduction. The result is identical (to the last
    bit) to ``F.max_pool2d(x[None,None], (kh,kw), stride=1, padding=(kh//2,kw//2))``
    cropped back to ``x``'s size — max-pool's implicit ``-inf`` padding makes both
    passes agree with the dense pool. ``x`` is 2-D ``(H, W)``; returns 2-D.
    """
    h, w = x.shape[-2], x.shape[-1]
    t = x.unsqueeze(0).unsqueeze(0)
    if kh > 1:
        t = F.max_pool2d(t, kernel_size=(kh, 1), stride=1, padding=(kh // 2, 0))
    if kw > 1:
        t = F.max_pool2d(t, kernel_size=(1, kw), stride=1, padding=(0, kw // 2))
    return t[0, 0, :h, :w]


def _greedy_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Pure-torch greedy IoU NMS — the fallback when torchvision is absent.

    Args:
        boxes: ``(N, 4)`` float tensor of ``[x1, y1, x2, y2]`` (half-open: x2/y2
            are exclusive edges, i.e. ``x + w``).
        scores: ``(N,)`` float tensor.
        iou_thr: Suppress a box if its IoU with an already-kept higher-scoring
            box exceeds this.

    Returns:
        LongTensor of kept indices into ``boxes``, in descending-score order.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = torch.argsort(scores, descending=True)

    # Greedy NMS is inherently sequential with only a few ops per kept box, so we
    # must NOT force a host<->device sync per iteration: `int(order[0])` would
    # block on the GPU every loop (thousands of stalls -> GPU far slower than
    # CPU). Instead we accumulate the kept *index tensors* and stack once at the
    # end — a single sync. (_run_nms additionally runs this on the CPU for CUDA
    # inputs, where this tiny sequential loop is faster still.)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        # Intersection of box i with every remaining box, vectorized.
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-9)
        # Keep only the boxes that do NOT overlap box i too much.
        order = rest[iou <= iou_thr]

    return torch.stack(keep)


def _run_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Dispatch to torchvision NMS if available, else the bundled greedy NMS."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    if _HAVE_TV_NMS:
        # torchvision.ops.nms wants float boxes/scores and returns sorted-desc
        # kept indices — identical contract to our fallback. Fused CUDA kernel,
        # so it runs in-place on the GPU.
        return _tv_nms(boxes.float(), scores.float(), float(iou_thr))
    # Pure-torch fallback. The greedy loop is sequential with tiny per-step work
    # — the worst case for a GPU (per-iteration kernel-launch latency dominates).
    # Running it on the CPU over the (small) candidate set and shipping the kept
    # indices back is dramatically faster for CUDA inputs, and a no-op transfer
    # for CPU inputs (so the CPU path — what the tests pin — is unchanged).
    keep = _greedy_nms(boxes.detach().to("cpu"), scores.detach().to("cpu"), iou_thr)
    return keep.to(boxes.device)


def _iou_against_box(
    boxes: torch.Tensor, ref: tuple[int, int, int, int], device: torch.device
) -> torch.Tensor:
    """IoU of every box in ``boxes`` (``[x1,y1,x2,y2]``) against one ref ``(x,y,w,h)``."""
    rx1, ry1 = float(ref[0]), float(ref[1])
    rx2, ry2 = float(ref[0] + ref[2]), float(ref[1] + ref[3])
    ref_area = max(ref[2], 0) * max(ref[3], 0)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xx1 = x1.clamp(min=rx1)
    yy1 = y1.clamp(min=ry1)
    xx2 = x2.clamp(max=rx2)
    yy2 = y2.clamp(max=ry2)
    inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = areas + ref_area - inter
    return inter / union.clamp(min=1e-9)


def _match_iou(a: Match, b: Match) -> float:
    """IoU of two half-open :class:`Match` boxes (orientation-agnostic).

    Used by the cross-orientation NMS that dedupes the pooled per-orientation
    results: two different orientations that fit the same physical instance have
    overlapping axis-aligned boxes, so this collapses them to the best-scoring one.
    """
    ix0 = max(a.x, b.x)
    iy0 = max(a.y, b.y)
    ix1 = min(a.x + a.w, b.x + b.w)
    iy1 = min(a.y + a.h, b.y + b.h)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


class Matcher:
    """Tiled, multi-scale NCC template matcher (§C.5, §D).

    A single instance is built per loaded image: :meth:`set_image` performs the
    one-time luminance conversion and tiling-plan computation, and every
    subsequent :meth:`match` reuses that precompute. The matcher is device-aware
    (CUDA when usable, CPU otherwise) via :func:`fastmatch.device.resolve_device`.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        compute_dtype: Literal["float32", "float16"] = "float32",
        channel_mode: Literal["luminance", "rgb"] = "luminance",
        conv_backend: Literal["auto", "spatial", "fft"] = "auto",
        vram_fraction: float = 0.6,
        core_tile: int | None = None,
        use_pyramid: bool = True,
    ) -> None:
        """Construct a matcher.

        Args:
            device: ``"auto"`` / ``"cuda"`` / ``"cpu"``; ``None`` means auto.
                Always canary-gated and CPU-fallback-safe via ``resolve_device``.
            compute_dtype: ``"float32"`` (default, recommended) or ``"float16"``
                (fast preview: conv *inputs* may be fp16 but the ``S1``/``S2``
                accumulators and final NCC stay fp32, and ``eps`` is raised).
            channel_mode: ``"luminance"`` (BT.601, default) or ``"rgb"`` (the
                three channel numerators and denominator terms are summed
                *before* the single division).
            conv_backend: ``"auto"`` (spatial for small templates, FFT above
                ``th*tw > 4096``), or forced ``"spatial"`` / ``"fft"``.
            vram_fraction: Fraction of *free* CUDA memory the compute-tile budget
                may consume (ceiling; CUDA only).
            core_tile: **Advanced / testing.** Forces the NCC compute-tile core
                size (square), overriding the VRAM-derived auto size. ``None``
                (default) auto-sizes from VRAM on CUDA / uses a large single tile
                on CPU. T6 tests set this to force tile seams; it does not change
                results, only how the output grid is partitioned. A forced
                ``core_tile`` ALSO disables the coarse-to-fine pyramid so the seam
                test always exercises the full-resolution path (§K.1 safeguard).
            use_pyramid: Enable the coarse-to-fine image-pyramid search for the
                convolution methods (§K.1). ``True`` (default) auto-enables the
                pyramid only when it is beneficial (large enough image *and*
                template); it falls back to the full-resolution
                :meth:`_search_all_scales` otherwise and on any error.
                ``False`` forces the full path — the recall-parity reference.
        """
        self._device = resolve_device(device if device is not None else "auto")
        self._compute_dtype = compute_dtype
        self._channel_mode = channel_mode
        self._conv_backend = conv_backend
        self._vram_fraction = float(vram_fraction)
        self._forced_core_tile = core_tile
        # A forced core tile means a test is exercising the tile-seam partitioning
        # on the full-resolution path; never override that with the pyramid.
        self._use_pyramid = bool(use_pyramid) and core_tile is None

        # fp16 inputs lose precision near the denominator zero-crossing, so the
        # spec raises eps an order of magnitude in that mode.
        self._eps = 1e-3 if compute_dtype == "float16" else 1e-6

        # Per-image precompute, populated by set_image().
        self._lum: torch.Tensor | None = None  # (1,1,H,W) fp32 luminance on device, in [0,1]
        # Per-channel RGB image planes (each (1,1,H,W) UINT8); staged only when
        # the source is 3-channel so channel_mode="rgb" has true per-channel
        # image statistics. Kept uint8 (3 B/px, no fp32 host copy) to honour the
        # gigapixel streaming contract; converted to fp32 [0,1] per tile in
        # _match_tile. None for grayscale sources (rgb mode falls back to
        # luminance there since there is no color information to sum).
        self._rgb: list[torch.Tensor] | None = None
        # Per-channel Y,Cb,Cr image planes (each (1,1,H,W) UINT8, BT.601 full-range),
        # for channel_mode="ycbcr". Derived LAZILY from self._rgb on the first ycbcr
        # query (kept uint8 like _rgb to honour the gigapixel contract) and cached;
        # None until used / for grayscale sources. Invalidated by set_image().
        self._ycbcr: list[torch.Tensor] | None = None
        self._h: int = 0
        self._w: int = 0

        # Compute-device image pyramid for the coarse-to-fine search (§K.1):
        # _pyr_lum[ell] is the staged luminance down-sampled by 2**ell via
        # avg_pool2d(2) (level 0 == self._lum), and _pyr_rgb[ell] the matching
        # 3-plane fp32 RGB stack when rgb is staged. Built lazily on first use
        # by _ensure_pyramid(), bounded by _MAX_PYR_LEVELS, cached and reused
        # across queries, and invalidated by set_image(). On a build OOM it is
        # left empty so match() transparently uses the full path.
        self._pyr_lum: list[torch.Tensor] | None = None
        self._pyr_rgb: list[list[torch.Tensor]] | None = None
        # Matching ycbcr pyramid, built lazily from self._ycbcr the first time the
        # pyramid path runs in ycbcr mode (mirrors _pyr_rgb). None otherwise.
        self._pyr_ycbcr: list[list[torch.Tensor]] | None = None
        # Host-side uint8 luminance for the OpenCV feature path (§J.3); the conv
        # methods never read it. Populated by set_image().
        self._lum_u8: np.ndarray | None = None
        # Host-side uint8 RGB (H,W,3) for the feature path's ``channel_mode="rgb"``
        # appearance verification. Built lazily from the staged per-channel planes
        # on the first rgb feature query and cached; None until then and for
        # grayscale sources (rgb mode falls back to luminance there).
        self._rgb_u8: np.ndarray | None = None

        # Lazily-built OpenCV feature backend (only when method=="features"). It
        # owns a per-(image, detector) cache of downsampled-image keypoints/
        # descriptors so repeated feature queries on the same image are cheap;
        # set_image() invalidates it. Kept as Any-typed to avoid importing the
        # feature module (and thus cv2) unless the feature path is exercised.
        self._feature_matcher = None  # type: ignore[var-annotated]
        # Lazily-built experimental Shazam matcher (only when method=="shazam",
        # which is disabled by default / not in the UI). It reuses the feature
        # backend's dense ORB detection + cache, so it owns no per-image state.
        self._shazam_matcher = None  # type: ignore[var-annotated]

    # -- public properties ---------------------------------------------------

    @property
    def image_size(self) -> tuple[int, int]:
        """Loaded image size as ``(H, W)``. Raises if no image is set."""
        if self._lum is None:
            raise RuntimeError("Matcher.set_image() must be called before image_size")
        return (self._h, self._w)

    @property
    def effective_device(self) -> torch.device:
        """The torch device the engine actually runs on (post canary gating)."""
        return self._device

    # -- image staging -------------------------------------------------------

    def set_image(self, image: np.ndarray) -> None:
        """Stage an image and precompute its luminance + tiling plan.

        Accepts ``(H, W, 3)`` uint8 RGB or ``(H, W)`` uint8 grayscale, and may be
        a ``np.memmap`` (we materialize luminance once into a single device
        tensor here; per-tile slicing happens during :meth:`match`). The result
        is reused across every subsequent :meth:`match` call.

        Args:
            image: The full-resolution image array (read-only; not mutated).
        """
        if image.ndim == 3:
            if image.shape[2] != 3:
                raise ValueError(f"3-D image must be (H,W,3), got {image.shape}")
            lum = self._to_luminance(image)
            # Stage the per-channel RGB planes too so channel_mode="rgb" can run
            # against true per-channel image statistics (the spec sums three
            # numerators/denominators before dividing — that needs S1/S2/cross
            # per channel, not just luminance). Staged as UINT8 (3 B/px, no fp32
            # host copy) to honour the gigapixel/memmap streaming contract — a
            # full fp32 plane set would be 12 B/px and defeat it. The per-tile
            # slice is converted to fp32 [0,1] inside _match_tile, mirroring how
            # the uint8-derived luminance is sliced per tile.
            self._rgb = [
                torch.from_numpy(np.ascontiguousarray(image[:, :, c]))
                .to(self._device)
                .unsqueeze(0)
                .unsqueeze(0)
                for c in range(3)
            ]
        elif image.ndim == 2:
            # Grayscale: luminance is just the (normalized) intensity itself, and
            # there is no color information for rgb mode to use.
            lum = np.asarray(image, dtype=np.float32) / 255.0
            self._rgb = None
        else:
            raise ValueError(f"image must be (H,W,3) or (H,W), got ndim={image.ndim}")

        self._h, self._w = int(lum.shape[0]), int(lum.shape[1])
        # Stage as (1,1,H,W) fp32 on the compute device once. np.ascontiguousarray
        # forces a real read of any memmap into RAM before the host->device copy
        # (a memmap row may be non-contiguous after the luminance combine).
        t = torch.from_numpy(np.ascontiguousarray(lum)).to(self._device)
        self._lum = t.unsqueeze(0).unsqueeze(0)

        # Stage a host-side uint8 luminance for the OpenCV feature path (§J.3),
        # which runs on CPU regardless of the conv device. Reconstructing it from
        # the fp32 [0,1] luminance here keeps a single canonical grayscale source.
        self._lum_u8 = np.ascontiguousarray(
            np.clip(np.rint(np.asarray(lum) * 255.0), 0, 255).astype(np.uint8)
        )

        # A new image invalidates the cached coarse-to-fine pyramid (§K.1); it is
        # rebuilt lazily on the next match() against the new staged luminance/rgb.
        self._pyr_lum = None
        self._pyr_rgb = None
        # ...the lazily-derived ycbcr planes + their pyramid...
        self._ycbcr = None
        self._pyr_ycbcr = None
        # ...and the host RGB the feature path's rgb mode reconstructs lazily.
        self._rgb_u8 = None

        # A new image invalidates any cached image features (§J.3/§J.4). The
        # matcher object itself is reused so its detector cache key still applies.
        if self._feature_matcher is not None:
            self._feature_matcher.invalidate_image()

    def _to_luminance(self, image: np.ndarray) -> np.ndarray:
        """RGB uint8 ``(H,W,3)`` -> fp32 luminance ``(H,W)`` in ``[0,1]`` (BT.601)."""
        # Compute in float32 directly so the /255 and the weighted sum never
        # overflow uint8 and stay bounded in [0,1].
        rgb = np.asarray(image, dtype=np.float32)
        wr, wg, wb = _BT601
        lum = rgb[:, :, 0] * wr + rgb[:, :, 1] * wg + rgb[:, :, 2] * wb
        return lum / 255.0

    def _ensure_ycbcr(self) -> bool:
        """Lazily derive the staged Y,Cb,Cr planes from the RGB planes.

        Returns ``True`` when ycbcr planes are available (3-channel source). The
        planes are computed once from :attr:`_rgb` (BT.601 full-range), stored as
        ``uint8`` ``(1,1,H,W)`` like :attr:`_rgb` (gigapixel-friendly), and cached
        until :meth:`set_image` invalidates them. ``False`` for grayscale sources,
        so ycbcr mode transparently falls back to luminance.
        """
        if self._ycbcr is not None:
            return True
        if self._rgb is None:
            return False
        r, g, b = (p.to(torch.float32) for p in self._rgb)
        planes: list[torch.Tensor] = []
        for wr, wg, wb, bias in _YCBCR_FROM_RGB:
            c = wr * r + wg * g + wb * b + bias
            planes.append(c.clamp(0.0, 255.0).round().to(torch.uint8))
        self._ycbcr = planes
        return True

    def _ensure_ycbcr_pyramid(self) -> list[list[torch.Tensor]] | None:
        """Build (lazily, once) the ycbcr image pyramid (mirrors :attr:`_pyr_rgb`).

        Pools each Y/Cb/Cr plane in fp32 ``[0,1]`` to the same level count as the
        luminance pyramid, so the level-``ell`` ROI conv runs against the same
        normalized signal as the template channels. Returns ``None`` (caller uses
        the full path) if ycbcr planes / the lum pyramid are absent, or on OOM.
        """
        if self._pyr_ycbcr is not None:
            return self._pyr_ycbcr
        if self._ycbcr is None or self._pyr_lum is None:
            return None
        try:
            base = [p.to(torch.float32) / 255.0 for p in self._ycbcr]
            levels: list[list[torch.Tensor]] = [base]
            for _ in range(len(self._pyr_lum) - 1):
                prev = levels[-1]
                levels.append([F.avg_pool2d(p, kernel_size=2) for p in prev])
            self._pyr_ycbcr = levels
            return levels
        except torch.cuda.OutOfMemoryError:
            self._pyr_ycbcr = None
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            return None

    # -- matching ------------------------------------------------------------

    def match(
        self,
        template: np.ndarray,
        params: MatchParams,
        *,
        exclude_box: tuple[int, int, int, int] | None = None,
        cancel: Callable[[], bool] | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> list[Match]:
        """Find every instance of ``template`` in the staged image.

        Args:
            template: ``(th, tw, 3)`` uint8 RGB or ``(th, tw)`` uint8 crop.
            params: Search parameters (thresholds, scales, NMS, channel mode...).
            exclude_box: ``(x, y, w, h)`` source region to exclude (its own
                detection plus anything overlapping it past ``exclude_iou``).
            cancel: Polled at every tile boundary; return True to abort early
                (a partial/empty result is returned).
            progress: Called with an int ``0..100`` after each finished tile.

        Returns:
            ``list[Match]`` sorted by score descending, capped at
            ``max_results``, in full-resolution image px (origin top-left).
        """
        if self._lum is None:
            raise RuntimeError("Matcher.set_image() must be called before match()")

        # Method dispatch (§J.4). The convolution methods (ncc/ssd/ccorr) share
        # ALL the tiled machinery below and differ only in the per-window score
        # formula (§J.2). "features" delegates to a wholly separate CPU/OpenCV
        # backend that tolerates rotation/scale/warp the conv methods cannot.
        method = params.method or "ncc"
        if method == "features":
            return self._match_features(
                template, params, exclude_box=exclude_box, cancel=cancel, progress=progress
            )
        if method == SHAZAM_METHOD:
            # Experimental, DISABLED-by-default landmark-pair-hash matcher: it is
            # not in METHODS (never offered in the UI), but reachable when set
            # explicitly so the approach stays kept + tested (see shazam_matcher.py).
            return self._match_shazam(
                template, params, exclude_box=exclude_box, cancel=cancel, progress=progress
            )
        if method not in CONV_METHODS:
            raise ValueError(
                f"unknown matching method {method!r}; expected one of {sorted(METHODS)}"
            )

        # Channel mode can be overridden per-call by params (the controller maps
        # the UI choice onto MatchParams); fall back to the constructor default.
        channel_mode = params.channel_mode or self._channel_mode
        # rgb/ycbcr modes are only meaningful when both the image and template carry
        # color; otherwise transparently fall back to luminance so the per-channel
        # pairing in _score_map always lines up. ycbcr additionally needs its
        # (lazily-derived) Y,Cb,Cr planes — if they cannot be built, fall back too.
        if channel_mode in ("rgb", "ycbcr") and (self._rgb is None or template.ndim != 3):
            channel_mode = "luminance"
        if channel_mode == "ycbcr" and not self._ensure_ycbcr():
            channel_mode = "luminance"

        # Per-channel weights for the active multi-channel mode (normalized to sum
        # to 1 so equal weights == the unweighted multi-channel behaviour). The
        # weights ride on the prepared template's per-channel dicts so the tiled /
        # pyramid machinery needs no new plumbing (see _prepare_template).
        if channel_mode == "rgb":
            weights = normalize_weights(params.rgb_weights, 3)
        elif channel_mode == "ycbcr":
            weights = normalize_weights(params.ycbcr_weights, 3)
        else:
            weights = (1.0,)

        # Orientation search (dihedral D4, §J / active_orientations). With both
        # checkboxes off this is exactly ("R0",) and the code below takes the
        # single-template path UNCHANGED — byte-for-byte the legacy behaviour the
        # existing tests pin. Otherwise the boxed template is transformed by each
        # active orientation, searched independently (reusing all the tiled multi-
        # scale machinery), the per-orientation hits are tagged with their
        # orientation, pooled, and deduped by ONE cross-orientation global NMS.
        orients = active_orientations(params.enable_rotation, params.enable_flipping)

        if orients == ("R0",):
            # --- legacy single-orientation path (no behaviour change) ----------
            # Prepare the device-resident template channels once (validated here).
            # The variance-floor rejection is gated on method (NCC only) inside.
            tmpl = self._prepare_template(template, channel_mode, method, weights)
            th0, tw0 = tmpl["th"], tmpl["tw"]
            cands = self._search_template(tmpl, params, channel_mode, method, cancel, progress)
            if cancel is not None and cancel():
                return []
            return self._finalize(cands, params, exclude_box, th0, tw0)

        # --- multi-orientation path ------------------------------------------
        # Per orientation: transform the boxed template, run the SAME per-template
        # multi-scale (coarse-to-fine) search, finalize it (source-exclusion +
        # per-orientation IoU-NMS + top-K), and tag every kept Match with the
        # orientation it was found under. The transformed template's box size is
        # used as-is, so R90/R270/diagonal boxes are correctly H/W-swapped.
        pooled: list[Match] = []
        n_orients = len(orients)
        for oi, orient in enumerate(orients):
            if cancel is not None and cancel():
                return []
            t_oriented = apply_orientation(template, orient)
            tmpl = self._prepare_template(t_oriented, channel_mode, method, weights)
            th0, tw0 = tmpl["th"], tmpl["tw"]

            # Scope the per-orientation progress into this orientation's slice of
            # the [0,100] bar so the overall sweep stays monotonic and determinate.
            def _scoped(pct: int, _oi: int = oi, _n: int = n_orients) -> None:
                if progress is not None:
                    progress(min(100, int(round((100.0 * _oi + pct) / _n))))

            sub_progress = _scoped if progress is not None else None
            cands = self._search_template(
                tmpl, params, channel_mode, method, cancel, sub_progress
            )
            if cancel is not None and cancel():
                return []
            for m in self._finalize(cands, params, exclude_box, th0, tw0):
                pooled.append(replace(m, orientation=orient))

        if progress is not None:
            progress(100)

        # One cross-orientation greedy IoU-NMS over the combined list keeps the
        # highest-scoring candidate per location (carrying its orientation tag),
        # then top-K — the global dedupe across the per-orientation results.
        return self._finalize_orientations(pooled, params)

    def _search_template(
        self,
        tmpl: dict,
        params: MatchParams,
        channel_mode: str,
        method: str,
        cancel: Callable[[], bool] | None,
        progress: Callable[[int], None] | None,
    ) -> dict:
        """Run the tiled multi-scale search for ONE (already-prepared) template.

        Picks the coarse-to-fine pyramid path when beneficial (§K.1) else the
        full-resolution search, wrapped in the CUDA OOM degradation ladder, and
        returns the stacked raw candidate dict (no exclusion/NMS) for
        :meth:`_finalize`. Factored out of :meth:`match` so the orientation search
        can drive it once per active orientation; the legacy single-orientation
        call site is byte-for-byte identical.
        """
        scales = tuple(params.scales) if params.scales else (1.0,)
        device = self._device

        # Choose the coarse-to-fine pyramid path when it is beneficial (§K.1),
        # else the full-resolution search. The pyramid path internally falls back
        # to the full path per (template, scale) when the level count K==0, and on
        # any error/OOM the whole job degrades to the full path below — so recall
        # parity holds and existing tests (which hit the not-beneficial branch or
        # a forced core_tile) exercise the byte-identical full path.
        use_pyramid = self._pyramid_beneficial(tmpl, scales)

        # The whole search is wrapped in the CUDA OOM degradation ladder. On CPU
        # the ladder collapses to a single straight run (no OOM expected).
        try:
            if use_pyramid:
                cands = self._search_all_scales_pyramid(
                    tmpl, scales, params, channel_mode, method, cancel, progress
                )
            else:
                cands = self._search_all_scales(
                    tmpl, scales, params, channel_mode, method, cancel, progress
                )
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise  # CPU OOM is a real failure, not something we degrade.
            # The pyramid participates in the OOM ladder (§K.1): drop it and let
            # the degradation ladder run the plain full-resolution search.
            self._pyr_lum = None
            self._pyr_rgb = None
            self._pyr_ycbcr = None
            cands = self._degrade_after_oom(
                tmpl, scales, params, channel_mode, method, cancel, progress
            )

        # empty_cache between *jobs* (not between tiles) — frees the tile
        # scratch this match() allocated before the next match() runs.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return cands

    @staticmethod
    def _finalize_orientations(pooled: list[Match], params: MatchParams) -> list[Match]:
        """Cross-orientation greedy IoU-NMS + top-K over pooled tagged Matches (§J).

        Each input is an already-finalized (per-orientation) :class:`Match` tagged
        with its orientation. This is the ONE global dedupe across orientations:
        sort by score desc, greedily keep a box unless its IoU with an already-kept
        higher-scoring box exceeds ``nms_iou`` (so the best-scoring orientation wins
        per location, carrying its tag), then cap at ``max_results``. Box IoU is
        orientation-agnostic, so an R0 and an R180 fit of the same instance overlap
        and collapse to one. Pure python (runs on the small, post-finalize list).
        """
        if not pooled:
            return []
        ordered = sorted(pooled, key=lambda m: m.score, reverse=True)
        kept: list[Match] = []
        nms_iou = float(params.nms_iou)
        for m in ordered:
            if all(_match_iou(m, k) <= nms_iou for k in kept):
                kept.append(m)
            if len(kept) >= params.max_results:
                break
        return kept

    # -- template preparation / validation -----------------------------------

    def _prepare_template(
        self,
        template: np.ndarray,
        channel_mode: str,
        method: str = "ncc",
        weights: tuple[float, ...] = (1.0,),
    ) -> dict:
        """Validate and stage the template channels on the compute device.

        Returns a dict with the per-channel ``[0,1]`` template tensors (one for
        luminance mode, three for rgb/ycbcr mode), the per-channel ``weights``
        (aligned with ``channels``, already normalized to sum to 1), and
        ``th``/``tw``. Per-scale zero-mean / norm are derived later in
        :meth:`_scaled_template`; the per-channel weight rides on each scaled
        channel dict so the score machinery combines the channels weighted.

        Args:
            template: ``(th, tw, 3)`` uint8 RGB or ``(th, tw)`` uint8 crop.
            channel_mode: ``"luminance"`` / ``"rgb"`` / ``"ycbcr"`` (controls the
                channels staged: one luma plane, R/G/B, or Y/Cb/Cr).
            method: The matching method (§J.2). The variance-floor rejection only
                applies to ``"ncc"`` — SSD/CCORR are *defined* to handle flat /
                low-variance templates (that is their whole use case, §J.1/§J.2),
                so they must not raise here.
            weights: Per-channel weights for the multi-channel modes (luminance
                ignores them — a single weight-1 channel).

        Raises:
            ValueError: side ``< 8`` px, or (``method=="ncc"`` only, per H1/§H)
                template std below the variance floor — a featureless crop can
                never produce a stable NCC peak and would otherwise flood the
                result with noise. SSD/CCORR skip this check.
        """
        if template.ndim == 3:
            if template.shape[2] != 3:
                raise ValueError(f"3-D template must be (th,tw,3), got {template.shape}")
            th, tw = int(template.shape[0]), int(template.shape[1])
        elif template.ndim == 2:
            th, tw = int(template.shape[0]), int(template.shape[1])
        else:
            raise ValueError(f"template must be (th,tw,3) or (th,tw), got ndim={template.ndim}")

        if th < _MIN_SIDE or tw < _MIN_SIDE:
            raise ValueError(
                f"template side too small: {tw}x{th} px (minimum {_MIN_SIDE}px per side)"
            )

        channels: list[torch.Tensor] = []
        if channel_mode in ("rgb", "ycbcr") and template.ndim == 3:
            if channel_mode == "ycbcr":
                # BT.601 full-range Y,Cb,Cr in [0,1] (matches the staged image
                # planes; the /255 keeps Cb/Cr centred near 0.5 — NCC subtracts the
                # mean, so the offset cancels).
                rgbf = np.asarray(template, dtype=np.float32)
                r, g, b = rgbf[:, :, 0], rgbf[:, :, 1], rgbf[:, :, 2]
                planes = [
                    (wr * r + wg * g + wb * b + bias) / 255.0
                    for wr, wg, wb, bias in _YCBCR_FROM_RGB
                ]
            else:
                rgbf = np.asarray(template, dtype=np.float32) / 255.0
                planes = [rgbf[:, :, 0], rgbf[:, :, 1], rgbf[:, :, 2]]
            for plane in planes:
                channels.append(
                    torch.from_numpy(np.ascontiguousarray(plane.astype(np.float32))).to(self._device)
                )
            chan_weights = list(normalize_weights(weights, 3))
        else:
            # Luminance (or a genuinely grayscale template) — single weight-1 plane.
            if template.ndim == 3:
                lum = self._to_luminance(template)
            else:
                lum = np.asarray(template, dtype=np.float32) / 255.0
            channels.append(torch.from_numpy(np.ascontiguousarray(lum)).to(self._device))
            chan_weights = [1.0]

        # Variance floor: reject a near-flat template — but ONLY for NCC (§J.2).
        # NCC normalizes by the patch variance, so a featureless crop has an
        # unstable (near-0/0) denominator and would flood results with noise.
        # SSD/CCORR have no variance normalization and are specifically the
        # right tools for flat / low-texture / exact-appearance regions (§J.1),
        # so they must accept such templates. Checked on the structural
        # (luminance / summed) signal so an all-one-color crop is rejected even
        # in rgb mode where each channel is individually flat.
        if method == "ncc":
            struct = torch.stack(channels, dim=0).mean(dim=0) if len(channels) > 1 else channels[0]
            if float(struct.std().item()) < _VAR_FLOOR:
                raise ValueError(
                    "template has near-zero variance (featureless region); "
                    "select a more distinctive area, or use the SSD method"
                )

        return {"channels": channels, "th": th, "tw": tw, "weights": chan_weights}

    def _scaled_template(
        self, tmpl: dict, scale: float
    ) -> tuple[list[dict], int, int]:
        """Resample the template to ``scale`` and precompute per-channel score terms.

        Returns ``(per_channel, sth, stw)`` where each per-channel dict holds the
        zero-mean weight ``Tz`` shaped ``(1,1,sth,stw)``, its L2 norm ``normTz``,
        the template mean ``meanT`` and sum-of-squares ``sumT2``, and the pixel
        count ``n``. The per-window image statistics (``S1``, ``S2``, cross term)
        are computed against these during tiling, and the dispatched score helper
        (§J.2) reuses ``meanT``/``sumT2`` to derive the raw cross term and SSD/
        CCORR maps without a second convolution.
        """
        th, tw = tmpl["th"], tmpl["tw"]
        if abs(scale - 1.0) < 1e-9:
            sth, stw = th, tw
            chans = tmpl["channels"]
        else:
            sth = max(_MIN_SIDE, int(round(th * scale)))
            stw = max(_MIN_SIDE, int(round(tw * scale)))
            chans = []
            for c in tmpl["channels"]:
                # Bilinear resample each channel template to the scaled size.
                r = F.interpolate(
                    c.unsqueeze(0).unsqueeze(0),
                    size=(sth, stw),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                chans.append(r)

        n = float(sth * stw)
        weights = tmpl.get("weights") or [1.0] * len(chans)
        per_channel: list[dict] = []
        for c, w in zip(chans, weights):
            mean = c.mean()
            tz = (c - mean)
            norm_tz = torch.sqrt((tz * tz).sum()).clamp(min=self._eps)
            # sum_w(T^2) per channel: needed by both CCORR (denominator) and SSD
            # (the +sumT2 term). Kept as a scalar tensor on-device.
            sum_t2 = (c * c).sum()
            per_channel.append(
                {
                    "tz": tz.unsqueeze(0).unsqueeze(0).contiguous(),  # (1,1,sth,stw)
                    "norm_tz": norm_tz,
                    "mean_t": mean,
                    "sum_t2": sum_t2,
                    "n": n,
                    "weight": float(w),
                }
            )
        return per_channel, sth, stw

    def _scaled_template_at_level(
        self, tmpl: dict, scale: float, level: int
    ) -> tuple[list[dict], int, int]:
        """Template per-channel terms at physical ``scale`` AND pyramid ``level``.

        The coarse/refine template MUST be down-sampled the SAME way the image
        pyramid is (``F.avg_pool2d(2)`` per level) — not by bilinear interpolation.
        Otherwise, for high-frequency content, an exact copy's coarse NCC is well
        below 1.0 (bilinear vs. average disagree pixel-for-pixel), the relaxed
        coarse threshold prunes it, and the pyramid misses real matches that the
        full search finds. Here we first resample to the requested physical scale
        (bilinear, matching the full path at ``level == 0``) and then ``avg_pool2d``
        ``level`` times, mirroring :meth:`_ensure_pyramid`. ``level == 0`` is
        identical to :meth:`_scaled_template`.
        """
        if level <= 0:
            return self._scaled_template(tmpl, scale)
        # Level-0 channels at the requested physical scale (bilinear, as the full path).
        if abs(scale - 1.0) < 1e-9:
            chans = list(tmpl["channels"])
        else:
            th, tw = tmpl["th"], tmpl["tw"]
            sth0 = max(_MIN_SIDE, int(round(th * scale)))
            stw0 = max(_MIN_SIDE, int(round(tw * scale)))
            chans = [
                F.interpolate(
                    c.unsqueeze(0).unsqueeze(0),
                    size=(sth0, stw0),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                for c in tmpl["channels"]
            ]
        # Average-pool `level` times to match the image pyramid construction.
        pooled: list[torch.Tensor] = []
        for c in chans:
            x = c.unsqueeze(0).unsqueeze(0)
            for _ in range(level):
                if min(x.shape[-2], x.shape[-1]) < 2:
                    break
                x = F.avg_pool2d(x, kernel_size=2)
            pooled.append(x[0, 0])

        sth, stw = pooled[0].shape[0], pooled[0].shape[1]
        n = float(sth * stw)
        weights = tmpl.get("weights") or [1.0] * len(pooled)
        per_channel: list[dict] = []
        for c, w in zip(pooled, weights):
            mean = c.mean()
            tz = c - mean
            norm_tz = torch.sqrt((tz * tz).sum()).clamp(min=self._eps)
            sum_t2 = (c * c).sum()
            per_channel.append(
                {
                    "tz": tz.unsqueeze(0).unsqueeze(0).contiguous(),
                    "norm_tz": norm_tz,
                    "mean_t": mean,
                    "sum_t2": sum_t2,
                    "n": n,
                    "weight": float(w),
                }
            )
        return per_channel, sth, stw

    # -- multi-scale orchestration -------------------------------------------

    def _search_all_scales(
        self,
        tmpl: dict,
        scales: tuple[float, ...],
        params: MatchParams,
        channel_mode: str,
        method: str,
        cancel: Callable[[], bool] | None,
        progress: Callable[[int], None] | None,
    ) -> dict:
        """Run the tiled score map at every scale, pooling raw candidates (no NMS).

        Shared by all convolution methods (``method`` selects the per-window
        score formula in :meth:`_score_map`). Returns a dict of stacked candidate
        tensors (``boxes``, ``scores``, ``scales``) on the compute device, to be
        deduped by a single global NMS in :meth:`_finalize`.
        """
        all_x: list[torch.Tensor] = []
        all_y: list[torch.Tensor] = []
        all_w: list[torch.Tensor] = []
        all_h: list[torch.Tensor] = []
        all_score: list[torch.Tensor] = []
        all_scale: list[torch.Tensor] = []

        # Build the per-scale tile plans up front so progress is a single
        # determinate sweep across all (scale, tile) pairs. Total tile count
        # drives progress; guard against an all-too-large-scale plan producing
        # zero tiles (template bigger than the image at that scale).
        scale_tiles: list[tuple[float, list, int, int, list]] = []
        total_tiles = 0
        for s in scales:
            per_channel, sth, stw = self._scaled_template(tmpl, s)
            tiles = self._plan_tiles(sth, stw)
            scale_tiles.append((s, per_channel, sth, stw, tiles))
            total_tiles += len(tiles)
        total_tiles = max(total_tiles, 1)

        done = 0
        for s, per_channel, sth, stw, tiles in scale_tiles:
            for tile in tiles:
                if cancel is not None and cancel():
                    # Honor cancellation at every tile boundary (§D.8).
                    return self._stack_candidates(
                        all_x, all_y, all_w, all_h, all_score, all_scale
                    )
                xs, ys, ws, hs, scs = self._match_tile(
                    per_channel, sth, stw, s, tile, params, channel_mode, method
                )
                if xs.numel() > 0:
                    all_x.append(xs)
                    all_y.append(ys)
                    all_w.append(ws)
                    all_h.append(hs)
                    all_score.append(scs)
                    all_scale.append(torch.full_like(scs, float(s)))
                done += 1
                if progress is not None:
                    progress(min(100, int(round(100.0 * done / total_tiles))))

        return self._stack_candidates(all_x, all_y, all_w, all_h, all_score, all_scale)

    @staticmethod
    def _stack_candidates(
        all_x: list[torch.Tensor],
        all_y: list[torch.Tensor],
        all_w: list[torch.Tensor],
        all_h: list[torch.Tensor],
        all_score: list[torch.Tensor],
        all_scale: list[torch.Tensor],
    ) -> dict:
        """Concatenate per-tile candidate lists into flat tensors (or empties)."""
        if not all_x:
            empty = torch.empty(0)
            return {
                "x": empty,
                "y": empty,
                "w": empty,
                "h": empty,
                "score": empty,
                "scale": empty,
            }
        return {
            "x": torch.cat(all_x),
            "y": torch.cat(all_y),
            "w": torch.cat(all_w),
            "h": torch.cat(all_h),
            "score": torch.cat(all_score),
            "scale": torch.cat(all_scale),
        }

    # -- coarse-to-fine image-pyramid search (§K.1) --------------------------

    def _pyramid_beneficial(self, tmpl: dict, scales: tuple[float, ...]) -> bool:
        """Decide whether coarse-to-fine is worth it for this query (§K.1).

        Auto-enable only when the pyramid actually pays: the full-resolution
        search must be expensive enough (large image AND a large-enough template
        so the coarse level is meaningfully smaller) and at least one requested
        scale must yield ``K >= 1``. Otherwise return ``False`` so the unchanged
        full path runs — this keeps every small-image/seam test on the
        byte-identical full path and avoids paying the pyramid's fixed overhead
        where it would lose.
        """
        if not self._use_pyramid:
            return False
        # Below these sizes the full-res search is already cheap; the pyramid's
        # template-pooling + ROI gather overhead would not be recovered.
        if (self._h * self._w) < _PYRAMID_MIN_IMAGE_PX:
            return False
        th, tw = tmpl["th"], tmpl["tw"]
        # At least one scale must produce a real (>=1) level count AND keep the
        # scaled template above the px floor where coarse localization is stable.
        for s in scales:
            sth = max(_MIN_SIDE, int(round(th * s)))
            stw = max(_MIN_SIDE, int(round(tw * s)))
            if (sth * stw) < _PYRAMID_MIN_TEMPLATE_PX:
                continue
            if self._level_count(sth, stw) >= 1:
                return True
        return False

    def _level_count(self, sth: int, stw: int) -> int:
        """Pyramid level count ``K`` for a template of size ``sth x stw`` (§K.1).

        ``K = clamp(floor(log2(min(sth,stw) / MIN_COARSE_SIDE)), 0, MAX_PYR_LEVELS)``
        — the coarsest level whose down-sampled template still keeps ``>=
        MIN_COARSE_SIDE`` px on its shorter side, capped at ``MAX_PYR_LEVELS``.
        """
        m = min(sth, stw)
        if m < _MIN_COARSE_SIDE:
            return 0
        k = int(np.floor(np.log2(m / _MIN_COARSE_SIDE)))
        return int(max(0, min(_MAX_PYR_LEVELS, k)))

    def _ncc_safe_level(self, tmpl: dict, scale: float, k: int, method: str) -> int:
        """Largest level ``<= k`` whose down-sampled NCC template keeps variance.

        For NCC, the coarse template must retain structure (variance on the
        ``[0,1]`` template ``>= _VAR_FLOOR``); otherwise its blurred-flat NCC peak
        is zeroed by the flat-region guard and the coarse pass misses the match.
        Returns the deepest level where the template at physical scale
        ``scale / 2**level`` still clears the floor (``0`` falls back to the full
        path). Non-NCC methods are unaffected and ``k`` is returned unchanged.
        """
        if method != "ncc" or k <= 0:
            return k
        lvl = k
        while lvl > 0:
            per_channel, sth_l, stw_l = self._scaled_template_at_level(tmpl, scale, lvl)
            n = max(1, sth_l * stw_l)
            # Variance on [0,1] = sum(Tz^2)/n = norm_tz^2 / n; keep the deepest
            # level where any channel still has structure (permissive: rgb needs
            # only one informative channel).
            vmax = max(float((c["norm_tz"] ** 2).item()) / n for c in per_channel)
            if vmax >= _VAR_FLOOR:
                break
            lvl -= 1
        return lvl

    def _ensure_pyramid(self) -> bool:
        """Build (lazily, once) the compute-device image pyramid (§K.1).

        Levels ``0..MAX_PYR_LEVELS``: level 0 is the staged luminance (and rgb
        planes when staged), level ``ell`` is the previous level down-sampled by
        ``F.avg_pool2d(2)`` (memory overhead about 1/3 of the base). Cached on the
        matcher and reused across queries. On a CUDA OOM during the build the
        partial pyramid is discarded and ``False`` is returned so the caller uses
        the full path (the pyramid participates in the OOM ladder, §K.1).
        """
        if self._pyr_lum is not None:
            return True
        if self._lum is None:
            return False
        try:
            lum_levels: list[torch.Tensor] = [self._lum]
            for _ in range(_MAX_PYR_LEVELS):
                prev = lum_levels[-1]
                if min(prev.shape[-2], prev.shape[-1]) < 2:
                    break  # cannot halve further
                lum_levels.append(F.avg_pool2d(prev, kernel_size=2))

            rgb_levels: list[list[torch.Tensor]] | None = None
            if self._rgb is not None:
                # Pool each RGB plane in fp32 [0,1] so the level-ell ROI conv runs
                # against the same normalized signal as the template channels
                # (matching how _match_tile converts the uint8 plane per tile).
                base = [p.to(torch.float32) / 255.0 for p in self._rgb]
                rgb_levels = [base]
                for _ in range(len(lum_levels) - 1):
                    prev_planes = rgb_levels[-1]
                    rgb_levels.append(
                        [F.avg_pool2d(p, kernel_size=2) for p in prev_planes]
                    )
            self._pyr_lum = lum_levels
            self._pyr_rgb = rgb_levels
            return True
        except torch.cuda.OutOfMemoryError:
            self._pyr_lum = None
            self._pyr_rgb = None
            self._pyr_ycbcr = None
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            return False

    def _level_planes(
        self, level: int, channel_mode: str, channels: int
    ) -> list[torch.Tensor]:
        """Image plane(s) at pyramid ``level`` matching the template channels.

        Returns fp32 ``(1,1,Hl,Wl)`` slabs — three RGB or Y,Cb,Cr planes when the
        matching multi-channel pyramid is available, else a single luminance plane
        — paired positionally with ``per_channel`` exactly as :meth:`_match_tile`.
        The ycbcr pyramid is built lazily here on first use (mirrors _pyr_rgb).
        """
        if channel_mode == "rgb" and self._pyr_rgb is not None and channels == 3:
            return list(self._pyr_rgb[level])
        if channel_mode == "ycbcr" and channels == 3:
            pyr = self._ensure_ycbcr_pyramid()
            if pyr is not None:
                return list(pyr[level])
        assert self._pyr_lum is not None
        return [self._pyr_lum[level]]

    def _search_all_scales_pyramid(
        self,
        tmpl: dict,
        scales: tuple[float, ...],
        params: MatchParams,
        channel_mode: str,
        method: str,
        cancel: Callable[[], bool] | None,
        progress: Callable[[int], None] | None,
    ) -> dict:
        """Coarse-to-fine multi-scale search, pooling raw candidates (§K.1).

        Per scale: pick ``K``; ``K == 0`` falls back to the full-resolution
        per-scale search (unchanged behaviour); otherwise localize candidates at
        level ``K`` and refine them down to level 0 with batched ROI convs. All
        scales' level-0 candidates are pooled and handed to the single global NMS
        in :meth:`_finalize`, exactly like the full path. Falls back to the full
        path entirely if the pyramid cannot be built.
        """
        if not self._ensure_pyramid():
            return self._search_all_scales(
                tmpl, scales, params, channel_mode, method, cancel, progress
            )

        all_x: list[torch.Tensor] = []
        all_y: list[torch.Tensor] = []
        all_w: list[torch.Tensor] = []
        all_h: list[torch.Tensor] = []
        all_score: list[torch.Tensor] = []
        all_scale: list[torch.Tensor] = []

        n_scales = max(len(scales), 1)
        for si, s in enumerate(scales):
            if cancel is not None and cancel():
                break
            sth = max(_MIN_SIDE, int(round(tmpl["th"] * s)))
            stw = max(_MIN_SIDE, int(round(tmpl["tw"] * s)))
            assert self._pyr_lum is not None
            # K cannot exceed the number of pyramid levels actually built.
            k = min(self._level_count(sth, stw), len(self._pyr_lum) - 1)
            # NCC needs the down-sampled template to keep its variance. A high-
            # frequency / low-structure template averaged down to a deep coarse
            # level can become near-flat; the flat-region guard then zeros its NCC
            # peak, so the coarse pass finds ZERO candidates and the pyramid misses
            # real matches that the full search finds (SSD/CCORR don't subtract the
            # mean, so they are unaffected). Reduce K to the deepest level where the
            # coarse template still clears the variance floor; K==0 -> full path.
            k = self._ncc_safe_level(tmpl, s, k, method)

            if k <= 0:
                # Small template at this scale: the unchanged full-res per-scale
                # path (keeps recall parity for the easy cases automatically).
                per_channel, sth0, stw0 = self._scaled_template(tmpl, s)
                xs, ys, ws, hs, scs = self._full_scale_candidates(
                    per_channel, sth0, stw0, s, params, channel_mode, method, cancel
                )
            else:
                xs, ys, ws, hs, scs = self._coarse_to_fine_scale(
                    tmpl, s, k, params, channel_mode, method, cancel
                )

            if xs.numel() > 0:
                all_x.append(xs)
                all_y.append(ys)
                all_w.append(ws)
                all_h.append(hs)
                all_score.append(scs)
                all_scale.append(torch.full_like(scs, float(s)))

            if progress is not None:
                progress(min(100, int(round(100.0 * (si + 1) / n_scales))))

        return self._stack_candidates(all_x, all_y, all_w, all_h, all_score, all_scale)

    def _full_scale_candidates(
        self,
        per_channel: list[dict],
        sth: int,
        stw: int,
        scale: float,
        params: MatchParams,
        channel_mode: str,
        method: str,
        cancel: Callable[[], bool] | None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the unchanged tiled full-res search for ONE scale (K==0 branch).

        Reuses :meth:`_plan_tiles` / :meth:`_match_tile` so this scale is computed
        byte-identically to the full path — the recall reference for the easy
        (small-template) cases the pyramid skips.
        """
        tiles = self._plan_tiles(sth, stw)
        xs_l: list[torch.Tensor] = []
        ys_l: list[torch.Tensor] = []
        ws_l: list[torch.Tensor] = []
        hs_l: list[torch.Tensor] = []
        sc_l: list[torch.Tensor] = []
        for tile in tiles:
            if cancel is not None and cancel():
                break
            gx, gy, ws, hs, scs = self._match_tile(
                per_channel, sth, stw, scale, tile, params, channel_mode, method
            )
            if gx.numel() > 0:
                xs_l.append(gx)
                ys_l.append(gy)
                ws_l.append(ws)
                hs_l.append(hs)
                sc_l.append(scs)
        if not xs_l:
            empty = torch.empty(0, device=self._device)
            return empty, empty, empty, empty, empty
        return (
            torch.cat(xs_l),
            torch.cat(ys_l),
            torch.cat(ws_l),
            torch.cat(hs_l),
            torch.cat(sc_l),
        )

    def _coarse_to_fine_scale(
        self,
        tmpl: dict,
        scale: float,
        k: int,
        params: MatchParams,
        channel_mode: str,
        method: str,
        cancel: Callable[[], bool] | None,
    ) -> tuple[torch.Tensor, ...]:
        """Coarse-to-fine localization for ONE scale (§K.1 steps 2-4).

        Returns flat ``(x, y, w, h, score)`` level-0 candidate tensors (full-res
        top-left px + the scaled box size + the exact level-0 score) ready for the
        shared :meth:`_finalize`. No Python per-candidate loop touches the GPU:
        the refinement is one batched conv over all ROIs per level.
        """
        device = self._device
        floor = float(params.threshold_floor)
        thr = float(params.threshold)
        # Relaxed coarse threshold so a true peak that is weaker once blurred is
        # not pruned before it can be refined (§K.1 step 2).
        coarse_thr = max(0.0, min(thr, floor) - _COARSE_MARGIN)
        empty = torch.empty(0, device=device)

        # --- step 2: coarse full score map at level K -----------------------
        # Template at level K: physical scale s, then avg-pooled k times so it is
        # down-sampled IDENTICALLY to the image pyramid (consistency is essential
        # for high-frequency templates — see _scaled_template_at_level).
        per_channel_k, sth_k, stw_k = self._scaled_template_at_level(tmpl, scale, k)
        coarse_map = self._level_score_map(
            k, per_channel_k, sth_k, stw_k, method, channel_mode
        )
        if coarse_map is None:
            return empty, empty, empty, empty, empty
        cy, cx = self._local_max_positions(coarse_map, sth_k, stw_k, coarse_thr)
        if cy.numel() == 0:
            return empty, empty, empty, empty, empty

        # --- step 3: refine ell = K-1 .. 0 ----------------------------------
        # Candidates carried as level-(ell+1) top-left positions; mapped ×2 into
        # the current level at the top of each iteration.
        last_scores = empty
        sth0, stw0 = sth_k, stw_k
        for ell in range(k - 1, -1, -1):
            if cancel is not None and cancel():
                return empty, empty, empty, empty, empty
            per_channel_l, sth_l, stw_l = self._scaled_template_at_level(tmpl, scale, ell)
            # Per-level threshold: relaxed for the intermediate (still-blurred)
            # levels, exact `floor` only at level 0 where positions are final.
            level_thr = floor if ell == 0 else coarse_thr
            cy, cx, sc = self._refine_level(
                ell, cy, cx, per_channel_l, sth_l, stw_l, method, channel_mode, level_thr
            )
            if cy.numel() == 0:
                return empty, empty, empty, empty, empty
            if ell == 0:
                last_scores = sc
                sth0, stw0 = sth_l, stw_l

        # --- step 4: level-0 candidates -> finalize-ready tensors -----------
        gx = cx.to(torch.float32)
        gy = cy.to(torch.float32)
        ws = torch.full_like(gx, float(int(round(stw0))))
        hs = torch.full_like(gx, float(int(round(sth0))))
        scores = last_scores.clamp(min=0.0)
        return gx, gy, ws, hs, scores

    def _level_score_map(
        self,
        level: int,
        per_channel: list[dict],
        sth: int,
        stw: int,
        method: str,
        channel_mode: str,
    ) -> torch.Tensor | None:
        """Full (single-slab) score map for one pyramid ``level`` (§K.1 step 2).

        The coarse level is tiny (image area / ``4**level``) so a single conv over
        the whole level is cheap — no tiling needed. Returns the sanitized 2-D map
        (the final ``max(0, ·)`` NOT yet applied), or ``None`` when the template is
        larger than the level.
        """
        channels = len(per_channel)
        planes = self._level_planes(level, channel_mode, channels)
        hl, wl = planes[0].shape[-2], planes[0].shape[-1]
        if hl - sth + 1 <= 0 or wl - stw + 1 <= 0:
            return None
        m = self._score_map(planes, per_channel, sth, stw, method)[0]
        return torch.nan_to_num(m, nan=-1.0, posinf=-1.0, neginf=-1.0)

    def _local_max_positions(
        self, score_map: torch.Tensor, sth: int, stw: int, thr: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Threshold + local-max prefilter on a 2-D score map (§D.5 reused).

        Identical neighborhood policy to :meth:`_match_tile` (template size capped
        at :data:`_LOCALMAX_MAX_KERNEL`) so coarse peaks are picked the same way
        the full path picks them. Returns ``(rows, cols)`` top-left positions.
        """
        kh = min(sth, _LOCALMAX_MAX_KERNEL)
        kw = min(stw, _LOCALMAX_MAX_KERNEL)
        pooled = _max_filter_2d(score_map, kh, kw)
        peaks = (score_map >= pooled) & (score_map >= thr)
        ys, xs = torch.nonzero(peaks, as_tuple=True)
        return ys, xs

    def _refine_level(
        self,
        level: int,
        cand_y_up: torch.Tensor,
        cand_x_up: torch.Tensor,
        per_channel: list[dict],
        sth: int,
        stw: int,
        method: str,
        channel_mode: str,
        thr: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine candidates at one finer ``level`` with ONE batched conv (§K.1).

        ``cand_*_up`` are top-left positions in the *coarser* (level+1) grid. Each
        is mapped ×2 into this level, a ``±_REFINE_RADIUS`` window opened around
        it, and the matching valid ROI (``window + (s*-1)`` so the conv is valid)
        batch-extracted into a single ``(N,1,Hroi,Wroi)`` tensor per channel. All
        N score maps are produced by ONE batched :meth:`_score_map` call (no
        per-candidate Python loop on the GPU), the per-ROI peak kept and deduped
        to one candidate per grid cell. Returns ``(rows, cols, scores)`` in THIS
        level's grid.
        """
        device = self._device
        empty_l = torch.empty(0, dtype=torch.long, device=device)
        empty_f = torch.empty(0, device=device)
        planes = self._level_planes(level, channel_mode, len(per_channel))
        hl, wl = planes[0].shape[-2], planes[0].shape[-1]

        # ROI spans the search window plus the template support so the conv is
        # valid; the valid output grid is then exactly the ±radius window.
        win_w = (2 * _REFINE_RADIUS + 1) + (stw - 1)
        win_h = (2 * _REFINE_RADIUS + 1) + (sth - 1)
        max_y0 = hl - win_h
        max_x0 = wl - win_w
        if max_y0 < 0 or max_x0 < 0:
            # Level smaller than one ROI: fall back to a single full score map.
            full = self._level_score_map(level, per_channel, sth, stw, method, channel_mode)
            if full is None:
                return empty_l, empty_l, empty_f
            ys, xs = self._local_max_positions(full, sth, stw, thr)
            return ys, xs, full[ys, xs].clamp(min=0.0)

        cy2 = cand_y_up.to(torch.long) * 2
        cx2 = cand_x_up.to(torch.long) * 2
        # ROI top-left = mapped position - REFINE_RADIUS, clamped in-bounds.
        roi_y0 = (cy2 - _REFINE_RADIUS).clamp(min=0, max=max_y0)
        roi_x0 = (cx2 - _REFINE_RADIUS).clamp(min=0, max=max_x0)

        # Build the (M,1,win_h,win_w) ROI batch per channel via vectorized gather
        # (an index grid added to each ROI's top-left — no Python per-candidate
        # loop). The gather + its index grids scale with the candidate count N, so
        # for large N on a VRAM-tight GPU (e.g. the display sharing the card) it
        # would OOM and force the whole query onto the slow full-res fallback. We
        # therefore process candidates in memory-budgeted CHUNKS: each chunk is an
        # independent batched refinement; results are concatenated and the single
        # cross-candidate dedupe below is unchanged, so the output is identical to
        # the one-shot path — it just never exceeds the free-VRAM budget.
        n = roi_y0.numel()
        ar_h = torch.arange(win_h, device=device)
        ar_w = torch.arange(win_w, device=device)
        chunk = self._refine_chunk(win_h, win_w, len(planes), sth, stw)

        gy_parts: list[torch.Tensor] = []
        gx_parts: list[torch.Tensor] = []
        val_parts: list[torch.Tensor] = []
        for start in range(0, n, chunk):
            ry0 = roi_y0[start : start + chunk]
            rx0 = roi_x0[start : start + chunk]
            m = ry0.numel()
            rows = (ry0.view(m, 1, 1) + ar_h.view(1, win_h, 1)).expand(m, win_h, win_w)
            cols = (rx0.view(m, 1, 1) + ar_w.view(1, 1, win_w)).expand(m, win_h, win_w)

            roi_planes = [p[0, 0][rows, cols].unsqueeze(1) for p in planes]
            maps = self._score_map(roi_planes, per_channel, sth, stw, method)  # (m,oh,ow)
            maps = torch.nan_to_num(maps, nan=-1.0, posinf=-1.0, neginf=-1.0)
            oh, ow = maps.shape[-2], maps.shape[-1]

            # Per-ROI argmax peak: the valid output grid is exactly the ±radius
            # window, so its best cell is a local maximum and gives one peak/ROI.
            flat = maps.reshape(m, oh * ow)
            best_val, best_idx = flat.max(dim=1)
            keep = best_val >= thr  # mask (no sync); empty chunks concat harmlessly
            bi = best_idx[keep]
            gy_parts.append(ry0[keep] + bi // ow)
            gx_parts.append(rx0[keep] + bi % ow)
            val_parts.append(best_val[keep])

        gy = torch.cat(gy_parts) if gy_parts else empty_l
        gx = torch.cat(gx_parts) if gx_parts else empty_l
        best_val = torch.cat(val_parts) if val_parts else empty_f
        if gy.numel() == 0:
            return empty_l, empty_l, empty_f

        # Dedupe candidates that landed on the same level cell (overlapping ROIs
        # from neighbouring coarse candidates can converge on one true peak),
        # keeping the max score per cell — vectorized, no Python loop.
        key = gy.to(torch.long) * wl + gx.to(torch.long)
        uniq, inv = torch.unique(key, return_inverse=True)
        u_score = torch.full((uniq.numel(),), -1.0, device=device)
        u_score = u_score.scatter_reduce(0, inv, best_val, reduce="amax", include_self=True)
        u_gy = (uniq // wl).to(torch.long)
        u_gx = (uniq % wl).to(torch.long)
        return u_gy, u_gx, u_score.clamp(min=0.0)

    # -- tiling --------------------------------------------------------------

    def _core_tile_size(self, sth: int, stw: int) -> int:
        """Choose the square core-tile edge for a template of size ``sth x stw``.

        Honors a forced ``core_tile`` (testing). On CUDA, derives a budget from
        free VRAM (§D.7) capped at the conservative 1024 start. On CPU there is
        no OOM ladder, but we still bound the tile so a long CPU match stays
        responsive to cancellation: ``cancel()`` is polled at each tile boundary,
        and a single huge conv2d is atomic/uninterruptible. A 1024 cap means a
        cancel (e.g. window close, or a new selection) is honored within roughly
        one tile's compute instead of after the whole image — preventing a
        wedged shutdown join (and the QThread-destroyed-while-running abort).
        """
        if self._forced_core_tile is not None:
            return max(1, int(self._forced_core_tile))

        if self._device.type == "cuda":
            try:
                free, _ = torch.cuda.mem_get_info()
            except Exception:
                free = 0
            budget = int(free * self._vram_fraction)
            dtype_size = 2 if self._compute_dtype == "float16" else 4
            bytes_per_px = (6 if self._cross_backend(sth, stw) == "spatial" else 9) * dtype_size
            if budget > 0 and bytes_per_px > 0:
                c = int((budget / bytes_per_px) ** 0.5) - max(sth, stw)
                core = max(256, c)
            else:
                core = _TILE_START
            return min(core, _TILE_START)

        # CPU: bound the tile so the match stays cancellable between tiles (see
        # the docstring) and per-tile scratch stays small. 1024 is a good balance
        # of halo overhead vs. cancellation latency on the (slow) CPU fallback.
        return 1024

    def _refine_chunk(
        self, win_h: int, win_w: int, channels: int, sth: int, stw: int
    ) -> int:
        """Max candidates per refinement batch that fit the free-VRAM budget (§K.1).

        The ROI gather in :meth:`_refine_level` allocates, per candidate, two
        ``int64`` index grids plus ``channels`` ROI planes of ``win_h x win_w``,
        and ``_score_map`` adds scratch. When the cross-correlation backend is FFT
        the dominant cost is instead the batched ``rfft2``/``irfft2`` transforms
        (and cuFFT's own work area), which are much larger than the spatial
        scratch — so we size the chunk to whichever the active backend uses. We
        keep a batch within ``vram_fraction`` of *currently free* VRAM — which on
        a display-shared GPU may be a small slice of the card — so a query with
        many coarse candidates refines in several passes instead of OOM-/cuFFT-
        failing onto the slow full-resolution search. CPU runs in a single pass.
        """
        if self._device.type != "cuda":
            return 1 << 30  # effectively "no chunking" on CPU
        try:
            free, _ = torch.cuda.mem_get_info()
        except Exception:
            free = 0
        # Halve the headroom: cuFFT's work area is not counted by PyTorch and the
        # transforms allocate transient complex buffers, so stay well clear of OOM.
        budget = int(free * self._vram_fraction * 0.5)
        dtype_size = 2 if self._compute_dtype == "float16" else 4
        # Always present: two int64 index grids + the gathered ROI plane(s).
        per_cand = win_h * win_w * (8 + 8 + channels * dtype_size)
        if self._cross_backend(sth, stw) == "fft":
            # FFT pads to a smooth (fh x fw); per ROI it holds a half-spectrum
            # complex buffer + a full real buffer, plus comparable cuFFT scratch.
            fh = _next_smooth(win_h + sth - 1)
            fw = _next_smooth(win_w + stw - 1)
            per_cand += fh * (fw // 2 + 1) * 8 * 2 + fh * fw * 4 * 3
        else:
            per_cand += win_h * win_w * 8 * dtype_size  # spatial conv scratch
        if budget <= 0 or per_cand <= 0:
            return 256  # conservative fixed batch when the probe is unavailable
        return max(1, budget // per_cand)

    def _plan_tiles(self, sth: int, stw: int) -> list[tuple[int, int, int, int]]:
        """Partition the valid output grid into halo-padded core tiles.

        The valid top-left grid is ``(H-sth+1, W-stw+1)``. We tile that grid into
        non-overlapping cores of ``core x core``; each tile's *input* region adds
        a halo of ``(sth-1, stw-1)`` on the right/bottom so every valid top-left
        whose core cell lies in this tile has its full window available — no
        valid position is split across a seam (§D.6, H5).

        Returns a list of ``(oy0, ox0, oh, ow)`` core-grid rectangles (output
        coordinates: top-left positions), in row-major order.
        """
        H, W = self._h, self._w
        out_h = H - sth + 1
        out_w = W - stw + 1
        if out_h <= 0 or out_w <= 0:
            return []  # template larger than the image at this scale — skip.

        core = self._core_tile_size(sth, stw)
        tiles: list[tuple[int, int, int, int]] = []
        oy = 0
        while oy < out_h:
            oh = min(core, out_h - oy)
            ox = 0
            while ox < out_w:
                ow = min(core, out_w - ox)
                tiles.append((oy, ox, oh, ow))
                ox += core
            oy += core
        return tiles

    def _cross_backend(self, sth: int, stw: int) -> str:
        """Resolve the cross-correlation backend for a template size (§D.2)."""
        if self._conv_backend == "spatial":
            return "spatial"
        if self._conv_backend == "fft":
            return "fft"
        return "spatial" if (sth * stw) <= _FFT_THRESHOLD else "fft"

    def _match_tile(
        self,
        per_channel: list[dict],
        sth: int,
        stw: int,
        scale: float,
        tile: tuple[int, int, int, int],
        params: MatchParams,
        channel_mode: str,
        method: str,
    ) -> tuple[torch.Tensor, ...]:
        """Compute the score map + peak extraction for one core tile.

        ``method`` (one of :data:`~fastmatch.types.CONV_METHODS`) only selects the
        per-window score formula via :meth:`_score_map`; the local-max prefilter,
        threshold floor, halo/core bookkeeping below are identical across methods.

        Returns flat ``(x, y, w, h, score)`` tensors (image-px top-lefts and the
        scaled box size) for peaks whose top-left falls inside the tile core.
        """
        oy0, ox0, oh, ow = tile
        # Input region for this tile: core plus right/bottom halo of (sth-1,stw-1)
        # so windows anchored at the last core row/col still have full support.
        iy0 = oy0
        ix0 = ox0
        iy1 = min(self._h, oy0 + oh + sth - 1)
        ix1 = min(self._w, ox0 + ow + stw - 1)

        # Select the image channel slab(s) matching the template channels. In rgb
        # mode (with RGB staged) each template channel pairs with its own image
        # channel so the per-channel numerators/denominators can be summed before
        # dividing (§D.1); otherwise a single luminance channel is used.
        if channel_mode == "rgb" and self._rgb is not None and len(per_channel) == 3:
            # The rgb planes are staged UINT8 (§ memory contract); convert the
            # tile slice to fp32 [0,1] here so it matches the template channels
            # (themselves [0,1]) and the already-normalized luminance path.
            img_planes = [
                p[:, :, iy0:iy1, ix0:ix1].to(torch.float32) / 255.0 for p in self._rgb
            ]
        elif channel_mode == "ycbcr" and self._ycbcr is not None and len(per_channel) == 3:
            # Y,Cb,Cr planes are staged UINT8 (BT.601 full-range); same per-tile
            # fp32 [0,1] conversion as rgb so they pair with the ycbcr template
            # channels. The per-channel weights live on per_channel.
            img_planes = [
                p[:, :, iy0:iy1, ix0:ix1].to(torch.float32) / 255.0 for p in self._ycbcr
            ]
        else:
            img_planes = [self._lum[:, :, iy0:iy1, ix0:ix1]]

        # The valid output grid produced by this image slab.
        out_h = (iy1 - iy0) - sth + 1
        out_w = (ix1 - ix0) - stw + 1
        if out_h <= 0 or out_w <= 0:
            empty = torch.empty(0, device=self._device)
            return empty, empty, empty, empty, empty

        # Dispatch on method (§J.2). The returned map is the method's natural
        # score map BEFORE the final ``max(0, ·)`` clamp: signed in [-1,1] for
        # NCC (so it is bit-for-bit the legacy ``ncc`` map and the existing tests
        # still pass), already non-negative in [0,1] for SSD/CCORR. The common
        # nan_to_num(-1) sentinel + clamp(min=0) below then handles all three
        # uniformly (negative correlation -> 0 for NCC; no-op for SSD/CCORR).
        # The score helpers keep a leading batch dim (B==1 for a single tile);
        # squeeze it back to the 2-D map this peak-extraction path expects.
        ncc = self._score_map(img_planes, per_channel, sth, stw, method)[0]

        # Sanitize before any thresholding so a stray inf/nan can't masquerade as
        # a peak (§D.1). -1 is below every valid score, so a sanitized cell can
        # never out-rank a real peak in the local-max prefilter.
        ncc = torch.nan_to_num(ncc, nan=-1.0, posinf=-1.0, neginf=-1.0)
        # The nan_to_num above already enforces the §D.3/H2 finiteness invariant,
        # so no per-tile assert is needed (an unconditional torch.isfinite(...).all()
        # forces a device->host sync every tile). An opt-in debug net is available
        # via the _debug_finite_check flag (default False) for diagnosing a
        # suspected sanitize regression without paying the sync in normal runs.
        if getattr(self, "_debug_finite_check", False):
            assert torch.isfinite(ncc).all(), "non-finite score map after sanitize"

        floor = float(params.threshold_floor)
        # Local-max prefilter via max-pool: a peak must equal the pooled max in
        # its neighborhood and clear the floor (§D.5). The neighborhood is the
        # template size, capped at _LOCALMAX_MAX_KERNEL so the (CPU) cost and
        # cancellation latency stay bounded for large templates; the global
        # IoU-NMS downstream merges any extra peaks within one footprint.
        kh = min(sth, _LOCALMAX_MAX_KERNEL)
        kw = min(stw, _LOCALMAX_MAX_KERNEL)
        # Separable stride-1 max filter (see _max_filter_2d): O(kh+kw)/px instead
        # of O(kh*kw), already cropped to the ncc grid. Identical peaks.
        pooled = _max_filter_2d(ncc, kh, kw)
        peaks = (ncc >= pooled) & (ncc >= floor)

        ys, xs = torch.nonzero(peaks, as_tuple=True)  # output-grid (row,col)
        if ys.numel() == 0:
            empty = torch.empty(0, device=self._device)
            return empty, empty, empty, empty, empty

        scores = ncc[ys, xs].clamp(min=0.0)  # score = max(0, ncc)

        # Box size at this scale (§D.4): round the scaled template extent.
        bw = int(round(stw))  # stw already == round(tw*s)
        bh = int(round(sth))

        # Translate output-grid positions to full-image top-left pixels.
        gx = xs + ix0
        gy = ys + iy0

        # Keep only detections whose TOP-LEFT falls in this tile's core cell so
        # every valid position is counted by exactly one tile (§D.6 / H5). The
        # halo guarantees the full window for any top-left in [ox0, ox0+ow) is
        # present in this tile's input slab; the *center* can land in the next
        # tile's core, so testing the center would drop seam-straddling windows
        # (no tile would own them). The final global NMS still dedupes any
        # genuine duplicate peaks across seams.
        in_core = (
            (gx >= ox0)
            & (gx < ox0 + ow)
            & (gy >= oy0)
            & (gy < oy0 + oh)
        )
        if not bool(in_core.any()):
            empty = torch.empty(0, device=self._device)
            return empty, empty, empty, empty, empty

        gx = gx[in_core].to(torch.float32)
        gy = gy[in_core].to(torch.float32)
        scores = scores[in_core]
        ws = torch.full_like(gx, float(bw))
        hs = torch.full_like(gx, float(bh))
        return gx, gy, ws, hs, scores

    # -- score-map core (method-dispatched) ----------------------------------

    def _score_map(
        self,
        img_planes: list[torch.Tensor],
        per_channel: list[dict],
        sth: int,
        stw: int,
        method: str,
        backend: str | None = None,
    ) -> torch.Tensor:
        """Compute the per-window score map for one of the convolution methods.

        Shared machinery for ``ncc``/``ssd``/``ccorr`` (§J.2): the *only*
        difference between the three is the final score formula. This thin wrapper
        computes the per-channel score *terms* (cross/box-filter quantities) with
        :meth:`_compute_score_terms` and then applies the pure-elementwise
        :meth:`_score_from_terms` (§K.1 refactor) — exactly the split the batched
        coarse-to-fine ROI path also reuses, so both paths share one formula and
        NCC stays bit-for-bit identical on the full path.

        Each plane in ``img_planes`` is a 4-D ``(B,1,H,W)`` slab (the full path
        passes ``B == 1`` tiles; the batched-ROI path passes ``B == N`` stacked
        ROIs). The returned map preserves the leading batch dim: ``(B, oh, ow)``.

        Luminance mode passes a single plane paired with the single template
        channel. ``rgb`` mode passes three image planes paired with the three
        template channels and **sums the per-channel terms before the single
        division** (§D.1/§J.2) — for ncc/ccorr the numerators and denominator
        terms are summed before dividing; for ssd the per-channel SSDs are summed
        then divided by ``n * channels``. ``img_planes`` and ``per_channel`` are
        paired positionally and must be the same length.

        Returns the method's natural map *before* the final ``max(0, ·)`` clamp
        (the caller sanitizes and clamps): signed in ``[-1, 1]`` for NCC (so it
        is bit-for-bit identical to the legacy NCC map — the regression baseline),
        and already in ``[0, 1]`` for SSD/CCORR.
        """
        terms = self._compute_score_terms(img_planes, per_channel, sth, stw, backend)
        return self._score_from_terms(method, terms)

    def _compute_score_terms(
        self,
        img_planes: list[torch.Tensor],
        per_channel: list[dict],
        sth: int,
        stw: int,
        backend: str | None = None,
    ) -> dict:
        """Compute the per-window score *terms*, summed across channels (§K.1).

        Returns the method-agnostic quantities every convolution score is built
        from — ``cross_z``, ``cross_raw``, ``S2``, the (scalar) template ``sumT2``,
        the NCC denominator ``sqrt(n·varI)·normTz``, the per-channel-summed SSD,
        the featureless-window guard ``flat``, the per-channel pixel count ``n``
        and the channel count — so the pure-elementwise :meth:`_score_from_terms`
        can produce ncc/ssd/ccorr without re-touching the (expensive) conv/box
        machinery. Shape-agnostic: ``img_planes`` are 4-D ``(B,1,H,W)`` slabs and
        every term keeps the leading batch dim (``B == 1`` for a full-path tile,
        ``B == N`` for the batched-ROI refinement). The accumulation order is the
        legacy one so the resulting NCC map is bit-for-bit unchanged.
        """
        n = per_channel[0]["n"]
        # A caller may force the backend (the batched ROI refinement forces
        # "spatial": FFT over a batch of small ROIs needs a huge work area and only
        # yields a tiny valid window — spatial conv is cheaper and VRAM-light, and
        # avoids cuFFT failing to allocate on a memory-tight/display-shared GPU).
        backend = backend or self._cross_backend(sth, stw)
        channels = len(per_channel)

        # Per-channel accumulators, summed BEFORE the single division (§J.2 RGB).
        cross_z_sum: torch.Tensor | None = None     # Σ_c conv(I_c, Tz_c)
        cross_raw_sum: torch.Tensor | None = None   # Σ_c (cross_z_c + meanT_c·S1_c)
        s2_sum: torch.Tensor | None = None          # Σ_c S2_c
        sum_t2_sum: torch.Tensor | None = None      # Σ_c sumT2_c (template scalar)
        ncc_denom_sum: torch.Tensor | None = None   # Σ_c sqrt(n·varI_c)·normTz_c
        ssd_sum: torch.Tensor | None = None         # Σ_c clamp(S2−2·cross_raw+sumT2, 0)
        flat: torch.Tensor | None = None            # featureless-window guard (NCC)

        # Sum of the per-channel weights (== 1 for normalized multi-channel modes,
        # == 1 for the single luminance channel). Used as the SSD normalizer so a
        # weighted average of the per-channel squared error is taken.
        sum_w = 0.0
        for img, ch in zip(img_planes, per_channel):
            img32 = img.to(torch.float32)
            # Per-channel weight (luminance == 1.0, so the legacy path is a no-op
            # multiply and stays bit-identical; multi-channel modes weight each
            # channel's contribution to the combined score, §channel modes).
            w = float(ch.get("weight", 1.0))
            sum_w += w

            # Separable box filters give the windowed sum S1 and sum-of-squares
            # S2 for this channel in fp32 accumulators (§D.1).
            s1 = self._box_filter(img32, sth, stw)
            s2 = self._box_filter(img32 * img32, sth, stw)

            # n*varI = clamp(S2 - S1^2/n, 0); clamp removes fp negatives.
            n_var = (s2 - (s1 * s1) / n).clamp(min=0.0)

            # Flat-region guard (NCC only): a window is featureless if EVERY
            # channel is flat. SSD/CCORR do not use it (flat is their use case).
            chan_flat = n_var < (_VAR_FLOOR * n)
            flat = chan_flat if flat is None else (flat & chan_flat)

            # Cross term: conv2d is cross-correlation (no flip), so conv(I, Tz)
            # directly yields sum_w(I * Tz). fp16 mode may run the conv input in
            # half precision (§D.3); the accumulation back to fp32 happens below.
            if backend == "spatial":
                conv_in = img.to(torch.float16) if self._compute_dtype == "float16" else img32
                cr = self._cross_spatial(conv_in, ch["tz"])
            else:
                cr = self._cross_fft(img32, ch["tz"], sth, stw)
            cr = cr.to(torch.float32)  # cross_z for this channel

            # Raw cross term derived without a second convolution (§J.2):
            # cross_raw = cross_z + meanT · S1.
            cross_raw = cr + ch["mean_t"] * s1

            # Each per-channel contribution is scaled by its weight before the
            # cross-channel sum (the single division then yields a weighted NCC /
            # CCORR; SSD divides by n·Σw below). For NCC/CCORR a common scale on
            # numerator and denominator cancels, so equal weights == unweighted.
            cr_w = w * cr
            cross_raw_w = w * cross_raw
            s2_w = w * s2
            sum_t2_w = w * ch["sum_t2"]
            denom_c = w * (torch.sqrt(n_var) * ch["norm_tz"])
            ssd_c = w * (s2 - 2.0 * cross_raw + ch["sum_t2"]).clamp(min=0.0)

            cross_z_sum = cr_w if cross_z_sum is None else (cross_z_sum + cr_w)
            cross_raw_sum = cross_raw_w if cross_raw_sum is None else (cross_raw_sum + cross_raw_w)
            s2_sum = s2_w if s2_sum is None else (s2_sum + s2_w)
            sum_t2_sum = sum_t2_w if sum_t2_sum is None else (sum_t2_sum + sum_t2_w)
            ncc_denom_sum = denom_c if ncc_denom_sum is None else (ncc_denom_sum + denom_c)
            ssd_sum = ssd_c if ssd_sum is None else (ssd_sum + ssd_c)

        assert (
            cross_z_sum is not None
            and cross_raw_sum is not None
            and s2_sum is not None
            and sum_t2_sum is not None
            and ncc_denom_sum is not None
            and ssd_sum is not None
            and flat is not None
        )

        return {
            "cross_z": cross_z_sum,
            "cross_raw": cross_raw_sum,
            "s2": s2_sum,
            "sum_t2": sum_t2_sum,
            "ncc_denom": ncc_denom_sum,
            "ssd": ssd_sum,
            "flat": flat,
            "n": n,
            "channels": channels,
            "sum_w": sum_w if sum_w > 0 else 1.0,
        }

    def _score_from_terms(self, method: str, terms: dict) -> torch.Tensor:
        """Pure-elementwise score formula from precomputed terms (§K.1 / §J.2).

        Shape-agnostic: operates on whatever shape the term tensors carry (a
        ``(oh, ow)`` tile map on the full path, a ``(N, oh, ow)`` batch on the
        ROI-refinement path) — there are no convolutions or box filters here, only
        the final per-method arithmetic, so the tiled full path and the batched
        coarse-to-fine path produce identical scores. NCC's expression and op
        order are unchanged from the original ``_score_map`` so the NCC regression
        tests still pass bit-for-bit.
        """
        eps = self._eps

        if method == "ncc":
            # coeff = cross_z / (sqrt(n·varI)·normTz + eps); flat windows -> 0.
            ncc = terms["cross_z"] / (terms["ncc_denom"] + eps)
            ncc = torch.where(terms["flat"], torch.zeros_like(ncc), ncc)
            return ncc

        if method == "ccorr":
            # Cosine cross-correlation: cc = cross_raw / (sqrt(S2)·sqrt(sumT2)+eps).
            # Summed per-channel numerators/denominator terms before dividing.
            cc = terms["cross_raw"] / (
                torch.sqrt(terms["s2"]) * torch.sqrt(terms["sum_t2"]) + eps
            )
            return cc.clamp(min=0.0, max=1.0)

        if method == "ssd":
            # Normalized squared difference -> flat-friendly similarity score.
            # rmse = sqrt(weighted-SSD / (n · Σw)); score = clamp(1 - rmse, 0, 1).
            # The per-channel SSD is weight-scaled (Σ w_c SSD_c), so dividing by
            # n·Σw_c takes a weighted average per-pixel squared error. For a single
            # channel (Σw=1) or equal weights this matches the legacy n·channels.
            rmse = torch.sqrt(terms["ssd"] / (terms["n"] * terms.get("sum_w", terms["channels"])))
            return (1.0 - rmse).clamp(min=0.0, max=1.0)

        raise ValueError(f"unsupported convolution method {method!r}")

    def _box_filter(self, x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
        """Separable windowed-sum box filter (O(kh+kw)), fp32 accumulators.

        Sums over each ``kh x kw`` window via two 1-D ``ones`` convolutions
        (vertical then horizontal). Input ``(B,1,H,W)``; output ``(B,H-kh+1,
        W-kw+1)`` — the single channel dim is squeezed but the batch dim is kept
        so the same kernel serves both the ``B==1`` full path and the ``B==N``
        batched-ROI refinement (§K.1) without any per-candidate Python loop.
        """
        # Vertical sum: a (kh,1) ones kernel reduces H -> H-kh+1.
        ones_v = torch.ones((1, 1, kh, 1), device=x.device, dtype=torch.float32)
        v = F.conv2d(x, ones_v)
        # Horizontal sum: a (1,kw) ones kernel reduces W -> W-kw+1.
        ones_h = torch.ones((1, 1, 1, kw), device=x.device, dtype=torch.float32)
        s = F.conv2d(v, ones_h)
        return s[:, 0]

    def _cross_spatial(self, img: torch.Tensor, tz: torch.Tensor) -> torch.Tensor:
        """Dense cross-correlation ``sum_w(I * Tz)`` via ``F.conv2d`` (no flip).

        Input ``(B,1,H,W)`` against the shared single-channel template weight
        ``(1,1,sth,stw)`` -> ``(B, oh, ow)`` (channel squeezed, batch kept) so one
        ``F.conv2d`` covers all ``B`` ROIs at once on the refinement path.
        """
        weight = tz.to(img.dtype)
        out = F.conv2d(img, weight)
        return out[:, 0]

    def _cross_fft(
        self, img: torch.Tensor, tz: torch.Tensor, sth: int, stw: int
    ) -> torch.Tensor:
        """Cross-correlation via FFT for large templates (§D.2).

        Circular convolution of the image with a *flipped* template equals linear
        cross-correlation once we pad both to a common length >= H+sth-1 (so the
        wrap-around tail does not contaminate the valid region) and crop the
        valid ``(H-sth+1, W-stw+1)`` block. Input ``(B,1,H,W)``; the rfft2 batches
        over the leading dims so the output keeps the batch: ``(B, oh, ow)``.
        """
        ih, iw = img.shape[-2], img.shape[-1]
        # Linear-conv length, padded to a 2-3-5-7 smooth size for FFT speed.
        fh = _next_smooth(ih + sth - 1)
        fw = _next_smooth(iw + stw - 1)

        img2 = img[:, 0]  # (B,H,W)
        tzf = torch.flip(tz[0, 0], dims=(0, 1))  # flip so circular conv == correlation

        # rfft2 transforms the trailing two dims, broadcasting the leading batch.
        # cuFFT can fail to allocate its work area on a VRAM-tight/display-shared
        # GPU (CUFFT_INTERNAL_ERROR) even when the tensors fit; spatial conv needs
        # no work area, so fall back to it rather than aborting the whole query.
        try:
            img_fft = torch.fft.rfft2(img2, s=(fh, fw))
            tz_fft = torch.fft.rfft2(tzf, s=(fh, fw))
            conv = torch.fft.irfft2(img_fft * tz_fft, s=(fh, fw))
        except RuntimeError as exc:
            if "fft" not in str(exc).lower() and "cufft" not in str(exc).lower():
                raise
            return self._cross_spatial(img, tz)

        # The valid cross-correlation output lives at offset (sth-1, stw-1) and
        # spans (ih-sth+1, iw-stw+1).
        out_h = ih - sth + 1
        out_w = iw - stw + 1
        return conv[:, sth - 1 : sth - 1 + out_h, stw - 1 : stw - 1 + out_w]

    # -- OOM degradation ladder (CUDA only) ----------------------------------

    def _degrade_after_oom(
        self,
        tmpl: dict,
        scales: tuple[float, ...],
        params: MatchParams,
        channel_mode: str,
        method: str,
        cancel: Callable[[], bool] | None,
        progress: Callable[[int], None] | None,
    ) -> dict:
        """Walk the CUDA OOM ladder: 1024 -> 512 tile -> fewer scales -> CPU.

        Each rung empties the cache and retries; the very last rung moves the
        staged image to CPU and runs there. Never silently fails — re-raises if
        even the CPU rung OOMs (which would be a host-memory problem).
        """
        original_forced = self._forced_core_tile
        original_device = self._device
        original_lum = self._lum
        original_rgb = self._rgb
        original_ycbcr = self._ycbcr

        # Rung 1-2: shrink the tile (only if not already forced by the caller).
        if original_forced is None:
            for tile_sz in _TILE_LADDER:
                torch.cuda.empty_cache()
                self._forced_core_tile = tile_sz
                try:
                    return self._search_all_scales(
                        tmpl, scales, params, channel_mode, method, cancel, progress
                    )
                except torch.cuda.OutOfMemoryError:
                    continue

        # Rung 3: fewer scales — keep just the native scale (1.0) if present,
        # else the median scale, on the smallest tile.
        torch.cuda.empty_cache()
        self._forced_core_tile = _TILE_LADDER[-1]
        reduced = (1.0,) if 1.0 in scales else (scales[len(scales) // 2],)
        try:
            return self._search_all_scales(
                tmpl, reduced, params, channel_mode, method, cancel, progress
            )
        except torch.cuda.OutOfMemoryError:
            pass

        # Rung 4: CPU fallback. Move the staged luminance, the staged RGB / YCbCr
        # planes (channel_mode rgb/ycbcr read these in _match_tile), and the
        # template channels to CPU, run there, then restore CUDA state. Leaving any
        # staged plane on the GPU would crash the CPU run with a device mismatch
        # (CUDA image × CPU template) in _cross_spatial/_cross_fft.
        torch.cuda.empty_cache()
        try:
            self._device = torch.device("cpu")
            self._forced_core_tile = original_forced  # CPU uses its own large tile
            self._lum = original_lum.to("cpu")
            self._rgb = (
                [p.to("cpu") for p in original_rgb] if original_rgb is not None else None
            )
            self._ycbcr = (
                [p.to("cpu") for p in original_ycbcr] if original_ycbcr is not None else None
            )
            cpu_tmpl = {
                "channels": [c.to("cpu") for c in tmpl["channels"]],
                "th": tmpl["th"],
                "tw": tmpl["tw"],
                "weights": tmpl.get("weights"),
            }
            return self._search_all_scales(
                cpu_tmpl, scales, params, channel_mode, method, cancel, progress
            )
        finally:
            # Restore CUDA state so the matcher stays GPU-backed for later jobs.
            self._device = original_device
            self._forced_core_tile = original_forced
            self._lum = original_lum
            self._rgb = original_rgb
            self._ycbcr = original_ycbcr

    # -- feature matching delegation (§J.3 / §J.4) ---------------------------

    def _match_features(
        self,
        template: np.ndarray,
        params: MatchParams,
        *,
        exclude_box: tuple[int, int, int, int] | None,
        cancel: Callable[[], bool] | None,
        progress: Callable[[int], None] | None,
    ) -> list[Match]:
        """Delegate ``method=="features"`` to the OpenCV feature backend (§J.4).

        The :class:`~fastmatch.feature_matcher.FeatureMatcher` is built lazily on
        first use (so importing :mod:`fastmatch.engine` never imports cv2) and
        reused across calls — it caches the staged full-image luminance's
        keypoints/descriptors per detector, invalidated by :meth:`set_image`.
        Feature matching always runs on CPU even when the conv methods use CUDA.

        Keypoint detection is always on luminance (ORB/AKAZE/SIFT are grayscale),
        but ``channel_mode="rgb"`` makes the *appearance verification* color-aware:
        we reconstruct a host ``(H,W,3)`` RGB array once from the staged per-channel
        planes (cached in ``self._rgb_u8``) and hand it to the backend, which then
        verifies each proposal with a per-channel-summed NCC (mirroring the conv
        path's rgb rule). For a grayscale source there is no color, so rgb mode
        transparently falls back to luminance.

        Raises:
            RuntimeError: if OpenCV is not importable (surfaced from the backend).
        """
        if self._lum_u8 is None:
            raise RuntimeError("Matcher.set_image() must be called before match()")

        if self._feature_matcher is None:
            # Import lazily INSIDE the feature path so the cv2 dependency is only
            # required by this method (§J.3/§J.4) — `import fastmatch.engine` and
            # the conv methods stay cv2-free.
            from .feature_matcher import FeatureMatcher

            self._feature_matcher = FeatureMatcher()
            # The image was staged before this matcher existed; prime its cache by
            # treating the current staged luminance as the active image.
            self._feature_matcher.invalidate_image()

        # rgb/ycbcr verification needs a host (H,W,3) RGB array (the feature path
        # converts it to YCbCr itself). Reconstruct it once from the staged
        # per-channel planes (works whether they live on CPU or CUDA) and cache it;
        # only built when a colour mode is actually requested on a colour image.
        channel_mode = params.channel_mode or self._channel_mode
        image_rgb = None
        if channel_mode in ("rgb", "ycbcr") and self._rgb is not None:
            if self._rgb_u8 is None:
                self._rgb_u8 = np.ascontiguousarray(
                    np.stack(
                        [p[0, 0].detach().to("cpu").numpy() for p in self._rgb],
                        axis=-1,
                    )
                )
            image_rgb = self._rgb_u8

        return self._feature_matcher.match(
            template,
            self._lum_u8,
            params,
            image_rgb=image_rgb,
            exclude_box=exclude_box,
            cancel=cancel,
            progress=progress,
            scale_back=1,
        )

    def _match_shazam(
        self,
        template: np.ndarray,
        params: MatchParams,
        *,
        exclude_box: tuple[int, int, int, int] | None,
        cancel: Callable[[], bool] | None,
        progress: Callable[[int], None] | None,
    ) -> list[Match]:
        """Delegate the experimental ``method="shazam"`` to the pair-hash backend.

        DISABLED by default (``"shazam"`` is not in METHODS, so the UI never selects
        it); reachable only when set explicitly. Reuses the feature backend's dense
        ORB detection + per-image cache for the image keypoints, then runs the
        landmark-pair-hash voting in :class:`~fastmatch.shazam_matcher.ShazamMatcher`.
        Feature detection runs on CPU (OpenCV) even when the conv methods use CUDA.
        """
        if self._lum_u8 is None:
            raise RuntimeError("Matcher.set_image() must be called before match()")
        if self._feature_matcher is None:
            from .feature_matcher import FeatureMatcher

            self._feature_matcher = FeatureMatcher()
            self._feature_matcher.invalidate_image()
        if self._shazam_matcher is None:
            from .shazam_matcher import ShazamMatcher

            self._shazam_matcher = ShazamMatcher()

        detector_name = getattr(params, "feature_detector", "orb") or "orb"
        img_kps, img_desc = self._feature_matcher._ensure_image_features(
            self._lum_u8, detector_name, 0, cancel
        )
        return self._shazam_matcher.match(
            template,
            self._lum_u8,
            params,
            feature_matcher=self._feature_matcher,
            img_kps=img_kps,
            img_desc=img_desc,
            exclude_box=exclude_box,
            cancel=cancel,
            progress=progress,
            scale_back=1,
        )

    # -- finalize: exclusion, global NMS, top-K ------------------------------

    def _finalize(
        self,
        cands: dict,
        params: MatchParams,
        exclude_box: tuple[int, int, int, int] | None,
        th0: int,
        tw0: int,
    ) -> list[Match]:
        """Apply source exclusion, one global NMS, top-K, and build Matches."""
        scores = cands["score"]
        if scores.numel() == 0:
            return []

        device = scores.device
        x = cands["x"]
        y = cands["y"]
        w = cands["w"]
        h = cands["h"]
        scale = cands["scale"]

        # NMS/IoU operate on [x1,y1,x2,y2] (half-open: x2=x+w).
        boxes = torch.stack([x, y, x + w, y + h], dim=1)

        # Source exclusion: drop any candidate overlapping the source box past
        # exclude_iou, or whose center lies inside it (§D.5 step 3 / H4).
        if exclude_box is not None:
            iou_src = _iou_against_box(boxes, exclude_box, device)
            cx = x + w / 2.0
            cy = y + h / 2.0
            ex, ey, ew, eh = exclude_box
            center_in = (
                (cx >= ex) & (cx < ex + ew) & (cy >= ey) & (cy < ey + eh)
            )
            drop = (iou_src > float(params.exclude_iou)) | center_in
            keep_mask = ~drop
            if not bool(keep_mask.any()):
                return []
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            scale = scale[keep_mask]

        # One global NMS across all scales/tiles so seam straddlers and
        # cross-scale duplicates collapse to a single hit (§D.4/§D.6).
        keep = _run_nms(boxes, scores, float(params.nms_iou))
        boxes = boxes[keep]
        scores = scores[keep]
        scale = scale[keep]

        # Top-K by score desc (§D.5 step 5).
        if scores.numel() > params.max_results:
            top = torch.argsort(scores, descending=True)[: params.max_results]
        else:
            top = torch.argsort(scores, descending=True)
        boxes = boxes[top]
        scores = scores[top]
        scale = scale[top]

        # Move to CPU once, build the plain-data Match list (no tensors cross out).
        boxes_c = boxes.cpu().numpy()
        scores_c = scores.cpu().numpy()
        scale_c = scale.cpu().numpy()

        results: list[Match] = []
        for i in range(boxes_c.shape[0]):
            bx1, by1, bx2, by2 = boxes_c[i]
            results.append(
                Match(
                    x=int(round(float(bx1))),
                    y=int(round(float(by1))),
                    w=int(round(float(bx2 - bx1))),
                    h=int(round(float(by2 - by1))),
                    score=float(scores_c[i]),
                    scale=float(scale_c[i]),
                )
            )
        return results
