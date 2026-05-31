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

from .device import resolve_device
from .types import CONV_METHODS, METHODS, Match, MatchParams

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

    keep: list[int] = []
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
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

    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


def _run_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Dispatch to torchvision NMS if available, else the bundled greedy NMS."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    if _HAVE_TV_NMS:
        # torchvision.ops.nms wants float boxes/scores and returns sorted-desc
        # kept indices — identical contract to our fallback.
        return _tv_nms(boxes.float(), scores.float(), float(iou_thr))
    return _greedy_nms(boxes, scores, iou_thr)


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
                results, only how the output grid is partitioned.
        """
        self._device = resolve_device(device if device is not None else "auto")
        self._compute_dtype = compute_dtype
        self._channel_mode = channel_mode
        self._conv_backend = conv_backend
        self._vram_fraction = float(vram_fraction)
        self._forced_core_tile = core_tile

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
        self._h: int = 0
        self._w: int = 0
        # Host-side uint8 luminance for the OpenCV feature path (§J.3); the conv
        # methods never read it. Populated by set_image().
        self._lum_u8: np.ndarray | None = None

        # Lazily-built OpenCV feature backend (only when method=="features"). It
        # owns a per-(image, detector) cache of downsampled-image keypoints/
        # descriptors so repeated feature queries on the same image are cheap;
        # set_image() invalidates it. Kept as Any-typed to avoid importing the
        # feature module (and thus cv2) unless the feature path is exercised.
        self._feature_matcher = None  # type: ignore[var-annotated]

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
        if method not in CONV_METHODS:
            raise ValueError(
                f"unknown matching method {method!r}; expected one of {sorted(METHODS)}"
            )

        # Channel mode can be overridden per-call by params (the controller maps
        # the UI choice onto MatchParams); fall back to the constructor default.
        channel_mode = params.channel_mode or self._channel_mode
        # rgb mode is only meaningful when both the image and template carry
        # color; otherwise transparently fall back to luminance so the per-channel
        # pairing in _score_map always lines up.
        if channel_mode == "rgb" and (self._rgb is None or template.ndim != 3):
            channel_mode = "luminance"

        # Prepare the device-resident template channels once (validated here).
        # The variance-floor rejection is gated on method (NCC only) inside.
        tmpl = self._prepare_template(template, channel_mode, method)
        th0, tw0 = tmpl["th"], tmpl["tw"]

        scales = tuple(params.scales) if params.scales else (1.0,)
        device = self._device

        # The whole search is wrapped in the CUDA OOM degradation ladder. On CPU
        # the ladder collapses to a single straight run (no OOM expected).
        try:
            cands = self._search_all_scales(
                tmpl, scales, params, channel_mode, method, cancel, progress
            )
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise  # CPU OOM is a real failure, not something we degrade.
            cands = self._degrade_after_oom(
                tmpl, scales, params, channel_mode, method, cancel, progress
            )

        # empty_cache between *jobs* (not between tiles) — frees the tile
        # scratch this match() allocated before the next match() runs.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if cancel is not None and cancel():
            return []

        return self._finalize(cands, params, exclude_box, th0, tw0)

    # -- template preparation / validation -----------------------------------

    def _prepare_template(
        self, template: np.ndarray, channel_mode: str, method: str = "ncc"
    ) -> dict:
        """Validate and stage the template channels on the compute device.

        Returns a dict with the per-channel ``[0,1]`` template tensors (one for
        luminance mode, three for rgb mode) plus ``th``/``tw``. Per-scale
        zero-mean / norm are derived later in :meth:`_scaled_template`.

        Args:
            template: ``(th, tw, 3)`` uint8 RGB or ``(th, tw)`` uint8 crop.
            channel_mode: ``"luminance"`` or ``"rgb"`` (controls channel staging).
            method: The matching method (§J.2). The variance-floor rejection only
                applies to ``"ncc"`` — SSD/CCORR are *defined* to handle flat /
                low-variance templates (that is their whole use case, §J.1/§J.2),
                so they must not raise here.

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
        if channel_mode == "rgb" and template.ndim == 3:
            rgb = np.asarray(template, dtype=np.float32) / 255.0
            for c in range(3):
                channels.append(torch.from_numpy(np.ascontiguousarray(rgb[:, :, c])).to(self._device))
        else:
            # Luminance (or a genuinely grayscale template).
            if template.ndim == 3:
                lum = self._to_luminance(template)
            else:
                lum = np.asarray(template, dtype=np.float32) / 255.0
            channels.append(torch.from_numpy(np.ascontiguousarray(lum)).to(self._device))

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

        return {"channels": channels, "th": th, "tw": tw}

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
        per_channel: list[dict] = []
        for c in chans:
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
        ncc = self._score_map(img_planes, per_channel, sth, stw, method)

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
        pooled = F.max_pool2d(
            ncc.unsqueeze(0).unsqueeze(0),
            kernel_size=(kh, kw),
            stride=1,
            padding=(kh // 2, kw // 2),
        )
        # Padding can make the pooled map larger by a pixel on even kernels;
        # crop back to the ncc grid before comparing.
        pooled = pooled[0, 0, : ncc.shape[0], : ncc.shape[1]]
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
    ) -> torch.Tensor:
        """Compute the per-window score map for one of the convolution methods.

        Shared machinery for ``ncc``/``ssd``/``ccorr`` (§J.2): the *only*
        difference between the three is the final score formula. All three reuse
        the same per-channel quantities the engine already computes —
        ``cross_z = conv(I, Tz)`` (zero-mean template), the box-filtered windowed
        sum ``S1`` and sum-of-squares ``S2``, and the template scalars ``meanT``,
        ``normTz``, ``sumT2``. The *raw* cross term is derived without a second
        convolution: ``cross_raw = cross_z + meanT * S1`` (since ``T = Tz +
        meanT``).

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
        n = per_channel[0]["n"]
        eps = self._eps
        backend = self._cross_backend(sth, stw)
        channels = len(per_channel)

        # Per-channel accumulators, summed BEFORE the single division (§J.2 RGB).
        cross_z_sum: torch.Tensor | None = None     # Σ_c conv(I_c, Tz_c)
        cross_raw_sum: torch.Tensor | None = None   # Σ_c (cross_z_c + meanT_c·S1_c)
        s2_sum: torch.Tensor | None = None          # Σ_c S2_c
        sum_t2_sum: torch.Tensor | None = None      # Σ_c sumT2_c (template scalar)
        ncc_denom_sum: torch.Tensor | None = None   # Σ_c sqrt(n·varI_c)·normTz_c
        ssd_sum: torch.Tensor | None = None         # Σ_c clamp(S2−2·cross_raw+sumT2, 0)
        flat: torch.Tensor | None = None            # featureless-window guard (NCC)

        for img, ch in zip(img_planes, per_channel):
            img32 = img.to(torch.float32)

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

            cross_z_sum = cr if cross_z_sum is None else (cross_z_sum + cr)
            cross_raw_sum = cross_raw if cross_raw_sum is None else (cross_raw_sum + cross_raw)
            s2_sum = s2 if s2_sum is None else (s2_sum + s2)
            sum_t2_sum = ch["sum_t2"] if sum_t2_sum is None else (sum_t2_sum + ch["sum_t2"])

            # NCC per-channel denominator term sqrt(n*varI)*normTz, summed across
            # channels BEFORE dividing (the legacy contract — kept bit-identical).
            denom_c = torch.sqrt(n_var) * ch["norm_tz"]
            ncc_denom_sum = denom_c if ncc_denom_sum is None else (ncc_denom_sum + denom_c)

            # SSD per channel: sum_w (I-T)^2 = S2 - 2·cross_raw + sumT2, summed
            # across channels (§J.2 ssd RGB rule), clamp removes fp negatives.
            ssd_c = (s2 - 2.0 * cross_raw + ch["sum_t2"]).clamp(min=0.0)
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

        if method == "ncc":
            # coeff = cross_z / (sqrt(n·varI)·normTz + eps); flat windows -> 0.
            # This expression and its op-order are unchanged from the original
            # _ncc_map so the NCC regression tests still pass bit-for-bit.
            ncc = cross_z_sum / (ncc_denom_sum + eps)
            ncc = torch.where(flat, torch.zeros_like(ncc), ncc)
            return ncc

        if method == "ccorr":
            # Cosine cross-correlation: cc = cross_raw / (sqrt(S2)·sqrt(sumT2)+eps).
            # Summed per-channel numerators/denominator terms before dividing.
            cc = cross_raw_sum / (torch.sqrt(s2_sum) * torch.sqrt(sum_t2_sum) + eps)
            return cc.clamp(min=0.0, max=1.0)

        if method == "ssd":
            # Normalized squared difference -> flat-friendly similarity score.
            # rmse = sqrt(SSD / (n · channels)); score = clamp(1 - rmse, 0, 1).
            rmse = torch.sqrt(ssd_sum / (n * channels))
            return (1.0 - rmse).clamp(min=0.0, max=1.0)

        raise ValueError(f"unsupported convolution method {method!r}")

    def _box_filter(self, x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
        """Separable windowed-sum box filter (O(kh+kw)), fp32 accumulators.

        Sums over each ``kh x kw`` window via two 1-D ``ones`` convolutions
        (vertical then horizontal). Output shape ``(H-kh+1, W-kw+1)``.
        """
        # Vertical sum: a (kh,1) ones kernel reduces H -> H-kh+1.
        ones_v = torch.ones((1, 1, kh, 1), device=x.device, dtype=torch.float32)
        v = F.conv2d(x, ones_v)
        # Horizontal sum: a (1,kw) ones kernel reduces W -> W-kw+1.
        ones_h = torch.ones((1, 1, 1, kw), device=x.device, dtype=torch.float32)
        s = F.conv2d(v, ones_h)
        return s[0, 0]

    def _cross_spatial(self, img: torch.Tensor, tz: torch.Tensor) -> torch.Tensor:
        """Dense cross-correlation ``sum_w(I * Tz)`` via ``F.conv2d`` (no flip)."""
        weight = tz.to(img.dtype)
        out = F.conv2d(img, weight)
        return out[0, 0]

    def _cross_fft(
        self, img: torch.Tensor, tz: torch.Tensor, sth: int, stw: int
    ) -> torch.Tensor:
        """Cross-correlation via FFT for large templates (§D.2).

        Circular convolution of the image with a *flipped* template equals linear
        cross-correlation once we pad both to a common length >= H+sth-1 (so the
        wrap-around tail does not contaminate the valid region) and crop the
        valid ``(H-sth+1, W-stw+1)`` block.
        """
        ih, iw = img.shape[-2], img.shape[-1]
        # Linear-conv length, padded to a 2-3-5-7 smooth size for FFT speed.
        fh = _next_smooth(ih + sth - 1)
        fw = _next_smooth(iw + stw - 1)

        img2 = img[0, 0]
        tzf = torch.flip(tz[0, 0], dims=(0, 1))  # flip so circular conv == correlation

        img_fft = torch.fft.rfft2(img2, s=(fh, fw))
        tz_fft = torch.fft.rfft2(tzf, s=(fh, fw))
        conv = torch.fft.irfft2(img_fft * tz_fft, s=(fh, fw))

        # The valid cross-correlation output lives at offset (sth-1, stw-1) and
        # spans (ih-sth+1, iw-stw+1).
        out_h = ih - sth + 1
        out_w = iw - stw + 1
        return conv[sth - 1 : sth - 1 + out_h, stw - 1 : stw - 1 + out_w]

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

        # Rung 4: CPU fallback. Move the staged luminance, the staged RGB planes
        # (channel_mode="rgb" reads these in _match_tile), and the template
        # channels to CPU, run there, then restore CUDA state for subsequent jobs.
        # Leaving the RGB planes on the GPU here would crash the CPU run with a
        # device mismatch (CUDA image × CPU template) in _cross_spatial/_cross_fft.
        torch.cuda.empty_cache()
        try:
            self._device = torch.device("cpu")
            self._forced_core_tile = original_forced  # CPU uses its own large tile
            self._lum = original_lum.to("cpu")
            self._rgb = (
                [p.to("cpu") for p in original_rgb] if original_rgb is not None else None
            )
            cpu_tmpl = {
                "channels": [c.to("cpu") for c in tmpl["channels"]],
                "th": tmpl["th"],
                "tw": tmpl["tw"],
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

        return self._feature_matcher.match(
            template,
            self._lum_u8,
            params,
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
