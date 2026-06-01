"""FastMatch — browser engine (pure NumPy, runs under Pyodide *and* CPython).

The desktop engine is built on PyTorch/CUDA, which has no WebAssembly build, so
the web port reimplements the convolution matchers (``ncc``/``ssd``/``ccorr``)
in plain NumPy. Same algorithm as the desktop's full-resolution path: FFT-based
cross-correlation for the numerator and an integral image for the windowed
sums/variances, with dihedral-orientation search, multi-scale, greedy IoU-NMS
and source exclusion. Feature matching (OpenCV) is intentionally not ported —
OpenCV is unavailable under Pyodide.

This module is dependency-light (NumPy only) so it loads fast in the browser,
and it is import-clean in CPython so ``tests/test_web_engine.py`` can pin it
against the desktop engine. The single entry point is :func:`match`.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6
# Per-window variance floor (sum of squared deviations < _VAR_FLOOR * n). NCC is
# 0/0 on a near-flat window; without this guard FFT-vs-integral rounding lets the
# ratio spike past 1 and invent matches on textureless regions. Matches the
# desktop engine's _VAR_FLOOR.
_VAR_FLOOR = 1e-3
ORIENTATIONS = ("R0", "R90", "R180", "R270", "MX", "MY", "MXR90", "MYR90")
CONV_METHODS = ("ncc", "ssd", "ccorr")
# BT.601 luminance weights — identical to the desktop engine's convention.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Orientation (dihedral D4) — mirrors fastmatch.types.
# --------------------------------------------------------------------------- #
def active_orientations(enable_rotation: bool, enable_flipping: bool) -> tuple[str, ...]:
    """Active D4 orientations for the two toggles (canonical order, always R0)."""
    active = {"R0"}
    if enable_rotation:
        active |= {"R90", "R180", "R270"}
    if enable_flipping:
        active |= {"MX", "MY"}
    if enable_rotation and enable_flipping:
        active |= {"MXR90", "MYR90"}
    return tuple(o for o in ORIENTATIONS if o in active)


def apply_orientation(arr: np.ndarray, orient: str) -> np.ndarray:
    """Transform ``arr`` (H,W[,C], numpy [row,col]==[y,x]) by orientation ``orient``."""
    if orient == "R0":
        out = arr
    elif orient == "R90":
        out = np.rot90(arr, 1)
    elif orient == "R180":
        out = np.rot90(arr, 2)
    elif orient == "R270":
        out = np.rot90(arr, 3)
    elif orient == "MX":
        out = arr[::-1, :]
    elif orient == "MY":
        out = arr[:, ::-1]
    elif orient == "MXR90":
        out = np.rot90(arr[::-1, :], 1)
    elif orient == "MYR90":
        out = np.rot90(arr[:, ::-1], 1)
    else:
        raise ValueError(f"unknown orientation {orient!r}")
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- #
# Core numeric helpers.
# --------------------------------------------------------------------------- #
def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3|4) uint8/float -> (H,W) float32 luminance in [0,1]."""
    a = np.asarray(rgb)
    if a.ndim == 2:
        g = a.astype(np.float32)
    else:
        g = a[..., :3].astype(np.float32) @ _LUMA
    if a.dtype == np.uint8 or g.max(initial=0.0) > 1.5:
        g = g / 255.0
    return np.ascontiguousarray(g, dtype=np.float32)


def _resize2d(img: np.ndarray, nh: int, nw: int) -> np.ndarray:
    """Bilinear resize of a 2-D float array to (nh, nw). Used to scale templates."""
    h, w = img.shape
    if (nh, nw) == (h, w):
        return img
    ys = np.linspace(0.0, h - 1.0, nh)
    xs = np.linspace(0.0, w - 1.0, nw)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def _box_sum(image: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """Windowed sums over every (kh, kw) window via an integral image (O(1)/window).

    Returns the valid grid ``(H-kh+1, W-kw+1)``. float64 accumulation keeps the
    cumulative sum accurate on large images.
    """
    h, w = image.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    np.cumsum(np.cumsum(image, axis=0, dtype=np.float64), axis=1, out=ii[1:, 1:])
    oh, ow = h - kh + 1, w - kw + 1
    return (
        ii[kh : kh + oh, kw : kw + ow]
        - ii[0:oh, kw : kw + ow]
        - ii[kh : kh + oh, 0:ow]
        + ii[0:oh, 0:ow]
    )


def _next_fast(n: int) -> int:
    """Smallest 2-3-5-7-smooth integer >= n (fast FFT length)."""
    if n <= 1:
        return 1
    best = 2 * n
    p2 = 1
    while p2 < best:
        p23 = p2
        while p23 < best:
            p235 = p23
            while p235 < best:
                v = p235
                while v < n:
                    v *= 7
                best = min(best, v)
                p235 *= 5
            p23 *= 3
        p2 *= 2
    return best


def _xcorr_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid cross-correlation ``sum_w image[y+i,x+j]*kernel[i,j]`` via FFT."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    fh, fw = _next_fast(ih + kh - 1), _next_fast(iw + kw - 1)
    fimg = np.fft.rfft2(image, s=(fh, fw))
    # Correlation == convolution with a flipped kernel.
    fker = np.fft.rfft2(kernel[::-1, ::-1], s=(fh, fw))
    full = np.fft.irfft2(fimg * fker, s=(fh, fw))
    oh, ow = ih - kh + 1, iw - kw + 1
    return full[kh - 1 : kh - 1 + oh, kw - 1 : kw - 1 + ow].astype(np.float32)


def _score_map(image: np.ndarray, template: np.ndarray, method: str) -> np.ndarray:
    """Per-window similarity map in [0,1]-ish for one (image, template), method."""
    th, tw = template.shape
    n = float(th * tw)
    win_sum = _box_sum(image, th, tw)
    win_sum2 = _box_sum(image * image, th, tw)
    if method == "ncc":
        t_zero = template - float(template.mean())
        t_ss = float((t_zero * t_zero).sum())
        num = _xcorr_valid(image, t_zero)  # sum (I_w - meanI_w)(T - meanT)
        var = np.clip(win_sum2 - win_sum * win_sum / n, 0.0, None)
        denom = np.sqrt(var * t_ss)
        ncc = num / (denom + _EPS)
        # Flat-window guard: NCC is undefined (0/0) where the window has no
        # texture; zero it so rounding can't push the ratio outside [-1, 1].
        ncc = np.where(var < _VAR_FLOOR * n, 0.0, ncc)
        return np.clip(ncc, -1.0, 1.0)
    cross = _xcorr_valid(image, template)  # sum I_w * T
    t_raw = float((template * template).sum())
    if method == "ccorr":
        denom = np.sqrt(np.clip(win_sum2, 0.0, None) * t_raw)
        return np.clip(cross / (denom + _EPS), 0.0, 1.0)
    # ssd: turn the sum of squared differences into a [0,1] similarity.
    ssd = np.clip(win_sum2 - 2.0 * cross + t_raw, 0.0, None)
    return np.clip(1.0 - ssd / (win_sum2 + t_raw + _EPS), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# NMS / IoU.
# --------------------------------------------------------------------------- #
def _greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy IoU NMS; returns kept row indices (descending score)."""
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.clip(areas[i] + areas[rest] - inter, _EPS, None)
        order = rest[iou <= iou_thr]
    return keep


def _iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def match(
    image,
    sel_x: int,
    sel_y: int,
    sel_w: int,
    sel_h: int,
    *,
    method: str = "ncc",
    channel_mode: str = "luminance",
    threshold: float = 0.8,
    scales=(1.0,),
    enable_rotation: bool = False,
    enable_flipping: bool = False,
    max_results: int = 200,
    nms_iou: float = 0.3,
    exclude_iou: float = 0.3,
):
    """Find instances of the selected region in ``image``.

    Args:
        image: ``(H,W,3|4)`` uint8 RGB(A) or ``(H,W)`` array (e.g. canvas pixels).
        sel_x/sel_y/sel_w/sel_h: the source selection rectangle (image px).
        method: ``"ncc"`` | ``"ssd"`` | ``"ccorr"``.
        channel_mode: ``"luminance"`` (default) or ``"rgb"`` (rgb averages the
            three per-channel score maps — a light approximation of the desktop's
            summed-numerator path, enough for the browser demo).
        threshold: keep matches with score >= this (in [0,1]).
        scales: template scale grid (e.g. ``(0.9, 1.0, 1.1)``).
        enable_rotation/enable_flipping: dihedral-orientation search.
        max_results, nms_iou, exclude_iou: as the desktop engine.

    Returns:
        ``list[dict]`` ``{x,y,w,h,score,scale,orientation}`` in image px, sorted
        by score desc, source region excluded, capped at ``max_results``.
    """
    method = method if method in CONV_METHODS else "ncc"
    arr = np.asarray(image)
    H, W = arr.shape[:2]
    sel_x, sel_y = int(sel_x), int(sel_y)
    sel_w, sel_h = int(sel_w), int(sel_h)
    if sel_w < 2 or sel_h < 2 or sel_x < 0 or sel_y < 0:
        return []
    sel_x2, sel_y2 = min(W, sel_x + sel_w), min(H, sel_y + sel_h)

    use_rgb = channel_mode == "rgb" and arr.ndim == 3 and arr.shape[2] >= 3
    if use_rgb:
        planes = [np.ascontiguousarray(arr[..., c].astype(np.float32) / 255.0) for c in range(3)]
    else:
        planes = [_to_gray(arr)]

    tmpls0 = [p[sel_y:sel_y2, sel_x:sel_x2] for p in planes]
    base_h, base_w = tmpls0[0].shape
    if base_h < 2 or base_w < 2:
        return []

    orients = active_orientations(bool(enable_rotation), bool(enable_flipping))
    scales = tuple(float(s) for s in scales) or (1.0,)
    cap_pre = max(4 * max_results, 4000)  # bound candidates fed to NMS

    boxes_all: list[tuple[float, float, float, float]] = []
    score_all: list[float] = []
    scale_all: list[float] = []
    orient_all: list[str] = []

    for orient in orients:
        tmpls_o = [apply_orientation(t, orient) for t in tmpls0]
        oth, otw = tmpls_o[0].shape
        for s in scales:
            sh, sw = max(2, round(oth * s)), max(2, round(otw * s))
            if sh > H or sw > W:
                continue
            tmpls_s = [_resize2d(t, sh, sw) for t in tmpls_o] if (sh, sw) != (oth, otw) else tmpls_o
            smap = None
            for img_p, tmpl_p in zip(planes, tmpls_s):
                m = _score_map(img_p, tmpl_p, method)
                smap = m if smap is None else smap + m
            smap = (smap / len(planes)).astype(np.float32)
            smap = np.nan_to_num(smap, nan=-1.0, posinf=-1.0, neginf=-1.0)

            ys, xs = np.nonzero(smap >= threshold)
            if ys.size == 0:
                continue
            sc = smap[ys, xs]
            if sc.size > cap_pre:  # keep the strongest candidates only
                top = np.argpartition(sc, -cap_pre)[-cap_pre:]
                ys, xs, sc = ys[top], xs[top], sc[top]
            for x, y, v in zip(xs.tolist(), ys.tolist(), sc.tolist()):
                boxes_all.append((x, y, x + sw, y + sh))
                score_all.append(float(v))
                scale_all.append(float(s))
                orient_all.append(orient)

    if not boxes_all:
        return []

    boxes = np.asarray(boxes_all, dtype=np.float32)
    scores = np.asarray(score_all, dtype=np.float32)
    keep = _greedy_nms(boxes, scores, float(nms_iou))

    src = (sel_x, sel_y, sel_w, sel_h)
    results = []
    for i in keep:
        bx, by, bx2, by2 = boxes[i]
        w, h = int(round(bx2 - bx)), int(round(by2 - by))
        cand = (int(round(bx)), int(round(by)), w, h)
        cx, cy = cand[0] + w / 2.0, cand[1] + h / 2.0
        in_src = sel_x <= cx < sel_x + sel_w and sel_y <= cy < sel_y + sel_h
        if in_src or _iou_xywh(cand, src) > float(exclude_iou):
            continue
        results.append(
            {
                "x": cand[0], "y": cand[1], "w": w, "h": h,
                "score": float(scores[i]),
                "scale": scale_all[i],
                "orientation": orient_all[i],
            }
        )
        if len(results) >= max_results:
            break
    return results
