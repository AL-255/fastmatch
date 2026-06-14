"""Rotation x scale stress benchmark for the Feature-matching method.

The saved test.json benchmark is all scale=1.0 / D4 orientations, so it does NOT
exercise feature matching's actual purpose: arbitrary rotated/scaled instances.
This plants a feature-rich motif at a grid of known (angle, scale) transforms on a
textured background, runs method="features" (rotation+flip enabled), and reports
recall per angle, per scale, overall, plus false positives — so we can see where
feature matching is poor.

Usage: python scripts/feature_eval.py [--detector orb|akaze|sift] [--seed N]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from fastmatch.engine import Matcher
from fastmatch.types import MatchParams

ANGLES = list(range(0, 360, 30))      # 0,30,...,330
SCALES = [0.7, 1.0, 1.4]
CELL = 300                            # px grid cell (fits motif*1.4 rotated)
MOTIF = 140


def _motif(rng):
    """Feature-rich asymmetric colour motif (dense blocks + landmarks)."""
    h = w = MOTIF
    m = np.full((h, w, 3), 25, np.uint8)
    bs = 4
    cy, cx = h // bs, w // bs
    field = rng.integers(40, 256, (cy, cx, 3)).astype(np.uint8)
    m[: cy * bs, : cx * bs] = np.repeat(np.repeat(field, bs, 0), bs, 1)
    m[: h // 6, : w // 2] = (245, 30, 30)      # red top-left bar (asymmetry)
    m[: h // 2, : w // 10] = (245, 30, 30)
    s = h // 7
    m[int(h * 0.30): int(h * 0.30) + s, int(w * 0.55): int(w * 0.55) + s] = (255, 255, 255)
    return m


def _bg(h, w, rng):
    base = rng.integers(96, 136, (h, w, 3)).astype(np.uint8)
    yy = np.linspace(0, 9 * np.pi, h)[:, None]
    xx = np.linspace(0, 9 * np.pi, w)[None, :]
    g = (7 * np.sin(yy) * np.cos(xx)).astype(np.int16)
    return np.clip(base.astype(np.int16) + g[:, :, None], 0, 255).astype(np.uint8)


def _stamp(canvas, motif, cx, cy, angle, scale):
    """Warp motif by (angle, scale) and paste centered at (cx, cy). Return AABB."""
    mh, mw = motif.shape[:2]
    diag = int(np.ceil(np.hypot(mh, mw) * max(1.0, scale))) + 4
    M = cv2.getRotationMatrix2D((mw / 2, mh / 2), angle, scale)
    M[0, 2] += diag / 2 - mw / 2
    M[1, 2] += diag / 2 - mh / 2
    warp = cv2.warpAffine(motif, M, (diag, diag), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(np.full((mh, mw), 255, np.uint8), M, (diag, diag),
                          flags=cv2.INTER_NEAREST) > 0
    x0, y0 = int(cx - diag / 2), int(cy - diag / 2)
    region = canvas[y0:y0 + diag, x0:x0 + diag]
    region[mask] = warp[mask]
    ys, xs = np.nonzero(mask)
    return (x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min(), ys.max() - ys.min())


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="orb")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    motif = _motif(rng)
    combos = [(a, s) for s in SCALES for a in ANGLES]
    ncols = 6
    nrows = (len(combos) + ncols - 1) // ncols + 1  # +1 row for the source
    H, W = nrows * CELL, ncols * CELL
    canvas = _bg(H, W, rng)

    # Source (upright, scale 1) top-left cell -> template.
    sx, sy = CELL // 2, CELL // 2
    src_box = _stamp(canvas, motif, sx, sy, 0, 1.0)
    planted = []  # (box, angle, scale)
    for i, (a, s) in enumerate(combos):
        r, c = divmod(i, ncols)
        cx, cy = c * CELL + CELL // 2, (r + 1) * CELL + CELL // 2
        planted.append((_stamp(canvas, motif, cx, cy, a, s), a, s))

    m = Matcher(device="cpu")
    m.set_image(canvas)
    template = np.ascontiguousarray(canvas[src_box[1]:src_box[1] + src_box[3],
                                           src_box[0]:src_box[0] + src_box[2]])
    params = MatchParams(method="features", channel_mode="luminance",
                         enable_rotation=True, enable_flipping=True,
                         threshold_floor=0.0, feature_min_inliers=8,
                         feature_max_instances=300, feature_detector=args.detector)
    t = time.perf_counter()
    res = m.match(template, params, exclude_box=src_box)
    dt = time.perf_counter() - t
    det = [(r.x, r.y, r.w, r.h) for r in res]

    # Recall: a planted copy is found if a detection's centre is near it (rotated
    # AABBs are loose, so use centre distance <= 0.5*motif + IoU>=0.25).
    used = [False] * len(det)
    found = {}
    for (box, a, s) in planted:
        bcx, bcy = box[0] + box[2] / 2, box[1] + box[3] / 2
        hit = False
        for j, d in enumerate(det):
            if used[j]:
                continue
            dcx, dcy = d[0] + d[2] / 2, d[1] + d[3] / 2
            if abs(dcx - bcx) <= 0.5 * MOTIF * s and abs(dcy - bcy) <= 0.5 * MOTIF * s \
               and _iou(box, d) >= 0.25:
                used[j] = True; hit = True; break
        found[(a, s)] = hit
    fp = sum(1 for u in used if not u)

    print(f"detector={args.detector}  planted={len(planted)}  detections={len(det)}  "
          f"false_pos={fp}  ({dt:.1f}s)")
    overall = sum(found.values())
    print(f"OVERALL recall: {overall}/{len(planted)} ({100*overall/len(planted):.0f}%)")
    print("\nrecall by SCALE:")
    for s in SCALES:
        v = [found[(a, s)] for a in ANGLES]
        print(f"  scale {s}: {sum(v)}/{len(v)}  " + "".join("#" if x else "." for x in v))
    print("\nrecall by ANGLE (across scales):")
    for a in ANGLES:
        v = [found[(a, s)] for s in SCALES]
        print(f"  {a:3d}°: {sum(v)}/{len(v)}  " + "".join("#" if x else "." for x in v))
    print("\nmiss grid (rows=scale, cols=angle 0..330):")
    print("        " + " ".join(f"{a:>3}" for a in ANGLES))
    for s in SCALES:
        print(f"  {s:>4}: " + "   ".join("#" if found[(a, s)] else "." for a in ANGLES))


if __name__ == "__main__":
    main()
