"""Comprehensive Feature-matching benchmark suite (the self-improvement objective).

Measures method="features" across several scenarios the single test.json benchmark
does not cover, so we can see where it is genuinely poor and track improvement:

  rotscale   - feature-rich motif at a rotation x scale grid (generality)
  lowtex     - low-texture upright copies (ORB-starved content)
  small      - small templates (48 px)
  manycopies - many identical upright copies (recall completeness)
  distractor - true copies among SIMILAR-but-different motifs (precision)
  perspective- perspective-warped copies (the README promises this)

Each scenario reports recall and false positives; rotscale/perspective also runtime.
Run: python scripts/feature_bench.py [--scenario name] [--detector orb]
Prints a compact summary; exits 0. Deterministic (fixed seeds).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from fastmatch.engine import Matcher
from fastmatch.types import MatchParams


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def textured_motif(rng, n=130):
    m = np.full((n, n, 3), 25, np.uint8)
    bs = 4
    cy, cx = n // bs, n // bs
    field = rng.integers(40, 256, (cy, cx, 3)).astype(np.uint8)
    m[: cy * bs, : cx * bs] = np.repeat(np.repeat(field, bs, 0), bs, 1)
    m[: n // 6, : n // 2] = (245, 30, 30)
    m[: n // 2, : n // 10] = (245, 30, 30)
    s = n // 7
    m[int(n * .3):int(n * .3) + s, int(n * .55):int(n * .55) + s] = (255, 255, 255)
    return m


def bg(h, w, rng):
    base = rng.integers(96, 136, (h, w, 3)).astype(np.uint8)
    yy = np.linspace(0, 9 * np.pi, h)[:, None]
    xx = np.linspace(0, 9 * np.pi, w)[None, :]
    g = (7 * np.sin(yy) * np.cos(xx)).astype(np.int16)
    return np.clip(base.astype(np.int16) + g[:, :, None], 0, 255).astype(np.uint8)


def stamp_affine(canvas, motif, cx, cy, M2x3, out):
    warp = cv2.warpAffine(motif, M2x3, (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(np.full(motif.shape[:2], 255, np.uint8), M2x3, (out, out),
                          flags=cv2.INTER_NEAREST) > 0
    x0, y0 = int(cx - out / 2), int(cy - out / 2)
    canvas[y0:y0 + out, x0:x0 + out][mask] = warp[mask]
    ys, xs = np.nonzero(mask)
    return (x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min(), ys.max() - ys.min())


def stamp_persp(canvas, motif, cx, cy, strength, out, rng):
    mh, mw = motif.shape[:2]
    src = np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]])
    j = strength * mw
    dst = np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]]) + rng.uniform(-j, j, (4, 2)).astype(np.float32)
    dst -= dst.min(0)
    H = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(motif, H, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpPerspective(np.full((mh, mw), 255, np.uint8), H, (out, out), flags=cv2.INTER_NEAREST) > 0
    x0, y0 = int(cx - out / 2), int(cy - out / 2)
    canvas[y0:y0 + out, x0:x0 + out][mask] = warp[mask]
    ys, xs = np.nonzero(mask)
    return (x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min(), ys.max() - ys.min())


def _run(canvas, src_box, **pk):
    m = Matcher(device="cpu")
    m.set_image(canvas)
    tmpl = np.ascontiguousarray(canvas[src_box[1]:src_box[1] + src_box[3],
                                       src_box[0]:src_box[0] + src_box[2]])
    p = MatchParams(method="features", threshold_floor=0.0, feature_max_instances=400, **pk)
    t = time.perf_counter()
    res = m.match(tmpl, p, exclude_box=src_box)
    return [(r.x, r.y, r.w, r.h) for r in res], time.perf_counter() - t


def _score(planted, det, tol_scale=1.0, msize=130, iou_hit=0.25):
    used = [False] * len(det)
    rec = 0
    for box in planted:
        bcx, bcy = box[0] + box[2] / 2, box[1] + box[3] / 2
        for j, d in enumerate(det):
            if used[j]:
                continue
            dcx, dcy = d[0] + d[2] / 2, d[1] + d[3] / 2
            if abs(dcx - bcx) <= 0.6 * msize * tol_scale and abs(dcy - bcy) <= 0.6 * msize * tol_scale \
               and iou(box, d) >= iou_hit:
                used[j] = True; rec += 1; break
    return rec


def sc_rotscale(det_name):
    rng = np.random.default_rng(5); motif = textured_motif(rng); n = motif.shape[0]
    angles = list(range(0, 360, 45)); scales = [0.7, 1.0, 1.4]; CELL = 300; ncols = 6
    combos = [(a, s) for s in scales for a in angles]
    nrows = (len(combos) + ncols - 1) // ncols + 1
    canvas = bg(nrows * CELL, ncols * CELL, rng)
    M = cv2.getRotationMatrix2D((n / 2, n / 2), 0, 1.0); M[0, 2] += CELL // 2 - n / 2; M[1, 2] += CELL // 2 - n / 2
    src = stamp_affine(canvas, motif, CELL // 2, CELL // 2, M, CELL)
    planted = []
    for i, (a, s) in enumerate(combos):
        r, c = divmod(i, ncols); cx, cy = c * CELL + CELL // 2, (r + 1) * CELL + CELL // 2
        diag = int(np.hypot(n, n) * max(1, s)) + 4
        Mt = cv2.getRotationMatrix2D((n / 2, n / 2), a, s); Mt[0, 2] += diag / 2 - n / 2; Mt[1, 2] += diag / 2 - n / 2
        planted.append(stamp_affine(canvas, motif, cx, cy, Mt, diag))
    det, dt = _run(canvas, src, enable_rotation=True, enable_flipping=True, feature_min_inliers=8, feature_detector=det_name)
    rec = _score(planted, det, tol_scale=1.4, msize=n)
    fp = len(det) - rec
    return dict(scenario="rotscale", recall=rec, total=len(planted), fp=max(0, fp), sec=round(dt, 1))


def sc_lowtex(det_name):
    rng = np.random.default_rng(7); n = 120
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.sqrt((xx - n * .4) ** 2 + (yy - n * .55) ** 2)
    base = np.clip(200 - .9 * r, 40, 230)
    motif = np.stack([base * .6, base * .8, base], -1)
    cv2.circle(motif, (int(n * .65), int(n * .4)), int(n * .18), (230, 180, 120), -1)
    motif = np.clip(cv2.GaussianBlur(motif, (0, 0), 3), 0, 255).astype(np.uint8)
    H, W = 900, 1200; canvas = bg(H, W, rng); pos = []
    for rr in range(3):
        for c in range(4):
            x, y = 40 + c * 290, 40 + rr * 290
            canvas[y:y + n, x:x + n] = motif; pos.append((x, y, n, n))
    det, _ = _run(canvas, pos[0], scales=(1.0,), feature_min_inliers=6, feature_detector=det_name)
    rec = _score(pos[1:], det, msize=n, iou_hit=0.5)
    return dict(scenario="lowtex", recall=rec, total=len(pos) - 1, fp=max(0, len(det) - rec))


def sc_small(det_name):
    rng = np.random.default_rng(9); n = 48; motif = textured_motif(rng, n)
    H, W = 700, 1000; canvas = bg(H, W, rng); pos = []
    for rr in range(3):
        for c in range(5):
            x, y = 30 + c * 190, 30 + rr * 190
            canvas[y:y + n, x:x + n] = motif; pos.append((x, y, n, n))
    det, _ = _run(canvas, pos[0], scales=(1.0,), feature_min_inliers=5, feature_detector=det_name)
    rec = _score(pos[1:], det, msize=n, iou_hit=0.4)
    return dict(scenario="small", recall=rec, total=len(pos) - 1, fp=max(0, len(det) - rec))


def sc_manycopies(det_name):
    rng = np.random.default_rng(11); n = 90; motif = textured_motif(rng, n)
    cols, rows = 8, 5; H, W = 40 + rows * 150, 40 + cols * 150
    canvas = bg(H, W, rng); pos = []
    for rr in range(rows):
        for c in range(cols):
            x, y = 30 + c * 150, 30 + rr * 150
            canvas[y:y + n, x:x + n] = motif; pos.append((x, y, n, n))
    det, _ = _run(canvas, pos[0], scales=(1.0,), feature_min_inliers=6, feature_detector=det_name)
    rec = _score(pos[1:], det, msize=n, iou_hit=0.5)
    return dict(scenario="manycopies", recall=rec, total=len(pos) - 1, fp=max(0, len(det) - rec))


def sc_distractor(det_name):
    """True copies among SIMILAR distractor motifs (precision test)."""
    rng = np.random.default_rng(13); n = 120
    motif = textured_motif(rng, n)
    H, W = 900, 1300; canvas = bg(H, W, rng)
    true_pos, distractors = [], []
    k = 0
    for rr in range(3):
        for c in range(4):
            x, y = 40 + c * 320, 40 + rr * 290
            if k % 2 == 0:
                canvas[y:y + n, x:x + n] = motif; true_pos.append((x, y, n, n))
            else:
                d = textured_motif(np.random.default_rng(100 + k), n)  # different motif
                canvas[y:y + n, x:x + n] = d; distractors.append((x, y, n, n))
            k += 1
    src = true_pos[0]
    det, _ = _run(canvas, src, scales=(1.0,), feature_min_inliers=8, feature_detector=det_name)
    rec = _score(true_pos[1:], det, msize=n, iou_hit=0.5)
    # FP = detections matching a distractor or nothing
    fp = sum(1 for d in det if not any(iou(d, t) >= 0.5 for t in true_pos))
    return dict(scenario="distractor", recall=rec, total=len(true_pos) - 1, fp=fp,
                note=f"{len(distractors)} distractors")


def sc_perspective(det_name):
    rng = np.random.default_rng(17); motif = textured_motif(rng, 130); n = 130
    CELL = 320; ncols = 5
    strengths = [0.05, 0.12, 0.20]; combos = [(s, i) for s in strengths for i in range(4)]
    nrows = (len(combos) + ncols - 1) // ncols + 1
    canvas = bg(nrows * CELL, ncols * CELL, rng)
    M = cv2.getRotationMatrix2D((n / 2, n / 2), 0, 1.0); M[0, 2] += CELL // 2 - n / 2; M[1, 2] += CELL // 2 - n / 2
    src = stamp_affine(canvas, motif, CELL // 2, CELL // 2, M, CELL)
    planted = []
    for i, (strength, _rep) in enumerate(combos):
        r, c = divmod(i, ncols); cx, cy = c * CELL + CELL // 2, (r + 1) * CELL + CELL // 2
        planted.append((stamp_persp(canvas, motif, cx, cy, strength, int(n * 1.6), np.random.default_rng(200 + i)), strength))
    det, dt = _run(canvas, src, enable_rotation=True, enable_flipping=True, feature_min_inliers=8, feature_detector=det_name)
    by_strength = {s: [0, 0] for s in strengths}
    boxes = [p[0] for p in planted]
    rec_total = _score(boxes, det, tol_scale=1.6, msize=n)
    # per-strength recall
    for (box, s) in planted:
        r = _score([box], det, tol_scale=1.6, msize=n)
        by_strength[s][0] += r; by_strength[s][1] += 1
    fp = max(0, len(det) - rec_total)
    return dict(scenario="perspective", recall=rec_total, total=len(planted), fp=fp, sec=round(dt, 1),
                note=" ".join(f"j{int(s*100)}:{by_strength[s][0]}/{by_strength[s][1]}" for s in strengths))


SCENARIOS = {f.__name__[3:]: f for f in
             (sc_rotscale, sc_lowtex, sc_small, sc_manycopies, sc_distractor, sc_perspective)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--detector", default="orb")
    args = ap.parse_args(argv)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    print(f"feature_bench detector={args.detector}")
    rows = []
    for nm in names:
        r = SCENARIOS[nm](args.detector)
        rows.append(r)
        pct = 100 * r["recall"] / max(1, r["total"])
        extra = f"  {r.get('sec','')}s" if "sec" in r else ""
        note = f"  [{r['note']}]" if "note" in r else ""
        print(f"  {r['scenario']:12s} recall {r['recall']:3d}/{r['total']:<3d} ({pct:3.0f}%)  fp={r['fp']:<3d}{extra}{note}")
    avg = np.mean([100 * r["recall"] / max(1, r["total"]) for r in rows])
    tfp = sum(r["fp"] for r in rows)
    print(f"  {'AVG':12s} recall {avg:.0f}%   total_fp={tfp}")


if __name__ == "__main__":
    main()
