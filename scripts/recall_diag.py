"""Recall diagnostic: replay each memory entry and measure how many of its
saved ground-truth matches the current engine recovers.

For every entry we crop its selection as the template, run match() with the
entry's saved params, then greedily pair each SAVED match to the closest engine
detection by IoU. A saved match is:
  * recovered@floor  if some engine detection overlaps it with IoU >= IOU_HIT
  * recovered@thr     additionally requires that detection's score >= entry threshold

Missed matches are reported with their best IoU and the engine score at that box,
so we can see *why* (below floor / suppressed / scale out of grid / etc.).

Usage:
    python scripts/recall_diag.py [memory.json] [--device cpu|cuda] [--iou 0.5]
                                  [--entries 0,2] [--no-pyramid]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmatch.engine import Matcher  # noqa: E402
from fastmatch.loader import load_image  # noqa: E402
from fastmatch.memory import load_store  # noqa: E402
from fastmatch.types import Match  # noqa: E402

IOU_HIT = 0.5


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _crop(full: np.ndarray, sel) -> np.ndarray:
    x, y, w, h = sel
    return np.ascontiguousarray(full[y : y + h, x : x + w])


def pair_recall(gt: list[Match], eng: list[Match], iou_hit: float, thr: float):
    """Greedily pair each GT match to the best-IoU engine match (1:1)."""
    eng_boxes = [(m.x, m.y, m.w, m.h, m.score) for m in eng]
    used = [False] * len(eng_boxes)
    rows = []
    for g in gt:
        gb = (g.x, g.y, g.w, g.h)
        best_j, best_iou = -1, 0.0
        for j, eb in enumerate(eng_boxes):
            if used[j]:
                continue
            i = _iou(gb, eb[:4])
            if i > best_iou:
                best_iou, best_j = i, j
        matched = best_j if best_iou >= iou_hit else -1
        if matched >= 0:
            used[matched] = True
            escore = eng_boxes[matched][4]
        else:
            escore = None
        rows.append((g, best_iou, escore))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("memory", nargs="?", default="samples/test.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iou", type=float, default=IOU_HIT)
    ap.add_argument("--entries", default=None, help="comma list of entry indices")
    ap.add_argument("--no-pyramid", action="store_true")
    ap.add_argument("--method", default=None, help="override every entry's method")
    ap.add_argument("--no-orient", action="store_true", help="force rotation/flipping off")
    ap.add_argument("--channel", default=None, help="override channel_mode (luminance|rgb)")
    args = ap.parse_args(argv)

    store = load_store(args.memory)
    doc = load_image(store.source_image)
    full = doc.full
    print(f"image {full.shape[1]}x{full.shape[0]}  entries={len(store.entries)}  device={args.device}")

    eng = Matcher(device=args.device, compute_dtype="float32",
                  use_pyramid=not args.no_pyramid)
    if eng.effective_device.type != args.device:
        print(f"  requested {args.device!r} resolved to {eng.effective_device.type!r}")
    eng.set_image(full)

    want = None
    if args.entries:
        want = {int(x) for x in args.entries.split(",")}

    grand_gt = grand_thr = grand_floor = 0
    for i, e in enumerate(store.entries):
        if want is not None and i not in want:
            continue
        tmpl = _crop(full, e.selection)
        params = dataclasses.replace(e.params, device=args.device)
        if args.method:
            params = dataclasses.replace(params, method=args.method)
        if args.no_orient:
            params = dataclasses.replace(params, enable_rotation=False, enable_flipping=False)
        if args.channel:
            params = dataclasses.replace(params, channel_mode=args.channel)
        t = time.perf_counter()
        try:
            eng_matches = eng.match(tmpl, params, exclude_box=e.selection)
        except ValueError as exc:
            print(f"\n=== entry {i}: method={params.method} REJECTED: {exc}")
            grand_gt += len(e.matches)
            continue
        dt = time.perf_counter() - t
        rows = pair_recall(e.matches, eng_matches, args.iou, params.threshold)
        n_gt = len(e.matches)
        rec_floor = sum(1 for _, iou, _ in rows if iou >= args.iou)
        rec_thr = sum(1 for _, iou, s in rows if iou >= args.iou and s is not None
                      and s >= params.threshold - 1e-6)
        grand_gt += n_gt
        grand_thr += rec_thr
        grand_floor += rec_floor
        print(f"\n=== entry {i}: method={params.method} thr={params.threshold:.3f} "
              f"scales={params.scales} rot/flip={params.enable_rotation}/{params.enable_flipping}")
        print(f"    sel={e.selection}  GT={n_gt}  engine_returned={len(eng_matches)}  ({dt:.1f}s)")
        print(f"    recovered@floor(IoU>={args.iou}): {rec_floor}/{n_gt}   "
              f"recovered@thr: {rec_thr}/{n_gt}")
        misses = [(g, iou, s) for g, iou, s in rows
                  if not (iou >= args.iou and s is not None and s >= params.threshold - 1e-6)]
        if misses:
            print(f"    MISSED {len(misses)} (gt_score, best_iou, engine_score@box):")
            for g, iou, s in misses[:20]:
                ss = f"{s:.3f}" if s is not None else "none"
                print(f"       gt@({g.x},{g.y},{g.w},{g.h}) gtscore={g.score:.3f} "
                      f"scale={g.scale} orient={g.orientation} | bestIoU={iou:.2f} engScore={ss}")
    print(f"\n=== TOTAL: GT={grand_gt}  recovered@floor={grand_floor}  "
          f"recovered@thr={grand_thr}  "
          f"(recall@thr={100.0*grand_thr/max(grand_gt,1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
