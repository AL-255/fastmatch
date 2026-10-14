"""Reproducible altered-copy recall / false-positive comparison on CPU.

Run: python scripts/fuzzy_eval.py [--seed 12]
The scene includes blur, JPEG compression, 50% occlusion, recolouring, and an
unrelated motif. This tests instance retrieval, not semantic recognition.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmatch.engine import Matcher
from fastmatch.types import MatchParams


def motif(rng, h=180, w=160):
    field = rng.integers(40, 256, (h // 3, w // 3, 3), dtype=np.uint8)
    result = np.full((h, w, 3), 25, np.uint8)
    result[:h // 3 * 3, :w // 3 * 3] = field.repeat(3, 0).repeat(3, 1)
    result[:h // 6, :w // 2] = (245, 30, 30)
    result[:h // 2, :w // 10] = (245, 30, 30)
    result[int(h * .3):int(h * .3) + h // 8,
           int(w * .55):int(w * .55) + h // 8] = 255
    return result


def build_scene(seed=12):
    rng = np.random.default_rng(seed)
    template = motif(rng)
    h, w = template.shape[:2]
    image = rng.integers(96, 136, (460, 1180, 3), dtype=np.uint8)
    source = (20, 20, w, h)
    image[20:20 + h, 20:20 + w] = template
    blur = cv2.GaussianBlur(template, (5, 5), .9)
    buffer = io.BytesIO()
    Image.fromarray(template).save(buffer, format="JPEG", quality=20)
    buffer.seek(0)
    compressed = np.asarray(Image.open(buffer))
    occluded = template.copy()
    occluded[:h // 2] = 110
    copies = {"blur": blur, "jpeg": compressed, "50% occlusion": occluded,
              "recolour": template[:, :, ::-1].copy()}
    truth = {}
    for number, (name, copy) in enumerate(copies.items()):
        x, y = 220 + number * 185, 240
        image[y:y + h, x:x + w] = copy
        truth[name] = (x, y, w, h)
    # Shares general texture and landmarks, but has unrelated local descriptors.
    image[240:240 + h, 970:970 + w] = motif(rng)
    return image, template, source, truth


def iou(a, b):
    intersection = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def evaluate(seed=12):
    image, template, source, truth = build_scene(seed)
    matcher = Matcher(device="cpu")
    matcher.set_image(image)
    report = {}
    for method in ("features", "fuzzy"):
        start = time.perf_counter()
        results = matcher.match(template, MatchParams(method=method, feature_min_inliers=8,
                                threshold_floor=.5), exclude_box=source)
        boxes = [(r.x, r.y, r.w, r.h) for r in results]
        found = {name: max((iou(box, target) for box in boxes), default=0) >= .8
                 for name, target in truth.items()}
        false_positives = sum(not any(iou(box, target) >= .8 for target in truth.values()) for box in boxes)
        report[method] = dict(found=found, false_positives=false_positives,
                              seconds=time.perf_counter() - start)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()
    for method, result in evaluate(args.seed).items():
        print(f"{method}: recall {sum(result['found'].values())}/4, "
              f"false positives {result['false_positives']}, {result['seconds']:.2f}s")
        for name, found in result["found"].items():
            print(f"  {name}: {'found' if found else 'MISSED'}")
