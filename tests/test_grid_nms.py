"""Grid-indexed NMS must return exactly what the plain greedy NMS returns."""
import time

import numpy as np
import pytest
import torch

from fastmatch.engine import Matcher, _greedy_nms, _grid_nms
from fastmatch.types import Match, MatchParams


def _random_boxes(rng, n, span, sizes):
    x = rng.integers(0, span, n)
    y = rng.integers(0, span, n)
    w = rng.choice(sizes, n)
    h = rng.choice(sizes, n)
    boxes = torch.tensor(np.stack([x, y, x + w, y + h], 1), dtype=torch.float32)
    scores = torch.tensor(rng.random(n), dtype=torch.float32)
    return boxes, scores


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("iou", [0.0, 0.3, 0.7])
def test_grid_nms_matches_greedy(seed, iou):
    rng = np.random.default_rng(seed)
    # Mixed box sizes (as a multi-scale search produces) and dense overlap.
    boxes, scores = _random_boxes(rng, 1500, 400, [8, 13, 40, 97])
    assert _grid_nms(boxes, scores, iou).tolist() == _greedy_nms(boxes, scores, iou).tolist()


def test_grid_nms_handles_huge_repeated_grids_quickly():
    # 100k non-overlapping copies of one cell, like a repeated chip structure.
    side, cell = 317, 60
    yy, xx = np.mgrid[0:side, 0:side]
    x = (xx.ravel() * cell).astype(np.float32)
    y = (yy.ravel() * cell).astype(np.float32)
    boxes = torch.tensor(np.stack([x, y, x + cell, y + cell], 1))
    scores = torch.rand(boxes.shape[0])
    t0 = time.time()
    keep = _grid_nms(boxes, scores, 0.3)
    assert keep.numel() == side * side
    assert time.time() - t0 < 20  # the quadratic loop took over ten minutes


def test_orientation_nms_matches_pairwise_reference():
    rng = np.random.default_rng(7)
    pooled = [
        Match(x=int(rng.integers(0, 300)), y=int(rng.integers(0, 300)),
              w=int(w), h=int(h), score=float(rng.random()), scale=1.0)
        for w, h in rng.choice([10, 16, 33], (800, 2))
    ]
    params = MatchParams(nms_iou=0.3, max_results=10_000)
    ordered = sorted(pooled, key=lambda m: m.score, reverse=True)

    def iou(a, b):
        iw = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
        ih = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        return inter / (a.w * a.h + b.w * b.h - inter)

    expected = []
    for m in ordered:
        if all(iou(m, k) <= 0.3 for k in expected):
            expected.append(m)
    assert Matcher._finalize_orientations(pooled, params) == expected
    capped = Matcher._finalize_orientations(pooled, MatchParams(nms_iou=0.3, max_results=5))
    assert capped == expected[:5]
