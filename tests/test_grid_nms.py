"""Grid-indexed NMS must return exactly what the plain greedy NMS returns."""
import time

import numpy as np
import pytest
import torch

from fastmatch.engine import (
    Matcher,
    _greedy_nms,
    _greedy_order,
    _grid_greedy_order,
    _grid_nms,
    _parallel_greedy_order,
)
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


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("iou", [0.0, 0.3, 0.5])
def test_parallel_rounds_match_sequential_loop(seed, iou):
    rng = np.random.default_rng(100 + seed)
    n = 4000
    x = rng.integers(0, 900, n).astype(np.float64)
    y = rng.integers(0, 900, n).astype(np.float64)
    w = rng.choice([6.0, 11.0, 28.0, 60.0], n)
    h = rng.choice([6.0, 22.0, 28.0], n)
    order = np.argsort(-rng.random(n), kind="stable")
    fast = _parallel_greedy_order(x, y, x + w, y + h, order, iou)
    assert fast is not None
    ref = _grid_greedy_order(x.tolist(), y.tolist(), (x + w).tolist(), (y + h).tolist(),
                             order.tolist(), iou)
    assert fast.tolist() == ref


def test_long_suppression_chain_falls_back_to_the_exact_loop():
    # Each box overlaps the next and scores rise along the row: greedy decisions
    # alternate down a 2000-long chain, beyond the parallel round budget.
    n = 2000
    x = np.arange(n, dtype=np.float64) * 4.0
    y = np.zeros(n)
    order = np.arange(n)[::-1].copy()
    assert _parallel_greedy_order(x, y, x + 10.0, y + 10.0, order, 0.3) is None
    got = _greedy_order(x, y, x + 10.0, y + 10.0, order, 0.3)
    ref = _grid_greedy_order(x.tolist(), y.tolist(), (x + 10).tolist(), (y + 10).tolist(),
                             order.tolist(), 0.3)
    assert got == ref


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
