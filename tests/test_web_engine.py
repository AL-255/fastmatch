"""Tests for the browser (NumPy) engine in ``docs/fastmatch_web.py``.

The web engine is a pure-NumPy reimplementation of the desktop convolution
matchers (torch has no WASM build). These tests pin its correctness in CPython —
the same code path Pyodide runs — and check recall parity against the desktop
``Matcher`` on a shared synthetic scene so the port is trustworthy, not just
plausible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# docs/ is a static-site folder, not a package — load the module by path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs"))
import fastmatch_web as fw  # noqa: E402


def _scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    h, w = 600, 800
    img = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
    motif = np.zeros((40, 32, 3), np.uint8)
    motif[:20, :16] = (240, 30, 30)
    motif[:20, 16:] = (30, 220, 30)
    motif[20:, :22] = (40, 40, 240)
    motif[20:, 22:] = (250, 250, 250)
    motif = np.clip(motif.astype(int) + rng.integers(-3, 4, motif.shape), 0, 255).astype(np.uint8)
    positions = [(50, 40), (300, 120), (600, 300), (150, 400), (450, 500)]
    for (x, y) in positions:
        img[y : y + 40, x : x + 32] = motif
    return img, positions, (40, 32)


def _recovered(results, truth, src_idx, tol=4):
    """Set of truth indices recovered by a center within ``tol`` px (source skipped)."""
    rec = set()
    for i, (tx, ty) in enumerate(truth):
        if i == src_idx:
            continue
        for r in results:
            if abs(r["x"] - tx) <= tol and abs(r["y"] - ty) <= tol:
                rec.add(i)
                break
    return rec


# --------------------------------------------------------------- numeric units
def test_box_sum_matches_naive():
    rng = np.random.default_rng(1)
    img = rng.standard_normal((40, 50)).astype(np.float32)
    kh, kw = 7, 9
    got = fw._box_sum(img, kh, kw)
    oh, ow = 40 - kh + 1, 50 - kw + 1
    assert got.shape == (oh, ow)
    # Spot-check a few windows against a direct sum.
    for (y, x) in [(0, 0), (10, 20), (oh - 1, ow - 1)]:
        assert np.isclose(got[y, x], img[y : y + kh, x : x + kw].sum(), atol=1e-3)


def test_xcorr_valid_matches_direct():
    rng = np.random.default_rng(2)
    img = rng.standard_normal((30, 36)).astype(np.float32)
    ker = rng.standard_normal((5, 7)).astype(np.float32)
    got = fw._xcorr_valid(img, ker)
    oh, ow = 30 - 5 + 1, 36 - 7 + 1
    assert got.shape == (oh, ow)
    for (y, x) in [(0, 0), (12, 18), (oh - 1, ow - 1)]:
        direct = float((img[y : y + 5, x : x + 7] * ker).sum())
        assert np.isclose(got[y, x], direct, atol=1e-3)


def test_resize_identity_and_shape():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((20, 24)).astype(np.float32)
    assert fw._resize2d(a, 20, 24) is a  # identity short-circuit
    assert fw._resize2d(a, 10, 30).shape == (10, 30)


# ------------------------------------------------------------------- behaviour
@pytest.mark.parametrize("method", ["ncc", "ssd", "ccorr"])
def test_recovers_all_other_instances(method):
    img, pos, (mh, mw) = _scene()
    sx, sy = pos[0]
    res = fw.match(img, sx, sy, mw, mh, method=method, threshold=0.85, max_results=50)
    rec = _recovered(res, pos, src_idx=0)
    assert rec == {1, 2, 3, 4}, f"{method}: recovered {rec}"
    assert all(0.0 <= r["score"] <= 1.0001 for r in res)


def test_source_region_is_excluded():
    img, pos, (mh, mw) = _scene()
    sx, sy = pos[0]
    res = fw.match(img, sx, sy, mw, mh, method="ncc", threshold=0.85)
    for r in res:
        assert fw._iou_xywh((r["x"], r["y"], r["w"], r["h"]), (sx, sy, mw, mh)) <= 0.3


def test_threshold_filters_results():
    img, pos, (mh, mw) = _scene()
    sx, sy = pos[0]
    hi = fw.match(img, sx, sy, mw, mh, method="ncc", threshold=0.99)
    lo = fw.match(img, sx, sy, mw, mh, method="ncc", threshold=0.5)
    assert len(lo) >= len(hi)


def test_orientation_tag_present_with_rotation():
    img, pos, (mh, mw) = _scene()
    sx, sy = pos[0]
    res = fw.match(img, sx, sy, mw, mh, method="ncc", threshold=0.85, enable_rotation=True)
    assert res and all(r["orientation"] in fw.ORIENTATIONS for r in res)


def test_tiny_selection_returns_empty():
    img, _, _ = _scene()
    assert fw.match(img, 10, 10, 1, 1) == []


# ------------------------------------------- parallel decomposition (workers)
def test_tasks_for_enumeration():
    assert fw.tasks_for(False, False, (1.0,)) == [("R0", 1.0)]
    assert fw.tasks_for(True, False, (1.0,)) == [(o, 1.0) for o in ("R0", "R90", "R180", "R270")]
    assert len(fw.tasks_for(True, True, (0.9, 1.0, 1.1))) == 8 * 3  # all orients x 3 scales


@pytest.mark.parametrize(
    "method,rot,flip,scales",
    [("ncc", False, False, (1.0,)),
     ("ssd", False, False, (1.0,)),
     ("ncc", False, False, (0.9, 1.0, 1.1)),
     ("ncc", True, True, (1.0,)),
     ("ncc", True, True, (0.8, 0.9, 1.0, 1.1, 1.25))],
)
def test_parallel_decomposition_equals_match(method, rot, flip, scales):
    """Per-task candidates_for + finalize (the worker path) == single-call match().

    This is the contract the browser's worker pool relies on: splitting the
    (orientation x scale) tasks across workers and merging must not change the
    result versus running match() in one shot.
    """
    img, pos, (mh, mw) = _scene(seed=3)
    sx, sy = pos[0]
    sel = (sx, sy, mw, mh)
    planes = fw.prepare_planes(img, "luminance")
    cap = max(4 * 200, 4000)
    cands = []
    for orient, s in fw.tasks_for(rot, flip, scales):
        cands += fw.candidates_for(planes, sel, method, 0.85, s, orient, cap)
    parallel = fw.finalize(cands, sel, 0.3, 0.3, 200)

    ref = fw.match(img, sx, sy, mw, mh, method=method, threshold=0.85,
                   scales=scales, enable_rotation=rot, enable_flipping=flip)

    def keyset(ms):
        return sorted((m["x"], m["y"], m["w"], m["h"]) for m in ms)

    assert keyset(parallel) == keyset(ref)


# ------------------------------------------------------- parity vs desktop torch
def test_recall_parity_with_desktop_engine():
    """The web NCC recovers the same instances the desktop torch engine does."""
    torch = pytest.importorskip("torch", reason="desktop parity needs torch")
    from fastmatch.engine import Matcher
    from fastmatch.types import MatchParams

    img, pos, (mh, mw) = _scene(seed=7)
    sx, sy = pos[0]
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])

    m = Matcher(device="cpu", use_pyramid=False)
    m.set_image(img)
    desk = m.match(
        template,
        MatchParams(method="ncc", threshold=0.85, threshold_floor=0.85, scales=(1.0,)),
        exclude_box=(sx, sy, mw, mh),
    )
    desk_rec = _recovered(
        [{"x": d.x, "y": d.y, "w": d.w, "h": d.h} for d in desk], pos, src_idx=0
    )
    web = fw.match(img, sx, sy, mw, mh, method="ncc", threshold=0.85)
    web_rec = _recovered(web, pos, src_idx=0)
    assert web_rec == desk_rec == {1, 2, 3, 4}
