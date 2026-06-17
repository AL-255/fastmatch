"""Masked (rectilinear-selection) matching for the convolution methods.

A template mask restricts the score to the SELECTED pixels only. The scene below
has a copy that matches the source on its LEFT half but differs on the right; with
no mask the copy scores ~0.5 (half matches), and with a left-half mask it scores
~1.0 — proving the matcher honors the mask. An all-True / mismatched mask must be
a no-op (the fast rectangular path).
"""

from __future__ import annotations

import numpy as np
import pytest

from fastmatch.engine import Matcher
from fastmatch.types import MatchParams

cpu = pytest.mark.cpu


def _scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 210, (200, 400, 3)).astype(np.uint8)
    P = rng.integers(0, 256, (40, 20, 3)).astype(np.uint8)     # shared left half
    q_src = rng.integers(0, 256, (40, 20, 3)).astype(np.uint8)  # source right half
    r_copy = rng.integers(0, 256, (40, 20, 3)).astype(np.uint8)  # copy right half (differs)

    def place(x, y, left, right):
        img[y : y + 40, x : x + 20] = left
        img[y : y + 40, x + 20 : x + 40] = right

    place(50, 50, P, q_src)     # source at (50,50): P | q_src
    place(250, 50, P, r_copy)   # copy   at (250,50): P | r_copy
    tmpl = np.ascontiguousarray(img[50:90, 50:90])
    return img, tmpl, (50, 50, 40, 40)


def _copy_score(res):
    return max(
        (r.score for r in res if abs(r.x - 250) < 8 and abs(r.y - 50) < 8), default=None
    )


@cpu
def test_mask_restricts_ncc_to_selected_region() -> None:
    img, tmpl, src = _scene()
    m = Matcher(device="cpu")
    m.set_image(img)
    mask = np.zeros((40, 40), bool)
    mask[:, :20] = True  # left half only

    p = MatchParams(method="ncc", threshold_floor=0.3, threshold=0.0)
    unmasked = _copy_score(m.match(tmpl, p, exclude_box=src))
    masked = _copy_score(m.match(tmpl, p, exclude_box=src, mask=mask))

    assert unmasked is not None and masked is not None
    assert masked > 0.95              # left half matches exactly under the mask
    assert masked - unmasked > 0.3    # masking out the differing half lifts the score


@cpu
@pytest.mark.parametrize("method", ["ssd", "ccorr"])
def test_mask_works_for_ssd_and_ccorr(method: str) -> None:
    img, tmpl, src = _scene()
    m = Matcher(device="cpu")
    m.set_image(img)
    mask = np.zeros((40, 40), bool)
    mask[:, :20] = True
    p = MatchParams(method=method, threshold_floor=0.3, threshold=0.0)
    masked = _copy_score(m.match(tmpl, p, exclude_box=src, mask=mask))
    assert masked is not None and masked > 0.95


@cpu
def test_mask_with_rotation_and_flipping_runs() -> None:
    img, tmpl, src = _scene()
    m = Matcher(device="cpu")
    m.set_image(img)
    mask = np.zeros((40, 40), bool)
    mask[:, :20] = True
    p = MatchParams(
        method="ncc", threshold_floor=0.3, threshold=0.0,
        enable_rotation=True, enable_flipping=True,
    )
    res = m.match(tmpl, p, exclude_box=src, mask=mask)
    assert _copy_score(res) is not None and _copy_score(res) > 0.95


@cpu
def test_full_or_mismatched_mask_is_a_noop() -> None:
    """An all-True or shape-mismatched mask matches the no-mask result exactly."""
    img, tmpl, src = _scene()
    m = Matcher(device="cpu")
    m.set_image(img)
    p = MatchParams(method="ncc", threshold_floor=0.3, threshold=0.0)
    base = _copy_score(m.match(tmpl, p, exclude_box=src))
    full = _copy_score(m.match(tmpl, p, exclude_box=src, mask=np.ones((40, 40), bool)))
    wrong = _copy_score(m.match(tmpl, p, exclude_box=src, mask=np.ones((9, 9), bool)))
    assert base == full == wrong
