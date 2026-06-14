"""The experimental Shazam matcher is KEPT but DISABLED.

Two contracts:
  * it is **disabled from the UI** — ``"shazam"`` is not in ``types.METHODS`` (so
    the params-panel dropdown never offers it), and
  * it is **kept and reachable** — the engine still dispatches ``method="shazam"``
    when set explicitly, runs without error, and returns valid Matches.

We do not pin its recall (it is known to underperform feature matching — see
fastmatch/shazam_matcher.py); only that the disabled-but-kept wiring works.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="shazam matching requires OpenCV (cv2)")

from fastmatch.types import METHODS, SHAZAM_METHOD, MatchParams  # noqa: E402

engine_mod = pytest.importorskip("fastmatch.engine", reason="engine not available")
Matcher = engine_mod.Matcher

cpu = pytest.mark.cpu


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def test_shazam_is_disabled_from_the_ui() -> None:
    """The method exists as a constant but is NOT selectable (not in METHODS)."""
    assert SHAZAM_METHOD == "shazam"
    assert SHAZAM_METHOD not in METHODS  # never offered in the dropdown


@cpu
def test_shazam_is_reachable_and_runs_when_set_explicitly() -> None:
    """Setting method='shazam' explicitly runs the kept backend and returns Matches.

    Uses a feature-rich (dense high-frequency) motif so ORB has plenty of
    keypoints to form the consistent landmark pairs the vote needs; the scene is
    small for a fast CPU test.
    """
    rng = np.random.default_rng(0)
    # Dense 4px colour blocks -> high-frequency luma -> many repeatable ORB corners.
    bs, cy, cx = 4, 34, 34
    field = rng.integers(20, 256, size=(cy, cx, 3)).astype(np.uint8)
    motif = np.repeat(np.repeat(field, bs, 0), bs, 1)
    mh, mw = motif.shape[:2]

    img = rng.integers(96, 140, (mh + 60, 40 + 3 * (mw + 40), 3)).astype(np.uint8)
    xs = [20, 20 + (mw + 40), 20 + 2 * (mw + 40)]
    y = 30
    for x in xs:
        img[y : y + mh, x : x + mw] = motif
    src = (xs[0], y, mw, mh)
    others = [(xs[1], y, mw, mh), (xs[2], y, mw, mh)]
    template = np.ascontiguousarray(img[y : y + mh, src[0] : src[0] + mw])

    m = Matcher(device="cpu")
    m.set_image(img)
    res = m.match(
        template,
        MatchParams(method="shazam", threshold_floor=0.0, feature_max_instances=20),
        exclude_box=src,
    )

    assert isinstance(res, list)
    assert all(0.0 <= r.score <= 1.0 for r in res)
    # Source excluded; recovers at least one of the identical copies.
    for r in res:
        assert _iou((r.x, r.y, r.w, r.h), src) <= MatchParams().exclude_iou
    recovered = sum(
        any(_iou((r.x, r.y, r.w, r.h), o) >= 0.5 for r in res) for o in others
    )
    assert recovered >= 1, f"shazam recovered {recovered} of {len(others)} copies"
