"""Channel-mode + per-channel-weight tests (luminance / rgb / ycbcr).

Covers the weighted multi-channel matching shared by every method:
  * ``normalize_weights`` (the sum-to-1 contract the UI relies on),
  * a Y-only ``ycbcr`` weight reproduces ``luminance`` (Y == BT.601 luma),
  * chroma-weighted matching is genuinely colour-discriminating (it rejects a
    same-luminance gray decoy that luminance matching accepts), for both the conv
    NCC path and the feature path,
  * all conv methods and the feature path run in every channel mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastmatch.types import MatchParams, normalize_weights

engine_mod = pytest.importorskip("fastmatch.engine", reason="engine not available")
Matcher = engine_mod.Matcher

cpu = pytest.mark.cpu


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _recovered(results, boxes, thr=0.5) -> int:
    return sum(
        any(_iou((r.x, r.y, r.w, r.h), b) >= thr for r in results) for b in boxes
    )


# --------------------------------------------------------------------------- #
# normalize_weights
# --------------------------------------------------------------------------- #
def test_normalize_weights_sums_to_one():
    w = normalize_weights((2.0, 1.0, 1.0), 3)
    assert pytest.approx(sum(w), abs=1e-9) == 1.0
    assert w == pytest.approx((0.5, 0.25, 0.25))


def test_normalize_weights_handles_zero_and_negative():
    assert normalize_weights((0.0, 0.0, 0.0), 3) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    # negatives clamp to 0; remainder renormalizes.
    assert normalize_weights((-5.0, 1.0, 1.0), 3) == pytest.approx((0.0, 0.5, 0.5))
    # pads / truncates to n.
    assert normalize_weights((1.0,), 3) == pytest.approx((1.0, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# Scene helpers: a colour motif, an identical copy, and a same-luminance gray decoy
# --------------------------------------------------------------------------- #
def _colour_scene(seed=3, n_side=64):
    rng = np.random.default_rng(seed)
    # Independent per-channel random blocks -> textured luma AND textured chroma,
    # so a gray (chroma-less) copy correlates poorly in chroma space.
    bs = 4
    field = rng.integers(25, 256, size=(n_side // bs, n_side // bs, 3)).astype(np.uint8)
    motif = np.repeat(np.repeat(field, bs, 0), bs, 1)
    mh, mw = motif.shape[:2]
    lum = motif[:, :, 0] * 0.299 + motif[:, :, 1] * 0.587 + motif[:, :, 2] * 0.114
    decoy = np.repeat(np.clip(np.rint(lum), 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)

    img = rng.integers(96, 140, (mh + 40, 20 + 3 * (mw + 30), 3)).astype(np.uint8)
    sx, ax, dx = 20, 20 + (mw + 30), 20 + 2 * (mw + 30)
    y = 20
    img[y : y + mh, sx : sx + mw] = motif   # source (colour)
    img[y : y + mh, ax : ax + mw] = motif   # identical colour copy
    img[y : y + mh, dx : dx + mw] = decoy    # same-luminance gray decoy
    return (
        img,
        (sx, y, mw, mh),
        (ax, y, mw, mh),   # colour copy
        (dx, y, mw, mh),   # gray decoy
    )


# --------------------------------------------------------------------------- #
# Y-only ycbcr == luminance (Y is the BT.601 luma plane)
# --------------------------------------------------------------------------- #
@cpu
def test_ycbcr_y_only_weight_matches_luminance_ncc():
    img, src, colour, decoy = _colour_scene()
    m = Matcher(device="cpu")
    m.set_image(img)
    tmpl = np.ascontiguousarray(img[src[1] : src[1] + src[3], src[0] : src[0] + src[2]])

    lum = m.match(
        tmpl, MatchParams(method="ncc", threshold=0.85, threshold_floor=0.85, scales=(1.0,)),
        exclude_box=src,
    )
    yon = m.match(
        tmpl,
        MatchParams(
            method="ncc", channel_mode="ycbcr", ycbcr_weights=(1.0, 0.0, 0.0),
            threshold=0.85, threshold_floor=0.85, scales=(1.0,),
        ),
        exclude_box=src,
    )
    # Both should recover the same instances (colour copy + the gray decoy, since
    # both share the source's luminance) at the same locations.
    targets = [colour, decoy]
    assert _recovered(lum, targets) == _recovered(yon, targets) == 2


# --------------------------------------------------------------------------- #
# Chroma weighting is colour-discriminating (conv NCC)
# --------------------------------------------------------------------------- #
@cpu
def test_ncc_chroma_weighting_rejects_same_luminance_decoy():
    img, src, colour, decoy = _colour_scene()
    m = Matcher(device="cpu")
    m.set_image(img)
    tmpl = np.ascontiguousarray(img[src[1] : src[1] + src[3], src[0] : src[0] + src[2]])

    # Luminance: the gray decoy shares the source's luma, so it IS matched.
    lum = m.match(
        tmpl, MatchParams(method="ncc", threshold=0.85, threshold_floor=0.5, scales=(1.0,)),
        exclude_box=src,
    )
    assert _recovered(lum, [colour]) == 1
    assert _recovered(lum, [decoy]) == 1

    # Chroma-only ycbcr (weight on Cb,Cr): the colour copy still matches, but the
    # gray decoy has flat chroma -> no chroma correlation -> rejected.
    chroma = m.match(
        tmpl,
        MatchParams(
            method="ncc", channel_mode="ycbcr", ycbcr_weights=(0.0, 0.5, 0.5),
            threshold=0.85, threshold_floor=0.5, scales=(1.0,),
        ),
        exclude_box=src,
    )
    assert _recovered(chroma, [colour]) == 1
    assert _recovered(chroma, [decoy]) == 0


# --------------------------------------------------------------------------- #
# Every conv method runs in every channel mode and returns valid scores
# --------------------------------------------------------------------------- #
@cpu
@pytest.mark.parametrize("method", ["ncc", "ssd", "ccorr"])
@pytest.mark.parametrize("mode", ["luminance", "rgb", "ycbcr"])
def test_conv_methods_run_in_all_channel_modes(method, mode):
    img, src, colour, decoy = _colour_scene()
    m = Matcher(device="cpu")
    m.set_image(img)
    tmpl = np.ascontiguousarray(img[src[1] : src[1] + src[3], src[0] : src[0] + src[2]])
    res = m.match(
        tmpl,
        MatchParams(method=method, channel_mode=mode, threshold=0.8, threshold_floor=0.5,
                    scales=(1.0,)),
        exclude_box=src,
    )
    assert isinstance(res, list)
    # Scores are ~[0,1]; NCC is clamped at 0 but not at 1, so an exact-copy peak
    # can land a hair above 1.0 from float rounding (pre-existing behaviour).
    assert all(0.0 <= r.score <= 1.0001 for r in res)
    # The identical colour copy is recovered in every mode/method.
    assert _recovered(res, [colour]) == 1


# --------------------------------------------------------------------------- #
# Feature path: ycbcr runs, and chroma weighting is colour-discriminating
# --------------------------------------------------------------------------- #
@cpu
def test_features_ycbcr_chroma_weighting_rejects_gray_decoy():
    pytest.importorskip("cv2", reason="feature matching requires OpenCV")
    img, src, colour, decoy = _colour_scene()
    m = Matcher(device="cpu")
    m.set_image(img)
    tmpl = np.ascontiguousarray(img[src[1] : src[1] + src[3], src[0] : src[0] + src[2]])

    def run(mode, **kw):
        return m.match(
            tmpl,
            MatchParams(method="features", channel_mode=mode, threshold_floor=0.0,
                        feature_min_inliers=8, feature_max_instances=20, **kw),
            exclude_box=src,
        )

    # luminance matches both the colour copy and the same-luminance gray decoy.
    lum = run("luminance")
    assert _recovered(lum, [colour]) == 1
    assert _recovered(lum, [decoy]) == 1

    # Chroma-only ycbcr (Y weight 0) keeps the colour copy but drops the gray
    # decoy — its chroma is flat, so it carries no colour correlation. (Any Y
    # weight would let the decoy's perfect luminance match rescue it, which is the
    # correct behaviour — discriminating by colour means weighting chroma.)
    chroma = run("ycbcr", ycbcr_weights=(0.0, 0.5, 0.5))
    assert all(0.0 <= r.score <= 1.0 for r in chroma)
    assert _recovered(chroma, [colour]) == 1
    assert _recovered(chroma, [decoy]) == 0
