"""Feature-matching tests: the warp-tolerant ``method="features"`` path (DESIGN §J.3).

The convolution methods (ncc/ssd/ccorr) score a *fixed-orientation, fixed-scale*
template window, so a rotated or perspective-warped instance falls outside their
score map and is missed. Feature matching (ORB keypoints + RANSAC homography)
recovers those instances precisely because it solves for the geometric transform.

This module pins exactly that contrast on a small, CPU-fast synthetic scene:

  * a distinctive, *texture-rich* asymmetric motif placed UPRIGHT,
  * a ROTATED (+ scaled) copy of it elsewhere on a textured background,

then asserts:

  * ``method="features"`` recovers BOTH instances (especially the rotated one)
    with the source region excluded, within a reasonable bbox tolerance, and
  * ``method="ncc"`` on the same image MISSES the rotated copy -- the concrete
    motivation for offering feature matching as a selectable method.

OpenCV (cv2) is required only for this method; the whole module is skipped when
it is not importable (``pytest.importorskip``). The engine's feature path is
written in parallel, so these tests are best-effort during the parallel build and
re-run at integration -- but they exercise only the public Matcher API of §C.5.
"""

from __future__ import annotations

import numpy as np
import pytest

# Feature matching's only backend is OpenCV; skip the whole module without it
# (mirrors the engine's documented RuntimeError when cv2 is missing -- DESIGN §J.3).
cv2 = pytest.importorskip("cv2", reason="feature matching requires OpenCV (cv2)")

from fastmatch.types import MatchParams  # noqa: E402  (after importorskip by design)

# The engine is a sibling module under parallel development (the feature path is
# added by another task); skip rather than error if it is not importable yet, so
# partial-build runs stay green. DESIGN §J.4.
engine_mod = pytest.importorskip("fastmatch.engine", reason="engine not yet implemented")
Matcher = engine_mod.Matcher

# Reuse the same fast-CPU marker style the engine tests use (registered in
# pyproject.toml's [tool.pytest.ini_options] markers). These scenes are tiny so
# both ORB detection and the NCC sweep finish in well under a second.
cpu = pytest.mark.cpu


# --------------------------------------------------------------------------- #
# Synthetic-scene helpers.
# --------------------------------------------------------------------------- #
def _make_textured_motif(rng: np.random.Generator, h: int = 180, w: int = 160) -> np.ndarray:
    """A distinctive, asymmetric, *feature-rich* RGB motif (h, w, 3) uint8.

    ORB latches onto FAST corners that survive its pyramid/edge-threshold, so a
    handful of big flat shapes is not enough on a ~100 px crop. This motif is a
    **dense field of small (3x3) random-colour blocks** on a dark base -- high
    spatial frequency, hence hundreds of repeatable keypoints at the *default*
    ``cv2.ORB_create()`` settings, plenty of which survive a 35-degree rotation.
    A few bold asymmetric landmarks (a red top-left ``L`` and an off-centre white
    square) keep the orientation unambiguous so the homography is well-posed.

    The motif is intentionally large (180x160) so that even after RANSAC consumes
    most matches on the upright instance, enough remain to also solve the rotated
    one under plain default ORB.
    """
    m = np.full((h, w, 3), 25, dtype=np.uint8)  # dark base so blocks pop
    bs = 3
    cells_y, cells_x = h // bs, w // bs
    field = rng.integers(40, 256, size=(cells_y, cells_x, 3)).astype(np.uint8)
    m[: cells_y * bs, : cells_x * bs] = np.repeat(np.repeat(field, bs, 0), bs, 1)
    # Bold asymmetric landmarks for an unambiguous orientation cue.
    m[: max(4, h // 6), : w // 2] = (245, 30, 30)     # red top bar (left-biased)
    m[: h // 2, : max(4, w // 10)] = (245, 30, 30)    # red left stub -> an L
    cy, cx, s = int(h * 0.30), int(w * 0.55), max(8, h // 8)
    m[cy : cy + s, cx : cx + s] = (255, 255, 255)     # off-centre white square
    return m


def _make_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Low-contrast textured noise background distinct from the motif's palette.

    Faint texture (so ORB does not blow up on a perfectly flat field) but far
    below the motif's contrast, so the motif's keypoints dominate.
    """
    base = rng.integers(96, 136, (h, w, 3)).astype(np.uint8)
    yy = np.linspace(0, 7 * np.pi, h)[:, None]
    xx = np.linspace(0, 7 * np.pi, w)[None, :]
    grating = (7 * np.sin(yy) * np.cos(xx)).astype(np.int16)
    return np.clip(base.astype(np.int16) + grating[:, :, None], 0, 255).astype(np.uint8)


def _build_rotated_scene() -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Scene with an upright motif (source) and a rotated+scaled copy elsewhere.

    Returns ``(image, src_box, rotated_box)`` where each box is half-open
    ``(x, y, w, h)`` in image px: ``src_box`` is the upright template region and
    ``rotated_box`` is the axis-aligned bounding box of the rotated copy.
    """
    rng = np.random.default_rng(3)

    motif = _make_textured_motif(rng)
    mh, mw = motif.shape[:2]

    # The rotated+scaled copy lands on a square canvas of side ``diag`` (its own
    # bounding box); 35 degrees + 1.15x is well beyond what a fixed-orientation
    # NCC window tolerates while still leaving ORB ample matchable structure.
    angle_deg, scale = 35.0, 1.15
    diag = int(np.ceil(np.hypot(mh, mw) * max(1.0, scale))) + 4

    # Size the image so the upright source (top-left) and the rotated copy
    # (lower-right) never overlap and both sit clear of the edges.
    src_x, src_y = 30, 28
    rot_x, rot_y = src_x + mw + 90, src_y + mh + 30
    H = rot_y + diag + 40
    W = rot_x + diag + 40
    img = _make_background(H, W, rng)

    # Upright source copy.
    img[src_y : src_y + mh, src_x : src_x + mw] = motif
    src_box = (src_x, src_y, mw, mh)

    # Single rotation matrix shared by the colour warp and the validity mask, so
    # the mask exactly marks the rotated motif's footprint (the warp uses
    # BORDER_REFLECT to avoid a dark fringe; the mask ensures we paste only real
    # motif pixels, never the reflected border).
    cx, cy = mw / 2.0, mh / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    rot[0, 2] += diag / 2.0 - cx
    rot[1, 2] += diag / 2.0 - cy
    warped = cv2.warpAffine(
        motif, rot, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    mask = cv2.warpAffine(
        np.full((mh, mw), 255, dtype=np.uint8), rot, (diag, diag), flags=cv2.INTER_NEAREST
    ) > 0

    region = img[rot_y : rot_y + diag, rot_x : rot_x + diag]
    region[mask] = warped[mask]
    rotated_box = (rot_x, rot_y, diag, diag)

    return img, src_box, rotated_box


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two half-open ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center_near(box: tuple[int, int, int, int], truth: tuple[int, int, int, int], tol: int) -> bool:
    """True if ``box``'s centre is within ``tol`` px of ``truth``'s centre."""
    bx = box[0] + box[2] / 2.0
    by = box[1] + box[3] / 2.0
    tx = truth[0] + truth[2] / 2.0
    ty = truth[1] + truth[3] / 2.0
    return abs(bx - tx) <= tol and abs(by - ty) <= tol


# --------------------------------------------------------------------------- #
# Feature matching recovers the rotated instance; NCC does not.
# --------------------------------------------------------------------------- #
@cpu
def test_features_recover_rotated_instance_and_exclude_source() -> None:
    """method='features' finds BOTH the upright and the rotated+scaled copy.

    The rotated copy is the point of the test: it sits well outside what a
    fixed-orientation NCC window can score, but ORB+homography solves for the
    transform and recovers it. The upright source region must be excluded.
    DESIGN §J.3.
    """
    img, src_box, rotated_box = _build_rotated_scene()
    sx, sy, sw, sh = src_box
    template = np.ascontiguousarray(img[sy : sy + sh, sx : sx + sw])

    m = Matcher(device="cpu")
    m.set_image(img)

    # ORB detector, generous instance cap; the synthetic copies are near-identical
    # so a modest inlier floor is plenty.
    params = MatchParams(
        method="features",
        feature_detector="orb",
        feature_ratio=0.75,
        feature_min_inliers=10,
        feature_max_instances=20,
        threshold_floor=0.0,  # keep every accepted instance; feature score is a confidence
    )
    results = m.match(template, params, exclude_box=src_box)

    assert isinstance(results, list)
    assert all(0.0 <= r.score <= 1.0 for r in results)

    # The ROTATED copy must be recovered: some returned box overlaps its
    # axis-aligned bbox well and is centred near it (bbox of a rotated quad is
    # looser than a template copy, hence the generous IoU / centre tolerances).
    rot_hits = [
        r
        for r in results
        if _center_near((r.x, r.y, r.w, r.h), rotated_box, tol=max(rotated_box[2], rotated_box[3]) // 4)
        and _iou((r.x, r.y, r.w, r.h), rotated_box) >= 0.4
    ]
    assert rot_hits, (
        "feature matching did not recover the rotated instance "
        f"(rotated_box={rotated_box}, results={[(r.x, r.y, r.w, r.h) for r in results]})"
    )

    # SOURCE EXCLUSION: the upright source region must not be returned.
    for r in results:
        assert _iou((r.x, r.y, r.w, r.h), src_box) <= params.exclude_iou, (
            "source region was not excluded by the feature path"
        )


@cpu
def test_ncc_misses_rotated_copy_that_features_finds() -> None:
    """Contrast: plain NCC MISSES the rotated copy (motivates feature matching).

    NCC scores a fixed-orientation window, so a 35-degree-rotated instance does
    not produce a high-correlation peak at the rotated location. We assert no
    strong NCC hit lands on the rotated copy's bounding box.
    """
    img, src_box, rotated_box = _build_rotated_scene()
    sx, sy, sw, sh = src_box
    template = np.ascontiguousarray(img[sy : sy + sh, sx : sx + sw])

    m = Matcher(device="cpu")
    m.set_image(img)

    params = MatchParams(method="ncc", threshold=0.85, threshold_floor=0.50, scales=(1.0,))
    results = m.match(template, params, exclude_box=src_box)

    # No kept NCC hit should coincide with the rotated copy -- that is the whole
    # reason a feature method is offered.
    rot_hits = [
        r
        for r in results
        if r.score >= params.threshold
        and _center_near((r.x, r.y, r.w, r.h), rotated_box, tol=max(rotated_box[2], rotated_box[3]) // 4)
    ]
    assert not rot_hits, (
        "NCC unexpectedly matched the rotated copy; the features-vs-NCC contrast "
        f"is invalid (hits={[(r.x, r.y, r.w, r.h, r.score) for r in rot_hits]})"
    )


# --------------------------------------------------------------------------- #
# Feature matching recovers MANY identical repeated instances (the demo image).
# --------------------------------------------------------------------------- #
@cpu
def test_features_recover_repeated_instances_on_demo() -> None:
    """Feature matching finds the demo's many identical motif copies (regression).

    The synthetic demo (``loader.generate_sample``) stamps one identical motif at
    N locations. The Lowe ratio test rejects exactly such repeated matches (every
    template feature has N equally-close neighbours), and ``knnMatch(k=2)`` only
    ever sees two copies — so the old matcher recovered ~0-2 of them ("cannot
    match anything"). The generalized-Hough vote + per-cluster homography must
    recover the great majority of the copies with no false positives.
    """
    loader = pytest.importorskip("fastmatch.loader", reason="loader not available")
    import tempfile
    import os

    path = tempfile.mktemp(suffix=".png")
    try:
        gt = loader.generate_sample(path, w=2400, h=1800, tile=160, n_targets=12, seed=0)
        doc = loader.load_image(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    src = gt[0]
    template = np.ascontiguousarray(
        doc.full[src.y : src.y + src.h, src.x : src.x + src.w]
    )
    exclude_box = (src.x, src.y, src.w, src.h)
    others = [g for g in gt if (g.x, g.y) != (src.x, src.y)]

    m = Matcher(device="cpu")
    m.set_image(doc.full)
    params = MatchParams(
        method="features", threshold=0.3, threshold_floor=0.0, feature_min_inliers=8
    )
    results = m.match(template, params, exclude_box=exclude_box)

    truth_boxes = [(g.x, g.y, g.w, g.h) for g in gt]
    recovered = sum(
        any(_iou((r.x, r.y, r.w, r.h), (g.x, g.y, g.w, g.h)) >= 0.5 for r in results)
        for g in others
    )
    false_pos = sum(
        1
        for r in results
        if not any(_iou((r.x, r.y, r.w, r.h), tb) >= 0.5 for tb in truth_boxes)
    )
    # Must recover the large majority of the identical copies (the old matcher got
    # ~0-2), with no spurious boxes.
    assert recovered >= int(0.8 * len(others)), (
        f"feature matching recovered only {recovered}/{len(others)} repeated copies"
    )
    assert false_pos == 0, f"feature matching produced {false_pos} false positive(s)"


# --------------------------------------------------------------------------- #
# Channel mode (luminance vs rgb): the appearance verification is colour-aware.
# --------------------------------------------------------------------------- #
def _make_colorful_motif(rng: np.random.Generator, h: int = 120, w: int = 120) -> np.ndarray:
    """Dense, **per-channel-independent** random colour field (h, w, 3) uint8.

    The luminance is high-frequency (so ORB fires plenty of keypoints), while the
    three channels are decorrelated — so a same-luminance *gray* copy correlates
    only weakly in rgb (≈0.5) even though it is identical in luminance (=1.0).
    """
    bs = 3
    cy, cx = h // bs, w // bs
    field = rng.integers(30, 256, size=(cy, cx, 3)).astype(np.uint8)
    return np.repeat(np.repeat(field, bs, 0), bs, 1)


@cpu
def test_features_rgb_verification_is_colour_aware() -> None:
    """``channel_mode="rgb"`` rejects a same-luminance/different-colour decoy.

    Scene: an upright colour motif (source), an identical **colour** copy, and a
    **gray** copy with the *exact same luminance* but no chroma. Keypoint
    detection is on luminance, so the gray decoy is detected/matched identically
    to the colour copy and is *proposed* in both modes. The difference is the
    appearance verification:

      * **luminance** mode -> the gray decoy correlates perfectly (ZNCC=1), so it
        IS recovered (same as the colour copy);
      * **rgb** mode -> the per-channel-summed NCC against the *colour* template
        drops well below the accept floor for the chroma-less decoy, so it is
        rejected — while the genuine colour copy (ZNCC=1) is still recovered.

    This pins that rgb feature matching genuinely uses colour (not just luminance).
    """
    rng = np.random.default_rng(7)
    motif = _make_colorful_motif(rng)
    mh, mw = motif.shape[:2]
    # Gray decoy: identical BT.601 luminance, zero chroma (R=G=B=luminance).
    lum = motif[:, :, 0] * 0.299 + motif[:, :, 1] * 0.587 + motif[:, :, 2] * 0.114
    lum_u8 = np.clip(np.rint(lum), 0, 255).astype(np.uint8)
    decoy = np.repeat(lum_u8[:, :, None], 3, axis=2)

    img = _make_background(mh + 40, 20 + 3 * (mw + 40), rng)
    sx, tx, dx = 20, 20 + (mw + 40), 20 + 2 * (mw + 40)
    y = 20
    img[y : y + mh, sx : sx + mw] = motif       # source (colour)
    img[y : y + mh, tx : tx + mw] = motif       # true colour copy
    img[y : y + mh, dx : dx + mw] = decoy        # same-luminance gray decoy
    src_box = (sx, y, mw, mh)
    true_box = (tx, y, mw, mh)
    decoy_box = (dx, y, mw, mh)
    template = np.ascontiguousarray(img[y : y + mh, sx : sx + mw])

    m = Matcher(device="cpu")
    m.set_image(img)

    def _run(mode: str):
        params = MatchParams(
            method="features", channel_mode=mode, threshold_floor=0.0,
            feature_min_inliers=8, feature_max_instances=20,
        )
        return m.match(template, params, exclude_box=src_box)

    def _hits(results, box):
        return [r for r in results if _iou((r.x, r.y, r.w, r.h), box) >= 0.5]

    lum_res = _run("luminance")
    assert _hits(lum_res, true_box), "luminance mode missed the colour copy"
    assert _hits(lum_res, decoy_box), "luminance mode should match the same-luminance decoy"

    rgb_res = _run("rgb")
    assert all(0.0 <= r.score <= 1.0 for r in rgb_res)
    assert _hits(rgb_res, true_box), "rgb mode missed the genuine colour copy"
    assert not _hits(rgb_res, decoy_box), (
        "rgb mode should REJECT the chroma-less decoy (it differs in colour)"
    )
    # Source region excluded in both modes.
    for r in rgb_res:
        assert _iou((r.x, r.y, r.w, r.h), src_box) <= MatchParams().exclude_iou
