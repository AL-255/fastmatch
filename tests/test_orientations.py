"""Dihedral-orientation (D4) search tests (DESIGN §L).

The boxed template can appear in the image under any of the 8 symmetries of a
square. Two UI checkboxes select the active subset and every matching method
honours them; each kept :class:`~fastmatch.types.Match` records the orientation
it was found under (``Match.orientation``). The shared contract lives in
``fastmatch.types`` and is **frozen**; this module pins both that contract and
the end-to-end engine/feature behaviour built on top of it.

Two layers of tests:

  * **Pure-contract** (no engine, always run): ``active_orientations`` for the
    four checkbox combos, ``apply_orientation`` producing 8 distinct arrays from
    an asymmetric template (with the H/W swap for the diagonal family), and
    ``orientation_from_matrix`` round-tripping the 8 canonical linear maps (with
    a scale applied) and classifying reflection by determinant sign.
  * **Engine / feature** (``@pytest.mark.cpu``, engine ``importorskip``ed): a
    small CPU-fast scene with a motif stamped at several true orientations, run
    with the checkboxes on/off, asserting the right instances are recovered AND
    tagged with the right orientation. The orientation feature in the engine is
    written in parallel, so these are best-effort during the parallel build and
    re-run at integration; they use only the public Matcher API of §C.5.

All scenes are tiny so the CPU-only torch build runs each test well under a
second (the conv pyramid keeps the sweep fast).
"""

from __future__ import annotations

import numpy as np
import pytest

# Pure-contract surface from the FROZEN shared types (always importable). The
# engine/feature tests below skip if the engine is not built yet, but these
# contract tests must pass today.
from fastmatch.types import (
    ORIENTATIONS,
    MatchParams,
    active_orientations,
    apply_orientation,
    orientation_from_matrix,
)

# Same fast-CPU marker the engine/feature suites use (registered in
# pyproject.toml's [tool.pytest.ini_options] markers).
cpu = pytest.mark.cpu

# Orientations that involve a reflection (odd number of flips). Used to assert a
# mirrored stamp is tagged as *some* reflection without demanding an exact code
# (the diagonal family MXR90/MYR90 is a flip AND a turn).
REFLECTIONS: frozenset[str] = frozenset({"MX", "MY", "MXR90", "MYR90"})


# =========================================================================== #
# Pure-contract tests (no engine needed; types.py is implemented and frozen).
# =========================================================================== #
def test_active_orientations_four_checkbox_combos() -> None:
    """The two checkboxes select exactly the documented orientation subsets.

    DESIGN §L / ``active_orientations``:
        (off, off)  -> R0
        (rot, off)  -> R0, R90, R180, R270
        (off, flip) -> R0, MX, MY
        (rot, flip) -> all 8
    Always in canonical :data:`ORIENTATIONS` order and always including R0.
    """
    assert active_orientations(False, False) == ("R0",)
    assert active_orientations(True, False) == ("R0", "R90", "R180", "R270")
    assert active_orientations(False, True) == ("R0", "MX", "MY")
    assert active_orientations(True, True) == ORIENTATIONS  # all 8, canonical order

    # R0 is in every subset, and every returned subset is in canonical order
    # (a subsequence of ORIENTATIONS) with no duplicates.
    for er in (False, True):
        for ef in (False, True):
            got = active_orientations(er, ef)
            assert got[0] == "R0"
            assert len(got) == len(set(got))
            order = [ORIENTATIONS.index(o) for o in got]
            assert order == sorted(order)


def _asymmetric_template() -> np.ndarray:
    """A small RGB template with NO dihedral symmetry, so all 8 transforms differ.

    Non-square (different H and W) so the H/W-swapping orientations are also
    structurally distinct, and asymmetric in content so e.g. MX != R180.
    """
    h, w = 5, 7
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # A unique value per pixel-channel breaks every reflection/rotation symmetry.
    arr[..., 0] = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    arr[..., 1] = np.arange(h * w, dtype=np.uint8).reshape(h, w) + 100
    arr[0, 0] = (255, 0, 0)      # a corner landmark
    arr[-1, -1] = (0, 0, 255)    # the opposite corner, a different colour
    return arr


def test_apply_orientation_produces_eight_distinct_arrays() -> None:
    """The 8 D4 transforms of an asymmetric template are all distinct (DESIGN §L)."""
    arr = _asymmetric_template()
    outs = {o: apply_orientation(arr, o) for o in ORIENTATIONS}

    # R0 is the identity (same content).
    np.testing.assert_array_equal(outs["R0"], arr)

    # All 8 results are pairwise distinct. Arrays of differing shape are trivially
    # distinct; same-shape ones must differ in content.
    keys = list(ORIENTATIONS)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = outs[keys[i]], outs[keys[j]]
            distinct = a.shape != b.shape or not np.array_equal(a, b)
            assert distinct, f"orientations {keys[i]} and {keys[j]} produced identical arrays"

    # Every result is C-contiguous (handed straight to torch/cv2 per the contract).
    for o, out in outs.items():
        assert out.flags["C_CONTIGUOUS"], f"{o} result is not C-contiguous"


def test_apply_orientation_swaps_hw_for_rotation_and_diagonal_family() -> None:
    """R90/R270/MXR90/MYR90 swap H and W; R180/MX/MY keep the shape (DESIGN §L)."""
    arr = _asymmetric_template()
    h, w = arr.shape[:2]
    assert h != w  # the test only means something for a non-square template

    swap = {"R90", "R270", "MXR90", "MYR90"}
    keep = {"R0", "R180", "MX", "MY"}
    for o in swap:
        out = apply_orientation(arr, o)
        assert out.shape[:2] == (w, h), f"{o} should swap H/W -> {(w, h)}, got {out.shape[:2]}"
    for o in keep:
        out = apply_orientation(arr, o)
        assert out.shape[:2] == (h, w), f"{o} should keep shape {(h, w)}, got {out.shape[:2]}"


def test_apply_orientation_works_on_2d_array() -> None:
    """``apply_orientation`` also handles a single-channel (H, W) array."""
    arr = _asymmetric_template()[..., 0]  # (H, W)
    h, w = arr.shape
    assert apply_orientation(arr, "R0").shape == (h, w)
    assert apply_orientation(arr, "R90").shape == (w, h)
    # The grayscale R180 must equal the channel-0 plane of the RGB R180 (consistency).
    rgb180 = apply_orientation(_asymmetric_template(), "R180")
    np.testing.assert_array_equal(apply_orientation(arr, "R180"), rgb180[..., 0])


def test_apply_orientation_rejects_unknown_code() -> None:
    """An unknown orientation code raises ValueError (defensive contract)."""
    with pytest.raises(ValueError):
        apply_orientation(_asymmetric_template(), "NOPE")


# Canonical 2x2 linear part of each D4 orientation, DERIVED from the actual
# ``apply_orientation`` numpy transform rather than hardcoded. This pins the real
# contract: ``orientation_from_matrix`` must label a transform with the SAME code
# ``apply_orientation`` produces (image coords: +x right, +y down). Deriving it
# here catches a convention mismatch between the two functions (e.g. np.rot90's
# array-CCW turn being clockwise in +y-down image coords) — exactly the kind of
# R90<->R270 swap a hardcoded copy would silently bless.
def _apply_orientation_linear(orient: str) -> np.ndarray:
    """Least-squares fit of the forward linear map of ``apply_orientation``."""
    h, w = 40, 20  # non-square so H/W swaps are observable
    xs = np.tile(np.arange(w), (h, 1)).astype(np.float64)
    ys = np.tile(np.arange(h).reshape(-1, 1), (1, w)).astype(np.float64)
    tx = apply_orientation(xs, orient).ravel()
    ty = apply_orientation(ys, orient).ravel()
    ho, wo = apply_orientation(xs, orient).shape
    xo = np.tile(np.arange(wo), (ho, 1)).ravel().astype(np.float64)
    yo = np.tile(np.arange(ho).reshape(-1, 1), (1, wo)).ravel().astype(np.float64)
    a = np.stack([tx, ty, np.ones_like(tx)], axis=1)
    lx = np.linalg.lstsq(a, xo, rcond=None)[0]
    ly = np.linalg.lstsq(a, yo, rcond=None)[0]
    return np.array([[round(lx[0]), round(lx[1])], [round(ly[0]), round(ly[1])]], dtype=np.float64)


_CANONICAL_LINEAR: dict[str, np.ndarray] = {o: _apply_orientation_linear(o) for o in ORIENTATIONS}


def test_orientation_from_matrix_round_trips_canonical_matrices() -> None:
    """Each canonical D4 linear map (× a scale) classifies back to itself (DESIGN §L)."""
    for scale in (0.5, 1.0, 2.3, 7.0):
        for code, lin in _CANONICAL_LINEAR.items():
            got = orientation_from_matrix(lin * scale)
            assert got == code, f"scale={scale} {code} misclassified as {got}"


def test_orientation_from_matrix_detects_reflection_by_det_sign() -> None:
    """Reflections (det < 0) classify to a reflection code; rotations to a rotation.

    A homography's linear part captures the mirror in the sign of its
    determinant; ``orientation_from_matrix`` must route a negative-det map to one
    of the reflection codes and a positive-det map to a pure rotation. DESIGN §L.
    """
    rotations = {"R0", "R90", "R180", "R270"}
    for code, lin in _CANONICAL_LINEAR.items():
        det = float(np.linalg.det(lin))
        if code in rotations:
            assert det > 0, f"{code} should be a proper rotation (det>0)"
            assert orientation_from_matrix(lin) in rotations
        else:
            assert det < 0, f"{code} should be a reflection (det<0)"
            assert orientation_from_matrix(lin) in REFLECTIONS


def test_orientation_from_matrix_tolerates_small_perturbation() -> None:
    """A noisy near-canonical linear map still classifies to the nearest D4 code.

    Recovered homographies are never exactly canonical; the classifier picks the
    nearest of the 8 after scale-normalization, so a small perturbation must not
    flip the result.
    """
    rng = np.random.default_rng(0)
    for code, lin in _CANONICAL_LINEAR.items():
        for _ in range(5):
            noisy = lin * 1.7 + rng.normal(0.0, 0.06, size=(2, 2))
            assert orientation_from_matrix(noisy) == code, f"{code} flipped under small noise"


def test_orientation_from_matrix_degenerate_returns_r0() -> None:
    """A (near-)singular linear part degrades to R0 rather than raising."""
    assert orientation_from_matrix(np.zeros((2, 2))) == "R0"


# =========================================================================== #
# Engine tests (CPU-fast; engine importorskip'd — orientation lands in parallel).
# =========================================================================== #
# The conv orientation path is implemented by another task; skip rather than
# error if the engine is not importable, so partial-build runs stay green.
engine_mod = pytest.importorskip("fastmatch.engine", reason="engine not yet implemented")
Matcher = engine_mod.Matcher


MOTIF_W, MOTIF_H = 24, 28  # asymmetric on purpose -> distinct under every D4 op


def _make_motif(rng: np.random.Generator) -> np.ndarray:
    """A distinctive, high-contrast, asymmetric RGB motif (MOTIF_H, MOTIF_W, 3).

    Asymmetric content so an upright (R0) template only correlates strongly with
    a copy stamped at the *same* orientation: matching a 90°/180°/mirror copy
    requires the engine to actually transform the template.
    """
    m = np.zeros((MOTIF_H, MOTIF_W, 3), dtype=np.uint8)
    m[:, :, 0] = 30
    m[: MOTIF_H // 2, : MOTIF_W // 2, 0] = 240          # top-left red
    m[: MOTIF_H // 2, MOTIF_W // 2 :, 1] = 220          # top-right green
    m[MOTIF_H // 2 :, : MOTIF_W // 3, 2] = 200          # bottom-left blue stripe
    m[MOTIF_H // 2 :, MOTIF_W // 3 :, :] = 255          # bottom-right white block
    m[2:6, MOTIF_W - 6 : MOTIF_W - 2, :] = 0            # off-centre dark notch
    m = np.clip(m.astype(np.int16) + rng.integers(-4, 5, m.shape), 0, 255).astype(np.uint8)
    return m


def _make_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Low-contrast textured noise background distinct from the motif's palette."""
    base = rng.integers(96, 136, (h, w, 3)).astype(np.uint8)
    yy = np.linspace(0, 6 * np.pi, h)[:, None]
    xx = np.linspace(0, 6 * np.pi, w)[None, :]
    grating = (8 * np.sin(yy) * np.cos(xx)).astype(np.int16)
    return np.clip(base.astype(np.int16) + grating[:, :, None], 0, 255).astype(np.uint8)


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


def _build_oriented_scene() -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Image with the motif stamped UPRIGHT, +90°, +180°, and MIRRORED (MY).

    Returns ``(image, upright_motif, truths)`` where ``upright_motif`` is the R0
    crop to use as the template and ``truths`` maps the true orientation code to
    the stamped instance's half-open ``(x, y, w, h)`` box. The stamped copies are
    produced with numpy rot90/flip so the geometry is exact and matches the D4
    transforms ``apply_orientation`` applies.

    Note: a numpy ``rot90(arr, 1)`` (counter-clockwise) is the ``"R90"`` code in
    :func:`apply_orientation`, and a left-right flip ``arr[:, ::-1]`` is ``"MY"``
    -- the same transforms the engine applies to the template, so the recovered
    orientation tag should equal the stamp's code below.
    """
    rng = np.random.default_rng(0)
    h, w = 360, 460
    img = _make_background(h, w, rng)
    motif = _make_motif(rng)
    mh, mw = motif.shape[:2]

    # Stamped copies, each generated with the SAME transform apply_orientation
    # uses for that code, so the truth orientation is unambiguous.
    stamps: dict[str, np.ndarray] = {
        "R0": motif,
        "R90": np.ascontiguousarray(np.rot90(motif, 1)),     # CCW 90° == code "R90"
        "R180": np.ascontiguousarray(np.rot90(motif, 2)),
        "MY": np.ascontiguousarray(motif[:, ::-1]),          # left-right mirror == "MY"
    }
    # Well-spread, non-overlapping top-left positions (mind the H/W swap for R90).
    positions: dict[str, tuple[int, int]] = {
        "R0": (24, 22),
        "R90": (260, 30),
        "R180": (60, 230),
        "MY": (320, 250),
    }
    truths: dict[str, tuple[int, int, int, int]] = {}
    for code, stamp in stamps.items():
        x, y = positions[code]
        sh, sw = stamp.shape[:2]
        img[y : y + sh, x : x + sw] = stamp
        truths[code] = (x, y, sw, sh)
    return img, motif, truths


def _default_params(**over) -> MatchParams:
    """MatchParams tuned for the small, high-SNR oriented synthetic scene."""
    kw = dict(threshold=0.85, threshold_floor=0.50, scales=(1.0,), max_results=100)
    kw.update(over)
    return MatchParams(**kw)


def _find_hit(results, truth: tuple[int, int, int, int], thr: float):
    """The best kept result whose box matches ``truth`` (IoU >= 0.5), or None."""
    cands = [r for r in results if r.score >= thr and _iou((r.x, r.y, r.w, r.h), truth) >= 0.5]
    return max(cands, key=lambda r: r.score) if cands else None


@cpu
def test_orientation_search_recovers_all_stamps_with_correct_tags() -> None:
    """All-orientations search recovers every stamped copy, tagged with its true code.

    With both checkboxes on, the engine searches all 8 D4 orientations of the
    template; the scene stamps R0/R90/R180/MY copies, so each must be recovered
    (IoU-located) AND carry the matching ``Match.orientation``. DESIGN §L.
    """
    img, motif, truths = _build_oriented_scene()
    mh, mw = motif.shape[:2]
    sx, sy, _sw, _sh = truths["R0"]
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])

    m = Matcher(device="cpu")
    m.set_image(img)
    params = _default_params(enable_rotation=True, enable_flipping=True)
    results = m.match(template, params, exclude_box=truths["R0"])

    assert isinstance(results, list)
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert all(r.orientation in ORIENTATIONS for r in results)
    # R0 is the source crop and is excluded; the other three must come back.
    for code in ("R90", "R180", "MY"):
        hit = _find_hit(results, truths[code], params.threshold)
        assert hit is not None, f"missed the {code} stamp at {truths[code]}"
        assert hit.orientation == code, (
            f"{code} stamp recovered but tagged {hit.orientation!r} (expected {code!r})"
        )


@cpu
def test_orientation_off_finds_only_upright_copy() -> None:
    """Both checkboxes off: ONLY the upright (R0) copy is found (baseline behaviour).

    This is the byte-for-byte-current path: no template transforms, so the 90°,
    180° and mirrored copies must NOT be recovered, and nothing carries a
    non-R0 tag. DESIGN §L. Here we make the *R90* copy the excluded source and
    confirm the upright copy is found while the rotated/mirrored ones are not.
    """
    img, motif, truths = _build_oriented_scene()
    mh, mw = motif.shape[:2]
    # Use the upright stamp as the template but DON'T exclude it; exclude a far-away
    # box so the upright R0 copy itself is returnable as the one valid same-orientation
    # match, while R90/R180/MY remain unrecoverable without orientation search.
    sx, sy, _sw, _sh = truths["R0"]
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])

    m = Matcher(device="cpu")
    m.set_image(img)
    params = _default_params(enable_rotation=False, enable_flipping=False)
    # Exclude a dummy region away from every stamp so the R0 copy is kept.
    results = m.match(template, params, exclude_box=(0, 0, 4, 4))

    assert all(r.orientation == "R0" for r in results), (
        "with both checkboxes off every Match must be tagged R0"
    )
    kept = [r for r in results if r.score >= params.threshold]
    # The upright copy IS found (same orientation as the template).
    assert _find_hit(kept, truths["R0"], params.threshold) is not None, "upright copy not found"
    # The rotated / mirrored copies are NOT found by a fixed-orientation search.
    for code in ("R90", "R180", "MY"):
        assert _find_hit(kept, truths[code], params.threshold) is None, (
            f"fixed-orientation search unexpectedly recovered the {code} copy"
        )


@cpu
def test_orientation_search_ssd_spot_check() -> None:
    """SSD honours the orientation checkboxes too (spot-check; DESIGN §L / §J.2).

    Orientation search is method-agnostic for the conv methods, so a quick SSD
    pass must also recover at least the rotated copies and tag them correctly.
    """
    img, motif, truths = _build_oriented_scene()
    mh, mw = motif.shape[:2]
    sx, sy, _sw, _sh = truths["R0"]
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])

    m = Matcher(device="cpu")
    m.set_image(img)
    params = _default_params(method="ssd", enable_rotation=True, enable_flipping=True)
    results = m.match(template, params, exclude_box=truths["R0"])

    assert all(r.orientation in ORIENTATIONS for r in results)
    for code in ("R90", "R180", "MY"):
        hit = _find_hit(results, truths[code], params.threshold)
        assert hit is not None, f"[ssd] missed the {code} stamp at {truths[code]}"
        assert hit.orientation == code, (
            f"[ssd] {code} stamp tagged {hit.orientation!r} (expected {code!r})"
        )


# =========================================================================== #
# Feature-mode orientation test (cv2 importorskip; ORB is approximate).
# =========================================================================== #
def test_features_orientation_recovers_mirrored_copy() -> None:
    """method='features' with flipping ON recovers a MIRRORED copy and tags a reflection.

    The feature path classifies a recovered homography's linear part into the
    nearest D4 orientation (via ``orientation_from_matrix``), so a mirrored
    instance must come back tagged as *some* reflection (MX/MY/MXR90/MYR90). With
    flipping OFF the mirrored copy is NOT reported. ORB is approximate, so we
    assert the reflection family / presence, not an exact code. DESIGN §L / §J.3.
    """
    cv2 = pytest.importorskip("cv2", reason="feature matching requires OpenCV (cv2)")
    rng = np.random.default_rng(3)

    # A feature-rich, asymmetric motif: a dense field of small random-colour
    # blocks (hundreds of repeatable ORB keypoints) plus bold asymmetric
    # landmarks so a mirror is unambiguous to the homography.
    mh, mw = 180, 160
    motif = np.full((mh, mw, 3), 25, dtype=np.uint8)
    bs = 3
    cy_n, cx_n = mh // bs, mw // bs
    field = rng.integers(40, 256, size=(cy_n, cx_n, 3)).astype(np.uint8)
    motif[: cy_n * bs, : cx_n * bs] = np.repeat(np.repeat(field, bs, 0), bs, 1)
    motif[: max(4, mh // 6), : mw // 2] = (245, 30, 30)      # red top bar (left-biased)
    motif[: mh // 2, : max(4, mw // 10)] = (245, 30, 30)     # red left stub -> an L
    s = max(8, mh // 8)
    motif[int(mh * 0.30) : int(mh * 0.30) + s, int(mw * 0.55) : int(mw * 0.55) + s] = (255, 255, 255)

    # Background distinct from the motif palette, with faint texture.
    H, W = 460, 560
    base = rng.integers(96, 136, (H, W, 3)).astype(np.uint8)
    yy = np.linspace(0, 7 * np.pi, H)[:, None]
    xx = np.linspace(0, 7 * np.pi, W)[None, :]
    grating = (7 * np.sin(yy) * np.cos(xx)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + grating[:, :, None], 0, 255).astype(np.uint8)

    # Upright source copy (top-left) and a left-right MIRRORED copy (lower-right).
    src_x, src_y = 30, 28
    img[src_y : src_y + mh, src_x : src_x + mw] = motif
    src_box = (src_x, src_y, mw, mh)

    mir = np.ascontiguousarray(motif[:, ::-1])  # "MY" mirror
    mir_x, mir_y = src_x + mw + 110, src_y + mh + 50
    img[mir_y : mir_y + mh, mir_x : mir_x + mw] = mir
    mir_box = (mir_x, mir_y, mw, mh)

    template = np.ascontiguousarray(img[src_y : src_y + mh, src_x : src_x + mw])

    m = Matcher(device="cpu")
    m.set_image(img)
    base_params = dict(
        method="features",
        feature_detector="orb",
        feature_ratio=0.75,
        feature_min_inliers=10,
        feature_max_instances=20,
        threshold_floor=0.0,
    )

    def _mirror_hits(results):
        return [
            r
            for r in results
            if _iou((r.x, r.y, r.w, r.h), mir_box) >= 0.4
        ]

    # Flipping ON: the mirrored copy is recovered and tagged as SOME reflection.
    on = m.match(template, MatchParams(enable_flipping=True, **base_params), exclude_box=src_box)
    assert all(0.0 <= r.score <= 1.0 for r in on)
    hits_on = _mirror_hits(on)
    assert hits_on, (
        "feature matching with flipping ON did not recover the mirrored instance "
        f"(mir_box={mir_box}, results={[(r.x, r.y, r.w, r.h, r.orientation) for r in on]})"
    )
    assert any(r.orientation in REFLECTIONS for r in hits_on), (
        "the mirrored instance was found but not tagged as a reflection "
        f"(tags={[r.orientation for r in hits_on]})"
    )
    # Source region excluded on the feature path.
    for r in on:
        assert _iou((r.x, r.y, r.w, r.h), src_box) <= MatchParams(**base_params).exclude_iou

    # Flipping OFF: the mirrored copy must NOT be reported (no reflection search).
    off = m.match(template, MatchParams(enable_flipping=False, **base_params), exclude_box=src_box)
    assert not _mirror_hits(off), (
        "feature matching with flipping OFF unexpectedly reported the mirrored copy "
        f"(results={[(r.x, r.y, r.w, r.h, r.orientation) for r in off]})"
    )
