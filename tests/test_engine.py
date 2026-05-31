"""Engine correctness tests: IoU recall/precision, tile seams, multi-scale.

Targets DESIGN §C.5 / §D for ``fastmatch.engine.Matcher``:

    Matcher(*, device, compute_dtype, channel_mode, conv_backend, vram_fraction,
            core_tile=int|None)
    .set_image(np.ndarray)
    .match(template, params, *, exclude_box=None, cancel=None, progress=None)
        -> list[Match]   # score desc, source excluded, image-px, scale-1 TL

Strategy: build a small **synthetic** image with a distinctive asymmetric
coloured motif stamped at known, non-overlapping locations on a low-contrast
textured background, crop one instance as the template, and assert:
  * RECALL  — every other ground-truth instance is recovered within +-2px (IoU),
  * PRECISION — no spurious hit above ``threshold`` lands away from a truth box,
  * source exclusion — the cropped instance is not returned,
  * true hits score high.
Plus a tile-seam case (``core_tile`` forces a small compute tile under a motif
straddling a seam), a multi-scale case, and the documented flat-template error.

All images are tiny so the CPU-only torch build runs each test in well under a
second. The engine is written in parallel; if not importable yet the module is
skipped rather than erroring, keeping partial-build runs green.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastmatch.types import MatchParams

# Engine is a sibling module under parallel development; skip if not ready.
engine_mod = pytest.importorskip("fastmatch.engine", reason="engine not yet implemented")
Matcher = engine_mod.Matcher

# These engine tests are deliberately sized to run on the CPU-only torch build
# in well under a second each. The ``cpu`` marker (applied per test below) lets a
# CI lane select just the fast CPU path with ``-m cpu``. The integrator should
# register it once in the project's ``pytest.ini``/``pyproject.toml`` --
#     [tool.pytest.ini_options]
#     markers = ["cpu: fast engine tests that run on CPU"]
# -- which this task is not permitted to create (only these three test files are
# in scope). Until then pytest emits a harmless PytestUnknownMarkWarning; the
# marker is functional regardless and selection by ``-m cpu`` works today.
cpu = pytest.mark.cpu


# --------------------------------------------------------------------------- #
# Synthetic-image helpers (no dependency on loader.generate_sample so engine
# behaviour is pinned independently of the generator's exact internals).
# --------------------------------------------------------------------------- #
MOTIF_W, MOTIF_H = 24, 28  # asymmetric on purpose -> no accidental rotational symmetry


def _make_motif(rng: np.random.Generator) -> np.ndarray:
    """A distinctive, high-contrast, asymmetric RGB motif of size (MOTIF_H, MOTIF_W, 3)."""
    m = np.zeros((MOTIF_H, MOTIF_W, 3), dtype=np.uint8)
    # Bold colour blocks with a clear left/right and top/bottom asymmetry so a
    # mismatched or mirror placement scores poorly.
    m[:, :, 0] = 30
    m[: MOTIF_H // 2, : MOTIF_W // 2, 0] = 240          # top-left red
    m[: MOTIF_H // 2, MOTIF_W // 2 :, 1] = 220          # top-right green
    m[MOTIF_H // 2 :, : MOTIF_W // 3, 2] = 200          # bottom-left blue stripe
    m[MOTIF_H // 2 :, MOTIF_W // 3 :, :] = 255          # bottom-right white block
    # A small off-centre dark notch to break any remaining symmetry.
    m[2:6, MOTIF_W - 6 : MOTIF_W - 2, :] = 0
    # Faint texture so it is never a perfectly flat patch.
    m = np.clip(m.astype(np.int16) + rng.integers(-4, 5, m.shape), 0, 255).astype(np.uint8)
    return m


def _make_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Low-contrast textured noise background, distinct from the motif's colours."""
    base = rng.integers(96, 136, (h, w, 3)).astype(np.uint8)  # mid-grey-ish
    # A faint sine grating adds structure without high-contrast edges.
    yy = np.linspace(0, 6 * np.pi, h)[:, None]
    xx = np.linspace(0, 6 * np.pi, w)[None, :]
    grating = (8 * np.sin(yy) * np.cos(xx)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + grating[:, :, None], 0, 255).astype(np.uint8)
    return base


def _stamp(img: np.ndarray, motif: np.ndarray, positions: list[tuple[int, int]]) -> None:
    """Stamp ``motif`` at each ``(x, y)`` top-left position into ``img`` in place."""
    mh, mw = motif.shape[:2]
    for (x, y) in positions:
        img[y : y + mh, x : x + mw] = motif


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two half-open ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center_near(box: tuple[int, int, int, int], truth: tuple[int, int, int, int], tol: int) -> bool:
    """True if ``box``'s centre is within ``tol`` px of ``truth``'s centre on both axes."""
    bx = box[0] + box[2] / 2.0
    by = box[1] + box[3] / 2.0
    tx = truth[0] + truth[2] / 2.0
    ty = truth[1] + truth[3] / 2.0
    return abs(bx - tx) <= tol and abs(by - ty) <= tol


def _build_scene() -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """A 320x420 image with 5 motif copies at known non-overlapping positions.

    Returns ``(image, positions, motif)`` where positions are TL ``(x, y)``.
    """
    rng = np.random.default_rng(0)
    h, w = 320, 420
    img = _make_background(h, w, rng)
    motif = _make_motif(rng)
    # Spread out so motifs never overlap and NMS/exclusion can't conflate them.
    positions = [(20, 18), (200, 30), (60, 180), (300, 150), (150, 250)]
    _stamp(img, motif, positions)
    return img, positions, motif


def _default_params(**over) -> MatchParams:
    """MatchParams tuned for these small, high-SNR synthetic scenes."""
    kw = dict(threshold=0.85, threshold_floor=0.50, scales=(1.0,), max_results=100)
    kw.update(over)
    return MatchParams(**kw)


# --------------------------------------------------------------------------- #
# Core recall / precision / exclusion
# --------------------------------------------------------------------------- #
@cpu
def test_recall_precision_and_source_exclusion() -> None:
    """All other instances found within +-2px, no spurious high hits, source excluded."""
    img, positions, motif = _build_scene()
    mh, mw = motif.shape[:2]
    truth_boxes = [(x, y, mw, mh) for (x, y) in positions]

    src_idx = 0
    sx, sy = positions[src_idx]
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])
    exclude_box = (sx, sy, mw, mh)

    m = Matcher(device="cpu")
    m.set_image(img)
    params = _default_params()
    results = m.match(template, params, exclude_box=exclude_box)

    assert isinstance(results, list)
    # Sorted by score descending (DESIGN §C.5 / §D.5 step 5).
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= r.score <= 1.0 for r in results)

    kept = [r for r in results if r.score >= params.threshold]

    # RECALL: every non-source truth box has a kept hit within +-2px (IoU high).
    others = [truth_boxes[i] for i in range(len(truth_boxes)) if i != src_idx]
    for tb in others:
        hit = next(
            (
                r
                for r in kept
                if _center_near((r.x, r.y, r.w, r.h), tb, tol=2)
                and _iou((r.x, r.y, r.w, r.h), tb) >= 0.5
            ),
            None,
        )
        assert hit is not None, f"missed ground-truth instance at {tb}"
        assert hit.score >= params.threshold  # true hits score high

    # SOURCE EXCLUSION: nothing kept overlaps the source box (IoU > exclude_iou).
    for r in kept:
        assert _iou((r.x, r.y, r.w, r.h), exclude_box) <= params.exclude_iou, (
            "source region was not excluded"
        )

    # PRECISION: every kept hit corresponds to some real (non-source) instance.
    for r in kept:
        assert any(
            _center_near((r.x, r.y, r.w, r.h), tb, tol=2) for tb in others
        ), f"spurious hit at {(r.x, r.y, r.w, r.h)} score={r.score}"

    # No duplicate detections of the same instance after NMS: one kept hit each.
    assert len(kept) == len(others)


@cpu
def test_match_box_size_equals_template_at_scale_one() -> None:
    """Scale-1 hits report the template's pixel dimensions (DESIGN §D.4)."""
    img, positions, motif = _build_scene()
    mh, mw = motif.shape[:2]
    sx, sy = positions[0]
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])

    m = Matcher(device="cpu")
    m.set_image(img)
    results = m.match(template, _default_params(), exclude_box=(sx, sy, mw, mh))
    kept = [r for r in results if r.score >= 0.85]
    assert kept, "expected at least one strong hit"
    for r in kept:
        assert r.scale == pytest.approx(1.0)
        assert (r.w, r.h) == (mw, mh)


@cpu
def test_image_size_property() -> None:
    """image_size reports (H, W) of the staged image (DESIGN §C.5)."""
    img, _positions, _motif = _build_scene()
    m = Matcher(device="cpu")
    m.set_image(img)
    assert m.image_size == (img.shape[0], img.shape[1])


# --------------------------------------------------------------------------- #
# (a) Tile-seam: motif straddling a small forced compute-tile boundary.
# --------------------------------------------------------------------------- #
@cpu
def test_tile_seam_motif_found_exactly_once() -> None:
    """With a small core_tile, a seam-straddling motif is found exactly once.

    DESIGN §D.6: tiles overlap by template-1 (halo) and only detections centred
    in a tile's core are kept, so a motif sitting across a seam must be recovered
    once -- never lost, never double-counted.
    """
    rng = np.random.default_rng(1)
    h, w = 300, 300
    img = _make_background(h, w, rng)
    motif = _make_motif(rng)
    mh, mw = motif.shape[:2]

    core_tile = 64  # small -> guarantees multiple compute tiles + seams
    # Place a reference (source) copy and a second copy straddling a tile seam at
    # x ~= 2*core_tile = 128 so the motif spans the boundary on the x axis.
    src_pos = (10, 10)
    seam_x = 2 * core_tile - mw // 2  # centre near the 128 px seam
    seam_pos = (seam_x, 140)
    _stamp(img, motif, [src_pos, seam_pos])

    sx, sy = src_pos
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])

    m = Matcher(device="cpu", core_tile=core_tile)
    m.set_image(img)
    results = m.match(template, _default_params(), exclude_box=(sx, sy, mw, mh))

    seam_truth = (seam_pos[0], seam_pos[1], mw, mh)
    seam_hits = [
        r
        for r in results
        if r.score >= 0.85 and _center_near((r.x, r.y, r.w, r.h), seam_truth, tol=2)
    ]
    assert len(seam_hits) == 1, (
        f"seam-straddling motif should be found exactly once, got {len(seam_hits)}"
    )
    assert _iou((seam_hits[0].x, seam_hits[0].y, seam_hits[0].w, seam_hits[0].h), seam_truth) >= 0.5


# --------------------------------------------------------------------------- #
# (b) Multi-scale: a slightly-resized instance is found at the right scale.
# --------------------------------------------------------------------------- #
@cpu
def test_multi_scale_finds_resized_instance() -> None:
    """A ~1.25x-resized instance is recovered and reports a >1 scale.

    DESIGN §D.4: the template is scaled, candidates pooled, one global NMS picks
    the winning scale per location. We stamp an enlarged motif and assert the
    returned scale is the nearest grid scale above 1.0.
    """
    from PIL import Image

    rng = np.random.default_rng(2)
    h, w = 360, 360
    img = _make_background(h, w, rng)
    motif = _make_motif(rng)
    mh, mw = motif.shape[:2]

    # Source copy at native size.
    src_pos = (20, 20)
    _stamp(img, motif, [src_pos])

    # An enlarged copy (~1.25x) elsewhere, resized with PIL (no cv2 in env).
    factor = 1.25
    big = np.asarray(
        Image.fromarray(motif).resize(
            (round(mw * factor), round(mh * factor)), Image.BILINEAR
        )
    )
    bh, bw = big.shape[:2]
    big_pos = (180, 180)
    img[big_pos[1] : big_pos[1] + bh, big_pos[0] : big_pos[0] + bw] = big

    sx, sy = src_pos
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])

    m = Matcher(device="cpu")
    m.set_image(img)
    scales = (0.9, 1.0, 1.1, 1.25)
    params = _default_params(scales=scales)
    results = m.match(template, params, exclude_box=(sx, sy, mw, mh))

    big_center = (big_pos[0] + bw / 2.0, big_pos[1] + bh / 2.0)
    # Find the hit nearest the enlarged instance's centre.
    enlarged = [
        r
        for r in results
        if r.score >= 0.80
        and abs(r.x + r.w / 2.0 - big_center[0]) <= 3
        and abs(r.y + r.h / 2.0 - big_center[1]) <= 3
    ]
    assert enlarged, "did not recover the enlarged instance"
    best = max(enlarged, key=lambda r: r.score)
    # Winning scale is > 1 and is the closest grid value to the true 1.25x.
    assert best.scale > 1.0
    nearest = min(scales, key=lambda s: abs(s - factor))
    assert best.scale == pytest.approx(nearest)
    # Reported box size tracks the chosen scale (round(tw*s), round(th*s)).
    assert best.w == round(mw * best.scale)
    assert best.h == round(mh * best.scale)


# --------------------------------------------------------------------------- #
# (c) Flat / low-variance template -> documented ValueError (DESIGN §H1).
# --------------------------------------------------------------------------- #
@cpu
def test_flat_template_raises_value_error() -> None:
    """A near-constant template is rejected with ValueError (var below floor)."""
    img, _positions, _motif = _build_scene()
    m = Matcher(device="cpu")
    m.set_image(img)

    # A perfectly flat patch: std == 0, well under var_floor (1e-3 on [0,1]).
    flat = np.full((MOTIF_H, MOTIF_W, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError):
        m.match(flat, _default_params())


@cpu
def test_effective_device_is_cpu_when_forced() -> None:
    """Forcing device='cpu' resolves to a CPU torch device (DESIGN §C1)."""
    import torch

    m = Matcher(device="cpu")
    dev = m.effective_device
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"


# --------------------------------------------------------------------------- #
# cancel / progress callback plumbing (DESIGN §D.8)
# --------------------------------------------------------------------------- #
@cpu
def test_progress_and_cancel_callbacks() -> None:
    """progress(0..100) is reported and cancel() short-circuits the search."""
    img, positions, motif = _build_scene()
    mh, mw = motif.shape[:2]
    sx, sy = positions[0]
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])

    m = Matcher(device="cpu", core_tile=64)
    m.set_image(img)

    seen: list[int] = []
    m.match(
        template,
        _default_params(),
        exclude_box=(sx, sy, mw, mh),
        progress=seen.append,
    )
    if seen:  # progress is optional per-tile granularity, but must stay in range.
        assert all(0 <= p <= 100 for p in seen)
        assert max(seen) <= 100

    # An always-true cancel must return promptly without raising.
    out = m.match(
        template,
        _default_params(),
        exclude_box=(sx, sy, mw, mh),
        cancel=lambda: True,
    )
    assert isinstance(out, list)


# --------------------------------------------------------------------------- #
# (d) Selectable convolution methods: SSD and CCORR (DESIGN §J.1 / §J.2).
#
# The tiled compute, halo, multi-scale, local-max prefilter, threshold_floor,
# source-exclusion, IoU-NMS, top-K and cancel/progress are all shared with NCC;
# only the per-window score formula differs. These tests pin that the alternate
# score maps still recover every stamped instance with near-1.0 scores on exact
# copies and exclude the source -- i.e. the shared machinery is method-agnostic.
# --------------------------------------------------------------------------- #
@cpu
@pytest.mark.parametrize("method", ["ssd", "ccorr"])
def test_conv_methods_recall_and_source_exclusion(method: str) -> None:
    """method='ssd'/'ccorr' recover all other instances (IoU +-2px), source excluded.

    The motifs are stamped as *exact* copies of the source crop, so for both
    alternate score maps the true hits must sit at ~1.0 (SSD: rmse~0 -> 1-rmse~1;
    CCORR: cosine of identical patches ~1). DESIGN §J.2.
    """
    img, positions, motif = _build_scene()
    mh, mw = motif.shape[:2]
    truth_boxes = [(x, y, mw, mh) for (x, y) in positions]

    src_idx = 0
    sx, sy = positions[src_idx]
    template = np.ascontiguousarray(img[sy : sy + mh, sx : sx + mw])
    exclude_box = (sx, sy, mw, mh)

    m = Matcher(device="cpu")
    m.set_image(img)
    params = _default_params(method=method)
    results = m.match(template, params, exclude_box=exclude_box)

    assert isinstance(results, list)
    # Same public-contract invariants as NCC: sorted desc, scores in [0,1].
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= r.score <= 1.0 for r in results)

    kept = [r for r in results if r.score >= params.threshold]
    others = [truth_boxes[i] for i in range(len(truth_boxes)) if i != src_idx]

    # RECALL: every non-source truth box has a kept hit within +-2px (IoU >= 0.5),
    # and -- because every stamp is an exact copy of the source -- that hit scores
    # near 1.0 under the method's score map.
    for tb in others:
        hit = next(
            (
                r
                for r in kept
                if _center_near((r.x, r.y, r.w, r.h), tb, tol=2)
                and _iou((r.x, r.y, r.w, r.h), tb) >= 0.5
            ),
            None,
        )
        assert hit is not None, f"[{method}] missed ground-truth instance at {tb}"
        assert hit.score >= 0.95, f"[{method}] exact copy scored low: {hit.score}"

    # SOURCE EXCLUSION: nothing kept overlaps the source box past exclude_iou.
    for r in kept:
        assert _iou((r.x, r.y, r.w, r.h), exclude_box) <= params.exclude_iou, (
            f"[{method}] source region was not excluded"
        )

    # PRECISION expectations differ by method's discriminative power:
    #   * SSD subtracts nothing but its 1-RMSE score is sharply peaked at an exact
    #     copy, so (like NCC) we demand a clean one-hit-per-instance result.
    #   * CCORR (CCORR_NORMED) does NOT subtract the window mean, so a bright,
    #     high-energy motif can push the cosine of an unrelated textured window
    #     past the 0.85 default -- exactly why DESIGN §J.1 frames CCORR as a
    #     weaker "alternative correlation measure". We therefore only require that
    #     every TRUE instance is recovered (above), not the absence of extra hits.
    if method == "ssd":
        for r in kept:
            assert any(
                _center_near((r.x, r.y, r.w, r.h), tb, tol=2) for tb in others
            ), f"[{method}] spurious hit at {(r.x, r.y, r.w, r.h)} score={r.score}"
        assert len(kept) == len(others), f"[{method}] expected one hit per instance"
    else:  # ccorr: every real instance must still be among the kept hits.
        for tb in others:
            assert any(
                _center_near((r.x, r.y, r.w, r.h), tb, tol=2) for r in kept
            ), f"[{method}] real instance {tb} not among kept hits"


# --------------------------------------------------------------------------- #
# (e) Method tradeoff: a FLAT / low-variance template.
#
# NCC's variance normalization is unstable (and rejected up front) on a
# near-uniform patch, while SSD's squared-difference score is well-defined and
# recovers the flat instance. This is the concrete motivation for offering SSD
# as a selectable method (DESIGN §J.1 / §J.2: "SSD/CCORR do NOT raise on
# low-variance templates -- that's their use case").
# --------------------------------------------------------------------------- #
def _build_flat_scene() -> tuple[np.ndarray, list[tuple[int, int]], int, int]:
    """Textured background with near-uniform 'flat' patches stamped at known spots.

    The flat patch is a single solid colour that differs from the (textured)
    background mean, so SSD can localize it even though it has near-zero internal
    variance. Returns ``(image, positions, patch_w, patch_h)``.
    """
    rng = np.random.default_rng(7)
    h, w = 300, 360
    img = _make_background(h, w, rng)
    pw, ph = 26, 30
    # A solid mid-bright patch -- internally flat (std ~ 0), distinct from the
    # ~96-136 grey textured background so a squared-difference search has a clear
    # minimum at each stamp.
    flat = np.full((ph, pw, 3), 200, dtype=np.uint8)
    positions = [(30, 24), (210, 40), (90, 200), (260, 180)]
    for (x, y) in positions:
        img[y : y + ph, x : x + pw] = flat
    return img, positions, pw, ph


@cpu
def test_flat_template_ncc_raises_but_ssd_recovers() -> None:
    """NCC rejects a flat template (ValueError); SSD recovers the flat instances.

    Demonstrates the documented method tradeoff: NCC keeps the variance-floor
    guard (§H1) and refuses a featureless crop, whereas SSD is built for exactly
    this case and localizes the other near-uniform copies. DESIGN §J.2.
    """
    img, positions, pw, ph = _build_flat_scene()
    truth_boxes = [(x, y, pw, ph) for (x, y) in positions]

    src_idx = 0
    sx, sy = positions[src_idx]
    template = np.ascontiguousarray(img[sy : sy + ph, sx : sx + pw])
    exclude_box = (sx, sy, pw, ph)

    m = Matcher(device="cpu")
    m.set_image(img)

    # NCC: the near-zero-variance template is rejected up front (§H1).
    with pytest.raises(ValueError):
        m.match(template, _default_params(method="ncc"), exclude_box=exclude_box)

    # SSD: must NOT raise, and must recover the other flat instances. A lower
    # threshold is used because flat patches against textured noise score a touch
    # below the exact-motif case, but exact copies still sit near 1.0.
    params = _default_params(method="ssd", threshold=0.80)
    results = m.match(template, params, exclude_box=exclude_box)
    assert isinstance(results, list)
    assert all(0.0 <= r.score <= 1.0 for r in results)

    kept = [r for r in results if r.score >= params.threshold]
    others = [truth_boxes[i] for i in range(len(truth_boxes)) if i != src_idx]

    for tb in others:
        hit = next(
            (
                r
                for r in kept
                if _center_near((r.x, r.y, r.w, r.h), tb, tol=2)
                and _iou((r.x, r.y, r.w, r.h), tb) >= 0.5
            ),
            None,
        )
        assert hit is not None, f"[ssd] missed flat instance at {tb}"

    # Source exclusion still holds for the flat-friendly method.
    for r in kept:
        assert _iou((r.x, r.y, r.w, r.h), exclude_box) <= params.exclude_iou, (
            "[ssd] source region was not excluded"
        )
