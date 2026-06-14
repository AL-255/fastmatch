"""OpenCV feature-matching backend for FastMatch (``method=="features"``).

This module implements :class:`FeatureMatcher`, the warp-tolerant alternative to
the GPU convolution methods (§J.3 of ``docs/DESIGN.md``). Where NCC/SSD/CCORR
need the instances to be roughly aligned and the same scale, feature matching
detects ORB/AKAZE/SIFT keypoints, matches descriptors, and recovers the instance
geometry — so it finds copies that are **rotated, scaled, or perspective-warped**
which the fixed-window template methods miss.

OpenCV (``cv2``) is required **only** for this method. It is imported lazily (at
:class:`FeatureMatcher` construction, not at module import) so importing
:mod:`fastmatch.engine` never drags in cv2; if cv2 is unavailable the
constructor raises the documented :class:`RuntimeError` and the UI disables the
"features" option.

Coordinate convention matches the rest of FastMatch: integer **image pixels**,
origin top-left, +x right / +y down, half-open boxes ``(x, y, w, h)`` (§F.3).

Pipeline — feature-PROPOSE + appearance-VERIFY
----------------------------------------------
The matcher's job here is to **reproduce the matches NCC/SSD would find** even on
feature-poor, highly repetitive content. Two facts drive the design:

* The dominant historical failure was **feature-density starvation** — too few
  keypoints per small repeated motif, so most true copies matched nothing. The
  fix is a *dense* detector (a high per-tile keypoint budget and lowered ORB
  edge/fast thresholds), which alone takes per-copy keypoint support on the
  benchmark PCB from ~0 to dozens at negligible extra cost (detection is tiled,
  one-time, and cached).
* Fitting a per-instance **homography to a spatially-gathered correspondence
  cluster** is fragile on repetitive images: the cluster mixes correspondences
  from several adjacent copies, so RANSAC latches onto spurious cross-copy
  transforms. Instead we **propose** candidate poses cheaply and then **verify by
  appearance**, which is exactly the criterion (normalized cross-correlation)
  that produced the NCC/SSD ground truth.

Steps (see §J.3):

1. **Dense tiled detection** — image keypoints/descriptors are detected once per
   ``(image, detector, level)`` over overlapping full-resolution tiles (each tile
   with its own budget so features stay spatially dense) and cached on this
   matcher; :meth:`invalidate_image` clears the cache when a new image is staged.
2. **Propose** — for each active D4 orientation the template is transformed
   (:func:`apply_orientation`), detected, and its descriptors multi-matched to the
   image (keep ALL good matches, one per copy — NOT the Lowe ratio test, which
   rejects exactly the repeated instances). Each correspondence votes for an
   instance centre under a *similarity* transform (carrying the matched keypoints'
   scale and angle), so an arbitrary rotated/scaled copy still forms a peak. Every
   centre-histogram bin above a low vote floor becomes a candidate pose
   ``(cx, cy, scale, angle)`` — permissive: high proposal recall, precision comes
   from the verify step.
3. **Verify by appearance (ZNCC)** — proposals near a D4 orientation at scale ≈ 1
   (every aligned/mirrored PCB instance) take a pixel-exact **axis** path: the
   net-oriented template is zero-mean normalized-cross-correlated against the
   image with a small ±-px alignment search. Genuinely off-axis proposals
   (arbitrary rotation/scale — the generality case) take an **inverse-warp** path:
   the image window is warped back into the template frame and ZNCC'd, refined
   over a small (angle, scale) grid. A proposal is accepted iff its ZNCC clears
   the acceptance floor; the floor *is* the NCC/SSD appearance criterion, so this
   reproduces their matches with near-zero false positives.
4. **Finalize** — honour the orientation checkboxes (keep an instance only if its
   classified D4 orientation is active), source-exclude, drop lattice phantoms,
   IoU-NMS, cap at ``feature_max_instances``. The ZNCC is the match score.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .types import (
    Match,
    _ORIENT_LINEAR,
    active_orientations,
    apply_orientation,
    normalize_weights,
    orientation_from_matrix,
)

# Image feature detection runs at FULL resolution. Downsampling the image forced
# the template to be downsampled by the same factor to keep descriptors at a
# comparable scale, which decimated any modest selection (a 200 px box on a
# 12000 px image shrank to 50 px) below ORB's keypoint threshold — so feature
# matching found nothing on exactly the large images this app targets. Instead we
# detect at native scale and bound cost/memory by TILING the detection, giving
# each tile its own keypoint budget so features stay spatially dense everywhere
# (a single global cap would leave any one small instance with too few keypoints
# to match). Boxes are already in full-res coordinates — no scale-back needed.
_DETECT_TILE = 2048        # full-res detection tile edge (px)
_DETECT_OVERLAP = 96       # tiles overlap on every side so a keypoint's descriptor
                           # patch and near-seam features have full context; a
                           # keypoint is assigned to exactly one tile's core, so
                           # there are no duplicate descriptors at seams (which
                           # would defeat per-seam descriptor matching).

# DENSE detection budgets (the load-bearing density fix). The benchmark PCB's
# small repeated motifs (~116 px) carry very few distinctive corners; at the old
# 1500/tile budget many true copies matched ZERO keypoints, capping recall far
# below what NCC/SSD (which correlate raw pixels) achieve. Raising the per-tile
# budget ~8x and lowering ORB's edge/fast thresholds lifts per-copy keypoint
# support from ~0 to dozens at negligible extra cost (detection is one-time/cached:
# ~1s for ~131k keypoints over a 42 MP image), and is what makes the propose step
# see every copy.
_FEATURES_PER_TILE = 12000  # ORB/SIFT keypoint budget per detection tile
_TEMPLATE_FEATURES = 8000   # denser budget for the (small) template crop
_ORB_EDGE_THRESHOLD = 8     # ORB default 31 discards keypoints within 31 px of any
                            # border (zero keypoints on a small template); 8 keeps
                            # near-edge keypoints. Identical for image AND template
                            # so descriptors stay comparable.
_ORB_FAST_THRESHOLD = 5     # lowered so faint low-contrast texture still fires.
_MIN_TEMPLATE_SIDE = 12     # below this a template cannot anchor an instance

# Coarsest-safe-level detection pyramid (§K.2). Detecting features over a
# gigapixel image at full resolution is the dominant cost of the feature path.
# We *can* detect at the COARSEST pyramid level where the downsampled template
# still keeps enough texture to fire ORB. The level is chosen FROM THE TEMPLATE so
# the template never decimates below ``MIN_TPL_FEATURE`` px on its shorter side —
# that exact over-decimation is the bug that made feature matching return nothing
# on large images, so this bound is load-bearing, NOT a tunable.
#
#   L = clamp(floor(log2(min(th, tw) / MIN_TPL_FEATURE)), 0, MAX_FEATURE_LEVEL)
#
# L == 0 reproduces full-resolution tiled detection exactly (small templates /
# small images are unchanged), so the small-image behaviour and the existing
# tests are preserved.
_MIN_TPL_FEATURE = 96      # coarsest level keeps the template's short side ≥ ~96 px
# Downsampling for detection is a SPEED optimization but silently drops keypoints,
# so we default the feature path to L=0 (full res) for reliability; set this > 0
# only to trade recall for first-query detection speed on very large templates.
_MAX_FEATURE_LEVEL = 0     # 0 -> always full resolution (see note above)

#: Multi-instance descriptor matching. The classic Lowe ratio test rejects a
#: match whose 2nd-nearest neighbour is (nearly) as close — which is EXACTLY what
#: happens for repeated/identical instances: each template feature recurs once per
#: copy, so its nearest neighbours (the copies) are all equally close and the ratio
#: test discards them, finding nothing. So instead of the ratio test we keep ALL
#: image matches within an absolute "good descriptor" distance of each template
#: feature (one correspondence per copy) and let the appearance verify reject the
#: spurious ones. ``_GOOD_DIST`` is the per-detector max descriptor distance for a
#: "good" match (random ORB Hamming ≈ 128; an exact copy ≈ 0); ``_KNN_K`` caps the
#: nearest-neighbour fan-out on repetitive texture (covers up to this many copies).
_GOOD_DIST = {
    "orb": 64.0,    # Hamming on the 256-bit ORB descriptor
    "akaze": 90.0,  # Hamming on AKAZE's (longer) binary descriptor
    "sift": 300.0,  # L2 on 128-float SIFT
}
_KNN_K = 96

#: Proposal vote floor: minimum correspondences in a centre-histogram bin for it
#: to become a candidate pose. Deliberately LOW — precision comes from the ZNCC
#: appearance gate, not the vote count, so faint copies (few keypoints) still get
#: a chance to be verified. ``feature_min_inliers`` caps this (default 12 -> 4).
_PROPOSAL_VOTE_FLOOR = 4

#: Global cap on proposals verified per query (after dedup). The cheap axis
#: centre-prefilter makes a generous cap affordable; some true copies are low-vote
#: so the cap stays high. Bounds worst-case verify cost on a pathological image.
_MAX_PROPOSALS = 3000

#: Appearance acceptance (zero-mean normalized cross-correlation). This floor IS
#: the NCC/SSD criterion that produced the ground truth, so it is what makes the
#: feature path reproduce their matches with near-zero false positives.
_NCC_ACCEPT = 0.70          # axis (D4 / scale~1) path acceptance
_NCC_ACCEPT_WARP = 0.70     # off-axis (arbitrary rotation/scale) path acceptance
#: Off-axis support gate: a genuinely warped instance must have at least this much
#: vote consensus before we pay for its (lossy, FP-prone) inverse-warp refine. The
#: PCB ground truth is all D4 and never takes this path, so it does not affect PCB
#: false positives. Reinterprets ``feature_min_inliers``.
_WARP_MIN_VOTES = 12
#: Skip the off-axis refine grid if the vote-median pose's centre ZNCC is this far
#: below the floor (cheap rejection of spurious warps).
_WARP_PREFILTER = 0.40
#: A residual rotation within this many degrees of a 90° multiple is treated as an
#: exact D4 turn (verified pixel-exactly via the axis path), NOT an arbitrary
#: rotation — keeps PCB D4 copies out of the resampling-lossy warp path.
_D4_ANGLE_TOL_DEG = 12.0
#: ± integer-px local alignment search on the axis path (the vote-median centre is
#: a few px noisy). ±2 -> 25 evals (was ±3 -> 49); the centre-prefilter below makes
#: the search cost only matter for promising proposals.
_NCC_SEARCH = 2
#: Axis centre-prefilter margin: reject a proposal in a single ZNCC if its centre
#: correlation is below ``_NCC_ACCEPT - this`` (a true copy within ±_NCC_SEARCH px
#: cannot recover from so low a centre, while spurious proposals sit near 0 and are
#: rejected cheaply). The dominant runtime win on large templates.
_AXIS_PREFILTER_MARGIN = 0.30

#: PERSPECTIVE fallback (the homography verify path). When a vote-rich proposal
#: fails the cheaper similarity paths, a per-proposal RANSAC homography is fit to
#: that proposal's OWN clean correspondence cluster and the image is warped back
#: through it (cv2.warpPerspective) for the SAME ZNCC>=_NCC_ACCEPT_WARP gate. Only
#: reachable for off-axis/failed-axis proposals with >= warp_min_votes support, so
#: the D4/PCB cases (axis-success) never enter it; the ZNCC gate + the geometric
#: guards below keep precision (~0 FP) and prevent repetitive-lattice phantoms.
_REPROJ_THRESH = 5.0        # RANSAC reprojection tolerance (px)
_RANSAC_ITERS = 3000
_MAX_CLUSTER_PTS = 600      # subsample a huge cluster before findHomography (speed)
_HOMOG_MIN_PTS = 8          # need > the 4-pt minimum so an 8-DoF fit is over-determined
_HOMOG_MAX_BOTTOM = 0.01    # |H[2,0]|,|H[2,1]| ceiling (px^-1): reject wild projective blow-ups
_MIN_AFFINE_DET = 1e-6      # reject near-degenerate / mirror-flipped affine parts
_MIN_SCALE = 1.0 / 1000.0   # sane linear-scale bounds on the recovered transform
_MAX_SCALE = 1000.0

#: IMAGE-SPACE CLUSTER homography fallback (the perspective recall fix). The
#: similarity-vote PROPOSE step scatters a single perspective copy into dozens of
#: conflicting (wrong-scale/angle) poses, so the per-proposal peak-bin cluster that
#: feeds :meth:`_verify_homography` is starved/contaminated and RANSAC settles on a
#: degenerate consensus. A perspective copy's IMAGE keypoints, however, stay
#: physically clustered regardless of the warp. So when every cheaper path
#: (axis / affine-warp / peak-bin homography) has REJECTED a vote-rich proposal,
#: gather ALL of that orientation's good correspondences whose IMAGE position lies
#: in a box around the proposal centre (one complete, high-inlier cluster), RANSAC
#: a homography on it, and apply the SAME ZNCC>=_NCC_ACCEPT_WARP gate. This is a
#: last-resort SUPPLEMENT: the gate + geometry guards keep precision (FP=0) and the
#: vote/centre gating keep it off the D4/PCB axis-successes.
_CLUSTER_RADIUS_FACTOR = 1.2  # half-box edge = this * max(th, tw) around the centre
_CLUSTER_MIN_PTS = 12         # min correspondences in the image-space box to attempt
#: The image-space cluster is COMPLETE (a copy's whole, high-inlier-ratio match set),
#: so RANSAC converges in a handful of samples — far fewer iterations than the
#: peak-bin path's contaminated cluster needs, and a tighter point cap keeps each fit
#: cheap. These bound the cluster path's cost so it does not blow up the bench.
_CLUSTER_RANSAC_ITERS = 200
_CLUSTER_MAX_PTS = 140
#: The peak-bin homography's ``_project_to_box`` rejects a NEGATIVE linear-2x2
#: determinant as a mirror-flip. Under real perspective the homography's linear
#: block can carry a negative det even for an orientation-PRESERVING copy (the
#: foreshortening sign lives in the projective terms, not the 2x2), so the
#: image-space cluster path validates orientation by the SIGNED AREA of the
#: projected quad instead (positive == not mirrored) — the same flip test applied
#: where it is geometrically correct. ``_project_to_box`` is left untouched so the
#: existing peak-bin path's (correct, FP-guarding) rejections are unchanged.


class FeatureMatcher:
    """Dense-feature propose + appearance-verify multi-instance matcher (§J.3).

    One instance is reused across queries by the engine's :class:`Matcher`. It
    caches the image keypoints/descriptors keyed by ``(detector, level)`` so
    repeated queries on the same staged image only re-detect the (small) template.
    :meth:`invalidate_image` is called by ``Matcher.set_image`` when a new image
    is staged.

    OpenCV is imported in :meth:`__init__`; if it is missing the documented
    :class:`RuntimeError` is raised there.
    """

    def __init__(self) -> None:
        """Construct the matcher, requiring OpenCV.

        Raises:
            RuntimeError: if ``cv2`` cannot be imported (the exact message the
                spec mandates so the UI can disable the option, §J.3).
        """
        try:
            import cv2  # noqa: F401  (kept as the module-level handle below)
        except Exception as exc:  # ImportError or a broken cv2 install
            raise RuntimeError(
                "Feature matching requires OpenCV: pip install opencv-python-headless"
            ) from exc

        self._cv2 = cv2
        # Cache of {(detector_name, level): (kps, desc)} for the staged image —
        # FULL-RES keypoint coordinates (absolute image coords, already ×2^level)
        # and their descriptors, detected on the image downsampled by 2^level
        # (§K.2). Cleared by invalidate_image().
        self._image_cache: dict[
            tuple[str, int], tuple[list[Any], np.ndarray | None]
        ] = {}
        self._image_shape: tuple[int, int] | None = None  # (H, W) at full res
        # Cheap fingerprint of the image the cache was built against. Defense in
        # depth: the engine calls invalidate_image() in set_image, but if a caller
        # reuses this matcher on a *different* array without invalidating, a stale
        # fingerprint mismatch forces a re-detect (§U2).
        self._image_fingerprint: tuple[Any, ...] | None = None
        # Test/diagnostic toggle: force the full-resolution path (L=0) regardless
        # of the template-selected level. Off by default.
        self._force_full_level: bool = False
        # The level chosen for the most recent query (read by tests/benchmarks).
        self._last_level: int | None = None
        # Cached host YCbCr image (uint8 H,W,3) for ycbcr-mode verification, plus
        # the id() of the rgb array it was derived from (so a new staged image's
        # rgb forces a rebuild). Cleared by invalidate_image().
        self._ycbcr_host_cache: np.ndarray | None = None
        self._ycbcr_host_src_id: int | None = None

    # -- cache management ----------------------------------------------------

    def invalidate_image(self) -> None:
        """Drop any cached per-image keypoints/descriptors (§J.3/§J.4).

        Called by ``Matcher.set_image`` so a freshly staged image is re-detected
        on the next feature query rather than reusing stale features.
        """
        self._image_cache.clear()
        self._image_shape = None
        self._image_fingerprint = None
        self._ycbcr_host_cache = None
        self._ycbcr_host_src_id = None

    # -- colour-space helpers ------------------------------------------------

    @staticmethod
    def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
        """RGB ``(H, W, 3)`` uint8 -> BT.601 full-range Y,Cb,Cr ``(H, W, 3)`` uint8.

        Matches the desktop engine's ycbcr staging so the feature path's appearance
        verification scores in the same colour space the conv methods do.
        """
        a = rgb.astype(np.float32)
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
        out = np.stack([y, cb, cr], axis=-1)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    def _ycbcr_host(self, image_rgb: np.ndarray) -> np.ndarray:
        """Cached host YCbCr conversion of the full RGB image (for ycbcr verify).

        Reuses the result across queries on the same staged image; rebuilds when a
        different rgb array is passed (id mismatch) or after :meth:`invalidate_image`.
        """
        if self._ycbcr_host_cache is not None and self._ycbcr_host_src_id == id(image_rgb):
            return self._ycbcr_host_cache
        self._ycbcr_host_cache = self._rgb_to_ycbcr(image_rgb)
        self._ycbcr_host_src_id = id(image_rgb)
        return self._ycbcr_host_cache

    # -- detector factory ----------------------------------------------------

    def _make_detector(self, name: str, nfeatures: int | None = None):
        """Build the requested OpenCV detector and its descriptor norm.

        Returns ``(detector, norm)`` where ``norm`` is the BFMatcher distance for
        that descriptor type: HAMMING for the binary ORB/AKAZE descriptors, L2
        for SIFT's float descriptors (§J.3 step 3).

        ``nfeatures`` caps the keypoint budget: per-tile for image detection and
        higher for the small template crop. ``None`` uses the per-tile default.

        Raises:
            ValueError: on an unknown detector name.
        """
        cv2 = self._cv2
        n = (name or "orb").lower()
        nf = _FEATURES_PER_TILE if nfeatures is None else int(nfeatures)
        if n == "orb":
            # Dense, low-threshold ORB: a high feature budget plus a reduced
            # ``edgeThreshold`` (default 31 discards every keypoint within 31 px of
            # a border — zero on a small template) and ``fastThreshold`` (so faint
            # texture still fires). IDENTICAL for image and template so descriptors
            # are comparable.
            return (
                cv2.ORB_create(
                    nfeatures=nf,
                    edgeThreshold=_ORB_EDGE_THRESHOLD,
                    fastThreshold=_ORB_FAST_THRESHOLD,
                ),
                cv2.NORM_HAMMING,
            )
        if n == "akaze":
            # AKAZE has no feature-count cap; per-tile detection bounds it instead.
            return cv2.AKAZE_create(), cv2.NORM_HAMMING
        if n == "sift":
            return cv2.SIFT_create(nfeatures=nf), cv2.NORM_L2
        raise ValueError(f"unknown feature_detector {name!r}; expected orb/akaze/sift")

    @staticmethod
    def _select_level(th: int, tw: int) -> int:
        """Coarsest pyramid level that keeps the template usable (§K.2).

        ``L = clamp(floor(log2(min(th, tw) / MIN_TPL_FEATURE)), 0,
        MAX_FEATURE_LEVEL)``. ``L == 0`` for any template whose short side is
        below ``2 * MIN_TPL_FEATURE``, reproducing full-resolution behaviour on
        small templates/images. With ``_MAX_FEATURE_LEVEL == 0`` this is always 0.
        """
        short = max(1, min(int(th), int(tw)))
        if short < 2 * _MIN_TPL_FEATURE:
            return 0
        level = int(np.floor(np.log2(short / _MIN_TPL_FEATURE)))
        return max(0, min(level, _MAX_FEATURE_LEVEL))

    @staticmethod
    def _fingerprint(image_gray: np.ndarray) -> tuple[Any, ...]:
        """Cheap identity fingerprint of a grayscale image for cache validation.

        Combines the shape, dtype, and a handful of evenly-spaced sampled pixels
        (constant work regardless of image size) so a *different* image — even one
        with the same shape — is detected without hashing the whole gigapixel
        buffer. Used only for defense-in-depth cache invalidation (§U2).
        """
        flat = image_gray.reshape(-1)
        n = flat.size
        if n == 0:
            sample: tuple[int, ...] = ()
        else:
            idx = np.linspace(0, n - 1, num=min(16, n), dtype=np.intp)
            sample = tuple(int(v) for v in flat[idx])
        return (image_gray.shape, str(image_gray.dtype), sample)

    def _detect_tiled(
        self,
        gray: np.ndarray,
        detector_name: str,
        cancel: Callable[[], bool] | None,
        level: int = 0,
    ) -> tuple[list[Any] | None, np.ndarray | None]:
        """Detect features over a (possibly downsampled) image in overlapping tiles.

        When ``level > 0`` the image is first downsampled by ``2^level`` with
        ``cv2.INTER_AREA`` and detection runs on that smaller image; every kept
        keypoint's coordinate is then multiplied by ``2^level`` so the cache always
        stores **full-resolution** keypoint coordinates (§K.2). ``level==0`` is
        full-resolution tiled detection.

        Each tile is detected with its own keypoint budget so features stay
        spatially uniform across a gigapixel image. Tiles overlap by
        ``_DETECT_OVERLAP``; each keypoint is assigned to exactly one tile's core,
        so no duplicate descriptors land at seams. Returns ``(kps, desc)``, or
        ``(None, None)`` if ``cancel()`` fired mid-detection.
        """
        cv2 = self._cv2
        factor = 1 << int(level)  # 2^level
        if factor > 1:
            fh, fw = gray.shape[:2]
            dw = max(1, fw // factor)
            dh = max(1, fh // factor)
            det_img = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            det_img = gray
        h, w = det_img.shape[:2]
        detector, _norm = self._make_detector(detector_name, nfeatures=_FEATURES_PER_TILE)
        step = _DETECT_TILE
        ov = _DETECT_OVERLAP
        all_kp: list[Any] = []
        all_desc: list[np.ndarray] = []
        for ty in range(0, h, step):
            for tx in range(0, w, step):
                if cancel is not None and cancel():
                    return None, None
                ox0, oy0 = max(0, tx - ov), max(0, ty - ov)
                ox1, oy1 = min(w, tx + step + ov), min(h, ty + step + ov)
                sub = np.ascontiguousarray(det_img[oy0:oy1, ox0:ox1])
                kp, desc = detector.detectAndCompute(sub, None)
                if desc is None or kp is None or len(kp) == 0:
                    continue
                cx1, cy1 = min(w, tx + step), min(h, ty + step)
                keep: list[int] = []
                for i, k in enumerate(kp):
                    ax, ay = k.pt[0] + ox0, k.pt[1] + oy0
                    if tx <= ax < cx1 and ty <= ay < cy1:
                        k.pt = (ax * factor, ay * factor)
                        keep.append(i)
                if not keep:
                    continue
                all_kp.extend(kp[i] for i in keep)
                all_desc.append(desc[keep])
        if not all_desc:
            return [], None
        return all_kp, np.vstack(all_desc)

    # -- image-feature cache -------------------------------------------------

    def _ensure_image_features(
        self,
        image_gray: np.ndarray,
        detector_name: str,
        level: int,
        cancel: Callable[[], bool] | None,
    ) -> tuple[list[Any], np.ndarray | None]:
        """Detect+cache the image's keypoints/descriptors per ``(detector, level)``.

        Runs the tiled detector once per ``(image, detector, level)`` and caches
        the result so subsequent queries with the same detector AND level reuse it.
        Keypoints are stored in absolute **full-resolution** image coordinates.

        Returns ``(image_kps, image_desc)``.
        """
        key = ((detector_name or "orb").lower(), int(level))
        # Defense-in-depth (§U2): if the image array differs from the one the cache
        # was built against, drop every cached detector/level and re-detect.
        fingerprint = self._fingerprint(image_gray)
        if self._image_fingerprint is not None and fingerprint != self._image_fingerprint:
            self._image_cache.clear()
            self._image_shape = None
        self._image_fingerprint = fingerprint

        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_shape = image_gray.shape[:2]
            return cached

        h, w = image_gray.shape[:2]
        self._image_shape = (h, w)

        kps, desc = self._detect_tiled(image_gray, key[0], cancel, level=level)
        if kps is None:
            # Cancelled mid-detection — return empty without caching the partial.
            return [], None
        result = (kps, desc)
        self._image_cache[key] = result
        return result

    # -- public match --------------------------------------------------------

    def match(
        self,
        template: np.ndarray,
        image_gray: np.ndarray,
        params: "Any",
        *,
        image_rgb: np.ndarray | None = None,
        exclude_box: tuple[int, int, int, int] | None = None,
        cancel: Callable[[], bool] | None = None,
        progress: Callable[[int], None] | None = None,
        scale_back: int = 1,
    ) -> list[Match]:
        """Find every (possibly warped) instance of ``template`` in the image.

        Keypoint **detection** is always on luminance (ORB/AKAZE/SIFT are
        grayscale); ``channel_mode`` only affects the **appearance verification**.
        With ``channel_mode == "rgb"`` and an ``image_rgb`` provided (and an RGB
        template), each proposal is verified with a per-channel-summed NCC over the
        three colour channels (mirroring the conv path's rgb rule); otherwise
        verification is single-channel luminance.

        Args:
            template: ``(th, tw, 3)`` uint8 RGB or ``(th, tw)`` uint8 grayscale
                template crop. Detection uses its luminance; rgb-mode verification
                uses its colour channels.
            image_gray: The staged full-image grayscale ``(H, W)`` uint8 the engine
                passes (its BT.601 luminance). Detection is cached on it; luminance
                verification reads it directly.
            params: A ``MatchParams``. Read: ``feature_detector``, ``feature_ratio``,
                ``feature_min_inliers`` (now the proposal/off-axis vote gate),
                ``feature_max_instances``, ``exclude_iou``, ``nms_iou``,
                ``channel_mode``, ``enable_rotation``/``enable_flipping``.
                ``scales`` is ignored (§J.3).
            image_rgb: Optional staged full-image RGB ``(H, W, 3)`` uint8 (same
                pixel frame as ``image_gray``); used only for rgb-mode appearance
                verification. ``None`` -> luminance verification.
            exclude_box: ``(x, y, w, h)`` source region (full-res) to exclude.
            cancel: Polled during detection and the verify loop.
            progress: Called with ``0..100`` (~0-40 detection, ~55-100 verify).
            scale_back: Extra integer factor mapping image coordinates to the
                caller's full-resolution frame (1 when the engine passes full-res).

        Returns:
            ``list[Match]`` sorted by score descending, source region excluded, in
            the caller's full-resolution image px. ``score`` is the verified ZNCC
            (a confidence in ``[0, 1]``); ``scale`` ≈ ``sqrt(bbox_area /
            template_area)``.

        Raises:
            RuntimeError: if OpenCV was unavailable (raised at construction).
        """
        cv2 = self._cv2
        detector_name = (getattr(params, "feature_detector", "orb") or "orb")
        ratio = float(getattr(params, "feature_ratio", 0.75))
        min_inliers = int(getattr(params, "feature_min_inliers", 12))
        # Proposal floor is deliberately LOW (precision is the ZNCC gate, not the
        # vote count); feature_min_inliers only caps it down from its default.
        vote_floor = max(2, min(min_inliers, _PROPOSAL_VOTE_FLOOR))
        # The off-axis (arbitrary-warp) path needs real vote consensus; that is
        # what feature_min_inliers now governs (default 12 == _WARP_MIN_VOTES).
        warp_min_votes = max(4, min_inliers)
        max_instances = int(getattr(params, "feature_max_instances", 100))
        exclude_iou = float(getattr(params, "exclude_iou", 0.30))
        nms_iou = float(getattr(params, "nms_iou", 0.30))
        channel_mode = getattr(params, "channel_mode", "luminance") or "luminance"
        rgb_weights = getattr(params, "rgb_weights", (1.0 / 3, 1.0 / 3, 1.0 / 3))
        ycbcr_weights = getattr(params, "ycbcr_weights", (1.0 / 3, 1.0 / 3, 1.0 / 3))

        # Orientation search (dihedral D4, §J): the template is proposed under each
        # ACTIVE orientation, and each verified instance is kept only if its
        # classified net orientation is active — so rotation OFF rejects rotated
        # fits and flip OFF rejects reflections. With both off this is R0 only.
        active = set(
            active_orientations(
                getattr(params, "enable_rotation", False),
                getattr(params, "enable_flipping", False),
            )
        )

        if progress is not None:
            progress(0)
        if cancel is not None and cancel():
            return []

        # --- 0. select the coarsest-safe detection level from the template ---
        tmpl_gray_full = self._to_gray(template)
        th_full, tw_full = tmpl_gray_full.shape[:2]
        if min(th_full, tw_full) < _MIN_TEMPLATE_SIDE:
            return []  # too small to anchor an instance
        level = 0 if self._force_full_level else self._select_level(th_full, tw_full)
        self._last_level = level

        # Verification colour space (detection always stays on luminance below).
        # rgb/ycbcr need a colour template AND a matching colour image (same pixel
        # frame); otherwise fall back to single-channel luminance. The verify step
        # is colour-blind: it takes the verify image + template in the chosen space
        # plus per-channel weights, and _zncc handles 1- or 3-channel weighted NCC.
        colour_ok = (
            image_rgb is not None
            and template.ndim == 3
            and image_rgb.ndim == 3
            and image_rgb.shape[:2] == image_gray.shape[:2]
        )
        verify_weights: list[float] = [1.0]
        is_multi = False
        if channel_mode == "rgb" and colour_ok:
            tmpl_verify_full = np.ascontiguousarray(template)
            verify_image = image_rgb
            verify_weights = list(normalize_weights(rgb_weights, 3))
            is_multi = True
        elif channel_mode == "ycbcr" and colour_ok:
            # BT.601 full-range conversion of both the image (cached) and template.
            verify_image = self._ycbcr_host(image_rgb)
            tmpl_verify_full = self._rgb_to_ycbcr(np.ascontiguousarray(template))
            verify_weights = list(normalize_weights(ycbcr_weights, 3))
            is_multi = True
        else:
            tmpl_verify_full = tmpl_gray_full
            verify_image = image_gray

        # --- 1. image features at level L (cached per (detector, level)) -----
        img_kps, img_desc = self._ensure_image_features(
            image_gray, detector_name, level, cancel
        )
        if progress is not None:
            progress(40)
        if cancel is not None and cancel():
            return []
        if img_desc is None or len(img_kps) < vote_floor:
            return []  # nothing detectable in the image for this detector

        full_h, full_w = self._image_shape  # type: ignore[misc]
        # Vectorized image keypoint geometry (built once; reused by every variant).
        ik_xy = np.array([k.pt for k in img_kps], np.float32).reshape(-1, 2)
        ik_sz = np.array([k.size for k in img_kps], np.float32)
        ik_ang = np.array([k.angle for k in img_kps], np.float32)

        # Oriented templates, built once per active orientation and reused. The
        # GRAY set drives keypoint detection (proposals); the VERIFY set is the
        # same arrays in luminance mode, or the colour template in rgb mode. The
        # verify step lazily adds any net-D4 code a proposal snaps to.
        tmpl_gray_oriented: dict[str, np.ndarray] = {"R0": tmpl_gray_full}
        for o in active:
            if o not in tmpl_gray_oriented:
                tmpl_gray_oriented[o] = np.ascontiguousarray(
                    apply_orientation(tmpl_gray_full, o)
                )
        if is_multi:
            tmpl_verify_oriented: dict[str, np.ndarray] = {"R0": tmpl_verify_full}
            for o in active:
                if o not in tmpl_verify_oriented:
                    tmpl_verify_oriented[o] = np.ascontiguousarray(
                        apply_orientation(tmpl_verify_full, o)
                    )
        else:
            tmpl_verify_oriented = tmpl_gray_oriented

        # "good descriptor" distance, widened/narrowed by feature_ratio.
        maxd = _GOOD_DIST.get(detector_name.lower(), 64.0) * max(0.25, ratio / 0.75)
        _det_unused, norm = self._make_detector(detector_name)

        # --- 2. PROPOSE across active orientations ---------------------------
        proposals: list[tuple] = []
        # Per-orientation COMPLETE correspondence sets, populated by the propose step
        # and consumed only by the image-space cluster homography fallback (§J.3).
        orient_corr: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for orient in active:
            if cancel is not None and cancel():
                return []
            proposals.extend(
                self._propose_for_variant(
                    tmpl_gray_oriented[orient], orient, level, detector_name, norm, maxd,
                    ik_xy, ik_sz, ik_ang, img_desc, full_h, full_w, vote_floor,
                    corr_out=orient_corr,
                )
            )
        if progress is not None:
            progress(55)
        # Dedupe across variants (a true copy can split votes across adjacent bins
        # / orientations) and cap so the verify set — and its cost — stays bounded.
        cell = max(8, int(min(tw_full, th_full) // 4))
        proposals = self._dedupe_proposals(proposals, cell)
        proposals.sort(key=lambda p: -p[4])  # highest-vote first
        if len(proposals) > _MAX_PROPOSALS:
            proposals = proposals[:_MAX_PROPOSALS]

        # --- 3. VERIFY by appearance (ZNCC) ----------------------------------
        total_f = max(1, int(scale_back))
        tmpl_area = float(th_full * tw_full)
        results: list[Match] = []
        n = max(1, len(proposals))
        for i, p in enumerate(proposals):
            if cancel is not None and (i & 63) == 0 and cancel():
                break
            r = self._verify(
                p, tmpl_verify_oriented, verify_image, full_h, full_w, warp_min_votes,
                verify_weights, orient_corr,
            )
            if r is None:
                continue
            box, score, code = r
            # Checkbox filter: keep only instances whose net orientation is active.
            if code not in active:
                continue
            bx_s, by_s, bw_s, bh_s = box
            rx = max(0, min(int(round(bx_s * total_f)), full_w * total_f))
            ry = max(0, min(int(round(by_s * total_f)), full_h * total_f))
            rw = max(1, int(round(bw_s * total_f)))
            rh = max(1, int(round(bh_s * total_f)))
            area = float(rw * rh)
            scale = float(np.sqrt(area / tmpl_area)) if tmpl_area > 0 else 1.0
            results.append(
                Match(x=rx, y=ry, w=rw, h=rh, score=float(score), scale=scale,
                      orientation=code)
            )
            if progress is not None and (i & 63) == 0:
                progress(min(99, 55 + int(44.0 * i / n)))

        if progress is not None:
            progress(100)

        # --- 4. finalize: source exclusion / phantom-drop / NMS / cap --------
        if exclude_box is not None:
            results = [
                r for r in results if not self._excluded(r, exclude_box, exclude_iou)
            ]
        results = self._drop_lattice_phantoms(results)
        results.sort(key=lambda r: r.score, reverse=True)
        results = self._nms(results, nms_iou)
        if len(results) > max_instances:
            results = results[:max_instances]
        return results

    # -- proposal generation -------------------------------------------------

    def _propose_for_variant(
        self,
        tmpl_oriented: np.ndarray,
        orient: str,
        level: int,
        detector_name: str,
        norm: int,
        maxd: float,
        ik_xy: np.ndarray,
        ik_sz: np.ndarray,
        ik_ang: np.ndarray,
        img_desc: np.ndarray,
        full_h: int,
        full_w: int,
        vote_floor: int,
        corr_out: "dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None" = None,
    ) -> list[tuple]:
        """Similarity-transform centre vote -> candidate poses for one variant.

        Detects the oriented template (at level L, scaled back to full-res coords),
        multi-matches its descriptors to the image (keep ALL good matches, one per
        copy), and votes for an instance centre under a similarity transform that
        carries each matched keypoint's scale and angle (so an arbitrary
        rotated/scaled copy still forms a peak — generality). Returns a list of
        ``(cx, cy, scale, angle_rad, votes, orient, th, tw)`` proposals (the bin
        members' MEDIAN pose, robust to contaminating neighbour votes); precision
        is left to the appearance verify.
        """
        cv2 = self._cv2
        factor = 1 << int(level)
        th, tw = tmpl_oriented.shape[:2]
        if min(th, tw) < _MIN_TEMPLATE_SIDE:
            return []

        # Detect the template downsampled by 2^level (INTER_AREA) so its descriptors
        # are scale-consistent with the level-L image detection, then scale the
        # keypoints back to full-res coords. At level 0 this is full-res detection.
        if factor > 1:
            tdw, tdh = max(1, tw // factor), max(1, th // factor)
            tmpl_det = cv2.resize(tmpl_oriented, (tdw, tdh), interpolation=cv2.INTER_AREA)
        else:
            tmpl_det = tmpl_oriented
        detector, _ = self._make_detector(detector_name, nfeatures=_TEMPLATE_FEATURES)
        tk, td = detector.detectAndCompute(np.ascontiguousarray(tmpl_det), None)
        tk = list(tk) if tk is not None else []
        if td is None or len(tk) < vote_floor:
            return []
        if factor > 1:
            for k in tk:
                k.pt = (k.pt[0] * factor, k.pt[1] * factor)

        bf = cv2.BFMatcher(norm)
        knn = bf.knnMatch(td, img_desc, k=min(len(img_desc), _KNN_K))
        good = []
        for ms in knn:
            for m in ms:
                if m.distance <= maxd:
                    good.append(m)
                else:
                    break  # knnMatch results are sorted by distance
        if len(good) < vote_floor:
            return []

        tk_xy = np.array([k.pt for k in tk], np.float32).reshape(-1, 2)
        tk_sz = np.array([k.size for k in tk], np.float32)
        tk_ang = np.array([k.angle for k in tk], np.float32)
        qi = np.fromiter((m.queryIdx for m in good), np.int64, len(good))
        ti = np.fromiter((m.trainIdx for m in good), np.int64, len(good))

        tpx, tpy = tk_xy[qi, 0], tk_xy[qi, 1]
        ipx, ipy = ik_xy[ti, 0], ik_xy[ti, 1]
        ts = np.where(tk_sz[qi] > 1e-3, tk_sz[qi], 1.0)
        rel_s = ik_sz[ti] / ts
        ta, ia = tk_ang[qi], ik_ang[ti]
        rel_a = np.deg2rad(np.where((ia >= 0) & (ta >= 0), ia - ta, 0.0))
        # Predicted instance centre = img_kp + s*R(da)*(tmpl_centre - tmpl_kp).
        vx = (0.5 * tw) - tpx
        vy = (0.5 * th) - tpy
        cos, sin = np.cos(rel_a), np.sin(rel_a)
        pcx = ipx + rel_s * (cos * vx - sin * vy)
        pcy = ipy + rel_s * (sin * vx + cos * vy)

        valid = (pcx >= 0) & (pcx < full_w) & (pcy >= 0) & (pcy < full_h)
        pcx, pcy, rel_s, rel_a = pcx[valid], pcy[valid], rel_s[valid], rel_a[valid]
        # Keep the matched-keypoint coords index-aligned with the (now-filtered)
        # vote arrays so each proposal can carry its OWN correspondence cluster for
        # the perspective (homography) verify fallback. tpx/tpy are in the oriented-
        # template detection frame; ipx/ipy are full-image coords.
        tpx, tpy = tpx[valid], tpy[valid]
        ipx, ipy = ipx[valid], ipy[valid]
        if pcx.size < vote_floor:
            return []

        # Record this orientation's COMPLETE correspondence set (template->image, in
        # full-res coords) for the image-space cluster homography fallback. Unlike
        # the per-proposal peak-bin clusters below, this is every good match for the
        # variant; the fallback box-gathers the subset around a failing proposal's
        # centre — a copy's image keypoints stay clustered regardless of perspective,
        # so that gather is the copy's COMPLETE cluster (high inlier ratio for RANSAC).
        if corr_out is not None:
            corr_out[orient] = (
                np.ascontiguousarray(tpx), np.ascontiguousarray(tpy),
                np.ascontiguousarray(ipx), np.ascontiguousarray(ipy),
            )

        # Coarse 2-D centre histogram (bin ≈ a quarter of the template short side).
        binsz = max(8, int(min(tw, th) // 4))
        bx = (pcx / binsz).astype(np.int64)
        by = (pcy / binsz).astype(np.int64)
        keys = bx * 1_000_000 + by
        order = np.argsort(keys, kind="stable")
        keys_s = keys[order]
        _uniq, starts, counts = np.unique(keys_s, return_index=True, return_counts=True)

        # Bin-key -> member indices, for 3x3 neighbourhood cluster enrichment below.
        # Under strong perspective the SIMILARITY vote predicts an instance centre per
        # correspondence; the warp makes those predicted centres scatter across
        # adjacent bins, so a single peak bin starves to a handful of points even
        # though hundreds of ORB matches physically sit on the copy. RANSAC then has
        # too few inliers (few_inliers) or fits an ill-conditioned H (project_box).
        bin_members = {int(k): order[s:s + c] for k, s, c in zip(keys_s[starts], starts, counts)}

        proposals: list[tuple] = []
        for st, ct in zip(starts, counts):
            if ct < vote_floor:
                continue
            members = order[st:st + ct]
            # Enrich the correspondence cluster with the ±1-bin (3x3) neighbourhood:
            # gather the coherent matches whose predicted centres scattered into
            # adjacent bins so RANSAC regains support. The vote count ``ct`` still
            # keys ONLY the peak bin (routing/gating + medians below are unchanged),
            # so precision is preserved. The 3x3 reach is ≤ 3*binsz (~96px for the
            # 130px motif) — far below the ~320px inter-copy spacing, so the gather
            # cannot cross into a neighbouring copy.
            kpx = int(bx[members[0]])
            kpy = int(by[members[0]])
            enriched = members
            for dxb in (-1, 0, 1):
                for dyb in (-1, 0, 1):
                    if dxb == 0 and dyb == 0:
                        continue
                    nbr = bin_members.get((kpx + dxb) * 1_000_000 + (kpy + dyb))
                    if nbr is not None:
                        enriched = np.concatenate((enriched, nbr))
            # This proposal's own (enriched) correspondences (template->image), for
            # the perspective homography fallback in _verify.
            sp = np.ascontiguousarray(np.stack([tpx[enriched], tpy[enriched]], 1).astype(np.float32))
            dp = np.ascontiguousarray(np.stack([ipx[enriched], ipy[enriched]], 1).astype(np.float32))
            proposals.append((
                float(np.median(pcx[members])),
                float(np.median(pcy[members])),
                float(np.median(rel_s[members])),
                float(np.median(rel_a[members])),
                int(ct),
                orient,
                th,
                tw,
                sp,
                dp,
            ))
        return proposals

    @staticmethod
    def _dedupe_proposals(proposals: list[tuple], cell: int) -> list[tuple]:
        """Collapse proposals in the same ``(orientation, grid-cell)`` to the
        highest-vote one. Bins are quarter-template, so a true copy can split its
        votes across adjacent bins/orientations into several near-identical
        proposals; deduping on a coarser cell (~half the template short side) keeps
        the verify set small without dropping distinct instances."""
        best: dict[tuple, tuple] = {}
        for p in proposals:
            cx, cy, orient = p[0], p[1], p[5]
            key = (orient, int(cx // cell), int(cy // cell))
            cur = best.get(key)
            if cur is None or p[4] > cur[4]:
                best[key] = p
        return list(best.values())

    # -- appearance verification ---------------------------------------------

    @staticmethod
    def _zncc(a: np.ndarray, b: np.ndarray, weights: "list[float] | None" = None) -> float:
        """(Weighted) zero-mean normalized cross-correlation of two patches.

        Works on single-channel ``(H, W)`` (luminance) and multi-channel
        ``(H, W, C)`` (rgb / ycbcr) inputs. For colour, each channel is zero-meaned
        independently and the per-channel numerators and per-channel denominators
        are summed *before* the single division, each scaled by its weight — the
        same weighted multi-channel rule the conv NCC uses
        (``Σ_c w_c <a_c,b_c> / Σ_c w_c ‖a_c‖·‖b_c‖``). Equal weights == the
        unweighted multi-channel NCC. A flat channel (no variance in either patch)
        contributes nothing rather than zeroing the whole score, so a copy that is
        textured in only some channels still verifies.
        """
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        if a.ndim == 2:
            a = a[:, :, None]
            b = b[:, :, None]
        c_n = a.shape[2]
        if weights is None or len(weights) != c_n:
            weights = [1.0] * c_n
        num = 0.0
        den = 0.0
        for c in range(c_n):
            w = float(weights[c])
            if w <= 0.0:
                continue
            ac = a[:, :, c]
            bc = b[:, :, c]
            ac = ac - ac.mean()
            bc = bc - bc.mean()
            na = np.sqrt(float((ac * ac).sum()))
            nb = np.sqrt(float((bc * bc).sum()))
            if na < 1e-6 or nb < 1e-6:
                continue  # flat in this channel: contributes no correlation signal
            num += w * float((ac * bc).sum())
            den += w * na * nb
        return num / den if den > 1e-6 else 0.0

    @staticmethod
    def _net_d4(orient: str, an: float) -> str | None:
        """Net D4 code if residual angle ``an`` ≈ a 90° multiple, else ``None``.

        Lets a copy whose keypoints recovered a ±90/180° rotation be verified by
        the pixel-exact axis path under its true D4 orientation (this variant's D4
        map composed with the k·90° turn), instead of leaking into the
        resampling-lossy, FP-prone warp path.
        """
        deg = float(np.rad2deg(an))
        k = int(round(deg / 90.0))
        if abs(deg - 90.0 * k) > _D4_ANGLE_TOL_DEG:
            return None
        turn = ("R0", "R90", "R180", "R270")[k % 4]
        net = (
            np.asarray(_ORIENT_LINEAR[turn], np.float64)
            @ np.asarray(_ORIENT_LINEAR[orient], np.float64)
        )
        return orientation_from_matrix(net)

    def _verify(
        self,
        proposal: tuple,
        tmpl_oriented_cache: dict[str, np.ndarray],
        verify_image: np.ndarray,
        full_h: int,
        full_w: int,
        warp_min_votes: int,
        weights: "list[float] | None" = None,
        orient_corr: "dict[str, tuple] | None" = None,
    ) -> tuple[tuple[int, int, int, int], float, str] | None:
        """Verify a proposal by appearance. Returns ``(box, zncc, orient)`` or None.

        ``verify_image`` and ``tmpl_oriented_cache`` are in the verification colour
        space — single-channel luminance, or 3-channel rgb / ycbcr — and ``weights``
        are the per-channel ZNCC weights; :meth:`_zncc` handles either, so this
        routine is colour-blind.

        Routing:
          * Near-D4 (residual angle ≈ a 90° multiple, scale ≈ 1) -> pixel-exact
            AXIS path: ZNCC the net-oriented template with a small local alignment
            search (a cheap centre-prefilter rejects clear non-matches in one ZNCC).
            This covers every aligned/mirrored PCB instance.
          * Genuinely off-axis (arbitrary rotation/scale) -> INVERSE-WARP path:
            warp the image window back into the upright-template frame and ZNCC,
            refined over a small (angle, scale) grid, gated on strong vote support.
            This is what the rotated-instance generality test exercises.
        """
        cv2 = self._cv2
        cx, cy, sc, an, votes, orient, th0, tw0 = proposal[:8]
        # This proposal's own correspondences (template->image), for the perspective
        # homography fallback. len-guarded so an 8-tuple (older callers/tests) works.
        src_pts = proposal[8] if len(proposal) > 8 else None
        dst_pts = proposal[9] if len(proposal) > 9 else None

        net_code = self._net_d4(orient, an) if 0.7 <= sc <= 1.4 else None
        if net_code is not None:
            tmpl = tmpl_oriented_cache.get(net_code)
            if tmpl is None:
                tmpl = np.ascontiguousarray(
                    apply_orientation(tmpl_oriented_cache["R0"], net_code)
                )
                tmpl_oriented_cache[net_code] = tmpl
            th, tw = tmpl.shape[:2]
            sw = max(1, int(round(tw * sc)))
            sh = max(1, int(round(th * sc)))
            tw_img = (
                cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
                if sc != 1.0 else tmpl
            )
            bx0 = int(round(cx - sw / 2.0))
            by0 = int(round(cy - sh / 2.0))
            # Centre-prefilter: one ZNCC at the vote centre; a true copy within
            # ±_NCC_SEARCH px cannot recover from a centre this far below the floor,
            # so reject cheaply (the dominant runtime win on large templates).
            # A genuine PERSPECTIVE copy can masquerade as near-D4 (vote angle≈0,
            # scale≈1) yet have a near-zero unwarped centre ZNCC, so a vote-RICH
            # proposal that fails the prefilter still falls through to the homography
            # / image-space cluster fallback; only a low-vote proposal is rejected
            # cheaply here (the fast PCB path — those never warrant a homography fit).
            axis_failed = False
            if 0 <= bx0 <= full_w - sw and 0 <= by0 <= full_h - sh:
                c = self._zncc(verify_image[by0:by0 + sh, bx0:bx0 + sw], tw_img, weights)
                if c < _NCC_ACCEPT - _AXIS_PREFILTER_MARGIN:
                    if votes < warp_min_votes:
                        return None
                    axis_failed = True  # skip the local search; go to the fallback
            if not axis_failed:
                # Local ±-px alignment search (the vote-median centre is a few px
                # noisy — exactly what NCC/SSD slides over); keep the best alignment.
                best_s, best_xy = -2.0, None
                for dy in range(-_NCC_SEARCH, _NCC_SEARCH + 1):
                    for dx in range(-_NCC_SEARCH, _NCC_SEARCH + 1):
                        x0, y0 = bx0 + dx, by0 + dy
                        if x0 < 0 or y0 < 0 or x0 + sw > full_w or y0 + sh > full_h:
                            continue
                        win = verify_image[y0:y0 + sh, x0:x0 + sw]
                        if win.shape != tw_img.shape:
                            continue
                        s = self._zncc(win, tw_img, weights)
                        if s > best_s:
                            best_s, best_xy = s, (x0, y0)
                if best_xy is not None and best_s >= _NCC_ACCEPT:
                    # Clamp the ZNCC confidence to [0, 1] (float rounding can nudge an
                    # exact-copy correlation a hair above 1.0).
                    return (best_xy[0], best_xy[1], sw, sh), min(1.0, max(0.0, best_s)), net_code
            # Axis verify FAILED. A perspective copy can masquerade as near-D4 (its
            # vote-median angle≈0, scale≈1) yet not match the unwarped template, so
            # fall through to the off-axis / homography fallback rather than reject.

        # General (off-axis) path — arbitrary rotation+scale, the generality test.
        # Inverse-warp the IMAGE window back into the upright-template frame and
        # ZNCC against the template (the smooth image is resampled once, so a
        # genuine copy keeps high correlation), refined over a small (angle, scale)
        # grid. Gated on strong vote support so the lenient floor cannot admit a
        # spurious cross-copy warp.
        if votes < warp_min_votes:
            return None
        tmpl = tmpl_oriented_cache[orient]
        th_o, tw_o = tmpl.shape[:2]
        corners = np.array([[0, 0], [tw_o, 0], [tw_o, th_o], [0, th_o]], np.float32)

        def _inv_warp_zncc(angle, scale):
            cos, sin = np.cos(angle), np.sin(angle)
            # Forward affine: oriented-template coords -> image coords.
            M = np.array([
                [scale * cos, -scale * sin,
                 cx - scale * (cos * (tw_o / 2.0) - sin * (th_o / 2.0))],
                [scale * sin, scale * cos,
                 cy - scale * (sin * (tw_o / 2.0) + cos * (th_o / 2.0))],
            ], np.float32)
            proj = (M[:, :2] @ corners.T).T + M[:, 2]
            x0 = int(np.floor(proj[:, 0].min())); y0 = int(np.floor(proj[:, 1].min()))
            x1 = int(np.ceil(proj[:, 0].max())); y1 = int(np.ceil(proj[:, 1].max()))
            if x0 < 0 or y0 < 0 or x1 > full_w or y1 > full_h or x1 <= x0 or y1 <= y0:
                return None
            Minv = cv2.invertAffineTransform(M)
            back = cv2.warpAffine(verify_image, Minv, (tw_o, th_o), flags=cv2.INTER_LINEAR)
            return self._zncc(back, tmpl, weights), (x0, y0, x1 - x0, y1 - y0)

        r0 = _inv_warp_zncc(an, sc)
        if r0 is not None and r0[0] >= _NCC_ACCEPT_WARP - _WARP_PREFILTER:
            best_s, best_box = r0
            best_a, best_sc = an, sc
            for da in (-4.0, -2.0, 0.0, 2.0, 4.0):
                for ds in (0.88, 0.94, 1.0, 1.06, 1.12):
                    if da == 0.0 and ds == 1.0:
                        continue
                    a, s = an + np.deg2rad(da), sc * ds
                    r = _inv_warp_zncc(a, s)
                    if r is not None and r[0] > best_s:
                        best_s, best_box, best_a, best_sc = r[0], r[1], a, s
            for da in (-1.5, -0.75, 0.0, 0.75, 1.5):
                for ds in (0.97, 0.985, 1.0, 1.015, 1.03):
                    if da == 0.0 and ds == 1.0:
                        continue
                    a, s = best_a + np.deg2rad(da), best_sc * ds
                    r = _inv_warp_zncc(a, s)
                    if r is not None and r[0] > best_s:
                        best_s, best_box = r[0], r[1]
            if best_box is not None and best_s >= _NCC_ACCEPT_WARP:
                return best_box, min(1.0, max(0.0, best_s)), orient

        # --- PERSPECTIVE homography fallback ---------------------------------
        # The cheaper similarity paths could not verify this vote-rich proposal. Fit
        # a homography to its OWN clean correspondence cluster and warp the image
        # back through it for the SAME ZNCC gate — recovering perspective/affine the
        # similarity model cannot. Reached only for >= warp_min_votes proposals (the
        # off-axis votes gate above), so D4/PCB axis-successes never get here.
        r = self._verify_homography(
            cv2, src_pts, dst_pts, tmpl, th_o, tw_o, verify_image, full_w, full_h, weights
        )
        if r is not None:
            return r

        # --- IMAGE-SPACE CLUSTER homography fallback -------------------------
        # The peak-bin cluster was starved/contaminated by the similarity vote's
        # perspective scatter (it keys only the centre-histogram peak). Re-gather
        # this orientation's COMPLETE correspondence set restricted to an image-space
        # box around the proposal centre — a perspective copy's image keypoints stay
        # physically clustered, so that box holds the copy's whole, high-inlier
        # cluster — and RANSAC + the SAME ZNCC gate on it. Last resort, so it only
        # ever runs for proposals every cheaper path already rejected.
        return self._verify_cluster_homography(
            cv2, orient, cx, cy, orient_corr, tmpl, th_o, tw_o,
            verify_image, full_w, full_h, weights,
        )

    def _verify_homography(self, cv2, src_pts, dst_pts, tmpl, th_o, tw_o,
                           verify_image, full_w, full_h, weights):
        """Perspective verify: fit a RANSAC homography to a proposal's own
        correspondence cluster, warp the image back into the template frame, and
        apply the ZNCC>=_NCC_ACCEPT_WARP appearance gate. Returns ``(box, score,
        "R0"-tagged orient)`` or ``None``. Geometric guards + the ZNCC gate keep
        precision and prevent repetitive-lattice phantoms.
        """
        if src_pts is None or dst_pts is None or len(src_pts) < _HOMOG_MIN_PTS:
            return None
        ps, pd = src_pts, dst_pts
        if len(ps) > _MAX_CLUSTER_PTS:  # bound RANSAC cost on dense clusters
            sel = np.linspace(0, len(ps) - 1, _MAX_CLUSTER_PTS).astype(np.int64)
            ps, pd = ps[sel], pd[sel]
        try:
            H, mask = cv2.findHomography(
                ps.reshape(-1, 1, 2), pd.reshape(-1, 1, 2),
                cv2.RANSAC, _REPROJ_THRESH, maxIters=_RANSAC_ITERS, confidence=0.999,
            )
        except cv2.error:
            return None
        if H is None or mask is None or int(mask.ravel().sum()) < _HOMOG_MIN_PTS:
            return None
        # Reject wild projective terms (a runaway perspective blow-up).
        if abs(float(H[2, 0])) > _HOMOG_MAX_BOTTOM or abs(float(H[2, 1])) > _HOMOG_MAX_BOTTOM:
            return None
        corners = np.array([[[0, 0], [tw_o, 0], [tw_o, th_o], [0, th_o]]], np.float32)
        box = self._project_to_box(cv2, H, corners, full_w, full_h)
        if box is None:
            return None
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        back = cv2.warpPerspective(verify_image, Hinv, (tw_o, th_o), flags=cv2.INTER_LINEAR)
        score = self._zncc(back, tmpl, weights)
        if score < _NCC_ACCEPT_WARP:
            return None
        bx, by, bw, bh = box
        return (int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))), \
            min(1.0, max(0.0, score)), "R0"

    def _verify_cluster_homography(self, cv2, orient, cx, cy, orient_corr, tmpl,
                                   th_o, tw_o, verify_image, full_w, full_h, weights):
        """Image-space cluster perspective verify (the perspective recall fix).

        Gathers ALL of ``orient``'s good template->image correspondences whose IMAGE
        position lies in a ``±_CLUSTER_RADIUS_FACTOR·max(th,tw)`` box around the
        proposal centre ``(cx, cy)`` — a perspective copy's image keypoints stay
        physically clustered, so this is the copy's COMPLETE, high-inlier cluster
        (the peak-bin cluster that ``_verify_homography`` uses was starved by the
        similarity vote's scatter). RANSAC a homography on it and apply the SAME
        ZNCC>=_NCC_ACCEPT_WARP gate; returns ``(box, score, "R0")`` or ``None``.

        Reached only after every cheaper path (and the peak-bin homography) rejected
        the proposal, so the D4/PCB axis-successes never enter it. Precision is the
        ZNCC gate plus the convex/positive-area/scale guards, which (unlike the
        peak-bin path's ``_project_to_box``) flip-test by the SIGNED AREA of the
        projected quad — correct under perspective, where the linear-2x2 det can go
        negative for an orientation-preserving copy.
        """
        if orient_corr is None:
            return None
        corr = orient_corr.get(orient)
        if corr is None:
            return None
        tpx, tpy, ipx, ipy = corr
        rad = _CLUSTER_RADIUS_FACTOR * max(th_o, tw_o)
        sel = (np.abs(ipx - cx) < rad) & (np.abs(ipy - cy) < rad)
        if int(sel.sum()) < _CLUSTER_MIN_PTS:
            return None
        ps = np.ascontiguousarray(np.stack([tpx[sel], tpy[sel]], 1).astype(np.float32))
        pd = np.ascontiguousarray(np.stack([ipx[sel], ipy[sel]], 1).astype(np.float32))
        if len(ps) > _CLUSTER_MAX_PTS:  # bound RANSAC cost on a dense cluster
            idx = np.linspace(0, len(ps) - 1, _CLUSTER_MAX_PTS).astype(np.int64)
            ps, pd = ps[idx], pd[idx]
        try:
            H, mask = cv2.findHomography(
                ps.reshape(-1, 1, 2), pd.reshape(-1, 1, 2),
                cv2.RANSAC, _REPROJ_THRESH, maxIters=_CLUSTER_RANSAC_ITERS, confidence=0.999,
            )
        except cv2.error:
            return None
        if H is None or mask is None or int(mask.ravel().sum()) < _HOMOG_MIN_PTS:
            return None
        # Reject wild projective terms (a runaway perspective blow-up) — same ceiling
        # as the peak-bin path.
        if abs(float(H[2, 0])) > _HOMOG_MAX_BOTTOM or abs(float(H[2, 1])) > _HOMOG_MAX_BOTTOM:
            return None
        corners = np.array([[[0, 0], [tw_o, 0], [tw_o, th_o], [0, th_o]]], np.float32)
        quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        if not np.all(np.isfinite(quad)):
            return None
        # Orientation-preserving + non-degenerate by the PROJECTED quad (perspective-
        # correct flip test, not the linear-2x2 det), then a sane scale bound from the
        # quad's area and convexity (a cross-copy lattice fit is non-convex / wrong area).
        area2 = self._signed_area2(quad)
        if area2 <= 0 or not self._is_convex(quad):
            return None
        lin = (0.5 * area2 / (tw_o * th_o)) ** 0.5 if tw_o * th_o > 0 else 0.0
        if lin < _MIN_SCALE or lin > _MAX_SCALE:
            return None
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        back = cv2.warpPerspective(verify_image, Hinv, (tw_o, th_o), flags=cv2.INTER_LINEAR)
        score = self._zncc(back, tmpl, weights)
        if score < _NCC_ACCEPT_WARP:
            return None
        x0 = max(0.0, min(float(quad[:, 0].min()), full_w))
        y0 = max(0.0, min(float(quad[:, 1].min()), full_h))
        x1 = max(0.0, min(float(quad[:, 0].max()), full_w))
        y1 = max(0.0, min(float(quad[:, 1].max()), full_h))
        bw, bh = x1 - x0, y1 - y0
        if bw < 1 or bh < 1:
            return None
        return (int(round(x0)), int(round(y0)), int(round(bw)), int(round(bh))), \
            min(1.0, max(0.0, score)), "R0"

    @staticmethod
    def _signed_area2(quad: np.ndarray) -> float:
        """Twice the signed polygon area (shoelace); >0 for a CCW quad."""
        x = quad[:, 0]
        y = quad[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def _is_convex(quad: np.ndarray) -> bool:
        """True if the ordered 4-point quad is convex (consistent turn direction)."""
        n = len(quad)
        sign = 0
        for i in range(n):
            dx1 = quad[(i + 1) % n, 0] - quad[i, 0]
            dy1 = quad[(i + 1) % n, 1] - quad[i, 1]
            dx2 = quad[(i + 2) % n, 0] - quad[(i + 1) % n, 0]
            dy2 = quad[(i + 2) % n, 1] - quad[(i + 1) % n, 1]
            cross = dx1 * dy2 - dy1 * dx2
            if cross != 0.0:
                s = 1 if cross > 0 else -1
                if sign == 0:
                    sign = s
                elif s != sign:
                    return False
        return True

    def _project_to_box(self, cv2, H, tmpl_corners, iw, ih):
        """Project template corners through ``H`` to an AABB, with degeneracy guards.

        Rejects (returns ``None``) a near-singular / mirror-flipped affine part, an
        out-of-range linear scale, a non-finite / non-convex / zero-area projected
        quad — the guards that stop a homography fit from latching onto a
        cross-copy lattice phantom. Returns ``(x, y, w, h)`` clamped to the image.
        """
        a, b, c, d = float(H[0, 0]), float(H[0, 1]), float(H[1, 0]), float(H[1, 1])
        det = a * d - b * c
        if det <= _MIN_AFFINE_DET:
            return None
        lin = det ** 0.5
        if lin < _MIN_SCALE or lin > _MAX_SCALE:
            return None
        quad = cv2.perspectiveTransform(tmpl_corners, H).reshape(-1, 2)
        if not np.all(np.isfinite(quad)):
            return None
        if self._signed_area2(quad) <= 0 or not self._is_convex(quad):
            return None
        x0 = max(0.0, min(float(quad[:, 0].min()), iw))
        y0 = max(0.0, min(float(quad[:, 1].min()), ih))
        x1 = max(0.0, min(float(quad[:, 0].max()), iw))
        y1 = max(0.0, min(float(quad[:, 1].max()), ih))
        w, h = x1 - x0, y1 - y0
        if w < 1 or h < 1:
            return None
        return (x0, y0, w, h)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _drop_lattice_phantoms(results: list[Match]) -> list[Match]:
        """Remove lattice phantoms: a box engulfing several real detections.

        On a repetitive pattern an oversized box can span several true instances;
        such a phantom is identified geometrically — its box contains the CENTRES
        of ≥ 2 other (smaller) detections — which is scale-agnostic, so it never
        drops a legitimate larger-scale copy.
        """
        n = len(results)
        if n < 3:
            return results
        cx = [r.x + r.w / 2.0 for r in results]
        cy = [r.y + r.h / 2.0 for r in results]
        area = [r.w * r.h for r in results]
        keep = [True] * n
        for i, a in enumerate(results):
            engulfed = 0
            for j in range(n):
                if i == j:
                    continue
                if (
                    a.x <= cx[j] < a.x + a.w
                    and a.y <= cy[j] < a.y + a.h
                    and area[i] > 1.5 * area[j]
                ):
                    engulfed += 1
            if engulfed >= 2:
                keep[i] = False
        return [r for r, k in zip(results, keep) if k]

    def _to_gray(self, arr: np.ndarray) -> np.ndarray:
        """Collapse a template to a C-contiguous uint8 grayscale image.

        Uses BT.601 luminance for RGB input so it lines up with the staged image
        grayscale the engine passes; a 2-D input is taken as-is.
        """
        if arr.ndim == 2:
            return np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        if arr.ndim == 3 and arr.shape[2] == 3:
            rgb = arr.astype(np.float32)
            lum = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
            return np.ascontiguousarray(np.clip(np.rint(lum), 0, 255).astype(np.uint8))
        raise ValueError(f"template must be (th,tw,3) or (th,tw), got shape {arr.shape}")

    @classmethod
    def _nms(cls, results: list[Match], nms_iou: float) -> list[Match]:
        """Greedy box-IoU non-max suppression over a score-sorted match list.

        Walks the (already score-descending) ``results`` keeping each box unless
        its IoU with a higher-scored, already-kept box exceeds ``nms_iou`` — so one
        physical instance proposed twice collapses to a single box, while two
        genuinely distinct instances that merely sit close are kept.
        """
        kept: list[Match] = []
        for m in results:
            if all(cls._box_iou(m, k) <= nms_iou for k in kept):
                kept.append(m)
        return kept

    @staticmethod
    def _box_iou(a: Match, b: Match) -> float:
        """IoU of two half-open ``Match`` boxes (area overlap / area union)."""
        ix0 = max(a.x, b.x)
        iy0 = max(a.y, b.y)
        ix1 = min(a.x + a.w, b.x + b.w)
        iy1 = min(a.y + a.h, b.y + b.h)
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        inter = iw * ih
        union = a.w * a.h + b.w * b.h - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _excluded(
        m: Match, exclude_box: tuple[int, int, int, int], exclude_iou: float
    ) -> bool:
        """True if ``m`` overlaps the source box past ``exclude_iou`` or is centered in it."""
        ex, ey, ew, eh = exclude_box
        ix0 = max(m.x, ex)
        iy0 = max(m.y, ey)
        ix1 = min(m.x + m.w, ex + ew)
        iy1 = min(m.y + m.h, ey + eh)
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        inter = iw * ih
        union = m.w * m.h + ew * eh - inter
        iou = inter / union if union > 0 else 0.0
        if iou > exclude_iou:
            return True
        cx = m.x + m.w / 2.0
        cy = m.y + m.h / 2.0
        return (ex <= cx < ex + ew) and (ey <= cy < ey + eh)
