"""OpenCV feature-matching backend for FastMatch (``method=="features"``).

This module implements :class:`FeatureMatcher`, the warp-tolerant alternative to
the GPU convolution methods (§J.3 of ``docs/DESIGN.md``). Where NCC/SSD/CCORR
need the instances to be roughly aligned and the same scale, feature matching
detects ORB/AKAZE/SIFT keypoints, matches descriptors with a Lowe ratio test,
and fits a RANSAC homography per instance — so it recovers copies that are
**rotated, scaled, or perspective-warped**, which the template methods miss.

OpenCV (``cv2``) is required **only** for this method. It is imported lazily (at
:class:`FeatureMatcher` construction, not at module import) so importing
:mod:`fastmatch.engine` never drags in cv2; if cv2 is unavailable the
constructor raises the documented :class:`RuntimeError` and the UI disables the
"features" option.

Coordinate convention matches the rest of FastMatch: integer **image pixels**,
origin top-left, +x right / +y down, half-open boxes ``(x, y, w, h)`` (§F.3).

Pipeline (see §J.3):

1. **Full-resolution tiled detection** — image (and template) keypoints are
   detected at native scale, because down-sampling drops keypoints and realistic
   templates (the demo's flat colour-block motif; any feature-poor pattern) have
   few to spare. Cost/memory is bounded by TILING (overlapping tiles, each with
   its own keypoint budget so features stay spatially dense); the result is cached
   per detector so the cost is one-time. (A coarsest-safe-level down-sampling
   pyramid is plumbed via ``_MAX_FEATURE_LEVEL`` but defaults OFF — it traded too
   much recall on feature-poor templates; see that constant's note.)
2. **Detector** — ORB (default), AKAZE, or SIFT on the grayscale image. Image
   keypoints/descriptors are computed **once per (image, detector)** and cached
   on this matcher; :meth:`invalidate_image` clears the cache when the engine
   stages a new image.
3. **Multi-instance descriptor matching** — for each template feature keep ALL
   image matches within an absolute "good descriptor" distance (``radiusMatch``),
   NOT just the best. The classic Lowe ratio test rejects a match whose 2nd
   neighbour is equally close — i.e. EVERY repeated/identical instance (each
   template feature recurs once per copy) — so it finds nothing on the demo's many
   copies. Keeping all good matches (one per copy) is what makes multi-instance
   detection work; the geometry step below rejects the spurious ones.
4. **Generalized-Hough vote + per-cluster homography** — each correspondence votes
   for an instance centre using the matched keypoints' scale and orientation (a
   similarity transform, à la Lowe 2004), so the correct correspondences for each
   copy form a sharp peak well above the diffuse noise. Each peak's
   correspondences are verified/refined with one homography (degeneracy-guarded),
   yielding the instance's axis-aligned bbox; the votes are consumed and the next
   peak is taken, up to ``feature_max_instances``. (Voting replaced a plain
   sequential RANSAC, which drowned in the large ambiguous correspondence set and
   recovered ~0-2 of many identical copies.)
5. **Score** — ``min(1.0, inliers / (2 * feature_min_inliers))`` (richly supported
   instances saturate near 1.0; ambiguous ones score lower).
6. **Exclude source** — same rule as the conv path (IoU > ``exclude_iou`` or
   center inside ``exclude_box``); overlapping detections merged by IoU-NMS.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .types import Match

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
                           # would defeat the Lowe ratio test there).
_FEATURES_PER_TILE = 1500  # ORB/SIFT keypoint budget per detection tile
_TEMPLATE_FEATURES = 5000  # denser budget for the (small) template crop
_MIN_TEMPLATE_SIDE = 12    # below this a template cannot anchor a homography

# Coarsest-safe-level detection pyramid (§K.2). Detecting features over a
# gigapixel image at full resolution is the dominant cost of the feature path AND
# floods the matcher with distractor descriptors (which hurt the Lowe ratio test
# → missed warped instances). Instead we detect at the COARSEST pyramid level
# where the downsampled template still keeps enough texture to fire ORB. The
# level is chosen FROM THE TEMPLATE so the template never decimates below
# ``MIN_TPL_FEATURE`` px on its shorter side — that exact over-decimation is the
# bug that made feature matching return nothing on large images, so this bound is
# load-bearing, NOT a tunable.
#
#   L = clamp(floor(log2(min(th, tw) / MIN_TPL_FEATURE)), 0, MAX_FEATURE_LEVEL)
#
# L == 0 reproduces today's full-resolution tiled detection exactly (small
# templates / small images are unchanged), so the small-image behaviour and the
# existing tests are preserved.
_MIN_TPL_FEATURE = 96      # coarsest level keeps the template's short side ≥ ~96 px
# Downsampling the image/template for feature detection is a SPEED optimization,
# but it silently drops keypoints — and realistic templates (the demo's flat
# colour-block motif; any feature-poor pattern) have few to begin with, so even a
# 2× shrink halves recall on repeated instances. Full-resolution detection is
# already tiled (memory-bounded) and cached (the cost is one-time, amortized
# across queries), so it is fast enough in practice. We therefore default the
# feature path to L=0 (full res) for reliability; set this > 0 only to trade
# recall for first-query detection speed on very large, feature-rich templates.
_MAX_FEATURE_LEVEL = 0     # 0 -> always full resolution (see note above)

#: RANSAC reprojection threshold (px, at detection scale). Inliers must lie
#: within this distance of the projected model — a few px tolerates the
#: sub-pixel keypoint and homography noise without admitting outliers.
_REPROJ_THRESH = 5.0

#: RANSAC iterations for the per-instance homography fit. Raised above the cv2
#: default because with MANY repeated copies a random 4-point sample is less
#: likely to land entirely on one instance, so more iterations are needed to hit
#: a geometrically consistent set.
_RANSAC_ITERS = 5000

#: Multi-instance descriptor matching. The classic Lowe ratio test rejects a
#: match whose 2nd-nearest neighbour is (nearly) as close — which is EXACTLY what
#: happens for repeated/identical instances: each template feature recurs once
#: per copy, so its nearest neighbours (the copies) are all equally close and the
#: ratio test discards them, finding nothing. The demo image (and the common
#: "find every copy" use case) is precisely this. So instead of the ratio test we
#: keep ALL image matches within an absolute "good descriptor" distance of each
#: template feature (one correspondence per copy) and let the geometric RANSAC
#: verification below reject the spurious ones. ``_GOOD_DIST`` is the per-detector
#: max descriptor distance for a "good" match (random ORB Hamming ≈ 128; an exact
#: copy ≈ 0); ``_MAX_MATCHES_PER_FEATURE`` caps the fan-out on repetitive texture.
_GOOD_DIST = {
    "orb": 64.0,    # Hamming on the 256-bit ORB descriptor
    "akaze": 90.0,  # Hamming on AKAZE's (longer) binary descriptor
    "sift": 300.0,  # L2 on 128-float SIFT (RANSAC filters the looser slack)
}
_MAX_MATCHES_PER_FEATURE = 256

#: Reject a homography whose affine-part determinant is below this magnitude
#: (near-singular / folded mapping) — a degenerate fit that does not correspond
#: to a real, orientation-preserving instance (§J.3 step 4).
_MIN_AFFINE_DET = 1e-6

#: Sane-scale guard: reject homographies that blow the template up or shrink it
#: past these absolute **linear scale** ratios (a runaway projective fit, not a
#: real copy). Compared against ``sqrt(det)`` so the bound is a per-axis length
#: ratio, not an area ratio (§J.3 step 4).
_MAX_SCALE = 1000.0
_MIN_SCALE = 1.0 / 1000.0


class FeatureMatcher:
    """ORB/AKAZE/SIFT + RANSAC-homography multi-instance matcher (§J.3).

    One instance is reused across queries by the engine's :class:`Matcher`. It
    caches the downsampled-image keypoints/descriptors keyed by detector name so
    repeated queries on the same staged image only re-detect the (small)
    template. :meth:`invalidate_image` is called by ``Matcher.set_image`` when a
    new image is staged.

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
        # (§K.2). Keying on the level lets two queries with different templates
        # (hence different safe levels) each reuse their own cached detection.
        # Cleared by invalidate_image().
        self._image_cache: dict[
            tuple[str, int], tuple[list[Any], np.ndarray | None]
        ] = {}
        self._image_shape: tuple[int, int] | None = None  # (H, W) at full res
        # Cheap fingerprint of the image the cache was built against. Defense in
        # depth: the engine calls invalidate_image() in set_image, but if a caller
        # reuses this matcher on a *different* array of the same detector without
        # invalidating, a stale fingerprint mismatch forces a re-detect (§U2).
        self._image_fingerprint: tuple[Any, ...] | None = None
        # Test/diagnostic toggle: force the full-resolution path (L=0) regardless
        # of the template-selected level. The full path stays the recall reference
        # so a test can assert pyramid-vs-full parity (§K.2/§K.3). Off by default.
        self._force_full_level: bool = False
        # The level chosen for the most recent query (read by tests/benchmarks to
        # report the selected L per template/image size). ``None`` before any
        # query runs.
        self._last_level: int | None = None

    # -- cache management ----------------------------------------------------

    def invalidate_image(self) -> None:
        """Drop any cached per-image keypoints/descriptors (§J.3/§J.4).

        Called by ``Matcher.set_image`` so a freshly staged image is re-detected
        on the next feature query rather than reusing stale features.
        """
        self._image_cache.clear()
        self._image_shape = None
        self._image_fingerprint = None

    # -- detector factory ----------------------------------------------------

    def _make_detector(self, name: str, nfeatures: int | None = None):
        """Build the requested OpenCV detector and its descriptor norm.

        Returns ``(detector, norm)`` where ``norm`` is the BFMatcher distance for
        that descriptor type: HAMMING for the binary ORB/AKAZE descriptors, L2
        for SIFT's float descriptors (§J.3 step 3).

        ``nfeatures`` caps the keypoint budget: per-tile for image detection
        (spatially dense) and higher for the small template crop. ``None`` uses
        the per-tile default.

        Raises:
            ValueError: on an unknown detector name.
        """
        cv2 = self._cv2
        n = (name or "orb").lower()
        nf = _FEATURES_PER_TILE if nfeatures is None else int(nfeatures)
        if n == "orb":
            # Reduced ``edgeThreshold``: ORB's default (31 px) discards every
            # keypoint within 31 px of any border, which on a modest template
            # (e.g. a 56 px crop) leaves *zero* keypoints. Shrinking it keeps
            # keypoints near small-template edges while staying identical between
            # image and template detection so descriptors are comparable.
            # ``fastThreshold`` is lowered so faint synthetic texture still fires.
            return (
                cv2.ORB_create(nfeatures=nf, edgeThreshold=15, fastThreshold=10),
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
        MAX_FEATURE_LEVEL)`` — the deepest level at which the template's shorter
        side, after downsampling by ``2^L``, is still ≥ ~``MIN_TPL_FEATURE`` px
        (enough texture for ORB). ``L == 0`` for any template whose short side is
        below ``2 * MIN_TPL_FEATURE``, reproducing today's full-resolution
        behaviour exactly on small templates/images.

        The cap is deliberately conservative: the template is NEVER decimated
        below ``MIN_TPL_FEATURE`` (the regression that made feature matching find
        nothing on large images, §J.3/§K.2), since ``2^L ≤ min(th,tw)/MIN_TPL``
        implies ``min(th,tw)/2^L ≥ MIN_TPL_FEATURE``.
        """
        short = max(1, min(int(th), int(tw)))
        if short < 2 * _MIN_TPL_FEATURE:
            # log2(ratio) < 1 → floor is 0; short-circuit (avoids a log of <1).
            return 0
        # floor(log2(short / MIN_TPL_FEATURE)): the largest L with
        # short / 2^L >= MIN_TPL_FEATURE, i.e. 2^L <= short / MIN_TPL_FEATURE.
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
            # Up to 16 evenly-spaced samples; cheap and stable across calls.
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
        ``cv2.INTER_AREA`` (the area-average resampler — anti-aliases, so coarse
        keypoints stay repeatable) and detection runs on that smaller image; every
        kept keypoint's coordinate is then multiplied by ``2^level`` so the cache
        always stores **full-resolution** keypoint coordinates (§K.2). ``level==0``
        is the original full-resolution tiled detection, bit-for-bit unchanged.

        Each tile is detected with its own keypoint budget so features stay
        spatially uniform across a gigapixel image (a single global cap would
        starve small instances). Tiles overlap by ``_DETECT_OVERLAP`` on every
        side so a keypoint near a seam is detected with its full descriptor
        patch; each keypoint is then assigned to exactly one tile's *core*, so no
        duplicate descriptors land at seams (duplicates would defeat the Lowe
        ratio test there). Keypoint coordinates are offset to absolute image px.

        Returns ``(kps, desc)``, or ``(None, None)`` if ``cancel()`` fired
        mid-detection (so the caller does not cache a partial result).
        """
        cv2 = self._cv2
        factor = 1 << int(level)  # 2^level
        if factor > 1:
            fh, fw = gray.shape[:2]
            # INTER_AREA downsample by 2^level (§K.2). Detection then runs on the
            # smaller image; keypoints are scaled back to full-res coords below.
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
                # Detection window = core + overlap on every side (clamped).
                ox0, oy0 = max(0, tx - ov), max(0, ty - ov)
                ox1, oy1 = min(w, tx + step + ov), min(h, ty + step + ov)
                sub = np.ascontiguousarray(det_img[oy0:oy1, ox0:ox1])
                kp, desc = detector.detectAndCompute(sub, None)
                if desc is None or kp is None or len(kp) == 0:
                    continue
                # Core bounds (half-open) this tile owns.
                cx1, cy1 = min(w, tx + step), min(h, ty + step)
                keep: list[int] = []
                for i, k in enumerate(kp):
                    ax, ay = k.pt[0] + ox0, k.pt[1] + oy0
                    if tx <= ax < cx1 and ty <= ay < cy1:
                        # Scale the detection-level coordinate back to full-res so
                        # the cache always speaks full-resolution image px (§K.2).
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

        Runs the tiled detector once per ``(image, detector, level)`` on the image
        downsampled by ``2^level`` (§K.2) and caches the result so subsequent
        queries with the same detector AND level reuse it. Keypoints are stored in
        absolute **full-resolution** image coordinates regardless of the detection
        level, so all downstream geometry (RANSAC, bbox clamp) is full-res.

        Returns ``(image_kps, image_desc)``.
        """
        key = ((detector_name or "orb").lower(), int(level))
        # Defense-in-depth (§U2): if the image array differs from the one the cache
        # was built against (caller reused the matcher without invalidate_image),
        # drop every cached detector/level so we re-detect against the current
        # image rather than returning stale keypoints/descriptors.
        fingerprint = self._fingerprint(image_gray)
        if self._image_fingerprint is not None and fingerprint != self._image_fingerprint:
            self._image_cache.clear()
            self._image_shape = None
        self._image_fingerprint = fingerprint

        cached = self._image_cache.get(key)
        if cached is not None:
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
        exclude_box: tuple[int, int, int, int] | None = None,
        cancel: Callable[[], bool] | None = None,
        progress: Callable[[int], None] | None = None,
        scale_back: int = 1,
    ) -> list[Match]:
        """Find every (possibly warped) instance of ``template`` in ``image_gray``.

        Args:
            template: ``(th, tw, 3)`` uint8 RGB or ``(th, tw)`` uint8 grayscale
                template crop (color is collapsed to grayscale — feature matching
                ignores ``channel_mode``, §J.3).
            image_gray: The staged full-image grayscale ``(H, W)`` uint8 the engine
                passes (its BT.601 luminance). Detection runs on a downsampled
                copy; boxes are mapped back to this frame.
            params: A ``MatchParams`` (``feature_detector``/``feature_ratio``/
                ``feature_min_inliers``/``feature_max_instances``/``exclude_iou``
                are read; ``scales``/``channel_mode`` are ignored, §J.3).
            exclude_box: ``(x, y, w, h)`` source region (full-res) to exclude.
            cancel: Polled during detection and between RANSAC iterations.
            progress: Called with ``0..100`` (~0-50 detection, ~50-100 instance
                loop) per §J.3 step 7.
            scale_back: Extra integer factor mapping ``image_gray`` coordinates to
                the caller's full-resolution frame (1 when the engine passes the
                full-res luminance directly). Combined with the internal
                detection-scale factor when emitting boxes.

        Returns:
            ``list[Match]`` sorted by score descending, source region excluded, in
            the caller's full-resolution image px. ``scale`` is approximated as
            ``sqrt(bbox_area / template_area)`` (§J.3 step 7).

        Raises:
            RuntimeError: if OpenCV was unavailable (raised at construction).
        """
        cv2 = self._cv2

        detector_name = getattr(params, "feature_detector", "orb")
        ratio = float(getattr(params, "feature_ratio", 0.75))
        # cv2.findHomography needs >= 4 correspondences; clamp once here so the
        # loop guard and the n_inliers test both honour the hard >= 4 floor and
        # a tiny feature_min_inliers (e.g. 2) cannot drive RANSAC into a crash.
        min_inliers = max(4, int(getattr(params, "feature_min_inliers", 12)))
        max_instances = int(getattr(params, "feature_max_instances", 100))
        exclude_iou = float(getattr(params, "exclude_iou", 0.30))
        nms_iou = float(getattr(params, "nms_iou", 0.30))

        if progress is not None:
            progress(0)
        if cancel is not None and cancel():
            return []

        # --- 0. select the coarsest-safe detection level from the template ---
        # The template (NOT the image) picks the level: L is the deepest pyramid
        # level at which the downsampled template's short side stays ≥
        # MIN_TPL_FEATURE px, so the template is never decimated into oblivion
        # (the regression that made features find nothing on large images, §K.2).
        tmpl_gray_full = self._to_gray(template)
        th_full, tw_full = tmpl_gray_full.shape[:2]
        if min(th_full, tw_full) < _MIN_TEMPLATE_SIDE:
            return []  # too small to anchor a homography
        level = 0 if self._force_full_level else self._select_level(th_full, tw_full)
        self._last_level = level
        factor = 1 << level  # 2^level

        # --- 1-2. image features at level L (cached per (detector, level)) ---
        # Image keypoints come back in FULL-RES coordinates (the detector scaled
        # them ×2^level), so all geometry below is full-resolution regardless of
        # the detection level.
        img_kps, img_desc = self._ensure_image_features(
            image_gray, detector_name, level, cancel
        )
        if progress is not None:
            progress(50)
        if cancel is not None and cancel():
            return []
        if img_desc is None or len(img_kps) < min_inliers:
            return []  # nothing detectable in the image for this detector

        # --- 3. template features at level L ---------------------------------
        # Detect the template downsampled by 2^level (INTER_AREA) so its
        # descriptors are scale-consistent with the level-L image detection, then
        # scale the keypoints back to full-res template coords (×2^level). At
        # level 0 this is identical to the original full-resolution detection, so
        # small templates/images are unchanged (§K.2). The level bound guarantees
        # the downsampled template's short side stays ≥ MIN_TPL_FEATURE px, so it
        # still carries enough texture for ORB — never the decimation bug.
        if factor > 1:
            tdw = max(1, tw_full // factor)
            tdh = max(1, th_full // factor)
            tmpl_det = cv2.resize(
                tmpl_gray_full, (tdw, tdh), interpolation=cv2.INTER_AREA
            )
        else:
            tmpl_det = tmpl_gray_full

        detector, norm = self._make_detector(detector_name, nfeatures=_TEMPLATE_FEATURES)
        tmpl_kps, tmpl_desc = detector.detectAndCompute(tmpl_det, None)
        tmpl_kps = list(tmpl_kps) if tmpl_kps is not None else []
        if tmpl_desc is None or len(tmpl_kps) < min_inliers:
            return []
        if factor > 1:
            # Map template keypoints back to full-res template coords so the
            # homography is fit in a single full-resolution frame (template and
            # image both full-res), and the projected box is already full-res.
            for k in tmpl_kps:
                k.pt = (k.pt[0] * factor, k.pt[1] * factor)

        # --- multi-instance descriptor matching (NOT the Lowe ratio test) ----
        # Keep EVERY image match within an absolute "good" descriptor distance of
        # each template feature — one correspondence per copy — so repeated /
        # identical instances survive (the ratio test would reject them; see the
        # _GOOD_DIST note). RANSAC below provides the geometric verification.
        bf = cv2.BFMatcher(norm)
        maxd = _GOOD_DIST.get((detector_name or "orb").lower(), 64.0)
        # feature_ratio (UI, default 0.75) widens/narrows acceptance: at 0.75 it
        # keeps the nominal threshold; higher = more permissive.
        maxd *= max(0.25, ratio / 0.75)
        matched = bf.radiusMatch(tmpl_desc, img_desc, maxDistance=float(maxd))
        good = []
        for ms in matched:
            if not ms:
                continue
            # radiusMatch results are not guaranteed sorted; keep the nearest few
            # per feature (cap the fan-out so a repetitive texture can't explode
            # the correspondence set).
            if len(ms) > _MAX_MATCHES_PER_FEATURE:
                ms = sorted(ms, key=lambda m: m.distance)[:_MAX_MATCHES_PER_FEATURE]
            good.extend(ms)
        if len(good) < min_inliers:
            return []

        # Template corners (full-res) for projection -> bbox.
        tmpl_corners = np.array(
            [[0, 0], [tw_full, 0], [tw_full, th_full], [0, th_full]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        tmpl_area_full = float(th_full * tw_full)

        # Image extent (full res) for clamping projected corners.
        full_h, full_w = self._image_shape  # type: ignore[misc]

        # --- 4. generalized-Hough instance voting + per-cluster homography ---
        # Without the ratio test the correspondence set is large and ambiguous
        # (each template feature matches one keypoint per copy, plus noise), so a
        # plain sequential RANSAC over the whole set drowns and finds almost
        # nothing on repeated instances. Instead, each correspondence VOTES for an
        # instance centre using the matched keypoints' scale and orientation (a
        # similarity transform, à la Lowe 2004): the correct correspondences for a
        # given copy all vote for the same centre, producing a sharp peak per
        # instance that stands well above the diffuse noise. We then verify/refine
        # each peak's correspondences with a single homography.
        total_f = max(1, int(scale_back))
        results: list[Match] = []

        n = len(good)
        tpx = np.empty(n, np.float32); tpy = np.empty(n, np.float32)
        ipx = np.empty(n, np.float32); ipy = np.empty(n, np.float32)
        rel_s = np.empty(n, np.float32); rel_a = np.empty(n, np.float32)
        for i, m in enumerate(good):
            tk = tmpl_kps[m.queryIdx]; ik = img_kps[m.trainIdx]
            tpx[i], tpy[i] = tk.pt
            ipx[i], ipy[i] = ik.pt
            ts = tk.size if tk.size > 1e-3 else 1.0
            rel_s[i] = ik.size / ts
            # ORB/SIFT keypoint angle is degrees in [0,360); -1 means unset.
            da = (ik.angle - tk.angle) if (ik.angle >= 0 and tk.angle >= 0) else 0.0
            rel_a[i] = np.deg2rad(da)
        # Predicted instance centre = img_kp + s*R(dθ)*(template_centre - tmpl_kp).
        vx = (0.5 * tw_full) - tpx
        vy = (0.5 * th_full) - tpy
        cos = np.cos(rel_a); sin = np.sin(rel_a)
        pcx = ipx + rel_s * (cos * vx - sin * vy)
        pcy = ipy + rel_s * (sin * vx + cos * vy)

        # Vote into a coarse 2-D histogram (bin ≈ a quarter of the template's
        # short side, so one instance's votes land in one bin / its neighbours).
        binsz = max(8, int(min(tw_full, th_full) // 4))
        nbx = int(full_w // binsz) + 2
        nby = int(full_h // binsz) + 2
        valid = (pcx >= 0) & (pcx < full_w) & (pcy >= 0) & (pcy < full_h)
        bx = np.clip((pcx / binsz).astype(np.int64), 0, nbx - 1)
        by = np.clip((pcy / binsz).astype(np.int64), 0, nby - 1)
        hist = np.zeros((nby, nbx), np.int32)
        np.add.at(hist, (by[valid], bx[valid]), 1)

        if int(hist.max()) >= min_inliers:
            # Stop once peaks fall to the diffuse-noise floor; the homography
            # verification below rejects any noise cluster that slips through.
            vote_floor = max(min_inliers, int(0.12 * int(hist.max())))
            used = np.zeros(n, bool)
            it = 0
            it_cap = 4 * max_instances + 8
            while len(results) < max_instances and it < it_cap:
                it += 1
                if cancel is not None and cancel():
                    break
                py, px = np.unravel_index(int(np.argmax(hist)), hist.shape)
                peak = int(hist[py, px])
                if peak < vote_floor:
                    break
                # Gather this peak's correspondences (its bin + 8 neighbours) and
                # consume them whether or not they yield an instance (so the loop
                # always makes progress and cannot spin on the same peak).
                cl = (
                    (~used)
                    & valid
                    & (np.abs(bx - px) <= 1)
                    & (np.abs(by - py) <= 1)
                )
                idx = np.nonzero(cl)[0]
                used[cl] = True
                if idx.size:
                    np.add.at(hist, (by[idx], bx[idx]), -1)
                if idx.size < min_inliers:
                    continue
                src_pts = np.stack([tpx[idx], tpy[idx]], 1).reshape(-1, 1, 2)
                dst_pts = np.stack([ipx[idx], ipy[idx]], 1).reshape(-1, 1, 2)
                try:
                    H, mask = cv2.findHomography(
                        src_pts, dst_pts, cv2.RANSAC, _REPROJ_THRESH,
                        maxIters=_RANSAC_ITERS, confidence=0.999,
                    )
                except cv2.error:
                    continue
                if H is None or mask is None:
                    continue
                n_inliers = int(mask.ravel().sum())
                if n_inliers < min_inliers:
                    continue
                box = self._project_to_box(cv2, H, tmpl_corners, full_w, full_h)
                if box is None:
                    continue
                bx_s, by_s, bw_s, bh_s = box
                rx = max(0, min(int(round(bx_s * total_f)), full_w))
                ry = max(0, min(int(round(by_s * total_f)), full_h))
                rw = max(1, min(int(round(bw_s * total_f)), full_w - rx))
                rh = max(1, min(int(round(bh_s * total_f)), full_h - ry))
                # 5. score saturates toward 1.0 for richly-supported instances.
                score = min(1.0, n_inliers / (2.0 * max(1, min_inliers)))
                # 7. scale ~= sqrt(bbox_area / template_area).
                area = float(rw * rh)
                scale = (
                    float(np.sqrt(area / tmpl_area_full)) if tmpl_area_full > 0 else 1.0
                )
                results.append(
                    Match(x=rx, y=ry, w=rw, h=rh, score=float(score), scale=scale)
                )
                if progress is not None:
                    progress(
                        min(100, 50 + int(round(50 * len(results) / max(1, max_instances))))
                    )

        if progress is not None:
            progress(100)

        # --- 6. source exclusion (IoU > exclude_iou or center inside) -------
        if exclude_box is not None:
            results = [
                r
                for r in results
                if not self._excluded(r, exclude_box, exclude_iou)
            ]

        # Sort by score descending to match the conv path's contract (§C.5).
        results.sort(key=lambda r: r.score, reverse=True)
        # Global IoU-NMS: sequential RANSAC can fit two near-identical homographies
        # to one physical instance and emit overlapping duplicate boxes. The conv
        # path's score map gets per-pixel NMS; the feature path bypasses it, so
        # suppress overlaps here using the box IoU (not centre distance, which
        # would wrongly merge two distinct touching instances) — §C.5/§J.3.
        results = self._nms(results, nms_iou)
        return results

    # -- helpers -------------------------------------------------------------

    def _to_gray(self, arr: np.ndarray) -> np.ndarray:
        """Collapse a template to a C-contiguous uint8 grayscale image.

        Uses BT.601 luminance for RGB input so it lines up with the staged image
        grayscale the engine passes; a 2-D input is taken as-is.
        """
        if arr.ndim == 2:
            return np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        if arr.ndim == 3 and arr.shape[2] == 3:
            rgb = arr.astype(np.float32)
            # BT.601 weights (R, G, B) — same convention as the engine's _BT601.
            lum = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
            return np.ascontiguousarray(np.clip(np.rint(lum), 0, 255).astype(np.uint8))
        raise ValueError(f"template must be (th,tw,3) or (th,tw), got shape {arr.shape}")

    def _project_to_box(
        self,
        cv2,
        H: np.ndarray,
        tmpl_corners: np.ndarray,
        iw: int,
        ih: int,
    ) -> tuple[float, float, float, float] | None:
        """Project the template corners through ``H`` and return an AABB.

        Applies the degeneracy guards (§J.3 step 4): the affine part of ``H`` must
        be well-conditioned (det not near-zero/negative — preserves orientation),
        the projected corner quad must be convex with a positive, sane-scale area.
        Returns ``(x, y, w, h)`` (detection-scale, clamped to the image) or
        ``None`` if any guard fails.
        """
        # Affine-part determinant: reject near-singular / orientation-flipping
        # (folded) maps. A negative det means the quad is mirrored — not a real
        # same-handedness instance.
        a, b, c, d = float(H[0, 0]), float(H[0, 1]), float(H[1, 0]), float(H[1, 1])
        det = a * d - b * c
        if det <= _MIN_AFFINE_DET:
            return None
        # Sane absolute scale: reject runaway blow-ups / collapses (§J.3 step 4).
        # ``det`` is the *area* scale of the affine part; convert to a *linear*
        # (per-axis) scale before comparing against the linear _MIN/_MAX bounds,
        # otherwise a real 5x copy (det=25) would trip the threshold far too early.
        lin = det ** 0.5
        if lin > _MAX_SCALE or lin < _MIN_SCALE:
            return None

        proj = cv2.perspectiveTransform(tmpl_corners, H)
        if proj is None:
            return None
        quad = proj.reshape(-1, 2)
        if not np.all(np.isfinite(quad)):
            return None

        # Convex + positive area via the shoelace formula on the (ordered) quad.
        area2 = self._signed_area2(quad)
        if area2 <= 0:  # zero/negative -> degenerate or flipped
            return None
        if not self._is_convex(quad):
            return None

        xs = quad[:, 0]
        ys = quad[:, 1]
        x0 = float(np.clip(xs.min(), 0, iw))
        y0 = float(np.clip(ys.min(), 0, ih))
        x1 = float(np.clip(xs.max(), 0, iw))
        y1 = float(np.clip(ys.max(), 0, ih))
        w = x1 - x0
        h = y1 - y0
        if w < 1 or h < 1:
            return None
        return x0, y0, w, h

    @staticmethod
    def _signed_area2(quad: np.ndarray) -> float:
        """Twice the signed area of a polygon (shoelace); >0 for CCW ordering."""
        x = quad[:, 0]
        y = quad[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def _is_convex(quad: np.ndarray) -> bool:
        """True if the ordered quad is convex (all cross products same sign)."""
        n = len(quad)
        if n < 4:
            return False
        signs = []
        for i in range(n):
            p0 = quad[i]
            p1 = quad[(i + 1) % n]
            p2 = quad[(i + 2) % n]
            e0 = p1 - p0
            e1 = p2 - p1
            cross = e0[0] * e1[1] - e0[1] * e1[0]
            signs.append(cross)
        signs = np.array(signs)
        # Convex iff every turn goes the same way (ignore ~0 collinear edges).
        pos = np.all(signs >= -1e-6)
        neg = np.all(signs <= 1e-6)
        return bool(pos or neg)

    @classmethod
    def _nms(cls, results: list[Match], nms_iou: float) -> list[Match]:
        """Greedy box-IoU non-max suppression over a score-sorted match list.

        Walks the (already score-descending) ``results`` keeping each box unless
        its IoU with a higher-scored, already-kept box exceeds ``nms_iou`` — the
        feature path's stand-in for the conv path's per-pixel NMS so one physical
        instance fit twice by sequential RANSAC collapses to a single box. Uses
        the box IoU (area overlap), so two genuinely distinct instances that merely
        sit close are kept (their IoU stays low). Pure numpy/python (§U1).
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
        # IoU of the two half-open boxes.
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
        # Center-inside test.
        cx = m.x + m.w / 2.0
        cy = m.y + m.h / 2.0
        return (ex <= cx < ex + ew) and (ey <= cy < ey + eh)
