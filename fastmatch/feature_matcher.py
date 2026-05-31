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

1. **Detection scale** — gigapixel feature detection is heavy, so the image (and
   template) are downsampled by an integer factor ``f`` chosen so the image's
   longest side is ``<= FEATURE_MAX_DIM``. Detection/matching happen at that
   scale and result boxes are mapped back to full-res by ``* f`` (and then by the
   caller's ``scale_back`` if the passed image was itself a downscale).
2. **Detector** — ORB (default), AKAZE, or SIFT on the grayscale image. Image
   keypoints/descriptors are computed **once per (image, detector)** and cached
   on this matcher; :meth:`invalidate_image` clears the cache when the engine
   stages a new image.
3. **Per query** — detect template keypoints/descriptors, ``knnMatch(k=2)``
   template->image with the right norm (HAMMING for ORB/AKAZE, L2 for SIFT), and
   keep matches passing the Lowe ratio test.
4. **Sequential RANSAC homography** — repeatedly fit a homography on the surviving
   matches; if it has enough inliers and passes degeneracy guards, record the
   instance (template-corner quad -> axis-aligned bbox), remove its inliers, and
   loop until too few matches remain or ``feature_max_instances`` is hit.
5. **Score** — ``min(1.0, inliers / (2 * feature_min_inliers))`` (exact copies
   saturate near 1.0; ambiguous ones score lower).
6. **Exclude source** — same rule as the conv path (IoU > ``exclude_iou`` or
   center inside ``exclude_box``).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .types import Match

# Cap the longest image side fed to the detector. Feature detection on a
# gigapixel image is both slow and memory-hungry; downsampling to <= 3000 px on
# the long edge keeps detection fast while preserving enough keypoints, and the
# integer downsample factor maps boxes back cleanly (§J.3 step 1).
FEATURE_MAX_DIM = 3000

#: RANSAC reprojection threshold (px, at detection scale). Inliers must lie
#: within this distance of the projected model — a few px tolerates the
#: sub-pixel keypoint and homography noise without admitting outliers.
_REPROJ_THRESH = 5.0

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
        # Cache of {detector_name: (down_f, kps, desc)} for the staged image. The
        # downsample factor is stored alongside so a detector switch reuses the
        # same factor and box-mapping math. Cleared by invalidate_image().
        self._image_cache: dict[str, tuple[int, list[Any], np.ndarray | None]] = {}
        # The grayscale image the cache was built against (downsampled). Stored so
        # corner projection / clamping can use the detection-scale image extent.
        self._image_gray: np.ndarray | None = None
        self._image_shape: tuple[int, int] | None = None  # (H, W) at full res
        # Cheap fingerprint of the image the cache was built against. Defense in
        # depth: the engine calls invalidate_image() in set_image, but if a caller
        # reuses this matcher on a *different* array of the same detector without
        # invalidating, a stale fingerprint mismatch forces a re-detect (§U2).
        self._image_fingerprint: tuple[Any, ...] | None = None

    # -- cache management ----------------------------------------------------

    def invalidate_image(self) -> None:
        """Drop any cached per-image keypoints/descriptors (§J.3/§J.4).

        Called by ``Matcher.set_image`` so a freshly staged image is re-detected
        on the next feature query rather than reusing stale features.
        """
        self._image_cache.clear()
        self._image_gray = None
        self._image_shape = None
        self._image_fingerprint = None

    # -- detector factory ----------------------------------------------------

    def _make_detector(self, name: str):
        """Build the requested OpenCV detector and its descriptor norm.

        Returns ``(detector, norm)`` where ``norm`` is the BFMatcher distance for
        that descriptor type: HAMMING for the binary ORB/AKAZE descriptors, L2
        for SIFT's float descriptors (§J.3 step 3).

        Raises:
            ValueError: on an unknown detector name.
        """
        cv2 = self._cv2
        n = (name or "orb").lower()
        if n == "orb":
            # A generous feature budget so large templates match densely, plus a
            # reduced ``edgeThreshold``: ORB's default (31 px) discards every
            # keypoint within 31 px of any border, which on a modest template
            # (e.g. a 56 px crop) leaves *zero* keypoints. Shrinking it keeps
            # keypoints near small-template edges while staying identical between
            # the image and the template detection so descriptors are comparable.
            # ``fastThreshold`` is lowered so faint synthetic texture still fires.
            return (
                cv2.ORB_create(nfeatures=5000, edgeThreshold=15, fastThreshold=10),
                cv2.NORM_HAMMING,
            )
        if n == "akaze":
            return cv2.AKAZE_create(), cv2.NORM_HAMMING
        if n == "sift":
            return cv2.SIFT_create(), cv2.NORM_L2
        raise ValueError(f"unknown feature_detector {name!r}; expected orb/akaze/sift")

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

    @staticmethod
    def _downsample_factor(h: int, w: int) -> int:
        """Smallest integer ``f >= 1`` so ``max(h, w) / f <= FEATURE_MAX_DIM``."""
        longest = max(h, w)
        if longest <= FEATURE_MAX_DIM:
            return 1
        # ceil division gives the smallest factor that brings the long edge under
        # the cap; integer factor keeps the box-mapping math exact (§J.3 step 1).
        return int((longest + FEATURE_MAX_DIM - 1) // FEATURE_MAX_DIM)

    def _downscale(self, gray: np.ndarray, f: int) -> np.ndarray:
        """Integer-factor area-average downscale of a grayscale image."""
        if f <= 1:
            return np.ascontiguousarray(gray)
        cv2 = self._cv2
        h, w = gray.shape[:2]
        # INTER_AREA is the right resampler for shrinking (anti-aliased).
        return cv2.resize(
            gray, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA
        )

    # -- image-feature cache -------------------------------------------------

    def _ensure_image_features(
        self,
        image_gray: np.ndarray,
        detector_name: str,
        cancel: Callable[[], bool] | None,
    ) -> tuple[int, list[Any], np.ndarray | None]:
        """Detect+cache the downsampled image's keypoints/descriptors per detector.

        Computes the integer downsample factor once, downsamples the image, and
        runs ``detectAndCompute`` a single time per ``(image, detector)`` — the
        result is cached so subsequent queries with the same detector reuse it.

        Returns ``(down_f, image_kps, image_desc)`` at detection scale.
        """
        key = (detector_name or "orb").lower()
        # Defense-in-depth (§U2): if the image array differs from the one the cache
        # was built against (caller reused the matcher without invalidate_image),
        # drop every cached detector so we re-detect against the current image
        # rather than returning stale keypoints/descriptors.
        fingerprint = self._fingerprint(image_gray)
        if self._image_fingerprint is not None and fingerprint != self._image_fingerprint:
            self._image_cache.clear()
            self._image_gray = None
            self._image_shape = None
        self._image_fingerprint = fingerprint

        cached = self._image_cache.get(key)
        if cached is not None:
            return cached

        h, w = image_gray.shape[:2]
        self._image_shape = (h, w)
        f = self._downsample_factor(h, w)
        small = self._downscale(image_gray, f)
        self._image_gray = small

        if cancel is not None and cancel():
            # Don't cache a cancelled (potentially partial) detection.
            return f, [], None

        detector, _norm = self._make_detector(detector_name)
        kps, desc = detector.detectAndCompute(small, None)
        kps = list(kps) if kps is not None else []
        result = (f, kps, desc)
        # Only cache a real detection (post-cancel guard above already returned).
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

        # --- 1-2. image features (cached per detector) ----------------------
        down_f, img_kps, img_desc = self._ensure_image_features(
            image_gray, detector_name, cancel
        )
        if progress is not None:
            progress(50)
        if cancel is not None and cancel():
            return []
        if img_desc is None or len(img_kps) < min_inliers:
            return []  # nothing detectable in the image for this detector

        # --- 3. template features at the same detection scale ---------------
        tmpl_gray = self._to_gray(template)
        tmpl_small = self._downscale(tmpl_gray, down_f)
        th_small, tw_small = tmpl_small.shape[:2]
        # Degenerate templates (smaller than a few px after downscale) cannot
        # define a homography-supporting corner quad.
        if th_small < 2 or tw_small < 2:
            return []

        detector, norm = self._make_detector(detector_name)
        tmpl_kps, tmpl_desc = detector.detectAndCompute(tmpl_small, None)
        tmpl_kps = list(tmpl_kps) if tmpl_kps is not None else []
        if tmpl_desc is None or len(tmpl_kps) < min_inliers:
            return []

        # --- knnMatch + Lowe ratio test -------------------------------------
        bf = cv2.BFMatcher(norm)
        # k=2 so the ratio test can compare best vs second-best (§J.3 step 3).
        knn = bf.knnMatch(tmpl_desc, img_desc, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, second = pair[0], pair[1]
            if m.distance < ratio * second.distance:
                good.append(m)
        if len(good) < min_inliers:
            return []

        # Template corners (at detection scale) for projection -> bbox.
        tmpl_corners = np.array(
            [[0, 0], [tw_small, 0], [tw_small, th_small], [0, th_small]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        # Full-res template area for the scale estimate.
        th_full, tw_full = tmpl_gray.shape[:2]
        tmpl_area_full = float(th_full * tw_full)

        # Detection-scale image extent for clamping projected corners.
        ih_small, iw_small = self._image_gray.shape[:2]  # type: ignore[union-attr]
        full_h, full_w = self._image_shape  # type: ignore[misc]

        # --- 4. sequential RANSAC multi-instance loop -----------------------
        # Total full-res-frame scale factor: detection downsample * caller's.
        total_f = down_f * max(1, int(scale_back))

        results: list[Match] = []
        remaining = list(good)
        while len(remaining) >= min_inliers and len(results) < max_instances:
            if cancel is not None and cancel():
                break

            src_pts = np.float32(
                [tmpl_kps[m.queryIdx].pt for m in remaining]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [img_kps[m.trainIdx].pt for m in remaining]
            ).reshape(-1, 1, 2)

            try:
                H, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, _REPROJ_THRESH
                )
            except cv2.error:
                # Belt-and-suspenders: a degenerate point set (e.g. < 4 usable
                # correspondences) raises rather than returning None — stop the
                # sequential loop cleanly instead of propagating the cv2.error.
                break
            if H is None or mask is None:
                break
            inlier_mask = mask.ravel().astype(bool)
            n_inliers = int(inlier_mask.sum())
            if n_inliers < min_inliers:
                break

            box = self._project_to_box(
                cv2, H, tmpl_corners, iw_small, ih_small
            )
            if box is not None:
                bx_s, by_s, bw_s, bh_s = box
                # Map detection-scale bbox to full-resolution and clamp to image.
                bx = int(round(bx_s * total_f))
                by = int(round(by_s * total_f))
                bw = int(round(bw_s * total_f))
                bh = int(round(bh_s * total_f))
                bx = max(0, min(bx, full_w))
                by = max(0, min(by, full_h))
                bw = max(1, min(bw, full_w - bx))
                bh = max(1, min(bh, full_h - by))

                # 5. score saturates toward 1.0 for richly-supported instances.
                score = min(1.0, n_inliers / (2.0 * max(1, min_inliers)))
                # 7. scale ~= sqrt(bbox_area / template_area).
                area = float(bw * bh)
                scale = float(np.sqrt(area / tmpl_area_full)) if tmpl_area_full > 0 else 1.0
                results.append(
                    Match(x=bx, y=by, w=bw, h=bh, score=float(score), scale=scale)
                )

            # Remove the inliers consumed by this instance, then look for the next
            # one in what's left (sequential RANSAC, §J.3 step 4).
            remaining = [m for m, keep in zip(remaining, inlier_mask) if not keep]

            if progress is not None:
                # Coarse 50->100 sweep across the instance budget.
                frac = len(results) / max(1, max_instances)
                progress(min(100, 50 + int(round(50 * frac))))

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
