"""Local instance retrieval with RootSIFT and geometric verification.

Unlike the strict feature matcher, acceptance never depends on pixel NCC.
Image descriptors query the *template* index: repeated copies do not compete
for a fixed number of neighbours. Detection is tiled and cached; descriptor
queries and RANSAC proposals are bounded and cancellable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .feature_matcher import FeatureMatcher, _TEMPLATE_FEATURES
from .types import Match, _ORIENT_LINEAR, active_orientations, apply_orientation, orientation_from_matrix


def rootsift(descriptors: np.ndarray) -> np.ndarray:
    """Hellinger embedding: L1-normalize nonnegative SIFT, then square root."""
    d = np.maximum(descriptors.astype(np.float32), 0)
    return np.ascontiguousarray(np.sqrt(d / np.maximum(d.sum(axis=1, keepdims=True), 1e-12)))


class FuzzyMatcher(FeatureMatcher):
    """RootSIFT retrieval, centre voting, then per-instance RANSAC homography."""

    def _detect_tiled(self, gray, detector_name, cancel, level=0):
        kps, desc = super()._detect_tiled(gray, "sift", cancel, level)
        return kps, None if desc is None else rootsift(desc)

    def match(self, template, image_gray, params, *, image_rgb=None,
              exclude_box=None, mask=None, cancel=None, progress=None, scale_back=1):
        cv2 = self._cv2
        if cancel is not None and cancel():
            return []
        gray = self._to_gray(template)
        if min(gray.shape) < 12:
            return []
        ik, desc = self._ensure_image_features(image_gray, "sift", 0, cancel)
        if progress is not None:
            progress(30)
        if desc is None or (cancel is not None and cancel()):
            return []
        xy = np.array([k.pt for k in ik], np.float32)
        sizes = np.array([k.size for k in ik], np.float32)
        angles = np.array([k.angle for k in ik], np.float32)
        active = active_orientations(params.enable_rotation, params.enable_flipping)
        # SIFT is rotation invariant. One reflected variant covers every mirror;
        # final transform classification enforces the selected D4 orientations.
        variants = ("R0", "MY") if params.enable_flipping else ("R0",)
        results = []
        min_support = max(4, int(params.feature_min_inliers))
        for variant in variants:
            if cancel is not None and cancel():
                return []
            oriented = np.ascontiguousarray(apply_orientation(gray, variant))
            oriented_mask = None if mask is None else np.ascontiguousarray(
                apply_orientation(mask.astype(np.uint8) * 255, variant))
            detector, _ = self._make_detector("sift", _TEMPLATE_FEATURES)
            tk, td = detector.detectAndCompute(oriented, oriented_mask)
            if td is None or len(tk) < min_support:
                continue
            td = rootsift(td)
            # Approximate search costs scale with image descriptors, not with
            # every image descriptor times every template descriptor.
            index = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
            index.add([td])
            index.train()
            qi, ti, distances = [], [], []
            ratio = min(0.98, max(0.5, float(params.feature_ratio) + 0.15))
            for start in range(0, len(desc), 2048):
                if cancel is not None and cancel():
                    return []
                neighbours = index.knnMatch(desc[start:start + 2048], k=2)
                for offset, pair in enumerate(neighbours):
                    if len(pair) < 2:
                        continue
                    first, second = pair
                    # Exact/near-exact matches may be internally repetitive in
                    # the template too. Let geometric consistency resolve them.
                    if first.distance <= 0.65 and (
                        first.distance < ratio * second.distance or first.distance < 0.18
                    ):
                        qi.append(first.trainIdx)
                        ti.append(start + offset)
                        distances.append(first.distance)
                if progress is not None:
                    progress(30 + int(25 * (start + len(neighbours)) / len(desc)))
            if len(qi) < min_support:
                continue
            qi, ti = np.asarray(qi), np.asarray(ti)
            distances = np.asarray(distances, np.float32)
            src = np.array([k.pt for k in tk], np.float32)[qi]
            dst = xy[ti]
            ts = np.array([k.size for k in tk], np.float32)[qi]
            ta = np.array([k.angle for k in tk], np.float32)[qi]
            scale = sizes[ti] / np.maximum(ts, 1e-6)
            angle = np.deg2rad(angles[ti] - ta)
            th, tw = oriented.shape
            delta = np.array([tw / 2, th / 2]) - src
            c, s = np.cos(angle), np.sin(angle)
            centres = dst + scale[:, None] * np.stack(
                [c * delta[:, 0] - s * delta[:, 1], s * delta[:, 0] + c * delta[:, 1]], axis=1)
            cell = max(8, min(tw, th) / 4)
            bins = np.floor(centres / cell).astype(np.int64)
            groups = {}
            for j, key in enumerate(map(tuple, bins)):
                groups.setdefault(key, []).append(j)
            # Neighbourhood support avoids losing a copy at bin boundaries.
            candidates = []
            for key, members in groups.items():
                neighbourhood = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                 for j in groups.get((key[0] + dx, key[1] + dy), ())]
                if len(neighbourhood) >= min_support:
                    candidates.append((len(members), neighbourhood))
            candidates.sort(key=lambda p: -p[0])
            for number, (_, members) in enumerate(candidates[:3000]):
                if cancel is not None and cancel():
                    return []
                ids = np.asarray(members)
                if len(ids) > 800:
                    ids = ids[np.argsort(distances[ids])[:800]]
                # Already solved instances need no second homography fit.
                centre = np.median(centres[ids], axis=0)
                if any(r.x <= centre[0] < r.x + r.w and r.y <= centre[1] < r.y + r.h
                       for r in results):
                    continue
                candidate = self._fit(src[ids], dst[ids], qi[ids], distances[ids],
                                      tw, th, variant, len(tk), min_support, image_gray.shape)
                if candidate is None:
                    continue
                match, transform = candidate
                if match.orientation not in active:
                    continue
                if exclude_box is not None and self._excluded(match, exclude_box, params.exclude_iou):
                    continue
                if image_rgb is not None and template.ndim == 3 and params.channel_mode != "luminance":
                    match = replace(match, score=match.score * self._colour_similarity(
                        template, image_rgb, transform, variant, mask, params))
                if match.score >= params.threshold_floor:
                    results.append(match)
                if progress is not None:
                    progress(55 + int(44 * (number + 1) / max(1, min(3000, len(candidates)))))
        results.sort(key=lambda r: -r.score)
        results = self._nms(results, params.nms_iou)
        results = results[:max(0, min(params.max_results, params.feature_max_instances))]
        if scale_back != 1:
            results = [replace(r, x=r.x * scale_back, y=r.y * scale_back,
                               w=r.w * scale_back, h=r.h * scale_back) for r in results]
        if progress is not None:
            progress(100)
        return results

    def _fit(self, src, dst, query_ids, distances, tw, th, variant,
             template_count, min_support, image_shape):
        cv2 = self._cv2
        if len(np.unique(query_ids)) < min_support:
            return None
        transform, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0,
                                               maxIters=1500, confidence=0.995)
        if transform is None or inliers is None or not np.isfinite(transform).all():
            return None
        keep = inliers.ravel().astype(bool)
        # Several SIFT orientations at the same point cannot provide independent
        # geometric support; count distinct locations, not descriptor duplicates.
        unique_points = np.unique(np.round(src[keep], 1), axis=0)
        if len(unique_points) < min_support:
            return None
        support = src[keep]
        hull_area = cv2.contourArea(cv2.convexHull(support))
        if hull_area < 0.025 * tw * th:
            return None
        corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
        denominators = (np.c_[corners.reshape(-1, 2), np.ones(4)] @ transform[2])
        if np.any(np.abs(denominators) < 1e-6) or np.any(denominators * denominators[0] <= 0):
            return None
        quad = cv2.perspectiveTransform(corners, transform).reshape(4, 2)
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
            return None
        area = abs(cv2.contourArea(quad))
        linear_scale = np.sqrt(area / (tw * th))
        if not 0.1 <= linear_scale <= 10:
            return None
        lo, hi = np.floor(quad.min(axis=0)), np.ceil(quad.max(axis=0))
        ih, iw = image_shape
        if lo[0] < -1 or lo[1] < -1 or hi[0] > iw + 1 or hi[1] > ih + 1:
            return None
        width, height = hi - lo
        if width <= 0 or height <= 0 or area < 0.2 * width * height:
            return None
        # Evaluate the projective Jacobian at the template centre, rather than
        # the homography's top-left block (which misclassifies perspective).
        cx, cy = tw / 2, th / 2
        z = transform[2] @ [cx, cy, 1]
        projected = (transform @ [cx, cy, 1])[:2] / z
        jacobian = (transform[:2, :2] - projected[:, None] * transform[2, :2]) / z
        code = orientation_from_matrix(jacobian @ np.asarray(_ORIENT_LINEAR[variant]))
        predicted = cv2.perspectiveTransform(src[keep].reshape(-1, 1, 2), transform).reshape(-1, 2)
        error = float(np.median(np.linalg.norm(predicted - dst[keep], axis=1)))
        descriptor_score = 1 - float(np.median(distances[keep])) / np.sqrt(2)
        coverage = min(1.0, np.sqrt(len(np.unique(query_ids[keep])) / template_count))
        geometry = np.exp(-error / 4) * min(1.0, len(unique_points) / (2 * min_support))
        score = float(np.clip(0.45 * descriptor_score + 0.25 * coverage + 0.30 * geometry, 0, 1))
        return Match(int(max(0, lo[0])), int(max(0, lo[1])), int(width), int(height),
                     score, float(linear_scale), code), transform

    def _colour_similarity(self, template, image_rgb, transform, variant, mask, params):
        """Weighted Hellinger colour-histogram similarity, with no pixel gate."""
        cv2 = self._cv2
        reference = np.ascontiguousarray(apply_orientation(template, variant))
        th, tw = reference.shape[:2]
        patch = cv2.warpPerspective(image_rgb, transform, (tw, th), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        selected = np.ones((th, tw), bool) if mask is None else apply_orientation(mask, variant).astype(bool)
        if not selected.any():
            return 0.0
        if params.channel_mode == "ycbcr":
            reference, patch = self._rgb_to_ycbcr(reference), self._rgb_to_ycbcr(patch)
            weights = params.ycbcr_weights
        else:
            weights = params.rgb_weights
        from .types import normalize_weights
        similarities = []
        for channel in range(3):
            a = np.histogram(reference[:, :, channel][selected], bins=16, range=(0, 256))[0]
            b = np.histogram(patch[:, :, channel][selected], bins=16, range=(0, 256))[0]
            similarities.append(float(np.sqrt(a * b).sum() / max(1, selected.sum())))
        return float(np.dot(normalize_weights(weights), similarities))
