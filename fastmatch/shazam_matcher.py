"""Experimental "Shazam-style" patch matcher — DISABLED by default.

This is the graphics analogue of Shazam's constellation/landmark hashing:

    Shazam:   pairs of spectral peaks -> hash(f_anchor, f_point, dt) -> vote on a
              consistent time-offset per song.
    Here:     pairs of keypoints      -> hash(word_A, word_B, dx, dy) -> vote on a
              consistent (x,y) offset per template instance.

A keypoint's "word" is the *template feature it matches* (its descriptor identity
from a knn descriptor match). A landmark PAIR hash therefore encodes "feature wi
sits at relative offset (dx, dy) from feature wj", so each matching image pair
votes the implied instance offset; offsets with many votes are copies. Run once
per active D4 orientation (Shazam is not rotation/mirror invariant).

**Status: kept but DISABLED.** ``"shazam"`` is intentionally NOT in
:data:`fastmatch.types.METHODS`, so it never appears in the UI method dropdown.
The engine still dispatches ``method == "shazam"`` when it is set explicitly
(``fastmatch.types.SHAZAM_METHOD``), so the approach is preserved, importable, and
unit-tested — but it is off by every default path. It was benchmarked against the
saved NCC/SSD ground truth and **underperforms the feature method badly** (≈38%
recall vs 97%, and far slower), because a pair-hash vote needs two correctly-
matched keypoints with preserved geometry — feature-sparse repeated copies often
lack enough consistent pairs — and the inverted-index speedup only pays off at
database scale (many images), not for one-template / one-image search. To surface
it in the UI, add ``"shazam"`` to ``METHODS``/``METHOD_LABELS`` and the params
panel; nothing else is required.

Detection is reused from the OpenCV :class:`~fastmatch.feature_matcher.FeatureMatcher`
(dense ORB, cached per image), so this module only implements the pair-hash voting.
Spatial neighbour search is a plain numpy grid bucket (no SciPy dependency).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from .feature_matcher import _GOOD_DIST, _KNN_K, _TEMPLATE_FEATURES
from .types import Match, active_orientations, apply_orientation

# --- tunables (see the module docstring for the benchmark caveat) -----------
_PAIR_QUANT = 8.0        # px bin for a pair's relative (dx, dy) hash key
_PAIR_RADIUS_FRAC = 0.7  # pair keypoints within this fraction of the template's long side
_PAIR_RADIUS_CAP = 160   # but never look farther than this (keeps the target zone local/fast)
_PAIR_K = 12             # max spatial neighbours paired per anchor
_VOTE_BIN = 8            # px bin for the instance-offset vote accumulator
_MIN_VOTES = 3           # min pair-votes for an offset to be proposed an instance
_NMS_IOU = 0.30
_VERIFY_ZNCC = True      # gate voted peaks by appearance (ZNCC) for precision
_NCC_ACCEPT = 0.70


class ShazamMatcher:
    """Landmark-pair-hash + offset-voting matcher (experimental, disabled by default).

    Reuses a :class:`~fastmatch.feature_matcher.FeatureMatcher` for dense ORB
    detection (and its cv2 handle); call :meth:`match` with the image's cached
    keypoints/descriptors.
    """

    def __init__(self) -> None:
        try:
            import cv2  # noqa: F401
        except Exception as exc:  # ImportError or a broken cv2 install
            raise RuntimeError(
                "Shazam matching requires OpenCV: pip install opencv-python-headless"
            ) from exc
        self._cv2 = cv2

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _zncc(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32) - a.mean()
        b = b.astype(np.float32) - b.mean()
        da, db = np.sqrt(float((a * a).sum())), np.sqrt(float((b * b).sum()))
        return float((a * b).sum() / (da * db)) if da > 1e-6 and db > 1e-6 else 0.0

    @staticmethod
    def _iou(a, b) -> float:
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        u = a[2] * a[3] + b[2] * b[3] - inter
        return inter / u if u > 0 else 0.0

    @staticmethod
    def _neighbours(pos: np.ndarray, radius: float, k: int) -> list[list[int]]:
        """Grid-bucketed spatial neighbours within ``radius`` (≤ ``k`` each), no SciPy."""
        n = pos.shape[0]
        if n == 0:
            return []
        cell = max(1.0, radius)
        gk = np.floor(pos / cell).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in range(n):
            buckets[(int(gk[i, 0]), int(gk[i, 1]))].append(i)
        r2 = radius * radius
        out: list[list[int]] = []
        for i in range(n):
            cx, cy = int(gk[i, 0]), int(gk[i, 1])
            cand: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(buckets.get((cx + dx, cy + dy), ()))
            px, py = pos[i]
            near = [
                j for j in cand
                if j != i and (pos[j, 0] - px) ** 2 + (pos[j, 1] - py) ** 2 <= r2
            ]
            out.append(near[:k])
        return out

    # -- public match --------------------------------------------------------

    def match(
        self,
        template: np.ndarray,
        image_gray: np.ndarray,
        params: "Any",
        *,
        feature_matcher,
        img_kps: list,
        img_desc: np.ndarray | None,
        exclude_box: tuple[int, int, int, int] | None = None,
        cancel: Callable[[], bool] | None = None,
        progress: Callable[[int], None] | None = None,
        scale_back: int = 1,
    ) -> list[Match]:
        """Find template instances by landmark-pair hashing + offset voting.

        Args mirror :meth:`fastmatch.feature_matcher.FeatureMatcher.match` plus the
        shared image detection: ``feature_matcher`` supplies the detector/grayscale
        helpers, and ``img_kps``/``img_desc`` are the image's cached ORB features.
        Returns ``list[Match]`` (full-res px, source-excluded, IoU-NMS'd).
        """
        cv2 = self._cv2
        if img_desc is None or len(img_kps) < _MIN_VOTES:
            return []
        full_h, full_w = image_gray.shape[:2]
        total_f = max(1, int(scale_back))
        ik_xy = np.array([k.pt for k in img_kps], np.float32).reshape(-1, 2)

        tmpl_gray = feature_matcher._to_gray(template)
        active = set(active_orientations(
            getattr(params, "enable_rotation", False),
            getattr(params, "enable_flipping", False),
        ))
        if progress is not None:
            progress(0)

        results: list[Match] = []
        for orient in active:
            if cancel is not None and cancel():
                break
            tg = np.ascontiguousarray(apply_orientation(tmpl_gray, orient))
            results.extend(
                self._match_orientation(tg, orient, ik_xy, img_desc, image_gray,
                                        feature_matcher, full_h, full_w, total_f, cancel)
            )
        if progress is not None:
            progress(100)

        # Source exclusion + greedy IoU-NMS (highest score wins).
        if exclude_box is not None:
            results = [r for r in results if not self._excluded(r, exclude_box,
                       float(getattr(params, "exclude_iou", 0.30)))]
        results.sort(key=lambda r: r.score, reverse=True)
        kept: list[Match] = []
        nms_iou = float(getattr(params, "nms_iou", _NMS_IOU))
        for m in results:
            if all(self._iou((m.x, m.y, m.w, m.h), (k.x, k.y, k.w, k.h)) <= nms_iou
                   for k in kept):
                kept.append(m)
        cap = int(getattr(params, "feature_max_instances", 100))
        return kept[:cap]

    def _match_orientation(self, tmpl_gray, orient, ik_xy, img_desc, image_gray,
                           fm, full_h, full_w, total_f, cancel) -> list[Match]:
        cv2 = self._cv2
        th, tw = tmpl_gray.shape[:2]
        detector, _ = fm._make_detector("orb", nfeatures=_TEMPLATE_FEATURES)
        tk, td = detector.detectAndCompute(tmpl_gray, None)
        if td is None or tk is None or len(tk) < _MIN_VOTES:
            return []
        t_xy = np.array([k.pt for k in tk], np.float32).reshape(-1, 2)

        # words: each image keypoint -> the template feature indices it matches.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = bf.knnMatch(td, img_desc, k=min(len(img_desc), _KNN_K))
        img_word: dict[int, list[int]] = defaultdict(list)
        maxd = _GOOD_DIST["orb"]
        for ms in knn:
            for m in ms:
                if m.distance <= maxd:
                    img_word[m.trainIdx].append(m.queryIdx)
                else:
                    break
        if len(img_word) < _MIN_VOTES:
            return []
        matched = np.array(sorted(img_word.keys()), np.int64)
        mpos = ik_xy[matched]

        radius = float(min(_PAIR_RADIUS_CAP, max(8.0, _PAIR_RADIUS_FRAC * max(tw, th))))

        # image inverted index: (wi, wj, qdx, qdy) -> [(px, py)]; wi is the anchor's word.
        index: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
        for a_local, nb in enumerate(self._neighbours(mpos, radius, _PAIR_K)):
            p_idx = int(matched[a_local])
            px, py = ik_xy[p_idx]
            wps = img_word[p_idx]
            for b_local in nb:
                q_idx = int(matched[b_local])
                qx, qy = ik_xy[q_idx]
                qdx = int(round((qx - px) / _PAIR_QUANT))
                qdy = int(round((qy - py) / _PAIR_QUANT))
                for wi in wps:
                    for wj in img_word[q_idx]:
                        index[(wi, wj, qdx, qdy)].append((px, py))

        # template pairs query the index, voting the implied instance offset.
        votes: dict[tuple[int, int], int] = defaultdict(int)
        members: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        for i, nb in enumerate(self._neighbours(t_xy, radius, _PAIR_K)):
            if cancel is not None and (i & 255) == 0 and cancel():
                return []
            tix, tiy = t_xy[i]
            for j in nb:
                tjx, tjy = t_xy[j]
                qdx = int(round((tjx - tix) / _PAIR_QUANT))
                qdy = int(round((tjy - tiy) / _PAIR_QUANT))
                for (px, py) in index.get((i, j, qdx, qdy), ()):
                    ox, oy = px - tix, py - tiy
                    key = (int(round(ox / _VOTE_BIN)), int(round(oy / _VOTE_BIN)))
                    votes[key] += 1
                    members[key].append((ox, oy))

        out: list[Match] = []
        for key, v in votes.items():
            if v < _MIN_VOTES:
                continue
            mem = np.array(members[key], np.float32)
            ox, oy = float(np.median(mem[:, 0])), float(np.median(mem[:, 1]))
            x0, y0 = int(round(ox)), int(round(oy))
            score = min(1.0, v / (2.0 * _MIN_VOTES))
            if _VERIFY_ZNCC:
                if 0 <= x0 <= full_w - tw and 0 <= y0 <= full_h - th:
                    z = self._zncc(image_gray[y0:y0 + th, x0:x0 + tw], tmpl_gray)
                else:
                    z = 0.0
                if z < _NCC_ACCEPT:
                    continue
                score = min(1.0, max(0.0, z))  # clamp (exact-copy ZNCC can exceed 1.0)
            rx = max(0, min(x0 * total_f, full_w * total_f))
            ry = max(0, min(y0 * total_f, full_h * total_f))
            out.append(Match(x=rx, y=ry, w=tw * total_f, h=th * total_f,
                             score=float(score), scale=1.0, orientation=orient))
        return out

    @staticmethod
    def _excluded(m: Match, exclude_box, exclude_iou: float) -> bool:
        ex, ey, ew, eh = exclude_box
        ix0, iy0 = max(m.x, ex), max(m.y, ey)
        ix1, iy1 = min(m.x + m.w, ex + ew), min(m.y + m.h, ey + eh)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        u = m.w * m.h + ew * eh - inter
        iou = inter / u if u > 0 else 0.0
        if iou > exclude_iou:
            return True
        cx, cy = m.x + m.w / 2.0, m.y + m.h / 2.0
        return (ex <= cx < ex + ew) and (ey <= cy < ey + eh)
