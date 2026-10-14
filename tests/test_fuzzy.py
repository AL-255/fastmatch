"""Fuzzy retrieval regression tests on altered, repeated and unrelated motifs."""

from dataclasses import replace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fastmatch.engine import Matcher
from fastmatch.fuzzy_matcher import FuzzyMatcher, rootsift
from fastmatch.types import MatchParams
from scripts.fuzzy_eval import build_scene, evaluate, iou, motif


def test_fuzzy_recovers_occluded_copy_strict_matcher_rejects():
    report = evaluate()
    assert not report["features"]["found"]["50% occlusion"]
    assert all(report["fuzzy"]["found"].values())
    assert report["fuzzy"]["false_positives"] == 0


def test_rootsift_normalization_and_empty_descriptor():
    result = rootsift(np.array([[1, 4, 9], [0, 0, 0]], np.float32))
    assert np.isfinite(result).all()
    assert np.linalg.norm(result[0]) == pytest.approx(1)
    np.testing.assert_array_equal(result[1], 0)


def test_retrieval_is_not_limited_to_96_descriptor_neighbours():
    template = motif(np.random.default_rng(42), 96, 96)
    image = np.full((10 * 128, 11 * 128, 3), 100, np.uint8)
    truth = []
    for row in range(10):
        for col in range(11):
            x, y = col * 128 + 16, row * 128 + 16
            image[y:y + 96, x:x + 96] = template
            truth.append((x, y, 96, 96))
    backend = FuzzyMatcher()
    results = backend.match(template, backend._to_gray(image), MatchParams(
        method="fuzzy", feature_min_inliers=6, max_results=200, feature_max_instances=200),
        exclude_box=truth[0])
    assert len(results) == 109
    assert all(any(iou((r.x, r.y, r.w, r.h), target) > .8 for r in results) for target in truth[1:])


def test_fuzzy_floor_cap_cancel_and_image_cache():
    image, template, source, _ = build_scene()
    matcher = Matcher(device="cpu")
    matcher.set_image(image)
    params = MatchParams(method="fuzzy", feature_min_inliers=8)
    hits = matcher.match(template, params, exclude_box=source)
    assert len(hits) == 4
    assert hits == sorted(hits, key=lambda r: -r.score)
    backend = matcher._fuzzy_matcher
    cached = backend._image_cache[("sift", 0)]
    assert len(matcher.match(template, replace(params, max_results=1), exclude_box=source)) == 1
    assert backend._image_cache[("sift", 0)] is cached
    assert matcher.match(template, replace(params, threshold_floor=1), exclude_box=source) == []
    assert matcher.match(template, params, cancel=lambda: True) == []
    matcher.set_image(np.zeros_like(image))
    assert not backend._image_cache
    assert matcher.match(template, params) == []


def test_fuzzy_mirror_checkbox_and_mask():
    template = motif(np.random.default_rng(43))
    h, w = template.shape[:2]
    image = np.full((h + 40, 2 * w + 100, 3), 100, np.uint8)
    image[20:20 + h, 20:20 + w] = template
    image[20:20 + h, w + 60:w * 2 + 60] = template[:, ::-1]
    target = (w + 60, 20, w, h)
    matcher = Matcher(device="cpu")
    matcher.set_image(image)
    params = MatchParams(method="fuzzy", feature_min_inliers=8)
    assert matcher.match(template, params, exclude_box=(20, 20, w, h)) == []
    mask = np.ones((h, w), bool)
    mask[:30] = False
    hits = matcher.match(template, replace(params, enable_flipping=True),
                         exclude_box=(20, 20, w, h), mask=mask)
    assert any(r.orientation == "MY" and iou((r.x, r.y, r.w, r.h), target) > .8 for r in hits)


def test_fuzzy_examples_do_not_use_pixel_verification():
    from fastmatch.examples import match_examples

    image, template, source, truth = build_scene()
    matcher = Matcher(device="cpu")
    matcher.set_image(image)
    hits = match_examples(matcher, [source], [(970, 240, 160, 180)],
                          MatchParams(method="fuzzy", feature_min_inliers=8))
    assert all(any(iou((r.x, r.y, r.w, r.h), box) > .8 for r in hits) for box in truth.values())
    assert any(iou((r.x, r.y, r.w, r.h), source) > .8 for r in hits)


def test_fuzzy_recovers_perspective_geometry():
    template = motif(np.random.default_rng(13))
    h, w = template.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[230, 35], [420, 60], [380, 245], [255, 200]])
    transform = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(template, transform, (460, 280))
    backend = FuzzyMatcher()
    hits = backend.match(template, backend._to_gray(image),
                         MatchParams(method="fuzzy", enable_rotation=True, feature_min_inliers=8))
    assert any(iou((r.x, r.y, r.w, r.h), (230, 35, 190, 210)) > .8 for r in hits)
