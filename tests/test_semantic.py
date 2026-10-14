"""Semantic retrieval orchestration, plus optional real-model offline smoke."""

from dataclasses import replace
import os

import numpy as np
import pytest
import torch

from fastmatch.engine import Matcher
from fastmatch.memory import params_from_dict, params_to_dict
from fastmatch.semantic_matcher import SemanticMatcher, semantic_available, window_positions
from fastmatch.types import MatchParams


def colour_encoder(crops):
    # Deterministic, deliberately non-pixel-exact stand-in to test the search
    # machinery. Real pretrained inference is checked separately below.
    return np.array([crop.reshape(-1, 3).mean(axis=0) for crop in crops], np.float32)


def test_semantic_retrieves_related_appearance_and_excludes_source():
    image = np.zeros((96, 192, 3), np.uint8)
    image[:, :, 2] = 200
    template = np.zeros((32, 32, 3), np.uint8)
    template[:, :, 0] = 240
    image[16:48, 16:48] = template
    image[48:80, 128:160] = (180, 30, 0)
    backend = SemanticMatcher(encoder=colour_encoder)
    params = MatchParams(method="semantic", threshold_floor=.9, max_results=10)
    progress = []
    hits = backend.match(template, image, params, exclude_box=(16, 16, 32, 32), progress=progress.append)
    assert hits
    assert (hits[0].x, hits[0].y, hits[0].w, hits[0].h) == (128, 48, 32, 32)
    assert all(not backend._excluded(r, (16, 16, 32, 32), params.exclude_iou) for r in hits)
    assert all(r.score >= .9 for r in hits)
    assert progress[-1] == 100
    assert backend.match(template, image, params, cancel=lambda: True) == []
    assert len(backend.match(template, image, replace(params, max_results=1), exclude_box=(16, 16, 32, 32))) == 1


def test_semantic_multiscale_edges_masks_and_cache_invalidation():
    image = np.zeros((96, 192, 3), np.uint8)
    image[:, :, 2] = 200
    image[32:96, 128:192] = (240, 0, 0)
    template = np.full((32, 32, 3), (240, 0, 0), np.uint8)
    backend = SemanticMatcher(encoder=colour_encoder)
    params = MatchParams(method="semantic", scales=(2,), threshold_floor=.99)
    hits = backend.match(template, image, params)
    assert (hits[0].x, hits[0].y, hits[0].w, hits[0].h) == (128, 32, 64, 64)
    assert hits[0].scale == 2
    assert backend.match(template, image, params, mask=np.zeros((32, 32), bool)) == []
    assert backend._cache
    backend.invalidate_image()
    assert not backend._cache
    assert window_positions(100, 32, 16)[-1] == 68
    assert window_positions(20, 32, 16) == []
    with pytest.raises(ValueError, match="finite and positive"):
        backend.match(template, image, replace(params, scales=(0,)))


def test_engine_delegates_semantic_and_memory_preserves_modes():
    image = np.full((64, 64, 3), (240, 0, 0), np.uint8)
    matcher = Matcher(device="cpu")
    matcher.set_image(image)
    matcher._semantic_matcher = SemanticMatcher(encoder=colour_encoder)
    params = MatchParams(method="semantic", scales=(.75, 1, 1.5))
    assert matcher.match(image[:32, :32], params)
    assert params_from_dict(params_to_dict(params)) == params
    assert params_from_dict(params_to_dict(replace(params, method="fuzzy"))).method == "fuzzy"
    matcher.set_image(np.zeros_like(image))
    assert not matcher._semantic_matcher._cache


def test_semantic_examples_preserve_variation_and_compare_negatives():
    from fastmatch.examples import match_examples

    image = np.full((96, 256, 3), (0, 0, 200), np.uint8)
    boxes = [(16, 16, 32, 32), (80, 16, 32, 32), (16, 64, 32, 32),
             (80, 64, 32, 32), (176, 16, 32, 32), (176, 64, 32, 32)]
    colours = [(240, 0, 0), (0, 240, 0), (180, 30, 0),
               (30, 180, 0), (240, 180, 0), (200, 150, 0)]
    for (x, y, w, h), colour in zip(boxes, colours):
        image[y:y + h, x:x + w] = colour
    matcher = Matcher(device="cpu")
    matcher.set_image(image)
    matcher._semantic_matcher = SemanticMatcher(encoder=colour_encoder)
    hits = match_examples(matcher, boxes[:2], [boxes[4]], MatchParams(
        method="semantic", threshold_floor=.5))
    assert {(r.x, r.y, r.w, r.h) for r in hits} == set(boxes[:4])


def test_missing_model_is_offline_and_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTMATCH_SEMANTIC_MODEL", str(tmp_path))
    ready, hint = semantic_available()
    assert not ready
    assert "--download-semantic-model" in hint
    with pytest.raises(RuntimeError, match="--download-semantic-model"):
        SemanticMatcher()._load()


def test_grayscale_semantic_search_converts_only_windows(monkeypatch):
    repeat = np.repeat

    def bounded_repeat(array, *args, **kwargs):
        assert array.shape[0] <= 32 and array.shape[1] <= 32
        return repeat(array, *args, **kwargs)

    monkeypatch.setattr("fastmatch.semantic_matcher.np.repeat", bounded_repeat)
    backend = SemanticMatcher(encoder=colour_encoder)
    image = np.full((128, 256), 100, np.uint8)
    hits = backend.match(image[:32, :32], image,
                         MatchParams(method="semantic", max_results=3))
    assert len(hits) == 3


def test_frozen_models_prefer_complete_user_cache_then_bundle(tmp_path, monkeypatch):
    import sys
    from fastmatch.semantic_matcher import model_directory

    bundled = tmp_path / "bundle" / "models" / "dinov2-small"
    bundled.mkdir(parents=True)
    (bundled / "model.safetensors").write_bytes(b"bundled")
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.delenv("FASTMATCH_SEMANTIC_MODEL", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setattr("fastmatch.semantic_matcher.user_model_directory", lambda: cache)
    assert model_directory() == bundled
    (cache / "model.safetensors").write_bytes(b"partial")
    assert model_directory() == bundled
    (cache / "config.json").write_text('{}')
    assert model_directory() == cache
    monkeypatch.setenv("FASTMATCH_SEMANTIC_MODEL", str(tmp_path / "override"))
    assert model_directory() == tmp_path / "override"


@pytest.mark.skipif(os.environ.get("FASTMATCH_TEST_SEMANTIC_MODEL") != "1",
                    reason="opt-in smoke with locally downloaded pretrained weights")
def test_pretrained_dinov2_offline_embeddings():
    backend = SemanticMatcher("cpu")
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (224, 224, 3), np.uint8)
    altered = np.clip(image.astype(np.float32) * .8 + 20, 0, 255).astype(np.uint8)
    features = backend._encode([image, altered])
    assert features.shape == (2, 768)
    assert torch.isfinite(features).all()
    assert float(features[0] @ features[1]) > .9
