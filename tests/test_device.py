"""Accelerator preference and launch-probe fallback, independent of hardware."""

import numpy as np
import pytest
import torch

from fastmatch import device
from fastmatch.engine import Matcher
from fastmatch.types import MatchParams


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(device, "canary_kernel_ok", lambda: True)
    monkeypatch.setattr(device, "mps_canary_kernel_ok", lambda: True)
    assert device.resolve_device("auto").type == "cuda"
    assert device.resolve_device("mps").type == "mps"
    assert device.resolve_device("cpu").type == "cpu"
    monkeypatch.setattr(device, "canary_kernel_ok", lambda: False)
    assert device.resolve_device("auto").type == "mps"
    assert device.resolve_device("cuda").type == "cpu"
    monkeypatch.setattr(device, "mps_canary_kernel_ok", lambda: False)
    assert device.resolve_device("auto").type == "cpu"
    assert device.resolve_device("mps").type == "cpu"


def test_rocm_alias_requires_a_rocm_wheel(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "canary_kernel_ok", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    assert device.resolve_device("rocm").type == "cpu"
    monkeypatch.setattr(torch.version, "hip", "7.2")
    assert device.resolve_device("rocm").type == "cuda"
    assert device.gpu_backend() == "rocm"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        device.resolve_device("unknown")


def test_mps_uses_spatial_kernels_and_bounded_batches():
    matcher = Matcher(device="cpu", conv_backend="fft")
    matcher._device = torch.device("mps")
    assert matcher._cross_backend(96, 96) == "spatial"
    assert matcher._core_tile_size(96, 96) == 512
    assert matcher._refine_chunk(100, 100, 3, 96, 96) == 16


@pytest.mark.skipif(device.resolve_device("mps").type != "mps", reason="requires a working Apple MPS device")
def test_mps_matching_matches_cpu_results():
    rng = np.random.default_rng(9)
    template = rng.integers(40, 240, (70, 66, 3), np.uint8)
    image = np.full((150, 220, 3), 100, np.uint8)
    image[15:85, 20:86] = template
    image[75:145, 150:216] = template
    matcher = Matcher(device="mps")
    assert matcher.effective_device.type == "mps"
    matcher.set_image(image)
    for method in ("ncc", "ssd", "ccorr"):
        hits = matcher.match(template, MatchParams(method=method, threshold_floor=.9),
                             exclude_box=(20, 15, 66, 70))
        assert any(r.x == 150 and r.y == 75 and r.score > .95 for r in hits)


"""Backend routing is tested without requiring AMD/NVIDIA hardware."""
import pytest
import torch
from fastmatch import device
from fastmatch.__main__ import _build_parser

@pytest.mark.parametrize("hip", [None, "7.2.1"])
def test_auto_gpu(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "canary_kernel_ok", lambda: True)
    assert device.resolve_device("auto").type == "cuda"
    assert device.resolve_device("rocm").type == ("cuda" if hip else "cpu")
    assert device.resolve_device("cpu").type == "cpu"

@pytest.mark.parametrize("pref", ["auto", "cuda", "rocm"])
def test_failed_canary_falls_back(monkeypatch, pref):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(device, "canary_kernel_ok", lambda: False)
    assert device.resolve_device(pref).type == "cpu"

def test_rocm_banner_and_cli(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: "AMD Radeon RX 7900 XT")
    assert device.device_banner_text(torch.device("cuda")) == "Engine: ROCm / HIP (AMD Radeon RX 7900 XT)"
    assert _build_parser().parse_args(["--device", "rocm"]).device == "rocm"

def test_invalid_preference():
    with pytest.raises(ValueError):
        device.resolve_device("typo")

def test_detection_exception_falls_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    def fail():
        raise RuntimeError("driver init failed")
    monkeypatch.setattr(torch.cuda, "is_available", fail)
    assert device.resolve_device("auto").type == "cpu"

@pytest.mark.parametrize("generation", ["RDNA1", "RDNA2", "RDNA3", "RDNA4"])
def test_no_generation_whitelist(monkeypatch, generation):
    monkeypatch.setattr(torch.version, "hip", "compatible-build")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: generation)
    monkeypatch.setattr(device, "canary_kernel_ok", lambda: True)
    assert device.resolve_device("rocm").type == "cuda"
