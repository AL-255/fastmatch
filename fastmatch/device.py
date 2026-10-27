"""Detect CUDA / ROCm and validate real GPU kernels before enabling acceleration."""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

def gpu_backend() -> str:
    """ROCm exposes its devices through torch.cuda, too."""
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def canary_kernel_ok() -> bool:
    """Check matrix, convolution and FFT support on the installed GPU runtime."""
    try:
        a = torch.ones(8, 8, device="cuda")
        # Force a real kernel launch + a device->host sync so a lazy failure surfaces here.
        if (a @ a).sum().item() != 512:
            return False
        conv = F.conv2d(a[None, None], torch.ones(1, 1, 3, 3, device="cuda"))
        if conv.sum().item() != 324:
            return False
        restored = torch.fft.irfft2(torch.fft.rfft2(a), s=a.shape)
        return bool(torch.allclose(restored, a))
    except Exception as exc:
        logging.getLogger(__name__).warning("GPU probe failed: %s", exc)
        return False


def mps_canary_kernel_ok() -> bool:
    """Gate Apple acceleration on real matmul and convolution execution."""
    try:
        a = torch.ones(8, 8, device="mps")
        _ = (a @ a).sum().item()
        image = a.reshape(1, 1, 8, 8)
        kernel = torch.ones(1, 1, 3, 3, device="mps")
        _ = torch.nn.functional.conv2d(image, kernel).sum().item()
        return True
    except Exception:
        return False


def resolve_device(pref: str = "auto") -> torch.device:
    """Resolve a usable torch device.

    Args:
        pref: ``auto`` tries CUDA/ROCm, then Apple MPS, then CPU. Explicit
            ``cuda``, ``rocm`` or ``mps`` falls back to CPU if unavailable;
            ``cpu`` always forces CPU.
    """
    if pref not in ("auto", "cpu", "cuda", "rocm", "mps"):
        raise ValueError(f"Unknown device preference: {pref}")
    if pref == "cpu":
        return torch.device("cpu")
    try:
        if (pref in ("auto", "cuda", "rocm")
                and (pref != "rocm" or gpu_backend() == "rocm")
                and torch.cuda.is_available() and canary_kernel_ok()):
            return torch.device("cuda")
    except Exception as exc:
        logging.getLogger(__name__).warning("GPU detection failed: %s", exc)
    try:
        if pref in ("auto", "mps") and torch.backends.mps.is_available() and mps_canary_kernel_ok():
            return torch.device("mps")
    except Exception as exc:
        logging.getLogger(__name__).warning("MPS detection failed: %s", exc)
    return torch.device("cpu")


def device_banner_text(dev: torch.device) -> str:
    """A short human-readable status banner for the resolved device."""
    if dev.type == "cuda":
        try:
            name = torch.cuda.get_device_name(dev)
        except Exception:
            name = "CUDA"
        return f"Engine: {'ROCm / HIP' if gpu_backend() == 'rocm' else 'CUDA'} ({name})"
    if dev.type == "mps":
        return "Engine: MPS (Apple GPU)"
    return "Engine: CPU (slow) — use a CUDA, ROCm, or Apple Silicon build for GPU acceleration"
