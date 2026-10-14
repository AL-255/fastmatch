"""Device resolution and a launch-time CUDA canary.

The hard reality on this machine: ``torch.cuda.is_available()`` can return True
while the installed wheel ships no kernel for the GPU's compute capability (the
RTX 5060 is Blackwell / sm_120 and needs a cu128+ build). Such a mismatch only
explodes when the *first real kernel* launches. So we gate CUDA on both
``is_available()`` **and** a tiny canary matmul, and fall back to CPU on any
failure — the app must always run, just slower on CPU.
"""

from __future__ import annotations

import torch


def gpu_backend() -> str:
    """ROCm exposes its devices through torch.cuda, too."""
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def canary_kernel_ok() -> bool:
    """Return True iff a trivial CUDA kernel actually executes.

    Catches the sm_120-wheel/driver mismatch that ``is_available()`` misses
    (e.g. ``CUDA error: no kernel image is available for execution``).
    """
    try:
        a = torch.ones(8, 8, device="cuda")
        # Force a real kernel launch + a device->host sync so a lazy failure surfaces here.
        _ = (a @ a).sum().item()
        return True
    except Exception:
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
    if (pref in ("auto", "cuda", "rocm")
            and (pref != "rocm" or gpu_backend() == "rocm")
            and torch.cuda.is_available() and canary_kernel_ok()):
        return torch.device("cuda")
    if pref in ("auto", "mps") and torch.backends.mps.is_available() and mps_canary_kernel_ok():
        return torch.device("mps")
    return torch.device("cpu")


def device_banner_text(dev: torch.device) -> str:
    """A short human-readable status banner for the resolved device."""
    if dev.type == "cuda":
        try:
            name = torch.cuda.get_device_name(dev)
        except Exception:
            name = "CUDA"
        return f"Engine: {'ROCm' if gpu_backend() == 'rocm' else 'CUDA'} ({name})"
    if dev.type == "mps":
        return "Engine: MPS (Apple GPU)"
    return "Engine: CPU (slow) — use a CUDA, ROCm, or Apple Silicon build for GPU acceleration"
