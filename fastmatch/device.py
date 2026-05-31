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


def resolve_device(pref: str = "auto") -> torch.device:
    """Resolve a usable torch device.

    Args:
        pref: ``"auto"`` (CUDA if usable else CPU), ``"cuda"`` (try CUDA, still
            fall back to CPU if the canary fails), or ``"cpu"`` (force CPU).
    """
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and canary_kernel_ok():
        return torch.device("cuda")
    return torch.device("cpu")


def device_banner_text(dev: torch.device) -> str:
    """A short human-readable status banner for the resolved device."""
    if dev.type == "cuda":
        try:
            name = torch.cuda.get_device_name(dev)
        except Exception:
            name = "CUDA"
        return f"Engine: CUDA ({name})"
    return "Engine: CPU (slow) — install the cu128 PyTorch build for GPU acceleration"
