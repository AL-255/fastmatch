"""Explicit CPU thread control for PyTorch and OpenCV compute kernels."""
import os
import torch
from PySide6.QtCore import QSettings

def available_threads() -> int:
    return max(1, os.cpu_count() or 1)

def load_threads() -> int:
    try:
        return max(0, int(QSettings("FastMatch", "FastMatch").value("compute/cpu_threads", 0)))
    except (TypeError, ValueError):
        return 0

def configure_threads(count: int = 0, *, persist: bool = False) -> int:
    # 0 uses all available logical processors. Inter-op concurrency remains
    # unchanged: the app has one search worker, avoiding nested thread pools.
    count = min(available_threads(), max(1, count or available_threads()))
    torch.set_num_threads(count)
    try:
        import cv2
        cv2.setNumThreads(count)
    except ImportError:
        pass
    if persist:
        QSettings("FastMatch", "FastMatch").setValue("compute/cpu_threads", count)
    return count
