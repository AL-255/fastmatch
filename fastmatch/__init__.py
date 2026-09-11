"""FastMatch — GPU-accelerated visual pattern matcher.

A PySide6 desktop app that loads a gigapixel image into a tiled, GPU-composited
viewport (scroll-wheel zoom, drag-pan, left-drag region select) and runs a fast
normalized-cross-correlation (NCC) template search to find every other region
that is visually similar to the boxed selection.

This package's ``__init__`` is intentionally light: it must be importable for
headless engine/loader use without dragging in Qt. GUI symbols live in
``fastmatch.app`` / ``fastmatch.viewport`` and are imported on demand.
"""

import os as _os

from PIL import Image as _PILImage

# ROCm: MIOpen's default find mode benchmarks and compiles kernels for every new
# convolution shape (7-21 s each on an RX 7900 XT). Each new selection size brings
# a fresh set of shapes (pyramid levels, box filters, tile edges), so the first
# search at a new size could take minutes and looked like it found nothing. FAST
# mode picks a solver heuristically instead. MIOpen reads this lazily when its
# handle is created, so setting it here (before any convolution) is enough.
# setdefault lets a user still override it. Ignored on CUDA/CPU.
_os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

# Allow decoding gigapixel images. loader.load_image performs an explicit pixel
# pre-flight before relying on this, so disabling Pillow's decompression-bomb
# guard here is deliberate and safe.
_PILImage.MAX_IMAGE_PIXELS = None

__version__ = "0.2.0"

__all__ = ["__version__"]
