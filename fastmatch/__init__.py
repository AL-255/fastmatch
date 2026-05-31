"""FastMatch — GPU-accelerated visual pattern matcher.

A PySide6 desktop app that loads a gigapixel image into a tiled, GPU-composited
viewport (scroll-wheel zoom, drag-pan, left-drag region select) and runs a fast
normalized-cross-correlation (NCC) template search to find every other region
that is visually similar to the boxed selection.

This package's ``__init__`` is intentionally light: it must be importable for
headless engine/loader use without dragging in Qt. GUI symbols live in
``fastmatch.app`` / ``fastmatch.viewport`` and are imported on demand.
"""

from PIL import Image as _PILImage

# Allow decoding gigapixel images. loader.load_image performs an explicit pixel
# pre-flight before relying on this, so disabling Pillow's decompression-bomb
# guard here is deliberate and safe.
_PILImage.MAX_IMAGE_PIXELS = None

__version__ = "0.1.0"

__all__ = ["__version__"]
