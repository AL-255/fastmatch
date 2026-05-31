"""The shared, read-only image source-of-truth.

``ImageDocument.full`` is the single full-resolution pixel buffer. The engine
reads it for matching (full res, possibly a ``np.memmap`` for gigapixel files);
the viewport derives its display pyramid from it. **All consumers treat it as
read-only** — nobody mutates ``full`` in place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageDocument:
    """A loaded image plus its provenance.

    Attributes:
        full: ``(H, W, 3)`` uint8 RGB, **C-contiguous**. May be a ``np.memmap``
            for images too large to hold comfortably in RAM. Read-only.
        path: Source path on disk (or a synthetic-source label).
        height: ``full.shape[0]``.
        width: ``full.shape[1]``.
    """

    full: np.ndarray
    path: str
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.full.ndim != 3 or self.full.shape[2] != 3:
            raise ValueError(f"ImageDocument.full must be (H, W, 3), got {self.full.shape}")
        if self.full.dtype != np.uint8:
            raise ValueError(f"ImageDocument.full must be uint8, got {self.full.dtype}")
        if self.height != self.full.shape[0] or self.width != self.full.shape[1]:
            raise ValueError(
                f"ImageDocument size mismatch: declared ({self.height}, {self.width}) "
                f"vs array {self.full.shape[:2]}"
            )
