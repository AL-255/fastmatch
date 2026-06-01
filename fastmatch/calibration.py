"""Physical-scale calibration for the viewport (pixels <-> physical units).

A :class:`Calibration` is established by picking two points on the image and
entering the physical length they span. Per the spec the entered length maps to
the **longer** of the horizontal / vertical pixel spans (``max(|dx|, |dy|)``) —
the natural choice when calibrating against a feature aligned to one axis (a
scale bar, a chip edge). The first picked point becomes the physical-grid
origin, so cursor coordinates and the selection-box area are all reported in the
same frame.

Pure math (no Qt) so it is unit-tested directly; the app passes plain
``(x, y)`` pixel tuples in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def fmt(value: float, sig: int = 4) -> str:
    """Format a physical magnitude with ``sig`` significant figures, trimmed."""
    if value == 0:
        return "0"
    return f"{value:.{sig}g}"


@dataclass(frozen=True)
class Calibration:
    """A pixel->physical mapping: ``scale`` units per pixel about ``origin``."""

    scale: float                      # physical units per pixel
    unit: str                         # e.g. "mm" (may be empty)
    origin: tuple[float, float]       # image-px point that is physical (0, 0)

    @classmethod
    def from_two_points(
        cls,
        p1: tuple[float, float],
        p2: tuple[float, float],
        length: float,
        unit: str = "",
    ) -> "Calibration":
        """Calibrate so ``length`` spans the longer pixel axis between p1 and p2.

        ``origin`` is ``p1``. Raises ``ValueError`` if the span or length is
        non-positive (a degenerate calibration).
        """
        ref_px = cls.reference_span(p1, p2)
        if ref_px <= 0 or length <= 0:
            raise ValueError("degenerate calibration: span and length must be positive")
        return cls(scale=float(length) / ref_px, unit=str(unit), origin=(float(p1[0]), float(p1[1])))

    @staticmethod
    def reference_span(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """The longer of the horizontal / vertical pixel spans between two points."""
        return max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))

    def physical_point(self, px: float, py: float) -> tuple[float, float]:
        """Pixel ``(px, py)`` -> physical coordinates relative to the origin."""
        return ((px - self.origin[0]) * self.scale, (py - self.origin[1]) * self.scale)

    def physical_length(self, px_dist: float) -> float:
        """Convert a pixel distance to physical units."""
        return px_dist * self.scale

    def physical_distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Euclidean physical distance between two image points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1]) * self.scale

    def physical_area(self, w_px: float, h_px: float) -> float:
        """Physical area of a ``w_px x h_px`` pixel rectangle (units squared)."""
        return abs(w_px) * abs(h_px) * self.scale * self.scale

    def format_point(self, px: float, py: float) -> str:
        x, y = self.physical_point(px, py)
        return f"({fmt(x)}, {fmt(y)}) {self.unit}".rstrip()

    def format_area(self, w_px: float, h_px: float) -> str:
        return f"{fmt(self.physical_area(w_px, h_px))} {self.unit}²".strip()

    def format_length(self, px_dist: float) -> str:
        return f"{fmt(self.physical_length(px_dist))} {self.unit}".rstrip()
