"""Cross-boundary data types for FastMatch.

These dataclasses are the contract that the engine, controller, worker, and
viewport all speak. They are deliberately **Qt-free and picklable** so they can
ride across QThread signal/slot boundaries (and a process boundary, if we ever
move matching out-of-process) without dragging GUI types along.

Coordinate convention (canonical, applied everywhere outside the viewport):
    integer **image pixels**, origin top-left, +x right / +y down (matches
    numpy ``arr[y, x]``, Pillow, and QImage). A box ``(x, y, w, h)`` is
    **half-open**: it covers ``x in [x, x+w)`` and ``y in [y, y+h)``, so
    ``full[y:y+h, x:x+w]`` yields an ``(h, w, 3)`` crop.
"""

from __future__ import annotations

from dataclasses import dataclass


# The selectable matching methods, in UI order, with one-line guidance on when each fits.
# Single source of truth shared by the engine dispatch and the params-panel dropdown.
METHODS: tuple[str, ...] = ("ncc", "ssd", "ccorr", "features")

METHOD_LABELS: dict[str, str] = {
    "ncc": "NCC — normalized cross-correlation (textured, illumination-robust)",
    "ssd": "SSD — squared difference (flat / low-texture / exact appearance)",
    "ccorr": "CCORR — cosine cross-correlation",
    "features": "Feature matching — ORB + homography (rotated / scaled / warped)",
}

# Methods whose score map is computed by the shared tiled-convolution machinery.
CONV_METHODS: frozenset[str] = frozenset({"ncc", "ssd", "ccorr"})


@dataclass(frozen=True)
class Match:
    """A single detected instance, in full-resolution image pixels.

    Attributes:
        x: Left edge (inclusive top-left), image px.
        y: Top edge (inclusive top-left), image px.
        w: Width in image px (half-open: covers columns ``x .. x+w``).
        h: Height in image px (half-open: covers rows ``y .. y+h``).
        score: NCC peak remapped to ``[0, 1]`` (raw NCC in ``[-1, 1]`` is
            clamped at 0 — negative correlation is "not a match").
        scale: Scale at which the instance was found (``1.0`` == template
            native size; ``> 1`` == larger instance, ``< 1`` == smaller).
    """

    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float


@dataclass(frozen=True)
class MatchParams:
    """Tunable search parameters.

    The engine returns every detection at or above ``threshold_floor`` so the
    UI can live-filter up to ``threshold`` with a slider without re-running the
    GPU search. ``scales`` defaults to ``(1.0,)`` here; the params panel widens
    it for CUDA devices (see the spec's DEFAULTS table) and keeps it narrow on
    CPU to avoid multi-minute freezes.
    """

    threshold: float = 0.85          # min score the UI keeps (live slider filter)
    threshold_floor: float = 0.50    # engine returns everything >= this; UI filters up
    scales: tuple[float, ...] = (1.0,)
    rotations: tuple[float, ...] | None = None  # degrees; None = off (documented unsupported)
    max_results: int = 500           # cap after NMS
    nms_iou: float = 0.30            # IoU above which overlapping detections are merged
    exclude_iou: float = 0.30        # drop hits overlapping the source box by more than this
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    compute_dtype: str = "float32"  # "float32" | "float16" (accumulators always fp32)
    channel_mode: str = "luminance"  # "luminance" | "rgb"

    # --- Matching method selection (see METHODS / METHOD_LABELS) ---------------
    # "ncc"/"ssd"/"ccorr" use the GPU tiled-convolution machinery (scales/NMS apply).
    # "features" uses an ORB-keypoint + RANSAC-homography backend that tolerates
    # rotation/scale/perspective warp; the scale/NMS/channel knobs do not apply to it.
    method: str = "ncc"

    # --- Feature-matching parameters (only used when method == "features") -------
    feature_detector: str = "orb"     # "orb" | "akaze" | "sift"
    feature_ratio: float = 0.75       # Lowe ratio test for descriptor matching
    feature_min_inliers: int = 12     # min RANSAC inliers to accept one instance
    feature_max_instances: int = 100  # cap on instances returned by the feature path
