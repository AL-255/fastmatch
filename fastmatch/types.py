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

import numpy as np


# --- Dihedral orientations (D4) ---------------------------------------------
# The 8 symmetries of a square: 4 rotations and 4 reflections. A boxed template
# can appear in the image under any of these, and all matching methods can search
# the active subset. Two independent bits select the subset:
#   * rotation  -> the 90°-family rotations (R90, R180, R270)
#   * flipping  -> the reflections (MX, MY); both bits also add the diagonal
#                  reflections (MXR90, MYR90), which are a flip AND a 90° turn.
ORIENTATIONS: tuple[str, ...] = ("R0", "R90", "R180", "R270", "MX", "MY", "MXR90", "MYR90")

ORIENTATION_LABELS: dict[str, str] = {
    "R0": "0°",
    "R90": "rotated 90°",
    "R180": "rotated 180°",
    "R270": "rotated 270°",
    "MX": "mirrored (vertical flip)",
    "MY": "mirrored (horizontal flip)",
    "MXR90": "mirrored + 90°",
    "MYR90": "mirrored + 90° (other diagonal)",
}


def active_orientations(enable_rotation: bool, enable_flipping: bool) -> tuple[str, ...]:
    """Select the active D4 orientations from the two UI checkboxes.

    Each orientation has a rotation bit (a 90/180/270° turn) and a reflection bit
    (a mirror); it is active iff the corresponding checkbox(es) are enabled:

        (off, off) -> R0
        (rot, off) -> R0, R90, R180, R270
        (off, flip) -> R0, MX, MY
        (rot, flip) -> all 8

    Returned in the canonical :data:`ORIENTATIONS` order; always includes R0.
    """
    active = {"R0"}
    if enable_rotation:
        active |= {"R90", "R180", "R270"}
    if enable_flipping:
        active |= {"MX", "MY"}
    if enable_rotation and enable_flipping:
        active |= {"MXR90", "MYR90"}
    return tuple(o for o in ORIENTATIONS if o in active)


def apply_orientation(arr: np.ndarray, orient: str) -> np.ndarray:
    """Return ``arr`` transformed by dihedral orientation ``orient``.

    Works on ``(H, W)`` or ``(H, W, C)`` uint8/float arrays (numpy ``[row, col]``
    == ``[y, x]``). R90/R270/MXR90/MYR90 swap the height and width. The result is
    C-contiguous so it can be handed straight to torch/cv2.
    """
    if orient == "R0":
        out = arr
    elif orient == "R90":
        out = np.rot90(arr, 1)            # 90° counter-clockwise
    elif orient == "R180":
        out = np.rot90(arr, 2)
    elif orient == "R270":
        out = np.rot90(arr, 3)
    elif orient == "MX":
        out = arr[::-1, :]                # mirror over the horizontal (x) axis
    elif orient == "MY":
        out = arr[:, ::-1]                # mirror over the vertical (y) axis
    elif orient == "MXR90":
        out = np.rot90(arr[::-1, :], 1)   # flip then rotate -> a diagonal reflection
    elif orient == "MYR90":
        out = np.rot90(arr[:, ::-1], 1)   # the other diagonal reflection
    else:
        raise ValueError(f"unknown orientation {orient!r}; expected one of {ORIENTATIONS}")
    return np.ascontiguousarray(out)


# Canonical 2x2 linear parts (image coords: +x right, +y down) of each D4
# orientation — the exact forward linear map of :func:`apply_orientation` (the
# numpy transform), so that ``orientation_from_matrix`` labels a recovered
# transform with the SAME code ``apply_orientation`` would produce. (These are
# derived from the numpy ops: e.g. np.rot90's array-CCW turn is a clockwise turn
# in +y-down image coords, so R90 == [[0,1],[-1,0]] here, not its transpose.)
_ORIENT_LINEAR: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "R0": ((1, 0), (0, 1)),
    "R90": ((0, 1), (-1, 0)),
    "R180": ((-1, 0), (0, -1)),
    "R270": ((0, -1), (1, 0)),
    "MX": ((1, 0), (0, -1)),
    "MY": ((-1, 0), (0, 1)),
    "MXR90": ((0, -1), (-1, 0)),
    "MYR90": ((0, 1), (1, 0)),
}


def orientation_from_matrix(linear: np.ndarray) -> str:
    """Classify a 2x2 linear transform into its nearest dihedral orientation.

    Used by the feature path: a homography's upper-left 2x2 (rotation + scale +
    reflection, ignoring shear) is scale-normalized and matched to the closest of
    the 8 canonical D4 matrices. The reflection is captured by the sign of the
    determinant, the rotation by the entry pattern.
    """
    m = np.asarray(linear, dtype=np.float64)[:2, :2]
    det = float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
    scale = float(np.sqrt(abs(det)))
    if scale < 1e-9:
        return "R0"
    mn = m / scale
    best, best_d = "R0", 1e18
    for code, ref in _ORIENT_LINEAR.items():
        ref_arr = np.asarray(ref, dtype=np.float64)
        d = float(((mn - ref_arr) ** 2).sum())
        if d < best_d:
            best, best_d = code, d
    return best


# The selectable matching methods, in UI order, with one-line guidance on when each fits.
# Single source of truth shared by the engine dispatch and the params-panel dropdown.
METHODS: tuple[str, ...] = ("ncc", "ssd", "ccorr", "features")

METHOD_LABELS: dict[str, str] = {
    "ncc": "NCC — normalized cross-correlation (textured, illumination-robust)",
    "ssd": "SSD — squared difference (flat / low-texture / exact appearance)",
    "ccorr": "CCORR — cosine cross-correlation",
    "features": "Feature matching — ORB keypoints + appearance verify (rotated / scaled)",
}

# Methods whose score map is computed by the shared tiled-convolution machinery.
CONV_METHODS: frozenset[str] = frozenset({"ncc", "ssd", "ccorr"})

# Experimental "Shazam-style" landmark-pair-hash matcher. Intentionally NOT in
# METHODS, so it never appears in the UI dropdown — it is DISABLED by default. The
# engine still dispatches it when ``method == SHAZAM_METHOD`` is set explicitly,
# so the approach is kept, importable and unit-tested. It underperforms feature
# matching badly on this workload (see fastmatch/shazam_matcher.py); to surface it
# in the UI, add it to METHODS/METHOD_LABELS and the params panel.
SHAZAM_METHOD: str = "shazam"


# --- Channel modes (colour space the matcher scores in) ---------------------
# Every method matches on one or more channels of the selected colour space; the
# per-channel score contributions are combined with user weights that sum to 1.
#   * luminance -> a single BT.601 luma plane (the fast default; no weights).
#   * rgb       -> the three R,G,B channels, weighted by ``rgb_weights``.
#   * ycbcr     -> the three Y,Cb,Cr channels (BT.601), weighted by ``ycbcr_weights``.
# Multi-channel modes are genuinely multi-channel (each channel correlated
# separately, contributions summed) — NOT a single projected plane — so colour
# differences are discriminative. Equal weights reproduce the unweighted multi-
# channel behaviour exactly.
CHANNEL_MODES: tuple[str, ...] = ("luminance", "rgb", "ycbcr")

CHANNEL_MODE_LABELS: dict[str, str] = {
    "luminance": "Luminance (single BT.601 luma plane)",
    "rgb": "RGB (weighted R, G, B channels)",
    "ycbcr": "YCbCr (weighted Y, Cb, Cr channels)",
}

#: Per-mode channel names, in order, for the weight sliders / UI labels.
CHANNEL_NAMES: dict[str, tuple[str, ...]] = {
    "luminance": ("Y",),
    "rgb": ("R", "G", "B"),
    "ycbcr": ("Y", "Cb", "Cr"),
}


def normalize_weights(weights, n: int = 3) -> tuple[float, ...]:
    """Clamp to non-negative and rescale ``weights`` so they sum to 1.0.

    Pads/truncates to ``n`` entries. An all-zero (or empty) input falls back to
    equal weights so a mode is never silently disabled. The matcher always
    normalizes, so the UI's sum-to-1 constraint is a convenience, not a contract
    the engine depends on.
    """
    w = [max(0.0, float(x)) for x in (weights or ())][:n]
    while len(w) < n:
        w.append(0.0)
    s = sum(w)
    if s <= 1e-9:
        return tuple(1.0 / n for _ in range(n))
    return tuple(x / s for x in w)


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
        orientation: Dihedral orientation the instance was found under (one of
            :data:`ORIENTATIONS`; ``"R0"`` == upright, unmirrored). Defaults to
            ``"R0"`` so existing callers/constructors are unaffected.
    """

    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float
    orientation: str = "R0"


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
    channel_mode: str = "luminance"  # "luminance" | "rgb" | "ycbcr" (see CHANNEL_MODES)

    # --- Per-channel weights for the multi-channel modes (see CHANNEL_MODES) ----
    # Used only when channel_mode is "rgb" / "ycbcr". The matcher normalizes them
    # to sum to 1 (normalize_weights), so equal weights == the unweighted multi-
    # channel behaviour. They weight each channel's contribution to the combined
    # score (all methods, conv + features), letting the user emphasise, e.g.,
    # chroma (Cb/Cr) to match by colour or a single RGB channel.
    rgb_weights: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    ycbcr_weights: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3)

    # --- Matching method selection (see METHODS / METHOD_LABELS) ---------------
    # "ncc"/"ssd"/"ccorr" use the GPU tiled-convolution machinery (scales/NMS apply).
    # "features" uses an ORB-keypoint + RANSAC-homography backend that tolerates
    # rotation/scale/perspective warp; the scale/NMS/channel knobs do not apply to it.
    method: str = "ncc"

    # --- Orientation search (dihedral D4; see active_orientations) --------------
    # Search the boxed template under additional orientations. All methods honor
    # these. enable_rotation adds 90/180/270° turns; enable_flipping adds mirrors;
    # both together also add the diagonal reflections. Each kept Match records the
    # orientation it was found under (Match.orientation).
    enable_rotation: bool = False
    enable_flipping: bool = False

    # --- Feature-matching parameters (only used when method == "features") -------
    feature_detector: str = "orb"     # "orb" | "akaze" | "sift"
    feature_ratio: float = 0.75       # Lowe ratio test for descriptor matching
    feature_min_inliers: int = 12     # min RANSAC inliers to accept one instance
    feature_max_instances: int = 100  # cap on instances returned by the feature path
