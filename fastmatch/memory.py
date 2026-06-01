"""Saved-match "Memory": data model + JSON persistence (Qt-free, testable).

A :class:`MemoryEntry` is a snapshot of one completed search — the boxed source
selection plus the matches it found (and the params used) — with derived stats
for display. A :class:`MemoryStore` is the source image plus a list of entries;
it round-trips to a JSON file via :func:`save_store` / :func:`load_store`.

This module deliberately has no Qt dependency: the GUI panel
(``fastmatch.memory_panel``) renders a store, while save/load and stats live here
so they can be unit-tested headlessly. Coordinates follow the project convention
(image px, half-open ``(x, y, w, h)``; see :mod:`fastmatch.types`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .types import ORIENTATIONS, Match, MatchParams

#: Bumped if the on-disk schema changes incompatibly. Readers tolerate older
#: minor shapes (missing optional keys) but refuse a newer major version.
MEMORY_JSON_VERSION = 1


@dataclass
class MemoryEntry:
    """One saved search: the source selection + the matches it produced.

    The complete search settings are captured as the frozen
    :class:`~fastmatch.types.MatchParams` used to run the search, so a saved
    entry records *everything* (method, threshold(s), scales, orientations,
    NMS/exclude IoU, channel mode, dtype, feature knobs) and not just a handful
    of flat fields. The most-used settings are re-exposed as read-only
    properties (:attr:`method`, :attr:`threshold`, :attr:`enable_rotation`,
    :attr:`enable_flipping`, :attr:`channel_mode`) that delegate to
    :attr:`params`, so existing callers keep working unchanged.

    Attributes:
        selection: The boxed template region ``(x, y, w, h)`` in image px.
        params: The full :class:`~fastmatch.types.MatchParams` in effect when the
            entry was captured (the matches stored are the ones visible at its
            ``threshold``).
        matches: The recorded matches (each a :class:`~fastmatch.types.Match`).
        label: Optional user/auto label; :meth:`summary` falls back to a
            descriptive string when empty.
    """

    selection: tuple[int, int, int, int]
    params: MatchParams
    matches: list[Match] = field(default_factory=list)
    label: str = ""

    # -- settings delegates (read-only views onto params) --------------------
    @property
    def method(self) -> str:
        """Matching method used ("ncc" | "ssd" | "ccorr" | "features")."""
        return self.params.method

    @property
    def threshold(self) -> float:
        """Score threshold in effect when the entry was captured."""
        return self.params.threshold

    @property
    def enable_rotation(self) -> bool:
        """Whether rotation orientations were searched."""
        return self.params.enable_rotation

    @property
    def enable_flipping(self) -> bool:
        """Whether flip orientations were searched."""
        return self.params.enable_flipping

    @property
    def channel_mode(self) -> str:
        """Channel mode used ("luminance" | "rgb")."""
        return self.params.channel_mode

    # -- derived stats -------------------------------------------------------
    def count(self) -> int:
        """Number of recorded matches."""
        return len(self.matches)

    def score_range(self) -> tuple[float, float]:
        """``(min, max)`` recorded score, or ``(0.0, 0.0)`` if empty."""
        if not self.matches:
            return (0.0, 0.0)
        scores = [m.score for m in self.matches]
        return (min(scores), max(scores))

    def mean_score(self) -> float:
        """Mean recorded score (``0.0`` if empty)."""
        if not self.matches:
            return 0.0
        return sum(m.score for m in self.matches) / len(self.matches)

    def orientation_counts(self) -> dict[str, int]:
        """Count of matches per orientation, in canonical order (nonzero only)."""
        counts: dict[str, int] = {}
        for m in self.matches:
            counts[m.orientation] = counts.get(m.orientation, 0) + 1
        return {o: counts[o] for o in ORIENTATIONS if o in counts}

    def orientation_summary(self) -> str:
        """Compact per-orientation breakdown, e.g. ``"R0:5 R90:3 MX:2"``."""
        return " ".join(f"{o}:{n}" for o, n in self.orientation_counts().items())

    def summary(self) -> str:
        """One-line human description used as the auto label."""
        if self.label:
            return self.label
        x, y, w, h = self.selection
        lo, hi = self.score_range()
        return (
            f"{self.method}/{self.channel_mode} · {self.count()} matches · "
            f"sel {w}×{h}@({x},{y}) · score {lo:.2f}–{hi:.2f}"
        )


@dataclass
class MemoryStore:
    """A source image plus the list of saved entries recorded against it."""

    source_image: str = ""
    image_size: tuple[int, int] = (0, 0)  # (width, height) in px
    entries: list[MemoryEntry] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# (De)serialization
# --------------------------------------------------------------------------- #
def _match_to_dict(m: Match) -> dict:
    return {
        "x": int(m.x),
        "y": int(m.y),
        "w": int(m.w),
        "h": int(m.h),
        "score": float(m.score),
        "scale": float(m.scale),
        "orientation": str(m.orientation),
    }


def _match_from_dict(d: dict) -> Match:
    return Match(
        x=int(d["x"]),
        y=int(d["y"]),
        w=int(d["w"]),
        h=int(d["h"]),
        score=float(d.get("score", 0.0)),
        scale=float(d.get("scale", 1.0)),
        orientation=str(d.get("orientation", "R0")),
    )


def params_to_dict(p: MatchParams) -> dict:
    """Serialize a :class:`~fastmatch.types.MatchParams` to a JSON-ready dict.

    Tuples are written as JSON lists; ``rotations`` (a tuple or ``None``) maps to
    a list or ``null``. The result feeds straight back into :func:`params_from_dict`.
    """
    return {
        "threshold": float(p.threshold),
        "threshold_floor": float(p.threshold_floor),
        "scales": [float(s) for s in p.scales],
        "rotations": (None if p.rotations is None else [float(r) for r in p.rotations]),
        "max_results": int(p.max_results),
        "nms_iou": float(p.nms_iou),
        "exclude_iou": float(p.exclude_iou),
        "device": str(p.device),
        "compute_dtype": str(p.compute_dtype),
        "channel_mode": str(p.channel_mode),
        "method": str(p.method),
        "enable_rotation": bool(p.enable_rotation),
        "enable_flipping": bool(p.enable_flipping),
        "feature_detector": str(p.feature_detector),
        "feature_ratio": float(p.feature_ratio),
        "feature_min_inliers": int(p.feature_min_inliers),
        "feature_max_instances": int(p.feature_max_instances),
    }


def params_from_dict(d: dict) -> MatchParams:
    """Rebuild a :class:`~fastmatch.types.MatchParams` from a dict.

    Missing keys fall back to the frozen :class:`MatchParams` defaults; ``scales``
    becomes a tuple and ``rotations`` a tuple-or-``None``.
    """
    defaults = MatchParams()
    rotations = d.get("rotations", defaults.rotations)
    return MatchParams(
        threshold=float(d.get("threshold", defaults.threshold)),
        threshold_floor=float(d.get("threshold_floor", defaults.threshold_floor)),
        scales=tuple(float(s) for s in d.get("scales", defaults.scales)),
        rotations=(None if rotations is None else tuple(float(r) for r in rotations)),
        max_results=int(d.get("max_results", defaults.max_results)),
        nms_iou=float(d.get("nms_iou", defaults.nms_iou)),
        exclude_iou=float(d.get("exclude_iou", defaults.exclude_iou)),
        device=str(d.get("device", defaults.device)),
        compute_dtype=str(d.get("compute_dtype", defaults.compute_dtype)),
        channel_mode=str(d.get("channel_mode", defaults.channel_mode)),
        method=str(d.get("method", defaults.method)),
        enable_rotation=bool(d.get("enable_rotation", defaults.enable_rotation)),
        enable_flipping=bool(d.get("enable_flipping", defaults.enable_flipping)),
        feature_detector=str(d.get("feature_detector", defaults.feature_detector)),
        feature_ratio=float(d.get("feature_ratio", defaults.feature_ratio)),
        feature_min_inliers=int(d.get("feature_min_inliers", defaults.feature_min_inliers)),
        feature_max_instances=int(
            d.get("feature_max_instances", defaults.feature_max_instances)
        ),
    )


def entry_to_dict(e: MemoryEntry) -> dict:
    """Serialize a :class:`MemoryEntry` to a JSON-ready dict.

    The full search settings are stored under ``"params"`` (see
    :func:`params_to_dict`); the selection, label and matches are stored as before.
    """
    return {
        # Persist the label exactly as set (empty stays empty) so the entry
        # round-trips to an equal object; the UI falls back to summary() for
        # display when the label is blank.
        "label": e.label,
        "selection": [int(v) for v in e.selection],
        "params": params_to_dict(e.params),
        "matches": [_match_to_dict(m) for m in e.matches],
    }


def entry_from_dict(d: dict) -> MemoryEntry:
    """Rebuild a :class:`MemoryEntry` from a dict (tolerant of missing keys).

    Current files carry a ``"params"`` block (full :class:`MatchParams`). Older
    files instead carry the flat ``method`` / ``threshold`` / ``enable_rotation``
    / ``enable_flipping`` keys; those are folded into a ``MatchParams`` (with the
    frozen defaults for every other field) so they still load.
    """
    sel = d.get("selection", [0, 0, 0, 0])
    if "params" in d:
        params = params_from_dict(d["params"])
    else:
        # Backward compatibility: build params from the old flat keys.
        defaults = MatchParams()
        params = MatchParams(
            method=str(d.get("method", defaults.method)),
            threshold=float(d.get("threshold", defaults.threshold)),
            enable_rotation=bool(d.get("enable_rotation", defaults.enable_rotation)),
            enable_flipping=bool(d.get("enable_flipping", defaults.enable_flipping)),
        )
    return MemoryEntry(
        selection=(int(sel[0]), int(sel[1]), int(sel[2]), int(sel[3])),
        params=params,
        matches=[_match_from_dict(m) for m in d.get("matches", [])],
        label=str(d.get("label", "")),
    )


def store_to_dict(store: MemoryStore) -> dict:
    """Serialize a :class:`MemoryStore` to a JSON-ready dict."""
    return {
        "version": MEMORY_JSON_VERSION,
        "source_image": store.source_image,
        "image_size": [int(store.image_size[0]), int(store.image_size[1])],
        "entries": [entry_to_dict(e) for e in store.entries],
    }


def store_from_dict(d: dict) -> MemoryStore:
    """Rebuild a :class:`MemoryStore` from a dict.

    Raises:
        ValueError: if the document is not a FastMatch memory file or its major
            version is newer than this build understands.
    """
    if not isinstance(d, dict) or "entries" not in d:
        raise ValueError("not a FastMatch memory file (missing 'entries')")
    version = int(d.get("version", MEMORY_JSON_VERSION))
    if version > MEMORY_JSON_VERSION:
        raise ValueError(
            f"memory file version {version} is newer than supported "
            f"({MEMORY_JSON_VERSION}); please update FastMatch"
        )
    size = d.get("image_size", [0, 0])
    return MemoryStore(
        source_image=str(d.get("source_image", "")),
        image_size=(int(size[0]), int(size[1])),
        entries=[entry_from_dict(e) for e in d.get("entries", [])],
    )


def save_store(store: MemoryStore, path: str | Path) -> None:
    """Write ``store`` to ``path`` as pretty-printed JSON (UTF-8)."""
    text = json.dumps(store_to_dict(store), indent=2)
    Path(path).write_text(text, encoding="utf-8")


def load_store(path: str | Path) -> MemoryStore:
    """Read a :class:`MemoryStore` from a JSON file at ``path``.

    Raises:
        ValueError: on malformed JSON or an unrecognized/too-new schema.
    """
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return store_from_dict(d)
