"""Saved-match "Memory" data-layer tests (DESIGN §M).

A :class:`~fastmatch.memory.MemoryEntry` is a snapshot of one completed search —
the boxed source selection plus the matches it found (and the params used) — with
derived stats for display; a :class:`~fastmatch.memory.MemoryStore` is the source
image plus a list of those entries, round-tripping to a JSON file.

The data model and JSON layer live in ``fastmatch.memory`` and are **Qt-free and
frozen**; this module pins that contract headlessly. It imports neither Qt nor the
torch/cv2 engine, so it is engine-free and runs instantly (no ``@pytest.mark.cpu``
needed). Coordinates follow the project convention — image px, half-open
``(x, y, w, h)`` — and match boxes are in the SOURCE image's pixel space.

Three groups:

  * **MemoryEntry stats**: ``count`` / ``score_range`` / ``mean_score`` /
    ``orientation_counts`` (canonical order, nonzero only) / ``orientation_summary``
    / ``summary``.
  * **JSON round-trip**: ``store_to_dict`` -> ``store_from_dict`` and
    ``save_store`` -> ``load_store`` preserve entries, matches (incl. orientation),
    the source image and its size EXACTLY.
  * **Robustness**: a too-new ``version`` and non-JSON text both raise
    ``ValueError``; ``store_from_dict`` tolerates entries missing optional keys.
"""

from __future__ import annotations

import json

import pytest

from fastmatch.memory import (
    MEMORY_JSON_VERSION,
    MemoryEntry,
    MemoryStore,
    entry_from_dict,
    entry_to_dict,
    load_store,
    params_from_dict,
    params_to_dict,
    save_store,
    store_from_dict,
    store_to_dict,
)
from fastmatch.types import ORIENTATIONS, Match, MatchParams


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _entry_with_varied_matches(*, label: str = "") -> MemoryEntry:
    """A MemoryEntry with several matches of varied scores and orientations.

    Orientations are deliberately out of canonical order in the match list and
    include duplicates so the stats helpers' canonical-order / counting behaviour
    is exercised. Boxes are in the source image's pixel space.

    ``label`` defaults to empty (so the stats tests see ``summary()``'s auto-label
    path). The JSON round-trip tests pass an explicit label, because the frozen
    ``entry_to_dict`` persists ``summary()`` when the label is empty — so only an
    explicitly-labelled entry is dataclass-equal to itself after a round-trip.
    """
    matches = [
        Match(x=100, y=200, w=32, h=24, score=0.91, scale=1.0, orientation="R0"),
        Match(x=400, y=120, w=24, h=32, score=0.72, scale=1.0, orientation="R90"),
        Match(x=640, y=300, w=32, h=24, score=0.88, scale=1.1, orientation="R0"),
        Match(x=50, y=500, w=32, h=24, score=0.65, scale=0.9, orientation="MY"),
        Match(x=720, y=480, w=24, h=32, score=0.80, scale=1.0, orientation="R90"),
    ]
    return MemoryEntry(
        selection=(100, 200, 32, 24),
        params=MatchParams(
            method="ncc",
            threshold=0.6,
            enable_rotation=True,
            enable_flipping=True,
        ),
        matches=matches,
        label=label,
    )


# =========================================================================== #
# MemoryEntry stats
# =========================================================================== #
def test_entry_count() -> None:
    """``count()`` is the number of recorded matches; ``0`` for an empty entry."""
    assert _entry_with_varied_matches().count() == 5
    empty = MemoryEntry(selection=(0, 0, 8, 8), params=MatchParams())
    assert empty.count() == 0


def test_entry_occurrences_includes_reference() -> None:
    """``occurrences()`` == matches + 1 (the reference selection counts too)."""
    assert _entry_with_varied_matches().occurrences() == 6  # 5 matches + reference
    empty = MemoryEntry(selection=(0, 0, 8, 8), params=MatchParams())
    assert empty.occurrences() == 1  # just the reference, even with no matches


def test_entry_score_range() -> None:
    """``score_range()`` returns ``(min, max)`` over the matches' scores."""
    e = _entry_with_varied_matches()
    lo, hi = e.score_range()
    assert lo == pytest.approx(0.65)
    assert hi == pytest.approx(0.91)
    # Empty -> (0.0, 0.0) per the frozen contract.
    empty = MemoryEntry(selection=(0, 0, 8, 8), params=MatchParams())
    assert empty.score_range() == (0.0, 0.0)


def test_entry_mean_score() -> None:
    """``mean_score()`` is the arithmetic mean of the match scores (``0.0`` if empty)."""
    e = _entry_with_varied_matches()
    expected = (0.91 + 0.72 + 0.88 + 0.65 + 0.80) / 5
    assert e.mean_score() == pytest.approx(expected)
    empty = MemoryEntry(selection=(0, 0, 8, 8), params=MatchParams())
    assert empty.mean_score() == 0.0


def test_entry_orientation_counts_canonical_order_nonzero_only() -> None:
    """``orientation_counts()`` counts per orientation, canonical order, nonzero only."""
    e = _entry_with_varied_matches()
    counts = e.orientation_counts()
    # The right tallies: 2x R0, 2x R90, 1x MY.
    assert counts == {"R0": 2, "R90": 2, "MY": 1}
    # Only orientations that actually occur appear (no zero entries for the
    # other D4 codes), and they are in canonical ORIENTATIONS order.
    assert all(v > 0 for v in counts.values())
    present = list(counts.keys())
    order = [ORIENTATIONS.index(o) for o in present]
    assert order == sorted(order), f"orientation_counts not in canonical order: {present}"
    # Total over orientations equals the match count.
    assert sum(counts.values()) == e.count()


def test_entry_orientation_summary_string() -> None:
    """``orientation_summary()`` is a compact ``"O:n"`` breakdown in canonical order."""
    e = _entry_with_varied_matches()
    # R0 before R90 before MY (canonical order), space-joined "code:count".
    assert e.orientation_summary() == "R0:2 R90:2 MY:1"
    # Empty entry -> empty string (no orientations).
    empty = MemoryEntry(selection=(0, 0, 8, 8), params=MatchParams())
    assert empty.orientation_summary() == ""


def test_entry_summary_mentions_method_occurrences_and_selection() -> None:
    """``summary()`` is a one-liner with the method, occurrence count and selection."""
    e = _entry_with_varied_matches()
    s = e.summary()
    assert isinstance(s, str)
    assert "ncc" in s                       # the method
    assert "occurrences" in s               # counts occurrences, not "matches"
    assert str(e.occurrences()) in s        # the occurrence count (6 = 5 + reference)
    # The selection's size and origin appear (sel is 32x24 @ (100, 200)).
    assert "32" in s and "24" in s
    assert "100" in s and "200" in s


def test_entry_summary_prefers_explicit_label() -> None:
    """A non-empty ``label`` is used verbatim as the summary (auto-label fallback)."""
    e = _entry_with_varied_matches()
    e.label = "my favourite motif"
    assert e.summary() == "my favourite motif"


# =========================================================================== #
# JSON round-trip
# =========================================================================== #
def _store_with_two_entries() -> MemoryStore:
    """A MemoryStore with a header and two entries, one using orientation flags.

    Both entries carry an explicit ``label`` so the store is dataclass-equal to
    itself after a JSON round-trip (the frozen ``entry_to_dict`` otherwise bakes
    ``summary()`` into the persisted label of a label-less entry).
    """
    e1 = _entry_with_varied_matches(label="oriented motif")  # rotation+flipping on
    e2 = MemoryEntry(
        selection=(10, 20, 64, 48),
        params=MatchParams(
            method="ssd",
            threshold=0.5,
            enable_rotation=False,
            enable_flipping=False,
        ),
        matches=[
            Match(x=500, y=500, w=64, h=48, score=0.99, scale=1.0, orientation="R0"),
            Match(x=900, y=720, w=64, h=48, score=0.95, scale=1.25, orientation="R0"),
        ],
        label="exact copies",
    )
    return MemoryStore(
        source_image="/data/scan.png",
        image_size=(4096, 3072),
        entries=[e1, e2],
    )


def test_store_dict_round_trip_equals_original() -> None:
    """``store_to_dict`` -> ``store_from_dict`` reproduces the store EXACTLY.

    Entries, their matches (incl. each Match's orientation), the source image
    path and the image size must all survive the round-trip unchanged. Because
    :class:`~fastmatch.types.Match` and :class:`MemoryStore` are dataclasses,
    ``==`` compares field-by-field, so an exact dataclass-equality assertion
    pins orientation and every numeric field at once.
    """
    store = _store_with_two_entries()
    back = store_from_dict(store_to_dict(store))
    assert back == store

    # Header preserved explicitly.
    assert back.source_image == "/data/scan.png"
    assert back.image_size == (4096, 3072)
    assert len(back.entries) == len(store.entries)

    # Spot-check the oriented matches survive incl. orientation/scale.
    e0 = back.entries[0]
    assert [m.orientation for m in e0.matches] == ["R0", "R90", "R0", "MY", "R90"]
    assert e0.enable_rotation is True and e0.enable_flipping is True
    assert back.entries[1].label == "exact copies"


def test_entry_dict_round_trip_preserves_matches_and_flags() -> None:
    """A single (explicitly labelled) entry round-trips through the dict helpers."""
    e = _entry_with_varied_matches(label="oriented motif")
    back = entry_from_dict(entry_to_dict(e))
    assert back == e
    # Match equality (a frozen dataclass) covers x/y/w/h/score/scale/orientation.
    assert back.matches == e.matches
    assert back.enable_rotation is True and back.enable_flipping is True
    assert back.selection == (100, 200, 32, 24)


def test_entry_round_trips_faithfully_when_unlabelled() -> None:
    """An unlabelled entry round-trips EXACTLY (the label is persisted as-is).

    Serialization must not invent a label: an empty label stays empty so a saved
    entry reloads equal to the original. The descriptive one-liner is generated
    on demand by ``summary()`` for display, independent of what is stored.
    """
    e = _entry_with_varied_matches()  # empty label
    assert e.label == ""
    d = entry_to_dict(e)
    assert d["label"] == ""  # not auto-filled — round-trip stays faithful
    back = entry_from_dict(d)
    assert back == e  # full identity, including the empty label
    # summary() still produces a descriptive string even with no stored label.
    assert back.summary() == e.summary()
    assert e.method in back.summary() and str(back.occurrences()) in back.summary()
    assert back.enable_rotation == e.enable_rotation
    assert back.enable_flipping == e.enable_flipping


def test_save_then_load_round_trips_exactly(tmp_path) -> None:
    """``save_store`` then ``load_store`` returns an equal store; matches exact.

    The on-disk JSON is the authoritative interchange format, so a written-then-
    read store must equal the original — including the source image, the image
    size, and every Match (with its orientation).
    """
    store = _store_with_two_entries()
    path = tmp_path / "memory.json"
    save_store(store, path)

    # A real JSON file was written.
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["version"] == MEMORY_JSON_VERSION
    assert on_disk["source_image"] == "/data/scan.png"
    assert on_disk["image_size"] == [4096, 3072]
    assert len(on_disk["entries"]) == 2

    loaded = load_store(path)
    assert loaded == store
    # Matches round-trip EXACTLY (Match equality includes orientation).
    for got, want in zip(loaded.entries, store.entries):
        assert got.matches == want.matches
        for m in got.matches:
            assert m.orientation in ORIENTATIONS


def test_save_load_accepts_str_path(tmp_path) -> None:
    """``save_store`` / ``load_store`` accept a plain ``str`` path, not just Path."""
    store = _store_with_two_entries()
    path = str(tmp_path / "memory_str.json")
    save_store(store, path)
    assert load_store(path) == store


# =========================================================================== #
# Full-params round-trip
# =========================================================================== #
def _entry_with_full_params() -> MemoryEntry:
    """An entry whose params set MANY non-default fields across the whole struct.

    Exercises every awkward corner of params (de)serialization at once: a
    non-default ``channel_mode``, a multi-element ``scales`` tuple, tweaked
    NMS/exclude IoU, both orientation bits, a non-default feature detector and
    min-inliers, and ``rotations`` left at its ``None`` default. The matches are
    orientation-tagged so Match equality (which includes ``orientation``) is
    pinned too.
    """
    params = MatchParams(
        threshold=0.7,
        threshold_floor=0.4,
        scales=(0.9, 1.0, 1.1),
        rotations=None,
        max_results=250,
        nms_iou=0.45,
        exclude_iou=0.2,
        device="cpu",
        compute_dtype="float16",
        channel_mode="rgb",
        method="features",
        enable_rotation=True,
        enable_flipping=True,
        feature_detector="akaze",
        feature_ratio=0.8,
        feature_min_inliers=20,
        feature_max_instances=42,
    )
    return MemoryEntry(
        selection=(12, 34, 56, 78),
        params=params,
        matches=[
            Match(x=12, y=34, w=56, h=78, score=0.95, scale=1.0, orientation="R0"),
            Match(x=300, y=400, w=78, h=56, score=0.81, scale=1.1, orientation="MX"),
        ],
        label="full params entry",
    )


def test_params_dict_round_trip_preserves_all_fields() -> None:
    """``params_to_dict`` -> ``params_from_dict`` reproduces a MatchParams EXACTLY.

    Tuples (``scales``) and the tuple-or-``None`` ``rotations`` survive the JSON
    list/null normalization; every scalar field is preserved (frozen-dataclass
    equality compares all fields at once).
    """
    p = _entry_with_full_params().params
    back = params_from_dict(params_to_dict(p))
    assert back == p
    # Spot-check the tuple normalization explicitly.
    assert isinstance(back.scales, tuple) and back.scales == (0.9, 1.0, 1.1)
    assert back.rotations is None
    assert back.channel_mode == "rgb"


def test_full_params_survive_dict_and_file_round_trip(tmp_path) -> None:
    """The FULL params (and matches) round-trip through both dict and file paths.

    A non-default ``channel_mode="rgb"``, multi-element ``scales``, tweaked
    ``nms_iou``, ``feature_detector="akaze"``, ``feature_min_inliers`` and the
    orientation bits must all reappear unchanged after ``store_to_dict`` ->
    ``store_from_dict`` AND ``save_store`` -> ``load_store``. The reloaded
    entry's ``params`` must equal the original, and matches (incl. orientation).
    """
    entry = _entry_with_full_params()
    store = MemoryStore(
        source_image="/data/full.png",
        image_size=(2048, 1536),
        entries=[entry],
    )

    # 1) dict round-trip is dataclass-equal.
    via_dict = store_from_dict(store_to_dict(store))
    assert via_dict == store

    # 2) file round-trip is dataclass-equal.
    path = tmp_path / "full_params.json"
    save_store(store, path)
    loaded = load_store(path)
    assert loaded == store

    # 3) The reloaded entry's params equal the original params, field-for-field.
    got = loaded.entries[0]
    assert got.params == entry.params
    # Pin the requested non-default settings explicitly.
    assert got.params.channel_mode == "rgb"
    assert got.params.scales == (0.9, 1.0, 1.1)
    assert got.params.nms_iou == pytest.approx(0.45)
    assert got.params.feature_detector == "akaze"
    assert got.params.feature_min_inliers == 20
    assert got.params.enable_rotation is True and got.params.enable_flipping is True
    # The settings delegates reflect params after the round-trip.
    assert got.method == "features"
    assert got.channel_mode == "rgb"
    assert got.threshold == pytest.approx(0.7)
    # Matches (incl. orientation) survive exactly.
    assert got.matches == entry.matches
    assert [m.orientation for m in got.matches] == ["R0", "MX"]

    # The serialized entry stores a "params" block (not the old flat keys).
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    entry_dict = on_disk["entries"][0]
    assert "params" in entry_dict
    assert entry_dict["params"]["channel_mode"] == "rgb"
    assert entry_dict["params"]["scales"] == [0.9, 1.0, 1.1]
    assert "method" not in entry_dict  # the flat key is gone


# =========================================================================== #
# Backward compatibility (old flat-key entries, no "params")
# =========================================================================== #
def test_entry_from_dict_loads_old_flat_keys_without_params() -> None:
    """An OLD-style entry dict (flat method/threshold/enable_*, no "params") loads.

    Files written before settings were captured as a full ``MatchParams`` stored
    the method / threshold / orientation bits as flat keys. ``entry_from_dict``
    must fold those into the entry's ``params`` so the delegating properties
    reflect them, with the frozen defaults for every other field.
    """
    old = {
        "label": "legacy entry",
        "selection": [100, 200, 32, 24],
        "method": "ssd",
        "threshold": 0.6,
        "enable_rotation": True,
        "enable_flipping": False,
        "matches": [
            {"x": 100, "y": 200, "w": 32, "h": 24, "score": 0.9, "scale": 1.0, "orientation": "R0"},
        ],
    }
    e = entry_from_dict(old)
    # The flat fields are reflected through the params delegates.
    assert e.method == "ssd"
    assert e.threshold == pytest.approx(0.6)
    assert e.enable_rotation is True
    assert e.enable_flipping is False
    # Everything else falls back to the frozen MatchParams defaults.
    defaults = MatchParams()
    assert e.params.threshold_floor == defaults.threshold_floor
    assert e.params.scales == defaults.scales
    assert e.params.nms_iou == defaults.nms_iou
    assert e.params.channel_mode == defaults.channel_mode
    assert e.channel_mode == "luminance"
    assert e.params.feature_detector == defaults.feature_detector
    # The non-params content still loads as before.
    assert e.selection == (100, 200, 32, 24)
    assert e.label == "legacy entry"
    assert e.matches == [Match(x=100, y=200, w=32, h=24, score=0.9, scale=1.0, orientation="R0")]


def test_store_from_dict_loads_old_flat_entries(tmp_path) -> None:
    """A whole OLD-style store (entries with flat keys, no "params") loads via file."""
    payload = {
        "version": MEMORY_JSON_VERSION,
        "source_image": "/data/legacy.png",
        "image_size": [800, 600],
        "entries": [
            {
                "label": "legacy",
                "selection": [1, 2, 3, 4],
                "method": "ccorr",
                "threshold": 0.55,
                "enable_rotation": False,
                "enable_flipping": True,
                "matches": [],
            }
        ],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = load_store(path)
    assert store.source_image == "/data/legacy.png"
    assert len(store.entries) == 1
    e = store.entries[0]
    assert e.method == "ccorr"
    assert e.threshold == pytest.approx(0.55)
    assert e.enable_rotation is False
    assert e.enable_flipping is True
    assert e.selection == (1, 2, 3, 4)


# =========================================================================== #
# Robustness
# =========================================================================== #
def test_load_store_rejects_newer_version(tmp_path) -> None:
    """A memory file with a newer major ``version`` raises ``ValueError``."""
    path = tmp_path / "future.json"
    payload = {
        "version": MEMORY_JSON_VERSION + 1,
        "source_image": "/data/scan.png",
        "image_size": [10, 10],
        "entries": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_store(path)
    # store_from_dict refuses the too-new dict directly as well.
    with pytest.raises(ValueError):
        store_from_dict(payload)


def test_store_from_dict_tolerates_missing_optional_keys() -> None:
    """Entries missing optional keys fall back to documented defaults.

    ``orientation`` defaults to ``"R0"``, ``enable_rotation`` / ``enable_flipping``
    to ``False``, and ``label`` to ``""``; a match missing ``score`` / ``scale``
    falls back to ``0.0`` / ``1.0`` per the frozen ``_match_from_dict``.
    """
    d = {
        "version": MEMORY_JSON_VERSION,
        "source_image": "/data/scan.png",
        "image_size": [800, 600],
        "entries": [
            {
                # No label / enable_* flags; one match with no orientation/score/scale.
                "selection": [5, 6, 7, 8],
                "method": "ncc",
                "threshold": 0.85,
                "matches": [{"x": 10, "y": 20, "w": 7, "h": 8}],
            }
        ],
    }
    store = store_from_dict(d)
    assert len(store.entries) == 1
    e = store.entries[0]
    assert e.label == ""
    assert e.enable_rotation is False
    assert e.enable_flipping is False
    assert e.selection == (5, 6, 7, 8)
    m = e.matches[0]
    assert m.orientation == "R0"
    assert m.score == 0.0
    assert m.scale == 1.0


def test_entry_from_dict_tolerates_empty_dict() -> None:
    """``entry_from_dict({})`` yields a valid, defaulted entry (no KeyError)."""
    e = entry_from_dict({})
    assert e.selection == (0, 0, 0, 0)
    assert e.method == "ncc"
    assert e.matches == []
    assert e.enable_rotation is False and e.enable_flipping is False
    assert e.label == ""


def test_load_store_rejects_non_json_text(tmp_path) -> None:
    """``load_store`` on non-JSON text raises ``ValueError`` (not a raw decode error)."""
    path = tmp_path / "garbage.json"
    path.write_text("this is not json {[", encoding="utf-8")
    with pytest.raises(ValueError):
        load_store(path)


def test_store_from_dict_rejects_non_memory_dict() -> None:
    """A dict that is not a FastMatch memory file (no ``entries``) raises ``ValueError``."""
    with pytest.raises(ValueError):
        store_from_dict({"source_image": "x", "image_size": [1, 1]})
    with pytest.raises(ValueError):
        store_from_dict("not a dict")  # type: ignore[arg-type]
