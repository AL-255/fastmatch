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
    save_store,
    store_from_dict,
    store_to_dict,
)
from fastmatch.types import ORIENTATIONS, Match


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
        method="ncc",
        threshold=0.6,
        matches=matches,
        enable_rotation=True,
        enable_flipping=True,
        label=label,
    )


# =========================================================================== #
# MemoryEntry stats
# =========================================================================== #
def test_entry_count() -> None:
    """``count()`` is the number of recorded matches; ``0`` for an empty entry."""
    assert _entry_with_varied_matches().count() == 5
    empty = MemoryEntry(selection=(0, 0, 8, 8), method="ncc", threshold=0.85)
    assert empty.count() == 0


def test_entry_score_range() -> None:
    """``score_range()`` returns ``(min, max)`` over the matches' scores."""
    e = _entry_with_varied_matches()
    lo, hi = e.score_range()
    assert lo == pytest.approx(0.65)
    assert hi == pytest.approx(0.91)
    # Empty -> (0.0, 0.0) per the frozen contract.
    empty = MemoryEntry(selection=(0, 0, 8, 8), method="ncc", threshold=0.85)
    assert empty.score_range() == (0.0, 0.0)


def test_entry_mean_score() -> None:
    """``mean_score()`` is the arithmetic mean of the match scores (``0.0`` if empty)."""
    e = _entry_with_varied_matches()
    expected = (0.91 + 0.72 + 0.88 + 0.65 + 0.80) / 5
    assert e.mean_score() == pytest.approx(expected)
    empty = MemoryEntry(selection=(0, 0, 8, 8), method="ncc", threshold=0.85)
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
    empty = MemoryEntry(selection=(0, 0, 8, 8), method="ncc", threshold=0.85)
    assert empty.orientation_summary() == ""


def test_entry_summary_mentions_method_count_and_selection() -> None:
    """``summary()`` is a one-liner containing the method, match count and selection."""
    e = _entry_with_varied_matches()
    s = e.summary()
    assert isinstance(s, str)
    assert "ncc" in s                      # the method
    assert str(e.count()) in s             # the match count (5)
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
        method="ssd",
        threshold=0.5,
        matches=[
            Match(x=500, y=500, w=64, h=48, score=0.99, scale=1.0, orientation="R0"),
            Match(x=900, y=720, w=64, h=48, score=0.95, scale=1.25, orientation="R0"),
        ],
        enable_rotation=False,
        enable_flipping=False,
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
    assert e.method in back.summary() and str(back.count()) in back.summary()
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
