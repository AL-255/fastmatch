"""Loader tests: synthetic generation, RAM load, and the memmap strip path.

Targets the DESIGN §C.4 / §F.4 contracts for ``fastmatch.loader``:

    generate_sample(path, w, h, *, tile, n_targets, seed) -> list[Match]
    load_image(path, *, max_ram_bytes=6e9) -> ImageDocument
    _load_via_memmap(path, w, h) -> np.memmap        # 1024-row strip decode

The key correctness claim is that the memmap path (forced via a tiny
``max_ram_bytes``) decodes pixel-identical output to the all-in-RAM path, and
that either way the document holds a ``(H, W, 3)`` uint8 C-contiguous array
whose declared width/height match the array.

These tests instantiate the real ``loader`` module, which is written in
parallel; if it is not yet importable they are skipped rather than collected as
hard errors, so this file stays green during the parallel build and is re-run at
integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastmatch.document import ImageDocument
from fastmatch.types import Match

# Loader is a sibling module under parallel development. Skip (don't error) the
# whole module if it cannot be imported yet, so partial-build runs stay clean.
loader = pytest.importorskip("fastmatch.loader", reason="loader not yet implemented")


# Small but seam-spanning sample. tile=100 on a 1200x900 canvas gives a real
# multi-tile grid so motifs can land on seams, yet it generates in well under a
# second on CPU.
SAMPLE_KW = dict(w=1200, h=900, tile=100, n_targets=6, seed=0)


def test_generate_sample_returns_ground_truth_and_writes_file(tmp_path) -> None:
    """generate_sample writes a PNG and returns one Match per stamped motif."""
    out = tmp_path / "sample.png"
    truth = loader.generate_sample(str(out), **SAMPLE_KW)

    assert out.exists(), "generate_sample must write the PNG to disk"
    assert out.stat().st_size > 0

    assert isinstance(truth, list)
    assert len(truth) == SAMPLE_KW["n_targets"]
    assert all(isinstance(m, Match) for m in truth)

    # Every ground-truth box must sit fully inside the (half-open) image bounds.
    for m in truth:
        assert m.w > 0 and m.h > 0
        assert 0 <= m.x and m.x + m.w <= SAMPLE_KW["w"]
        assert 0 <= m.y and m.y + m.h <= SAMPLE_KW["h"]


def test_generate_sample_is_deterministic(tmp_path) -> None:
    """Same seed -> identical ground-truth boxes (reproducible self-test)."""
    a = loader.generate_sample(str(tmp_path / "a.png"), **SAMPLE_KW)
    b = loader.generate_sample(str(tmp_path / "b.png"), **SAMPLE_KW)
    assert [(m.x, m.y, m.w, m.h) for m in a] == [(m.x, m.y, m.w, m.h) for m in b]


def test_load_image_returns_well_formed_document(tmp_path) -> None:
    """load_image yields a (H,W,3) uint8 C-contiguous doc with matching size."""
    out = tmp_path / "sample.png"
    loader.generate_sample(str(out), **SAMPLE_KW)

    doc = loader.load_image(str(out))
    assert isinstance(doc, ImageDocument)

    arr = doc.full
    assert arr.ndim == 3 and arr.shape[2] == 3
    assert arr.dtype == np.uint8
    assert arr.flags["C_CONTIGUOUS"]

    # Declared dimensions must agree with both the array and the requested size.
    assert arr.shape == (SAMPLE_KW["h"], SAMPLE_KW["w"], 3)
    assert doc.height == arr.shape[0] == SAMPLE_KW["h"]
    assert doc.width == arr.shape[1] == SAMPLE_KW["w"]


def test_load_image_accepts_path_object(tmp_path) -> None:
    """load_image accepts a Path (not only str) per its ``str | Path`` signature."""
    out = tmp_path / "sample.png"
    loader.generate_sample(out, **SAMPLE_KW)  # also exercise Path on generate_sample
    doc = loader.load_image(out)
    assert doc.full.shape == (SAMPLE_KW["h"], SAMPLE_KW["w"], 3)


def test_memmap_path_matches_full_ram_load(tmp_path) -> None:
    """Forcing the memmap path yields pixel-identical, (H,W,3) uint8 output.

    A tiny ``max_ram_bytes`` pushes ``load_image`` onto ``_load_via_memmap``
    (1024-row strip decode). The decoded pixels must be byte-for-byte equal to
    the all-in-RAM decode (np.array_equal), proving the strip seams are correct.
    """
    out = tmp_path / "sample.png"
    loader.generate_sample(str(out), **SAMPLE_KW)

    # Generous budget -> in-RAM path; near-zero budget -> memmap strip path.
    doc_ram = loader.load_image(str(out), max_ram_bytes=6_000_000_000)
    doc_mm = loader.load_image(str(out), max_ram_bytes=1)

    # Shape / dtype invariants hold on the memmap document too.
    assert doc_mm.full.shape == (SAMPLE_KW["h"], SAMPLE_KW["w"], 3)
    assert doc_mm.full.dtype == np.uint8
    assert doc_mm.full.shape == doc_ram.full.shape

    # The whole point of the strip decoder: identical pixels to the RAM path.
    # Compare via np.asarray so a memmap is materialised for the equality check.
    assert np.array_equal(np.asarray(doc_mm.full), np.asarray(doc_ram.full))


def test_load_via_memmap_returns_memmap(tmp_path) -> None:
    """_load_via_memmap returns an actual np.memmap of the right shape/dtype."""
    out = tmp_path / "sample.png"
    loader.generate_sample(str(out), **SAMPLE_KW)

    mm = loader._load_via_memmap(str(out), SAMPLE_KW["w"], SAMPLE_KW["h"])
    assert isinstance(mm, np.memmap)
    assert mm.shape == (SAMPLE_KW["h"], SAMPLE_KW["w"], 3)
    assert mm.dtype == np.uint8
