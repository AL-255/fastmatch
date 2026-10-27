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
from PIL import Image

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


@pytest.mark.parametrize("max_ram_bytes", [1, 6_000_000_000])
@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_white_artwork_keeps_alpha_geometry(tmp_path, max_ram_bytes, mode) -> None:
    """White ink encoded only in alpha must not turn into a blank white image."""
    alpha = np.array([[0, 128, 255], [255, 0, 128]], dtype=np.uint8)
    rgba = np.full((2, 3, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = alpha
    im = Image.fromarray(rgba)
    if mode == "LA":
        im = im.convert("LA")
    elif mode == "P":
        # A palette PNG can encode both partial and full transparency in tRNS.
        im = Image.fromarray(np.array([[0, 1, 2], [2, 0, 1]], np.uint8)).convert("P")
        im.putpalette([255, 255, 255] * 256)
        im.info["transparency"] = bytes([0, 128, 255])
    path = tmp_path / "white-pads.png"
    im.save(path)
    doc = loader.load_image(path, max_ram_bytes=max_ram_bytes)
    assert (doc.width, doc.height) == (3, 2)
    np.testing.assert_array_equal(doc.full, np.repeat(alpha[:, :, None], 3, axis=2))


@pytest.mark.parametrize("max_ram_bytes", [1, 6_000_000_000])
def test_dark_artwork_ignores_invisible_rgb(tmp_path, max_ram_bytes) -> None:
    """Dark lines need a white matte, regardless of hidden transparent colors."""
    rgba = np.full((3, 4, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = 0  # invisible white must not bias the matte toward black
    rgba[1, 1] = (0, 0, 0, 255)
    rgba[1, 2] = (0, 0, 0, 128)
    path = tmp_path / "dark-lines.png"
    Image.fromarray(rgba).save(path)
    doc = loader.load_image(path, max_ram_bytes=max_ram_bytes)
    expected = np.full((3, 4, 3), 255, dtype=np.uint8)
    expected[1, 1] = 0
    expected[1, 2] = 127
    np.testing.assert_array_equal(doc.full, expected)


def test_transparent_image_uses_one_matte_across_strip_boundary(tmp_path) -> None:
    rgba = np.zeros((loader._STRIP_ROWS + 1, 4, 4), dtype=np.uint8)
    rgba[:, :3, 3] = 255
    rgba[-1, :3, :3] = 255  # this strip alone would choose the opposite matte
    path = tmp_path / "seam.png"
    Image.fromarray(rgba).save(path)
    ram = loader.load_image(path)
    mapped = loader.load_image(path, max_ram_bytes=1)
    np.testing.assert_array_equal(mapped.full, ram.full)
    np.testing.assert_array_equal(mapped.full[:, 3], 255)
    np.testing.assert_array_equal(mapped.full[:, :3], rgba[:, :3, :3])


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_opaque_pixels_are_preserved(tmp_path, mode) -> None:
    rgb = np.random.default_rng(5).integers(0, 256, (12, 13, 3), dtype=np.uint8)
    path = tmp_path / "opaque.png"
    Image.fromarray(rgb).convert(mode).save(path)
    for budget in (1, 6_000_000_000):
        np.testing.assert_array_equal(loader.load_image(path, max_ram_bytes=budget).full, rgb)


@pytest.mark.parametrize("max_ram_bytes", [1, 6_000_000_000])
def test_rgb_transparent_color_key(tmp_path, max_ram_bytes) -> None:
    path = tmp_path / "color-key.png"
    rgb = np.array([[[255, 255, 255], [255, 0, 255]]], dtype=np.uint8)
    Image.fromarray(rgb).save(path, transparency=(255, 0, 255))
    doc = loader.load_image(path, max_ram_bytes=max_ram_bytes)
    np.testing.assert_array_equal(doc.full, [[[255, 255, 255], [0, 0, 0]]])
