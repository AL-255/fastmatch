"""ROCm-only engine tuning: FFT correlation + larger tiles, same results."""
import numpy as np
import pytest

from fastmatch.device import gpu_backend, resolve_device
from fastmatch.engine import Matcher
from fastmatch.types import MatchParams

ROCM = gpu_backend() == "rocm" and resolve_device("cuda").type == "cuda"


@pytest.mark.skipif(not ROCM, reason="needs a working ROCm runtime")
def test_rocm_uses_fft_and_big_tiles_and_still_finds_copies():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (3000, 3000, 3), dtype=np.uint8)
    tmpl = img[100:122, 200:228].copy()
    for x, y in ((900, 400), (2500, 2600), (1700, 1300)):
        img[y:y + 22, x:x + 28] = tmpl
    m = Matcher(device="cuda")
    m.set_image(img)
    assert m._cross_backend(22, 28) == "fft"
    assert m._core_tile_size(22, 28) > 1024
    res = m.match(tmpl, MatchParams(threshold_floor=0.9), exclude_box=(200, 100, 28, 22))
    assert sorted((r.x, r.y) for r in res if r.score > 0.99) == [(900, 400), (1700, 1300), (2500, 2600)]


def test_cpu_keeps_the_spatial_path():
    m = Matcher(device="cpu")
    m.set_image(np.zeros((64, 64), np.uint8))
    assert m._cross_backend(22, 28) == "spatial"
