"""Real HIP integration checks; skipped when no working AMD runtime exists."""
import numpy as np
import pytest
from fastmatch.device import gpu_backend, resolve_device
from fastmatch.engine import Matcher
from fastmatch.types import MatchParams

@pytest.mark.parametrize("method", ["ncc", "ssd", "ccorr"])
@pytest.mark.parametrize("backend", ["spatial", "fft"])
def test_rocm_finds_exact_copy(method, backend):
    if gpu_backend() != "rocm" or resolve_device("rocm").type != "cuda":
        pytest.skip("No working ROCm runtime")
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    template = image[8:24, 8:24].copy()
    image[72:88, 80:96] = template
    matcher = Matcher(device="rocm", conv_backend=backend, use_pyramid=False)
    matcher.set_image(image)
    results = matcher.match(template, MatchParams(method=method, threshold=0.99),
                            exclude_box=(8, 8, 16, 16))
    assert any(r.x == 80 and r.y == 72 and r.score > 0.99 for r in results)
