"""FFT padding lengths: the ROCm bucket set stays smooth, bounded and sparse."""
from fastmatch.engine import _next_fft_bucket, _next_smooth


def _smooth(n: int) -> bool:
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def test_bucket_is_smooth_and_bounded():
    for n in range(1, 20000):
        b = _next_fft_bucket(n)
        assert b >= n
        assert _smooth(b)
        assert b <= max(_next_smooth(n), n * 1.25)


def test_bucket_collapses_nearby_lengths():
    # Tile inputs for different selection sizes (core 1024 + template - 1) must
    # share a handful of lengths so rocFFT's per-length kernels get reused.
    lengths = {_next_fft_bucket(1024 + t - 1) for t in range(16, 400)}
    assert lengths == {1280, 1536}
