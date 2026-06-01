"""Unit tests for the physical-scale calibration math (Qt-free)."""

from __future__ import annotations

import math

import pytest

from fastmatch.calibration import Calibration, fmt


def test_reference_span_uses_longer_axis():
    # dx=100, dy=40 -> longer axis is 100.
    assert Calibration.reference_span((10, 20), (110, 60)) == 100
    # dy dominates here.
    assert Calibration.reference_span((0, 0), (30, 80)) == 80


def test_from_two_points_scale_and_origin():
    cal = Calibration.from_two_points((10, 20), (110, 70), length=5.0, unit="mm")
    # longer span = max(100, 50) = 100 px -> 5 mm / 100 px = 0.05 mm/px.
    assert cal.scale == pytest.approx(0.05)
    assert cal.unit == "mm"
    assert cal.origin == (10.0, 20.0)  # first point is the origin


def test_from_two_points_rejects_degenerate():
    with pytest.raises(ValueError):
        Calibration.from_two_points((5, 5), (5, 5), length=1.0)      # zero span
    with pytest.raises(ValueError):
        Calibration.from_two_points((0, 0), (10, 0), length=0.0)     # zero length


def test_physical_point_relative_to_origin():
    cal = Calibration(scale=0.05, unit="mm", origin=(10.0, 20.0))
    assert cal.physical_point(10, 20) == (0.0, 0.0)
    assert cal.physical_point(110, 20) == pytest.approx((5.0, 0.0))
    assert cal.physical_point(10, 120) == pytest.approx((0.0, 5.0))


def test_physical_distance_is_euclidean():
    cal = Calibration(scale=0.05, unit="mm", origin=(0.0, 0.0))
    # 3-4-5 triangle: 50 px hypotenuse -> 2.5 mm.
    assert cal.physical_distance((0, 0), (30, 40)) == pytest.approx(0.05 * 50)
    assert cal.physical_length(50) == pytest.approx(2.5)


def test_physical_area_is_squared_scale():
    cal = Calibration(scale=0.05, unit="mm", origin=(0.0, 0.0))
    # 100 x 200 px -> (5 mm)(10 mm) = 50 mm^2.
    assert cal.physical_area(100, 200) == pytest.approx(50.0)


def test_formatters():
    cal = Calibration(scale=0.05, unit="mm", origin=(0.0, 0.0))
    assert cal.format_point(100, 0) == "(5, 0) mm"
    assert cal.format_area(100, 200) == "50 mm²"
    assert cal.format_length(50) == "2.5 mm"
    # Empty unit: no trailing space.
    bare = Calibration(scale=1.0, unit="", origin=(0.0, 0.0))
    assert bare.format_length(3) == "3"


def test_fmt_significant_figures():
    assert fmt(0) == "0"
    assert fmt(1234.5678) == "1235"      # 4 sig figs
    assert fmt(0.0012345) == "0.001234" or fmt(0.0012345) == "0.001235"
