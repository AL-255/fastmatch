"""MainWindow integration tests (offscreen, Qt-only).

Focus: the **&Engine** menu that switches the compute backend at runtime. The
behaviours pinned here have crash/lie potential and have been hand-verified:

  * the menu is an exclusive radio of Auto / CUDA / CPU, the entry matching the
    current device preference starts checked, and CUDA is disabled when no
    working CUDA device exists (so the user can never pick an absent backend);
  * switching the engine rebuilds the worker-thread controller on the new device
    (never two live engines), refreshes the device banner, re-gates the GPU-only
    multi-scale control, and keeps the radio honest — without leaking a thread;
  * re-selecting the live engine is a no-op.

These run under ``QT_QPA_PLATFORM=offscreen`` with a raster viewport (no GL
surface). They never run a real match, so they are CUDA-agnostic: the CUDA arm
is only asserted when this machine actually has a working GPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="app tests need PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from fastmatch.app import MainWindow  # noqa: E402
from fastmatch.device import resolve_device  # noqa: E402
from fastmatch.document import ImageDocument  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    doc = ImageDocument(np.zeros((400, 500, 3), np.uint8), "mem", 400, 500)
    w = MainWindow(doc, device="cpu")
    w._viewport.setViewport(QWidget())  # raster viewport: avoid GL on offscreen
    yield w
    w.close()


CUDA_AVAILABLE = resolve_device("cuda").type == "cuda"


def _checked_keys(w: MainWindow) -> list[str]:
    return [k for k, a in w._engine_actions.items() if a.isChecked()]


def test_engine_menu_structure(window) -> None:
    """Three exclusive radios; CPU checked at start; CUDA gated on availability."""
    assert window._engine_menu.title() == "&Engine"
    assert set(window._engine_actions) == {"auto", "cuda", "cpu"}
    assert window._engine_group.isExclusive()
    assert _checked_keys(window) == ["cpu"]  # constructed with device="cpu"
    assert window._engine_actions["cuda"].isEnabled() == CUDA_AVAILABLE
    assert window._engine_actions["cpu"].isEnabled()
    assert window._engine_actions["auto"].isEnabled()


def test_select_same_engine_is_noop(window) -> None:
    """Re-selecting the live engine neither rebuilds the controller nor churns UI."""
    ctrl = window._controller
    window._on_select_engine("cpu")
    assert window._controller is ctrl
    assert window._device_pref == "cpu"


def test_switch_engine_rebuilds_controller_and_updates_ui(window) -> None:
    """Switching CPU->Auto rebuilds the controller, refreshes banner, no leak."""
    ctrl_before = window._controller
    window._on_select_engine("auto")

    assert window._device_pref == "auto"
    assert window._controller is not ctrl_before  # fresh worker-thread controller
    assert _checked_keys(window) == ["auto"]  # radio reflects the live engine
    assert window._banner.text()  # banner refreshed for the resolved device
    assert window._params.device == "auto"  # cached params carry the preference
    assert not window._orphaned_controllers  # old controller joined, none parked

    resolved = window._resolved_device.type
    assert resolved == ("cuda" if CUDA_AVAILABLE else "cpu")
    # Multi-scale is GPU-only: enabled exactly when the resolved device is CUDA.
    assert window._params_panel._multiscale.isEnabled() == (resolved == "cuda")


def test_switch_to_cpu_locks_multiscale(window) -> None:
    """Landing on CPU forces single-scale and disables the multi-scale control."""
    window._on_select_engine("auto")
    window._on_select_engine("cpu")
    assert window._device_pref == "cpu"
    assert window._resolved_device.type == "cpu"
    assert not window._params_panel._multiscale.isEnabled()
    assert not window._params_panel._multiscale.isChecked()
    assert not window._orphaned_controllers
