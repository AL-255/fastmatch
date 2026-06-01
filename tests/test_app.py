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

from fastmatch import theme  # noqa: E402
from fastmatch.app import MainWindow  # noqa: E402
from fastmatch.device import resolve_device  # noqa: E402
from fastmatch.document import ImageDocument  # noqa: E402
from fastmatch.memory_panel import _ensure_json_suffix  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_theme(qapp, monkeypatch):
    """Isolate theme persistence (never touch the user's real QSettings) and
    restore the app to the default palette after each test so global Qt state
    never leaks between tests."""
    cell = {"key": theme.DEFAULT_THEME}
    monkeypatch.setattr(
        theme, "save_theme", lambda k: cell.__setitem__("key", theme.normalize_theme(k))
    )
    monkeypatch.setattr(theme, "load_theme", lambda: cell["key"])
    yield cell
    theme.apply_theme(qapp, "system", persist=False)  # reset shared palette/style


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


# ----------------------------------------------------------- Empty-canvas startup
@pytest.fixture
def empty_window(qapp):
    """A window launched with no image (the no-argument startup path)."""
    w = MainWindow(None, device="cpu")
    w._viewport.setViewport(QWidget())
    yield w
    w.close()


def test_launch_with_no_image_starts_empty(empty_window) -> None:
    """No document -> no controller, empty canvas, and an idle, honest UI."""
    w = empty_window
    assert w._doc is None
    assert w._controller is None  # engine/worker deferred until an image opens
    assert w._viewport.image_size() == (0, 0)
    assert not w._act_close_image.isEnabled()  # nothing to close
    assert not w._params_panel._run_button.isEnabled()  # nothing to run
    assert w._count_label.text() == "No image"


def test_open_image_from_empty_builds_controller(empty_window) -> None:
    """Opening the first image builds + wires the controller and enables Close."""
    w = empty_window
    doc = ImageDocument(np.zeros((300, 400, 3), np.uint8), "/tmp/x.png", 300, 400)
    w._swap_document(doc)
    assert w._doc is doc
    assert w._controller is not None
    assert w._act_close_image.isEnabled()


def test_engine_switch_with_no_image_is_safe(empty_window) -> None:
    """Switching the engine on an empty canvas must not crash or build a worker."""
    w = empty_window
    w._on_select_engine("auto")
    assert w._controller is None  # still nothing to search
    assert w._banner.text()  # banner still refreshed for the new device
    assert w._device_pref == "auto"


def test_close_empty_window_does_not_crash(qapp) -> None:
    """closeEvent on an empty canvas must not dereference a None controller."""
    w = MainWindow(None, device="cpu")
    w._viewport.setViewport(QWidget())
    w.close()  # would raise AttributeError if closeEvent assumed a controller


# --------------------------------------------------------- Save Memory .json suffix
@pytest.mark.parametrize(
    "given, expected",
    [
        ("patterns", "patterns.json"),                 # bare name -> append
        ("patterns.json", "patterns.json"),            # already correct -> unchanged
        ("PATTERNS.JSON", "PATTERNS.JSON"),            # case-insensitive match -> kept
        ("/tmp/sub.dir/notes", "/tmp/sub.dir/notes.json"),  # dotted dir, no file ext
        ("backup.txt", "backup.txt.json"),             # foreign ext -> force .json
        ("a.json.json", "a.json.json"),                # trailing .json honoured
    ],
)
def test_ensure_json_suffix(given, expected) -> None:
    """Save Memory normalizes any typed name to a .json file (dialogs don't always)."""
    assert _ensure_json_suffix(given) == expected


# --------------------------------------------------------------- Theme menu
from PySide6.QtGui import QPalette  # noqa: E402


def _window_rgb(qapp) -> tuple[int, int, int]:
    return qapp.palette().color(QPalette.ColorRole.Window).getRgb()[:3]


def _canvas_rgb(w: MainWindow) -> tuple[int, int, int]:
    return w._viewport.backgroundBrush().color().getRgb()[:3]


def test_theme_menu_structure(window) -> None:
    """Three exclusive radios (System/Light/Dark); the active theme is checked."""
    assert window._theme_menu.title() == "&Theme"
    assert list(window._theme_actions) == list(theme.THEME_KEYS)
    assert window._theme_group.isExclusive()
    checked = [k for k, a in window._theme_actions.items() if a.isChecked()]
    assert checked == [window._theme]  # exactly the active theme


def test_switch_to_dark_themes_chrome_and_canvas(window, qapp) -> None:
    """Dark theme darkens both the Qt palette and the image canvas."""
    window._on_select_theme("dark")
    assert window._theme == "dark"
    assert [k for k, a in window._theme_actions.items() if a.isChecked()] == ["dark"]
    assert qapp.style().objectName().lower() == "fusion"
    assert _window_rgb(qapp) == (53, 53, 53)
    assert _canvas_rgb(window) == (30, 30, 30)


def test_switch_to_light_themes_chrome_and_canvas(window, qapp) -> None:
    """Light theme yields a light palette and a light canvas."""
    window._on_select_theme("light")
    assert window._theme == "light"
    assert sum(_window_rgb(qapp)) > 600  # clearly a light window colour
    assert _canvas_rgb(window) == (225, 225, 225)


def test_select_same_theme_is_noop(window, qapp) -> None:
    """Re-selecting the live theme leaves the palette untouched."""
    window._on_select_theme("dark")
    before = _window_rgb(qapp)
    window._on_select_theme("dark")
    assert _window_rgb(qapp) == before
    assert window._theme == "dark"


def test_theme_choice_is_persisted(window, isolated_theme) -> None:
    """Switching themes writes the choice through the persistence layer."""
    window._on_select_theme("dark")
    assert isolated_theme["key"] == "dark"
    assert theme.load_theme() == "dark"


def test_normalize_theme_coerces_unknown_to_default() -> None:
    """Any unknown/blank key falls back to the default theme."""
    assert theme.normalize_theme("DARK") == "dark"
    assert theme.normalize_theme(None) == theme.DEFAULT_THEME
    assert theme.normalize_theme("chartreuse") == theme.DEFAULT_THEME
    assert theme.DEFAULT_THEME in theme.THEME_KEYS
