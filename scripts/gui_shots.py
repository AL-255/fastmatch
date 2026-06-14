"""Headless screenshots of the FastMatch GUI for visual critique.

Renders the PySide6 UI under the offscreen Qt platform and grabs PNGs of
representative states (main window light/dark, the params panel in several
method/channel modes, the memory panel). The widget canvas is GPU/OpenGL and may
render blank offscreen, but all the chrome (panels, toolbar, menus, controls,
theming) renders — which is what GUI critique is about.

Usage:  QT_QPA_PLATFORM=offscreen python scripts/gui_shots.py [outdir]
Outputs <outdir>/*.png  (default: scripts/gui_shots_out/)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "gui_shots_out"
OUT.mkdir(parents=True, exist_ok=True)


def _settle(app, n=6):
    for _ in range(n):
        app.processEvents()


def _grab(widget, name):
    _settle(QApplication.instance())
    pm = widget.grab()
    p = OUT / f"{name}.png"
    pm.save(str(p))
    print(f"  wrote {p.name}  ({pm.width()}x{pm.height()})")


def main():
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)
    app = QApplication([sys.argv[0]])

    from fastmatch import theme
    from fastmatch.app import MainWindow
    from fastmatch.loader import generate_sample, load_image
    from fastmatch.memory import MemoryEntry, MemoryStore
    from fastmatch.params_panel import ParamsPanel
    from fastmatch.memory_panel import MemoryPanel
    from fastmatch.types import Match, MatchParams
    import torch

    # Small sample image so staging is instant.
    tmp = Path(tempfile.mkdtemp()) / "sample.png"
    generate_sample(tmp, w=1400, h=1000, tile=120, n_targets=10, seed=1)
    doc = load_image(str(tmp))

    win = MainWindow(doc, device="cpu")
    win.resize(1320, 880)
    win._on_select_theme("light")   # palette + window QSS + canvas
    win.show()
    _grab(win, "01_mainwindow_light")

    win._on_select_theme("dark")
    win.show()
    _grab(win, "02_mainwindow_dark")

    # Params panel in isolation, light, across method/channel states. Standalone
    # widgets need the theme QSS applied directly (no parent window to cascade it).
    theme.apply_theme(app, "light", persist=False)
    pp = ParamsPanel(effective_device=torch.device("cpu"))
    pp.setStyleSheet(theme.theme_qss("light"))
    pp.resize(340, 760)
    pp.show()
    pp._run_button.setEnabled(True)   # so the primary-action accent (:enabled) renders
    pp.set_match_count(7)             # so the muted match-count readout renders
    _grab(pp, "03_params_ncc_luminance")

    pp._channel_mode.setCurrentIndex(pp._channel_mode.findData("rgb"))
    _grab(pp, "04_params_rgb_weights")

    pp._channel_mode.setCurrentIndex(pp._channel_mode.findData("ycbcr"))
    _grab(pp, "05_params_ycbcr_weights")

    pp._select_method("features")
    pp._sync_method_page()
    _grab(pp, "06_params_features")

    # Dark params panel: verify bevels, disabled-text legibility, muted readout,
    # and the enabled primary-Run accent under the dark theme.
    theme.apply_theme(app, "dark", persist=False)
    pp.setStyleSheet(theme.theme_qss("dark"))
    pp._select_method("ncc")
    pp._sync_method_page()
    pp._channel_mode.setCurrentIndex(pp._channel_mode.findData("luminance"))
    _grab(pp, "08_params_dark")

    # Memory panel populated.
    theme.apply_theme(app, "light", persist=False)
    store = MemoryStore(source_image=str(tmp), image_size=(1400, 1000))
    for i in range(4):
        store.entries.append(MemoryEntry(
            selection=(100 + i * 30, 80, 120, 120),
            params=MatchParams(method=["ncc", "ssd", "features", "ccorr"][i],
                               channel_mode=["luminance", "rgb", "ycbcr", "luminance"][i]),
            matches=[Match(x=200, y=200, w=120, h=120, score=0.9 - 0.05 * i, scale=1.0,
                           orientation="R0")] * (i + 1),
        ))
    mp = MemoryPanel()
    mp.setStyleSheet(theme.theme_qss("light"))  # muted hint/secondary text
    if hasattr(mp, "set_store"):
        mp.set_store(store)
    elif hasattr(mp, "refresh"):
        mp._store = store
        mp.refresh()
    mp.resize(360, 420)
    mp.show()
    _grab(mp, "07_memory_panel")

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
