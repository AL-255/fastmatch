"""The FastMatch main window — the integration hub.

``MainWindow`` wires the three decoupled facets together exactly as ``DESIGN.md``
§A/§F.2 prescribe:

* :class:`~fastmatch.viewport.ImageViewport` — the tiled, GPU-composited canvas
  that emits ``regionSelected(QRect)`` (full-res image px, half-open).
* :class:`~fastmatch.controller.MatchController` — owns the worker thread, runs
  the NCC engine, emits ``matches_ready`` / ``busy_changed`` / ``progress`` /
  ``failed``.
* :class:`~fastmatch.params_panel.ParamsPanel` — emits ``MatchParams``.

The data flow is: a selection (or a param change re-using the last selection) is
sent to the controller; results come back on ``matches_ready`` and are pushed to
the viewport overlay. The one subtlety is the **threshold**: it is a live UI
filter, so a threshold-only change goes straight to ``viewport.set_match_threshold``
(no engine re-run); any *other* param change re-issues the last selection.

Heavy modules (viewport, controller) are imported lazily inside the constructor
so this module stays importable for tooling even before its sibling modules
exist, and so a headless ``import fastmatch.app`` does not force Qt/torch eagerly
at module top-level beyond what PySide6 already requires.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRect, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .calibration import Calibration
from .device import device_banner_text, resolve_device
from .document import ImageDocument
from .memory import MemoryEntry
from .about import AboutDialog
from .memory_panel import MemoryPanel
from .params_panel import _HIST_BIN_PRESETS, _HIST_BINS, ParamsPanel
from .types import CONV_METHODS, MatchParams
from . import theme

# File ▸ Open Recent: persisted history of opened image paths (most-recent first).
_RECENT_KEY = "files/recent"
_RECENT_MAX = 10


class MainWindow(QMainWindow):
    """Top-level window wiring viewport, controller, and params panel."""

    def __init__(self, doc: ImageDocument | None = None, *, device: str = "auto") -> None:
        """Construct the window, optionally around a loaded document.

        Args:
            doc: The image to display and search, or ``None`` to start with an
                empty canvas (no image) — e.g. launching with no file argument.
                An image is then loaded via *File ▸ Open Image…*.
            device: Device preference passed to the engine ("auto"/"cuda"/"cpu").
        """
        super().__init__()
        # Imported here (not at module top) so app.py imports cleanly even while
        # its sibling modules are still being written in parallel.
        from .controller import MatchController
        from .viewport import ImageViewport

        self.setWindowTitle("FastMatch")
        self._doc = doc

        # Resolve the device once for the banner + the params panel. The
        # controller resolves it again internally for the engine; both go through
        # the same canary-gated resolve_device, so they agree.
        self._device_pref = device
        self._resolved_device = resolve_device(device)

        # Seed params so the first request is well-formed AND carries the device
        # preference (§C.5). ParamsPanel.current_params() legitimately returns
        # device="auto", so we re-apply self._device_pref every time we read the
        # panel (here and in the params_changed handler) to keep the CLI/UI
        # preference from being silently dropped to "auto".
        self._params: MatchParams = MatchParams(device=device)
        # Remember the last selection so non-threshold param changes can re-run it.
        self._last_rect: QRect | None = None
        # Mask for the last selection (rectilinear/orthogonal-polygon), or None for
        # a plain rectangle; threaded to the matcher on every (re-)run of this rect.
        self._last_mask = None
        # Physical-scale calibration (None until the user calibrates); reset per
        # image. _tool_return_mode restores Pan/Select after a one-shot tool.
        self._calibration: Calibration | None = None
        self._last_cursor: tuple[int, int] | None = None
        self._tool_return_mode = None
        # Last full result set from the engine, kept so a live threshold change can
        # refresh the orientation breakdown without re-running the engine.
        self._all_matches: list = []
        # Track whether we've shown the one-time "CPU is slow" notice.
        self._cpu_notice_shown = False
        # Controllers whose worker thread was still running at swap/close time
        # are parked here (not destroyed) so the live QThread is never GC'd while
        # running; each one's thread.finished -> deleteLater reclaims it once its
        # wedged job finally returns (§C.3 / §C.6).
        self._orphaned_controllers: list = []

        # --- Central viewport ---------------------------------------------
        self._viewport = ImageViewport(self)
        if doc is not None:
            self._viewport.set_image(doc)  # else: start on the empty canvas
        self.setCentralWidget(self._viewport)

        # --- Controller (owns worker thread) -------------------------------
        # With no image there is nothing to search, so we defer building the
        # engine/worker thread until an image is opened (_swap_document). A None
        # controller is the same idle state reached via File > Close Image.
        self._controller = MatchController(doc, self._params) if doc is not None else None

        # --- Params panel + device banner dock -----------------------------
        self._params_panel = ParamsPanel(self._resolved_device.type, self)
        # Re-apply the device preference: the panel returns device="auto", which
        # would otherwise overwrite the seeded preference (§C.5).
        self._params = dataclasses.replace(
            self._params_panel.current_params(), device=self._device_pref
        )
        # Auto Run state (panel default is on -> preserves the run-on-change UX).
        self._auto_run = self._params_panel.auto_run_enabled()
        # Render the image as the matcher sees it: grayscale in luminance mode,
        # colour in rgb mode (the channel-mode combo drives the display).
        self._viewport.set_display_grayscale(self._params.channel_mode == "luminance")
        # Colour the image canvas for the active application theme. The app-level
        # palette/style is applied at bootstrap (__main__); the canvas is themed
        # separately because it is not a Qt palette role.
        self._theme = theme.load_theme()
        self._viewport.set_background_color(theme.viewport_background(self._theme))
        # Window-level QSS (muted secondary text + primary Run accent). Applied to
        # the window, not the QApplication, so the global Fusion style stays intact
        # while the QSS still cascades to the dock's banner/match-count/Run.
        self.setStyleSheet(theme.theme_qss(self._theme))

        dock_body = QWidget(self)
        dock_layout = QVBoxLayout(dock_body)
        # Zero the wrapper margins so the ParamsPanel's own 8px inset is the sole
        # one — matching the Memory dock (whose panel is the dock widget) so both
        # docks share one left edge.
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(0)
        # The primary controls (Run / Method / Match parameters) lead the dock; the
        # device/engine banner is a low-priority diagnostic, so it sits at the
        # bottom (and the same hint is in the status bar).
        self._banner = QLabel(device_banner_text(self._resolved_device), dock_body)
        self._banner.setWordWrap(True)
        self._banner.setObjectName("deviceBanner")
        # Share the params panel's 8px horizontal inset so the dock column has one
        # left edge (the panel uses setContentsMargins(8,...); the banner is a
        # sibling in the dock layout and would otherwise sit 8px further left).
        self._banner.setContentsMargins(8, 0, 8, 0)
        dock_layout.addWidget(self._params_panel)
        dock_layout.addWidget(self._banner)
        dock_layout.addStretch(1)

        self._dock = QDockWidget("Search", self)
        self._dock.setWidget(dock_body)
        self._dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)

        # --- Memory panel dock --------------------------------------------
        # A bottom dock so the per-entry stats table has horizontal room. The
        # panel holds the list of saved searches; the source header is seeded
        # here and re-seeded on every document swap so saved coords stay tied to
        # the right image.
        self._memory = MemoryPanel(self)
        if self._doc is not None:
            self._memory.set_source(self._doc.path, self._viewport.image_size())
        self._memory_dock = QDockWidget("Memory", self)
        self._memory_dock.setWidget(self._memory)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._memory_dock)

        # --- Toolbar + menu bar -------------------------------------------
        self._build_toolbar()
        self._build_menus()

        # --- Status bar ----------------------------------------------------
        self._cursor_label = QLabel("Cursor: (-, -)", self)
        self._area_label = QLabel("", self)        # physical area of the selection
        self._count_label = QLabel("0 matches", self)
        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setMaximumWidth(180)
        self._progress.setVisible(False)
        # Left-aligned focus-mode hint (always shown; text reflects on/off state).
        self._focus_label = QLabel("Press Space to enter focus mode", self)
        sb = self.statusBar()
        sb.addWidget(self._focus_label)            # left side
        sb.addPermanentWidget(self._cursor_label)  # right side
        sb.addPermanentWidget(self._area_label)
        sb.addPermanentWidget(self._count_label)
        sb.addPermanentWidget(self._progress)

        # --- Wiring --------------------------------------------------------
        self._connect_signals()

        # On CPU, notify once that searches will be slow. This MUST be
        # non-blocking: a modal QMessageBox.exec on startup hangs headless /
        # automated runs (no one clicks "OK"), so we surface it as a status-bar
        # message instead. Deferred to a 0ms single-shot so it lands after the
        # event loop is up and construction has returned.
        if self._resolved_device.type == "cpu":
            QTimer.singleShot(0, self._maybe_show_cpu_notice)

        # Empty-canvas startup (no document): mirror the File > Close Image idle
        # state — nothing to close/run and a "No image" readout until one is opened.
        if self._doc is None:
            self._act_close_image.setEnabled(False)
            self._params_panel.set_run_enabled(False)
            self._count_label.setText("No image")
        else:
            # A CLI-launched image is "opened" too — record it in the recent list.
            self._record_recent(self._doc.path)

        # Fit the freshly loaded image into the view (no-op when empty).
        self._viewport.fit_in_view()

    # ------------------------------------------------------------------ build
    def _build_toolbar(self) -> None:
        """Create the toolbar actions (Open, mode toggle, Fit, Clear, Self-test)."""
        tb = QToolBar("Main", self)
        tb.setObjectName("mainToolbar")
        self.addToolBar(tb)

        # Toolbar uses sentence case ("Open image…") to match its siblings (Select
        # mode / Clear matches); the File menu gets its own Title-Case "Open Image…"
        # action below (menu convention: Close Image, Open Memory…).
        self._act_open = QAction("Open image…", self)
        self._act_open.setToolTip("Open an image to search (PNG / JPEG / TIFF / BMP).")
        self._act_open.triggered.connect(self._on_open)
        tb.addAction(self._act_open)

        tb.addSeparator()

        # Mode toggle: checkable; unchecked == Select, checked == Pan. The label
        # names the CURRENT mode (see _on_mode_toggled), so it starts "Select mode"
        # to match the unchecked/Select startup state.
        self._act_mode = QAction("Select mode", self)
        self._act_mode.setCheckable(True)
        self._act_mode.setToolTip("Toggle between Select (draw region) and Pan (drag to move).")
        self._act_mode.toggled.connect(self._on_mode_toggled)
        tb.addAction(self._act_mode)

        # Rectilinear select: an orthogonal-polygon region tool. Checkable and
        # mutually exclusive with the Select/Pan toggle (each unchecks the other).
        self._act_rectilinear = QAction("Rectilinear select", self)
        self._act_rectilinear.setCheckable(True)
        self._act_rectilinear.setToolTip(
            "Rectilinear (right-angle polygon) select: click to drop vertices "
            "(edges snap horizontal/vertical); click the first vertex or double-click "
            "/ Enter to close, Esc to cancel."
        )
        self._act_rectilinear.toggled.connect(self._on_rectilinear_toggled)
        tb.addAction(self._act_rectilinear)

        self._act_fit = QAction("Fit", self)
        self._act_fit.setShortcut(QKeySequence("F"))  # F = zoom-to-fit
        self._act_fit.setToolTip("Zoom to fit the whole image (F).")
        self._act_fit.triggered.connect(self._viewport.fit_in_view)
        tb.addAction(self._act_fit)

        tb.addSeparator()  # view group | results group

        self._act_clear = QAction("Clear matches", self)
        self._act_clear.setToolTip(
            "Clear the result boxes and the current selection (draw a new region to "
            "search again)."
        )
        self._act_clear.triggered.connect(self._on_clear_matches)
        tb.addAction(self._act_clear)
        # "Add to Memory" lives only on the Memory dock now (was duplicated here).

        tb.addSeparator()

        # Measurement tools (also listed under the Tools menu).
        self._act_calibrate = QAction("Calibrate", self)
        self._act_calibrate.setToolTip(
            "Drag a line of known physical length, then enter that length to set the "
            "pixel↔physical scale (uses the longer of the horizontal/vertical span; "
            "the first point becomes the physical origin)."
        )
        self._act_calibrate.triggered.connect(self._on_calibrate_tool)
        tb.addAction(self._act_calibrate)

        self._act_measure = QAction("Measure", self)
        self._act_measure.setToolTip("Drag a line on the image to measure its physical distance.")
        self._act_measure.triggered.connect(self._on_measure_tool)
        tb.addAction(self._act_measure)

        # Self-test is a diagnostic, not a primary tool — the action is created here
        # (toolbar build runs before the menus) but lives in the Help menu, not the
        # toolbar (see _build_help_menu).
        self._act_selftest = QAction("Self-test", self)
        self._act_selftest.setToolTip(
            "Generate a labelled sample, search one motif, and report recall."
        )
        self._act_selftest.triggered.connect(self._on_self_test)

    def _build_menus(self) -> None:
        """Menu bar — File (image + memory ops) and View (box appearance)."""
        # Keep references on self: a menu held only by a local can have its C++
        # object GC-deleted by shiboken (-> "QMenu already deleted" on access).
        file_menu = self._file_menu = self.menuBar().addMenu("&File")
        # Image operations. A dedicated Title-Case menu action (the toolbar keeps
        # its own sentence-case "Open image…"); both trigger _on_open.
        self._act_open_menu = QAction("Open Image…", self)
        self._act_open_menu.triggered.connect(self._on_open)
        file_menu.addAction(self._act_open_menu)
        self._act_close_image = QAction("Close Image", self)
        self._act_close_image.setToolTip("Close the current image and clear the view.")
        self._act_close_image.triggered.connect(self._on_close_image)
        file_menu.addAction(self._act_close_image)
        # Recently opened images (most-recent first; persisted across sessions).
        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        # Memory operations (backed by the Memory panel's file methods).
        act_open_mem = QAction("Open Memory…", self)
        act_open_mem.triggered.connect(self._memory.open_memory)
        file_menu.addAction(act_open_mem)
        act_save_mem = QAction("Save Memory", self)
        act_save_mem.setShortcut(QKeySequence.StandardKey.Save)  # Ctrl+S
        act_save_mem.triggered.connect(self._memory.save_memory)
        file_menu.addAction(act_save_mem)
        act_save_mem_as = QAction("Save Memory As…", self)
        act_save_mem_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_mem_as.triggered.connect(self._memory.save_memory_as)
        file_menu.addAction(act_save_mem_as)
        act_close_mem = QAction("Close Memory", self)
        act_close_mem.setToolTip("Clear all memory entries.")
        act_close_mem.triggered.connect(self._memory.close_memory)
        file_menu.addAction(act_close_mem)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = self._view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._act_fit)  # Fit (F), also on the toolbar
        view_menu.addSeparator()

        boxes_menu = view_menu.addMenu("Match &boxes")

        # Line width: an exclusive set of presets (cosmetic device px).
        width_menu = boxes_menu.addMenu("Line &width")
        self._box_width_group = QActionGroup(self)
        self._box_width_group.setExclusive(True)
        for px in (1, 2, 3, 4, 6):
            act = QAction(f"{px} px", self, checkable=True)
            act.setChecked(px == 1)  # default 1 px
            act.triggered.connect(lambda _checked, w=px: self._viewport.set_box_line_width(w))
            self._box_width_group.addAction(act)
            width_menu.addAction(act)

        # Score histogram (above the threshold slider): tunable bin count, an
        # exclusive set of presets. "Hidden" here in the View menu rather than the
        # always-visible params panel.
        bins_menu = view_menu.addMenu("Histogram &bins")
        self._hist_bins_group = QActionGroup(self)
        self._hist_bins_group.setExclusive(True)
        for nbins in _HIST_BIN_PRESETS:
            act = QAction(f"{nbins} bins", self, checkable=True)
            act.setChecked(nbins == _HIST_BINS)  # default
            act.triggered.connect(
                lambda _checked, n=nbins: self._params_panel.set_histogram_bins(n)
            )
            self._hist_bins_group.addAction(act)
            bins_menu.addAction(act)

        # XOR-with-background toggle.
        self._act_box_xor = QAction("&XOR with background", self, checkable=True)
        self._act_box_xor.setToolTip(
            "Draw box outlines XORed with the pixels underneath, so they stay "
            "visible on any background."
        )
        self._act_box_xor.toggled.connect(self._viewport.set_box_xor)
        boxes_menu.addAction(self._act_box_xor)

        # Dock visibility toggles — a closed Search/Memory dock (the X on its title
        # bar) would otherwise be unrecoverable without restarting. Qt's built-in
        # toggleViewAction is a checkable show/hide that stays in sync automatically.
        view_menu.addSeparator()
        act_dock_search = self._dock.toggleViewAction()
        act_dock_search.setText("&Search panel")
        view_menu.addAction(act_dock_search)
        act_dock_memory = self._memory_dock.toggleViewAction()
        act_dock_memory.setText("&Memory panel")
        view_menu.addAction(act_dock_memory)

        self._build_tools_menu()
        self._build_theme_menu()
        self._build_engine_menu()

        # Help menu (last, by convention) — home of the Self-test diagnostic, which
        # was demoted off the primary toolbar.
        self._help_menu = self.menuBar().addMenu("&Help")
        self._help_menu.addAction(self._act_selftest)
        self._help_menu.addSeparator()
        self._act_about = QAction("&About FastMatch", self)
        self._act_about.setMenuRole(QAction.MenuRole.AboutRole)  # -> app menu on macOS
        self._act_about.triggered.connect(self._on_about)
        self._help_menu.addAction(self._act_about)

    def _build_tools_menu(self) -> None:
        """&Tools menu — physical-scale calibration and the measuring ruler.

        Reuses the Calibrate / Measure actions created on the toolbar.
        """
        tools_menu = self._tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction(self._act_calibrate)
        tools_menu.addAction(self._act_measure)

        tools_menu.addSeparator()
        self._act_clear_measure = QAction("Clear measurement", self)
        self._act_clear_measure.triggered.connect(self._on_clear_measurement)
        tools_menu.addAction(self._act_clear_measure)
        self._act_clear_calib = QAction("Clear calibration", self)
        self._act_clear_calib.triggered.connect(self._on_clear_calibration)
        tools_menu.addAction(self._act_clear_calib)

    def _build_theme_menu(self) -> None:
        """&Theme menu — switch the application look (System / Light / Dark).

        An exclusive radio group; the entry matching the persisted theme starts
        checked. Selecting one re-themes the whole app live and remembers the
        choice for next launch.
        """
        theme_menu = self._theme_menu = self.menuBar().addMenu("&Theme")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self._theme_actions: dict[str, QAction] = {}
        for key in theme.THEME_KEYS:
            act = QAction(theme.THEME_LABELS[key], self, checkable=True)
            act.setChecked(key == self._theme)
            act.triggered.connect(lambda _checked, k=key: self._on_select_theme(k))
            self._theme_group.addAction(act)
            theme_menu.addAction(act)
            self._theme_actions[key] = act

    def _on_select_theme(self, key: str) -> None:
        """Re-theme the whole application at runtime (menu callback)."""
        key = theme.normalize_theme(key)
        if key == self._theme:
            return
        app = QApplication.instance()
        if app is None:  # pragma: no cover - GUI always has an app
            return
        self._theme = theme.apply_theme(app, key)  # palette + style + persist
        self._theme_actions[self._theme].setChecked(True)  # keep the radio honest
        self._viewport.set_background_color(theme.viewport_background(self._theme))
        self.setStyleSheet(theme.theme_qss(self._theme))  # secondary text + Run accent

    def _build_engine_menu(self) -> None:
        """&Engine menu — pick the compute backend (CPU / CUDA / Auto) at runtime.

        An exclusive radio group of three presets mirroring the device-preference
        strings understood by :func:`resolve_device`. The CUDA entry is disabled
        when no working CUDA device is present (the canary-gated probe failed), so
        the user can never select an unavailable backend. The entry matching the
        current :attr:`_device_pref` starts checked.
        """
        engine_menu = self._engine_menu = self.menuBar().addMenu("&Engine")
        self._engine_group = QActionGroup(self)
        self._engine_group.setExclusive(True)

        cuda_available = resolve_device("cuda").type == "cuda"
        # (preference-string, label, tooltip, enabled).
        specs = (
            ("auto", "&Auto (prefer GPU)",
             "Use the GPU when available, otherwise fall back to the CPU."),
            ("cuda", "&CUDA (GPU)",
             "Force the CUDA GPU backend." if cuda_available
             else "No working CUDA device was detected on this machine."),
            ("cpu", "C&PU",
             "Force the CPU backend (slower, but always available)."),
        )
        self._engine_actions: dict[str, QAction] = {}
        for key, label, tip in specs:
            act = QAction(label, self, checkable=True)
            act.setToolTip(tip)
            if key == "cuda" and not cuda_available:
                act.setEnabled(False)
            act.setChecked(key == self._device_pref)
            act.triggered.connect(lambda _checked, k=key: self._on_select_engine(k))
            self._engine_group.addAction(act)
            engine_menu.addAction(act)
            self._engine_actions[key] = act

    def _on_select_engine(self, key: str) -> None:
        """Switch the compute backend at runtime (menu callback).

        Rebuilds the controller on the new device, re-gates device-dependent
        controls (the multi-scale grid is GPU-only), refreshes the banner, and
        re-runs the last query if Auto Run is on. If the in-flight controller is
        still busy and won't join, the switch is aborted and the radio reverts to
        the previous backend so the menu never lies about the live engine.
        """
        if key == self._device_pref:
            return

        previous = self._device_pref
        if not self._teardown_controller():
            # Busy worker parked: undo the radio selection and tell the user.
            self._engine_actions[previous].setChecked(True)
            QMessageBox.warning(
                self, "Busy", "A search is still finishing; please wait and try again."
            )
            return

        # Commit the new device across all the places that cache it.
        self._device_pref = key
        self._engine_actions[key].setChecked(True)  # keep the radio honest
        self._resolved_device = resolve_device(key)
        self._params_panel.set_device(self._resolved_device.type)
        self._params = dataclasses.replace(
            self._params_panel.current_params(), device=self._device_pref
        )
        self._banner.setText(device_banner_text(self._resolved_device))

        # Build a fresh controller bound to the new device, then reset the busy UI.
        self._install_new_controller()
        self._on_busy_changed(False)

        # Re-run the last query so results reflect the new engine immediately.
        if self._auto_run and self._last_rect is not None:
            self._controller.request(self._last_rect, self._params, self._last_mask)

    def _connect_signals(self) -> None:
        """Connect viewport/controller/panel signals per DESIGN.md §F.2."""
        # Selection -> controller request.
        self._viewport.regionSelected.connect(self._on_region_selected)
        # Live cursor readout.
        self._viewport.cursorImagePos.connect(self._on_cursor_pos)
        # Measurement tools: two-point picks for calibration and the ruler.
        self._viewport.calibrationPicked.connect(self._on_calibration_picked)
        self._viewport.measurePicked.connect(self._on_measure_picked)
        # Focus mode (Space): show a left-aligned status-bar hint while active.
        self._viewport.focusModeChanged.connect(self._on_focus_mode_changed)

        # Controller results / state. With no image at startup there is no
        # controller yet; _install_new_controller wires these when one is opened.
        if self._controller is not None:
            self._controller.matches_ready.connect(self._on_matches_ready)
            self._controller.busy_changed.connect(self._on_busy_changed)
            self._controller.progress.connect(self._on_progress)
            self._controller.failed.connect(self._on_failed)

        # Params panel: threshold is a live filter, everything else re-runs.
        self._params_panel.params_changed.connect(self._on_params_changed)
        # Run button + Auto Run toggle.
        self._params_panel.run_clicked.connect(self._on_run)
        self._params_panel.auto_run_changed.connect(self._on_auto_run_changed)

        # Memory panel: add the current search, revisit a saved entry's boxes,
        # and react to a freshly loaded store (possibly switching images).
        self._memory.add_requested.connect(self._on_add_to_memory)
        self._memory.entry_activated.connect(self._on_revisit_entry)
        self._memory.store_loaded.connect(self._on_memory_loaded)

    # ----------------------------------------------------------------- actions
    def _on_open(self) -> None:
        """Open a file dialog, load the chosen image, and rebuild around it."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)",
        )
        if not path:
            return
        # Import lazily to avoid a hard top-level dependency cycle with loader.
        from .loader import load_image

        try:
            doc = load_image(path)
        except Exception as exc:  # surface load failures rather than crash
            QMessageBox.critical(self, "Open failed", f"Could not load image:\n{exc}")
            return
        self._swap_document(doc)

    # ------------------------------------------------------- recent images
    def _load_recent(self) -> list[str]:
        """Persisted recent-image paths, most-recent first, dropping any that no
        longer exist on disk (so stale entries self-prune)."""
        raw = theme.settings().value(_RECENT_KEY, [])
        if isinstance(raw, str):          # a lone entry can round-trip as a bare str
            raw = [raw]
        elif not isinstance(raw, (list, tuple)):
            raw = []
        out: list[str] = []
        seen: set[str] = set()
        for p in raw:
            p = str(p)
            if p and p not in seen and Path(p).is_file():
                out.append(p)
                seen.add(p)
        return out[:_RECENT_MAX]

    def _record_recent(self, path: str) -> None:
        """Push a successfully-opened image to the front of the recent list."""
        if not path:
            return
        try:
            ap = str(Path(path).resolve())
        except Exception:
            ap = path
        if not Path(ap).is_file():        # skip synthetic / in-memory sources
            return
        items = [p for p in self._load_recent() if p != ap]
        items.insert(0, ap)
        theme.settings().setValue(_RECENT_KEY, items[:_RECENT_MAX])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        """Repopulate the *Open Recent* submenu from the persisted list."""
        menu = self._recent_menu
        menu.clear()
        items = self._load_recent()
        if not items:
            act = menu.addAction("(No recent images)")
            act.setEnabled(False)
            return
        for p in items:
            # File name, plus the parent dir for disambiguation; full path on hover.
            parent = Path(p).parent.name
            label = f"{Path(p).name}  —  {parent}" if parent else Path(p).name
            act = QAction(label, self)
            act.setToolTip(p)
            act.triggered.connect(lambda _checked=False, path=p: self._open_recent(path))
            menu.addAction(act)
        menu.addSeparator()
        clear = menu.addAction("Clear Recent")
        clear.triggered.connect(self._clear_recent)

    def _open_recent(self, path: str) -> None:
        """Open an image chosen from the recent list (validating it still exists)."""
        if not Path(path).is_file():
            QMessageBox.warning(
                self, "Open Recent", f"This image is no longer available:\n{path}"
            )
            theme.settings().setValue(
                _RECENT_KEY, [p for p in self._load_recent() if p != path]
            )
            self._rebuild_recent_menu()
            return
        from .loader import load_image

        try:
            doc = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", f"Could not load image:\n{exc}")
            return
        self._swap_document(doc)

    def _clear_recent(self) -> None:
        """Forget the recent-image history."""
        theme.settings().remove(_RECENT_KEY)
        self._rebuild_recent_menu()

    def _on_close_image(self) -> None:
        """Close the current image: clear the view, matches, and selection.

        The matching controller is left idle (no image -> no selections can be
        drawn); opening another image rebuilds it via _swap_document.
        """
        self._viewport.clear_matches()
        self._viewport.clear_template()
        self._viewport.clear_image()
        self._last_rect = None
        self._last_mask = None
        self._all_matches = []
        self._calibration = None  # calibration is image-specific
        self._area_label.setText("")
        self._params_panel.set_run_enabled(False)
        self._params_panel.set_match_count(0)
        self._params_panel.set_score_histogram([])
        self._memory.set_source("", (0, 0))
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._count_label.setText("No image")
        self._act_close_image.setEnabled(False)  # nothing to close now
        self.statusBar().showMessage("Image closed.", 4000)

    def _teardown_controller(self) -> bool:
        """Shut down the current controller; park it (alive) if still busy.

        Returns True if the old controller's worker thread joined and was handed
        to ``deleteLater`` (safe to build a replacement), or False if the bounded
        join timed out — in which case the controller is PARKED (kept alive so its
        still-running QThread is never GC'd -> SIGABRT) and the caller must not
        build a new engine (§C.3 / §C.6).
        """
        old = self._controller
        if old is None:
            return True  # empty-canvas state: no controller to tear down
        if not old.shutdown():
            self._orphaned_controllers.append(old)
            return False
        old.deleteLater()
        return True

    def _install_new_controller(self) -> None:
        """Build a fresh controller for the current doc + params and wire signals.

        The previous controller must already have been relinquished via
        :meth:`_teardown_controller`. With no document (empty canvas) there is
        nothing to search, so the controller stays ``None``.
        """
        from .controller import MatchController

        if self._doc is None:
            self._controller = None
            return
        self._controller = MatchController(self._doc, self._params)
        self._controller.matches_ready.connect(self._on_matches_ready)
        self._controller.busy_changed.connect(self._on_busy_changed)
        self._controller.progress.connect(self._on_progress)
        self._controller.failed.connect(self._on_failed)

    def _swap_document(self, doc: ImageDocument) -> None:
        """Replace the current document, controller, and viewport image.

        Tears the old controller's worker thread down *before* building a new one
        (never two CUDA contexts / worker threads at once). If the join times out
        (a wedged job), the old controller is parked and we bail with a warning.
        """
        if not self._teardown_controller():
            QMessageBox.warning(
                self, "Busy", "A search is still finishing; please wait and try again."
            )
            return

        self._doc = doc
        self._record_recent(doc.path)  # remember it in File ▸ Open Recent
        self._last_rect = None
        self._last_mask = None
        self._all_matches = []
        self._calibration = None  # calibration is per-image; drop it on swap
        self._area_label.setText("")
        self._params_panel.set_run_enabled(False)  # new image -> no selection yet
        self._viewport.clear_matches()
        self._viewport.clear_template()
        self._viewport.set_image(doc)
        # Keep the display mode (grayscale/colour) consistent for the new image.
        self._viewport.set_display_grayscale(self._params.channel_mode == "luminance")
        self._install_new_controller()
        # Reset the busy UI for the fresh controller: the old controller may have
        # been mid-search, leaving the progress bar visible (§C.17).
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._viewport.fit_in_view()
        self._count_label.setText("0 matches")
        self._params_panel.set_match_count(0)
        self._params_panel.set_score_histogram([])
        # Re-seed the Memory panel's source header so entries added after the
        # swap record the new image (existing entries are left untouched).
        self._memory.set_source(self._doc.path, self._viewport.image_size())
        self._act_close_image.setEnabled(True)  # an image is open again

    def _on_mode_toggled(self, checked: bool) -> None:
        """Switch the viewport between Pan (checked) and Select (unchecked)."""
        from .viewport import ImageViewport

        # Leaving for Select/Pan turns the rectilinear tool off (mutually exclusive).
        if self._act_rectilinear.isChecked():
            self._act_rectilinear.blockSignals(True)
            self._act_rectilinear.setChecked(False)
            self._act_rectilinear.blockSignals(False)
        mode = ImageViewport.Mode.PAN if checked else ImageViewport.Mode.SELECT
        self._viewport.set_mode(mode)
        self._act_mode.setText("Pan mode" if checked else "Select mode")

    def _on_rectilinear_toggled(self, checked: bool) -> None:
        """Enter the rectilinear-polygon select tool (or fall back to box Select)."""
        from .viewport import ImageViewport

        if checked:
            # Drop the Pan toggle back to its Select label without re-entering it.
            if self._act_mode.isChecked():
                self._act_mode.blockSignals(True)
                self._act_mode.setChecked(False)
                self._act_mode.setText("Select mode")
                self._act_mode.blockSignals(False)
            self._viewport.set_mode(ImageViewport.Mode.RECTILINEAR)
        else:
            self._viewport.set_mode(ImageViewport.Mode.SELECT)

    def _on_clear_matches(self) -> None:
        """Clear overlay matches and the current template selection."""
        self._viewport.clear_matches()
        self._viewport.clear_template()
        self._last_rect = None
        self._last_mask = None
        self._all_matches = []
        self._params_panel.set_run_enabled(False)  # nothing to Run anymore
        self._count_label.setText("0 matches")
        self._params_panel.set_match_count(0)
        self._params_panel.set_score_histogram([])

    # ----------------------------------------------------------------- memory
    def _on_add_to_memory(self) -> None:
        """Append the current selection + visible matches as a Memory entry.

        The matches stored are the ones currently *above* the live threshold
        (what the user sees in the overlay), captured from the engine's full
        result set in :attr:`_all_matches`. A no-op (with a status hint) if there
        is no completed search to save.
        """
        if self._last_rect is None or not self._all_matches:
            self.statusBar().showMessage("Run a match first to add to Memory.", 6000)
            return
        visible = [m for m in self._all_matches if m.score >= self._params.threshold]
        rect = self._last_rect
        sel = (rect.x(), rect.y(), rect.width(), rect.height())
        # Record the COMPLETE settings (channel mode, orientation flags, scales,
        # NMS/feature params, ...) via the full MatchParams snapshot.
        entry = MemoryEntry(selection=sel, params=self._params, matches=list(visible))
        self._memory.add_entry(entry)
        self.statusBar().showMessage(f"Added {len(visible)} matches to Memory.", 6000)

    def _on_revisit_entry(self, entry: object) -> None:
        """Recall a saved entry: restore its reference box and matches (revisit).

        Pushes the entry's recorded matches onto the overlay, restores the blue
        reference box to the entry's remembered selection (so the source is shown
        where it was captured), and drops the live threshold to 0 so every saved
        box shows regardless of the current slider. Revisit only makes sense for
        the current image; if the entry was recorded against a different image the
        boxes will not line up, which is the user's call.
        """
        matches = list(entry.matches)  # type: ignore[attr-defined]
        self._viewport.set_matches(matches)
        # Restore the reference (blue) box to the remembered selection, and make
        # it the active selection so a subsequent Run / param change uses it.
        sx, sy, sw, sh = entry.selection  # type: ignore[attr-defined]
        rect = QRect(int(sx), int(sy), int(sw), int(sh))
        self._viewport.set_template_rect(rect)
        self._last_rect = QRect(rect)
        self._params_panel.set_run_enabled(True)
        # Show every saved box: the entry already holds the matches the user
        # chose to remember, so do not re-filter them by the current slider.
        try:
            self._viewport.set_match_threshold(0.0)
        except Exception:
            pass
        self._all_matches = matches
        try:
            shown = self._viewport.visible_match_count()
        except Exception:
            shown = len(matches)
        self._count_label.setText(self._count_text(shown, matches, 0.0))
        self._params_panel.set_match_count(shown)
        self.statusBar().showMessage(
            f"Revisiting memory entry: showing {len(matches)} saved matches.", 6000
        )

    def _on_memory_loaded(self, store: object) -> None:
        """React to a store loaded from disk: optionally switch to its image.

        If the loaded store names a different source image than the current
        document, offer to open it (so the entries' coords line up). If the file
        is missing, warn non-blockingly that the entries were recorded for a
        different image. The store's entries are already loaded into the panel by
        the panel itself; we only handle the image side here.
        """
        source = getattr(store, "source_image", "")
        if not source or source == self._doc.path:
            return
        if os.path.exists(source):
            reply = QMessageBox.question(
                self,
                "Open source image?",
                "These saved searches were recorded against a different image:\n"
                f"{source}\n\nOpen it so the saved boxes line up?",
            )
            if reply == QMessageBox.StandardButton.Yes:
                from .loader import load_image

                try:
                    doc = load_image(source)
                except Exception as exc:  # surface load failures rather than crash
                    QMessageBox.critical(
                        self, "Open failed", f"Could not load image:\n{exc}"
                    )
                    return
                self._swap_document(doc)
        else:
            self.statusBar().showMessage(
                "Loaded memory entries were recorded for a different image "
                f"({source}), which was not found; saved boxes may not line up.",
                10000,
            )

    # --------------------------------------------------------- viewport events
    def _on_region_selected(self, rect: QRect) -> None:
        """Record a freshly drawn selection; search now only if Auto Run is on."""
        self._last_rect = QRect(rect)  # copy: caller may reuse/mutate its QRect
        # Capture the rectilinear mask (None for a plain rectangle) for this rect.
        self._last_mask = self._viewport.selection_mask()
        self._params_panel.set_run_enabled(True)  # there is now something to Run
        self._update_area_label()
        if self._auto_run:
            self._controller.request(rect, self._params, self._last_mask)
        else:
            self.statusBar().showMessage("Selection set — press Run to search.", 4000)

    # ------------------------------------------------- calibration / measure
    def _enter_tool(self, mode) -> None:
        """Enter a one-shot measurement tool, remembering the mode to restore."""
        if self._doc is None:
            self.statusBar().showMessage("Open an image first.", 4000)
            return
        from .viewport import ImageViewport

        self._tool_return_mode = (
            ImageViewport.Mode.PAN if self._act_mode.isChecked() else ImageViewport.Mode.SELECT
        )
        self._viewport.set_mode(mode)

    def _exit_tool(self) -> None:
        """Restore the Pan/Select mode after a one-shot tool completes."""
        from .viewport import ImageViewport

        self._viewport.set_mode(self._tool_return_mode or ImageViewport.Mode.SELECT)

    def _on_calibrate_tool(self) -> None:
        from .viewport import ImageViewport

        self._enter_tool(ImageViewport.Mode.CALIBRATE)
        if self._doc is not None:
            self.statusBar().showMessage(
                "Calibrate: drag a line along a span of known physical length.", 6000
            )

    def _on_measure_tool(self) -> None:
        from .viewport import ImageViewport

        self._enter_tool(ImageViewport.Mode.MEASURE)
        if self._doc is not None:
            self.statusBar().showMessage("Measure: drag a line to measure its distance.", 6000)

    def _on_calibration_picked(self, p1: QPointF, p2: QPointF) -> None:
        """Two points picked: ask for the physical length and set the scale."""
        self._exit_tool()
        ref_px = Calibration.reference_span((p1.x(), p1.y()), (p2.x(), p2.y()))
        if ref_px < 1.0:
            self.statusBar().showMessage("Calibration span too short — try a longer drag.", 5000)
            self._viewport.clear_calibration()
            return
        default = self._calibration.unit if self._calibration is not None else "mm"
        text, ok = QInputDialog.getText(
            self,
            "Calibrate scale",
            f"The longer span is {ref_px:.1f} px.\n"
            "Enter its physical length (e.g. '5.36 mm'):",
            text=f"1 {default}",
        )
        if not ok or not text.strip():
            return
        length, unit = self._parse_length(text, default)
        if length is None or length <= 0:
            self.statusBar().showMessage("Could not read a positive length from that input.", 5000)
            return
        try:
            cal = Calibration.from_two_points((p1.x(), p1.y()), (p2.x(), p2.y()), length, unit)
        except ValueError:
            self.statusBar().showMessage("Degenerate calibration; please try again.", 5000)
            return
        self._set_calibration(cal)
        self.statusBar().showMessage(
            f"Calibrated: {length:g} {unit} over {ref_px:.1f} px "
            f"→ {cal.scale:.4g} {unit}/px.",
            8000,
        )

    def _on_measure_picked(self, p1: QPointF, p2: QPointF) -> None:
        """Two points picked: report the distance (physical if calibrated)."""
        self._exit_tool()
        import math

        px = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
        if self._calibration is not None:
            phys = self._calibration.format_length(px)
            self.statusBar().showMessage(f"Distance: {phys}  ({px:.1f} px)", 0)
        else:
            self.statusBar().showMessage(
                f"Distance: {px:.1f} px  (calibrate to show physical units)", 0
            )

    @staticmethod
    def _parse_length(text: str, fallback_unit: str) -> tuple[float | None, str]:
        """Parse '<number> [unit]' -> (value, unit); unit falls back if omitted."""
        import re

        m = re.match(r"\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*(.*)$", text.strip())
        if not m:
            return None, fallback_unit
        value = float(m.group(1))
        unit = m.group(2).strip() or fallback_unit
        return value, unit

    def _set_calibration(self, cal: Calibration | None) -> None:
        """Store the calibration and refresh everything that depends on it."""
        self._calibration = cal
        self._viewport.set_calibration(cal.scale if cal else None, cal.unit if cal else "")
        self._update_area_label()
        if self._last_cursor is not None:  # refresh the physical-coord readout
            self._on_cursor_pos(*self._last_cursor)

    def _update_area_label(self) -> None:
        """Show the physical area of the current selection (blank if not calibrated)."""
        if self._calibration is not None and self._last_rect is not None:
            self._area_label.setText(
                "area " + self._calibration.format_area(self._last_rect.width(), self._last_rect.height())
            )
        else:
            self._area_label.setText("")

    def _on_clear_measurement(self) -> None:
        self._viewport.clear_measurement()
        self.statusBar().showMessage("Measurement cleared.", 3000)

    def _on_clear_calibration(self) -> None:
        self._set_calibration(None)
        self._viewport.clear_calibration()
        self.statusBar().showMessage("Calibration cleared.", 3000)

    def _on_run(self) -> None:
        """Run the search on the current selection (the Run button / manual)."""
        if self._last_rect is None:
            self.statusBar().showMessage("Draw a selection box first.", 4000)
            return
        self._controller.request(self._last_rect, self._params, self._last_mask)

    def _on_auto_run_changed(self, on: bool) -> None:
        """Track the Auto Run toggle; turning it on runs the pending selection."""
        self._auto_run = bool(on)
        if self._auto_run and self._last_rect is not None:
            self._controller.request(self._last_rect, self._params, self._last_mask)

    def _on_about(self) -> None:
        """Show the modal About dialog (author, build ID, full license)."""
        AboutDialog(self).exec()

    def _on_focus_mode_changed(self, on: bool) -> None:
        """Update the left-aligned focus-mode hint in the status bar."""
        self._focus_label.setText(
            "Focus mode — Space to exit" if on else "Press Space to enter focus mode"
        )

    def _on_cursor_pos(self, x: int, y: int) -> None:
        """Update the status-bar cursor readout (-1,-1 == outside image).

        Shows pixel coords always, plus physical coordinates (relative to the
        calibration origin) once calibrated.
        """
        self._last_cursor = None if (x < 0 or y < 0) else (x, y)
        if self._last_cursor is None:
            self._cursor_label.setText("Cursor: (-, -)")
            return
        text = f"Cursor: ({x}, {y}) px"
        if self._calibration is not None:
            text += "   " + self._calibration.format_point(x, y)
        self._cursor_label.setText(text)

    # ------------------------------------------------------- controller events
    def _on_matches_ready(self, matches: object) -> None:
        """Push results to the overlay and update the count readouts."""
        match_list = list(matches)  # type: ignore[arg-type]
        self._viewport.set_matches(match_list)
        # The viewport's client-side threshold may show fewer than len(matches);
        # prefer the viewport's post-filter count for the user-facing readout.
        try:
            shown = self._viewport.visible_match_count()
        except Exception:
            shown = len(match_list)
        # Remember the engine's full result so threshold-only changes can refresh
        # the orientation breakdown without re-running the engine.
        self._all_matches = match_list
        self._count_label.setText(self._count_text(shown, match_list, self._params.threshold))
        self._params_panel.set_match_count(shown)
        # Feed the full (pre-threshold) score distribution to the slider histogram.
        self._params_panel.set_score_histogram([m.score for m in match_list])

    def _count_text(self, shown: int, matches: list, threshold: float) -> str:
        """Build the status-bar count string, with an orientation breakdown.

        When more than one orientation is searched (or any non-R0 hit is present)
        we append a per-orientation tally of the *displayed* matches, e.g.
        ``"12 matches — R0:5 R90:3 MX:4"``. With orientation search off this stays
        the plain ``"N matches"`` it has always been.
        """
        base = f"{shown} matches"
        if not matches:
            return base
        # Tally only the matches currently above the live threshold so the
        # breakdown agrees with the shown count.
        thr = threshold
        counts: dict[str, int] = {}
        for m in matches:
            if getattr(m, "score", 0.0) < thr:
                continue
            o = getattr(m, "orientation", "R0")
            counts[o] = counts.get(o, 0) + 1
        # Only show a breakdown when something other than plain upright is in play.
        if not counts or (len(counts) == 1 and "R0" in counts):
            return base
        from .types import ORIENTATIONS

        parts = [f"{o}:{counts[o]}" for o in ORIENTATIONS if o in counts]
        return f"{base} — {' '.join(parts)}"

    def _on_busy_changed(self, busy: bool) -> None:
        """Show/hide the determinate progress bar with engine busy state."""
        self._progress.setVisible(busy)
        if busy:
            self._progress.setValue(0)

    def _on_progress(self, pct: int) -> None:
        """Forward worker progress (0..100) to the status-bar progress bar."""
        self._progress.setValue(pct)

    def _on_failed(self, message: str) -> None:
        """Surface an engine failure in the status bar (transient)."""
        self.statusBar().showMessage(f"Match failed: {message}", 8000)

    # ----------------------------------------------------------- params events
    def _on_params_changed(self, params: object) -> None:
        """Handle a params change: threshold is live-only; others re-run.

        We compare against the cached params to decide whether *only* the
        threshold moved. If so we apply it to the viewport overlay (no engine
        re-run); otherwise we adopt the new params and re-issue the last
        selection's request, if any.
        """
        # Re-apply the device preference: the panel emits device="auto", which
        # must not clobber the CLI/UI preference carried in self._params (§C.5).
        new = dataclasses.replace(params, device=self._device_pref)  # MatchParams
        old = self._params

        # Always reflect the threshold in the overlay's live filter.
        self._viewport.set_match_threshold(new.threshold)  # type: ignore[attr-defined]
        # Channel mode also drives the display: luminance -> grayscale, rgb ->
        # colour (a no-op in the viewport when unchanged).
        if new.channel_mode != old.channel_mode:
            self._viewport.set_display_grayscale(new.channel_mode == "luminance")
        # Keep the count readout in sync with the new live filter.
        try:
            shown = self._viewport.visible_match_count()
            self._count_label.setText(self._count_text(shown, self._all_matches, new.threshold))
            self._params_panel.set_match_count(shown)
        except Exception:
            pass

        # Did anything other than the threshold change? A *method* switch is
        # engine-relevant (different score map / backend), as are the feature-
        # matching knobs and the orientation-search checkboxes (which change
        # which transforms of the template are searched), so re-issue the last
        # selection on any of them.
        engine_relevant_changed = (
            new.method != old.method
            or new.scales != old.scales
            or new.max_results != old.max_results
            or new.channel_mode != old.channel_mode
            or new.compute_dtype != old.compute_dtype
            or new.nms_iou != old.nms_iou
            or new.exclude_iou != old.exclude_iou
            or new.threshold_floor != old.threshold_floor
            or new.enable_rotation != old.enable_rotation
            or new.enable_flipping != old.enable_flipping
            or new.feature_detector != old.feature_detector
            or new.feature_ratio != old.feature_ratio
            or new.feature_min_inliers != old.feature_min_inliers
            or new.feature_max_instances != old.feature_max_instances
        )

        # If the method changed, surface where it runs: conv methods use the
        # resolved engine device; feature matching always runs on CPU via OpenCV
        # (DESIGN.md §J.3), even when conv methods would use CUDA.
        if new.method != old.method:
            self._announce_method(new.method)

        self._params = new  # type: ignore[assignment]

        if engine_relevant_changed and self._last_rect is not None and self._auto_run:
            # Auto Run on: re-run the last selection with the new parameters
            # (latest-wins debounce in the controller coalesces rapid edits).
            # With Auto Run off, the new params are stored and applied on Run.
            self._controller.request(self._last_rect, self._params, self._last_mask)

    def _announce_method(self, method: str) -> None:
        """Surface, in the status bar, where the selected method runs.

        Feature matching is CPU-only via OpenCV regardless of the engine device;
        the convolution methods run on the resolved engine device. Non-blocking
        (status-bar message), never modal.
        """
        if method not in CONV_METHODS:  # i.e. "features"
            self.statusBar().showMessage(
                "Feature matching runs on the CPU via OpenCV (independent of the "
                "engine GPU device).",
                10000,
            )
        else:
            dev = device_banner_text(self._resolved_device)
            self.statusBar().showMessage(f"Method '{method}' — {dev}", 6000)

    # ------------------------------------------------------------- self-test
    def _on_self_test(self) -> None:
        """Run the built-in recall self-test and report the outcome in a dialog.

        Generates a small labelled sample, loads it, programmatically selects one
        known motif, runs a synchronous match against a throwaway engine, and
        reports how many of the *other* ground-truth instances were recovered
        within a small center tolerance (DESIGN.md §H7).
        """
        try:
            result = self._run_self_test()
        except Exception as exc:  # never let the self-test crash the app
            QMessageBox.warning(self, "Self-test error", f"Self-test could not run:\n{exc}")
            return
        QMessageBox.information(self, "Self-test", result)

    def _run_self_test(self) -> str:
        """Execute the self-test and return a human-readable summary string."""
        # Local imports: keep self-test machinery out of the hot import path.
        from .engine import Matcher
        from .loader import generate_sample, load_image

        # A small sample keeps the CPU path fast enough for an interactive test.
        with tempfile.TemporaryDirectory(prefix="fastmatch_selftest_") as tmp:
            sample_path = str(Path(tmp) / "selftest.png")
            truth = generate_sample(
                sample_path, w=1600, h=1200, tile=120, n_targets=8, seed=7
            )
            doc = load_image(sample_path)

            # Pick the first ground-truth instance as the template/source region.
            src = truth[0]
            template = doc.full[src.y : src.y + src.h, src.x : src.x + src.w]
            import numpy as np

            template = np.ascontiguousarray(template)

            # Pin the self-test engine to CPU (§C.19): this runs synchronously on
            # the GUI thread, so building it on the real device would establish a
            # CUDA context off the worker thread (the one place CUDA is allowed).
            # It is a tiny correctness check; CPU is plenty fast and keeps all
            # CUDA confined to the worker thread.
            matcher = Matcher(device="cpu")
            matcher.set_image(doc.full)
            params = MatchParams()  # defaults: threshold 0.85, floor 0.50
            found = matcher.match(
                template, params, exclude_box=(src.x, src.y, src.w, src.h)
            )

            # For every *other* ground-truth stamp, check a detection center lands
            # within a tolerance of it.
            tol = max(4, src.w // 8)
            others = truth[1:]
            recovered = 0
            for gt in others:
                gcx, gcy = gt.x + gt.w / 2.0, gt.y + gt.h / 2.0
                for m in found:
                    mcx, mcy = m.x + m.w / 2.0, m.y + m.h / 2.0
                    if abs(mcx - gcx) <= tol and abs(mcy - gcy) <= tol:
                        recovered += 1
                        break

            ok = recovered == len(others)
            verdict = "PASS" if ok else "INCOMPLETE"
            return (
                f"Self-test: {verdict}\n\n"
                f"Device: {matcher.effective_device} (self-test)\n"
                f"Ground-truth instances (excluding source): {len(others)}\n"
                f"Recovered within +/-{tol}px: {recovered}\n"
                f"Total detections returned: {len(found)}"
            )

    # ---------------------------------------------------------------- helpers
    def _maybe_show_cpu_notice(self) -> None:
        """Surface the one-time 'CPU is slow' notice, non-blocking.

        Deliberately a status-bar message rather than a modal ``QMessageBox`` —
        a modal startup dialog blocks the event loop until dismissed, which
        hangs headless/automated runs forever. The same non-blocking treatment
        applies to any startup notice.
        """
        if self._cpu_notice_shown:
            return
        self._cpu_notice_shown = True
        self.statusBar().showMessage(
            "Running on CPU: no usable CUDA device found — matching may take "
            "seconds to minutes on large images. Install the cu128 PyTorch build "
            "for GPU acceleration.",
            15000,
        )

    # ----------------------------------------------------------------- close
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        """Shut the controller's worker thread down cleanly before closing.

        If the join times out (a wedged job), the live controller is parked
        rather than dropped so its still-running QThread is never GC'd while
        running; ``thread.finished -> deleteLater`` reclaims it when the job
        finally returns (§C.3 / §C.6). We still let the window close.
        """
        try:
            old = self._controller
            if old is not None and not old.shutdown():
                self._orphaned_controllers.append(old)
        finally:
            super().closeEvent(event)


def build_main_window(
    doc: ImageDocument | None = None, *, device: str = "auto"
) -> MainWindow:
    """Construct a :class:`MainWindow` for ``doc`` (helper for __main__ and tests).

    Args:
        doc: The image document to display and search, or ``None`` to start with
            an empty canvas (no image loaded).
        device: Device preference ("auto"/"cuda"/"cpu").

    Returns:
        A ready-to-``show()`` ``MainWindow``.
    """
    return MainWindow(doc, device=device)
