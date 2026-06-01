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

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .device import device_banner_text, resolve_device
from .document import ImageDocument
from .memory import MemoryEntry
from .memory_panel import MemoryPanel
from .params_panel import ParamsPanel
from .types import CONV_METHODS, MatchParams


class MainWindow(QMainWindow):
    """Top-level window wiring viewport, controller, and params panel."""

    def __init__(self, doc: ImageDocument, *, device: str = "auto") -> None:
        """Construct the window around a loaded document.

        Args:
            doc: The image to display and search.
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
        self._viewport.set_image(doc)
        self.setCentralWidget(self._viewport)

        # --- Controller (owns worker thread) -------------------------------
        self._controller = MatchController(doc, self._params)

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

        dock_body = QWidget(self)
        dock_layout = QVBoxLayout(dock_body)
        self._banner = QLabel(device_banner_text(self._resolved_device), dock_body)
        self._banner.setWordWrap(True)
        self._banner.setObjectName("deviceBanner")
        dock_layout.addWidget(self._banner)
        dock_layout.addWidget(self._params_panel)
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
        self._memory.set_source(self._doc.path, self._viewport.image_size())
        self._memory_dock = QDockWidget("Memory", self)
        self._memory_dock.setWidget(self._memory)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._memory_dock)

        # --- Toolbar + menu bar -------------------------------------------
        self._build_toolbar()
        self._build_menus()

        # --- Status bar ----------------------------------------------------
        self._cursor_label = QLabel("(-, -)", self)
        self._count_label = QLabel("0 matches", self)
        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setMaximumWidth(180)
        self._progress.setVisible(False)
        sb = self.statusBar()
        sb.addPermanentWidget(self._cursor_label)
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

        # Fit the freshly loaded image into the view.
        self._viewport.fit_in_view()

    # ------------------------------------------------------------------ build
    def _build_toolbar(self) -> None:
        """Create the toolbar actions (Open, mode toggle, Fit, Clear, Self-test)."""
        tb = QToolBar("Main", self)
        tb.setObjectName("mainToolbar")
        self.addToolBar(tb)

        self._act_open = QAction("Open", self)
        self._act_open.triggered.connect(self._on_open)
        tb.addAction(self._act_open)

        tb.addSeparator()

        # Mode toggle: checkable; unchecked == Select, checked == Pan.
        self._act_mode = QAction("Pan mode", self)
        self._act_mode.setCheckable(True)
        self._act_mode.setToolTip("Toggle between Select (draw region) and Pan (drag to move).")
        self._act_mode.toggled.connect(self._on_mode_toggled)
        tb.addAction(self._act_mode)

        self._act_fit = QAction("Fit", self)
        self._act_fit.setShortcut(QKeySequence("F"))  # F = zoom-to-fit
        self._act_fit.setToolTip("Zoom to fit the whole image (F)")
        self._act_fit.triggered.connect(self._viewport.fit_in_view)
        tb.addAction(self._act_fit)

        self._act_clear = QAction("Clear matches", self)
        self._act_clear.triggered.connect(self._on_clear_matches)
        tb.addAction(self._act_clear)

        self._act_add_memory = QAction("Add to Memory", self)
        self._act_add_memory.setToolTip(
            "Save the current selection and its matches as a Memory entry."
        )
        self._act_add_memory.triggered.connect(self._on_add_to_memory)
        tb.addAction(self._act_add_memory)

        tb.addSeparator()

        self._act_selftest = QAction("Self-test", self)
        self._act_selftest.setToolTip(
            "Generate a labelled sample, search one motif, and report recall."
        )
        self._act_selftest.triggered.connect(self._on_self_test)
        tb.addAction(self._act_selftest)

    def _build_menus(self) -> None:
        """Menu bar — overlay box appearance lives under 'View > Match boxes'."""
        view_menu = self.menuBar().addMenu("&View")
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

        # XOR-with-background toggle.
        self._act_box_xor = QAction("&XOR with background", self, checkable=True)
        self._act_box_xor.setToolTip(
            "Draw box outlines XORed with the pixels underneath, so they stay "
            "visible on any background."
        )
        self._act_box_xor.toggled.connect(self._viewport.set_box_xor)
        boxes_menu.addAction(self._act_box_xor)

    def _connect_signals(self) -> None:
        """Connect viewport/controller/panel signals per DESIGN.md §F.2."""
        # Selection -> controller request.
        self._viewport.regionSelected.connect(self._on_region_selected)
        # Live cursor readout.
        self._viewport.cursorImagePos.connect(self._on_cursor_pos)

        # Controller results / state.
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

    def _swap_document(self, doc: ImageDocument) -> None:
        """Replace the current document, controller, and viewport image.

        Tears the old controller's worker thread down *before* building a new
        one so two CUDA contexts / worker threads never live at once. If that
        join times out (a wedged job), we MUST NOT drop the live thread (it
        would be GC'd while running -> SIGABRT) nor build a second engine: park
        the old controller and bail with a warning. Its
        ``thread.finished -> deleteLater`` reclaims it once the job returns
        (§C.3 / §C.6).
        """
        from .controller import MatchController

        old = self._controller
        if not old.shutdown():
            # Still running after the bounded join: park it (alive) and abort the
            # swap. Do not build a new engine; do not drop the running thread.
            self._orphaned_controllers.append(old)
            QMessageBox.warning(
                self,
                "Busy",
                "A search is still finishing; please wait and try again.",
            )
            return
        # Shutdown succeeded: the old controller's thread is down and its engine
        # refs are dropped. It is safe to reclaim it and build the new one.
        old.deleteLater()

        self._doc = doc
        self._last_rect = None
        self._all_matches = []
        self._params_panel.set_run_enabled(False)  # new image -> no selection yet
        self._viewport.clear_matches()
        self._viewport.clear_template()
        self._viewport.set_image(doc)
        # Keep the display mode (grayscale/colour) consistent for the new image.
        self._viewport.set_display_grayscale(self._params.channel_mode == "luminance")
        self._controller = MatchController(doc, self._params)
        # Reconnect the new controller's signals.
        self._controller.matches_ready.connect(self._on_matches_ready)
        self._controller.busy_changed.connect(self._on_busy_changed)
        self._controller.progress.connect(self._on_progress)
        self._controller.failed.connect(self._on_failed)
        # Reset the busy UI for the fresh controller: the old controller may have
        # been mid-search, leaving the progress bar visible (§C.17).
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._viewport.fit_in_view()
        self._count_label.setText("0 matches")
        self._params_panel.set_match_count(0)
        # Re-seed the Memory panel's source header so entries added after the
        # swap record the new image (existing entries are left untouched).
        self._memory.set_source(self._doc.path, self._viewport.image_size())

    def _on_mode_toggled(self, checked: bool) -> None:
        """Switch the viewport between Pan (checked) and Select (unchecked)."""
        from .viewport import ImageViewport

        mode = ImageViewport.Mode.PAN if checked else ImageViewport.Mode.SELECT
        self._viewport.set_mode(mode)
        self._act_mode.setText("Pan mode" if checked else "Select mode")

    def _on_clear_matches(self) -> None:
        """Clear overlay matches and the current template selection."""
        self._viewport.clear_matches()
        self._viewport.clear_template()
        self._last_rect = None
        self._all_matches = []
        self._params_panel.set_run_enabled(False)  # nothing to Run anymore
        self._count_label.setText("0 matches")
        self._params_panel.set_match_count(0)

    # ----------------------------------------------------------------- memory
    def _on_add_to_memory(self) -> None:
        """Append the current selection + visible matches as a Memory entry.

        The matches stored are the ones currently *above* the live threshold
        (what the user sees in the overlay), captured from the engine's full
        result set in :attr:`_all_matches`. A no-op (with a status hint) if there
        is no completed search to save.
        """
        if self._last_rect is None or not self._all_matches:
            self.statusBar().showMessage("Run a match first to add to memory.", 6000)
            return
        visible = [m for m in self._all_matches if m.score >= self._params.threshold]
        rect = self._last_rect
        sel = (rect.x(), rect.y(), rect.width(), rect.height())
        # Record the COMPLETE settings (channel mode, orientation flags, scales,
        # NMS/feature params, ...) via the full MatchParams snapshot.
        entry = MemoryEntry(selection=sel, params=self._params, matches=list(visible))
        self._memory.add_entry(entry)
        self.statusBar().showMessage(f"Added {len(visible)} matches to memory.", 6000)

    def _on_revisit_entry(self, entry: object) -> None:
        """Show a saved entry's boxes on the viewport (revisit).

        Pushes the entry's recorded matches straight onto the overlay and drops
        the live threshold to 0 so every saved box shows regardless of the
        current slider. Revisit only makes sense for the current image; if the
        entry was recorded against a different image the boxes will not line up,
        which is the user's call.
        """
        matches = list(entry.matches)  # type: ignore[attr-defined]
        self._viewport.set_matches(matches)
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
        self._params_panel.set_run_enabled(True)  # there is now something to Run
        if self._auto_run:
            self._controller.request(rect, self._params)
        else:
            self.statusBar().showMessage("Selection set — press Run to search.", 4000)

    def _on_run(self) -> None:
        """Run the search on the current selection (the Run button / manual)."""
        if self._last_rect is None:
            self.statusBar().showMessage("Draw a selection box first.", 4000)
            return
        self._controller.request(self._last_rect, self._params)

    def _on_auto_run_changed(self, on: bool) -> None:
        """Track the Auto Run toggle; turning it on runs the pending selection."""
        self._auto_run = bool(on)
        if self._auto_run and self._last_rect is not None:
            self._controller.request(self._last_rect, self._params)

    def _on_cursor_pos(self, x: int, y: int) -> None:
        """Update the status-bar cursor readout (-1,-1 == outside image)."""
        if x < 0 or y < 0:
            self._cursor_label.setText("(-, -)")
        else:
            self._cursor_label.setText(f"({x}, {y})")

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
            self._controller.request(self._last_rect, self._params)

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
            if not old.shutdown():
                self._orphaned_controllers.append(old)
        finally:
            super().closeEvent(event)


def build_main_window(doc: ImageDocument, *, device: str = "auto") -> MainWindow:
    """Construct a :class:`MainWindow` for ``doc`` (helper for __main__ and tests).

    Args:
        doc: The image document to display and search.
        device: Device preference ("auto"/"cuda"/"cpu").

    Returns:
        A ready-to-``show()`` ``MainWindow``.
    """
    return MainWindow(doc, device=device)
