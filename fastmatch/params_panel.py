"""The search-parameter side panel.

``ParamsPanel`` turns user-facing widgets (method selector, threshold slider,
result cap, channel mode, multi-scale toggle, feature-matching controls) into a
frozen :class:`~fastmatch.types.MatchParams` and emits it on any change. It is
*device-aware*: the full multi-scale grid is only meaningful (and affordable) on
CUDA, so on CPU the scale control is forced to single-scale and disabled,
matching ``DESIGN.md`` §H — CUDA scales ``(0.8, 0.9, 1.0, 1.1, 1.25)``, CPU
``(1.0,)``.

A **Method** selector sits at the top (``DESIGN.md`` §J.5). The convolution
methods (``ncc``/``ssd``/``ccorr``) share the GPU tiled machinery and expose the
scale + channel controls; ``features`` (ORB/AKAZE/SIFT + RANSAC homography) is
inherently scale/rotation tolerant and grayscale, so it instead exposes a
detector + minimum-inliers control. The two control sets are swapped with a
``QStackedWidget`` so only the relevant knobs are visible. The ``features``
option is disabled (greyed, with an install hint) when OpenCV is not importable;
we probe ``import cv2`` once at construction and *never* hard-depend on it.

Threshold is special: it is a *live UI filter* (the engine returns everything at
or above ``threshold_floor`` and the overlay filters up to ``threshold`` with no
re-run), so the main window can route threshold changes straight to the viewport
while routing every other change to a fresh engine request. We still emit the
full ``MatchParams`` on a threshold change so a single signal carries everything.
The threshold *default* tracks the method (conv methods favour a strict ``0.85``;
feature matching's inlier-ratio score is softer, so it defaults to ``0.5``), but
we only nudge the default on a method switch and otherwise respect manual edits.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .types import (
    CONV_METHODS,
    METHOD_LABELS,
    METHODS,
    MatchParams,
    active_orientations,
)

# Default multi-scale grids (DESIGN.md §H). CPU stays single-scale to avoid the
# len(scales)x cost on a backend that is already minutes-slow.
_SCALES_CUDA: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.25)
_SCALES_CPU: tuple[float, ...] = (1.0,)

# The threshold slider is integer-valued in Qt; we map [0, 1000] <-> [0.0, 1.0]
# so the user gets 0.001 resolution on a [0, 1] score.
_THRESH_TICKS = 1000

# Per-method default for the threshold *slider* (DESIGN.md §J.5). Conv methods
# want a strict cutoff against repetitive-texture floods; the feature path scores
# by inlier ratio and so reads lower for genuine instances, hence a softer 0.5.
_THRESH_DEFAULT_CONV = 0.85
_THRESH_DEFAULT_FEATURES = 0.50

# Feature detectors offered, in UI order (DESIGN.md §J.3). Keys match
# ``MatchParams.feature_detector``.
_FEATURE_DETECTORS: tuple[str, ...] = ("orb", "akaze", "sift")
_FEATURE_DETECTOR_LABELS: dict[str, str] = {
    "orb": "ORB (fast, default)",
    "akaze": "AKAZE",
    "sift": "SIFT",
}


def _cv2_available() -> bool:
    """Return whether OpenCV (``cv2``) can be imported.

    Probed once at panel construction so the "features" method can be greyed out
    with an install hint when the optional backend is absent. We never keep a
    reference to the module — the convolution methods must not pull cv2 in.
    """
    try:
        import cv2  # noqa: F401  (probe only)
    except Exception:
        # Any import failure (missing wheel, broken install) means no features.
        return False
    return True


class ParamsPanel(QWidget):
    """Search-parameter editor that emits :class:`MatchParams` on change.

    Signals:
        params_changed: Emitted with the current ``MatchParams`` whenever any
            control changes (including the live threshold slider).
        run_clicked: Emitted when the user presses the **Run** button (manually
            trigger the search on the current selection).
        auto_run_changed: Emitted with the new state of the **Auto Run** checkbox.
            When Auto Run is on, the main window re-issues the search on a new
            selection or an engine-relevant settings change; when off, it waits
            for Run.
    """

    params_changed = Signal(object)  # MatchParams
    run_clicked = Signal()
    auto_run_changed = Signal(bool)

    def __init__(self, effective_device: str, parent: QWidget | None = None) -> None:
        """Build the panel for a resolved device.

        Args:
            effective_device: The device the engine actually resolved to, one of
                ``"cuda"`` / ``"cpu"`` (case-insensitive). Drives whether the
                multi-scale control is offered.
            parent: Optional Qt parent.
        """
        super().__init__(parent)
        # Normalize: a torch.device prints as e.g. "cuda:0"; accept that too.
        self._device = str(effective_device).lower()
        self._is_cuda = self._device.startswith("cuda")

        # Whether the optional OpenCV backend is present (gates the "features"
        # method item). Probed exactly once, here, at construction.
        self._features_available = _cv2_available()

        # Guard so programmatic widget updates (e.g. forcing CPU single-scale or
        # nudging the threshold default on a method switch) don't recursively
        # re-emit while we are mid-construction or mid-sync.
        self._emitting = False

        root = QVBoxLayout(self)

        # --- Run controls (top) --------------------------------------------
        # A manual "Run" button plus an "Auto Run" toggle. Auto Run defaults ON
        # so the established behaviour (draw a box / change a setting -> search
        # runs) is preserved; turning it off lets the user set up the selection
        # and parameters and trigger a single search with Run (useful when each
        # run is expensive, e.g. CPU multi-scale).
        run_row = QHBoxLayout()
        self._run_button = QPushButton("Run", self)
        self._run_button.setToolTip("Run the search on the current selection.")
        self._run_button.setEnabled(False)  # enabled once a region is selected
        self._run_button.clicked.connect(lambda *_: self.run_clicked.emit())
        run_row.addWidget(self._run_button, 1)
        self._auto_run = QCheckBox("Auto Run", self)
        self._auto_run.setChecked(True)
        self._auto_run.setToolTip(
            "When on, the search runs automatically as you draw a selection or "
            "change settings. When off, click Run to search."
        )
        self._auto_run.toggled.connect(self.auto_run_changed)
        run_row.addWidget(self._auto_run)
        root.addLayout(run_row)

        # --- Method selector (top, DESIGN.md §J.5) -------------------------
        method_box = QGroupBox("Method", self)
        method_layout = QVBoxLayout(method_box)
        self._method = QComboBox(self)
        for key in METHODS:
            # Store the method key as item data; show the human-readable label.
            self._method.addItem(METHOD_LABELS.get(key, key), userData=key)
            idx = self._method.count() - 1
            self._method.setItemData(idx, METHOD_LABELS.get(key, key), Qt.ItemDataRole.ToolTipRole)
            if key == "features" and not self._features_available:
                # Grey out + explain how to enable it, without hard-depending on cv2.
                self._set_combo_item_enabled(self._method, idx, False)
                self._method.setItemData(
                    idx,
                    "Install opencv-python-headless",
                    Qt.ItemDataRole.ToolTipRole,
                )
        # Default to the MatchParams default method ("ncc"); fall back to the
        # first enabled item if that key is somehow absent.
        self._select_method(MatchParams.method)
        self._method.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self._method)
        root.addWidget(method_box)

        # --- Match parameters group ----------------------------------------
        params_box = QGroupBox("Match parameters", self)
        form = QFormLayout(params_box)

        # Threshold: live [0, 1] filter, default tracks the selected method.
        self._threshold = QSlider(Qt.Orientation.Horizontal, self)
        self._threshold.setRange(0, _THRESH_TICKS)
        self._threshold.setValue(int(round(self._default_threshold() * _THRESH_TICKS)))
        self._threshold.setToolTip(
            "Minimum match score to display. This filters live (no re-run)."
        )
        self._threshold_label = QLabel(self)
        self._threshold.valueChanged.connect(self._on_threshold_changed)
        form.addRow("Threshold", self._threshold)
        form.addRow("", self._threshold_label)

        # Max results: cap after NMS, default 500.
        self._max_results = QSpinBox(self)
        self._max_results.setRange(1, 100_000)
        self._max_results.setValue(MatchParams.max_results)
        self._max_results.setToolTip("Maximum number of matches returned (cap after NMS).")
        self._max_results.valueChanged.connect(self._on_param_changed)
        form.addRow("Max results", self._max_results)

        # Channel mode combo: luminance (fast, default) or rgb. Conv-only; the
        # feature path is grayscale, so this lives in the conv control page.
        self._channel_mode = QComboBox(self)
        self._channel_mode.addItems(["luminance", "rgb"])
        self._channel_mode.setToolTip(
            "luminance: single-channel NCC (3x less VRAM). rgb: combine all three "
            "channels before normalizing."
        )
        self._channel_mode.currentIndexChanged.connect(self._on_param_changed)

        root.addWidget(params_box)

        # --- Method-specific controls (stacked: conv vs features) ----------
        # Page 0 == convolution controls (scale grid + channel mode), Page 1 ==
        # feature controls (detector + min inliers). We swap pages on method
        # change so only the relevant knobs are visible (DESIGN.md §J.5).
        self._method_stack = QStackedWidget(self)

        # Page 0: convolution-only controls.
        conv_box = QGroupBox("Scale search", self)
        conv_layout = QVBoxLayout(conv_box)
        conv_form = QFormLayout()
        conv_form.addRow("Channel mode", self._channel_mode)
        conv_layout.addLayout(conv_form)
        self._multiscale = QCheckBox("Search multiple scales", self)
        if self._is_cuda:
            self._multiscale.setChecked(True)
            self._multiscale.setToolTip(
                "Search the scale grid "
                f"{', '.join(f'{s:g}x' for s in _SCALES_CUDA)} (GPU only)."
            )
            self._multiscale.toggled.connect(self._on_param_changed)
        else:
            # On CPU the full grid is unaffordable; force and lock single-scale.
            self._multiscale.setChecked(False)
            self._multiscale.setEnabled(False)
            self._multiscale.setToolTip(
                "Multi-scale search is GPU-only; CPU is locked to 1.0x to keep "
                "queries responsive."
            )
        conv_layout.addWidget(self._multiscale)
        self._method_stack.addWidget(conv_box)  # index 0

        # Page 1: feature-matching controls.
        feature_box = QGroupBox("Feature matching", self)
        feature_form = QFormLayout(feature_box)
        self._feature_detector = QComboBox(self)
        for det in _FEATURE_DETECTORS:
            self._feature_detector.addItem(_FEATURE_DETECTOR_LABELS[det], userData=det)
        self._select_detector(MatchParams.feature_detector)
        self._feature_detector.setToolTip(
            "Keypoint detector: ORB (fast binary, default), AKAZE, or SIFT. "
            "Feature matching runs on CPU via OpenCV."
        )
        self._feature_detector.currentIndexChanged.connect(self._on_param_changed)
        feature_form.addRow("Detector", self._feature_detector)

        self._feature_min_inliers = QSpinBox(self)
        self._feature_min_inliers.setRange(4, 10_000)
        self._feature_min_inliers.setValue(MatchParams.feature_min_inliers)
        self._feature_min_inliers.setToolTip(
            "Minimum RANSAC inliers required to accept one instance. Lower finds "
            "more (and weaker) instances; higher is stricter."
        )
        self._feature_min_inliers.valueChanged.connect(self._on_param_changed)
        feature_form.addRow("Min inliers", self._feature_min_inliers)
        self._method_stack.addWidget(feature_box)  # index 1

        root.addWidget(self._method_stack)

        # --- Orientation (dihedral D4) search ------------------------------
        # Applies to ALL methods, so it lives *outside* the method-specific
        # stack as a common group (DESIGN.md / types.active_orientations). Two
        # independent bits: rotation adds the 90/180/270° turns; flipping adds
        # the mirrors; both together also add the diagonal reflections. Both
        # default off, so with neither checked the search is the upright "R0"
        # template only — byte-for-byte the prior behaviour.
        orient_box = QGroupBox("Orientation", self)
        orient_layout = QVBoxLayout(orient_box)
        self._enable_rotation = QCheckBox("Enable Rotation", self)
        self._enable_rotation.setChecked(MatchParams.enable_rotation)
        self._enable_rotation.setToolTip(
            "Also search the template rotated 90°, 180°, and 270°."
        )
        self._enable_rotation.toggled.connect(self._on_orientation_changed)
        orient_layout.addWidget(self._enable_rotation)

        self._enable_flipping = QCheckBox("Enable Flipping", self)
        self._enable_flipping.setChecked(MatchParams.enable_flipping)
        self._enable_flipping.setToolTip(
            "Also search mirrored templates. Combined with rotation this adds the "
            "diagonal reflections too (8 orientations total)."
        )
        self._enable_flipping.toggled.connect(self._on_orientation_changed)
        orient_layout.addWidget(self._enable_flipping)

        # Tiny readout of how many orientations the two checkboxes select.
        self._orientation_label = QLabel(self)
        self._orientation_label.setTextFormat(Qt.TextFormat.PlainText)
        orient_layout.addWidget(self._orientation_label)
        root.addWidget(orient_box)

        # --- Match-count readout -------------------------------------------
        self._count_label = QLabel(self)
        self._count_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._count_label)

        root.addStretch(1)

        # Initialize derived labels + the correct stack page without emitting.
        self._sync_threshold_label()
        self._sync_orientation_label()
        self._sync_method_page()
        self.set_match_count(0)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _set_combo_item_enabled(combo: QComboBox, index: int, enabled: bool) -> None:
        """Enable/disable a single combo item (greys it and blocks selection).

        Qt has no direct per-item enable API on ``QComboBox``; we toggle the
        item's ``ItemIsEnabled`` flag on its model row, which both greys the
        entry and prevents the user from choosing it.
        """
        model = combo.model()
        item = model.index(index, 0)
        if enabled:
            flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        else:
            flags = Qt.ItemFlag.NoItemFlags
        model.setData(item, int(flags.value), Qt.ItemDataRole.UserRole - 1)

    def _select_method(self, key: str) -> None:
        """Select the combo item whose stored key is ``key`` (else first item)."""
        idx = self._method.findData(key)
        self._method.setCurrentIndex(idx if idx >= 0 else 0)

    def _select_detector(self, key: str) -> None:
        """Select the feature-detector item whose stored key is ``key``."""
        idx = self._feature_detector.findData(key)
        self._feature_detector.setCurrentIndex(idx if idx >= 0 else 0)

    def _current_method(self) -> str:
        """Return the method key currently selected in the Method combo."""
        data = self._method.currentData()
        return str(data) if data is not None else MatchParams.method

    def _is_conv_method(self) -> bool:
        """Whether the current method uses the GPU tiled-convolution machinery."""
        return self._current_method() in CONV_METHODS

    def _default_threshold(self) -> float:
        """The threshold-slider default appropriate to the current method."""
        return _THRESH_DEFAULT_CONV if self._is_conv_method() else _THRESH_DEFAULT_FEATURES

    def _scales(self) -> tuple[float, ...]:
        """The active scale grid given device + the multi-scale checkbox.

        Feature matching ignores scales entirely (it is scale-tolerant), so we
        report the single-scale default there to keep the params object tidy.
        """
        if not self._is_conv_method():
            return _SCALES_CPU
        if self._is_cuda and self._multiscale.isChecked():
            return _SCALES_CUDA
        return _SCALES_CPU

    def _sync_threshold_label(self) -> None:
        """Refresh the numeric readout beside the threshold slider."""
        self._threshold_label.setText(f"{self._threshold.value() / _THRESH_TICKS:.3f}")

    def _sync_orientation_label(self) -> None:
        """Refresh the readout of how many orientations the checkboxes select."""
        active = active_orientations(
            self._enable_rotation.isChecked(), self._enable_flipping.isChecked()
        )
        n = len(active)
        plural = "orientation" if n == 1 else "orientations"
        self._orientation_label.setText(f"{n} {plural}: {', '.join(active)}")

    def _sync_method_page(self) -> None:
        """Show the control page (conv vs features) matching the current method."""
        self._method_stack.setCurrentIndex(0 if self._is_conv_method() else 1)

    # ------------------------------------------------------------------- slots
    def _on_method_changed(self, *args: object) -> None:
        """Method switched: swap controls, nudge the threshold default, emit.

        We set the threshold *default* for the new method, but only as a default:
        Qt's ``setValue`` is a programmatic move guarded by ``_emitting`` (set
        inside ``_emit``), so it does not double-fire; later manual slider edits
        are untouched. The single ``_emit`` at the end carries the new method
        plus the (possibly re-defaulted) threshold in one signal.
        """
        self._sync_method_page()
        # Nudge the threshold to the method's default. Block the slider's own
        # signal so this programmatic move does not emit twice; the trailing
        # _emit() reflects the final state.
        target = int(round(self._default_threshold() * _THRESH_TICKS))
        blocked = self._threshold.blockSignals(True)
        try:
            self._threshold.setValue(target)
        finally:
            self._threshold.blockSignals(blocked)
        self._sync_threshold_label()
        self._emit()

    def _on_threshold_changed(self) -> None:
        """Threshold moved: update its label, then emit (live filter path)."""
        self._sync_threshold_label()
        self._emit()

    def _on_param_changed(self, *args: object) -> None:
        """A non-threshold control changed: emit a fresh params object."""
        self._emit()

    def _on_orientation_changed(self, *args: object) -> None:
        """An orientation checkbox toggled: refresh the readout, then emit.

        Like the other non-threshold controls this is engine-relevant (it changes
        which transforms of the template are searched), so the trailing _emit
        carries the new enable_rotation/enable_flipping for the main window to
        re-issue the search.
        """
        self._sync_orientation_label()
        self._emit()

    def _emit(self) -> None:
        """Build and emit the current ``MatchParams`` unless suppressed."""
        if self._emitting:
            return
        self._emitting = True
        try:
            self.params_changed.emit(self.current_params())
        finally:
            self._emitting = False

    # ------------------------------------------------------------------ public
    def current_params(self) -> MatchParams:
        """Return the parameters currently described by the widgets.

        ``device`` is left as ``"auto"`` — the controller already owns a Matcher
        bound to the resolved device, so the panel does not re-specify it.
        ``threshold_floor`` keeps its default so the engine always returns a
        super-set the live threshold slider can filter within. The feature-only
        fields are always populated (they are simply ignored by the conv path),
        so the emitted object fully reflects the panel regardless of method.
        """
        return MatchParams(
            threshold=self._threshold.value() / _THRESH_TICKS,
            threshold_floor=MatchParams.threshold_floor,
            scales=self._scales(),
            rotations=None,
            max_results=self._max_results.value(),
            nms_iou=MatchParams.nms_iou,
            exclude_iou=MatchParams.exclude_iou,
            device="auto",
            compute_dtype=MatchParams.compute_dtype,
            channel_mode=self._channel_mode.currentText(),
            method=self._current_method(),
            enable_rotation=self._enable_rotation.isChecked(),
            enable_flipping=self._enable_flipping.isChecked(),
            feature_detector=str(self._feature_detector.currentData()),
            feature_ratio=MatchParams.feature_ratio,
            feature_min_inliers=self._feature_min_inliers.value(),
            feature_max_instances=MatchParams.feature_max_instances,
        )

    def set_match_count(self, n: int) -> None:
        """Update the match-count readout label."""
        self._count_label.setText(f"Matches shown: {n}")

    def auto_run_enabled(self) -> bool:
        """Whether the **Auto Run** checkbox is currently checked."""
        return self._auto_run.isChecked()

    def set_run_enabled(self, enabled: bool) -> None:
        """Enable/disable the **Run** button (e.g. only once a region exists)."""
        self._run_button.setEnabled(bool(enabled))
