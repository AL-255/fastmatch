"""The search-parameter side panel.

``ParamsPanel`` turns user-facing widgets (method selector, threshold slider,
result cap, channel mode, multi-scale toggle, feature-matching controls) into a
frozen :class:`~fastmatch.types.MatchParams` and emits it on any change. It is
*device-aware*: the full multi-scale grid is only meaningful (and affordable) on
CUDA, so on CPU the scale control is forced to single-scale and disabled,
matching ``DESIGN.md`` §H — CUDA scales ``(0.8, 0.9, 1.0, 1.1, 1.25)``, CPU
``(1.0,)``.

A **Method** selector sits at the top (``DESIGN.md`` §J.5). The convolution
methods (``ncc``/``ssd``/``ccorr``) share the GPU tiled machinery and expose a
scale-search control; ``features`` (ORB/AKAZE/SIFT, propose + appearance-verify)
is inherently scale/rotation tolerant, so it instead exposes a detector +
match-count control. Those method-specific sets are swapped with a
``QStackedWidget`` so only the relevant knobs are visible. The **channel mode**
(luminance/rgb) is common to all methods — it lives in the always-visible params
box (conv methods combine the channels before normalizing; the feature method's
appearance verification becomes colour-aware while detection stays grayscale).
The ``features`` option is disabled (greyed, with an install hint) when OpenCV is
not importable; we probe ``import cv2`` once at construction and *never*
hard-depend on it.

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
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from .types import (
    CHANNEL_MODE_LABELS,
    CHANNEL_MODES,
    CHANNEL_NAMES,
    CONV_METHODS,
    METHOD_LABELS,
    METHODS,
    MatchParams,
    active_orientations,
    normalize_weights,
)

#: Channel-weight slider resolution (ticks per slider; value/100 == the weight).
_WEIGHT_TICKS = 100

#: Labels in the parent "Match parameters" form. Their widest text sets that
#: form's label-column width; nested sub-forms (the weight groups, the feature
#: form) pin their own labels to the same width so every field in the panel
#: shares one left edge instead of stepping left at each sub-form.
_PARENT_FORM_LABELS = ("Threshold", "Max results", "Channel mode")

#: Display casing for a multi-channel mode's weight-group title (mode key -> label).
_MODE_DISPLAY = {"rgb": "RGB", "ycbcr": "YCbCr"}

# Default multi-scale grids (DESIGN.md §H). CPU stays single-scale to avoid the
# len(scales)x cost on a backend that is already minutes-slow.
_SCALES_CUDA: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.25)
_SCALES_CPU: tuple[float, ...] = (1.0,)

# The threshold slider is integer-valued in Qt; we map [0, 1000] <-> [0.0, 1.0]
# so the user gets 0.001 resolution on a [0, 1] score.
_THRESH_TICKS = 1000

# Score histogram drawn directly above the threshold slider (aligned to its track).
_HIST_HEIGHT = 38    # px tall (excludes the small gap to the slider below)
_HIST_BINS = 100     # DEFAULT score bins across [0, 1]; tunable via the View menu
# Bin-count presets offered in the View ▸ Score histogram menu (the tunable param).
_HIST_BIN_PRESETS: tuple[int, ...] = (50, 100, 200, 400)

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


class _ThresholdHistogram(QWidget):
    """A small bar chart of match-score counts, drawn to align with a horizontal
    :class:`QSlider`'s track so the bars sit directly above the slider value axis.

    Each bar is one score bin across ``[0, 1]``; its height is that bin's hit count
    (shared linear scale, tallest bin == full height). Bars at or above the current
    threshold are drawn in the accent colour (kept / shown); bars below it are muted
    (filtered out), so dragging the slider gives live feedback on how many detections
    survive. Purely a readout — it forwards no input; the slider beneath owns
    interaction. Colours come from the widget palette, so it tracks the app theme.
    """

    def __init__(self, slider: QSlider, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self._slider = slider
        self._scores: list[float] = []   # raw scores, kept so a bin change re-bins
        self._bins = _HIST_BINS
        self._counts: list[int] = []
        self._max = 0
        self.setFixedHeight(_HIST_HEIGHT)
        # Non-interactive: let clicks fall through to whatever is beneath.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setToolTip(
            "Distribution of match scores. Bars at/above the threshold are shown; "
            "bars below it are filtered out. Bin count: View ▸ Score histogram."
        )

    def set_scores(self, scores: "list[float]") -> None:
        """Store the match scores ([0, 1]), re-bin, and repaint."""
        self._scores = [min(1.0, max(0.0, float(s))) for s in scores]
        self._rebin()

    def set_bins(self, bins: int) -> None:
        """Set the number of histogram bins across [0, 1] (re-bins the stored scores)."""
        bins = max(1, int(bins))
        if bins != self._bins:
            self._bins = bins
            self._rebin()

    def bins(self) -> int:
        """Current bin count."""
        return self._bins

    def _rebin(self) -> None:
        counts = [0] * self._bins
        for s in self._scores:
            b = int(s * self._bins)
            if b >= self._bins:  # the score == 1.0 edge falls in the last bin
                b = self._bins - 1
            counts[b] += 1
        self._counts = counts
        self._max = max(counts) if counts else 0
        self.update()

    def refresh_threshold(self) -> None:
        """Repaint after the slider moved (recolours kept vs filtered bars)."""
        self.update()

    def _track_span(self) -> "tuple[float, float]":
        """Pixel x-range (value 0 .. value 1) of the slider's handle-centre travel.

        Mapping each bin edge through this makes a bar boundary line up with the
        slider handle when it sits at the same score — the alignment the chart wants.
        The slider and this widget share the grid column, so the slider's local x
        coordinates equal ours.
        """
        opt = QStyleOptionSlider()
        self._slider.initStyleOption(opt)
        st = self._slider.style()
        groove = st.subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderGroove, self._slider,
        )
        handle = st.subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderHandle, self._slider,
        )
        half = handle.width() / 2.0
        return groove.x() + half, groove.x() + groove.width() - half

    def paintEvent(self, event: object) -> None:  # noqa: N802 (Qt override)
        if not self._counts or self._max <= 0:
            return
        x0, x1 = self._track_span()
        if x1 <= x0:
            return
        span = x1 - x0
        bins = self._bins
        h = self.height()
        base = h - 1                       # leave a 1px baseline at the bottom
        top_pad = 2                        # headroom so the tallest bar isn't clipped
        usable = max(1.0, base - top_pad)

        # Theme-aware colours, both derived from the palette accent so the chart
        # reads as one distribution. Kept bars use the full accent; filtered bars are
        # the accent blended ~42% over the panel background — same hue, clearly
        # de-emphasised, yet OPAQUE so they never wash out to "gray on gray".
        pal = self.palette()
        accent = pal.color(QPalette.ColorRole.Highlight)
        win = pal.color(QPalette.ColorRole.Window)
        kept = accent
        a = 0.42
        filtered = QColor(
            round(accent.red() * a + win.red() * (1 - a)),
            round(accent.green() * a + win.green() * (1 - a)),
            round(accent.blue() * a + win.blue() * (1 - a)),
        )
        thr = self._slider.value() / float(self._slider.maximum() or 1)

        slot = span / bins
        gap = 1.0 if slot > 3.0 else 0.0   # drop the inter-bar gap once bars get thin
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(Qt.PenStyle.NoPen)
        for i, c in enumerate(self._counts):
            if c <= 0:
                continue
            bx0 = x0 + i * slot
            bw = max(1.0, slot - gap)
            bh = (c / self._max) * usable
            bin_centre = (i + 0.5) / bins
            painter.setBrush(kept if bin_centre >= thr else filtered)
            painter.drawRect(int(round(bx0)), int(round(base - bh)),
                             int(round(bw)), int(round(bh)))
        painter.end()


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

        # Width of the parent "Match parameters" form's label column (the widest
        # of its labels). Nested sub-forms pin their labels to this so all fields
        # share one left edge. Computed from the panel font so it tracks the theme.
        fm = QFontMetrics(self.font())
        self._label_col_width = max(fm.horizontalAdvance(t) for t in _PARENT_FORM_LABELS)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)  # even outer breathing room
        root.setSpacing(8)                    # consistent gap between sibling groups

        # --- Run controls (top) --------------------------------------------
        # A manual "Run" button plus an "Auto Run" toggle. Auto Run defaults ON
        # so the established behaviour (draw a box / change a setting -> search
        # runs) is preserved; turning it off lets the user set up the selection
        # and parameters and trigger a single search with Run (useful when each
        # run is expensive, e.g. CPU multi-scale).
        run_row = QHBoxLayout()
        self._run_button = QPushButton("Run", self)
        # objectName + default flag mark Run as the primary action; theme._theme_qss
        # gives it an accent on :enabled (the disabled state keeps the palette grey).
        self._run_button.setObjectName("runButton")
        self._run_button.setDefault(True)
        self._run_button.setToolTip("Run the search on the current selection.")
        self._run_button.setEnabled(False)  # enabled once a region is selected
        self._run_button.clicked.connect(lambda *_: self.run_clicked.emit())
        run_row.addWidget(self._run_button, 1)
        self._auto_run = QCheckBox("Auto run", self)
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
        method_layout.setContentsMargins(8, 4, 8, 8)
        method_layout.setSpacing(6)
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
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setContentsMargins(8, 4, 8, 8)
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(8)

        # Threshold: live [0, 1] filter, default tracks the selected method. The
        # numeric value sits right-aligned on the slider's own row (no orphan row).
        self._threshold = QSlider(Qt.Orientation.Horizontal, self)
        self._threshold.setRange(0, _THRESH_TICKS)
        self._threshold.setValue(int(round(self._default_threshold() * _THRESH_TICKS)))
        self._threshold.setToolTip(
            "Minimum match score to display. This filters live (no re-run)."
        )
        self._threshold_label = QLabel(self)
        self._threshold_label.setMinimumWidth(44)
        self._threshold_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._threshold.valueChanged.connect(self._on_threshold_changed)
        # Score histogram sits directly above the slider, in the SAME grid column so
        # it shares the slider's width/x; it aligns its bars to the slider track in
        # paintEvent. The numeric readout stays beside the slider (column 1).
        self._threshold_hist = _ThresholdHistogram(self._threshold, self)
        thresh_grid = QGridLayout()
        thresh_grid.setContentsMargins(0, 0, 0, 0)
        thresh_grid.setHorizontalSpacing(8)
        thresh_grid.setVerticalSpacing(2)
        thresh_grid.addWidget(self._threshold_hist, 0, 0)
        thresh_grid.addWidget(self._threshold, 1, 0)
        thresh_grid.addWidget(self._threshold_label, 1, 1)
        thresh_grid.setColumnStretch(0, 1)
        form.addRow(self._form_label("Threshold"), thresh_grid)

        # Max results: cap after NMS, default 500.
        self._max_results = QSpinBox(self)
        self._max_results.setRange(1, 100_000)
        self._max_results.setValue(MatchParams.max_results)
        self._max_results.setToolTip(
            "Maximum number of matches returned (the cap applied after overlapping "
            "detections are merged)."
        )
        self._max_results.valueChanged.connect(self._on_param_changed)
        form.addRow(self._form_label("Max results"), self._max_results)

        # Channel mode combo: luminance (fast, default), rgb, or ycbcr. Applies to
        # ALL methods — the conv methods combine the channels before normalizing,
        # and the feature method's appearance verification becomes colour-aware
        # (detection stays grayscale) — so it lives in the always-visible params box.
        self._channel_mode = QComboBox(self)
        # Descriptive labels (mirroring the Method combo), keyed by the mode key in
        # userData so the emitted value stays "luminance"/"rgb"/"ycbcr".
        for _key in CHANNEL_MODES:
            self._channel_mode.addItem(CHANNEL_MODE_LABELS.get(_key, _key), userData=_key)
        self._channel_mode.setToolTip(
            "Luminance: single BT.601 luma plane (faster, less VRAM). RGB / YCbCr: "
            "weighted multi-channel matching (per-channel weights below) — applies "
            "to NCC/SSD/CCORR and feature matching's appearance verification."
        )
        self._channel_mode.currentIndexChanged.connect(self._on_channel_mode_changed)
        form.addRow(self._form_label("Channel mode"), self._channel_mode)

        # Per-channel weight sliders for the multi-channel modes (one group each;
        # only the active mode's group is shown). The three sliders are relative
        # weights; the matcher normalizes them to sum to 1.0 and the readout shows
        # the normalized values. Default equal weights == unweighted multi-channel.
        self._rgb_weight_box, self._rgb_weight_sliders, self._rgb_weight_readout = (
            self._make_weight_group("rgb")
        )
        self._ycbcr_weight_box, self._ycbcr_weight_sliders, self._ycbcr_weight_readout = (
            self._make_weight_group("ycbcr")
        )
        form.addRow(self._rgb_weight_box)
        form.addRow(self._ycbcr_weight_box)

        root.addWidget(params_box)

        # --- Method-specific controls (stacked: conv vs features) ----------
        # Page 0 == convolution controls (scale grid + channel mode), Page 1 ==
        # feature controls (detector + min inliers). We swap pages on method
        # change so only the relevant knobs are visible (DESIGN.md §J.5).
        self._method_stack = QStackedWidget(self)

        # Page 0: convolution-only controls (scale search). Channel mode moved to
        # the always-visible params box above (it now applies to features too).
        conv_box = QGroupBox("Scale search", self)
        conv_box.setFlat(True)  # secondary group: title rule only, no full frame
        conv_layout = QVBoxLayout(conv_box)
        conv_layout.setContentsMargins(8, 4, 8, 4)
        conv_layout.setSpacing(6)
        self._multiscale = QCheckBox("Search multiple scales", self)
        # Always connect the signal (even though it starts disabled on CPU) so it
        # works after a runtime CPU->CUDA engine switch; enable/checked/tooltip and
        # the hint below are set by _apply_device_to_multiscale (here + set_device).
        self._multiscale.toggled.connect(self._on_param_changed)
        # Muted reason shown only when the control is disabled (CPU), so the greyed
        # checkbox is not an unexplained dead control. Reuses the secondary-text QSS.
        self._multiscale_hint = QLabel(self)
        self._multiscale_hint.setObjectName("matchCount")
        self._multiscale_hint.setWordWrap(True)
        self._apply_device_to_multiscale()
        conv_layout.addWidget(self._multiscale)
        conv_layout.addWidget(self._multiscale_hint)
        # The stacked widget sizes to the taller (feature) page; pin content to the
        # top instead of letting it float centered in the slack.
        conv_layout.addStretch(1)
        self._method_stack.addWidget(conv_box)  # index 0

        # Page 1: feature-matching controls.
        feature_box = QGroupBox("Feature matching", self)
        feature_box.setFlat(True)  # secondary group, matching Scale search / Orientation
        feature_form = QFormLayout(feature_box)
        feature_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        feature_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        feature_form.setContentsMargins(8, 4, 8, 8)
        feature_form.setVerticalSpacing(8)
        feature_form.setHorizontalSpacing(8)
        self._feature_detector = QComboBox(self)
        for det in _FEATURE_DETECTORS:
            self._feature_detector.addItem(_FEATURE_DETECTOR_LABELS[det], userData=det)
        self._select_detector(MatchParams.feature_detector)
        self._feature_detector.setToolTip(
            "Keypoint detector: ORB (fast binary, default), AKAZE, or SIFT. "
            "Feature matching runs on CPU via OpenCV."
        )
        self._feature_detector.currentIndexChanged.connect(self._on_param_changed)
        feature_form.addRow(self._form_label("Detector"), self._feature_detector)

        self._feature_min_inliers = QSpinBox(self)
        self._feature_min_inliers.setRange(4, 10_000)
        self._feature_min_inliers.setValue(MatchParams.feature_min_inliers)
        self._feature_min_inliers.setToolTip(
            "Minimum number of verified feature matches required to accept one "
            "instance. Lower finds more (and weaker) instances; higher is stricter."
        )
        self._feature_min_inliers.valueChanged.connect(self._on_param_changed)
        feature_form.addRow(self._form_label("Min matches"), self._feature_min_inliers)
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
        orient_box.setFlat(True)  # secondary group: title rule only, no full frame
        orient_layout = QVBoxLayout(orient_box)
        orient_layout.setContentsMargins(8, 4, 8, 4)
        orient_layout.setSpacing(6)
        self._enable_rotation = QCheckBox("Enable rotation", self)
        self._enable_rotation.setChecked(MatchParams.enable_rotation)
        self._enable_rotation.setToolTip(
            "Also search the template rotated 90°, 180°, and 270°."
        )
        self._enable_rotation.toggled.connect(self._on_orientation_changed)
        orient_layout.addWidget(self._enable_rotation)

        self._enable_flipping = QCheckBox("Enable flipping", self)
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
        self._count_label.setObjectName("matchCount")  # muted secondary (theme QSS)
        self._count_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._count_label)

        root.addStretch(1)

        # Initialize derived labels + the correct stack page without emitting.
        self._sync_threshold_label()
        self._sync_orientation_label()
        self._sync_method_page()
        self._sync_weight_readouts()
        self._sync_weight_visibility()
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
        # Old notation: list the D4 orientation CODES (R0, R90, MX, MY, ...).
        self._orientation_label.setText(f"{n} {plural}: {', '.join(active)}")

    def _sync_method_page(self) -> None:
        """Show the control page (conv vs features) matching the current method."""
        self._method_stack.setCurrentIndex(0 if self._is_conv_method() else 1)

    def _form_label(self, text: str) -> QLabel:
        """A right-aligned form label pinned to the parent label-column width.

        Used for the nested sub-forms (weight groups, feature form) so their
        fields share the same left edge as the parent "Match parameters" form
        instead of stepping left under their own (narrower) label column.
        """
        lbl = QLabel(text, self)
        lbl.setMinimumWidth(self._label_col_width)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    # ------------------------------------------------------- channel weights
    def _make_weight_group(self, mode: str):
        """Build a 3-slider weight group for ``mode`` ("rgb" / "ycbcr").

        Returns ``(box, sliders, readout)``. The sliders are relative weights
        (0..``_WEIGHT_TICKS``); the matcher normalizes them to sum to 1.0 and the
        readout shows the normalized values. Defaults track ``MatchParams`` (equal
        weights == unweighted multi-channel). Signals are connected AFTER the
        initial ``setValue`` so construction never emits.
        """
        names = CHANNEL_NAMES[mode]
        init = (
            MatchParams().rgb_weights if mode == "rgb" else MatchParams().ycbcr_weights
        )
        box = QGroupBox(f"{_MODE_DISPLAY.get(mode, mode.upper())} channel weights", self)
        box.setFlat(True)  # nested sub-section: avoid a frame-within-a-frame
        lay = QFormLayout(box)
        lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.setContentsMargins(8, 4, 8, 6)
        lay.setVerticalSpacing(6)
        sliders: list[QSlider] = []
        for i, nm in enumerate(names):
            s = QSlider(Qt.Orientation.Horizontal, self)
            s.setRange(0, _WEIGHT_TICKS)
            s.setValue(int(round(float(init[i]) * _WEIGHT_TICKS)))
            s.setToolTip(
                f"Relative weight of the {nm} channel "
                "(the three are normalized to sum to 1.0)."
            )
            s.valueChanged.connect(self._on_weight_changed)
            # Pin the (single-letter) label to the parent form's label-column
            # width so the slider lines up under the Channel-mode combo above
            # instead of jutting left of every other field in the box.
            lay.addRow(self._form_label(nm), s)
            sliders.append(s)
        readout = QLabel(self)
        readout.setTextFormat(Qt.TextFormat.PlainText)
        lay.addRow(readout)
        return box, sliders, readout

    def _weights_for(self, mode: str) -> tuple[float, float, float]:
        """Normalized weights from the ``mode`` group's sliders (sum to 1.0)."""
        sliders = self._rgb_weight_sliders if mode == "rgb" else self._ycbcr_weight_sliders
        return normalize_weights([s.value() for s in sliders], 3)  # type: ignore[return-value]

    def _sync_weight_readouts(self) -> None:
        """Refresh both groups' normalized-weight readout labels."""
        for mode, readout in (
            ("rgb", self._rgb_weight_readout),
            ("ycbcr", self._ycbcr_weight_readout),
        ):
            w = self._weights_for(mode)
            names = CHANNEL_NAMES[mode]
            readout.setText(
                " ".join(f"{n} {v:.2f}" for n, v in zip(names, w)) + " (sum 1.00)"
            )

    def _sync_weight_visibility(self) -> None:
        """Show only the active multi-channel mode's weight group (none for luma)."""
        mode = self._channel_mode.currentData()
        self._rgb_weight_box.setVisible(mode == "rgb")
        self._ycbcr_weight_box.setVisible(mode == "ycbcr")

    def _on_channel_mode_changed(self, *args: object) -> None:
        """Channel mode switched: show the right weight group, then emit."""
        self._sync_weight_visibility()
        self._on_param_changed()

    def _on_weight_changed(self, *args: object) -> None:
        """A weight slider moved: refresh the readout, then emit (re-runs)."""
        self._sync_weight_readouts()
        self._on_param_changed()

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
        """Threshold moved: update its label + histogram, then emit (live filter)."""
        self._sync_threshold_label()
        self._threshold_hist.refresh_threshold()
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
            channel_mode=self._channel_mode.currentData(),
            rgb_weights=self._weights_for("rgb"),
            ycbcr_weights=self._weights_for("ycbcr"),
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

    def set_score_histogram(self, scores: "list[float]") -> None:
        """Feed the per-detection scores ([0, 1]) for the hit-count histogram drawn
        above the threshold slider. Pass an empty list to clear it (e.g. on a new
        image or cleared matches)."""
        self._threshold_hist.set_scores(list(scores))

    def set_histogram_bins(self, bins: int) -> None:
        """Set the threshold-histogram bin count (the View-menu tunable)."""
        self._threshold_hist.set_bins(bins)

    def auto_run_enabled(self) -> bool:
        """Whether the **Auto Run** checkbox is currently checked."""
        return self._auto_run.isChecked()

    def set_run_enabled(self, enabled: bool) -> None:
        """Enable/disable the **Run** button (e.g. only once a region exists)."""
        self._run_button.setEnabled(bool(enabled))

    def _apply_device_to_multiscale(self) -> None:
        """Set the multi-scale control for the current device (no signal emit).

        CUDA: enabled + checked (the full grid is affordable on the GPU). CPU:
        forced single-scale and disabled (the grid is unaffordable). Blocks the
        toggled signal so this programmatic update never emits params_changed.
        """
        blocked = self._multiscale.blockSignals(True)
        try:
            if self._is_cuda:
                self._multiscale.setEnabled(True)
                self._multiscale.setChecked(True)
                self._multiscale.setToolTip(
                    "Search the scale grid "
                    f"{', '.join(f'{s:g}×' for s in _SCALES_CUDA)} (GPU only)."
                )
                self._multiscale_hint.setVisible(False)
            else:
                self._multiscale.setChecked(False)
                self._multiscale.setEnabled(False)
                self._multiscale.setToolTip(
                    "Multi-scale search is GPU-only; CPU is locked to 1.0× to keep "
                    "queries responsive."
                )
                self._multiscale_hint.setText("GPU only — CPU is locked to 1.0×.")
                self._multiscale_hint.setVisible(True)
        finally:
            self._multiscale.blockSignals(blocked)

    def set_device(self, effective_device: str) -> None:
        """Re-gate device-dependent controls after a runtime engine switch.

        Updates whether the multi-scale grid is offered (CUDA) or locked off
        (CPU). Does not emit ``params_changed`` — the caller re-reads
        :meth:`current_params` after switching the engine.
        """
        self._device = str(effective_device).lower()
        new_is_cuda = self._device.startswith("cuda")
        if new_is_cuda == self._is_cuda:
            return
        self._is_cuda = new_is_cuda
        self._apply_device_to_multiscale()
