# GUI self-improvement log

An automatic critique→improve loop for the desktop GUI. Each iteration renders the
PySide6 UI headlessly (`scripts/gui_shots.py`, offscreen Qt → PNGs), runs a
multi-lens critique (parallel design-lens agents + a synthesis judge), applies a
small batch of low-risk fixes, and verifies the full test suite stays green plus
the change is visible in a re-screenshot. Screenshots live under
`scripts/gui_shots_out/` (gitignored; regenerate on demand).

## Iteration 1 — labels, hierarchy, form polish

Correctness / copy:
- Method dropdown: "Feature matching — ORB + homography (rotated / scaled / warped)"
  → "… ORB keypoints + appearance verify (rotated / scaled)" (the matcher was
  rewritten away from homography). `types.py`
- Feature panel "Min inliers" → "Min matches"; tooltip de-RANSAC'd. `params_panel.py`
- Weight-group title casing: "YCBCR channel weights" → "YCbCr channel weights". `params_panel.py`
- Channel-mode combo now shows descriptive labels ("luminance (single BT.601 luma
  plane)", etc.) keyed by `userData`, mirroring the Method combo; emitted key
  unchanged. `params_panel.py`

Hierarchy / layout:
- Device/engine banner moved from the TOP of the Search dock to the bottom, so
  Run / Method / Match parameters lead the panel (the hint is also in the status
  bar). `app.py`
- Threshold value moved onto the slider's row (right-aligned), removing the orphan
  form row. `params_panel.py`
- Channel-weight readout spans the group full-width with single spacing. `params_panel.py`
- Match-parameters and Feature-matching forms: right-aligned label column +
  `AllNonFixedFieldsGrow` for a consistent field grid. `params_panel.py`

Verification: 153 tests pass; changes confirmed in re-screenshots (01 main window,
06 feature panel).

## Iteration 2 — theming, contrast & primary-action emphasis

Theme/contrast (`theme.py`):
- DARK: set the 3D bevel roles (Light/Midlight/Mid/Dark/Shadow = 75/64/44/25/15)
  — Fusion left them at LIGHT greys, so group-box frames and the slider groove
  were nearly invisible on the dark window; now they read as recessed edges.
- Disabled text raised for legibility: dark 120→150; light gains a darkened
  disabled foreground (Fusion #bebebe→140) so greyed controls aren't ghosts.
- Accent harmonized: dark Highlight 42,130,218→58,150,235 with a separate lighter
  Link (80,170,235); light Highlight→#1769aa (AA white-on-blue) + Link→#0b5fb0.

Emphasis / secondary text (`theme.py` + `app.py` + `params_panel.py`):
- Run is now the primary action: objectName `runButton` + `setDefault(True)` + a
  per-theme accent QSS scoped to `:enabled` (disabled state untouched).
- `deviceBanner` + `matchCount` muted as secondary text via objectName QSS.
- QSS is applied at the **window** level (`MainWindow.setStyleSheet(theme.theme_qss(key))`),
  NOT the QApplication — an app stylesheet wraps the style in a proxy and blanks
  `app.style().objectName()` (broke a theme test); a widget stylesheet keeps the
  global Fusion style while still cascading to the dock children.

Verification: 153 tests pass; verified in light+dark re-screenshots (02 dark main
window frames/slider now visible; 08 dark panel + 03 light panel show the enabled
Run accent, muted readout, legible disabled text).

## Iteration 3 — toolbar/menu structure, de-duplication, spacing & copy

Toolbar / menus (`app.py`):
- Toolbar grouped by task with separators: Open | Pan mode / Fit | Clear matches |
  Calibrate / Measure. The duplicate "Add to Memory" action was removed from the
  toolbar (it lives only on the Memory dock now); "Self-test" (a diagnostic) was
  demoted off the toolbar into a new **Help** menu. Actions/handlers unchanged.

Layout / density (`params_panel.py`):
- Explicit, even margins/spacing on the root + Method/Match-parameters/Feature
  forms for a consistent vertical rhythm.
- Low-priority single-control groups flattened (`setFlat(True)`): Scale search,
  Orientation, and the RGB/YCbCr weight sub-groups — establishing framed-primary
  (Method, Match parameters, Feature matching) vs flat-secondary hierarchy and
  removing the frame-within-a-frame around the weight sliders.

Copy / consistency (`params_panel.py`):
- Sentence-case checkboxes: "Auto run", "Enable rotation", "Enable flipping"
  (matching "Search multiple scales").
- Friendly orientation readout via `ORIENTATION_LABELS`: "1 orientation: 0°"
  (was the internal code "R0").

Verification: 153 tests pass (test traps avoided — Tools-menu list, Theme/Engine
titles, status-bar count label all untouched); confirmed in re-screenshots.
