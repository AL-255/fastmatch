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
