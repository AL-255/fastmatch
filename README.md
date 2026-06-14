# FastMatch

FastMatch is a single-process PySide6 desktop app for exploring **gigapixel
images** and finding repeated visual patterns inside them. It loads the image
into a tiled, GPU-composited viewport (scroll-wheel zoom under the cursor,
drag-pan, left-drag region select) and runs a fast template search on whatever
box you draw, then overlays every other region of the image that looks similar.
The default matcher is **normalized cross-correlation (NCC)**, but you can pick a
different **matching method** from a dropdown — SSD for flat / low-texture
regions, CCORR for an alternate correlation measure, or Feature matching for
rotated / scaled / warped instances (see [Matching methods](#matching-methods)).
The correlation methods are built on PyTorch and run on the GPU when a suitable
CUDA build is installed, falling back transparently to CPU otherwise.

---

## Web (browser) edition — `docs/`

A zero-install version runs **entirely in the browser** via
[Pyodide](https://pyodide.org) and is hostable on **GitHub Pages** as static
files. Because PyTorch and Qt have no WebAssembly builds, the web edition is a
self-contained **NumPy reimplementation** of the convolution matchers
(`ncc`/`ssd`/`ccorr`) — same algorithm as the desktop's full-resolution path
(FFT cross-correlation + integral-image windowed statistics, dihedral-orientation
and multi-scale search, greedy IoU-NMS, source exclusion). Feature matching is
desktop-only (it needs OpenCV, unavailable under Pyodide).

The browser engine (`docs/fastmatch_web.py`) is pure NumPy, so it runs in
CPython too and is unit-tested (`tests/test_web_engine.py`), including a recall
**parity check against the desktop torch engine** and a check that the parallel
worker decomposition reproduces the single-call `match()`.

**Multi-core + responsive:** the search runs in a **pool of Web Workers** (one
Pyodide per core — `docs/worker.js`), not on the UI thread. The work is split
into independent *(orientation × scale)* tasks dispatched across the pool, and
the page merges the candidates (greedy NMS / source-exclusion / top-K). Because
the UI thread stays free, a **spinner** animates during a search and it is
**interruptible** — drawing a new box, changing a parameter, **Cancel**, or
`Esc` supersedes the in-flight search immediately (a monotonic generation token
discards stale worker results). Multi-core uses a worker pool rather than
Pyodide's pthreads because `SharedArrayBuffer` needs COOP/COEP headers that
GitHub Pages can't send.

**Try it locally:**

```bash
python -m http.server -d docs 8000   # then open http://localhost:8000
```

Open an image (or **Load sample**), drag a box around a pattern, and it finds
every similar region. Everything is client-side; it scales across CPU cores, but
very large images (more than a few thousand px per side) are still heavier in
WASM than on the native desktop engine.

**Deploy on GitHub Pages:** the repo ships a workflow
(`.github/workflows/pages.yml`) that publishes `docs/` via the official Pages
Actions. One-time setup: **Settings → Pages → Source = GitHub Actions**. After
that, every push that touches `docs/` builds and deploys automatically (or run
the workflow manually from the Actions tab); the app is served at
`https://<user>.github.io/<repo>/`. There is no build step — `docs/` already
contains everything (`index.html`, `app.js`, `style.css`, `fastmatch_web.py`,
`sample.png`) and `index.html` becomes the site root, so the app's relative
fetches resolve exactly as they do locally. Pyodide + NumPy load from a CDN on
first visit (~10 MB, cached). *(Alternatively, set Source to "Deploy from a
branch" → `main` → `/docs`, which needs no workflow.)*

---

## Install

FastMatch needs PySide6, NumPy, Pillow, and PyTorch. **torchvision is optional**
(used only for a slightly faster NMS; a pure-torch fallback ships in the engine),
and **OpenCV is optional too** — it is required *only* for the "Feature matching"
method (see [Matching methods](#matching-methods)); the NCC / SSD / CCORR methods
never touch it. The `requirements.txt` quickstart installs the `-headless` OpenCV
build so it does not clash with PySide6's Qt plugins.

### CPU quickstart

```bash
pip install -r requirements.txt
```

This installs everything, including the **CPU-only** PyTorch wheel. The app is
fully functional on CPU — every code path is device-agnostic — it is just
slower for large multi-scale searches.

### GPU upgrade (CUDA / cu128)

The PyPI PyTorch wheel ships **no sm_120 (Blackwell, e.g. RTX 5060) kernels**,
so even when `torch.cuda.is_available()` returns `True` the first real kernel
fails. To get genuine GPU acceleration, install a CUDA build from the PyTorch
CUDA index *after* the quickstart above:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
# cu129 is an alternative if cu128 is unavailable for your platform:
# pip install torch --index-url https://download.pytorch.org/whl/cu129
```

Verify the GPU build is active:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

- On the **CPU** wheel this prints `False`.
- On a **working CUDA** wheel it prints `True`, and FastMatch's status banner
  reads `Engine: CUDA (<your GPU name>)` instead of `Engine: CPU (slow) …`.

FastMatch never assumes CUDA exists: it detects the device at runtime with both
`torch.cuda.is_available()` and a launch-time **canary kernel**, so a broken or
mismatched CUDA wheel cleanly degrades to CPU rather than crashing.

---

## Run

```bash
# Open an image in the viewer
python -m fastmatch path/to/image.png

# Launch with no image — starts on an empty canvas; open one from
# File > Open Image…
python -m fastmatch

# Generate a synthetic ground-truth test image (textured noise + known motifs),
# then open it
python -m fastmatch --generate-sample sample.png --w 12000 --h 12000
python -m fastmatch sample.png

# Force a specific device (default is "auto": CUDA if usable, else CPU)
python -m fastmatch path/to/image.png --device cpu
python -m fastmatch path/to/image.png --device cuda
python -m fastmatch path/to/image.png --device auto
```

If no image path is given, FastMatch opens with an **empty canvas** (no
auto-generated demo) — load an image from *File ▸ Open Image…*. `--device`
accepts `auto`, `cuda`, or `cpu`. `cuda` still falls back to CPU if the canary
kernel fails. `--generate-sample <path>` writes a synthetic image (default
`12000x12000`, override with `--w` / `--h`) with a set of known motif stamps and
exits; load that file separately to search it.

---

## Interaction guide

| Action | Result |
|---|---|
| **Scroll wheel** | Zoom in/out, always anchored **under the cursor** (max 64x). The canvas is treated as unbounded, so the cursor stays the pivot even at the image edges. |
| **F** | Zoom to fit the whole image. |
| **Hold Space + drag**, or **middle-mouse drag** | Pan the view. |
| **Left-drag** | Draw the selection box (the search template). |
| **Run** button / **Auto Run** checkbox | With *Auto Run* on (default) the search runs whenever you draw a selection or change a setting. Turn it off to stage the selection + parameters and trigger a single search with **Run** (handy when each run is expensive). |
| **Channel dropdown** | The colour space the matcher scores in: `luminance` (a single BT.601 luma plane), `rgb`, or `ycbcr`. The two colour modes are **multi-channel** (each channel correlated separately) and expose **per-channel weight sliders** (R,G,B or Y,Cb,Cr) — see [Channel modes & weights](#channel-modes--weights). |
| Release a new selection | Runs the search (if Auto Run is on); any in-flight search is **auto-cancelled** (latest-wins). |
| **Method dropdown** | Pick the matching method (NCC / SSD / CCORR / Feature matching). Changing it **re-runs** the search on the current selection. See [Matching methods](#matching-methods). |
| **Rotation / Flipping checkboxes** | Also search the template under quarter-turn rotations and/or mirror reflections. Changing either **re-runs** the search. See [Orientation search](#orientation-search). |
| **Threshold slider** | **Live-filters** the displayed results — no re-run (works for every method). |
| **View ▸ Match boxes** menu | Configure the overlay box outlines: **Line width** (1–6 px, zoom-independent) and **XOR with background** (invert the outline against whatever is underneath so it stays visible on any background). |
| **Engine** menu | Switch the compute backend on the fly: **Auto (prefer GPU)**, **CUDA (GPU)**, or **CPU**. CUDA is greyed out when no working GPU is detected. Switching rebuilds the engine on the new device, updates the status banner, re-gates the GPU-only multi-scale search, and re-runs the current selection. |
| **Theme** menu | Switch the application look: **System** (follow the desktop), **Light**, or **Dark**. The whole UI and the image canvas re-theme instantly, and the choice is remembered for next launch. |
| **Tools** menu | **Calibrate scale** and **Measure distance** (physical units). See [Measurement & calibration](#measurement--calibration). |
| **Add to Memory** | Save the current selection + all current matches as an entry in the Memory list. See [Saved-match Memory](#saved-match-memory). |
| **Double-click a Memory entry** | Revisit that saved search — restores the blue reference box to its remembered selection and re-shows its matches. |

The **source region you selected is excluded** from the matches (it would
otherwise always be a perfect self-match), and that exclusion is shown in the
UI. Selections that are too small (< 8 px on a side), nearly the whole image, or
extremely elongated (aspect > 20) are rejected.

---

## How matching works (brief)

FastMatch computes **normalized cross-correlation** between your selected
template and the whole image on the GPU (or CPU). NCC is brightness- and
contrast-invariant, so it tolerates the lighting jitter common across repeated
instances. Key points:

- **Multi-scale:** the template is matched at several scales (a wider grid on
  GPU, a single scale on CPU to stay responsive); candidates from all scales
  are pooled and resolved with a single global **non-maximum suppression (NMS)**
  so the best scale wins per location.
- **Rotation is NOT searched by default.** Rotated instances are generally
  missed unless you opt in (which multiplies cost by the number of angles).
- **Out-of-grid scales may be missed.** Instances much larger or smaller than
  the scanned scale grid can fall through; widen the scales for such cases.
- Matching runs on a single background worker thread with cooperative
  cancellation and a 120 ms debounce, so dragging a new box smoothly replaces
  the previous search.

---

## Matching methods

A "perfectly flat" region and a "warped" region call for different matchers, so
FastMatch lets you choose one from the **Method dropdown** in the params panel.
Changing the method re-runs the search on your current selection; the threshold
slider stays a live, no-re-run client-side filter for every method.

| Method | What it does | When to use it | Runs on |
|---|---|---|---|
| **NCC** (default) | Normalized cross-correlation (CCOEFF). Subtracts the mean and divides by per-window variance, so it is **illumination-robust**. | **Textured, aligned, same-scale** instances — the general default. Needs some internal texture to normalize against. | **GPU** (CPU fallback) |
| **SSD** | Normalized squared difference: `1 − RMSE` of the pixel-wise difference. | **Flat / low-texture / exact-appearance** regions, where NCC's variance normalization is unstable (or rejects the template outright). **Not** illumination-invariant — use it when brightness is consistent. | **GPU** (CPU fallback) |
| **CCORR** | Cosine cross-correlation (CCORR_NORMED) — an alternate correlation measure that does not subtract the mean. | A correlation alternative to NCC; useful as a cross-check when NCC behaves oddly. | **GPU** (CPU fallback) |
| **Feature matching** | Dense ORB/AKAZE/SIFT keypoints **propose** a candidate at every repeated copy (one correspondence per copy — not the Lowe ratio test), and each candidate is **verified by appearance** (normalized cross-correlation). | **Rotated / scaled / perspective-warped** instances that the template (window-based) methods miss — and a feature-driven way to **reproduce the correlation methods' matches** on repetitive content. | **CPU**, via OpenCV |

Notes:

- **NCC, SSD and CCORR are GPU-accelerated** (PyTorch) and share all of the same
  machinery — tiling, halos, the multi-scale sweep, non-maximum suppression,
  source exclusion, the result cap, progress and cancellation. They differ only
  in the per-window score formula, so switching among them is cheap.
- **NCC rejects near-flat (featureless) templates** with a message, because its
  variance normalization is ill-defined there. **SSD does not** — that is exactly
  its use case, so switch to SSD for solid-colour or very-low-texture targets.
- **Feature matching runs on the CPU via OpenCV**, even when the correlation
  methods are using CUDA. It detects keypoints **densely at full resolution**
  (tiled so memory stays bounded; the per-image detection is cached and reused
  across queries), proposes a candidate instance at every repeated copy, and
  **verifies each by appearance** (zero-mean normalized cross-correlation) so its
  hits line up with what the correlation methods would find — with very few false
  positives. It ignores the scale grid (it is inherently scale/rotation-tolerant),
  but it **honours the channel mode**: keypoint *detection* is always grayscale
  (ORB/AKAZE/SIFT are), while in **rgb / ycbcr** mode the appearance verification
  is colour-aware (a weighted per-channel NCC over the three channels) — useful
  when copies match in luminance but differ in colour. Its detector (ORB / AKAZE /
  SIFT) and match-count controls are exposed in the panel.
- Feature matching requires **OpenCV** (`opencv-python-headless`, installed by
  the quickstart). If OpenCV is missing the dropdown greys that option out with a
  tooltip and the other three methods keep working.

---

## Channel modes & weights

The **Channel mode** dropdown chooses the colour space every method scores in,
and applies to **all four methods**:

| Mode | What it matches on |
|---|---|
| **luminance** (default) | A single BT.601 luma plane — fastest, least memory. |
| **rgb** | The three **R, G, B** channels, each correlated separately and combined with the **RGB weight sliders**. |
| **ycbcr** | The three **Y, Cb, Cr** channels (BT.601), combined with the **YCbCr weight sliders**. |

The colour modes are genuinely **multi-channel** — each channel is correlated on
its own and the per-channel scores are summed (not flattened into one projected
grey plane), so colour differences stay discriminative. When **rgb** or **ycbcr**
is selected, three **weight sliders** appear (R/G/B or Y/Cb/Cr); they are
**normalized to sum to 1.0** (the readout shows the normalized values) and weight
each channel's contribution to the combined score. This works the same way for
the GPU correlation methods (NCC/SSD/CCORR) and for feature matching's appearance
verification. Some useful points in the space:

- **Equal weights** reproduce the unweighted multi-channel behaviour.
- A single **Y weight (1, 0, 0)** in ycbcr is exactly luminance matching.
- **Chroma-weighted** ycbcr (weight on Cb/Cr, little or no Y) matches by *colour*:
  it accepts only same-coloured copies and rejects ones that merely share the same
  brightness — and conversely, weighting Y ignores colour and matches by shape.
- Luminance stays the cheapest default; the colour modes stage extra per-channel
  planes (ycbcr is derived lazily, only when first used).

---

## Orientation search

By default the search looks for the template in its **upright, unmirrored**
orientation only. Two checkboxes in the params panel widen the search to the
**8 symmetries of a square** (the dihedral group D4):

- **Rotation** — also match the template rotated by 90°, 180° and 270°.
- **Flipping** — also match the template mirrored (horizontal and vertical
  flips). With **both** boxes on, the two diagonal reflections (mirror **and** a
  quarter-turn) are searched as well, for all 8 orientations.

| Rotation | Flipping | Orientations searched |
|---|---|---|
| off | off | upright only (default — identical to before this feature) |
| on  | off | upright + 90° / 180° / 270° |
| off | on  | upright + horizontal / vertical mirror |
| on  | on  | all 8 (rotations, mirrors, and diagonal reflections) |

**All four methods honor these checkboxes.** The correlation methods
(NCC / SSD / CCORR) re-run their score-map search once per active orientation
and keep the best one per location; feature matching proposes the template under
each active orientation, appearance-verifies each instance, and classifies the
orientation it was found under. Each result records the orientation it was found
under, so a hit can be a rotated or mirrored copy of your selection. With both
boxes off, behaviour is exactly the upright-only search described above (no
extra cost). Changing either checkbox re-runs the search on your current
selection; the threshold slider stays a live, no-re-run filter.

---

## Saved-match Memory

The **Memory** panel keeps a list of saved searches so you can collect and
compare interesting matches across a session.

- **Add to Memory** — after a search completes, click *Add to Memory* to append
  an entry that captures the **current selection box**, **all current matches**,
  and the **complete settings** used (the full `MatchParams`: method, channel
  mode, threshold(s), scales, orientation flags, NMS/exclude IoU, max results,
  compute dtype, and the feature-matching parameters).
- **Per-entry stats** — each row shows the method, channel mode, selection,
  **occurrences** (the number of matches **plus the reference selection** — e.g.
  2 matches show as 3 occurrences), score range, and a compact per-orientation
  breakdown (e.g. `R0:2 R90:1 MY:1`); hovering a row shows **all** of that
  entry's settings.
- **Rename…** — give the selected entry a custom name (a blank name reverts to
  the auto summary); the name is shown in the list and saved to JSON.
- **Remove** — deletes the selected line(s) from the list.
- **Double-click** an entry to **revisit** it — FastMatch restores the blue
  reference box to the entry's remembered selection and re-shows its matches.
- **File menu** — all file operations live under **File**: *Open Image…*,
  *Close Image*, *Open Memory…*, *Save Memory* (writes to the current file, or
  prompts if none yet; `Ctrl+S`), *Save Memory As…* (`Ctrl+Shift+S`), and
  *Close Memory* (clear the list). Saving always writes a `.json` file — a name
  typed without an extension (e.g. `patterns`) is saved as `patterns.json`
  automatically. A Memory `.json` records the **source image**
  (its path and pixel size) and **every entry** (selection, settings, and each
  match with its score, scale and orientation). Match coordinates are in the
  **source image's pixel space**, so a Memory opened against the same image lines
  its boxes up exactly; opening one recorded for a different image offers to open
  that image. A file written by a newer FastMatch (a higher schema version), or
  any malformed/non-JSON file, is refused with a clear error rather than crashing.
- **Engine menu** — switch the compute backend without restarting: *Auto (prefer
  GPU)*, *CUDA (GPU)*, or *CPU*. The CUDA entry is disabled when no working GPU
  is detected (the same canary-gated probe used at launch), so you can never
  select an unavailable backend. Switching tears down the old worker thread,
  rebuilds the engine on the chosen device, refreshes the status banner, re-gates
  the GPU-only multi-scale search, and re-runs the current selection — matching
  the `--device` flag you could have launched with. The view, selection, and
  Memory list are preserved across the switch.
- **Theme menu** — switch the application appearance at runtime: *System*
  (follow the desktop's own theme), *Light*, or *Dark*. The whole window and the
  image canvas re-theme instantly. Light/Dark use hand-built palettes on Qt's
  palette-faithful **Fusion** style so they look identical on every platform;
  *System* restores the desktop theme captured at startup. The choice is
  persisted (via `QSettings`) and re-applied on the next launch, before the first
  paint, so there is no startup flash. Match/selection box colours are kept
  constant across themes (a match always reads as the same green).

---

## Measurement & calibration

The **Tools** menu (and toolbar) add physical-scale measurement on top of the
pixel grid:

- **Calibrate scale** — click *Calibrate*, then drag a line along a span of known
  physical length (a scale bar, a chip edge). Enter the length when prompted
  (e.g. `5.36 mm` — a bare number reuses the last unit). The entered length maps
  to the **longer** of the horizontal/vertical pixel spans (`max(|Δx|, |Δy|)`),
  the natural choice for a feature aligned to one axis. The **first point becomes
  the physical-grid origin**. The reference line stays drawn (orange) labelled
  with its length.
- **Measure distance** — click *Measure*, then drag a line; its **physical
  distance** (true Euclidean length × scale) is labelled on the line (amber) and
  shown in the status bar. Before calibration it reports pixels.
- **Physical cursor coordinates** — once calibrated, the status bar shows the
  cursor position in physical units relative to the calibration origin, alongside
  the pixel coordinate, e.g. `(1203, 540) px   (5.36, 2.41) mm`.
- **Selection area** — the status bar shows the **physical area** of the current
  selection box, e.g. `area 31.3 mm²` (pixel `w·h × scale²`).

Both tools are one-shot (a single drag, then the previous Pan/Select mode is
restored). Calibration is per-image and resets when you open/close an image;
*Clear calibration* and *Clear measurement* remove them.

---

## Self-test

FastMatch ships a built-in correctness check:

- **Self-test menu action** — synthesizes an image, stamps a known motif at N
  positions, runs the matcher, and confirms it recovers every planted instance
  (centers within ±1 px) with the source region excluded.
- **`--generate-sample` workflow** — the same generator is exposed on the CLI so
  you can produce a ground-truth image, open it, draw a box around one motif,
  and visually confirm all other instances light up:

  ```bash
  python -m fastmatch --generate-sample sample.png --w 12000 --h 12000
  python -m fastmatch sample.png
  ```

---

## Troubleshooting

- **`CUDA error: no kernel image is available for execution`** — your PyTorch
  wheel has no kernel for your GPU's compute capability (the sm_120 / Blackwell
  case). Install the CUDA build from the cu128 index (see
  [GPU upgrade](#gpu-upgrade-cuda--cu128)). FastMatch detects this at launch via
  its canary kernel and **auto-falls back to CPU**, so the app keeps working in
  the meantime — just slower, with a `Engine: CPU (slow) …` banner.

- **Very large / gigapixel images** — images that do not fit comfortably in RAM
  are decoded through a **memory-mapped (memmap)** path and streamed tile-by-tile
  to the GPU and the display pyramid; no full-image texture is ever uploaded.
  Truly oversized images are refused up front with a dialog rather than being
  allowed to OOM-kill the process.

- **Out of memory on the GPU** — the engine starts conservatively (1024x1024
  compute tiles) and, on `OutOfMemoryError`, **auto-degrades** along a ladder:
  1024 → 512 tile → fewer scales → CPU fallback, surfacing a message at each
  rung. You do not need to tune anything manually.

- **Searches feel slow** — you are likely on the CPU build. Confirm with
  `python -c "import torch; print(torch.cuda.is_available())"`; if it prints
  `False`, install the cu128 GPU build as described above.
