# FastMatch

[English](README.md) | [简体中文](readme_cn.md)

![FastMatch](docs/top.png)

FastMatch is a single-process PySide6 desktop app for exploring **gigapixel
images** and finding repeated visual patterns inside them. It loads the image
into a tiled, GPU-composited viewport (scroll-wheel zoom under the cursor,
drag-pan, left-drag region select) and runs a fast template search on whatever
box you draw, then overlays every other region of the image that looks similar.
The default matcher is **normalized cross-correlation (NCC)**, but you can pick a
different **matching method** from a dropdown — SSD for flat / low-texture
regions, CCORR for an alternate correlation measure, or Feature matching for
rotated / scaled / warped instances (see [Matching methods](#matching-methods)).
The correlation methods are built on PyTorch and support NVIDIA CUDA, AMD ROCm,
and Apple MPS, falling back to CPU when an accelerator is unavailable.
For image-search-style similarity, choose **Fuzzy instance search** to find
altered copies, or **Semantic similarity** to find visually related objects.
See [Fuzzy and semantic search](#fuzzy-and-semantic-search).

---

## Install

### Download an executable

Tagged versions build standalone applications on [GitHub Releases](https://github.com/AL-255/fastmatch/releases).
No Python installation is needed to run an extracted application.

| Platform | Builds |
|---|---|
| Windows x64 | CPU, NVIDIA CUDA 12.8 |
| Linux x64 (Ubuntu 22.04 or newer / compatible desktop) | CPU, NVIDIA CUDA 12.8, AMD ROCm 7.2 |
| macOS Apple Silicon | Apple MPS GPU, with CPU fallback |

Choose the runtime for your hardware, extract the archive, and keep the complete
folder together. Launch `FastMatch.exe` on Windows, `FastMatch.app` on macOS, or
`FastMatch/FastMatch` on Linux. **Auto** selects a working accelerator; **Engine**
also offers CPU, CUDA/ROCm, and Apple MPS. CUDA/ROCm builds require compatible GPU
drivers. macOS builds are ad-hoc signed, not notarized; Windows builds are unsigned.
The semantic model is included and works offline.

Large GPU archives are split into `.part001`, `.part002`, etc. Download **all**
parts for your chosen build and its `.parts.json` manifest into one folder.
The release includes helpers that join the parts and verify SHA256:

```bash
python assemble_release.py FastMatch-vX.Y.Z-linux-x64-rocm7.2.tar.gz.parts.json
```

On Windows, use the PowerShell helper without Python:

```powershell
./assemble_release.ps1 -Manifest FastMatch-vX.Y.Z-windows-x64-cuda-cu128.zip.parts.json
```

Extract the reconstructed archive afterward. Each build also includes a `.sha256`
checksum file and its packaged-app smoke-test report. See [Release builds](docs/RELEASING.md)
for local builds and the tag-triggered workflow.

### Run from source

FastMatch needs PySide6, NumPy, Pillow, and PyTorch. **torchvision is optional**
(used only for a slightly faster NMS; a pure-torch fallback ships in the engine),
and **OpenCV is optional too** — it is required for strict feature matching and
fuzzy instance search (see [Matching methods](#matching-methods)); the NCC / SSD / CCORR methods
never touch it. The `requirements.txt` quickstart installs the `-headless` OpenCV
build so it does not clash with PySide6's Qt plugins.

### CPU quickstart

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Installing PyTorch from its CPU index first explicitly selects the **CPU-only**
runtime on Windows/Linux (the macOS wheel also includes MPS). The app is
fully functional on CPU — every code path is device-agnostic — it is just
slower for large multi-scale searches.

### GPU upgrade (CUDA / cu128)

For NVIDIA acceleration, replace the CPU runtime with the tested CUDA 12.8 build.
A CUDA availability flag alone does not guarantee that the driver and GPU
architecture are supported; FastMatch also executes a canary kernel at launch.

```bash
pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
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

For AMD on Linux, install the ROCm runtime. Apple Silicon uses MPS from the
normal macOS PyTorch wheel, with a separate canary probe:

```bash
pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

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

# Force a device (auto tries CUDA/ROCm, then Apple MPS, then CPU)
python -m fastmatch path/to/image.png --device cpu
python -m fastmatch path/to/image.png --device cuda
python -m fastmatch path/to/image.png --device rocm
python -m fastmatch path/to/image.png --device mps
python -m fastmatch path/to/image.png --device auto
```

If no image path is given, FastMatch opens with an **empty canvas** (no
auto-generated demo) — load an image from *File ▸ Open Image…*. `--device`
accepts `auto`, `cuda`, `rocm`, `mps`, or `cpu`. CUDA/ROCm and Apple MPS are probed
with real kernels before use; an unavailable accelerator falls back to CPU.
`cuda` still falls back to CPU if the canary
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
| **Method dropdown** | Pick NCC / SSD / CCORR, Fuzzy instance search, Semantic similarity, or Strict feature matching. Changing it **re-runs** the search on the current selection. See [Matching methods](#matching-methods). |
| **Rotation / Flipping checkboxes** | Also search the template under quarter-turn rotations and/or mirror reflections. Changing either **re-runs** the search. See [Orientation search](#orientation-search). |
| **Threshold slider** | **Live-filters** the displayed results — no re-run (works for every method). |
| **View ▸ Match boxes** menu | Configure the overlay box outlines: **Line width** (1–6 px, zoom-independent) and **XOR with background** (invert the outline against whatever is underneath so it stays visible on any background). |
| **Engine** menu | Switch between **Auto (prefer GPU)**, **CUDA (NVIDIA GPU)** or **ROCm (AMD GPU)** according to the installed runtime, **MPS (Apple GPU)**, and **CPU**. Unavailable accelerators are greyed out. Switching rebuilds the engine, updates the status banner and multi-scale controls, and re-runs the current selection. |
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
| **Fuzzy instance search** | RootSIFT descriptors, repeated-instance retrieval, and per-instance RANSAC geometric verification. Scores descriptor similarity, keypoint coverage, and geometric consistency, without a pixel-correlation gate. | **Blurred / compressed / recoloured / partly occluded** copies with distinctive local details, including rotation, scale and perspective changes. | **CPU**, via OpenCV |
| **Semantic similarity** | Pretrained DINOv2 embeddings and cosine similarity over overlapping image windows, with local refinement. | **Visually related objects with different appearances**, including regions that have no matching keypoints. | **GPU or CPU**, via PyTorch; optional local model |
| **Strict feature matching** | Dense ORB/AKAZE/SIFT keypoints **propose** a candidate at every repeated copy (one correspondence per copy — not the Lowe ratio test), and each candidate is **verified by appearance** (normalized cross-correlation). | **Rotated / scaled / perspective-warped** copies whose pixels still correlate strongly after alignment. | **CPU**, via OpenCV |

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

The **Channel mode** dropdown chooses the matching colour space,
and applies to **NCC / SSD / CCORR, strict feature matching, and fuzzy instance search**.
Semantic similarity always uses RGB learned features and disables these controls.

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

**All methods honor these checkboxes.** The correlation methods
(NCC / SSD / CCORR) re-run their score-map search once per active orientation
and keep the best one per location; feature matching proposes the template under
each active orientation, appearance-verifies each instance, and classifies the
orientation it was found under. Each result records the orientation it was found
under, so a hit can be a rotated or mirrored copy of your selection. With both
boxes off, behaviour is exactly the upright-only search described above (no
extra cost). Changing either checkbox re-runs the search on your current
selection; the threshold slider stays a live, no-re-run filter.
Fuzzy search classifies the recovered geometric transform; semantic search
compares embeddings of the selected template orientations and adjusts window
aspect ratios for quarter turns.

---

## Fuzzy and semantic search

These are separate retrieval modes: finding an altered copy and finding a
different but visually related object need different evidence.

- **Fuzzy instance search** uses [RootSIFT](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
  with geometric verification. Image descriptors query the template index, so
  repeated copies do not compete for a fixed nearest-neighbour quota. **Min matches**
  controls independent keypoint support; the threshold filters a combined
  descriptor/geometry score. Use luminance when recolouring should be tolerated;
  RGB/YCbCr weights add colour-histogram similarity. It infers scale and needs
  distinctive details; plain, textureless shapes may yield no matches.
- **Semantic similarity** uses local [DINOv2-small](https://github.com/facebookresearch/dinov2)
  weights. It ranks overlapping windows, refines promising positions, and
  merges overlaps. Multi-scale search also works on CPU. Results are approximate
  regions based on the selection's size and aspect ratio, rather than object
  segmentations. Very large images with small selections require many encoder
  calls and can be slow; cancellation and progress remain available. Domain
  performance depends on the pretrained features.
- Scores are similarities, **not probabilities**, and differ between methods.
  Start around `0.5` and adjust the threshold. Positive examples are searched
  separately; negative examples use the chosen retrieval evidence rather than
  a pixel-correlation filter. Both modes support selection masks, source
  exclusion, result limits, orientation search, and saved Memory entries.

Fuzzy search works with the normal quickstart. To enable semantic search:

```bash
pip install -r requirements-semantic.txt
python -m fastmatch --download-semantic-model
```

The explicit download stores model data in `models/dinov2-small` (ignored by Git).
Searches then run **offline**; no image is uploaded and searching never downloads
weights. The method is greyed out until dependencies and weights are present.
Restart the app after setup. For another location, download and launch with:

```bash
python -m fastmatch --download-semantic-model /path/to/dinov2-small
FASTMATCH_SEMANTIC_MODEL=/path/to/dinov2-small python -m fastmatch
```

Reproducible CPU comparisons:

```bash
python scripts/fuzzy_eval.py
python scripts/feature_eval.py --method fuzzy
```

The default synthetic altered-copy scene recovers **4/4 copies with no false
positives** (the strict matcher recovers 3/4, missing 50% occlusion). The separate
rotation/scale scene recovers **36/36 with no false positives**. These controlled
scenes are regression checks, not guarantees for arbitrary images.

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

- **Out of memory on the GPU** — the correlation engine starts conservatively (1024x1024
  compute tiles) and, on `OutOfMemoryError`, **auto-degrades** along a ladder:
  1024 → 512 tile → fewer scales → CPU fallback, surfacing a message at each
  rung. You do not need to tune anything manually.

- **Searches feel slow** — you are likely on the CPU build. Confirm with
  `python -c "import torch; print(torch.cuda.is_available())"`; if it prints
  `False`, install the cu128 GPU build as described above.
