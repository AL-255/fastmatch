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

# Generate a synthetic ground-truth test image (textured noise + known motifs),
# then open it
python -m fastmatch --generate-sample sample.png --w 12000 --h 12000
python -m fastmatch sample.png

# Force a specific device (default is "auto": CUDA if usable, else CPU)
python -m fastmatch path/to/image.png --device cpu
python -m fastmatch path/to/image.png --device cuda
python -m fastmatch path/to/image.png --device auto
```

`--device` accepts `auto`, `cuda`, or `cpu`. `cuda` still falls back to CPU if
the canary kernel fails. `--generate-sample <path>` writes a synthetic image
(default `12000x12000`, override with `--w` / `--h`) with a set of known motif
stamps and exits; load that file separately to search it.

---

## Interaction guide

| Action | Result |
|---|---|
| **Scroll wheel** | Zoom in/out, anchored **under the cursor** (max 64x). |
| **Hold Space + drag**, or **middle-mouse drag** | Pan the view. |
| **Left-drag** | Draw the selection box (the search template). |
| Release a new selection | Re-runs the search; any in-flight search is **auto-cancelled** (latest-wins). |
| **Method dropdown** | Pick the matching method (NCC / SSD / CCORR / Feature matching). Changing it **re-runs** the search on the current selection. See [Matching methods](#matching-methods). |
| **Threshold slider** | **Live-filters** the displayed results — no re-run (works for every method). |

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
| **Feature matching** | ORB keypoints + Lowe ratio test + sequential RANSAC **homography**, returning multiple instances. | **Rotated / scaled / perspective-warped** instances that the template (window-based) methods miss — they only score a single fixed orientation and scale. | **CPU**, via OpenCV |

Notes:

- **NCC, SSD and CCORR are GPU-accelerated** (PyTorch) and share all of the same
  machinery — tiling, halos, the multi-scale sweep, non-maximum suppression,
  source exclusion, the result cap, progress and cancellation. They differ only
  in the per-window score formula, so switching among them is cheap.
- **NCC rejects near-flat (featureless) templates** with a message, because its
  variance normalization is ill-defined there. **SSD does not** — that is exactly
  its use case, so switch to SSD for solid-colour or very-low-texture targets.
- **Feature matching runs on the CPU via OpenCV**, even when the correlation
  methods are using CUDA. It detects features on a downsampled copy of the image
  for speed and maps results back to full resolution. It ignores the scale grid
  and channel-mode controls (it is inherently scale/rotation-tolerant and works
  in grayscale); its detector (ORB / AKAZE / SIFT) and minimum-inlier count are
  exposed in the panel instead.
- Feature matching requires **OpenCV** (`opencv-python-headless`, installed by
  the quickstart). If OpenCV is missing the dropdown greys that option out with a
  tooltip and the other three methods keep working.

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
