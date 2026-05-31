# FastMatch — Final Implementation-Ready Specification

> Authoritative design + interface contract. T0 shared types (`fastmatch/types.py`,
> `fastmatch/document.py`, `fastmatch/device.py`, `fastmatch/__init__.py`) are already
> implemented and **frozen** — code against them exactly, do not edit them.

## A. OVERVIEW & TECH-STACK DECISION

FastMatch is a single-process PySide6 desktop app that loads a gigapixel image into a tiled, GPU-composited viewport (zoom-under-cursor, pan, left-drag region select) and runs a fast normalized-cross-correlation (NCC) template search on the boxed region to find and overlay all other visually similar instances. We build entirely on the already-installed stack — **PySide6 6.11, PyTorch (device-agnostic), numpy, Pillow** — and add **no new runtime dependencies** (torchvision is *optional*, used for `nms` only if importable, else a bundled pure-torch NMS). The single hard external requirement for GPU acceleration is replacing the present CPU-only `torch 2.11.0+cpu` with a CUDA wheel from the PyTorch CUDA index (`--index-url https://download.pytorch.org/whl/cu128` or `/cu129`) that ships sm_120 (Blackwell) kernels; the code never assumes CUDA exists — it runtime-detects with `torch.cuda.is_available()` *and* a launch-time canary kernel, falling back to CPU otherwise. The three heavy facets (viewport, engine, integration) are decoupled by a strict contract: **everything outside the viewport speaks half-open integer image-pixel boxes `(x,y,w,h)`; only `QRect`/`QRectF` ever touch the viewport/controller boundary; the engine has zero Qt imports.**

## B. FILE / MODULE LAYOUT

```
fastmatch/
├── __init__.py          # [T0 DONE] version, Image.MAX_IMAGE_PIXELS=None side-effect
├── __main__.py          # CLI entry: argparse, QApplication attrs, builds MainWindow; --device/--generate-sample/--w/--h
├── app.py               # MainWindow(QMainWindow): toolbar/dock/statusbar, wires Viewport+Controller+ParamsPanel, closeEvent
├── viewport.py          # ImageViewport(QGraphicsView): tiled pyramid item, overlay item, zoom/pan/select, OpenGL viewport
├── pyramid.py           # ImagePyramid, Level, TileCache (LRU), tile-decode QRunnable workers
├── overlay.py           # MatchOverlayItem(QGraphicsItem): batched numpy-backed box drawing + client-side threshold filter
├── controller.py        # MatchController(QObject): QThread owner, debounce, cancel, job_id, busy
├── worker.py            # MatchWorker(QObject): run/cancel slots, progress/finished/error signals
├── engine.py            # Matcher: device-agnostic torch NCC engine, tiled, multi-scale, NMS. NO Qt imports.
├── device.py            # [T0 DONE] resolve_device(), canary_kernel_ok(), device_banner_text()
├── document.py          # [T0 DONE] ImageDocument dataclass (the shared source-of-truth array)
├── types.py             # [T0 DONE] Match, MatchParams dataclasses (picklable, Qt-free)
├── loader.py            # load_image, _load_via_memmap, generate_sample (synthetic ground-truth image)
└── params_panel.py      # ParamsPanel(QWidget): threshold/scales/max_results/device widgets -> MatchParams
tests/
├── test_loader.py       # memmap strip-decode correctness, MAX_IMAGE_PIXELS, contiguity
├── test_engine.py       # generate_sample -> match -> IoU recall/precision vs ground truth; tile-seam + scale cases
└── test_coords.py       # selection->crop->overlay round-trip; half-open invariants; <=1px drift
requirements.txt         # PySide6, numpy, Pillow pinned; torch with CPU/GPU install notes in comments
README.md                # install (CPU + cu128 GPU), usage, interaction, self-test, troubleshooting
```

## C. INTERFACE CONTRACTS

### C.1 `types.py` — cross-boundary data (Qt-free, picklable) — IMPLEMENTED (frozen)

```python
@dataclass(frozen=True)
class Match:
    x: int          # left,  image px, INCLUSIVE top-left
    y: int          # top,   image px, INCLUSIVE top-left
    w: int          # width  in image px (half-open: covers x .. x+w)
    h: int          # height in image px (half-open: covers y .. y+h)
    score: float    # NCC peak in [0,1]  (raw NCC in [-1,1] is min-clamped/remapped; see D)
    scale: float    # scale at which found (1.0 == template native size)

@dataclass(frozen=True)
class MatchParams:
    threshold: float = 0.85
    threshold_floor: float = 0.50
    scales: tuple[float, ...] = (1.0,)
    rotations: tuple[float, ...] | None = None
    max_results: int = 500
    nms_iou: float = 0.30
    exclude_iou: float = 0.30
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    compute_dtype: str = "float32"  # "float32" | "float16" (accumulators stay fp32)
    channel_mode: str = "luminance"  # "luminance" | "rgb"
```

### C.2 `document.py` — IMPLEMENTED (frozen)

```python
@dataclass
class ImageDocument:
    full: np.ndarray   # (H,W,3) uint8 RGB, C-CONTIGUOUS; may be np.memmap. READ-ONLY by all consumers.
    path: str
    height: int        # == full.shape[0]
    width: int         # == full.shape[1]
    # __post_init__ validates shape (H,W,3), dtype uint8, and size consistency.
```

### C.3 `device.py` — IMPLEMENTED (frozen)

```python
def resolve_device(pref: str = "auto") -> torch.device   # "auto"|"cuda"|"cpu"; canary-gated; CPU fallback
def canary_kernel_ok() -> bool                            # 8x8 cuda matmul + .item(); False on any exception
def device_banner_text(dev: torch.device) -> str          # "Engine: CUDA (NVIDIA ...)" / "Engine: CPU (slow) ..."
```

### C.4 `loader.py`

```python
def load_image(path: str | Path, *, max_ram_bytes: int = 6_000_000_000) -> ImageDocument
def _load_via_memmap(path, w: int, h: int) -> "np.memmap"        # strip-decode, 1024-row strips
def generate_sample(path: str | Path, w: int = 12000, h: int = 12000, *,
                    tile: int = 200, n_targets: int = 40, seed: int = 0) -> list[Match]
    # writes textured-noise PNG with N known motif stamps; RETURNS ground-truth Match list
```

### C.5 `engine.py` — `Matcher`

```python
class Matcher:
    def __init__(self, *,
        device: str | None = None,                                  # None=auto
        compute_dtype: Literal["float32","float16"] = "float32",
        channel_mode: Literal["luminance","rgb"] = "luminance",
        conv_backend: Literal["auto","spatial","fft"] = "auto",
        vram_fraction: float = 0.6) -> None: ...

    def set_image(self, image: np.ndarray) -> None: ...
        # (H,W,3) uint8 or (H,W) uint8. Computes luminance, builds tiling plan, stages to pinned host mem.
        # Reused across all match() calls. Accepts memmap (streams tiles, never one big upload).

    def match(self, template: np.ndarray, params: MatchParams, *,
        exclude_box: tuple[int,int,int,int] | None = None,
        cancel: Callable[[], bool] | None = None,
        progress: Callable[[int], None] | None = None) -> list[Match]: ...
        # Returns matches sorted by score desc, source region excluded. Image-px, scale-1, origin TL, +x/+y down.
        # cancel() polled at every tile boundary; progress(0..100) per finished tile.

    @property
    def image_size(self) -> tuple[int, int]: ...        # (H, W)
    @property
    def effective_device(self) -> torch.device: ...
```

The `device.py` resolution and `vram_fraction` budget are encapsulated inside `Matcher`; the public `device` arg of `MatchParams` is mapped to `Matcher` construction by the controller.

### C.6 `worker.py` — `MatchWorker(QObject)`

```python
class MatchWorker(QObject):
    progress = Signal(int)          # 0..100
    finished = Signal(object, int)  # (list[Match], job_id)
    error    = Signal(str, int)     # (traceback_str, job_id)
    def __init__(self, engine: "Matcher") -> None: ...
    @Slot(object, object, object, int)
    def run(self, template: "np.ndarray", params: "MatchParams",
            exclude_box: "tuple|None", job_id: int) -> None: ...
    def request_cancel(self) -> None: ...     # sets internal threading.Event
```

Emitted payloads are **plain CPU data only** (`list[Match]` dataclasses, str). No CUDA tensors, numpy arrays >MB, or QPixmaps ever cross a signal. `job_id` rides on every result for stale-drop.

### C.7 `controller.py` — `MatchController(QObject)`

```python
class MatchController(QObject):
    matches_ready = Signal(object)    # list[Match]  -> MainWindow -> Viewport
    busy_changed  = Signal(bool)
    progress      = Signal(int)       # 0..100, forwarded from worker
    failed        = Signal(str)
    def __init__(self, doc: "ImageDocument", params0: "MatchParams") -> None: ...
        # builds Matcher(device=...), calls engine.set_image(doc.full), owns QThread+MatchWorker
    @Slot(QRect, object)
    def request(self, rect_img: QRect, params: "MatchParams") -> None: ...
        # validates+crops, cancels in-flight, (re)arms 120ms debounce, latest-wins
    def shutdown(self) -> None: ...   # cancel, thread.quit(), thread.wait()
```

### C.8 `viewport.py` — `ImageViewport(QGraphicsView)`

```python
class ImageViewport(QGraphicsView):
    class Mode(enum.Enum):
        SELECT = enum.auto()
        PAN    = enum.auto()

    # signals
    regionSelected = Signal(QRect)        # template rect, FULL-RES image px, half-open
    matchesChanged = Signal(int)          # count currently shown (post client-side threshold)
    cursorImagePos = Signal(int, int)     # live image px under cursor (-1,-1 if outside)
    zoomChanged    = Signal(float, int)   # (view_scale, current_pyramid_level)
    viewChanged    = Signal(QRect)        # visible image-px rect (prefetch hint)

    def __init__(self, parent: QWidget | None = None) -> None: ...
    # image
    def set_image(self, doc: "ImageDocument", *, tile: int = 256) -> None: ...
    def image_size(self) -> tuple[int, int]: ...           # (W, H)
    def clear_image(self) -> None: ...
    # matches / overlay
    def set_matches(self, matches: list["Match"]) -> None: ...   # converts to numpy internally
    def clear_matches(self) -> None: ...
    def set_match_threshold(self, t: float) -> None: ...   # 0..1 client-side filter, no re-run
    def visible_match_count(self) -> int: ...
    # selection / template
    def template_rect(self) -> QRect | None: ...
    def clear_template(self) -> None: ...
    # view control
    def set_mode(self, mode: "ImageViewport.Mode") -> None: ...
    def fit_in_view(self) -> None: ...
    def set_zoom(self, view_scale: float) -> None: ...
    def visible_image_rect(self) -> QRect: ...
    @property
    def view_scale(self) -> float: ...                     # viewport px per full-res image px
    @property
    def current_level(self) -> int: ...
```

**Resolved conflict (overlay payload):** `set_matches` takes `list[Match]` and converts internally to the `(N,4) int32` + `(N,) float32` numpy arrays the overlay draws from. The viewport never receives raw numpy across the public boundary.

### C.9 `pyramid.py`

```python
TILE = 256                                  # tile edge px at each level's own resolution
class Level:
    index: int; scale: float; w: int; h: int; cols: int; rows: int
class ImagePyramid:
    full_w: int; full_h: int; levels: list[Level]
    def __init__(self, doc: "ImageDocument", tile: int = TILE) -> None: ...
    def num_levels(self) -> int: ...
    def region(self, level: int, x: int, y: int, w: int, h: int) -> "np.ndarray": ...  # level-px crop, uint8
class TileCache:
    def __init__(self, max_pixmaps: int = 256, pinned_levels: int = 2) -> None: ...
    def get(self, key: tuple[int,int,int]) -> "QPixmap | None": ...   # (level,cx,cy)
    def put(self, key: tuple[int,int,int], pm: "QPixmap") -> None: ...
    def pin_top_levels(self) -> None: ...
```

### C.10 `overlay.py`

```python
class MatchOverlayItem(QGraphicsItem):
    def set_matches(self, boxes: "np.ndarray", scores: "np.ndarray") -> None: ...  # (N,4)int32,(N,)f32
    def set_threshold(self, t: float) -> None: ...    # numpy mask; triggers update(), no re-run
    def visible_count(self) -> int: ...
    def set_source_box(self, rect: "QRect | None") -> None: ...   # drawn in distinct color
```

### C.11 `params_panel.py`

```python
class ParamsPanel(QWidget):
    params_changed = Signal(object)   # MatchParams
    def __init__(self, effective_device: str, parent=None) -> None: ...
    def current_params(self) -> "MatchParams": ...
    def set_match_count(self, n: int) -> None: ...   # readout label
```

## D. ENGINE SPEC

### D.1 Fast NCC (exact)
Single **luminance** channel by default (BT.601 `L = 0.299R + 0.587G + 0.114B`); 3× less VRAM, one conv, structural similarity. `channel_mode="rgb"` sums the three numerators and three denominator terms *before* dividing (not averaging three NCCs). Image and template normalized to `[0,1]` (`/255`).

Per scale, precompute zero-mean template `Tz = T - mean(T)`, `normTz = sqrt(sum(Tz²))`, pixel count `n = th*tw`. With image patch `I` under a window at top-left `(x,y)`:

```
cross(x,y) = sum_w(I · Tz)                       # F.conv2d(I, Tz_weight)  — conv2d is cross-correlation, no flip
S1(x,y)    = sum_w(I)                             # separable box filter (ones_h then ones_w)
S2(x,y)    = sum_w(I²)                            # separable box filter on I²
n·varI     = clamp(S2 - S1²/n, min=0)            # clamp removes fp negatives
denom      = sqrt(n·varI) · normTz
ncc(x,y)   = cross / (denom + eps)               # eps=1e-6 fp32 / 1e-3 fp16
```

`ncc ∈ [-1,1]`, valid map shape `(H-th+1, W-tw+1)`. **Score remap to [0,1] for the public `Match.score`:** `score = max(0, ncc)`. Flat-region guard: where `n·varI < flat_eps` (`flat_eps = 1e-3 * n`) force score 0. Sanitize: `ncc = torch.nan_to_num(ncc, nan=-1, posinf=-1, neginf=-1)` before any thresholding.

**Separable box filter** is mandatory (O(th+tw) not O(th·tw)). The cross term stays a full 2-D conv (spatial) or FFT.

**Why box filters, not a global integral image:** float32's 24-bit mantissa makes a 40k×40k summed-area table catastrophically lose LSBs; box-filter error is bounded by window size, not image size.

### D.2 conv backend
`conv_backend="auto"`: spatial `F.conv2d` for `th*tw <= 4096` (≤~64×64), FFT (`torch.fft.rfft2`, flip template so conv==correlation, pad to 2·3·5·7-smooth sizes) above. Box sums always separable spatial conv.

### D.3 dtype / accumulators (resolved)
`compute_dtype="float32"` default and recommended. `float16` allowed only as fast-preview: conv *input* may be fp16 but **`S1`, `S2`, and the final NCC accumulate in fp32**; raise `eps` to 1e-3. `bfloat16` not offered. Debug assert: `assert torch.isfinite(ncc).all()`.

### D.4 Multi-scale (resolved)
Scale the **template** (image precompute reused). Pool candidates across all scales, then **one global NMS** so the true scale wins per location. Hit box size = `(round(tw*s), round(th*s))`. Rotation OFF by default (`rotations=None`); opt-in is `len(scales)*len(rotations)`× cost.

### D.5 Peak extraction
1. Threshold at `params.threshold_floor`.
2. On-GPU local-max prefilter: `mx = F.max_pool2d(ncc, (th,tw), stride=1, padding=(th//2,tw//2)); peaks = (ncc==mx) & (ncc>=floor)`.
3. Exclude source: drop candidates whose box IoU with `exclude_box` > `exclude_iou` (0.30), or whose center is inside it.
4. Vectorized GPU IoU-NMS (`torchvision.ops.nms` if importable, else bundled ~30-line pure-torch greedy) at `nms_iou` (0.30).
5. Top-K `max_results` (500), score desc. Move to CPU (`.cpu()`) on the worker thread, build `list[Match]`.

### D.6 Tiled compute + halo math
Process the **output grid** `(H-th+1, W-tw+1)` in non-overlapping core tiles; each tile's **input** is its core plus a halo of `(th-1, tw-1)` on right/bottom (inputs of adjacent tiles overlap by exactly template-1) so **no valid top-left position is ever split** — seamless, no border matches lost. Keep only detections whose center falls in the tile's core interior to avoid double-count; the final global NMS dedupes seam straddlers. `cancel()` polled at every tile boundary.

### D.7 VRAM budget (8 GB, resolved)
```
free, _ = torch.cuda.mem_get_info()
budget  = int(free * vram_fraction)               # vram_fraction=0.6 -> ~40% headroom
bytes_per_px = (6 if backend=="spatial" else 9) * dtype_size
C = int((budget / bytes_per_px) ** 0.5) - max(th, tw)
core = max(256, C)                                 # never below 256
```
**Conservative override / degradation ladder:** start at **core tile 1024×1024, one tile × one scale per GPU step**; wrap every GPU step in `except torch.cuda.OutOfMemoryError: empty_cache()` then degrade **1024 → 512 tile → fewer scales → CPU fallback**, with a user-visible message per rung. `empty_cache()` between *jobs*, not between tiles. The engine starts at min(computed, 1024) and only grows after a successful tile, never exceeding the formula.

### D.8 Progress / cancellation
`progress(int 0..100)` per finished tile (determinate). `cancel()` checked per tile. First query on a new image pays luminance + tiling-plan + pinned-host staging cost; warm queries reuse it.

## E. VIEWPORT SPEC

### E.1 Pyramid (resolved tile size)
Scene units == full-res image px. `PyramidItem(QGraphicsItem)` at scene origin, `boundingRect == (0,0,full_w,full_h)`. Level 0 = full res; each level halves both dims (2×2 block-mean). `L = max(1, ceil(log2(max(full_w,full_h)/TILE)) + 1)`. Levels built **lazily per tile** off the UI thread; child tiles derive from parent tiles. Source ingestion via `ImagePyramid.region(level,x,y,w,h)` (works on memmap).

**TILE = 256** for the *display* pyramid (well under the 32768 GL limit on this RTX 5060; query at runtime via `glGetIntegerv(GL_MAX_TEXTURE_SIZE)`, never assume). **A full-image single texture is forbidden.**

### E.2 LOD selection
`view_scale` = viewport px per full-res image px. `ideal = max(0, log2(1/view_scale)); level = clamp(round(ideal), 0, L-1)` with ±0.25 hysteresis to stop boundary thrash.

### E.3 LRU cache
`TileCache(max_pixmaps=256, pinned_levels=2)` ≈ 64 MB. Top 2 levels pinned (never evicted) so paint always has a coarse blurry fallback while fine tiles load.

### E.4 Pan vs draw (resolved)
**Hybrid: default mode = SELECT; hold Space = hand-pan; middle-mouse-drag always pans; wheel always zooms.** Pan driven manually via scrollbar adjustment (coexists with custom rubber-band + OpenGL viewport). Space tracked with `keyPressEvent`/`keyReleaseEvent` guarded by `if not e.isAutoRepeat()`. Cursors: Cross (select), Open/Closed-hand (pan).

### E.5 Wheel zoom
```python
def wheelEvent(self, e):
    self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    step   = 1.0015 ** e.angleDelta().y()      # angleDelta is 1/8-degree units
    target = clamp(self._view_scale*step, self._min_scale, self._max_scale)  # max_scale<=64x
    self.scale(target/self._view_scale, target/self._view_scale)
    self._view_scale = target; self._update_lod(); e.accept()   # do NOT chain to super()
```
Pan/zoom authoritative state kept in **double**; clamp max zoom 64× and recenter scene rect on the visible region.

### E.6 Overlay drawing
Single `MatchOverlayItem`, **never N `QGraphicsRectItem`**. Boxes `(N,4) int32`, scores `(N,) f32`. `paint()` culls to `opt.exposedRect` via vectorized numpy mask, draws with **cosmetic pen (width 0)** so lines stay 1 device px at any zoom, batched `p.drawRects(...)`. Threshold filter is a numpy mask → instant. Source box drawn in a distinct color. zValue above pyramid.

### E.7 Tile pixmap pipeline
Decode/downsample on `QThreadPool`/`QRunnable` → returns ready `np.ndarray`; **QImage→QPixmap creation on the GUI thread** (Qt pixmap not reliably thread-safe to create off-GUI). Trap: `QImage(np.ascontiguousarray(tile).data, w, h, arr.strides[0], QImage.Format_RGB888)` — **explicit stride required**, keep ndarray ref alive; `QPixmap.fromImage` copies so the QImage/ndarray can then be freed. Tag every tile request with a monotonic `view_generation`; discard stale worker results.

### E.8 OpenGL viewport — YES
`view.setViewport(QOpenGLWidget())`, `setViewportUpdateMode(SmartViewportUpdate)`. After async tile arrival call `viewport().update()`. Set `QApplication` attribute `Qt.AA_UseDesktopOpenGL` at startup; no MSAA. Set each `QPixmap.setDevicePixelRatio(1.0)`; trust DPR-aware `mapToScene`.

### E.9 Coordinate mapping
Chain: `screen px --mapToScene--> scene px (== full-res image px) == image px`. Engine works in scene/image px; match boxes return in scene coords → overlay draws with **no conversion**, pixel-locked at every zoom/pan.

## F. THREADING & DATA FLOW

### F.1 Model (resolved)
One persistent `QThread` owning one `MatchWorker` (via `moveToThread`); **single in-flight job, latest-wins, 120 ms debounce.** One long-lived CUDA context on exactly one thread + simple cooperative cancellation. The display-tile decoding uses a *separate* `QThreadPool` (no CUDA, GUI-thread pixmap finalization).

### F.2 Flow
```
Viewport.regionSelected(QRect)               # image px, half-open
   -> MainWindow caches current MatchParams, calls Controller.request(rect, params)
   -> Controller: validate rect, if busy -> worker.request_cancel(), arm debounce
   -> _dispatch: template = doc.full[y:y+h, x:x+w] (ascontiguousarray); job_id++; busy_changed(True)
   -> _run_requested.emit(template, params, exclude_box, job_id)   # QUEUED, plain CPU payload
   -> MatchWorker.run (worker thread): engine.match(...); finished.emit(list[Match], job_id)
   -> Controller._on_finished: if job_id==current: busy_changed(False); matches_ready.emit(matches)
   -> MainWindow: Viewport.set_matches(matches); statusbar count
```
Stale results dropped by `job_id != current`. CUDA tensors never cross a signal (`.cpu()` on worker first). Shutdown: `closeEvent -> controller.shutdown()` cancels, `thread.quit()`, `thread.wait()`.

### F.3 Coordinate contract (one canonical frame)
Integer **image px**, origin top-left, **+x right / +y down**. Box `(x,y,w,h)` is **half-open**. **Rounding rule:** float→int selection uses **floor top-left / ceil bottom-right** (box grows to cover the drag); cursor readout/hit-test uses **floor**; engine outputs are already integer. Round-trip drift ≤1px.

### F.4 Image loading
`Image.MAX_IMAGE_PIXELS = None` set at package import. `load_image`: if `w*h*3 <= max_ram_bytes` (6e9) → `np.ascontiguousarray(np.asarray(im.convert("RGB")))`; else `_load_via_memmap` decodes in 1024-row strips into a `np.memmap`. Engine streams memmap tiles to GPU; viewport strided-downsamples it for the pyramid. Pre-flight: if estimated bytes exceed a configured RAM budget, refuse with a dialog rather than OOM-kill.

### F.5 Synthetic test-image generator
`generate_sample(path,w,h,tile,n_targets,seed) -> list[Match]`: low-contrast sine-grating noise background, same distinctive asymmetric colored motif stamped at `n_targets` non-overlapping random positions (some on tile seams, some lightly scaled, half brightness-jittered ±5%), written strip-by-strip. **Returns ground-truth boxes.**

## G. EDGE-CASE REQUIREMENTS (checklist)

**Critical**
- C1 — Single `resolve_device()` gating on `torch.cuda.is_available()` (never hardcode "cuda"); CPU path must fully run.
- C1b — Launch-time canary kernel; on any exception fall back to CPU with banner.
- C2 — Engine runs on worker thread only; on CPU default to 1 scale + autoscale search so longest side ≤~4096; determinate progress + Cancel + one-time "CPU is slow" banner.
- C3 — Start at 1024 tile, one tile × one scale/step; OOM-aware retry/degrade ladder 1024→512→fewer scales→CPU; `empty_cache()` between jobs.
- C4 — Never upload full image as one texture; tiled pyramid, display tiles ≤256 (hard wall 32768, queried at runtime); memmap >RAM images; pre-flight RAM refuse dialog.
- C5 — Single worker, single in-flight, cooperative cancel token + monotonic `job_id` stale-drop; all CUDA tensors created/used on the one worker thread.

**High**
- H1 — `eps` in denominator; reject low-variance templates (`std < var_floor`) with message; `nan_to_num` score maps before NMS.
- H2 — Accumulators + final NCC always fp32 even in fp16 mode; debug `assert torch.isfinite`.
- H3 — Validate selection in image px: clip to image, reject `< 8px` side, reject ≥ whole image, reject aspect > 20; tile = `max(base, next_pow2(template_side + 2*halo))` so template never exceeds tile.
- H4 — Exclude source region (IoU > 0.30 or center inside) in full-image coords; state exclusion in UI.
- H5 — Halo = template−1 overlap on every side; keep only detections centered in core interior.
- H6 — QImage from numpy: `ascontiguousarray` + explicit `strides[0]` bytesPerLine + keep buffer alive until `QPixmap.fromImage` copies.
- H7 — Built-in self-test (synthesize → stamp N → recover all centers ±1px, source excluded); "Self-test" menu action.

**Medium**
- M1 — One canonical coord space; floor-TL/ceil-BR mapping; ≤1px round-trip.
- M2 — Honor `devicePixelRatioF()` consistently; never hand-roll zoom math.
- M3 — Pan/zoom state in double; clamp zoom ≤64×; recenter scene rect; tile-based off-screen culling.
- M4 — Defaults threshold 0.85 / NMS 0.30 / cap 500; live threshold slider w/ result count; document rotation unsupported + out-of-grid scales missed.
- M5 — closeEvent: cancel all jobs, `thread.quit()`, `thread.wait(timeout)`; free CUDA tensors on worker first; slot guards (job_id + widget-alive).
- M6 — (canary) log exact error, CPU fallback, point at cu128 index-url.
- M7 — Workers emit only plain CPU data; `.cpu().numpy()` before emit.

## H. DEFAULTS

| Setting | Default | Notes |
|---|---|---|
| `threshold` (UI live filter) | **0.85** | stricter cuts repetitive-texture floods |
| `threshold_floor` (engine returns ≥) | **0.50** | UI filters up without re-running engine |
| `scales` (GPU) | **(0.8, 0.9, 1.0, 1.1, 1.25)** | full grid when device==cuda |
| `scales` (CPU) | **(1.0,)** | 1 scale + autoscale ≤4096 |
| `rotations` | **None** (off) | opt-in; documented cost |
| `max_results` | **500** | cap after NMS |
| `nms_iou` | **0.30** | |
| `exclude_iou` | **0.30** | source-region exclusion |
| `compute_dtype` | **float32** | accumulators always fp32 |
| `channel_mode` | **luminance** | |
| `conv_backend` | **auto** | spatial ≤64×64, FFT above |
| `vram_fraction` | **0.6** | ceiling; engine starts at min(computed, 1024) tile |
| compute tile (start) | **1024×1024** | degrade 1024→512→fewer scales→CPU |
| min template side | **8 px** | reject below |
| max zoom | **64×** | |
| display TILE | **256** | well under 32768 GL cap |
| `TileCache` size | **256 pixmaps** (~64 MB), pin top **2** levels | |
| debounce | **120 ms** | |
| `max_ram_bytes` (loader→memmap) | **6e9** | |
| `var_floor` (reject template) | **1e-3** on [0,1] | |
| `eps` | **1e-6 fp32 / 1e-3 fp16** | |
| sample image | **12000×12000**, motif 200, 40 targets | |

## I. CITATIONS
PyTorch sm_120/cu128 support (issues #164342, #159207); NumPy ndarray↔QImage stride/bytesPerLine trap; Qt 6.11 QImage docs.

---

# J. EXTENSION — Selectable Matching Methods

User requirement: support multiple, user-selectable matching methods, because a "perfectly flat" image
and a "warped" image suit different methods. `MatchParams` is already extended (frozen, done):
`method ∈ METHODS = ("ncc","ssd","ccorr","features")` plus `feature_detector`, `feature_ratio`,
`feature_min_inliers`, `feature_max_instances`. `CONV_METHODS = {"ncc","ssd","ccorr"}`.

### J.1 Method guidance (also the UI tooltips, from `METHOD_LABELS`)
- **ncc** — normalized cross-correlation (CCOEFF). Default. Illumination-robust; needs texture; aligned, same-scale instances. (GPU)
- **ssd** — normalized squared difference. **Best for perfectly flat / low-texture / exact-appearance** regions where NCC's variance normalization is unstable. NOT illumination-invariant. (GPU)
- **ccorr** — cosine cross-correlation (CCORR_NORMED). Alternative correlation measure. (GPU)
- **features** — ORB keypoints + RANSAC homography, multi-instance. **Tolerant to rotation / scale / perspective warp** that template methods miss. (CPU via OpenCV)

### J.2 Conv-method score maps (ncc/ssd/ccorr share ALL existing machinery)
The tiled compute, halo, multi-scale, local-max prefilter, threshold_floor, source-exclusion, IoU-NMS,
top-K, cancel/progress are **unchanged**. ONLY the per-window score formula is dispatched on `method`.
Refactor the per-window scoring into a single dispatched helper; keep **NCC numerically identical** so the
baseline tests still pass.

Pixels normalized to `[0,1]`. Per window (n = th·tw px per channel), with the quantities the engine
already computes: `cross_z = conv(I, Tz)` (zero-mean template), `S1 = Σ_w I`, `S2 = Σ_w I²`, and template
scalars `meanT`, `normTz = sqrt(Σ Tz²)`, `sumT2 = Σ T²`. Derive the raw cross **without a second
convolution**: `cross_raw = cross_z + meanT·S1` (since `T = Tz + meanT`).
```
n·varI = clamp(S2 - S1²/n, min=0)
ncc    : coeff = cross_z / (sqrt(n·varI)·normTz + eps);          score = max(0, coeff)            # == current
ccorr  : cc    = cross_raw / (sqrt(S2)·sqrt(sumT2) + eps);       score = clamp(cc, 0, 1)
ssd    : SSD   = clamp(S2 - 2·cross_raw + sumT2, min=0)          # = Σ_w (I-T)²
         rmse  = sqrt(SSD / n);                                   score = clamp(1 - rmse, 0, 1)    # flat-friendly
```
RGB mode: sum the per-channel terms (cross_z/cross_raw, S1, S2, sumT2, normTz²) **before** the single
division (ncc/ccorr); for ssd sum SSD across channels then divide by `n·channels`. Flat-region guard, eps
(1e-6 fp32 / 1e-3 fp16), and `nan_to_num` exactly as §D.1. SSD/CCORR do NOT raise on low-variance templates
(that's their use case) — only NCC keeps the variance-floor ValueError.

### J.3 Feature matching — `fastmatch/feature_matcher.py` (new), used when `method=="features"`
Backend: **OpenCV (cv2)**. REQUIRED only for this method; if cv2 is not importable, `match()` raises
`RuntimeError("Feature matching requires OpenCV: pip install opencv-python-headless")` and the UI disables
the option. Convolution methods never import cv2.

Pipeline (multi-instance, warp-tolerant):
1. **Detection scale**: detect at **full resolution** so a template selected at native scale keeps all its
   keypoints. (An earlier design downsampled image *and template* by a shared factor to bound cost; that
   decimated any modest selection below ORB's keypoint threshold and made feature matching find nothing on
   large images — the bug.) Bound cost/memory instead by **tiling** detection (`_DETECT_TILE` with
   `_DETECT_OVERLAP`, each tile given `_FEATURES_PER_TILE` so keypoints stay spatially dense; each keypoint
   assigned to one tile core so seams produce no duplicate descriptors). Boxes come out in full-res px
   directly. Cache the full-res keypoints/descriptors on the Matcher, keyed by detector; invalidate in
   `set_image`. The template is detected at full resolution too (denser `_TEMPLATE_FEATURES` budget).
2. **Detector** per `feature_detector`: ORB (`cv2.ORB_create`, default), AKAZE, or SIFT. Grayscale (reuse
   staged luminance). Compute IMAGE kpts/desc once per (image, detector) and cache.
3. **Per query**: detect TEMPLATE kpts/desc; `knnMatch(k=2)` template→image with the right norm
   (HAMMING for ORB/AKAZE, L2 for SIFT); Lowe ratio test (`feature_ratio`).
4. **Sequential RANSAC homography**: loop — `cv2.findHomography(src_tmpl_pts, dst_img_pts, cv2.RANSAC,
   reproj_thresh)`; if inliers ≥ `feature_min_inliers` and H passes degeneracy guards (affine-part det not
   near-zero/negative, projected template-corner quad convex with positive, sane area — reject scale
   >1000× or <1/1000×), project the template's 4 corners → quad → axis-aligned bbox (clamp to image, ×f to
   full-res) → record a `Match`; remove inlier matches; repeat until matches < min or `feature_max_instances`.
5. **Score → [0,1]**: `score = min(1.0, inliers / (2·feature_min_inliers))` (identical instances ≈ 1.0,
   ambiguous ones lower). Document if you choose a different monotonic map.
6. **Exclude source** (IoU > `exclude_iou` or center inside `exclude_box`) like the conv path.
7. `cancel()` checked during detection and between RANSAC iterations; `progress` ≈ detection (0–50) +
   instance loop (50–100). `scale ≈ sqrt(bbox_area / template_area)`.

Feature matching ignores `params.scales` and `channel_mode` (inherently scale/rotation-tolerant,
grayscale) and runs on CPU even when conv methods use CUDA — document this.

### J.4 Engine integration (`engine.py`, edited by the same owner)
`Matcher.match()` branches at the top: `params.method in CONV_METHODS` → existing tiled path with the
dispatched `score_map`; `== "features"` → delegate to a lazily-constructed feature matcher (image-feature
cache on the Matcher, invalidated in `set_image`). `effective_device` unchanged.

### J.5 UI (`params_panel.py`, `app.py`)
- `ParamsPanel`: a **Method** `QComboBox` at top from `METHODS`/`METHOD_LABELS`; emits
  `params_changed(MatchParams(method=...))`. Show scales/channel controls only for conv methods; show
  detector (+ min-inliers) only for features (QStackedWidget or show/hide). Disable the "features" item
  (grey + tooltip "Install opencv-python-headless") if cv2 is not importable (detect at construction).
  On switch to features default the threshold slider to **0.5**; conv methods **0.85**.
- `app.py`: a method change is engine-relevant → re-issue the last selection's search. The threshold slider
  stays a live client-side filter for all methods.

### J.6 Tests
- Extend `tests/test_engine.py`: SSD and CCORR recall on the stamped-motif image (exact copies score ≈1);
  a flat/low-variance case where SSD recovers instances. Keep NCC tests unchanged (regression).
- New `tests/test_features.py`: `pytest.importorskip("cv2")`; image with an upright motif AND a
  rotated(+scaled) copy; `method="features"` recovers BOTH (esp. the rotated one) with source excluded;
  contrast that plain NCC misses the rotated copy. CPU-fast.

### J.7 requirements / README
- `requirements.txt`: add `opencv-python-headless` with a comment that it powers the "features" method only
  (and `-headless` avoids Qt-plugin clashes with PySide6).
- `README.md`: document the 4 methods + when to use each (flat→SSD, textured→NCC, warped/rotated→features),
  that conv methods are GPU and feature matching is CPU via OpenCV, and the new method dropdown.

---

# K. PERFORMANCE — Image-Pyramid Optimization (per mode)

Matching cost scales ~linearly with image pixels (baseline RTX 5060: NCC×1 855 ms, NCC×5 4276 ms, SSD/CCORR
~1.7 s at 8000²; ×~50 on CPU). Each mode gets an image-pyramid optimization. **Hard rule: recall parity** —
the optimized path must recover the SAME true instances the full-resolution path does (verified scene-by-
scene against the full search); it trades a tiny, bounded precision/recall margin only where provably safe,
and falls back to the full path when the pyramid can't help (so existing tests/behaviour never regress).

### K.1 Conv modes (ncc/ssd/ccorr): coarse-to-fine search

The full search computes the score map over every full-res position. Coarse-to-fine instead localizes
candidates cheaply at a downsampled level, then refines only those candidates at finer levels.

**Compute image pyramid** (in `engine.set_image`, on the compute device, cached, reused across queries):
levels ℓ=0..K, level ℓ = the staged luminance (and rgb planes, if staged) downsampled by 2^ℓ via
`F.avg_pool2d(2)`. Memory overhead ≈ 1/3 of the base. Build lazily/bounded so it never blows the VRAM
budget (it participates in the OOM ladder; on OOM, skip pyramid and use the full path).

**Per (template, scale) algorithm:**
1. **Level count** `K = clamp(floor(log2(min(sth,stw) / MIN_COARSE_SIDE)), 0, MAX_PYR_LEVELS)`
   (`MIN_COARSE_SIDE≈16`, `MAX_PYR_LEVELS≈4`). If `K==0` (template/image too small) → use the existing
   full-res `_search_all_scales` path (no behaviour change; this is what the small-image tests exercise).
2. **Coarse search** at level K: downsample the template to level K (scale·2^-K), compute the FULL score map
   on the (tiny) level-K image via the existing per-tile machinery, local-max + threshold at a **relaxed**
   `coarse_thr = max(0, min(threshold, threshold_floor) - COARSE_MARGIN)` (`COARSE_MARGIN≈0.20` so a true
   peak that looks weaker when blurred is NOT pruned). Collect candidate top-left positions.
3. **Refine** ℓ=K-1…0: map each candidate ×2; around the mapped position open a search window of
   `±REFINE_RADIUS` (≈3 px, absorbs the ×2 + downsample-rounding uncertainty); **batch-extract** the
   level-ℓ ROIs (each `window + (sth_ℓ-1)` so the conv is valid), stack as `(N,1,Hroi,Wroi)`, and compute
   the score map for all ROIs in ONE batched `F.conv2d` (+ batched box filters). Keep local-max peaks ≥ the
   level threshold; dedupe candidates landing in the same cell. These become the next level's candidates.
4. **Level 0** peaks are the final candidates (exact positions + scores) → feed the existing
   `_finalize` (source-exclusion, IoU-NMS, top-K), identical to the full path.
5. **Multi-scale**: run coarse-to-fine per requested scale; pool all candidates; ONE global NMS (as today).

**Refactor enabling the ROI batch:** split the per-window score into (a) term computation (cross_z via
conv, S1/S2 via box filters — done for a tile OR a batch of ROIs) and (b) a pure-elementwise
`_score_from_terms(method, cross_z, S1, S2, n, meanT, normTz, sumT2, …)` that is shape-agnostic, so both
the tiled full path and the batched ROI path reuse the exact same NCC/SSD/CCORR formula (§J.2). NCC stays
numerically identical on the full path.

**Performance kernel:** the ROI refinement MUST be batched (one conv over N ROIs), not a Python per-
candidate loop, so refinement cost is ~O(#candidates · template²) on the GPU regardless of image size. The
coarse full search is ~O(pixels/4^K). Target: NCC×5 @8000² from ~4.3 s to < ~0.8 s; NCC×1 @8000² < ~0.2 s;
similar or better on CPU; **recall identical** to full search on the test scenes.

**Safeguards:** keep `_search_all_scales` (full) intact as the fallback and the recall reference; an internal
toggle/flag forces the full path so a test can assert pyramid-vs-full recall parity. A forced `core_tile`
(used by the seam test) takes the full path. The coarse margin is conservative; if a benchmark scene shows
the pyramid missing a true instance the full path finds, widen `COARSE_MARGIN` / raise `MIN_COARSE_SIDE`.

### K.2 Feature mode: coarsest-safe-level detection pyramid

The feature path's cost is dominated by global image detection. Detecting at FULL res over a gigapixel image
is slow AND produces many distractor descriptors (which hurt the Lowe ratio test → missed warped instances).
Optimization: detect at the **coarsest pyramid level that still keeps the template usable**.

- **Level selection from the template:** `L = clamp(floor(log2(min(th,tw) / MIN_TPL_FEATURE)), 0,
  MAX_FEATURE_LEVEL)` with `MIN_TPL_FEATURE≈96` — the coarsest level where the downsampled template still has
  ≥ ~96 px (enough ORB keypoints). `L=0` reproduces today's full-res tiled detection (so small templates /
  small images are unchanged).
- **Per-level cached tiled detection:** detect the image downsampled by `2^L` (INTER_AREA), tiled with the
  per-tile budget as today, cached per `(detector, L)`; keypoints stored in full-res coords (×2^L).
- **Per query:** pick L from the template, downsample the template by `2^L`, detect/match/RANSAC at that
  level, map boxes back ×2^L. Fewer features at level L ⇒ faster knnMatch + RANSAC AND fewer distractors
  ⇒ **equal-or-better recall** (this should also help the gigapixel rotated-instance case).
- **Safety:** the level is bounded so the template never decimates below `MIN_TPL_FEATURE` (this is the exact
  failure that made feature matching return nothing before — do NOT reintroduce it). Verify recall parity
  vs L=0 on the rotated-instance scene at several image sizes.

### K.3 Verification (both modes)
- **Recall parity:** on a battery of scenes (stamped motifs at known locations; rotated/scaled copies; tile
  seams; flat regions), assert the optimized path recovers every true instance the full path recovers
  (IoU ≥ 0.7), with precision no worse beyond a small tolerance.
- **Benchmarks:** report before/after match() time per mode at 4000²/8000² (GPU) and a CPU spot-check; the
  optimization must be substantially faster (target ≥ 3× on NCC×5, ≥ 2× elsewhere) with no recall loss.
- The existing 30 tests must still pass; add tests asserting pyramid-vs-full recall parity on a larger scene
  and that `core_tile`/small-image paths still work.
