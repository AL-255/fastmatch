"use strict";
/*
 * FastMatch — browser edition front-end.
 *
 * Boots Pyodide + NumPy, loads the pure-NumPy engine (fastmatch_web.py), and
 * wires an HTML5-canvas viewport (pan / zoom-to-cursor / draw selection) to it.
 * Everything runs client-side; no server, so it hosts as static files on
 * GitHub Pages. See README for the matching algorithm (it mirrors the desktop
 * app's full-resolution NCC/SSD/CCORR path).
 */

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");

const state = {
  pyodide: null,
  ready: false,
  busy: false,
  bitmap: null,        // ImageBitmap for fast rendering
  imgW: 0,
  imgH: 0,
  view: { scale: 1, ox: 0, oy: 0 },
  mode: "select",       // "select" | "pan"
  selection: null,      // {x,y,w,h} in image px
  matches: [],
  dragging: null,       // {kind, ...}
};

// ----------------------------------------------------------------- rendering
function dpr() { return window.devicePixelRatio || 1; }

function resizeCanvas() {
  const r = canvas.getBoundingClientRect();
  const d = dpr();
  canvas.width = Math.max(1, Math.round(r.width * d));
  canvas.height = Math.max(1, Math.round(r.height * d));
  render();
}

function screenToImage(clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  const v = state.view;
  return {
    x: (clientX - r.left - v.ox) / v.scale,
    y: (clientY - r.top - v.oy) / v.scale,
  };
}

function fitView() {
  if (!state.bitmap) return;
  const r = canvas.getBoundingClientRect();
  const s = Math.min(r.width / state.imgW, r.height / state.imgH) * 0.96;
  state.view.scale = s > 0 ? s : 1;
  state.view.ox = (r.width - state.imgW * state.view.scale) / 2;
  state.view.oy = (r.height - state.imgH * state.view.scale) / 2;
  render();
}

function render() {
  const d = dpr();
  const v = state.view;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.bitmap) return;

  // image -> backing-store px (folds in devicePixelRatio).
  ctx.setTransform(d * v.scale, 0, 0, d * v.scale, d * v.ox, d * v.oy);
  ctx.imageSmoothingEnabled = v.scale < 3;
  ctx.drawImage(state.bitmap, 0, 0);

  const lw = 1.5 / v.scale; // constant ~1.5 CSS px outline regardless of zoom

  // matches: green
  ctx.lineWidth = lw;
  ctx.strokeStyle = "rgba(40,220,70,0.95)";
  for (const m of state.matches) ctx.strokeRect(m.x, m.y, m.w, m.h);

  // selection (source): cyan
  if (state.selection) {
    const s = state.selection;
    ctx.lineWidth = lw * 1.4;
    ctx.strokeStyle = "rgba(0,200,255,0.98)";
    ctx.strokeRect(s.x, s.y, s.w, s.h);
  }
}

// ---------------------------------------------------------------- readout/UI
function setHint(text) {
  const h = $("hint");
  if (text === null) { h.classList.add("hidden"); return; }
  h.classList.remove("hidden");
  h.innerHTML = text;
}
function setReadout(text) { $("readout").textContent = text || ""; }
function setBusy(b) {
  state.busy = b;
  $("topbar").classList.toggle("busy", b);
  updateRunEnabled();
}
function updateRunEnabled() {
  $("runBtn").disabled = !(state.ready && state.bitmap && state.selection && !state.busy);
}

// ------------------------------------------------------------ image loading
async function loadImageFromBlobOrURL(src) {
  let blob;
  if (typeof src === "string") {
    const resp = await fetch(src);
    if (!resp.ok) throw new Error(`could not fetch ${src}`);
    blob = await resp.blob();
  } else {
    blob = src;
  }
  const bitmap = await createImageBitmap(blob);
  state.bitmap = bitmap;
  state.imgW = bitmap.width;
  state.imgH = bitmap.height;
  state.selection = null;
  state.matches = [];
  $("matchCount").textContent = "0";
  $("imgInfo").textContent = `${bitmap.width}×${bitmap.height}px`;

  // Extract RGBA pixels and hand them to the Python engine once.
  const off = document.createElement("canvas");
  off.width = bitmap.width;
  off.height = bitmap.height;
  const octx = off.getContext("2d", { willReadFrequently: true });
  octx.drawImage(bitmap, 0, 0);
  const id = octx.getImageData(0, 0, bitmap.width, bitmap.height);
  if (state.ready) {
    state.pyodide.globals.set("_imgbuf", new Uint8Array(id.data.buffer));
    await state.pyodide.runPythonAsync(`fm_set_image(_imgbuf, ${bitmap.width}, ${bitmap.height})`);
    state.pyodide.globals.delete("_imgbuf");
  }
  setHint(state.ready ? null : "Engine still loading…");
  fitView();
  updateRunEnabled();
}

// --------------------------------------------------------------- parameters
function gatherParams() {
  const s = state.selection;
  const multiscale = $("multiscale").checked;
  return {
    x: Math.round(s.x), y: Math.round(s.y),
    w: Math.round(s.w), h: Math.round(s.h),
    method: $("method").value,
    channel: $("channel").value,
    threshold: parseFloat($("threshold").value),
    maxResults: parseInt($("maxResults").value, 10),
    scales: multiscale ? [0.8, 0.9, 1.0, 1.1, 1.25] : [1.0],
    rotation: $("rotation").checked,
    flipping: $("flipping").checked,
  };
}

async function runMatch() {
  if (!state.ready || !state.bitmap || !state.selection || state.busy) return;
  setBusy(true);
  setReadout("matching…");
  const t0 = performance.now();
  try {
    const params = gatherParams();
    const fn = state.pyodide.globals.get("fm_match");
    const json = await fn(JSON.stringify(params));
    fn.destroy?.();
    state.matches = JSON.parse(json);
    $("matchCount").textContent = String(state.matches.length);
    const ms = Math.round(performance.now() - t0);
    setReadout(`${state.matches.length} matches · ${ms} ms`);
    render();
  } catch (err) {
    console.error(err);
    setReadout("error: " + (err && err.message ? err.message : err));
  } finally {
    setBusy(false);
  }
}

function autoRun() {
  if ($("autorun").checked) runMatch();
}

// --------------------------------------------------------------- interaction
function setMode(mode) {
  state.mode = mode;
  $("modeBtn").textContent = "Mode: " + (mode === "pan" ? "Pan" : "Select");
  canvas.classList.toggle("pan", mode === "pan");
}

canvas.addEventListener("wheel", (e) => {
  if (!state.bitmap) return;
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  const cx = e.clientX - r.left, cy = e.clientY - r.top;
  const before = screenToImage(e.clientX, e.clientY);
  const factor = Math.exp(-e.deltaY * 0.0015);
  const v = state.view;
  v.scale = Math.min(64, Math.max(0.02, v.scale * factor));
  // keep the image point under the cursor fixed (zoom-to-cursor).
  v.ox = cx - before.x * v.scale;
  v.oy = cy - before.y * v.scale;
  render();
}, { passive: false });

canvas.addEventListener("pointerdown", (e) => {
  if (!state.bitmap) return;
  canvas.setPointerCapture(e.pointerId);
  const panning = state.mode === "pan" || e.button === 1 || e.shiftKey;
  if (panning) {
    state.dragging = { kind: "pan", sx: e.clientX, sy: e.clientY, ox: state.view.ox, oy: state.view.oy };
    canvas.classList.add("panning");
  } else if (e.button === 0) {
    const p = screenToImage(e.clientX, e.clientY);
    state.dragging = { kind: "select", x0: p.x, y0: p.y };
    state.matches = [];
    $("matchCount").textContent = "0";
  }
});

canvas.addEventListener("pointermove", (e) => {
  const p = screenToImage(e.clientX, e.clientY);
  if (state.bitmap) {
    setReadout(`(${Math.max(0, Math.min(state.imgW - 1, Math.round(p.x)))}, ` +
               `${Math.max(0, Math.min(state.imgH - 1, Math.round(p.y)))})`);
  }
  const dr = state.dragging;
  if (!dr) return;
  if (dr.kind === "pan") {
    state.view.ox = dr.ox + (e.clientX - dr.sx);
    state.view.oy = dr.oy + (e.clientY - dr.sy);
    render();
  } else if (dr.kind === "select") {
    const x = Math.max(0, Math.min(state.imgW, Math.min(dr.x0, p.x)));
    const y = Math.max(0, Math.min(state.imgH, Math.min(dr.y0, p.y)));
    const x2 = Math.max(0, Math.min(state.imgW, Math.max(dr.x0, p.x)));
    const y2 = Math.max(0, Math.min(state.imgH, Math.max(dr.y0, p.y)));
    state.selection = { x, y, w: x2 - x, h: y2 - y };
    render();
  }
});

canvas.addEventListener("pointerup", (e) => {
  const dr = state.dragging;
  state.dragging = null;
  canvas.classList.remove("panning");
  if (!dr) return;
  if (dr.kind === "select") {
    if (state.selection && state.selection.w >= 4 && state.selection.h >= 4) {
      updateRunEnabled();
      autoRun();
    } else {
      state.selection = null;
      render();
      updateRunEnabled();
    }
  }
});

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "m" || e.key === "M") setMode(state.mode === "pan" ? "select" : "pan");
  else if (e.key === "f" || e.key === "F") fitView();
});

// ------------------------------------------------------------ control wiring
$("file").addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) loadImageFromBlobOrURL(f).catch((err) => setReadout("load error: " + err.message));
});
$("sampleBtn").addEventListener("click", () =>
  loadImageFromBlobOrURL("sample.png").catch((err) => setReadout("sample missing: " + err.message)));
$("modeBtn").addEventListener("click", () => setMode(state.mode === "pan" ? "select" : "pan"));
$("runBtn").addEventListener("click", runMatch);
$("clearBtn").addEventListener("click", () => {
  state.selection = null; state.matches = [];
  $("matchCount").textContent = "0"; setReadout(""); render(); updateRunEnabled();
});

$("threshold").addEventListener("input", () => { $("thVal").textContent = $("threshold").value; });
$("threshold").addEventListener("change", autoRun);
$("maxResults").addEventListener("input", () => { $("mrVal").textContent = $("maxResults").value; });
$("maxResults").addEventListener("change", autoRun);
for (const id of ["method", "channel"]) $(id).addEventListener("change", autoRun);
for (const id of ["multiscale", "rotation", "flipping"]) $(id).addEventListener("change", autoRun);

new ResizeObserver(resizeCanvas).observe($("stage"));

// ------------------------------------------------------------ engine bootstrap
async function boot() {
  try {
    state.pyodide = await loadPyodide();
    setHint("Loading NumPy…");
    await state.pyodide.loadPackage("numpy");
    setHint("Loading the matching engine…");
    const src = await (await fetch("fastmatch_web.py")).text();
    state.pyodide.FS.writeFile("fastmatch_web.py", src);
    await state.pyodide.runPythonAsync(`
import json, numpy as np
import fastmatch_web as fw
_IMG = None
def fm_set_image(buf, w, h):
    global _IMG
    if hasattr(buf, "to_py"):
        buf = buf.to_py()   # JsProxy(Uint8Array) -> memoryview
    _IMG = np.frombuffer(buf, dtype=np.uint8).reshape(int(h), int(w), 4).copy()
def fm_match(params_json):
    if _IMG is None:
        return "[]"
    p = json.loads(params_json)
    res = fw.match(
        _IMG, p["x"], p["y"], p["w"], p["h"],
        method=p["method"], channel_mode=p["channel"],
        threshold=p["threshold"], scales=tuple(p["scales"]),
        enable_rotation=p["rotation"], enable_flipping=p["flipping"],
        max_results=p["maxResults"],
    )
    return json.dumps(res)
`);
    state.ready = true;
    setHint(state.bitmap ? null :
      "Engine ready. <b>Open an image</b> or <b>Load sample</b>, then drag a box around a pattern.");
    // If an image was loaded before the engine finished, push its pixels now.
    if (state.bitmap) await loadImageFromBlobOrURL(await bitmapToBlob(state.bitmap));
    updateRunEnabled();
  } catch (err) {
    console.error(err);
    setHint("Failed to load the Python engine.<br>" + (err && err.message ? err.message : err));
  }
}

async function bitmapToBlob(bitmap) {
  const c = document.createElement("canvas");
  c.width = bitmap.width; c.height = bitmap.height;
  c.getContext("2d").drawImage(bitmap, 0, 0);
  return await new Promise((res) => c.toBlob(res, "image/png"));
}

setMode("select");
resizeCanvas();
boot();
