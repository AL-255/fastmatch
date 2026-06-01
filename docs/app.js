"use strict";
/*
 * FastMatch — browser edition front-end.
 *
 * The matching runs in a POOL of Web Workers (one Pyodide + NumPy per core) so
 * it uses multiple cores AND stays off the UI thread — that is what makes the
 * spinner animate and lets a search be interrupted. The work is split into
 * (orientation x scale) tasks dispatched across the pool; the page pools the
 * candidates and runs the final greedy NMS / source-exclusion / top-K here.
 * A monotonic "generation" token cancels an in-flight search the instant a new
 * selection is drawn or a parameter changes.
 */

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");

const POOL_SIZE = Math.max(1, Math.min((navigator.hardwareConcurrency || 4), 6));
const ORIENTATION_ORDER = ["R0", "R90", "R180", "R270", "MX", "MY", "MXR90", "MYR90"];

const state = {
  poolReady: false,
  imageReady: false,
  bitmap: null,
  rgbaBuf: null,        // ArrayBuffer of RGBA pixels (cloned to each worker)
  imgW: 0,
  imgH: 0,
  view: { scale: 1, ox: 0, oy: 0 },
  mode: "select",
  selection: null,
  matches: [],
  dragging: null,
};

const workers = [];     // {worker, ready, busy, readyResolve, imgResolve}
let currentGen = 0;     // bumping this cancels any in-flight search
let genState = null;    // {gen, tasks, params, next, pending, collected, resolve}

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
  return { x: (clientX - r.left - v.ox) / v.scale, y: (clientY - r.top - v.oy) / v.scale };
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

  ctx.setTransform(d * v.scale, 0, 0, d * v.scale, d * v.ox, d * v.oy);
  ctx.imageSmoothingEnabled = v.scale < 3;
  ctx.drawImage(state.bitmap, 0, 0);

  const lw = 1.5 / v.scale;
  ctx.lineWidth = lw;
  ctx.strokeStyle = "rgba(40,220,70,0.95)";
  for (const m of state.matches) ctx.strokeRect(m.x, m.y, m.w, m.h);

  if (state.selection) {
    const s = state.selection;
    ctx.lineWidth = lw * 1.4;
    ctx.strokeStyle = "rgba(0,200,255,0.98)";
    ctx.strokeRect(s.x, s.y, s.w, s.h);
  }
}

// ---------------------------------------------------------------- UI helpers
function setHint(text) {
  const h = $("hint");
  if (text === null) { h.classList.add("hidden"); return; }
  h.classList.remove("hidden");
  h.innerHTML = text;
}
function setReadout(text) { $("readout").textContent = text || ""; }
function showSpinner(on, label) {
  $("spinner").classList.toggle("hidden", !on);
  if (label) $("spinLabel").textContent = label;
}
function updateRunEnabled() {
  $("runBtn").disabled = !(state.poolReady && state.imageReady && state.selection);
}

// --------------------------------------------------------------- worker pool
function handleWorkerMessage(wk, msg) {
  if (msg.type === "ready") { wk.readyResolve && wk.readyResolve(); }
  else if (msg.type === "imageAck") { wk.imgResolve && wk.imgResolve(); }
  else if (msg.type === "taskDone") {
    if (msg.error) console.warn("[worker task error]", msg.error);
    onTaskDone(wk, msg);
  }
}

async function bootPool() {
  setHint(`Loading ${POOL_SIZE} Python engine${POOL_SIZE > 1 ? "s" : ""}… ` +
          `first load fetches Pyodide + NumPy (~10&nbsp;MB, cached after).`);
  const readies = [];
  for (let i = 0; i < POOL_SIZE; i++) {
    const worker = new Worker("worker.js");
    const wk = { worker, ready: false, busy: false, readyResolve: null, imgResolve: null };
    workers.push(wk);
    worker.onmessage = (e) => handleWorkerMessage(wk, e.data);
    worker.onerror = (e) => { console.error("worker error", e.message || e); };
    readies.push(new Promise((res) => { wk.readyResolve = res; }));
    worker.postMessage({ type: "init" });
  }
  await Promise.all(readies);
  workers.forEach((wk) => { wk.ready = true; });
  state.poolReady = true;
}

async function setImageOnWorkers() {
  if (!state.poolReady || !state.rgbaBuf) return;
  state.imageReady = false;
  const acks = workers.map((wk) => new Promise((res) => { wk.imgResolve = res; }));
  for (const wk of workers) {
    // Clone the buffer per worker (no transfer) so each holds its own copy.
    wk.worker.postMessage({ type: "image", buf: state.rgbaBuf.slice(0), w: state.imgW, h: state.imgH });
  }
  await Promise.all(acks);
  state.imageReady = true;
}

// ------------------------------------------------------ dispatch / cancel
function startMatch(tasks, params) {
  const gen = ++currentGen;
  if (genState) { const old = genState; genState = null; old.resolve(old.collected); } // supersede
  return new Promise((resolve) => {
    genState = { gen, tasks, params, next: 0, pending: 0, collected: [], resolve };
    pump();
  });
}

function pump() {
  const gs = genState;
  if (!gs) return;
  for (const wk of workers) {
    if (!wk.ready || wk.busy) continue;
    if (gs.next >= gs.tasks.length) break;
    const [orient, scale] = gs.tasks[gs.next++];
    wk.busy = true;
    gs.pending++;
    const s = gs.params.sel;
    wk.worker.postMessage({
      type: "task",
      gen: gs.gen,
      params: {
        x: s.x, y: s.y, w: s.w, h: s.h,
        method: gs.params.method, channel: gs.params.channel,
        threshold: gs.params.threshold, scale, orient, cap: gs.params.cap,
      },
    });
  }
  if (gs.next >= gs.tasks.length && gs.pending === 0) {
    const resolve = gs.resolve, collected = gs.collected;
    genState = null;
    resolve(collected);
  }
}

function onTaskDone(wk, msg) {
  wk.busy = false;
  const gs = genState;
  if (gs && msg.gen === gs.gen) {
    if (msg.candidates && msg.candidates.length) gs.collected.push(...msg.candidates);
    gs.pending--;
  }
  pump(); // dispatch the next queued task to this now-idle worker (any generation)
}

function cancelMatch() {
  const had = !!genState;
  currentGen++;                 // in-flight worker results become stale
  if (genState) { const r = genState.resolve, c = genState.collected; genState = null; r(c); }
  showSpinner(false);
  if (had) setReadout("cancelled");
}

// ------------------------------------------------------------- finalize (JS)
function iouBox(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy, uni = a.w * a.h + b.w * b.h - inter;
  return uni > 0 ? inter / uni : 0;
}

function finalize(cands, sel, nmsIou, excludeIou, maxResults) {
  if (!cands.length) return [];
  const order = cands.map((_, i) => i).sort((a, b) => cands[b].score - cands[a].score);
  const kept = [];
  for (const i of order) {
    const c = cands[i];
    let suppressed = false;
    for (const k of kept) { if (iouBox(c, cands[k]) > nmsIou) { suppressed = true; break; } }
    if (!suppressed) kept.push(i);
  }
  const out = [];
  for (const i of kept) {
    const c = cands[i], cx = c.x + c.w / 2, cy = c.y + c.h / 2;
    const inSrc = sel.x <= cx && cx < sel.x + sel.w && sel.y <= cy && cy < sel.y + sel.h;
    if (inSrc || iouBox(c, sel) > excludeIou) continue;
    out.push(c);
    if (out.length >= maxResults) break;
  }
  return out;
}

// --------------------------------------------------------------- the search
function activeOrientations(rot, flip) {
  const a = new Set(["R0"]);
  if (rot) { a.add("R90"); a.add("R180"); a.add("R270"); }
  if (flip) { a.add("MX"); a.add("MY"); }
  if (rot && flip) { a.add("MXR90"); a.add("MYR90"); }
  return ORIENTATION_ORDER.filter((o) => a.has(o));
}

function gatherParams() {
  const s = state.selection;
  const maxResults = parseInt($("maxResults").value, 10);
  const scales = $("multiscale").checked ? [0.8, 0.9, 1.0, 1.1, 1.25] : [1.0];
  const orients = activeOrientations($("rotation").checked, $("flipping").checked);
  return {
    sel: { x: Math.round(s.x), y: Math.round(s.y), w: Math.round(s.w), h: Math.round(s.h) },
    method: $("method").value,
    channel: $("channel").value,
    threshold: parseFloat($("threshold").value),
    maxResults,
    cap: Math.max(2 * maxResults, 1500),
    nms_iou: 0.3,
    exclude_iou: 0.3,
    scales,
    tasks: orients.flatMap((o) => scales.map((sc) => [o, sc])),
  };
}

async function runMatch() {
  if (!(state.poolReady && state.imageReady && state.selection)) return;
  const params = gatherParams();
  showSpinner(true, `matching… (${POOL_SIZE} cores)`);
  const t0 = performance.now();
  const gen = currentGen + 1; // the gen startMatch will assign
  const cands = await startMatch(params.tasks, params);
  if (gen !== currentGen) return; // superseded by a newer search; let it finish
  const matches = finalize(cands, params.sel, params.nms_iou, params.exclude_iou, params.maxResults);
  state.matches = matches;
  $("matchCount").textContent = String(matches.length);
  setReadout(`${matches.length} matches · ${Math.round(performance.now() - t0)} ms · ` +
             `${params.tasks.length} tasks / ${POOL_SIZE} cores`);
  render();
  showSpinner(false);
}

function autoRun() { if ($("autorun").checked) runMatch(); }

// ------------------------------------------------------------ image loading
async function loadImageFromBlobOrURL(src) {
  cancelMatch();   // abandon any in-flight search on the previous image
  let blob = (typeof src === "string") ? await (await fetch(src)).blob() : src;
  const bitmap = await createImageBitmap(blob);
  state.bitmap = bitmap;
  state.imgW = bitmap.width;
  state.imgH = bitmap.height;
  state.selection = null;
  state.matches = [];
  state.imageReady = false;
  $("matchCount").textContent = "0";
  $("imgInfo").textContent = `${bitmap.width}×${bitmap.height}px`;

  const off = document.createElement("canvas");
  off.width = bitmap.width; off.height = bitmap.height;
  const octx = off.getContext("2d", { willReadFrequently: true });
  octx.drawImage(bitmap, 0, 0);
  state.rgbaBuf = octx.getImageData(0, 0, bitmap.width, bitmap.height).data.buffer;

  fitView();
  if (state.poolReady) {
    setHint(null);
    showSpinner(true, "staging image…");
    await setImageOnWorkers();
    showSpinner(false);
  }
  updateRunEnabled();
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
  const v = state.view;
  v.scale = Math.min(64, Math.max(0.02, v.scale * Math.exp(-e.deltaY * 0.0015)));
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

canvas.addEventListener("pointerup", () => {
  const dr = state.dragging;
  state.dragging = null;
  canvas.classList.remove("panning");
  if (!dr || dr.kind !== "select") return;
  if (state.selection && state.selection.w >= 4 && state.selection.h >= 4) {
    updateRunEnabled();
    autoRun();
  } else {
    state.selection = null; render(); updateRunEnabled();
  }
});

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "m" || e.key === "M") setMode(state.mode === "pan" ? "select" : "pan");
  else if (e.key === "f" || e.key === "F") fitView();
  else if (e.key === "Escape") cancelMatch();
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
$("cancelBtn").addEventListener("click", cancelMatch);
$("clearBtn").addEventListener("click", () => {
  cancelMatch();
  state.selection = null; state.matches = [];
  $("matchCount").textContent = "0"; setReadout(""); render(); updateRunEnabled();
});

$("threshold").addEventListener("input", () => { $("thVal").textContent = $("threshold").value; });
$("threshold").addEventListener("change", autoRun);
$("maxResults").addEventListener("input", () => { $("mrVal").textContent = $("maxResults").value; });
$("maxResults").addEventListener("change", autoRun);
for (const id of ["method", "channel", "multiscale", "rotation", "flipping"]) {
  $(id).addEventListener("change", autoRun);
}

new ResizeObserver(resizeCanvas).observe($("stage"));

// ------------------------------------------------------------------- bootstrap
async function boot() {
  try {
    await bootPool();
    $("poolInfo").textContent = `${POOL_SIZE} core${POOL_SIZE > 1 ? "s" : ""}`;
    if (state.rgbaBuf) {           // image was loaded before the pool finished
      showSpinner(true, "staging image…");
      await setImageOnWorkers();
      showSpinner(false);
    }
    setHint(state.bitmap ? null :
      "Engines ready. <b>Open an image</b> or <b>Load sample</b>, then drag a box around a pattern.");
    updateRunEnabled();
  } catch (err) {
    console.error(err);
    setHint("Failed to start the Python engines.<br>" + (err && err.message ? err.message : err));
  }
}

setMode("select");
resizeCanvas();
boot();
