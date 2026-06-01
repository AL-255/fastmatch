"use strict";
/*
 * FastMatch worker — one Pyodide + NumPy instance per CPU core.
 *
 * GitHub Pages cannot send the COOP/COEP headers that SharedArrayBuffer (and
 * thus Pyodide's pthreads) require, so multi-core is achieved the portable way:
 * the page spawns a POOL of these workers and hands each an independent
 * (orientation x scale) task. Each worker holds its own copy of the image and
 * computes one task's candidate boxes; the page merges them. Keeping matching
 * off the main thread is also what lets the UI stay live (spinner, cancel).
 */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js");

let pyodide = null;
let wSetImage = null;
let wTask = null;

async function init() {
  pyodide = await loadPyodide();
  await pyodide.loadPackage("numpy");
  const src = await (await fetch("fastmatch_web.py")).text();
  pyodide.FS.writeFile("fastmatch_web.py", src);
  await pyodide.runPythonAsync(`
import json, numpy as np
import fastmatch_web as fw
_IMG = None
_PLANES = {}   # channel_mode -> planes, cached per image so 40 tasks don't redo it
def w_set_image(buf, w, h):
    global _IMG, _PLANES
    if hasattr(buf, "to_py"):
        buf = buf.to_py()
    _IMG = np.frombuffer(buf, dtype=np.uint8).reshape(int(h), int(w), 4).copy()
    _PLANES = {}
def _planes(channel):
    p = _PLANES.get(channel)
    if p is None:
        p = fw.prepare_planes(_IMG, channel)
        _PLANES[channel] = p
    return p
def w_task(params_json):
    if _IMG is None:
        return "[]"
    p = json.loads(params_json)
    cands = fw.candidates_for(
        _planes(p["channel"]), (p["x"], p["y"], p["w"], p["h"]),
        p["method"], p["threshold"], p["scale"], p["orient"], p["cap"],
    )
    return json.dumps(cands)
`);
  wSetImage = pyodide.globals.get("w_set_image");
  wTask = pyodide.globals.get("w_task");
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await init();
      self.postMessage({ type: "ready" });
    } else if (msg.type === "image") {
      pyodide.globals.set("_imgbuf", new Uint8Array(msg.buf));
      await pyodide.runPythonAsync(`w_set_image(_imgbuf, ${msg.w}, ${msg.h})`);
      pyodide.globals.delete("_imgbuf");
      self.postMessage({ type: "imageAck" });
    } else if (msg.type === "task") {
      const candidates = JSON.parse(wTask(JSON.stringify(msg.params)));
      self.postMessage({ type: "taskDone", gen: msg.gen, candidates });
    }
  } catch (err) {
    const detail = (err && err.message) ? err.message : String(err);
    // Report as an (empty) task completion so the pool keeps draining, plus a log.
    self.postMessage({ type: "taskDone", gen: msg.gen, candidates: [], error: detail });
  }
};
