"""Replay a FastMatch memory file as a performance benchmark (CPU vs CUDA).

This mirrors how the GUI uses the engine: build ONE Matcher per device, stage
the source image ONCE, then run every saved entry as a query (template = the
entry's selection, with that region excluded), timing each query. It reports
per-entry and total query time per device and the GPU/CPU speedup, so engine
changes can be measured against a real workload.

Usage:
    python -m scripts.bench_memory [memory.json] [--devices cpu,cuda]
                                   [--repeat N] [--method MM] [--csv]

Defaults to ``samples/test.json``. ``--method`` overrides every entry's method
(e.g. ``ncc``) to isolate one code path. CUDA timings synchronize the device.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow ``python scripts/bench_memory.py`` as well as ``-m scripts.bench_memory``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmatch.engine import Matcher  # noqa: E402
from fastmatch.loader import load_image  # noqa: E402
from fastmatch.memory import load_store  # noqa: E402
import dataclasses  # noqa: E402


def _sync(device: str) -> None:
    if device == "cuda":
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _crop(full: np.ndarray, sel: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = sel
    return np.ascontiguousarray(full[y : y + h, x : x + w])


def bench_device(device: str, full: np.ndarray, entries, repeat: int, method: str | None):
    """Stage the image once on ``device`` and time each entry's query."""
    print(f"\n=== device: {device} ===")
    t0 = time.perf_counter()
    eng = Matcher(device=device, compute_dtype="float32")
    eff = eng.effective_device.type
    if eff != device:
        print(f"  (requested {device!r} resolved to {eff!r} — skipping)")
        return None
    eng.set_image(full)
    _sync(device)
    stage_s = time.perf_counter() - t0
    print(f"  stage image {full.shape[1]}x{full.shape[0]}: {stage_s:6.2f} s")

    per_entry_best = []
    for i, e in enumerate(entries):
        tmpl = _crop(full, e.selection)
        params = e.params
        if method is not None:
            params = dataclasses.replace(params, method=method)
        params = dataclasses.replace(params, device=device)
        times = []
        n = 0
        for _ in range(repeat):
            t = time.perf_counter()
            matches = eng.match(tmpl, params, exclude_box=e.selection)
            _sync(device)
            times.append(time.perf_counter() - t)
            n = len(matches)
        best = min(times)
        per_entry_best.append(best)
        th, tw = tmpl.shape[:2]
        print(
            f"  entry {i}: {params.method:4s} tmpl {tw:3d}x{th:3d} "
            f"-> {n:3d} hits | best {best:7.3f} s  (runs: "
            + ", ".join(f"{x:.3f}" for x in times) + ")"
        )
    total = sum(per_entry_best)
    print(f"  TOTAL (sum of best per entry): {total:7.3f} s")
    return {"stage_s": stage_s, "per_entry": per_entry_best, "total": total}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("memory", nargs="?", default="samples/test.json")
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--repeat", type=int, default=2, help="runs per entry; best is kept")
    ap.add_argument("--method", default=None, help="override every entry's method")
    args = ap.parse_args(argv)

    store = load_store(args.memory)
    print(f"memory: {args.memory}  ({len(store.entries)} entries)")
    print(f"source: {store.source_image}")
    doc = load_image(store.source_image)
    full = doc.full

    results = {}
    for device in [d.strip() for d in args.devices.split(",") if d.strip()]:
        results[device] = bench_device(device, full, store.entries, args.repeat, args.method)

    cpu, cuda = results.get("cpu"), results.get("cuda")
    if cpu and cuda:
        print("\n=== summary (best per entry) ===")
        for i, (c, g) in enumerate(zip(cpu["per_entry"], cuda["per_entry"])):
            sp = c / g if g else float("inf")
            flag = "  <-- GPU SLOWER" if sp < 1.0 else ""
            print(f"  entry {i}: cpu {c:7.3f} s  cuda {g:7.3f} s  speedup {sp:5.2f}x{flag}")
        tsp = cpu["total"] / cuda["total"] if cuda["total"] else float("inf")
        print(f"  TOTAL: cpu {cpu['total']:7.3f} s  cuda {cuda['total']:7.3f} s  "
              f"speedup {tsp:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
