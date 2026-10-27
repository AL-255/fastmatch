"""Small repeatable CPU matching benchmark (seconds, lower is better)."""
import sys
from pathlib import Path
import time
import json
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastmatch.cpu import available_threads, configure_threads
from fastmatch.engine import Matcher
from fastmatch.types import MatchParams

if __name__ == "__main__":
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (768, 768, 3), dtype=np.uint8)
    template = image[20:84, 20:84].copy()
    image[400:464, 400:464] = template
    report = {}
    for threads in sorted({1, 4, max(1, available_threads() // 2), available_threads()}):
        configure_threads(threads)
        matcher = Matcher(device="cpu", use_pyramid=False)
        matcher.set_image(image)
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            matcher.match(template, MatchParams(threshold_floor=0.95), exclude_box=(20, 20, 64, 64))
            samples.append(time.perf_counter() - start)
        report[threads] = round(min(samples), 4)
        print(json.dumps(report), flush=True)
