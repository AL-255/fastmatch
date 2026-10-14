"""Exercise the actual frozen application without leaving a GUI running."""

import json
from pathlib import Path
import platform
import traceback


def run_smoke_test(report_path: str, device: str = "cpu") -> int:
    report = {"ok": False, "platform": platform.platform()}
    try:
        import numpy as np
        import torch
        from PySide6.QtWidgets import QApplication, QWidget
        from . import __version__
        from .app import MainWindow
        from .about import build_id, license_text
        from .device import resolve_device
        from .engine import Matcher
        from .semantic_matcher import SemanticMatcher
        from .types import MatchParams

        torch.set_num_threads(2)
        report.update(version=__version__, build_id=build_id(), torch=torch.__version__,
                      cuda=torch.version.cuda, hip=torch.version.hip,
                      mps_built=torch.backends.mps.is_built())
        app = QApplication.instance() or QApplication([])
        window = MainWindow(None, device="cpu")
        window._viewport.setViewport(QWidget())  # headless runners lack an OpenGL display
        window.show()
        app.processEvents()
        assert "GNU GENERAL PUBLIC LICENSE" in license_text(), "Bundled license is missing"
        window.close()

        rng = np.random.default_rng(42)
        motif = rng.integers(40, 256, (32, 32, 3), dtype=np.uint8).repeat(3, 0).repeat(3, 1)
        image = np.full((256, 256, 3), 100, np.uint8)
        image[20:116, 20:116] = motif
        image[140:236, 140:236] = motif
        resolved = resolve_device(device)
        # CPU smoke is mandatory even for GPU wheels. An explicit GPU smoke
        # requested on real hardware must never silently become a CPU test.
        if device not in ("cpu", "auto") and resolved.type == "cpu":
            raise RuntimeError(f"Requested {device} accelerator is unavailable")
        matcher = Matcher(device=device)
        matcher.set_image(image)
        report["device"] = str(matcher.effective_device)
        for method in ("ncc", "ssd", "ccorr", "fuzzy"):
            results = matcher.match(motif, MatchParams(method=method, threshold_floor=.8,
                                    feature_min_inliers=6), exclude_box=(20, 20, 96, 96))
            assert any(abs(r.x - 140) <= 2 and abs(r.y - 140) <= 2 for r in results), method
        report["matching"] = "ncc, ssd, ccorr, fuzzy"
        features = SemanticMatcher(resolved)._encode([motif, motif])
        assert features.shape == (2, 768) and torch.isfinite(features).all()
        assert float(features[0] @ features[1]) > .99
        report["semantic"] = "pretrained DINOv2 inference passed"
        report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1
