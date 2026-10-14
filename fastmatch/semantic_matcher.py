"""Offline semantic region retrieval using optional pretrained DINOv2.

Search streams overlapping windows through the encoder in bounded batches,
then refines the strongest candidates. Similarity is cosine in learned feature
space, not a pixel correlation or an estimated probability. Model downloads
only happen through the explicit CLI setup command.
"""

from __future__ import annotations

from collections import OrderedDict
from importlib.util import find_spec
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .types import Match, active_orientations, apply_orientation

MODEL_ID = "facebook/dinov2-small"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "dinov2-small"
SETUP_HINT = "pip install -r requirements-semantic.txt && python -m fastmatch --download-semantic-model"


def model_directory() -> Path:
    return Path(os.environ.get("FASTMATCH_SEMANTIC_MODEL", str(DEFAULT_MODEL_DIR))).expanduser()


def semantic_available() -> tuple[bool, str]:
    if find_spec("transformers") is None:
        return False, f"Semantic search needs optional dependencies: {SETUP_HINT}"
    path = model_directory()
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        return False, f"Download DINOv2 once: {SETUP_HINT}"
    return True, "Offline DINOv2 visual similarity; searches run on the selected engine device."


def download_semantic_model(path: str | Path | None = None) -> Path:
    """Explicitly download model data (no remote model code) for offline search."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(f"Install semantic dependencies first: {SETUP_HINT}") from exc
    target = Path(path) if path is not None else model_directory()
    snapshot_download(MODEL_ID, local_dir=str(target), token=False,
                      allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE"])
    return target.resolve()


def window_positions(length: int, window: int, stride: int) -> list[int]:
    """Cover a dimension, including its final valid window at the edge."""
    if window > length:
        return []
    positions = list(range(0, length - window + 1, max(1, stride)))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


class SemanticMatcher:
    def __init__(self, device="cpu", *, encoder=None):
        self.device = torch.device(device)
        # The injected encoder supports deterministic retrieval tests without
        # downloading weights. Production always loads pretrained local data.
        self._encoder = encoder
        self._model = None
        self._cache = OrderedDict()
        self._image_ref = None

    def invalidate_image(self):
        self._cache.clear()
        self._image_ref = None

    def _load(self):
        if self._encoder is not None or self._model is not None:
            return
        ready, hint = semantic_available()
        if not ready:
            raise RuntimeError(hint)
        try:
            from transformers import Dinov2Model
            self._model = Dinov2Model.from_pretrained(
                str(model_directory()), local_files_only=True,
                use_safetensors=True).eval().to(self.device)
        except Exception as exc:
            raise RuntimeError(f"Cannot load local DINOv2 model. {SETUP_HINT}: {exc}") from exc

    def _encode(self, crops):
        crops = [np.repeat(crop[:, :, None], 3, axis=2) if crop.ndim == 2 else crop for crop in crops]
        if self._encoder is not None:
            return F.normalize(torch.as_tensor(self._encoder(crops), dtype=torch.float32), dim=1)
        self._load()
        arrays = [np.asarray(Image.fromarray(np.asarray(crop, np.uint8)).resize(
            (224, 224), Image.Resampling.BICUBIC)).copy() for crop in crops]
        pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device).float() / 255
        mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.inference_mode():
            tokens = self._model(pixel_values=(pixels - mean) / std).last_hidden_state
            # Global object context plus mean local patch evidence.
            descriptor = torch.cat([F.normalize(tokens[:, 0], dim=1),
                                    F.normalize(tokens[:, 1:].mean(dim=1), dim=1)], dim=1)
            return F.normalize(descriptor, dim=1).cpu()

    def _windows(self, image, boxes, cancel):
        batch_size = 16 if self.device.type == "cuda" else 4
        for start in range(0, len(boxes), batch_size):
            if cancel is not None and cancel():
                return
            batch = boxes[start:start + batch_size]
            missing = [box for box in batch if box not in self._cache]
            if missing:
                crops = [image[y:y + h, x:x + w] for x, y, w, h in missing]
                encoded = self._encode(crops)
                for box, feature in zip(missing, encoded):
                    self._cache[box] = feature
            features = torch.stack([self._cache[box] for box in batch])
            for box in batch:
                self._cache.move_to_end(box)
            while len(self._cache) > 4096:
                self._cache.popitem(last=False)
            yield batch, features

    def match(self, template, image, params, *, exclude_box=None, mask=None,
              cancel=None, progress=None):
        if cancel is not None and cancel():
            return []
        self._load()
        if self._image_ref is not image:
            self.invalidate_image()
            self._image_ref = image
        if template.ndim == 2:
            template = np.repeat(template[:, :, None], 3, axis=2)
        template = template.copy()
        if mask is not None:
            if not mask.any():
                return []
            template[~mask] = (124, 116, 104)  # neutral mean colour
        active = active_orientations(params.enable_rotation, params.enable_flipping)
        queries = self._encode([np.ascontiguousarray(apply_orientation(template, code)) for code in active])
        height, width = image.shape[:2]
        th, tw = template.shape[:2]
        # Each orientation can change the window aspect ratio.
        shapes = set()
        for code in active:
            oh, ow = apply_orientation(template, code).shape[:2]
            for scale in params.scales:
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("Semantic search scales must be finite and positive")
                shapes.add((max(8, round(ow * scale)), max(8, round(oh * scale))))
        total = sum(len(window_positions(width, w, w // 2)) *
                    len(window_positions(height, h, h // 2)) for w, h in shapes)
        seen, proposals = 0, []
        # Iterate by rows so the gigapixel candidate grid is never materialized.
        for w, h in sorted(shapes):
            xs = window_positions(width, w, w // 2)
            for y in window_positions(height, h, h // 2):
                row = [(x, y, w, h) for x in xs]
                for boxes, features in self._windows(image, row, cancel):
                    scores, indices = (features @ queries.T).max(dim=1)
                    for box, score, index in zip(boxes, scores.tolist(), indices.tolist()):
                        if score >= params.threshold_floor:
                            candidate = Match(*box, float(np.clip(score, 0, 1)),
                                              float(np.sqrt(box[2] * box[3] / (tw * th))), active[index])
                            if not self._excluded(candidate, exclude_box, params.exclude_iou):
                                proposals.append(candidate)
                    seen += len(boxes)
                    if progress is not None:
                        progress(int(80 * seen / max(1, total)))
                    # Keep enough alternatives to refine and perform global NMS,
                    # bounding storage even when every window clears the floor.
                    if len(proposals) > max(2000, params.max_results * 8):
                        proposals = self._nms(proposals, params.nms_iou)[:max(500, params.max_results * 4)]
                if cancel is not None and cancel():
                    return []
        coarse = self._nms(proposals, params.nms_iou)[:min(128, max(16, params.max_results))]
        refine = set()
        for r in coarse:
            for dy in (-(r.h // 4), 0, r.h // 4):
                for dx in (-(r.w // 4), 0, r.w // 4):
                    x, y = r.x + dx, r.y + dy
                    if 0 <= x <= width - r.w and 0 <= y <= height - r.h:
                        refine.add((x, y, r.w, r.h))
        boxes_to_refine = sorted(refine)
        done = 0
        for boxes, features in self._windows(image, boxes_to_refine, cancel):
            scores, indices = (features @ queries.T).max(dim=1)
            for box, score, index in zip(boxes, scores.tolist(), indices.tolist()):
                if score >= params.threshold_floor:
                    r = Match(*box, float(np.clip(score, 0, 1)),
                              float(np.sqrt(box[2] * box[3] / (tw * th))), active[index])
                    if not self._excluded(r, exclude_box, params.exclude_iou):
                        proposals.append(r)
            done += len(boxes)
            if progress is not None:
                progress(80 + int(19 * done / max(1, len(boxes_to_refine))))
        if cancel is not None and cancel():
            return []
        results = self._nms(proposals, params.nms_iou)[:max(0, params.max_results)]
        if progress is not None:
            progress(100)
        return results

    @staticmethod
    def _excluded(candidate, box, iou):
        if box is None:
            return False
        # These helpers only use coordinates; no OpenCV dependency is imported.
        from .feature_matcher import FeatureMatcher
        if FeatureMatcher._excluded(candidate, box, iou):
            return True
        x, y, w, h = box
        intersection = max(0, min(candidate.x + candidate.w, x + w) - max(candidate.x, x)) * max(
            0, min(candidate.y + candidate.h, y + h) - max(candidate.y, y))
        # Broad retrieval windows must not rank a fragment of the query object
        # as a new object simply because their centre lies outside the selection.
        return intersection / max(1, min(w * h, candidate.w * candidate.h)) > iou

    @staticmethod
    def _nms(matches, iou):
        from .feature_matcher import FeatureMatcher
        return FeatureMatcher._nms(sorted(matches, key=lambda r: -r.score), iou)
