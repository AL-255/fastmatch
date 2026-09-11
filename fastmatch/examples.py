"""Multi-example search: several hand-picked boxes instead of one template.

One box of a small, noisy structure (a TSV on a die shot, say) is a poor
template: its own noise dominates, true copies score low and background texture
of similar darkness scores high. Letting the user box several instances (the
*positives*) and a few false hits (the *negatives*) fixes both problems:

1. **Align + average the positives.** Every positive is re-centred on the first
   one's size, then snapped (small local ZNCC search) so hand-drawn boxes that
   are off by a few px line up. Their mean is a far cleaner template; it drives
   the normal tiled search (all methods, scales, orientations, channel modes).
2. **Verify against the examples.** Each candidate patch is compared (ZNCC over
   the RGB pixels) with every positive and every negative; a candidate closer to
   a negative than to any positive is dropped, and so is anything overlapping a
   negative box. This is a nearest-neighbour classifier over the user's
   examples: marking a few false hits removes the whole family they belong to.

The positives themselves are real instances, so they stay in the result (the
count is the total number of instances, examples included).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .types import ORIENTATIONS, Match, MatchParams, apply_orientation

Box = tuple[int, int, int, int]  # (x, y, w, h), half-open image px

#: Candidates verified per batch (bounds the gathered patch memory).
_VERIFY_CHUNK = 50_000

#: max_results for the proposal search: every candidate must reach the
#: verification step; the user's cap applies to the final list only.
_UNCAPPED = 1 << 30


def _inverse_orientations() -> dict[str, str]:
    probe = np.arange(6).reshape(2, 3)
    inv = {}
    for o in ORIENTATIONS:
        t = apply_orientation(probe, o)
        inv[o] = next(p for p in ORIENTATIONS
                      if np.array_equal(apply_orientation(t, p), probe))
    return inv


_INVERSE = _inverse_orientations()


def _recentre(box: Box, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int]:
    """Top-left of a ``w x h`` box with the same centre as ``box``, inside the image."""
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    return min(max(0, x), img_w - w), min(max(0, y), img_h - h)


def _as_float(patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch, dtype=np.float32)
    return patch if patch.ndim == 3 else patch[..., None]


def _znorm(rows: np.ndarray) -> np.ndarray:
    rows = rows - rows.mean(axis=-1, keepdims=True)
    return rows / (np.linalg.norm(rows, axis=-1, keepdims=True) + 1e-6)


def _snap(image: np.ndarray, ref: np.ndarray, x: int, y: int, w: int, h: int, r: int) -> tuple[int, int]:
    """Shift ``(x, y)`` by up to ``r`` px to maximise ZNCC with ``ref``."""
    img_h, img_w = image.shape[:2]
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(img_w, x + w + r), min(img_h, y + h + r)
    region = _as_float(image[y0:y1, x0:x1])
    if region.shape[0] < h or region.shape[1] < w:
        return x, y
    # All (dy, dx) windows of the region at once: (ny, nx, C, h, w).
    wins = np.lib.stride_tricks.sliding_window_view(region, (h, w), axis=(0, 1))
    ny, nx = wins.shape[:2]
    rows = _znorm(wins.reshape(ny * nx, -1))
    ref_row = _znorm(np.ascontiguousarray(ref.transpose(2, 0, 1)).reshape(1, -1))
    best = int(np.argmax(rows @ ref_row[0]))
    return x0 + best % nx, y0 + best // nx


def prepare_examples(
    image: np.ndarray, positives: list[Box], negatives: list[Box]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Box], list[Box]]:
    """Align the examples and build the mean template.

    Returns ``(mean_template, pos_patches, neg_patches, pos_boxes, neg_boxes)``:
    the uint8 template in the image's own layout, float32 ``(K, h, w, C)`` patch
    stacks, and the aligned boxes, all at the first positive's size.
    """
    if not positives:
        raise ValueError("at least one positive example is required")
    img_h, img_w = image.shape[:2]
    _, _, w, h = positives[0]
    r = max(2, min(w, h) // 4)
    ref = _as_float(image[positives[0][1]:positives[0][1] + h, positives[0][0]:positives[0][0] + w])
    pos_boxes: list[Box] = [(positives[0][0], positives[0][1], w, h)]
    for box in positives[1:]:
        x, y = _recentre(box, w, h, img_w, img_h)
        x, y = _snap(image, ref, x, y, w, h, r)
        pos_boxes.append((x, y, w, h))
    neg_boxes = [(*_recentre(b, w, h, img_w, img_h), w, h) for b in negatives]

    def stack(boxes: list[Box]) -> np.ndarray:
        if not boxes:
            return np.empty((0, h, w, image.shape[2] if image.ndim == 3 else 1), np.float32)
        return np.stack([_as_float(image[y:y + h, x:x + w]) for x, y, _, _ in boxes])

    pos, neg = stack(pos_boxes), stack(neg_boxes)
    mean = np.clip(np.rint(pos.mean(axis=0)), 0, 255).astype(np.uint8)
    if image.ndim == 2:
        mean = mean[..., 0]
    return mean, pos, neg, pos_boxes, neg_boxes


def _iou_many(boxes: np.ndarray, box: Box) -> np.ndarray:
    """IoU of each ``(x, y, w, h)`` row of ``boxes`` against one ``box``."""
    x, y, w, h = box
    iw = np.minimum(boxes[:, 0] + boxes[:, 2], x + w) - np.maximum(boxes[:, 0], x)
    ih = np.minimum(boxes[:, 1] + boxes[:, 3], y + h) - np.maximum(boxes[:, 1], y)
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    return inter / (boxes[:, 2] * boxes[:, 3] + w * h - inter)


def _boxes(cands: list[Match]) -> np.ndarray:
    return np.array([(m.x, m.y, m.w, m.h) for m in cands], dtype=np.float64).reshape(-1, 4)


def _candidate_patches(image: np.ndarray, cands: list[Match], h: int, w: int) -> torch.Tensor:
    """Candidate patches in the examples' frame: ``(N, h*w*C)`` float32 rows.

    Undoes each candidate's orientation and resizes scaled hits back to the
    example size, so every row lines up with the example patches.
    """
    img_h, img_w = image.shape[:2]
    out = torch.empty((len(cands), h * w * (image.shape[2] if image.ndim == 3 else 1)))
    groups: dict[tuple[int, int, str], list[int]] = {}
    for i, m in enumerate(cands):
        groups.setdefault((m.w, m.h, m.orientation), []).append(i)
    for (cw, ch, orient), idx in groups.items():
        xs = np.array([cands[i].x for i in idx])
        ys = np.array([cands[i].y for i in idx])
        oy, ox = np.mgrid[0:ch, 0:cw]
        rows = np.clip(ys[:, None, None] + oy, 0, img_h - 1)
        cols = np.clip(xs[:, None, None] + ox, 0, img_w - 1)
        patch = np.asarray(image[rows, cols], dtype=np.float32)  # (n, ch, cw[, C])
        if image.ndim == 2:
            patch = patch[..., None]
        if orient != "R0":
            patch = np.stack([apply_orientation(p, _INVERSE[orient]) for p in patch])
        t = torch.from_numpy(np.ascontiguousarray(patch))  # (n, ph, pw, C)
        if t.shape[1:3] != (h, w):
            t = F.interpolate(t.permute(0, 3, 1, 2), size=(h, w), mode="bilinear",
                              align_corners=False).permute(0, 2, 3, 1)
        out[idx] = t.reshape(len(idx), -1)
    return out


def _verify(image: np.ndarray, cands: list[Match], pos: np.ndarray, neg: np.ndarray,
            device: torch.device, cancel: Callable[[], bool] | None) -> np.ndarray | None:
    """Boolean keep-mask: nearest example of each candidate is a positive."""
    k_pos = torch.from_numpy(_znorm(pos.reshape(len(pos), -1))).to(device)
    k_neg = torch.from_numpy(_znorm(neg.reshape(len(neg), -1))).to(device)
    h, w = pos.shape[1:3]
    keep = np.ones(len(cands), dtype=bool)
    for a in range(0, len(cands), _VERIFY_CHUNK):
        if cancel is not None and cancel():
            return None
        rows = _candidate_patches(image, cands[a:a + _VERIFY_CHUNK], h, w).to(device)
        rows = rows - rows.mean(dim=1, keepdim=True)
        rows = rows / (rows.norm(dim=1, keepdim=True) + 1e-6)
        best_pos = (rows @ k_pos.T).max(dim=1).values
        best_neg = (rows @ k_neg.T).max(dim=1).values
        keep[a:a + len(rows)] = (best_pos >= best_neg).cpu().numpy()
    return keep


def match_examples(
    matcher,
    positives: list[Box],
    negatives: list[Box],
    params: MatchParams,
    *,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[Match]:
    """Search with several positive (and optional negative) example boxes.

    Args:
        matcher: A :class:`fastmatch.engine.Matcher` with an image staged.
        positives: Example boxes of the wanted structure; the first sets the
            template size. At least one.
        negatives: Boxes of things that must NOT be matched (e.g. false hits).
        params: Search parameters, as for :meth:`Matcher.match`.

    Returns:
        ``list[Match]`` sorted by score, examples included, capped at
        ``params.max_results`` (0 = unlimited).
    """
    image = matcher.host_image
    mean, pos, neg, pos_boxes, neg_boxes = prepare_examples(image, positives, negatives)
    cands = matcher.match(mean, replace(params, max_results=_UNCAPPED), exclude_box=None,
                          cancel=cancel, progress=progress)
    if cancel is not None and cancel():
        return []

    if neg_boxes and cands:
        # Drop anything the user explicitly marked as "not this" (vectorised:
        # there can be a million candidates).
        b = _boxes(cands)
        overlaps = np.zeros(len(cands), dtype=bool)
        for nb in neg_boxes:
            overlaps |= _iou_many(b, nb) > float(params.exclude_iou)
        cands = [m for m, o in zip(cands, overlaps) if not o]
        keep = _verify(image, cands, pos, neg, matcher.effective_device, cancel)
        if keep is None:
            return []
        cands = [m for m, k in zip(cands, keep) if k]

    # The positives are instances by definition: keep them even when their own
    # score fell under the search floor (a very noisy example).
    b = _boxes(cands)
    for box in pos_boxes:
        if not b.shape[0] or not (_iou_many(b, box) > float(params.nms_iou)).any():
            cands.append(Match(x=box[0], y=box[1], w=box[2], h=box[3], score=1.0, scale=1.0))

    cands.sort(key=lambda m: m.score, reverse=True)
    if params.max_results > 0:
        cands = cands[: params.max_results]
    return cands
