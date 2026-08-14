"""Bbox geometry: clipping, IoU, action-consistency checks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence


def clip01k(v: float) -> int:
    return max(0, min(1000, int(round(float(v)))))


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_area(bb: Sequence[int]) -> int:
    return max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])


def is_duplicate_bbox(bbox: Sequence[int], regions: List[Dict[str, Any]], thr: float = 0.7) -> bool:
    return any(iou(bbox, r["bbox_1000"]) >= thr for r in regions)


def is_reasonable_next_bbox(action: str, bbox: Sequence[int], regions: List[Dict[str, Any]]) -> bool:
    if not regions:
        return True
    last = regions[-1]["bbox_1000"]
    if action == "REFINE_SUBREGION":
        return bbox_area(bbox) <= bbox_area(last) * 1.2
    if action == "CHECK_ADJACENT_CONTEXT":
        gap_x = max(last[0] - bbox[2], bbox[0] - last[2], 0)
        gap_y = max(last[1] - bbox[3], bbox[1] - last[3], 0)
        scale = max(last[2] - last[0], last[3] - last[1], 1)
        return iou(bbox, last) < 0.7 and math.hypot(gap_x, gap_y) <= 0.5 * scale
    if action == "VERIFY_SYMMETRY":
        return iou(bbox, last) < 0.5
    return True
