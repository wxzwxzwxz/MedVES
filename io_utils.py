"""Path, sample-key, and JSONL helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def to_file_url(path: str) -> str:
    if path.startswith("file://"):
        return path
    return "file://" + os.path.abspath(path)


def resolve_image_path(img_entry: Any, json_dir: str) -> str:
    path = img_entry[0] if isinstance(img_entry, list) else img_entry
    if not isinstance(path, str) or not path:
        raise ValueError(f"Unsupported image field: {img_entry!r}")
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(json_dir, path))


def make_sample_key(sample: Dict[str, Any]) -> str:
    ds = (sample.get("dataset") or "").strip()
    sub = (sample.get("subset") or "").strip()
    q = (sample.get("question") or "").strip()
    o = ",".join(sample.get("options", [])).strip()
    a = (sample.get("label") or sample.get("answer") or "").strip()

    if "image_original" in sample:
        img0 = os.path.basename(sample["image_original"])
    else:
        img = sample.get("image", "")
        img0 = os.path.basename(img[0] if isinstance(img, list) and img else img if isinstance(img, str) else "")
    return "␟".join([ds, sub, img0.strip(), q, o, a])


def load_done_keys(path: str) -> set:
    done = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                done.add(make_sample_key(obj))
    return done
