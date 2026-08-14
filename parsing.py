"""Parsers for the 3-line evidence reply and the answer letter."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from constants import VALID_NEXT_ACTIONS
from geometry import clip01k


def parse_bbox_obs_next(text: str) -> Optional[Tuple[Optional[List[int]], str, str]]:
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]

    def pick(prefix: str) -> Optional[str]:
        return next((ln for ln in lines if ln.upper().startswith(prefix)), None)

    bbox_line, obs_line, next_line = pick("BBOX:"), pick("OBS:"), pick("NEXT:")
    if not (bbox_line and obs_line and next_line):
        return None

    payload = bbox_line.split(":", 1)[1].strip()
    obs = obs_line.split(":", 1)[1].strip()
    nxt = next_line.split(":", 1)[1].strip().upper()

    if payload.upper() in ("NONE", "NULL", "NIL"):
        if obs.upper() == "NONE" and nxt == "STOP":
            return None, "NONE", "STOP"
        return None

    nums = re.findall(r"-?\d+", payload)
    if len(nums) < 4:
        return None
    bbox = [clip01k(n) for n in nums[:4]]

    if not obs or nxt not in VALID_NEXT_ACTIONS:
        return None
    return bbox, obs, nxt


def extract_choice_letter(text: Optional[str], valid_letters: Sequence[str]) -> Optional[str]:
    if not text or not valid_letters:
        return None
    allowed_set = {x.upper() for x in valid_letters}
    stripped = text.strip()
    if stripped.upper() in allowed_set:
        return stripped.upper()
    first = stripped.splitlines()[0].strip().upper() if stripped else ""
    if first in allowed_set:
        return first
    allowed = "".join(re.escape(x) for x in valid_letters)
    patterns = [
        rf"\b(?:the\s+)?answer\b\s*(?:is\s*)?(?::\s*)?\(?([{allowed}])\)?(?!\w)",
        rf"(?m)^\s*([{allowed}])\.\s*",
        rf"\(\s*([{allowed}])\s*\)",
        rf"\*\*\s*\(?([{allowed}])\)?",
    ]
    matches = [
        (m.start(1), m.group(1).upper())
        for pattern in patterns
        for m in re.finditer(pattern, text, re.IGNORECASE)
    ]
    return max(matches, default=(0, None), key=lambda x: x[0])[1]
