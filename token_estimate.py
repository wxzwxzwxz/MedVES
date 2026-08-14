"""Text and image token estimation."""

from __future__ import annotations

import math
from typing import Any, Dict, List
from urllib.parse import urlparse

from PIL import Image as PILImage

from constants import (
    QWEN3_VL_MAX_PIXELS,
    QWEN3_VL_MERGE_SIZE,
    QWEN3_VL_MIN_PIXELS,
    QWEN3_VL_PATCH_SIZE,
)


def _flatten_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" or "text" in item:
                parts.append(str(item.get("text", "")))
            elif "image" in item or item.get("type") in ("image_url", "image"):
                parts.append("<image>")
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        if "image" in content or content.get("type") in ("image_url", "image"):
            return "<image>"
    return str(content)


def estimate_qwen3vl_image_tokens(
    width: int,
    height: int,
    min_pixels: int = QWEN3_VL_MIN_PIXELS,
    max_pixels: int = QWEN3_VL_MAX_PIXELS,
) -> int:
    total_pixels = width * height
    if total_pixels <= 0:
        return 0
    if total_pixels < min_pixels:
        scale = math.sqrt(min_pixels / total_pixels)
        width, height = int(width * scale), int(height * scale)
    elif total_pixels > max_pixels:
        scale = math.sqrt(max_pixels / total_pixels)
        width, height = int(width * scale), int(height * scale)

    grid_size = QWEN3_VL_PATCH_SIZE * QWEN3_VL_MERGE_SIZE
    width = max(grid_size, round(width / grid_size) * grid_size)
    height = max(grid_size, round(height / grid_size) * grid_size)
    return (height // grid_size) * (width // grid_size)


def _image_value_from_content_item(item: Dict[str, Any]) -> Any:
    image = item.get("image")
    if image is not None:
        return image
    image_url = item.get("image_url")
    if isinstance(image_url, dict):
        return image_url.get("url")
    return image_url


def _count_image_tokens_from_content(content: Any) -> int:
    if not isinstance(content, list):
        return 0

    total = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        image_value = _image_value_from_content_item(item)
        if image_value is None:
            continue

        if isinstance(image_value, PILImage.Image):
            width, height = image_value.size
        elif isinstance(image_value, str):
            if image_value.startswith("file://"):
                image_path = urlparse(image_value).path
            elif "://" not in image_value:
                image_path = image_value
            else:
                continue
            with PILImage.open(image_path) as image:
                width, height = image.size
        else:
            continue

        total += estimate_qwen3vl_image_tokens(width, height)
    return total


def estimate_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, len(text.split()))


def estimate_prompt_tokens_from_history(tokenizer: Any, history: List[Dict[str, Any]]) -> int:
    if tokenizer is None:
        return 0

    text_chunks: List[str] = []
    image_tokens = 0
    for message in history or []:
        role = (message.get("role") or "user").upper()
        content = message.get("content")
        text = _flatten_content_to_text(content)
        text_chunks.append(f"{role}: {text}" if text else f"{role}:")
        image_tokens += _count_image_tokens_from_content(content)

    prompt_text = "\n".join(text_chunks) + "\nASSISTANT:"
    return estimate_tokens(tokenizer, prompt_text) + image_tokens
