from __future__ import annotations

import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import unquote, urlparse

import requests
from PIL import Image, ImageDraw, ImageFont

ImageRef = Union[str, os.PathLike[str]]
BBox = Sequence[float]

COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 165, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 64, 0),
)


def _source_stem(image_ref: str) -> str:
    if image_ref.startswith("file://"):
        image_ref = unquote(urlparse(image_ref).path)
    elif image_ref.startswith(("http://", "https://")):
        image_ref = unquote(urlparse(image_ref).path)

    stem = Path(image_ref).stem
    return stem or "image"


def _load_image(image_ref: ImageRef, timeout: float = 30.0) -> Image.Image:
    ref = os.fspath(image_ref)

    if ref.startswith("file://"):
        ref = unquote(urlparse(ref).path)

    if ref.startswith(("http://", "https://")):
        response = requests.get(ref, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    path = Path(ref).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _load_font(img_w: int, img_h: int) -> Tuple[ImageFont.ImageFont, int]:
    font_size = max(14, int(min(img_w, img_h) * 0.05))
    candidates = (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/local/share/fonts/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size), font_size
        except OSError:
            continue

    return ImageFont.load_default(), font_size


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_bbox(bbox: BBox) -> List[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Each bbox must contain four numbers, got: {bbox!r}")

    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bbox values: {bbox!r}") from exc

    x1, x2 = sorted((_clamp(x1, 0.0, 1000.0), _clamp(x2, 0.0, 1000.0)))
    y1, y2 = sorted((_clamp(y1, 0.0, 1000.0), _clamp(y2, 0.0, 1000.0)))
    return [x1, y1, x2, y2]


def draw_bbox_phrase_overlay(
    image_ref: ImageRef,
    bboxes_2d: Sequence[BBox],
    output_dir: ImageRef,
    phrases: Optional[Sequence[str]] = None,
    *,
    id_offset: int = 0,
    draw_phrase_on_image: bool = False,
    max_phrase_chars: int = 60,
    request_timeout: float = 30.0,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not bboxes_2d:
        raise ValueError("At least one bbox is required.")

    if phrases is None:
        normalized_phrases = [""] * len(bboxes_2d)
    else:
        normalized_phrases = [str(item or "") for item in phrases]
        if len(normalized_phrases) != len(bboxes_2d):
            raise ValueError("phrases and bboxes_2d must have the same length.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    image_rgb = _load_image(image_ref, timeout=request_timeout)
    width, height = image_rgb.size
    image = image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(image)
    font, font_size = _load_font(width, height)
    line_width = max(3, int(min(width, height) * 0.006))

    regions: List[Dict[str, Any]] = []
    for local_index, (raw_bbox, raw_phrase) in enumerate(
        zip(bboxes_2d, normalized_phrases), start=1
    ):
        bbox = _normalize_bbox(raw_bbox)
        region_id = int(id_offset) + local_index
        color = COLORS[(region_id - 1) % len(COLORS)]

        x1, y1, x2, y2 = bbox
        px1 = _clamp((x1 / 1000.0) * width, 0, width - 1)
        py1 = _clamp((y1 / 1000.0) * height, 0, height - 1)
        px2 = _clamp((x2 / 1000.0) * width, 0, width - 1)
        py2 = _clamp((y2 / 1000.0) * height, 0, height - 1)

        draw.rectangle(
            [px1, py1, px2, py2],
            outline=color + (255,),
            width=line_width,
        )

        phrase = raw_phrase.strip()
        if max_phrase_chars > 0 and len(phrase) > max_phrase_chars:
            phrase = phrase[:max_phrase_chars].rstrip() + "…"

        label = f"[{region_id}]"
        text_on_image = (
            f"{label} {phrase}" if draw_phrase_on_image and phrase else label
        )

        text_x = int(px1) + 4
        text_y = int(py1) - int(font_size * 1.2)
        if text_y < 0:
            text_y = int(py1) + 4

        left, top, right, bottom = draw.textbbox(
            (text_x, text_y), text_on_image, font=font
        )
        padding = max(2, int(font_size * 0.25))
        draw.rectangle(
            [
                left - padding,
                top - padding,
                right + padding,
                bottom + padding,
            ],
            fill=(0, 0, 0, 140),
        )
        draw.text(
            (text_x, text_y),
            text_on_image,
            fill=color + (255,),
            font=font,
        )

        regions.append(
            {
                "id": region_id,
                "bbox_1000": [int(round(value)) for value in bbox],
                "phrase": phrase,
                "color_rgb": list(color),
            }
        )

    source_stem = _source_stem(os.fspath(image_ref))
    filename = f"{source_stem}_overlay_{uuid.uuid4().hex[:8]}.png"
    destination = output_path / filename
    image.convert("RGB").save(destination)

    return str(destination), regions
