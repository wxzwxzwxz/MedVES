"""Message construction, overlay bookkeeping, and the chat call."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from constants import OVERLAY_MARKER


def user_msg(text: Optional[str] = None, image: Optional[str] = None) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    if image:
        content.append({"type": "image_url", "image_url": {"url": image}})
    if text:
        content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def is_overlay_msg(msg: Dict[str, Any]) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(p, dict) and p.get("type") == "text" and str(p.get("text", "")).startswith(OVERLAY_MARKER)
        for p in content
    )


def attach_overlay(history: List[Dict[str, Any]],
                   overlay: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if overlay is None:
        return history
    return [m for m in history if not is_overlay_msg(m)] + [overlay]


def chat(client: OpenAI, model: str, messages: List[Dict[str, Any]], *,
         temperature: float, top_p: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "")
