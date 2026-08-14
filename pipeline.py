"""The evidence loop and the direct-VLM baseline."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from chat_utils import attach_overlay, chat, user_msg
from geometry import is_duplicate_bbox, is_reasonable_next_bbox
from image_overlay import draw_bbox_phrase_overlay
from io_utils import to_file_url
from parsing import extract_choice_letter, parse_bbox_obs_next
from prompts import (
    BASELINE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_anchor_prompt,
    build_answer_prompt,
    build_answer_prompt_fresh,
    build_retry_prompt,
    build_step_prompt,
)
from token_estimate import estimate_prompt_tokens_from_history, estimate_tokens


def run_evidence_loop(client: OpenAI, args, prompt: str, img_url: str,
                      valid_letters: List[str]) -> Dict[str, Any]:
    system = {"role": "system", "content": SYSTEM_PROMPT}
    history: List[Dict[str, Any]] = [user_msg(image=img_url)]
    evidence: List[Dict[str, Any]] = []
    start = time.time()
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "current_prompt_tokens": 0,
        "current_completion_tokens": 0,
    }

    def ask(msgs: List[Dict[str, Any]], max_tokens: int,
            system_override: Optional[str] = None) -> str:
        request_messages = [{"role": "system", "content": system_override} if system_override
                            else system] + msgs
        if args.tokenizer is not None:
            current_prompt_tokens = estimate_prompt_tokens_from_history(args.tokenizer, msgs)
            token_usage["current_prompt_tokens"] = current_prompt_tokens
            token_usage["prompt_tokens"] += current_prompt_tokens

        text = chat(client, args.model, request_messages,
                    temperature=args.temperature, top_p=args.top_p, max_tokens=max_tokens)

        if args.tokenizer is not None:
            current_completion_tokens = estimate_tokens(args.tokenizer, text)
            token_usage["current_completion_tokens"] = current_completion_tokens
            token_usage["completion_tokens"] += current_completion_tokens
            token_usage["total_tokens"] = (
                token_usage["prompt_tokens"] + token_usage["completion_tokens"]
            )
        return text

    latest_overlay: Optional[Dict[str, Any]] = None
    planned_next = "ANCHOR"
    forced: Optional[str] = None
    steps = overlay_calls = overlay_fails = evidence_calls = ok = 0

    while ok < args.target_tool_calls_ok:
        if steps >= args.max_steps:
            forced = "MAX_STEPS"
            break
        if overlay_calls >= args.max_tool_calls:
            forced = "MAX_TOOL_CALLS"
            break
        if evidence_calls >= args.max_evidence_calls:
            forced = "MAX_EVIDENCE_CALLS"
            break
        if time.time() - start >= args.loop_timeout:
            forced = "TIMEOUT"
            break

        action = planned_next
        step_prompt = (build_anchor_prompt(prompt) if action == "ANCHOR"
                       else build_step_prompt(action))

        if len(evidence) + 1 >= args.overlay_from_step:
            history = attach_overlay(history, latest_overlay)
        ckpt = list(history)
        history.append(user_msg(text=step_prompt))

        def retry_msg(reason: str) -> Dict[str, Any]:
            if not args.retry_keeps_instruction:
                return user_msg(text=build_retry_prompt(reason))
            return user_msg(text=build_retry_prompt(
                reason, step_prompt, action, evidence[-1]["bbox_1000"] if evidence else None))

        success = False
        for _ in range(args.max_retries_per_step + 1):
            reply = ask(history, args.max_tokens)
            steps += 1

            parsed = parse_bbox_obs_next(reply)
            if parsed is None:
                history = ckpt + [retry_msg("format error")]
                continue

            bbox, obs, nxt = parsed
            evidence_calls += 1

            if bbox is None:
                history.append({"role": "assistant", "content": reply})
                planned_next, forced, success = "STOP", "EARLY_EXIT", True
                break

            if action != "REFINE_SUBREGION" and is_duplicate_bbox(bbox, evidence):
                history = ckpt + [retry_msg("bbox overlaps previous evidence (IoU>=0.7). Pick a NEW bbox elsewhere.")]
                continue
            if action != "ANCHOR" and not is_reasonable_next_bbox(action, bbox, evidence):
                history = ckpt + [retry_msg(f"the proposed bbox does not match the requested action {action}. Pick a more suitable NEW bbox.")]
                continue

            planned_next = nxt
            history.append({"role": "assistant", "content": reply})
            region = {"id": len(evidence) + 1, "bbox_1000": bbox, "phrase": obs}

            draw = evidence + [region]
            overlay_calls += 1
            try:
                overlay_path, _ = draw_bbox_phrase_overlay(
                    img_url,
                    [r["bbox_1000"] for r in draw],
                    args.overlay_dir,
                    phrases=[r["phrase"] for r in draw],
                    id_offset=0,
                    draw_phrase_on_image=False,
                    max_phrase_chars=200,
                )
            except Exception as exc:
                overlay_fails += 1
                history = ckpt + [retry_msg(f"overlay failed: {exc!r}. Propose a DIFFERENT bbox.")]
                continue

            evidence.append(region)
            ok += 1
            caption = "Latest overlay image (all collected bboxes so far)."
            latest_overlay = user_msg(text=caption, image=to_file_url(overlay_path))
            success = True
            break

        if not success:
            forced = forced or "STEP_RETRY_EXHAUSTED"
            break
        if planned_next == "STOP":
            forced = forced or "EARLY_STOP"
            break

    if args.answer_fresh:
        history = [user_msg(text="Original image", image=img_url)]
        if latest_overlay is not None:
            history.append(latest_overlay)
        history.append(user_msg(text=build_answer_prompt_fresh(prompt, evidence)))
        answer = ask(history, 8, system_override=BASELINE_SYSTEM_PROMPT)
    else:
        history = attach_overlay(history, latest_overlay)
        history.append(user_msg(text=build_answer_prompt(evidence)))
        answer = ask(history, args.max_tokens)
    history.append({"role": "assistant", "content": answer})

    if not extract_choice_letter(answer, valid_letters):
        forced = forced or "FORCED_CONCLUDE"
        fallback_prompt = (
            f"{prompt}\n\n"
            "Answer with ONLY the option letter from the given choices."
        )

        text = chat(client, args.model,
                       [{"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                        user_msg(text=fallback_prompt, image=img_url)],
                       temperature=0.0, top_p=1.0, max_tokens=8)
        answer = text

    return {
        "raw": answer,
        "messages": history,
        "forced": forced,
        "stats": {
            "steps": steps,
            "tool_calls": overlay_calls,
            "tool_calls_ok": ok,
            "tool_calls_fail": overlay_fails,
            "evidence_calls": evidence_calls,
            "forced": forced,
        },
        "evidence": evidence,
        "token_usage": token_usage,
    }


def run_baseline(client: OpenAI, args, prompt: str, img_url: str,
                 valid_letters: List[str]) -> Dict[str, Any]:
    system_prompt = BASELINE_SYSTEM_PROMPT + " Do not call any tools."
    baseline_prompt = (
        f"{prompt.rstrip()}\n"
        "Answer with the option's letter from the given choices directly."
    )
    content = [{"type": "text", "text": baseline_prompt},
                {"type": "image_url", "image_url": {"url": img_url}}]
    answer_max_tokens = 8
    history = [{"role": "user", "content": content}]
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "current_prompt_tokens": 0,
        "current_completion_tokens": 0,
    }

    if args.tokenizer is not None:
        token_usage["current_prompt_tokens"] = estimate_prompt_tokens_from_history(
            args.tokenizer, history)
        token_usage["prompt_tokens"] = token_usage["current_prompt_tokens"]

    answer = chat(client, args.model,
                  [{"role": "system", "content": system_prompt}] + history,
                  temperature=args.temperature, top_p=args.top_p, max_tokens=answer_max_tokens)

    if args.tokenizer is not None:
        token_usage["current_completion_tokens"] = estimate_tokens(args.tokenizer, answer)
        token_usage["completion_tokens"] = token_usage["current_completion_tokens"]
        token_usage["total_tokens"] = (
            token_usage["prompt_tokens"] + token_usage["completion_tokens"]
        )

    history.append({"role": "assistant", "content": answer})
    return {
        "raw": answer,
        "messages": history,
        "forced": None,
        "stats": {
            "steps": 1,
            "tool_calls": 0,
            "tool_calls_ok": 0,
            "tool_calls_fail": 0,
            "evidence_calls": 0,
            "forced": None,
        },
        "evidence": [],
        "token_usage": token_usage,
    }
