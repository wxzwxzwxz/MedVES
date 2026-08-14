"""System prompts and prompt builders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

SYSTEM_PROMPT = """
You are a visual evidence extractor for medical VQA. You DO NOT diagnose or answer during EVIDENCE steps.

At each EVIDENCE STEP, propose one localized visual cue that is directly verifiable from the image pixels, or NONE when localization would not help. The output must contain exactly three lines:
BBOX: x1,y1,x2,y2 or NONE
OBS: neutral visual observation within the BBOX, or NONE
NEXT: what evidence to look for next, or STOP

OBS rules:
The observation must be benchmark-agnostic and restricted to pixel-verifiable properties, such as intensity/density, texture, shape, size, location, boundary, symmetry, or the presence/absence of a visible structure. Use the structured schema target/attr/value/conf. Do not include diagnoses, clinical labels, or generic conclusion words that imply diagnosis, such as normal, abnormal, pathological, atrophy, enlarged, widened, effaced, volume loss, lesion, mass, fracture, edema, or hemorrhage. Do not use inference language such as likely, suggests, consistent with, or indicative of.

Redundancy rule:
Avoid proposing bounding boxes that substantially overlap with previous ones (preferably IoU < 0.7), except for REFINE_SUBREGION.

Return:
BBOX: NONE
OBS: NONE
NEXT: STOP
when no informative local box is needed or visible, or no additional box would help. This includes decisions relying mainly on global context, such as organ identity, whole-region appearance, diffuse or layered structure, or whole-image symmetry. Do not invent a focal cue.

Coordinates:
All coordinates must be integers in the range [0, 1000].

When asked for the ANSWER STEP, output only one option letter and nothing else.
""".strip()


BASELINE_SYSTEM_PROMPT = (
    "Your role is that of a research assistant specializing in medical visual understanding. "
    "Answer questions about images by looking at them closely."
)


def build_prompt(question: str, options: List[str]) -> Tuple[str, List[str]]:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    option_lines = "\n".join(f"{l}. {opt}" for l, opt in zip(letters, options))
    prompt = f"{question}\n\n{option_lines}\n"
    return prompt, letters


def _exit_hint() -> str:
    return (
        "If no additional localized evidence is needed, output:\n"
        "BBOX: NONE\nOBS: NONE\nNEXT: STOP\n\n"
        "Otherwise output EXACTLY 3 lines and NOTHING else:\n"
    )


_FORMAT_BLOCK = (
    "BBOX: x1,y1,x2,y2\n"
    "OBS: short neutral visual observation\n"
    "NEXT: VERIFY_SYMMETRY or CHECK_ADJACENT_CONTEXT or REFINE_SUBREGION\n"
)

_ACTION_HINTS = {
    "VERIFY_SYMMETRY": "Inspect a NEW region that helps verify whether the current finding is symmetric, "
                       "anatomically expected, or globally consistent.",
    "CHECK_ADJACENT_CONTEXT": "Inspect a NEW region adjacent to the current evidence that best clarifies "
                              "surrounding anatomy, cavity, or tissue context.",
    "REFINE_SUBREGION": "Inspect a NEW subregion or boundary of the current evidence.",
}


def build_anchor_prompt(question_prompt: str) -> str:
    return (
        f"{question_prompt}\n\n"
        "EVIDENCE STEP: find ONE bbox containing the most important candidate evidence.\n\n"
        f"{_exit_hint()}{_FORMAT_BLOCK}"
    )


def build_step_prompt(action: str) -> str:
    hint = _ACTION_HINTS.get(action, "Inspect a NEW region.")
    return f"EVIDENCE STEP:\n{hint}"


_ACTION_CONSTRAINTS = {
    "VERIFY_SYMMETRY": "The new bbox must cover a clearly different region than {last} (IoU < 0.5), "
                       "typically the mirrored / contralateral position.",
    "CHECK_ADJACENT_CONTEXT": "The new bbox must sit next to {last} - touching it or within half its size - "
                              "and must not repeat it (IoU < 0.7).",
    "REFINE_SUBREGION": "The new bbox must be a sub-part or boundary of {last} and must NOT be larger than it.",
}


def build_retry_prompt(reason: str, step_prompt: str = "", action: str = "",
                       last_bbox: Optional[Sequence[int]] = None) -> str:
    parts = [f"RETRY: {reason}"]
    constraint = _ACTION_CONSTRAINTS.get(action)
    if constraint and last_bbox is not None:
        parts.append(constraint.format(last=list(last_bbox)))
    if step_prompt:
        parts.append(step_prompt)
    return "\n".join(parts)


def format_evidence(regions: List[Dict[str, Any]], max_items: int = 12, obs_max_len: int = 120) -> str:
    if not regions:
        return "EVIDENCE_SO_FAR: (none)"
    lines = [f"EVIDENCE_SO_FAR (n={len(regions)}):"]
    for r in regions[-max_items:]:
        phrase = r["phrase"]
        if len(phrase) > obs_max_len:
            phrase = phrase[:obs_max_len].rstrip() + "…"
        lines.append(f"- {r['id']} | bbox={r['bbox_1000']} | obs={phrase}")
    return "\n".join(lines)


def build_answer_prompt_fresh(question_prompt: str, evidence: List[Dict[str, Any]]) -> str:
    return (
        "ANSWER STEP:\n"
        f"{question_prompt}\n\n"
        f"{format_evidence(evidence)}\n\n"
        "Given the original image, the latest overlay image, and the accumulated evidence "
        "observations, answer the question with ONLY the option letter.\n"
    )


def build_answer_prompt(evidence: List[Dict[str, Any]]) -> str:
    return (
        "ANSWER STEP:\n"
        "Use the latest overlay image to verify evidence if needed.\n"
        "Now answer with ONLY the option letter.\n"
        f"{format_evidence(evidence)}"
    )
