"""Argument parsing and per-benchmark defaults."""

from __future__ import annotations

import argparse
import os

REPO = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(REPO, "output", "Qwen3-VL-4B-Instruct")


def parse_args():
    ap = argparse.ArgumentParser(description="Medical VQA evaluation with configurable run settings.")
    ap.add_argument("--bench", required=True, choices=("omni", "4dataset"))
    ap.add_argument("--run_id", default="main", help="Result file is results_<run_id>.jsonl.")

    ap.add_argument("--base_url", default="http://127.0.0.1:22000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--loop_timeout", type=float, default=600.0)
    ap.add_argument("--target_tool_calls_ok", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--max_tool_calls", type=int, default=12)
    ap.add_argument("--max_evidence_calls", type=int, default=12)
    ap.add_argument("--retry_keeps_instruction", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overlay_from_step", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use_baseline", action=argparse.BooleanOptionalAction, default=False,
                    help="Skip the evidence loop and answer with a single direct VLM call.")

    ap.add_argument("--input_json")
    ap.add_argument("--overlay_dir")
    ap.add_argument("--output_dir")
    ap.add_argument("--answer_fresh", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--max_retries_per_step", type=int)
    args = ap.parse_args()

    if args.bench == "omni":
        args.input_json = args.input_json or os.path.join(
            REPO, "data", "Medical_Multimodal_Evaluation_Data_clean",
            "medical_multimodel_evaluation_data_omni.json")
        args.overlay_dir = args.overlay_dir or os.path.join(OUTPUT_ROOT, "overlay_omni")
        args.output_dir = args.output_dir or os.path.join(
            OUTPUT_ROOT, "medical_multimodel_evaluation_data_omni")
        args.answer_fresh = False if args.answer_fresh is None else args.answer_fresh
        args.max_retries_per_step = (
            1 if args.max_retries_per_step is None else args.max_retries_per_step)
    else:
        args.input_json = args.input_json or os.path.join(
            REPO, "data", "Medical_Multimodal_Evaluation_Data_clean",
            "medical_multimodel_evaluation_data_4dataset.json")
        args.overlay_dir = args.overlay_dir or os.path.join(OUTPUT_ROOT, "overlay_4dataset")
        args.output_dir = args.output_dir or os.path.join(
            OUTPUT_ROOT, "medical_multimodel_evaluation_data_4dataset")
        args.answer_fresh = True if args.answer_fresh is None else args.answer_fresh
        args.max_retries_per_step = (
            1 if args.max_retries_per_step is None else args.max_retries_per_step)

    suffix = "_base" if args.use_baseline else ""
    args.output_jsonl = os.path.join(args.output_dir, f"results_{args.run_id}{suffix}.jsonl")
    return args
