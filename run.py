#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple

from openai import OpenAI
from tqdm import tqdm

from cli import parse_args
from io_utils import load_done_keys, make_sample_key, resolve_image_path, to_file_url
from parsing import extract_choice_letter
from pipeline import run_baseline, run_evidence_loop
from prompts import build_prompt

def main():
    args = parse_args()
    try:
        from transformers import AutoTokenizer
        args.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as exc:
        print(f"Failed to load tokenizer: {exc!r}")
        args.tokenizer = None

    input_json = os.path.abspath(args.input_json)
    json_dir = os.path.dirname(input_json)
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    output_jsonl = args.output_jsonl
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)
    os.makedirs(args.overlay_dir, exist_ok=True)

    samples = list(enumerate(data))
    mode_out = "w"
    if args.resume and os.path.exists(output_jsonl):
        done = load_done_keys(output_jsonl)
        if done:
            print(f"[RESUME] {output_jsonl}: {len(done)} done, appending.")
            samples = [(i, s) for i, s in samples if make_sample_key(s) not in done]
            mode_out = "a"

    mode = "baseline" if args.use_baseline else "agent"

    def evaluate(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        idx, sample = item
        question = sample.get("question", "")
        options = sample.get("options", [])
        if not isinstance(options, list):
            raise ValueError(f"options must be a list, got {type(options)} at idx={idx}")

        prompt, valid_letters = build_prompt(question, options)
        img_abs = resolve_image_path(sample.get("image", []), json_dir)
        if not os.path.exists(img_abs):
            raise FileNotFoundError(f"Image not found: {img_abs} (idx={idx})")
        img_url = to_file_url(img_abs)

        try:
            runner = run_baseline if args.use_baseline else run_evidence_loop
            result = runner(client, args, prompt, img_url, valid_letters)
        except Exception as exc:
            print(f"[ERROR] idx={idx}: {exc!r}")
            result = {"raw": f"[ERROR] {exc!r}", "messages": None, "forced": "ERROR",
                      "stats": None, "evidence": [], "token_usage": None}

        raw = result["raw"]
        return {
            "idx": idx,
            "dataset": sample.get("dataset", ""),
            "subset": sample.get("subset", ""),
            "question": question,
            "options": options,
            "prompt": prompt,
            "image_original": img_abs,
            "predict": extract_choice_letter(raw, valid_letters) if options else raw,
            "label": sample.get("answer"),
            "raw_response": raw,
            "final_messages": result["messages"],
            "evidence": result["evidence"],
            "mode": mode,
            "forced": result["forced"],
            "token_usage": result["token_usage"],
            "stats": result["stats"],
        }

    write_lock = threading.Lock()
    with open(output_jsonl, mode_out, encoding="utf-8") as fout:
        def emit(record: Dict[str, Any]) -> None:
            with write_lock:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

        if args.workers <= 1:
            for item in tqdm(samples, desc=f"Evaluating ({mode})"):
                emit(evaluate(item))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(evaluate, item) for item in samples]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Evaluating ({mode}, workers={args.workers})"):
                    emit(future.result())

if __name__ == "__main__":
    main()
