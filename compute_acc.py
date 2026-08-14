#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

FOUR_DATASETS = ["PMC-VQA", "PathVQA", "SLAKE", "VQA-RAD"]
OMNI_DATASET = "OmniMedVQA"
OMNI_SUBSETS = [
    ("CT(Computed Tomography)", "CT"),
    ("Fundus Photography", "FP"),
    ("MR (Mag-netic Resonance Imaging)", "MR"),
    ("OCT (Optical Coherence Tomography", "OCT"),
    ("Dermoscopy", "D"),
    ("Microscopy Images", "MI"),
    ("X-Ray", "X-Ray"),
    ("ultrasound", "US"),
]
TARGET_DATASETS = set(FOUR_DATASETS) | {OMNI_DATASET}
OPTION_RE = re.compile(r"([A-K])\.\s*([^\n]+)")
OPTION_LETTERS = "abcdefghijk"
WIDTH = 170


def iter_json_objects(path: str) -> Iterator[Dict[str, Any]]:
    buffer = ""
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            buffer += line
            try:
                obj = json.loads(buffer)
            except json.JSONDecodeError:
                continue
            buffer = ""
            if isinstance(obj, dict):
                yield obj
    if buffer:
        print(f"[WARN] Unparseable JSON remains at the end of the file: {path}")


def find_jsonl_files(root_dir: str) -> List[str]:
    return sorted(
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(root_dir)
        for filename in filenames
        if filename.endswith(".jsonl")
    )


def parse_path(root_dir: str, path: str) -> Tuple[str, str, str]:
    parts = os.path.relpath(path, root_dir).split(os.sep)
    if len(parts) >= 3:
        algorithm, bench = parts[0], parts[1]
    elif len(parts) == 2:  # root_dir itself is the algorithm directory
        algorithm, bench = os.path.basename(os.path.abspath(root_dir)), parts[0]
    else:
        algorithm, bench = "unknown_algorithm", "unknown_bench"
    return algorithm, bench, os.path.basename(path)


def normalize_dataset_name(value: Any) -> str:
    name = re.sub(r"_(?:test|val|validation)$", "", str(value or "unknown"), flags=re.I)
    lower = name.lower()
    if "pmc" in lower and "vqa" in lower:
        return "PMC-VQA"
    if "pathvqa" in lower:
        return "PathVQA"
    if "slake" in lower:
        return "SLAKE"
    if "vqa-rad" in lower or "vqarad" in lower:
        return "VQA-RAD"
    if "omni" in lower:
        return OMNI_DATASET
    return name.strip()

def process_answer(prediction: Any, prompt: Any) -> str:
    prediction = str(prediction or "").replace("addCriterion", "").strip()
    if "is:" in prediction:
        prediction = prediction.rsplit("is:", 1)[-1].strip()

    prediction = (
        prediction.replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .strip()
        .lower()
    )
    if not prediction:
        return ""

    options = [text.strip().lower() for _, text in OPTION_RE.findall(str(prompt or ""))]
    first_word = prediction.split(maxsplit=1)[0].strip(",.?!")

    if first_word in OPTION_LETTERS:
        index = OPTION_LETTERS.index(first_word)
        if index < len(options):
            return options[index]

    full_answer = prediction.strip(",.?!")
    return full_answer if full_answer in options else first_word


def get_raw_prediction(item: Dict[str, Any]) -> str:
    prediction = item.get("predict")
    if prediction is None:
        prediction = item.get("raw_response", "")
    return str(prediction or "")


def is_correct(item: Dict[str, Any]) -> bool:
    raw_prediction = get_raw_prediction(item)
    label = str(item.get("label") or "").strip().lower()
    return (
        process_answer(raw_prediction, item.get("prompt", "")) == label
        or raw_prediction.strip().lower() == label
    )


def get_total_tokens(item: Dict[str, Any]) -> Optional[float]:
    for usage in (
        item.get("token_usage"),
        (item.get("tts_meta") or {}).get("token_usage"),
    ):
        if isinstance(usage, dict):
            value = usage.get("total_tokens")
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return None


def accuracy(correct: int, total: int) -> float:
    return correct / total * 100 if total else 0.0


def new_entry() -> Dict[str, Any]:
    return {
        "subsets": defaultdict(lambda: {"correct": 0, "total": 0}),
        "correct": 0,
        "total": 0,
        "token_sum": 0.0,
        "token_count": 0,
    }


def collect_stats(root_dir: str, files: List[str]) -> Dict[str, Any]:
    # stats[bench][dataset][algorithm][run]
    stats = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(new_entry)))
    )

    for path in files:
        algorithm, bench, run = parse_path(root_dir, path)
        for item in iter_json_objects(path):
            dataset = normalize_dataset_name(item.get("dataset"))
            if dataset not in TARGET_DATASETS:
                continue

            subset = str(item.get("subset") or "default").strip() or "default"
            entry = stats[bench][dataset][algorithm][run]
            entry["subsets"][subset]["total"] += 1
            entry["total"] += 1

            tokens = get_total_tokens(item)
            if tokens is not None:
                entry["token_sum"] += tokens
                entry["token_count"] += 1

            if is_correct(item):
                entry["subsets"][subset]["correct"] += 1
                entry["correct"] += 1

    return stats


def avg_tokens(entry: Dict[str, Any]) -> float:
    return entry["token_sum"] / entry["token_count"] if entry["token_count"] else 0.0


def show_rows(rows: List[Any], topk: int) -> List[Any]:
    return rows if topk <= 0 else rows[:topk]


def print_omni(bench: str, datasets: Dict[str, Any], args: argparse.Namespace) -> bool:
    rows = []
    for algorithm, runs in datasets.get(OMNI_DATASET, {}).items():
        for run, entry in runs.items():
            if entry["total"] < args.min_total:
                continue
            subset_accs = {
                subset: accuracy(stat["correct"], stat["total"])
                for subset, _ in OMNI_SUBSETS
                if (stat := entry["subsets"].get(subset)) and stat["total"]
            }
            macro = sum(subset_accs.values()) / len(subset_accs) if subset_accs else 0.0
            rows.append((macro, accuracy(entry["correct"], entry["total"]), algorithm, run, entry, subset_accs))

    rows = show_rows(sorted(rows, reverse=True, key=lambda row: row[0]), args.topk)
    if not rows:
        return False

    print("=" * WIDTH)
    print(f"DATASET: {OMNI_DATASET}   (bench={bench})   sort_by=macro_avg(desc)")
    print("-" * WIDTH)
    header = f"{'Rank':>4}  {'Macro%':>7}"
    if args.show_micro:
        header += f"  {'Micro%':>7}"
    header += f"  {'AvgTks':>7}  {'Total':>5}  {'Algorithm':<36}  {'Run(file)':<46}  "
    header += "  ".join(f"{abbr:>7}" for _, abbr in OMNI_SUBSETS)
    print(header)
    print("-" * WIDTH)

    for rank, (macro, micro, algorithm, run, entry, subset_accs) in enumerate(rows, 1):
        line = f"{rank:>4}  {macro:>7.2f}"
        if args.show_micro:
            line += f"  {micro:>7.2f}"
        line += f"  {avg_tokens(entry):>7.0f}  {entry['total']:>5}  {algorithm:<36}  {run:<46}  "
        line += "  ".join(
            f"{subset_accs[subset]:>7.1f}" if subset in subset_accs else " " * 7
            for subset, _ in OMNI_SUBSETS
        )
        print(line)
    print()
    return True


def print_four_datasets(bench: str, datasets: Dict[str, Any], args: argparse.Namespace) -> bool:
    algorithm_runs = {
        (algorithm, run)
        for dataset in FOUR_DATASETS
        for algorithm, runs in datasets.get(dataset, {}).items()
        for run in runs
    }
    rows = []

    for algorithm, run in algorithm_runs:
        values, correct, total, token_sum, token_count = {}, 0, 0, 0.0, 0
        for dataset in FOUR_DATASETS:
            entry = datasets.get(dataset, {}).get(algorithm, {}).get(run)
            if not entry or not entry["total"]:
                values[dataset] = None
                continue
            values[dataset] = accuracy(entry["correct"], entry["total"])
            correct += entry["correct"]
            total += entry["total"]
            token_sum += entry["token_sum"]
            token_count += entry["token_count"]

        available = [value for value in values.values() if value is not None]
        if available and total >= args.min_total:
            rows.append((sum(available) / len(available), accuracy(correct, total), token_sum / token_count if token_count else 0.0, total, algorithm, run, values))

    rows = show_rows(sorted(rows, reverse=True, key=lambda row: row[0]), args.topk)
    if not rows:
        return False

    print("=" * WIDTH)
    print(f"DATASETS: {', '.join(FOUR_DATASETS)}   (bench={bench})   sort_by=macro_avg(desc)")
    print("-" * WIDTH)
    header = f"{'Rank':>4}  {'Macro%':>7}"
    if args.show_micro:
        header += f"  {'Micro%':>7}"
    header += f"  {'AvgTks':>7}  {'Total':>5}  {'Algorithm':<36}  {'Run(file)':<46}  "
    header += "  ".join(f"{dataset:>9}" for dataset in FOUR_DATASETS)
    print(header)
    print("-" * WIDTH)

    for rank, (macro, micro, tokens, total, algorithm, run, values) in enumerate(rows, 1):
        line = f"{rank:>4}  {macro:>7.2f}"
        if args.show_micro:
            line += f"  {micro:>7.2f}"
        line += f"  {tokens:>7.0f}  {total:>5}  {algorithm:<36}  {run:<46}  "
        line += "  ".join(
            f"{values[dataset]:>9.1f}" if values[dataset] is not None else " " * 9
            for dataset in FOUR_DATASETS
        )
        print(line)
    print()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize OmniMedVQA, PMC-VQA, PathVQA, SLAKE, and VQA-RAD results."
    )
    parser.add_argument("root_dir", help="Results root directory: root/<algorithm>/<bench>/<run>.jsonl")
    parser.add_argument("--min_total", type=int, default=1)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--show_micro", action="store_true")
    args = parser.parse_args()

    files = find_jsonl_files(args.root_dir)
    if not files:
        parser.error(f"No .jsonl files found: {args.root_dir}")

    stats = collect_stats(args.root_dir, files)
    printed = False
    for bench, datasets in sorted(stats.items()):
        print("\n" + "#" * WIDTH)
        print(f"BENCH: {bench}")
        print("#" * WIDTH + "\n")
        printed |= print_omni(bench, datasets, args)
        printed |= print_four_datasets(bench, datasets, args)

    if not printed:
        print("[WARN] No data matched OmniMedVQA or the four target datasets.")
    print("=" * WIDTH)
    print("DONE.")


if __name__ == "__main__":
    main()
