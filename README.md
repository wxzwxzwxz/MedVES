# MedVES: Visual Evidence Scaling at Test Time for VLMs in Medical VQA

> **TL;DR.** More test-time text is not the same as more visual evidence. MedVES makes a frozen VLM *look again*: it localizes a region, records a neutral visual observation, updates a cumulative visual state, and decides what to inspect next before answering.

<p align="center">
  <a href="assets/framework.pdf">
    <img src="assets/framework.png" width="100%" alt="MedVES framework">
  </a>
</p>

## What is MedVES?

MedVES is a training-free test-time scaling framework for medical VQA that uses extra inference compute to iteratively localize useful visual evidence. At each step, a frozen VLM predicts a bounding box, a visual observation, and the next evidence-seeking action. Accepted evidence is accumulated as overlays on the original image, preserving global context while guiding subsequent steps. The process stops once sufficient evidence is found, allowing harder questions to receive more test-time compute.

### Why it matters

- **Training-free:** no parameter updates and no extra annotated data.
- **Visual test-time scaling:** additional compute is spent on finding complementary image evidence rather than only generating longer text.
- **Bounded visual state:** VOS keeps the search history spatially explicit without growing the dialogue with one new image per step.
- **Adaptive compute:** early stopping trades a small amount of peak fixed-budget accuracy for substantially lower token cost.

## Installation

```bash
conda create -n medves python=3.10 -y
conda activate medves

pip install -r requirements.txt
pip install transformers
```

MedVES talks to OpenAI-compatible `/chat/completions` servers. Images are passed as `file://` URLs, so the serving process must be allowed to read them from disk.

Example with vLLM:

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --port 22000 \
  --allowed-local-media-path /
```

> For shared or production machines, prefer restricting `--allowed-local-media-path` to the dataset/project directory instead of `/`.

## Data

Place the benchmark JSON files under `data_process/Medical_Multimodal_Evaluation_Data_clean/`:

```text
data_process/Medical_Multimodal_Evaluation_Data_clean/
    medical_multimodel_evaluation_data_omni.json
    medical_multimodel_evaluation_data_4dataset.json
```

Each entry should have the form:

```json
{
  "dataset": "...",
  "subset": "...",
  "question": "...",
  "options": ["..."],
  "answer": "...",
  "image": "relative/path/to/image.png"
}
```

`image` is interpreted relative to the JSON file.

The `4dataset` benchmark groups **VQA-RAD, SLAKE, PathVQA, and PMC-VQA**.

## Quick Start

### MedVES evidence-search inference

```bash
python run.py --bench omni --workers 32
python run.py --bench 4dataset --workers 32
```

### Single-pass baseline

```bash
python run.py --bench omni --use_baseline --workers 32
python run.py --bench 4dataset --use_baseline --workers 32
```

### Run all four settings

```bash
./run.sh
```

Logs are written under `logs/`.

## Important CLI Options

Every option has a default in `cli.py`; 

| Flag | Default | Meaning |
| --- | --- | --- |
| `--base_url` | `http://127.0.0.1:22000/v1` | OpenAI-compatible endpoint |
| `--model` | `Qwen/Qwen3-VL-4B-Instruct` | Served model name |
| `--use_baseline` | off | Skip the evidence loop and run direct VLM inference |
| `--workers` | `1` | Number of samples evaluated in parallel |
| `--resume` | off | Skip samples already present in the output file |

By default, `--workers` is set to `1`; increasing it (e.g., `--workers 32`) can substantially speed up inference if the serving backend has sufficient capacity.

## Outputs and Evaluation

Results are written to:

```text
output/<model>/<benchmark>/results_<run_id>.jsonl
```

Baseline runs use:

```text
output/<model>/<benchmark>/results_<run_id>_base.jsonl
```

Compute accuracy with:

```bash
python compute_acc.py output/
```# MedVES
