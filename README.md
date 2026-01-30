# LLM Systematic Review Classifier

A fine-tuned language model workflow for automating the screening of academic papers in systematic reviews. The project uses Unsloth-based SFT/LoRA training and evaluation pipelines to classify papers as included (1) or excluded (0) based on predefined inclusion criteria.

## 📋 Overview

Systematic reviews require screening large volumes of papers against strict inclusion criteria. This repository provides:

- Training scripts for LFM2.5 and Nemotron models
- Evaluation pipelines with detailed logging
- Phase I screening pipeline for multi-pass inference
- Agreement metric utilities (Cohen’s kappa, PABAK, Gwet AC1)

## 🗂️ Repository Layout

```
llm_systematic_review/
├── data/                          # Datasets (CSV)
├── models/                        # Saved LoRA adapters / model artifacts
├── results/                       # Evaluation outputs, logs, and phase results
├── scripts/
│   ├── train/                      # Training scripts
│   ├── eval/                       # Evaluation + agreement scripts
│   └── pipeline/                   # End-to-end inference pipelines
├── utils/                          # Helper utilities (e.g., data splitting)
├── env/                            # Local virtual environment (ignored)
├── caches/                         # Unsloth caches/checkpoints (ignored)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Set up a Python environment

```bash
python3 -m venv env/llm_env
source env/llm_env/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data

```bash
python utils/split_data.py
```

This splits the original dataset into train/test CSVs under data/.

## 🏋️ Train a Model

### LFM2.5 SFT (LoRA)

```bash
python scripts/train/lfm2_sft_with_unsloth.py
```

Outputs:
- Model adapters saved to models/
- Evaluation JSON saved to results/

### Nemotron SFT (LoRA)

```bash
python scripts/train/nemotron_3_nano_30b_a3b_a100.py
```

Outputs:
- Model adapters saved to models/
- Evaluation JSON saved to results/

## ✅ Evaluate a Model

```bash
python scripts/eval/evaluate_model.py \
    --model-dir models/lfm-2.5-1.2b-instruct_for_sys_review \
    --test-path data/test_data_LFM.csv \
    --verbose
```

Artifacts are written to results/, including:

- eval_results_*.json
- skipped_examples_*.jsonl
- evaluation_detailed_log_*.jsonl

## 🧪 Phase I Screening Pipeline

Runs multi-pass inference at different temperatures and computes agreement statistics.

```bash
python scripts/pipeline/phase1_inference_pipeline.py
```

Outputs to results/phase1_results_lfm2.5/.

## 📏 Agreement Metrics

### Binary agreement metrics

```bash
python scripts/eval/agreement_metrics.py \
    --input results/eval_results_lfm2.5_1.2b_instruct.json
```

### Multi-rater agreement

```bash
python scripts/eval/multi_rater_agreement.py \
    --input results/phase1_results_lfm2.5/phase1_predictions_*.csv
```

## 🧠 Sample Inference (Python)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="models/lfm-2.5-1.2b-instruct_for_sys_review",
        max_seq_length=4096,
)
FastLanguageModel.for_inference(model)

prompt = """Assess inclusion for the review.
Title: 'ChatGPT as a Software Development Bot: A Project-Based Study'.
Abstract: The study examines ChatGPT as a support tool for undergraduate
software development projects and evaluates learning outcomes.
Output 0 or 1."""

messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", tokenize=True, return_dict=True
).to("cuda")

output = model.generate(**inputs, max_new_tokens=64, temperature=0.0)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 📝 Notes

- Run commands from the repository root to keep relative paths consistent.
- Large artifacts (models, caches, env) are kept outside source code and ignored by git.

If you need a different layout or script entry points, update paths inside the scripts under scripts/.
