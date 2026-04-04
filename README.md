# LLM Systematic Review Classifier

A fine-tuned language model workflow for automating Phase I screening of academic papers in systematic reviews. The project uses Unsloth-based SFT/LoRA training and evaluation pipelines to classify papers as included (`1`) or excluded (`0`) based on predefined inclusion criteria.

- **Dataset:** [IARG-UF/llm_sys_review](https://huggingface.co/datasets/IARG-UF/llm_sys_review)
- **Published model:** [IARG-UF/lfm-2.5-1.2b-instruct_for_sys_review](https://huggingface.co/IARG-UF/lfm-2.5-1.2b-instruct_for_sys_review)

---

## Repository Layout

```
llm_systematic_review/
├── scripts/
│   ├── train/
│   │   ├── lfm2_sft_with_unsloth.py        # SFT training for LFM2 / LFM2.5 (LoRA)
│   │   └── nemotron_3_nano_30b_a3b_a100.py # SFT training for Nemotron-3-Nano-30B-A3B (LoRA)
│   ├── eval/
│   │   ├── evaluate_model.py               # Evaluate fine-tuned model on full dataset
│   │   ├── evaluate_model_v2.py            # Evaluate base model (no fine-tuning)
│   │   ├── agreement_metrics.py            # Binary agreement: Cohen's kappa, PABAK, Gwet AC1
│   │   └── multi_rater_agreement.py        # Multi-rater: Fleiss kappa, AC1, AC2 with bootstrap CI
│   └── pipeline/
│       └── phase1_inference_pipeline.py    # 3-pass temperature inference + agreement metrics
├── utils/
│   └── split_data.py                       # Reference: how the train/test split was produced
├── models/                                 # LoRA adapters saved here after training
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python3 -m venv env/llm_env
source env/llm_env/bin/activate
pip install -r requirements.txt
```

---

## Dataset

All data files are published on HuggingFace. Download them before running any scripts:

```bash
# Install HuggingFace Hub CLI if needed
pip install huggingface_hub

# Download the dataset
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="IARG-UF/llm_sys_review", repo_type="dataset", local_dir="data")
EOF
```

The dataset contains:
- `train_data_LFM.csv` — training split (315 rows, 85%)
- `test_data_LFM.csv` — test split (56 rows, 15%)
- `phase I screening_ALL studies_cleaned_prompts.csv` — full Phase I screening set (8,277 rows)
- `200_0_172_1 LFM.csv` — raw labeled source data (371 rows)

`utils/split_data.py` documents how the train/test split was generated from the raw labeled file.

---

## Training

### LFM2 / LFM2.5 (recommended)

```bash
python scripts/train/lfm2_sft_with_unsloth.py
```

- Base model: `unsloth/LFM2.5-1.2B-Instruct`
- LoRA: r=16, alpha=16, trained on response tokens only
- Saves adapter to `models/lfm-2.5-1.2b-instruct_for_sys_review/`

### Nemotron-3-Nano-30B-A3B (requires A100)

```bash
python scripts/train/nemotron_3_nano_30b_a3b_a100.py
```

- Base model: `unsloth/Nemotron-3-Nano-30B-A3B`
- LoRA: r=8, alpha=16
- Saves adapter to `models/nemotron_3_nano_30b_a3b_for_sys_review/`

---

## Evaluation

### Fine-tuned model on the full Phase I dataset

```bash
python scripts/eval/evaluate_model.py \
    --model-dir models/lfm-2.5-1.2b-instruct_for_sys_review \
    --test-path "data/phase I screening_ALL studies_cleaned_prompts.csv"
```

Outputs written to `results/` (created on first run):
- `eval_results_*.json` — classification metrics
- `evaluation_detailed_log_*.jsonl` — per-example predictions
- `skipped_examples_*.jsonl` — samples where no valid label was parsed

### Base model without fine-tuning

```bash
python scripts/eval/evaluate_model_v2.py
```

Uses `unsloth/LFM2.5-1.2B-Instruct` directly from HuggingFace (no local adapter needed).

---

## Phase I Screening Pipeline

Runs 3 inference passes at temperatures [0.1, 0.4, 0.8] over the full Phase I dataset and computes inter-rater agreement between passes and human annotations.

```bash
python scripts/pipeline/phase1_inference_pipeline.py
```

Outputs written to `results/phase1_results_lfm2.5/`:
- `phase1_predictions_<timestamp>.csv` — per-example predictions across all passes
- `phase1_metrics_<timestamp>.json` — classification metrics and agreement statistics

---

## Agreement Metrics

### Binary (model vs. human)

```bash
python scripts/eval/agreement_metrics.py \
    --input results/eval_results_lfm2.5_1.2b_instruct.json
```

Computes Cohen's kappa, PABAK, and Gwet AC1 from a results JSON file.

### Multi-rater (human + 3 model passes)

```bash
python scripts/eval/multi_rater_agreement.py \
    --input results/phase1_results_lfm2.5/phase1_predictions_<timestamp>.csv
```

Computes Fleiss' kappa, multi-rater Gwet AC1, and AC2 with bootstrapped 95% CIs.

---

## Sample Inference

Using the published model without cloning this repo:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="IARG-UF/lfm-2.5-1.2b-instruct_for_sys_review",
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

output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Notes

- Run all commands from the repository root so relative paths in scripts resolve correctly.
- Model weights, caches, and virtual environments are git-ignored. The `models/` directory is populated by running a training script or by downloading the published adapter from HuggingFace.
- The full Phase I dataset has severe class imbalance: ~99.6% excluded (label 0), ~0.4% included (label 1). Evaluation metrics should be interpreted accordingly.
