#!/usr/bin/env python3
"""
Standalone evaluation script for fine-tuned language models on systematic review screening.

This script:
1. Loads the saved LoRA model from disk
2. Evaluates it on the test dataset
3. Logs ALL skipped examples (not just snippets)
4. Provides detailed per-example logging for debugging
5. Improves label parsing to handle the model's thinking output format

Usage:
    python evaluate_model.py [--model-dir MODEL_DIR] [--test-path TEST_PATH]
                             [--max-samples N] [--verbose]
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from unsloth import FastLanguageModel
from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_DIR = "models/lfm-2.5-1.2b-instruct_for_sys_review"
DEFAULT_TEST_PATH = "data/phase I screening_ALL studies_cleaned_prompts - phase I screening_ALL studies_cleaned_prompts.csv"
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_LOAD_IN_4BIT = False

# Output paths
EVAL_RESULTS_PATH = "results/eval_results_full_dataset_lfm2.5.json"
SKIPPED_EXAMPLES_PATH = "results/skipped_examples_full_dataset_lfm2.5.jsonl"
DETAILED_LOG_PATH = "results/evaluation_detailed_log_full_dataset_lfm2.5.jsonl"

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
DO_SAMPLE = False

# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with timestamps and appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# ============================================================================
# Data Processing Functions
# ============================================================================


def normalize_text(text: str) -> str:
    """Normalize text: fix escaped newlines, strip special prefixes and trailing quote artifacts."""
    text = text.replace("\\n", "\n").replace("/n", "\n")
    if text.startswith("<|startoftext|>"):
        text = text[len("<|startoftext|>"):]
    text = text.rstrip("'\"")
    return text.strip()


def parse_binary_label(text: str) -> Optional[int]:
    """
    Parse a binary label (0 or 1) from model output.

    Tries, in order:
    1. Label immediately after a </think> tag
    2. Label at the end of the text
    3. Last standalone 0 or 1 (word boundary)
    4. Last 0 or 1 character anywhere in text

    Returns 0, 1, or None if no valid label found.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    think_match = re.search(r"</think>\s*([01])", text)
    if think_match:
        return int(think_match.group(1))

    end_match = re.search(r"\s*([01])\s*$", text)
    if end_match:
        return int(end_match.group(1))

    all_matches = re.findall(r"\b([01])\b", text)
    if all_matches:
        return int(all_matches[-1])

    for char in reversed(text):
        if char in ("0", "1"):
            return int(char)

    return None


# ============================================================================
# Model Loading and Inference
# ============================================================================


def load_model(
    model_dir: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
    logger: Optional[logging.Logger] = None,
):
    """
    Load the fine-tuned LoRA model from disk.
    
    Returns:
        (model, tokenizer) tuple
    """
    if logger:
        logger.info(f"Loading model from: {model_dir}")
        logger.info(f"Max sequence length: {max_seq_length}")
        logger.info(f"Load in 4-bit: {load_in_4bit}")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_path = Path(model_dir)
    config_path = model_path/"config.json"
    adapter_config_path = model_path / "adapter_config.json"

    used_unsloth = False
    if config_path.exists():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )
        used_unsloth = True
    elif adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(
                f"Missing base_model_name_or_path in {adapter_config_path}"
            )
        if logger:
            logger.info(f"Adapter detected. Base model: {base_model}")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_dir)
            used_unsloth = True
        except Exception as exc:
            if logger:
                logger.warning(
                    "Unsloth base load failed, falling back to PEFT. "
                    f"Reason: {exc}"
                )
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer

            model = AutoPeftModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
            )
    else:
        raise FileNotFoundError(
            "Model directory is missing config.json or adapter_config.json."
        )

    if used_unsloth:
        FastLanguageModel.for_inference(model)

    if logger:
        logger.info("Model loaded successfully")

    return model, tokenizer


def predict_label(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[Optional[int], str]:
    """Run inference on prompt and return (predicted_label, raw_output)."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = output[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
    raw_output_clean = raw_output.replace("<|im_end|>", "").strip()
    label = parse_binary_label(raw_output_clean)
    return label, raw_output


# ============================================================================
# Metrics Computation
# ============================================================================


def compute_classification_metrics(labels: list, preds: list) -> dict:
    """
    Compute binary classification metrics.
    
    Returns dict with: accuracy, precision, recall, f1, agreement percentage,
    and percentage-based confusion matrix values.
    """
    if len(labels) != len(preds):
        raise ValueError("Labels and predictions must have the same length")
    
    total = len(labels)
    if total == 0:
        return {
            "total": 0, "agreement_percentage": 0.0, "accuracy": 0.0, 
            "precision": 0.0, "recall": 0.0, "f1": 0.0, 
            "tp_count": 0, "tn_count": 0, "fp_count": 0, "fn_count": 0,
            "tp_pct": 0.0, "tn_pct": 0.0, "fp_pct": 0.0, "fn_pct": 0.0,
        }
    
    # Compute confusion matrix using sklearn (returns [[TN, FP], [FN, TP]])
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn_count, fp_count, fn_count, tp_count = cm.ravel()
    
    # Compute percentage-based confusion matrix (normalized by total)
    cm_pct = cm / total * 100
    tn_pct, fp_pct, fn_pct, tp_pct = cm_pct.ravel()
    
    # Overall agreement: percentage of cases where prediction matches label
    agreement_count = int(tp_count + tn_count)
    agreement_percentage = (agreement_count / total) * 100
    
    # Sklearn metrics with weighted averaging (best for imbalanced datasets)
    accuracy = balanced_accuracy_score(labels, preds)
    precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # Also compute macro (unweighted) and per-class metrics for comparison
    un_weighted_accuracy = accuracy_score(labels, preds)    
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    
    return {
        "total": int(total),
        # Overall agreement
        "agreement_percentage": float(agreement_percentage),
        "agreement_count": int(agreement_count),
        # Balanced accuracy (weighted)
        "accuracy_weighted": accuracy,
        # Weighted metrics (recommended for imbalanced data)
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        # Macro metrics (unweighted average)
        "accuracy_macro": un_weighted_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        # Per-class metrics
        "precision_class_0": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
        "precision_class_1": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
        "recall_class_0": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
        "recall_class_1": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
        "f1_class_0": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "f1_class_1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        # Confusion matrix counts
        "tp_count": int(tp_count),
        "tn_count": int(tn_count),
        "fp_count": int(fp_count),
        "fn_count": int(fn_count),
        # Confusion matrix percentages
        "tp_pct": float(tp_pct),
        "tn_pct": float(tn_pct),
        "fp_pct": float(fp_pct),
        "fn_pct": float(fn_pct),
    }


# ============================================================================
# Main Evaluation Function
# ============================================================================


def evaluate_model(
    model,
    tokenizer,
    test_path: str,
    model_dir: str,
    results_path: str = EVAL_RESULTS_PATH,
    skipped_path: str = SKIPPED_EXAMPLES_PATH,
    detailed_log_path: str = DETAILED_LOG_PATH,
    max_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Evaluate the model on the test dataset and save results, skipped examples, and detailed logs."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Loading test dataset from: {test_path}")
    dataset = load_dataset("csv", data_files=test_path, split="train")
    dataset = dataset.map(
        lambda ex: {
            "text": normalize_text(ex["prompt"]),
            "label": int(ex["Decision"]) if ex.get("Decision") is not None else None,
        }
    )

    total_rows = len(dataset)
    num_samples = max_samples if max_samples else total_rows
    if max_samples:
        logger.info(f"Evaluating on {max_samples} / {total_rows} samples")
    else:
        logger.info(f"Evaluating on all {total_rows} samples")

    labels = []
    preds = []
    skipped_examples = []
    detailed_logs = []
    skip_reasons = {}

    model.eval()

    for idx, row in enumerate(tqdm(dataset, total=num_samples, desc="Evaluating", unit="sample")):
        if max_samples and len(labels) + len(skipped_examples) >= max_samples:
            break
        
        raw_text = row["text"]
        label = row.get("label")
        log_entry = {"index": idx, "ground_truth_label": label, "full_text": raw_text}

        if label is None or label not in (0, 1):
            skip_reason = "invalid_label"
            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
            skipped_examples.append({"index": idx, "reason": skip_reason, "full_text": raw_text, "label_value": label})
            log_entry["status"] = "skipped_extraction"
            log_entry["skip_reason"] = skip_reason
            detailed_logs.append(log_entry)
            logger.debug(f"[{idx}] Skipped (extraction): {skip_reason}")
            continue

        if not raw_text or not raw_text.strip():
            skip_reason = "empty_prompt"
            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
            skipped_examples.append({"index": idx, "reason": skip_reason, "full_text": raw_text})
            log_entry["status"] = "skipped_extraction"
            log_entry["skip_reason"] = skip_reason
            detailed_logs.append(log_entry)
            logger.debug(f"[{idx}] Skipped (extraction): {skip_reason}")
            continue

        pred, raw_output = predict_label(model, tokenizer, raw_text)
        log_entry["model_raw_output"] = raw_output
        log_entry["predicted_label"] = pred

        if pred is None:
            skip_reasons["missing_prediction"] = skip_reasons.get("missing_prediction", 0) + 1
            skipped_examples.append({
                "index": idx,
                "reason": "missing_prediction",
                "full_text": raw_text,
                "model_raw_output": raw_output,
            })
            log_entry["status"] = "skipped_prediction"
            log_entry["skip_reason"] = "missing_prediction"
            detailed_logs.append(log_entry)
            logger.debug(f"[{idx}] Skipped (prediction): could not parse output")
            logger.debug(f"    Raw output: {raw_output[:200]}...")
            continue

        labels.append(label)
        preds.append(pred)
        correct = (pred == label)
        log_entry["status"] = "success"
        log_entry["correct"] = correct
        detailed_logs.append(log_entry)
        if logger.level <= logging.DEBUG:
            logger.debug(f"[{idx}] {'✓' if correct else '✗'} pred={pred}, truth={label}")
    
    metrics = compute_classification_metrics(labels, preds)

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total evaluated: {metrics['total']}")
    logger.info(f"Total skipped: {len(skipped_examples)}")
    logger.info("-" * 40)
    logger.info(f"Overall Agreement: {metrics['agreement_percentage']:.2f}% ({metrics['agreement_count']}/{metrics['total']})")
    logger.info("-" * 40)
    logger.info(f"Accuracy:  {metrics['accuracy_macro']:.4f}")
    logger.info(f"Precision: {metrics['precision_macro']:.4f}")
    logger.info(f"Recall:    {metrics['recall_macro']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1_macro']:.4f}")
    logger.info("-" * 40)
    logger.info("Confusion Matrix (counts):")
    logger.info(f"                 Predicted 0    Predicted 1")
    logger.info(f"  Actual 0       TN={metrics['tn_count']:>6}      FP={metrics['fp_count']:>6}")
    logger.info(f"  Actual 1       FN={metrics['fn_count']:>6}      TP={metrics['tp_count']:>6}")
    logger.info("-" * 40)
    logger.info("Confusion Matrix (percentages):")
    logger.info(f"                 Predicted 0    Predicted 1")
    logger.info(f"  Actual 0       TN={metrics['tn_pct']:>5.2f}%     FP={metrics['fp_pct']:>5.2f}%")
    logger.info(f"  Actual 1       FN={metrics['fn_pct']:>5.2f}%     TP={metrics['tp_pct']:>5.2f}%")
    
    if skip_reasons:
        logger.info("-" * 40)
        logger.info("Skip Reasons:")
        for reason, count in sorted(skip_reasons.items()):
            logger.info(f"  {reason}: {count}")
    
    logger.info(f"Saving detailed logs to: {detailed_log_path}")
    with open(detailed_log_path, "w", encoding="utf-8") as f:
        for entry in detailed_logs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if skipped_examples:
        logger.info(f"Saving {len(skipped_examples)} skipped examples to: {skipped_path}")
        with open(skipped_path, "w", encoding="utf-8") as f:
            for ex in skipped_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    config_name = getattr(getattr(model, "config", None), "_name_or_path", None)
    resolved_model_name = (
        str(Path(config_name).name) if config_name else str(Path(model_dir).name)
    )
    results = {
        "model_dir": resolved_model_name,
        "test_path": test_path,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "skipped_count": len(skipped_examples),
        "skip_reasons": skip_reasons,
        "generation_config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "do_sample": DO_SAMPLE,
        },
    }
    
    logger.info(f"Saving results summary to: {results_path}")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on systematic review classification"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to saved model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=DEFAULT_TEST_PATH,
        help=f"Path to test CSV file (default: {DEFAULT_TEST_PATH})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=EVAL_RESULTS_PATH,
        help=f"Path to save results JSON (default: {EVAL_RESULTS_PATH})",
    )
    parser.add_argument(
        "--skipped-path",
        type=str,
        default=SKIPPED_EXAMPLES_PATH,
        help=f"Path to save skipped examples (default: {SKIPPED_EXAMPLES_PATH})",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    logger.info("=" * 60)
    logger.info("SYSTEMATIC REVIEW MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Check paths
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.test_path):
        logger.error(f"Test data file not found: {args.test_path}")
        sys.exit(1)
    
    # Load model
    try:
        model, tokenizer = load_model(
            model_dir=args.model_dir,
            load_in_4bit=args.load_4bit,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Run evaluation
    try:
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_path=args.test_path,
            model_dir=args.model_dir,
            results_path=args.results_path,
            skipped_path=args.skipped_path,
            max_samples=args.max_samples,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    logger.info("Evaluation complete!")
    
    return results


if __name__ == "__main__":
    main()
