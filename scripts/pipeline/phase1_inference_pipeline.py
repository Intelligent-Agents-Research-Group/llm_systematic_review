#!/usr/bin/env python3
"""
Phase I Screening Pipeline with LFM Model.

This script runs three passes over the Phase I screening data with different
temperatures (0.1, 0.4, 0.8), logs predictions, and computes Fleiss' Kappa
between human decisions and model predictions.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# Set environment variables before imports
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from unsloth import FastModel

# Constants
MODEL_DIR = "models/lfm-2.5-1.2b-instruct_for_sys_review" #"models/lfm-1.2_for_sys_review"
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = False
PHASE1_DATA_PATH = "data/phase I screening_ALL studies_cleaned_prompts - phase I screening_ALL studies_cleaned_prompts.csv"
RESULTS_DIR = "results/phase1_results_lfm2.5"
TEMPERATURES = [0.1, 0.4, 0.8]

# Conservative prompt wrapper to reduce false positives
# This adds stricter guidance without explicitly mentioning class imbalance
CONSERVATIVE_PROMPT_PREFIX = """You are an expert systematic review screener with high precision standards. Your task is to determine if a study should be INCLUDED in a systematic review.

CRITICAL GUIDELINES:
- A study should ONLY be labeled as '1' (include) if it CLEARLY and UNAMBIGUOUSLY meets ALL inclusion criteria
- When in doubt, exclude the study (output '0')
- The burden of proof is on inclusion: the abstract must provide strong, direct evidence of meeting each criterion
- Tangential relevance, potential implications, or partial matches are NOT sufficient for inclusion
- Be skeptical: most studies in a screening pool do not meet strict inclusion criteria

"""

CONSERVATIVE_PROMPT_SUFFIX = """

REMINDER: Only output '1' if you are highly confident that ALL inclusion criteria are met based on clear evidence in the abstract. If any criterion is uncertain or only partially met, output '0'."""


def load_model_for_inference(model_dir=MODEL_DIR):
    """Load the fine-tuned LFM model for inference."""
    from transformers import Lfm2ForCausalLM
    
    print(f"Loading model from {model_dir}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        auto_model=Lfm2ForCausalLM,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastModel.for_inference(model)
    print("Model loaded successfully!")
    return model, tokenizer


def predict_label(model, tokenizer, prompt, temperature=0.1, use_conservative_prompt=True):
    """
    Generate prediction for a single prompt.
    
    Args:
        model: The LFM model
        tokenizer: The tokenizer
        prompt: The prompt text
        temperature: Sampling temperature
        use_conservative_prompt: If True, wrap prompt with conservative guidance
        
    Returns:
        Tuple of (predicted_label, raw_decoded_text)
    """
    # Apply conservative prompt wrapper to reduce false positives
    if use_conservative_prompt:
        enhanced_prompt = CONSERVATIVE_PROMPT_PREFIX + prompt + CONSERVATIVE_PROMPT_SUFFIX
    else:
        enhanced_prompt = prompt
    
    messages = [{"role": "user", "content": enhanced_prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    
    # Use do_sample=True when temperature > 0 for proper sampling
    do_sample = temperature > 0
    
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=do_sample,
        temperature=temperature,
        top_k=20,
        top_p=0.1,
        repetition_penalty=1.05,
    )
    
    gen_tokens = output[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    
    # Extract the first 0 or 1 from the response
    for ch in decoded:
        if ch in ("0", "1"):
            return int(ch), decoded
    return None, decoded


def run_inference_pass(model, tokenizer, df, temperature, pass_name):
    """
    Run inference on all prompts with a specific temperature.
    
    Args:
        model: The LFM model
        tokenizer: The tokenizer
        df: DataFrame with prompts
        temperature: Sampling temperature
        pass_name: Name for this pass (for logging)
        
    Returns:
        List of predictions (0, 1, or None for skipped)
    """
    predictions = []
    skipped = 0
    
    print(f"\n{'='*60}")
    print(f"Running {pass_name} (temperature={temperature})")
    print(f"{'='*60}")
    
    model.eval()
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Pass {pass_name}"):
            prompt = row['prompt']
            
            try:
                pred, raw_output = predict_label(model, tokenizer, prompt, temperature)
                if pred is None:
                    skipped += 1
                predictions.append(pred)
            except Exception as e:
                print(f"Error on row {idx}: {e}")
                predictions.append(None)
                skipped += 1
    
    valid_preds = [p for p in predictions if p is not None]
    print(f"Completed {pass_name}: {len(valid_preds)} valid predictions, {skipped} skipped")
    
    return predictions


def compute_fleiss_kappa(ratings_matrix):
    """
    Compute Fleiss' Kappa for inter-rater reliability.
    
    Args:
        ratings_matrix: numpy array of shape (n_subjects, n_categories)
                       where each row sums to the number of raters
                       
    Returns:
        Fleiss' Kappa value
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix[0].sum()
    
    # Proportion of all assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)
    
    # P_i for each subject (extent of agreement)
    P_i = (ratings_matrix ** 2).sum(axis=1) - n_raters
    P_i = P_i / (n_raters * (n_raters - 1))
    
    # Mean of P_i
    P_bar = P_i.mean()
    
    # P_e (expected agreement by chance)
    P_e = (p_j ** 2).sum()
    
    # Fleiss' Kappa
    if P_e == 1:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)
    
    return kappa


def compute_cohen_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa between two raters.
    
    Args:
        rater1: List of ratings from rater 1
        rater2: List of ratings from rater 2
        
    Returns:
        Cohen's Kappa value
    """
    # Filter out None values
    valid_pairs = [(r1, r2) for r1, r2 in zip(rater1, rater2) if r1 is not None and r2 is not None]
    if not valid_pairs:
        return None
    
    rater1_valid = [p[0] for p in valid_pairs]
    rater2_valid = [p[1] for p in valid_pairs]
    
    n = len(valid_pairs)
    
    # Observed agreement
    p_o = sum(r1 == r2 for r1, r2 in valid_pairs) / n
    
    # Expected agreement
    p1_0 = sum(r == 0 for r in rater1_valid) / n
    p1_1 = sum(r == 1 for r in rater1_valid) / n
    p2_0 = sum(r == 0 for r in rater2_valid) / n
    p2_1 = sum(r == 1 for r in rater2_valid) / n
    
    p_e = p1_0 * p2_0 + p1_1 * p2_1
    
    if p_e == 1:
        return 1.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def compute_classification_metrics(labels, preds):
    """Compute classification metrics using sklearn with weighted averaging for imbalanced data."""
    # Filter out None predictions
    valid_pairs = [(y, p) for y, p in zip(labels, preds) if p is not None and y is not None]
    if not valid_pairs:
        return {"error": "No valid predictions"}
    
    labels_valid = [p[0] for p in valid_pairs]
    preds_valid = [p[1] for p in valid_pairs]
    
    total = len(labels_valid)
    
    # Confusion matrix components
    tp = sum(p == 1 and y == 1 for p, y in zip(preds_valid, labels_valid))
    tn = sum(p == 0 and y == 0 for p, y in zip(preds_valid, labels_valid))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds_valid, labels_valid))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds_valid, labels_valid))

    # Sklearn metrics with weighted averaging (best for imbalanced datasets)
    accuracy = balanced_accuracy_score(labels_valid, preds_valid)
    precision_weighted = precision_score(labels_valid, preds_valid, average='weighted', zero_division=0)
    recall_weighted = recall_score(labels_valid, preds_valid, average='weighted', zero_division=0)
    f1_weighted = f1_score(labels_valid, preds_valid, average='weighted', zero_division=0)
    
    # Also compute macro (unweighted) and per-class metrics for comparison
    un_weighted_accuracy = accuracy_score(labels_valid, preds_valid)
    precision_macro = precision_score(labels_valid, preds_valid, average='macro', zero_division=0)
    recall_macro = recall_score(labels_valid, preds_valid, average='macro', zero_division=0)
    f1_macro = f1_score(labels_valid, preds_valid, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(labels_valid, preds_valid, average=None, zero_division=0)
    recall_per_class = recall_score(labels_valid, preds_valid, average=None, zero_division=0)
    f1_per_class = f1_score(labels_valid, preds_valid, average=None, zero_division=0)

    return {
        "total": total,
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
        # Confusion matrix
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load the model
    model, tokenizer = load_model_for_inference()
    
    # Load Phase I data
    print(f"\nLoading Phase I data from {PHASE1_DATA_PATH}...")
    df = pd.read_csv(PHASE1_DATA_PATH)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Get human decisions
    human_decisions = df['Decision'].astype(int).tolist()
    
    # Run three passes with different temperatures
    all_predictions = {}
    for temp in TEMPERATURES:
        pass_name = f"temp_{temp}"
        predictions = run_inference_pass(model, tokenizer, df, temp, pass_name)
        all_predictions[pass_name] = predictions
        
        # Save predictions for this pass
        df[f'pred_{pass_name}'] = predictions
    
    # Save detailed results to CSV (5 columns: prompt, human_decision, 3 prediction passes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_df = pd.DataFrame({
        'prompt': df['prompt'],
        'human_decision': df['Decision'],
        'pred_temp_0.1': all_predictions['temp_0.1'],
        'pred_temp_0.4': all_predictions['temp_0.4'],
        'pred_temp_0.8': all_predictions['temp_0.8'],
    })
    
    results_file = os.path.join(RESULTS_DIR, f"phase1_predictions_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved predictions CSV to {results_file}")
    
    # Compute metrics for each pass
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    
    all_metrics = {}
    for temp in TEMPERATURES:
        pass_name = f"temp_{temp}"
        preds = all_predictions[pass_name]
        metrics = compute_classification_metrics(human_decisions, preds)
        all_metrics[pass_name] = metrics
        
        print(f"\n{pass_name}:")
        print(f"  Accuracy:  {metrics['accuracy_weighted']:.4f}")
        print(f"  Weighted Metrics (for imbalanced data):")
        print(f"    Precision: {metrics['precision_weighted']:.4f}")
        print(f"    Recall:    {metrics['recall_weighted']:.4f}")
        print(f"    F1:        {metrics['f1_weighted']:.4f}")
        print(f"  Macro Metrics (unweighted):")
        print(f"  accuracy={metrics['accuracy_macro']:.4f}, ")
        print(f"    Precision: {metrics['precision_macro']:.4f}")
        print(f"    Recall:    {metrics['recall_macro']:.4f}")
        print(f"    F1:        {metrics['f1_macro']:.4f}")
        print(f"  Per-Class Metrics:")
        print(f"    Class 0 (Exclude): P={metrics['precision_class_0']:.4f}, R={metrics['recall_class_0']:.4f}, F1={metrics['f1_class_0']:.4f}")
        print(f"    Class 1 (Include): P={metrics['precision_class_1']:.4f}, R={metrics['recall_class_1']:.4f}, F1={metrics['f1_class_1']:.4f}")
        print(f"  Confusion: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    
    # Compute Cohen's Kappa between human and each model pass
    print("\n" + "="*60)
    print("COHEN'S KAPPA (Human vs Model)")
    print("="*60)
    
    kappa_results = {}
    for temp in TEMPERATURES:
        pass_name = f"temp_{temp}"
        preds = all_predictions[pass_name]
        kappa = compute_cohen_kappa(human_decisions, preds)
        kappa_results[f"human_vs_{pass_name}"] = kappa
        print(f"Human vs {pass_name}: κ = {kappa:.4f}")
    
    # Compute Fleiss' Kappa across all raters (human + 3 model passes)
    print("\n" + "="*60)
    print("FLEISS' KAPPA (All Raters)")
    print("="*60)
    
    # Build ratings matrix for Fleiss' Kappa
    # Each row is a subject (prompt), columns are [count_of_0, count_of_1]
    n_subjects = len(df)
    n_raters = 4  # human + 3 model passes
    
    ratings_matrix = np.zeros((n_subjects, 2))  # 2 categories: 0 and 1
    
    valid_subjects = 0
    for i in range(n_subjects):
        ratings = [human_decisions[i]]
        for temp in TEMPERATURES:
            pass_name = f"temp_{temp}"
            pred = all_predictions[pass_name][i]
            if pred is not None:
                ratings.append(pred)
        
        # Only include subjects where all raters provided a rating
        if len(ratings) == n_raters:
            for r in ratings:
                ratings_matrix[valid_subjects, r] += 1
            valid_subjects += 1
    
    # Trim to valid subjects only
    ratings_matrix = ratings_matrix[:valid_subjects]
    
    if valid_subjects > 0:
        fleiss_kappa = compute_fleiss_kappa(ratings_matrix)
        print(f"Fleiss' Kappa (Human + 3 Model Passes): κ = {fleiss_kappa:.4f}")
        print(f"Valid subjects: {valid_subjects}/{n_subjects}")
    else:
        fleiss_kappa = None
        print("Could not compute Fleiss' Kappa - no valid subjects")
    
    # Compute pairwise Cohen's Kappa between model passes
    print("\n" + "="*60)
    print("COHEN'S KAPPA (Between Model Passes)")
    print("="*60)
    
    for i, temp1 in enumerate(TEMPERATURES):
        for temp2 in TEMPERATURES[i+1:]:
            pass1 = f"temp_{temp1}"
            pass2 = f"temp_{temp2}"
            preds1 = all_predictions[pass1]
            preds2 = all_predictions[pass2]
            kappa = compute_cohen_kappa(preds1, preds2)
            kappa_results[f"{pass1}_vs_{pass2}"] = kappa
            print(f"{pass1} vs {pass2}: κ = {kappa:.4f}")
    
    # Save metrics and kappa scores to JSON
    metrics_json = {
        "timestamp": timestamp,
        "model_dir": MODEL_DIR,
        "data_path": PHASE1_DATA_PATH,
        "total_samples": len(df),
        "temperatures": TEMPERATURES,
        "classification_metrics": all_metrics,
        "cohen_kappa_human_vs_model": {
            f"temp_{temp}": kappa_results[f"human_vs_temp_{temp}"] 
            for temp in TEMPERATURES
        },
        "cohen_kappa_between_passes": {
            f"temp_{t1}_vs_temp_{t2}": kappa_results.get(f"temp_{t1}_vs_temp_{t2}")
            for i, t1 in enumerate(TEMPERATURES) 
            for t2 in TEMPERATURES[i+1:]
        },
        "fleiss_kappa": fleiss_kappa,
        "valid_subjects_for_fleiss": valid_subjects,
    }
    
    metrics_file = os.path.join(RESULTS_DIR, f"phase1_metrics_{timestamp}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()