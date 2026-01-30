"""
LFM2 SFT with Unsloth.

Cleaned from the original Colab notebook while preserving the linear flow.
"""

import json
import os

if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from transformers import TextStreamer
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from trl import SFTConfig, SFTTrainer


MODEL_NAME = "unsloth/LFM2.5-1.2B-Instruct" #"unsloth/LFM2-1.2B"
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = False
DATA_PATH = "data/train_data_LFM.csv"
TEST_DATA_PATH = "data/test_data_LFM.csv"
RANDOM_STATE = 3407
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
END_PART = "<|im_end|>"
EVAL_RESULTS_PATH = "results/eval_results_lfm2.5_1.2b_instruct.json"
SAVED_MODEL_DIR = "models/lfm-2.5-1.2b-instruct_for_sys_review"

FOURBIT_MODELS = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/LFM2-1.2B-unsloth-bnb-4bit",
    "unsloth/LFM2-700M-unsloth-bnb-4bit",
    "unsloth/LFM2-350M-unsloth-bnb-4bit",
    # Full 16bit unquantized models
    "unsloth/LFM2-1.2B",
    "unsloth/LFM2-700M",
    "unsloth/LFM2-350M",
]

def load_base_model(
    model_name=MODEL_NAME,
    dtype=DTYPE,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    full_finetuning=True,
):
    # make tokenizer globally accessible for unsloth functions
    global tokenizer, model

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=dtype,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=full_finetuning,
        # token = "hf_...", # use one if using gated models
    )
    return model, tokenizer


def add_lora_adapters(model, random_state=RANDOM_STATE):
    return FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # LFM for now is just text only
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # Should leave on always!
        r=16,  # Larger = higher accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=random_state,
    )


def normalize_text_column(example, column):
    text = example[column]
    # Normalize escaped newlines to actual newlines for template matching.
    text = text.replace("\\n", "\n").replace("/n", "\n")
    return {"text": text}


def load_and_prepare_dataset(
    data_path,
    text_column,
    sample_index=0,
    print_sample=False,
):
    dataset = load_dataset("csv", data_files=data_path, split="train")
    dataset = dataset.map(lambda ex: normalize_text_column(ex, text_column))
    if print_sample:
        print(dataset[sample_index]["text"])
    return dataset


def build_trainer(model, tokenizer, dataset):
    config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Use GA to mimic batch size!
        warmup_steps=5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=320,
        learning_rate=2e-5,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=RANDOM_STATE,
        report_to="none",  # Use this for WandB etc
    )
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,  # Can set up evaluation!
        args=config,
    )


def apply_response_only_mask(trainer):
    return train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )


def validate_response_masking(
    dataset,
    tokenizer,
    instruction_part=INSTRUCTION_PART,
    response_part=RESPONSE_PART,
    sample_index=0,
):
    sample = dataset[sample_index]["text"]
    if instruction_part not in sample or response_part not in sample:
        print(
            "WARNING: Instruction/response markers not found in raw text. "
            "Masking will zero out all labels."
        )
    tokens = tokenizer(sample, return_tensors="pt")
    decoded = tokenizer.decode(tokens["input_ids"][0])
    if instruction_part not in decoded or response_part not in decoded:
        print(
            "WARNING: Instruction/response markers not found after tokenization. "
            "Tokenizer/template mismatch likely."
        )
    print(
        f"Marker counts in raw text: user={sample.count(instruction_part)}, "
        f"assistant={sample.count(response_part)}"
    )


def extract_prompt_and_label(text):
    """Extract the user prompt and ground truth label from formatted text.
    
    Returns:
        tuple: (prompt, label, skip_reason) where skip_reason is None if successful
    """
    user_start = text.find(INSTRUCTION_PART)
    if user_start == -1:
        return None, None, "INSTRUCTION_PART not found"
    user_start += len(INSTRUCTION_PART)
    user_end = text.find(END_PART, user_start)
    if user_end == -1:
        return None, None, "END_PART after user not found"
    user_text = text[user_start:user_end]

    response_start = text.find(RESPONSE_PART, user_end)
    if response_start == -1:
        return None, None, "RESPONSE_PART not found"
    response_start += len(RESPONSE_PART)
    response_end = text.find(END_PART, response_start)
    if response_end == -1:
        return None, None, "END_PART after response not found"
    response_text = text[response_start:response_end].strip()

    label = None
    for ch in response_text:
        if ch in ("0", "1"):
            label = int(ch)
            break
    
    if label is None:
        return user_text, None, f"No 0/1 in response: '{response_text[:50]}...'"
    
    return user_text, label, None


def predict_label(model, tokenizer, prompt, temperature=0.1, fallback_to_majority=True):
    """Generate a prediction (0 or 1) from the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The user prompt text
        temperature: Sampling temperature (default 0.1 for near-deterministic)
        fallback_to_majority: If True, return 0 when no clear prediction found
        
    Returns:
        tuple: (prediction, raw_output, skip_reason) where skip_reason is None if successful
    """
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    
    # Generate with slightly more tokens to capture edge cases
    output = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        temperature=temperature,  # Avoid division by zero
        top_k=50,
        top_p=0.1,
        repetition_penalty=1.05,
    )
    gen_tokens = output[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # Primary: Look for 0 or 1 in the output
    for ch in decoded:
        if ch in ("0", "1"):
            return int(ch), decoded, None
    
    # Secondary: Check for keywords that indicate include/exclude
    decoded_lower = decoded.lower()
    if any(word in decoded_lower for word in ["include", "yes", "accept"]):
        return 1, decoded, None
    if any(word in decoded_lower for word in ["exclude", "no", "reject"]):
        return 0, decoded, None
    
    # Fallback: If enabled, default to 0 (exclude) as it's typically the majority class
    if fallback_to_majority:
        return 0, decoded, f"Fallback to 0 (no 0/1 found in: '{decoded[:50]}...')"
    
    return None, decoded, f"No prediction found in: '{decoded[:50]}...'"


def compute_classification_metrics(labels, preds):
    """Compute classification metrics using sklearn with weighted averaging for imbalanced data."""
    total = len(labels)
    
    # Confusion matrix components
    tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(preds, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels))

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


def evaluate_on_testset(
    model,
    tokenizer,
    test_path=TEST_DATA_PATH,
    results_path=EVAL_RESULTS_PATH,
    verbose=True,
    skipped_examples_path="results/skipped_examples.jsonl",
):
    dataset = load_and_prepare_dataset(
        test_path,
        text_column="test",
        print_sample=False,
    )
    max_samples = int(os.environ.get("EVAL_MAX_SAMPLES", "0") or 0)
    total_rows = len(dataset)
    if max_samples:
        print(
            f"Evaluating on test set: {test_path} "
            f"({max_samples} / {total_rows} samples)"
        )
    else:
        print(f"Evaluating on test set: {test_path} ({total_rows} samples)")

    labels = []
    preds = []
    skipped_extraction = 0
    skipped_prediction = 0
    fallback_used = 0
    skipped_details = []

    model.eval()
    with torch.no_grad():
        for idx, row in enumerate(dataset):
            if max_samples and len(labels) >= max_samples:
                break
            
            # Extract prompt and ground truth label
            prompt, label, extract_reason = extract_prompt_and_label(row["text"])

            # Log extraction failures
            if label is None or not prompt:
                skipped_extraction += 1
                skipped_details.append({
                    "idx": idx,
                    "stage": "extraction",
                    "reason": extract_reason or "Empty prompt",
                    "text_preview": row["text"][:200] + "..."
                })
                if verbose:
                    print(f"  [SKIP idx={idx}] Extraction failed: {extract_reason}")
                continue
            
            # Get model prediction
            pred, raw_output, pred_reason = predict_label(model, tokenizer, prompt)
            
            # Log prediction issues
            if pred is None:
                skipped_prediction += 1
                skipped_details.append({
                    "idx": idx,
                    "stage": "prediction",
                    "reason": pred_reason,
                    "raw_output": raw_output
                })
                if verbose:
                    print(f"  [SKIP idx={idx}] Prediction failed: {pred_reason}")
                continue
            
            # Track fallback usage
            if pred_reason and "Fallback" in pred_reason:
                fallback_used += 1
                if verbose:
                    print(f"  [FALLBACK idx={idx}] {pred_reason}")
            labels.append(label)
            preds.append(pred)

    # Compute metrics
    metrics = compute_classification_metrics(labels, preds)
    total_skipped = skipped_extraction + skipped_prediction
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples in dataset: {total_rows}")
    print(f"Successfully evaluated: {metrics['total']}")
    print(f"Skipped (extraction): {skipped_extraction}")
    print(f"Skipped (prediction): {skipped_prediction}")
    print(f"Fallback predictions used: {fallback_used}")
    print()
    print("Weighted Metrics (recommended for imbalanced data):")
    print(
        f"  accuracy={metrics['accuracy_weighted']:.4f}, "
        f"precision={metrics['precision_weighted']:.4f}, "
        f"recall={metrics['recall_weighted']:.4f}, "
        f"f1={metrics['f1_weighted']:.4f}"
    )
    print("Macro Metrics (unweighted average):")
    print(
        f"  accuracy={metrics['accuracy_macro']:.4f}, "
        f"  precision={metrics['precision_macro']:.4f}, "
        f"recall={metrics['recall_macro']:.4f}, "
        f"f1={metrics['f1_macro']:.4f}"
    )
    print("Per-Class Metrics:")
    print(
        f"  Class 0 (Exclude): precision={metrics['precision_class_0']:.4f}, "
        f"recall={metrics['recall_class_0']:.4f}, f1={metrics['f1_class_0']:.4f}"
    )
    print(
        f"  Class 1 (Include): precision={metrics['precision_class_1']:.4f}, "
        f"recall={metrics['recall_class_1']:.4f}, f1={metrics['f1_class_1']:.4f}"
    )
    print(
        "Confusion matrix: "
        f"TP={metrics['tp']}, TN={metrics['tn']}, "
        f"FP={metrics['fp']}, FN={metrics['fn']}"
    )
    print("="*60)
    
    # Save results
    results = {
        "model_name": "lfm-1.2B",
        "metrics": metrics,
        "skipped": {
            "total": total_skipped,
            "extraction_failures": skipped_extraction,
            "prediction_failures": skipped_prediction,
            "fallback_used": fallback_used,
        },
        "test_path": test_path,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Saved evaluation results to {results_path}")
    
    # Save skipped examples for debugging
    if skipped_details:
        with open(skipped_examples_path, "w", encoding="utf-8") as f:
            for detail in skipped_details:
                f.write(json.dumps(detail) + "\n")
        print(f"Saved {len(skipped_details)} skipped examples to {skipped_examples_path}")


def inspect_masking(trainer, tokenizer, sample_index=100):
    print(tokenizer.decode(trainer.train_dataset[sample_index]["input_ids"]))
    masked_labels = [
        tokenizer.pad_token_id if x == -100 else x
        for x in trainer.train_dataset[sample_index]["labels"]
    ]
    print(
        tokenizer.decode(masked_labels).replace(tokenizer.pad_token, " ")
    )


def report_memory_start():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(
        torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
    )
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return start_gpu_memory, max_memory


def report_memory_end(start_gpu_memory, max_memory, trainer_stats):
    used_memory = round(
        torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
    )
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    runtime = trainer_stats.metrics["train_runtime"]
    print(f"{runtime} seconds used for training.")
    print(f"{round(runtime / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(
        "Peak reserved memory for training % of max memory = "
        f"{lora_percentage} %."
    )


def run_inference(model, tokenizer, prompt, max_new_tokens=128):
    print("Inference example:\n")
    print(prompt)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # Increase for longer outputs!
        # Recommended Liquid settings!
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


def run_inference_examples(model, tokenizer):
    run_inference(
        model,
        tokenizer,
        "Assess inclusion for the review. Title: 'ChatGPT as a Software "
        "Development Bot: A Project-Based Study'. Abstract: The study "
        "examines ChatGPT as a support tool for undergraduate software "
        "development projects and evaluates learning outcomes. Output 0 or 1. \n\n",
    )
    run_inference(
        model,
        tokenizer,
        "Assess inclusion for the review. Title: 'Soil biogeochemical models "
        "for climate change'. Abstract: This paper focuses on soil carbon "
        "modeling and parameter estimation, with no mention of generative AI "
        "or undergraduate CS. Output 0 or 1. \n\n",
    )


def save_lora(model, tokenizer, output_dir=SAVED_MODEL_DIR):
    model.save_pretrained(output_dir)  # Local saving
    tokenizer.save_pretrained(output_dir)
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


def load_lora_for_inference(
    lora_dir="lora_model",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
):
    from transformers import Lfm2ForCausalLM

    model, tokenizer = FastModel.from_pretrained(
        model_name=lora_dir,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        auto_model=Lfm2ForCausalLM,
        load_in_4bit=load_in_4bit,
    )
    FastModel.for_inference(model)  # Enable native 2x faster inference
    return model, tokenizer


def load_lora_with_peft(lora_dir="lora_model", load_in_4bit=LOAD_IN_4BIT):
    # This path is slower and lacks 4bit model downloading.
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_dir,  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    return model, tokenizer


def save_merged_and_gguf_examples(model, tokenizer):
    # Saving to float16 for VLLM
    if False:
        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="merged_16bit",
        )
    if False:
        model.push_to_hub_merged(
            "hf/model",
            tokenizer,
            save_method="merged_16bit",
            token="",
        )

    # Merge to 4bit
    if False:
        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="merged_4bit",
        )
    if False:
        model.push_to_hub_merged(
            "hf/model",
            tokenizer,
            save_method="merged_4bit",
            token="",
        )

    # Just LoRA adapters
    if False:
        model.save_pretrained("model")
        tokenizer.save_pretrained("model")
    if False:
        model.push_to_hub("hf/model", token="")
        tokenizer.push_to_hub("hf/model", token="")

    # GGUF / llama.cpp conversion
    if False:
        model.save_pretrained_gguf("model", tokenizer)
    if False:
        model.push_to_hub_gguf("hf/model", tokenizer, token="")

    if False:
        model.save_pretrained_gguf(
            "model", tokenizer, quantization_method="f16"
        )
    if False:
        model.push_to_hub_gguf(
            "hf/model",
            tokenizer,
            quantization_method="f16",
            token="",
        )

    if False:
        model.save_pretrained_gguf(
            "model", tokenizer, quantization_method="q4_k_m"
        )
    if False:
        model.push_to_hub_gguf(
            "hf/model",
            tokenizer,
            quantization_method="q4_k_m",
            token="",
        )

    if False:
        model.push_to_hub_gguf(
            "hf/model",  # Change hf to your username!
            tokenizer,
            quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
            token="",
        )


def main():
    model, tokenizer = load_base_model()
    model = add_lora_adapters(model)

    
    dataset = load_and_prepare_dataset(
        DATA_PATH,
        text_column="train",
        print_sample=True,
    )

    validate_response_masking(dataset, tokenizer)
    trainer = build_trainer(model, tokenizer, dataset)
    trainer = apply_response_only_mask(trainer)
    inspect_masking(trainer, tokenizer)

    start_gpu_memory, max_memory = report_memory_start()
    trainer_stats = trainer.train()
    report_memory_end(start_gpu_memory, max_memory, trainer_stats)

    evaluate_on_testset(model, tokenizer)

    run_inference_examples(model, tokenizer)

    save_lora(model, tokenizer)

    # if False:
    #     model, tokenizer = load_lora_for_inference()

    # run_inference(
    #     model,
    #     tokenizer,
    #     "Assess inclusion for the review. Title: 'ChatGPT-generated feedback "
    #     "for Python students'. Abstract: A quasi-experiment in an "
    #     "undergraduate CS course compares AI-generated reflective feedback "
    #     "to guided reflection, measuring student learning outcomes. Output 0 "
    #     "or 1. \n\n",
    # )

    # if False:
    #     model, tokenizer = load_lora_with_peft()

    # save_merged_and_gguf_examples(model, tokenizer)


if __name__ == "__main__":
    main()