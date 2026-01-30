"""
Nemotron-3-Nano-30B-A3B SFT with Unsloth.

Cleaned from the original Colab notebook while preserving the linear flow.
"""

import json
import os
import re

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "unsloth/Nemotron-3-Nano-30B-A3B"
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
DATA_PATH = "data/train_data_LFM.csv"
TEST_DATA_PATH = "data/test_data_LFM.csv"
RANDOM_STATE = 3407
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
END_PART = "<|im_end|>"
EVAL_RESULTS_PATH = "results/eval_results_nemotron.json"
SKIPPED_EXAMPLES_PATH = "results/skipped_examples_nemotron.jsonl"
SAVED_MODEL_DIR = "models/nemotron_3_nano_30b_a3b_for_sys_review"

FOURBIT_MODELS = [
    "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit",
]


def load_base_model(
    model_name=MODEL_NAME,
    dtype=DTYPE,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_8bit=LOAD_IN_8BIT,
    full_finetuning=False,
):
    global tokenizer, model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=full_finetuning,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        # token = "hf_...", # use one if using gated models
    )
    return model, tokenizer


def add_lora_adapters(model, random_state=RANDOM_STATE):
    return FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj",
            "out_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
        use_rslora=False,
        loftq_config=None,
    )


def normalize_text_column(example, column):
    text = example[column]
    text = text.replace("\\n", "\n").replace("/n", "\n")
    if text.startswith("<|startoftext|>"):
        text = text.replace("<|startoftext|>", "", 1)
    return {"text": text}


def load_and_prepare_dataset(data_path, text_column, sample_index=0, print_sample=False):
    dataset = load_dataset("csv", data_files=data_path, split="train")
    dataset = dataset.map(lambda ex: normalize_text_column(ex, text_column))
    if print_sample:
        print(dataset[sample_index]["text"])
    return dataset


def validate_nemotron_format(dataset, sample_index=0):
    total = len(dataset)
    starts_with_user = 0
    has_startoftext = 0
    for row in dataset:
        text = row["text"]
        if text.startswith("<|im_start|>user\n"):
            starts_with_user += 1
        if "<|startoftext|>" in text:
            has_startoftext += 1
    print(
        "Nemotron format check: "
        f"{starts_with_user}/{total} start with '<|im_start|>user\\n', "
        f"{has_startoftext} contain '<|startoftext|>'"
    )
    if starts_with_user == 0:
        print(
            "WARNING: No samples start with the Nemotron user prefix. "
            "Check template/normalization."
        )
    if has_startoftext:
        print(
            "WARNING: Found '<|startoftext|>' tokens. "
            "These should be stripped for Nemotron."
        )
    if os.environ.get("FORMAT_FAIL_ON_INVALID", "0") == "1":
        if starts_with_user != total or has_startoftext:
            raise RuntimeError("Nemotron format validation failed.")


def build_trainer(model, tokenizer, dataset):
    config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=200,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=RANDOM_STATE,
        report_to="none",
    )
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
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


def parse_binary_label(text):
    match = re.search(r"</think>\\s*([01])", text)
    if match:
        return int(match.group(1))
    matches = re.findall(r"\\b[01]\\b", text)
    if matches:
        return int(matches[-1])
    for ch in reversed(text):
        if ch in ("0", "1"):
            return int(ch)
    return None


def extract_prompt_and_label(text):
    user_match = re.search(
        r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)",
        text,
        flags=re.S,
    )
    if not user_match:
        return None, None, "missing_user"
    user_text = user_match.group(1).strip()

    assistant_match = re.search(
        r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)",
        text,
        flags=re.S,
    )
    if not assistant_match:
        return user_text, None, "missing_assistant"
    response_text = assistant_match.group(1).strip()

    label = parse_binary_label(response_text)
    if label is None:
        return user_text, None, "missing_label"
    return user_text, label, None


def predict_label(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=False,
        temperature=0.0,
        use_cache=False,
    )
    gen_tokens = output[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    decoded = decoded.replace("<|im_end|>", "").strip()
    label = parse_binary_label(decoded)
    return label, decoded


def compute_classification_metrics(labels, preds):
    total = len(labels)
    tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(preds, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels))

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
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
    skipped_path=SKIPPED_EXAMPLES_PATH,
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
    skipped = 0
    skip_reasons = {}
    skipped_examples = []
    max_skip_log = int(os.environ.get("EVAL_MAX_SKIPPED_LOG", "5") or 5)

    model.eval()
    with torch.no_grad():
        for row in dataset:
            if max_samples and len(labels) >= max_samples:
                break
            prompt, label, reason = extract_prompt_and_label(row["text"])
            if reason or not prompt:
                skipped += 1
                skip_reasons[reason or "missing_prompt"] = (
                    skip_reasons.get(reason or "missing_prompt", 0) + 1
                )
                if len(skipped_examples) < max_skip_log:
                    skipped_examples.append(
                        {
                            "reason": reason or "missing_prompt",
                            "text_snippet": row["text"][:500],
                        }
                    )
                continue
            pred, _ = predict_label(model, tokenizer, prompt)
            if pred is None:
                skipped += 1
                skip_reasons["missing_prediction"] = (
                    skip_reasons.get("missing_prediction", 0) + 1
                )
                if len(skipped_examples) < max_skip_log:
                    skipped_examples.append(
                        {
                            "reason": "missing_prediction",
                            "text_snippet": row["text"][:500],
                        }
                    )
                continue
            labels.append(label)
            preds.append(pred)

    metrics = compute_classification_metrics(labels, preds)
    print(
        "Evaluation metrics: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"total={metrics['total']}, skipped={skipped}"
    )
    print(
        "Confusion matrix: "
        f"tp={metrics['tp']}, tn={metrics['tn']}, "
        f"fp={metrics['fp']}, fn={metrics['fn']}"
    )
    if skip_reasons:
        print(f"Skipped breakdown: {skip_reasons}")
        with open(skipped_path, "w", encoding="utf-8") as f:
            for item in skipped_examples:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        print(f"Saved skipped examples to {skipped_path}")
    results = {
        "model_name": "nemotron-3-nano-30b-a3b",
        "metrics": metrics,
        "skipped": skipped,
        "skip_reasons": skip_reasons,
        "test_path": test_path,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Saved evaluation results to {results_path}")
    if skipped and os.environ.get("EVAL_FAIL_ON_SKIPS", "0") == "1":
        raise RuntimeError(
            f"Evaluation skipped {skipped} rows. See {skipped_path}."
        )


def run_inference(model, tokenizer, prompt, max_new_tokens=4096):
    print("Inference example:")
    print(prompt)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        min_p=0.15,
        repetition_penalty=1.05,
        use_cache=False,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


def run_inference_examples(model, tokenizer):
    run_inference(
        model,
        tokenizer,
        "Assess inclusion for the review. Title: 'ChatGPT as a Software "
        "Development Bot: A Project-Based Study'. Abstract: The study "
        "examines ChatGPT as a support tool for undergraduate software "
        "development projects and evaluates learning outcomes. Output 0 or 1.",
    )
    run_inference(
        model,
        tokenizer,
        "Assess inclusion for the review. Title: 'Soil biogeochemical models "
        "for climate change'. Abstract: This paper focuses on soil carbon "
        "modeling and parameter estimation, with no mention of generative AI "
        "or undergraduate CS. Output 0 or 1.",
    )


def save_lora(model, tokenizer, output_dir=SAVED_MODEL_DIR):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


def load_lora_for_inference(
    lora_dir=SAVED_MODEL_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def save_merged_and_gguf_examples(model, tokenizer):
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

    if False:
        model.save_pretrained("model")
        tokenizer.save_pretrained("model")
    if False:
        model.push_to_hub("hf/model", token="")
        tokenizer.push_to_hub("hf/model", token="")

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
            "hf/model",
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
    validate_nemotron_format(dataset)

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

    if False:
        model, tokenizer = load_lora_for_inference()

    run_inference(
        model,
        tokenizer,
        "Assess inclusion for the review. Title: 'ChatGPT-generated feedback "
        "for Python students'. Abstract: A quasi-experiment in an "
        "undergraduate CS course compares AI-generated reflective feedback "
        "to guided reflection, measuring student learning outcomes. Output 0 "
        "or 1.",
    )

    save_merged_and_gguf_examples(model, tokenizer)


if __name__ == "__main__":
    main()
