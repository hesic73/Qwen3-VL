"""
Evaluation script for Qwen3-VL finetuned on cloth coverage prediction.

Usage:
    # Finetuned model only
    python tools/evaluate.py \
        --model_path ./output/qwen3vl_lora_run/checkpoint-xxx \
        --base_model Qwen/Qwen3-VL-4B-Instruct \
        --test_file /path/to/test.jsonl \
        --image_dir /path/to/images

    # Compare finetuned vs base model
    python tools/evaluate.py \
        --model_path ./output/qwen3vl_lora_run/checkpoint-xxx \
        --base_model Qwen/Qwen3-VL-4B-Instruct \
        --test_file /path/to/test.jsonl \
        --image_dir /path/to/images \
        --compare_base
"""

import argparse
import json
import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to LoRA checkpoint dir")
    parser.add_argument("--base_model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--within_n", type=int, default=5)
    parser.add_argument("--compare_base", action="store_true", help="Also evaluate base model without LoRA")
    parser.add_argument("--plot_out", default="eval_result.png")
    return parser.parse_args()


def extract_integer(text: str):
    """Extract first integer in [0, 100] from model output. Returns None if not found."""
    m = re.search(r"\b(\d{1,3})\b", text.strip())
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val
    return None


def build_messages(item, image_dir: Path):
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]
    image_pool = [{"type": "image", "image": str(image_dir / img)} for img in images]

    messages = []
    for turn in item["conversations"]:
        if turn["from"] != "human":
            continue
        content = []
        for seg in re.split(r"(<image>)", turn["value"]):
            if seg == "<image>":
                content.append(image_pool.pop(0))
            elif seg.strip():
                content.append({"type": "text", "text": seg.strip()})
        messages.append({"role": "user", "content": content})
    return messages


@torch.inference_mode()
def predict(model, processor, messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=16)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_evaluation(model, processor, samples, image_dir, label=""):
    preds, gts, invalid = [], [], []
    for item in tqdm(samples, desc=label):
        gt_text = next(t["value"] for t in item["conversations"] if t["from"] != "human")
        gt = extract_integer(gt_text)
        if gt is None:
            continue
        raw = predict(model, processor, build_messages(item, image_dir))
        pred = extract_integer(raw)
        if pred is None:
            invalid.append(raw)
        else:
            preds.append(pred)
            gts.append(gt)
    return preds, gts, invalid


def print_metrics(preds, gts, invalid, n_total, within_n, label=""):
    n_invalid = len(invalid)
    n_valid = len(preds)
    print(f"\n{'='*50}")
    print(f"[{label}]")
    print(f"Total:           {n_total}")
    print(f"Invalid outputs: {n_invalid} ({100*n_invalid/n_total:.1f}%)")
    if n_valid == 0:
        return None
    errors = [abs(p - g) for p, g in zip(preds, gts)]
    mae = sum(errors) / n_valid
    exact = sum(1 for e in errors if e == 0) / n_valid
    within = sum(1 for e in errors if e <= within_n) / n_valid
    print(f"MAE:             {mae:.2f}")
    print(f"Exact match:     {100*exact:.1f}%")
    print(f"Within-{within_n}:      {100*within:.1f}%")
    if invalid:
        print(f"Sample invalid outputs: {[repr(o) for o in invalid[:3]]}")
    return {"errors": errors, "mae": mae, "exact": exact, "within": within, "preds": preds, "gts": gts}


def plot_results(results_list, within_n, plot_out):
    """
    results_list: list of (label, metrics_dict)
    Left panel: error histogram (one per model)
    Right panel: pred vs gt scatter (one per model)
    """
    n_models = len(results_list)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models), squeeze=False)

    for row, (label, m) in enumerate(results_list):
        errors = m["errors"]
        preds = m["preds"]
        gts = m["gts"]

        # Left: error histogram
        ax = axes[row][0]
        ax.hist(errors, bins=range(0, max(errors) + 2), color="steelblue", edgecolor="white")
        ax.axvline(within_n, color="red", linestyle="--", label=f"within-{within_n}")
        ax.set_xlabel("|pred - gt|")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} — Error Histogram\nMAE={m['mae']:.2f}, Within-{within_n}={100*m['within']:.1f}%")
        ax.legend()

        # Right: pred vs gt scatter
        # How to read: each dot is one sample.
        # X = ground truth, Y = prediction.
        # Dots on the red line = perfect prediction.
        # Dots forming a horizontal band = model ignores input and guesses a fixed number.
        # Dots spread along the diagonal = model has learned the mapping.
        ax = axes[row][1]
        ax.scatter(gts, preds, alpha=0.15, s=8, color="steelblue")
        ax.plot([0, 100], [0, 100], "r--", linewidth=1.2, label="perfect (y=x)")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{label} — Pred vs GT\n(dots on diagonal = good; horizontal band = model guessing)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    print(f"\nPlot saved to: {plot_out}")


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)

    with open(args.test_file) as f:
        samples = [json.loads(l) for l in f]
    if args.max_samples:
        samples = samples[: args.max_samples]

    processor = AutoProcessor.from_pretrained(args.base_model)
    results_to_plot = []

    # --- Base model (optional) ---
    if args.compare_base:
        print(f"\nLoading base model (no LoRA): {args.base_model}")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        base_model.eval()
        preds, gts, invalid = run_evaluation(base_model, processor, samples, image_dir, label="Base")
        m = print_metrics(preds, gts, invalid, len(samples), args.within_n, label="Base Model")
        if m:
            results_to_plot.append(("Base Model", m))
        del base_model
        torch.cuda.empty_cache()

    # --- Finetuned model ---
    print(f"\nLoading finetuned model: {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()
    preds, gts, invalid = run_evaluation(model, processor, samples, image_dir, label="Finetuned")
    m = print_metrics(preds, gts, invalid, len(samples), args.within_n, label="Finetuned")
    if m:
        results_to_plot.append(("Finetuned", m))

    if results_to_plot:
        plot_results(results_to_plot, args.within_n, args.plot_out)


if __name__ == "__main__":
    main()
