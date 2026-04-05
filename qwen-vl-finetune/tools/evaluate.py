"""
Evaluation script for Qwen3-VL finetuned on cloth coverage prediction.

Usage:
    # Finetuned model only
    python tools/evaluate.py \
        --model_path ./output/qwen3vl_lora_run/checkpoint-xxx \
        --base_model Qwen/Qwen3-VL-4B-Instruct \
        --test_file /path/to/test.jsonl

    # Compare finetuned vs base model
    python tools/evaluate.py \
        --model_path ./output/qwen3vl_lora_run/checkpoint-xxx \
        --base_model Qwen/Qwen3-VL-4B-Instruct \
        --test_file /path/to/test.jsonl \
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
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--within_n", type=int, default=5)
    parser.add_argument("--tokenizer_path", default=None, help="Custom tokenizer dir (e.g. ./custom_tokenizer for remapped tokens)")
    parser.add_argument("--compare_base", action="store_true", help="Also evaluate base model without LoRA")
    parser.add_argument(
        "--output_mode", default="special", choices=["special", "integer"],
        help="(\'special\') expect <AREA_N> tokens from finetuned model; "
             "(\'integer\') expect bare 0-100 integers from base/plain model",
    )
    parser.add_argument(
        "--base_output_mode", default="integer", choices=["special", "integer"],
        help="output_mode used when evaluating the base model (default: \'integer\')",
    )
    parser.add_argument("--plot_out", default="eval_result.png")
    parser.add_argument("--lora", action="store_true", help="Load finetuned model as a LoRA adapter (default: full finetune)")
    return parser.parse_args()


def extract_special_token(text: str):
    """Strict: accept only <AREA_N> special tokens (0-100). Returns None for anything else."""
    m = re.search(r"<AREA_(\d{1,3})>", text.strip())
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val
    return None


def extract_plain_int(text: str):
    """Strict: accept only a bare integer in [0, 100]. Returns None for anything else."""
    m = re.fullmatch(r"\s*(\d{1,3})\s*", text.strip())
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val
    return None


def make_extractor(output_mode: str):
    """Return the right extraction function for the specified output_mode.

    output_mode choices:
      'special'  – expect <AREA_N> tokens (finetuned with remapped tokenizer)
      'integer'  – expect a bare integer 0-100 (base model / plain prompting)
    """
    if output_mode == "special":
        return extract_special_token
    elif output_mode == "integer":
        return extract_plain_int
    else:
        raise ValueError(f"Unknown output_mode {output_mode!r}. Choose 'special' or 'integer'.")


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
    # Don't skip special tokens: <AREA_N> is a special token in the remapped tokenizer
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()


def run_evaluation(model, processor, samples, image_dir, label="", output_mode="special"):
    extractor = make_extractor(output_mode)
    # GT is always in <AREA_N> format in the dataset
    gt_extractor = extract_special_token
    preds, gts, invalid = [], [], []
    for item in tqdm(samples, desc=label):
        gt_text = next(t["value"] for t in item["conversations"] if t["from"] != "human")
        gt = gt_extractor(gt_text)
        if gt is None:
            continue
        raw = predict(model, processor, build_messages(item, image_dir))
        pred = extractor(raw)
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
    image_dir = Path(args.test_file).parent  # Assume images are in the same dir as test_file

    with open(args.test_file) as f:
        samples = [json.loads(l) for l in f]
    if args.max_samples:
        samples = samples[: args.max_samples]

    processor = AutoProcessor.from_pretrained(args.base_model)
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        processor.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    results_to_plot = []

    # --- Base model (optional) ---
    if args.compare_base:
        print(f"\nLoading base model (no LoRA): {args.base_model}")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        base_model.eval()
        preds, gts, invalid = run_evaluation(base_model, processor, samples, image_dir, label="Base", output_mode=args.base_output_mode)
        m = print_metrics(preds, gts, invalid, len(samples), args.within_n, label=f"Base Model [{args.base_output_mode}]")
        if m:
            results_to_plot.append(("Base Model", m))
        del base_model
        torch.cuda.empty_cache()

    # --- Finetuned model ---
    print(f"\nLoading finetuned model: {args.model_path}")
    if args.lora:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
    preds, gts, invalid = run_evaluation(model, processor, samples, image_dir, label="Finetuned", output_mode=args.output_mode)
    m = print_metrics(preds, gts, invalid, len(samples), args.within_n, label=f"Finetuned [{args.output_mode}]")
    if m:
        results_to_plot.append(("Finetuned", m))

    if results_to_plot:
        plot_results(results_to_plot, args.within_n, args.plot_out)


if __name__ == "__main__":
    main()
