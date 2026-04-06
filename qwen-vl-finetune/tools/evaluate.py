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
    parser.add_argument("--model_path", required=True, help="Path to checkpoint dir")
    parser.add_argument("--base_model", default=None, help="Path to base model (required for LoRA or compare_base)")
    parser.add_argument("--dataset_path", required=True, help="Path to evaluation dataset jsonl file")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--within_n", type=int, default=5)
    parser.add_argument("--tokenizer_path", default=None, help="Custom tokenizer dir")
    parser.add_argument(
        "--label_format", choices=["special", "integer"], required=True,
    )
    parser.add_argument("--plot_out", default="eval_result.png")
    parser.add_argument("--lora", action="store_true", help="Load finetuned model as a LoRA adapter")
    
    args = parser.parse_args()
    if args.lora and not args.base_model:
        parser.error("--base_model is required when using --lora")
    return args


def extract_special_token(text: str):
    m = re.search(r"<AREA_(\d{1,3})>", text.strip())
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val
    return None


def extract_plain_int(text: str):
    m = re.fullmatch(r"\s*(\d{1,3})\s*", text.strip())
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val
    return None


def make_extractor(label_format: str):
    if label_format == "special":
        return extract_special_token
    elif label_format == "integer":
        return extract_plain_int
    else:
        raise ValueError(f"Unknown format {label_format}")


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
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()


def run_evaluation(model, processor, samples, image_dir, label="", label_format="special"):
    extractor = make_extractor(label_format)
    gt_extractor = extractor
    preds, gts, invalid = [], [], []
    for item in tqdm(samples, desc=label):
        gt_text = next(t["value"] for t in item["conversations"] if t["from"] != "human")
        gt = gt_extractor(gt_text)
        if gt is None:
            raise ValueError(f"Dataset Ground Truth {gt_text!r} does not match the expected label_format '{label_format}'. "
                             "Please check your dataset or pass the correct --label_format.")
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
    print(f"Valid outputs:   {n_valid} ({100*n_valid/n_total:.1f}% success rate)")
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
    n_models = len(results_list)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models), squeeze=False)

    for row, (label, m) in enumerate(results_list):
        errors = m["errors"]
        preds = m["preds"]
        gts = m["gts"]

        ax = axes[row][0]
        ax.hist(errors, bins=range(0, max(errors) + 2), color="steelblue", edgecolor="white")
        ax.axvline(within_n, color="red", linestyle="--", label=f"within-{within_n}")
        ax.set_xlabel("|pred - gt|")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} - Error Histogram\nMAE={m['mae']:.2f}, Within-{within_n}={100*m['within']:.1f}%")
        ax.legend()

        ax = axes[row][1]
        ax.scatter(gts, preds, alpha=0.15, s=8, color="steelblue")
        ax.plot([0, 100], [0, 100], "r--", linewidth=1.2, label="perfect (y=x)")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{label} - Pred vs GT")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    print(f"\nPlot saved to: {plot_out}")


def main():
    args = parse_args()
    image_dir = Path(args.dataset_path).parent

    with open(args.dataset_path) as f:
        samples = [json.loads(l) for l in f]
    if args.max_samples:
        samples = samples[: args.max_samples]

    processor_path = args.base_model if args.base_model else args.model_path
    processor = AutoProcessor.from_pretrained(processor_path)
    
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        processor.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    
    results_to_plot = []


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
    
    preds, gts, invalid = run_evaluation(model, processor, samples, image_dir, label="Finetuned", label_format=args.label_format)
    m = print_metrics(preds, gts, invalid, len(samples), args.within_n, label=f"Finetuned [{args.label_format}]")
    if m:
        results_to_plot.append(("Finetuned", m))

    if results_to_plot:
        plot_results(results_to_plot, args.within_n, args.plot_out)


if __name__ == "__main__":
    main()
