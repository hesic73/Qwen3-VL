"""
Remap 202 least-frequent tokens in the Qwen3 tokenizer to:
  <ACT_0>..<ACT_100>  (101 tokens, for action values)
  <AREA_0>..<AREA_100> (101 tokens, for area/coverage output)

Strategy: take the 202 highest-numbered regular tokens (which BPE assigned last =
least frequent) and rename them in-place in tokenizer.json.
Vocab size stays UNCHANGED - no model resize needed.

Usage:
    python tools/remap_tokenizer.py \
        --base_model Qwen/Qwen3-VL-4B-Instruct \
        --output_dir ./custom_tokenizer

    python tools/remap_tokenizer.py --verify --output_dir ./custom_tokenizer
"""

import argparse
import json
import shutil
from pathlib import Path
from transformers import AutoTokenizer


ACT_TOKENS  = [f"<ACT_{i}>"  for i in range(101)]
AREA_TOKENS = [f"<AREA_{i}>" for i in range(101)]
ALL_NEW_TOKENS = ACT_TOKENS + AREA_TOKENS  # 202 total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output_dir", default="./custom_tokenizer")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def build_tokenizer(base_model, output_dir):
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    tj_path = Path(output_dir) / "tokenizer.json"
    tc_path = Path(output_dir) / "tokenizer_config.json"
    with open(tj_path) as f:
        tj = json.load(f)
    with open(tc_path) as f:
        tc = json.load(f)

    vocab = tj["model"]["vocab"]               # {str: int}
    added_ids = {e["id"] for e in tj.get("added_tokens", [])}

    # 202 highest-ID regular tokens → these become our special tokens
    id2tok = {v: k for k, v in vocab.items()}
    candidates = sorted(
        [i for i in id2tok if i not in added_ids],
        reverse=True,
    )[:202]
    assert len(candidates) == 202, f"Only found {len(candidates)} candidates"

    # Build rename map: old_string -> new_string
    rename = {}
    id2new = {}
    for new_name, old_id in zip(ALL_NEW_TOKENS, candidates):
        old_str = id2tok[old_id]
        rename[old_str] = new_name
        id2new[old_id] = new_name

    # Apply rename in vocab
    tj["model"]["vocab"] = {
        rename.get(k, k): v for k, v in vocab.items()
    }

    # Remove any BPE merge rule whose result string was renamed.
    # After save_pretrained, each merge is stored as [left, right] (result = left+right).
    renamed_old_strings = set(rename.keys())
    kept_merges = []
    for merge in tj["model"].get("merges", []):
        if isinstance(merge, list):
            left, right = merge[0], merge[1]
        else:
            left, right = merge.split(" ", 1)
        if (left + right) in renamed_old_strings:
            continue  # drop this merge
        kept_merges.append(merge)
    removed = len(tj["model"].get("merges", [])) - len(kept_merges)
    tj["model"]["merges"] = kept_merges
    print(f"Removed {removed} merge rules that produced renamed tokens")

    # Add as added_tokens so they are tokenized atomically (not split by BPE)
    existing_added_ids = {e["id"] for e in tj["added_tokens"]}
    for old_id, new_name in sorted(id2new.items()):
        if old_id not in existing_added_ids:
            tj["added_tokens"].append({
                "id": old_id,
                "content": new_name,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            })
    tj["added_tokens"].sort(key=lambda x: x["id"])

    # Update tokenizer_config additional_special_tokens
    existing_special = set(tc.get("additional_special_tokens", []))
    existing_special.update(ALL_NEW_TOKENS)
    tc["additional_special_tokens"] = sorted(existing_special)

    with open(tj_path, "w") as f:
        json.dump(tj, f, ensure_ascii=False, indent=2)
    with open(tc_path, "w") as f:
        json.dump(tc, f, ensure_ascii=False, indent=2)

    print(f"Remapped IDs: {min(candidates)}..{max(candidates)}")
    print(f"Vocab size unchanged: {len(tj['model']['vocab'])}")
    print(f"Example: id={candidates[0]} renamed to {id2new[candidates[0]]!r}")

    # Sanity check via re-loaded tokenizer
    verify(output_dir)


def verify(output_dir):
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    errors = 0
    for token in ALL_NEW_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid == tokenizer.unk_token_id:
            print(f"FAIL: {token!r} mapped to unk_token_id")
            errors += 1
            continue
        decoded = tokenizer.decode([tid])
        if decoded != token:
            print(f"FAIL: {token!r} -> id {tid} -> decoded {decoded!r}")
            errors += 1
    if errors == 0:
        print(f"All {len(ALL_NEW_TOKENS)} tokens verified  (vocab size: {len(tokenizer)})")
    else:
        print(f"{errors} verification errors.")


def main():
    args = parse_args()
    if args.verify:
        verify(args.output_dir)
    else:
        build_tokenizer(args.base_model, args.output_dir)


if __name__ == "__main__":
    main()
