import argparse
import json
import time
from pathlib import Path
import sys
import os
import numpy as np

import torch
from datasets import load_dataset, load_from_disk

# Ensure we can import from the current directory even if launched elsewhere
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

# Import ControlledDecoder from util_decode.py 
from util_decoding.util_decode_new import ControlledDecoder


def _resolve_prompts(ds, dataset_name: str):
    """Normalize to a simple list[str] of prompts."""
    if dataset_name == "Dahoas/full-hh-rlhf":
        return ds["prompt"]
    elif "ultrafeedback" in dataset_name.lower():
        return [f"\n\nHuman: {q}\n\nAssistant:" for q in ds["instruction"]]
    elif "harmfulqa" in dataset_name.lower():
        return [f"\n\nHuman: {q}\n\nAssistant:" for q in ds["question"]]
    elif dataset_name == "stanfordnlp/SHP":
        unique_prompts, seen_posts = [], set()
        for post_id, histr in zip(ds["post_id"], ds["history"]):
            if post_id in seen_posts:
                continue
            unique_prompts.append(f" Human: {histr} Assistant: ")
            seen_posts.add(post_id)
        return unique_prompts
    else:
        # If the split already yields plain strings, try that:
        first = ds[0]
        if isinstance(first, str):
            return list(ds)
        # Otherwise, try common columns:
        for col in ("prompt", "question", "inputs", "text"):
            if col in ds.column_names:
                return ds[col]
        raise ValueError(
            "Could not resolve prompts for this dataset; "
            "specify a supported dataset or adapt the code."
        )


def main():
    ap = argparse.ArgumentParser("Collect outputs with Controlled Decoding")
    # Data
    ap.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    ap.add_argument("--dataset_local_dir", type=str, default=None,
                help="Local offline dataset folder (e.g., /n/.../datasets/HH-RLHF)")
    ap.add_argument("--setting", type=str, required=True,
                choices=["validation", "test", "transfer"],
                help="validation: last run percent% of TRAIN; test: first run percent% of TEST.")
    # ap.add_argument("--run_percent", type=float, default=2.0)
    ap.add_argument("--model_name", type=str, required=True, help="E.g., llama3-8b")
    ap.add_argument("--reward_model", type=str, required=True, help="E.g., skywork")
    ap.add_argument("--evaluation", type=str, required=True)
    ap.add_argument("--out_file", type=str, required=True)

    # ControlledDecoder config
    ap.add_argument("--value_ckpt", type=str, required=True, help="Path to value_agent_epochXX.pt")
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--lambda_coef", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)

    # Decoding mode
    ap.add_argument("--mode", type=str, default="greedy", choices=["greedy", "beam", "sample"])
    ap.add_argument("--num_beams", type=int, default=4)                 # for beam
    ap.add_argument("--sample_temperature", type=float, default=1.0)    # for sample
    ap.add_argument("--sample_top_k", type=int, default=None)           # optional override for sample

    # Generation control
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--skip_if_prompt_too_long", action="store_true",
                    help="Skip prompts that exceed model context instead of attempting anyway.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for test sampling")
    ap.add_argument("--dataset_split", type=str, default=None,
                help="Override the split to load (e.g. 'train'). "
                     "By default: 'train' for validation, 'test' for test/transfer.")

    args = ap.parse_args()
    
    out_arg = args.out_file

    if args.setting == "validation":
        root_folder = "validation"
    elif args.setting == "test":
        root_folder = "outputs"
    elif args.setting == "transfer":
        root_folder = "transfer"
    else:
        raise ValueError(f"Unknown setting: {args.setting}")

    combined = f"{args.model_name}-{args.evaluation}"
    out_path = Path(root_folder) / combined / args.reward_model / out_arg

    if out_path.exists():
        raise SystemExit(f"ERROR: out_file already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing results to: {out_path}")

    # --- load dataset ---
    print(f"[INFO] Setting={args.setting}  dataset={args.dataset}")
    if args.setting == "validation":
        target_split = "train"
    elif "ultrafeedback" in args.dataset.lower():
        target_split = "train"
    else:
        target_split = "test"

    # Allow explicit override (e.g. datasets that only have a 'train' split)
    if args.dataset_split is not None:
        target_split = args.dataset_split

    if args.dataset_local_dir:
        if args.dataset == "stanfordnlp/SHP":
            print(f"[INFO] Loading SHP from disk at {args.dataset_local_dir}")
            ds_dict = load_from_disk(args.dataset_local_dir)  # DatasetDict
            ds = ds_dict[target_split]  # pick 'train' or 'test'
        else:
            # Auto-detect Arrow datasets (saved with save_to_disk) by checking
            # for state.json inside the split subdirectory.
            local_split_dir = os.path.join(args.dataset_local_dir, target_split)
            if os.path.isdir(local_split_dir) and os.path.exists(
                os.path.join(local_split_dir, "state.json")
            ):
                print(f"[INFO] Detected Arrow dataset; loading with load_from_disk: {local_split_dir}")
                ds = load_from_disk(local_split_dir)
            else:
                print(f"[INFO] Loading split = {target_split} locally from {args.dataset_local_dir}")
                ds = load_dataset(args.dataset_local_dir, split=target_split)
    else:
        ds = load_dataset(args.dataset, split=target_split)
        print(f"[INFO] Loading split = {target_split} from Hub: {args.dataset}")
    # ds = load_dataset(args.dataset, split=args.split)
    prompts_all = _resolve_prompts(ds, args.dataset)
    N = len(prompts_all)

    k = 1000 if args.setting == "validation" else 1000
    VAL_POOL = 10_000

    if args.setting == "validation":
        rng = np.random.RandomState(args.seed)
        if "ultrafeedback" in args.dataset.lower():
            pool = prompts_all[-20000:-10000]
            assert len(pool) >= 300, "UltraFeedback validation pool smaller than 300"
            idxs = rng.choice(len(pool), size=300, replace=False)
            prompts = [pool[i] for i in idxs]
            slice_desc = f"train[-20000:-10000], random 300 (seed={args.seed})"
        else:
            pool = prompts_all[-VAL_POOL:]
            assert len(pool) >= k, "Validation pool smaller than k"
            idxs = rng.choice(len(pool), size=k, replace=False)
            prompts = [pool[i] for i in idxs]
            slice_desc = f"train last {VAL_POOL}, random {k} (seed={args.seed})"
    else:
        rng = np.random.RandomState(args.seed)
        if "ultrafeedback" in args.dataset.lower():
            pool = prompts_all[-10000:]
            assert len(pool) >= k, "UltraFeedback test pool smaller than k"
            idxs = rng.choice(len(pool), size=k, replace=False)
            prompts = [pool[i] for i in idxs]
            slice_desc = f"train last 10000, random {k} (seed={args.seed})"
        else:
            idxs = rng.permutation(len(prompts_all))[:k]
            prompts = [prompts_all[i] for i in idxs]
            slice_desc = f"test random {k} samples (seed={args.seed})"

    print(f"[INFO] Using {len(prompts)} prompts from {slice_desc} (N={N}).")
    # --- build controlled decoder ---
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    print(f"[INFO] Loading ControlledDecoder model_id={args.model_id}")
    dec = ControlledDecoder(
        value_ckpt_path=args.value_ckpt,
        model_id=args.model_id,
        device=None if args.device == "auto" else args.device,
        dtype=dtype_map[args.dtype],
        lambda_coef=args.lambda_coef,
        top_k=args.top_k,
        temperature=args.temperature,
        debug = True,
        debug_max_steps = 12
    )

    # --- helper for one prompt ---
    def run_one(prompt: str):
        t0 = time.time()
        if args.mode == "greedy":
            full_text = dec.decode_greedy(
                prompt,
                max_new_tokens=args.max_new_tokens,
                stop_on_eos=True,
            )
        elif args.mode == "beam":
            full_text = dec.decode_beam(
                prompt,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                stop_on_eos=True,
            )
        else:  # "sample"
            full_text = dec.decode_sample(
                prompt,
                max_new_tokens=args.max_new_tokens,
                stop_on_eos=True,
                sample_temperature=args.sample_temperature,
                top_k=args.sample_top_k,  # falls back to self.top_k if None
            )
        elapsed = time.time() - t0

        # Derive continuation robustly (decoder may return full text including prompt)
        if isinstance(full_text, str) and full_text.startswith(prompt):
            continuation = full_text[len(prompt):]
        else:
            # Fall back to treating the returned text as the continuation itself
            continuation = full_text

        return continuation, elapsed

    # --- iterate & write JSONL incrementally ---
    num_skipped = 0
    num_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            try:
                result_text, elapsed = run_one(prompt)
                obj = {
                    "prompt": prompt,
                    "result": result_text,                   # continuation only
                    "response": f"{prompt}{result_text}",    # full string for convenience
                    "elapsed": elapsed,
                    # provenance
                    "mode": args.mode,
                    "model_id": args.model_id,
                    "lambda_coef": args.lambda_coef,
                    "top_k": args.top_k,
                    "temperature": args.temperature,
                    "num_beams": args.num_beams if args.mode == "beam" else None,
                    "sample_temperature": args.sample_temperature if args.mode == "sample" else None,
                    "sample_top_k": args.sample_top_k if args.mode == "sample" else None,
                    "max_new_tokens": args.max_new_tokens,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                num_written += 1
            except RuntimeError as e:
                # Optionally skip on context overflow or CUDA OOM
                print(f"[WARN] Skipped idx={i}: {e}")
                num_skipped += 1
                continue

    print(f"[INFO] Done. Wrote: {out_path}  (written={num_written}, skipped={num_skipped})")


if __name__ == "__main__":
    main()