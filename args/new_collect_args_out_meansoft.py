from datasets import load_dataset, load_from_disk
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from new_argsearch_meansoft import ARGS  
import os
import time
import sys
import random
import numpy as np
import torch

# -------------------- CLI --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
parser.add_argument("--rm", type=str, required=True)
parser.add_argument("--llm", type=str, required=True)
parser.add_argument("--max_new_token", type=int, default=128)

parser.add_argument("--llm_gpu", type=str, default="cuda:0")
parser.add_argument("--rm_gpu", type=str, default="cuda:1")
parser.add_argument("--recover", action="store_true", default=False)

parser.add_argument("--config", type=str, required=True, help="JSONL file of run configs (one per line)")
parser.add_argument("--out_file", type=str, required=True, help="Output prefix (we write <prefix>_{i}.jsonl)")
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--setting", type=str, default="test", choices=["validation", "test"])
parser.add_argument("--K", type=int, default=0)
parser.add_argument("--dataset_local_dir", type=str, default=None)

parser.add_argument("--B", type=float, required=True, help="Soft shaping magnitude B")
parser.add_argument("--alpha", type=float, required=True, help="Soft shaping steepness alpha")
parser.add_argument("--noise", action="store_true", default=False,
                    help="If set, apply Gaussian noise to rewards before fusion.")
parser.add_argument("--noise_var", type=float, default=1.0,
                    help="Variance of the Gaussian noise added to rewards (default: 1.0).")

args = parser.parse_args()
print(f"{args=}")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.max_new_token <= 0:
    print("ERROR: --max_new_token must be > 0", file=sys.stderr)
    sys.exit(1)

if args.B <= 0:
    print("ERROR: --B must be > 0", file=sys.stderr)
    sys.exit(1)

if args.alpha <= 0:  
    print("ERROR: --alpha must be > 0", file=sys.stderr)  
    sys.exit(1) 

cfg_path = Path(args.config)
if not cfg_path.exists():
    print("ERROR: --config not found", file=sys.stderr)
    sys.exit(1)

with cfg_path.open("r", encoding="utf-8") as f:
    run_configs = [json.loads(line) for line in f if line.strip()]

required_keys = {"rm_weight", "topk", "mode", "sample_temp"}
for i, rc in enumerate(run_configs):
    miss = required_keys - set(rc.keys())
    if miss:
        print(f"ERROR: config line {i} missing keys: {miss}", file=sys.stderr)
        sys.exit(1)

print(f"[INFO] Loaded {len(run_configs)} run configs")

# -------------------- Load dataset --------------------
ds_split = "train" if args.setting == "validation" else "test"

raw_ds = None
if args.dataset_local_dir:
    local_path = Path(args.dataset_local_dir)
    if local_path.exists():
        try:
            if args.dataset == "stanfordnlp/SHP":
                ds_dict = load_from_disk(str(local_path))
                raw_ds = ds_dict[ds_split]
            else:
                raw_ds = load_dataset(str(local_path), split=ds_split)
        except Exception as e:
            print(f"[WARN] Failed to load local dataset from {local_path}: {e}", file=sys.stderr)
            raw_ds = None

if raw_ds is None:
    raw_ds = load_dataset(args.dataset, split=ds_split)

if args.dataset == "Dahoas/full-hh-rlhf":
    ds_list = raw_ds["prompt"]
elif args.dataset == "stanfordnlp/SHP":
    unique_prompts, seen = [], set()
    for post_id, histr in zip(raw_ds["post_id"], raw_ds["history"]):
        if post_id in seen:
            continue
        unique_prompts.append(" Human: " + histr + " Assistant: ")
        seen.add(post_id)
    ds_list = unique_prompts
else:
    if isinstance(raw_ds, list):
        ds_list = raw_ds
    elif "prompt" in raw_ds.column_names:
        ds_list = raw_ds["prompt"]
    else:
        print("ERROR: could not infer prompt field for this dataset", file=sys.stderr)
        sys.exit(1)

K = int(args.K)
if K > 0:
    selected_ds = ds_list[-K:] if args.setting == "validation" else ds_list[:K]
else:
    # Default number of prompts differs by validation/test**
    if args.setting == "validation":
        selected_ds = ds_list[-300:]
    else:
        n = 1000 if K <= 0 else K
        rng = np.random.RandomState(args.seed)  # fixed given --seed
        idxs = rng.permutation(len(ds_list))[:n]
        selected_ds = [ds_list[i] for i in idxs]

print(f"[INFO] After (setting,K) selection: {len(selected_ds)} prompts (K={K}, setting={args.setting})")
truncated_ds = selected_ds

# -------------------- Models --------------------
print(f"[INFO] Loading models (llm={args.llm}, rm={args.rm})")
search = ARGS(llm_path=args.llm, rm_path=args.rm, llm_dev=args.llm_gpu, rm_dev=args.rm_gpu)
print("[INFO] Models ready")

# -------------------- Helpers --------------------
def _extract_tokens(out):
    if out is None:
        return None
    if isinstance(out, tuple):
        return out[0]
    return out

def _empty_tokens(tok):
    if tok is None:
        return True
    try:
        if hasattr(tok, "numel"):
            return tok.numel() == 0
        if isinstance(tok, (list, tuple)):
            return len(tok) == 0
    except Exception:
        pass
    return False

def runprompt(prompt: str, rm_weight=0.0, topk=5, new_token=24, mode="greedy_large", sample_temp=None,
              noise: bool = False, noise_var: float = 1.0):
    temp = sample_temp if sample_temp is not None else 0.7

    out = search.generate(
        prompt,
        method=mode,
        topk=topk,
        max_new_token=new_token,
        temperature=temp,
        weight=rm_weight,             # keep config meaning
        reward_shaping=True,
        B=float(args.B),
        beta=None,                    # keep your convention (weight is lambda)
        alpha=float(args.alpha),
        seed=args.seed,
        debug=False,
        noise=noise, noise_var=noise_var,
    )

    tokens = _extract_tokens(out)
    if _empty_tokens(tokens):
        return None, False

    try:
        text_out = search.tokens_to_text(tokens)
        text = text_out[0] if isinstance(text_out, (list, tuple)) else str(text_out)
    except Exception as e:
        print(f"[WARN] tokens_to_text failed: {e}", file=sys.stderr)
        return None, False

    pure = text[len(prompt):] if text.startswith(prompt) else text
    return pure, True

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# -------------------- Main loop per run-config --------------------
for config_num, rc in enumerate(run_configs):
    out_path = Path(f"{args.out_file}.jsonl")
    print(f"[INFO] Config {config_num}: {rc}")
    print(f"[INFO] Writing: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.recover:
        print(f"ERROR: {out_path} exists. Use --recover to append.", file=sys.stderr)
        sys.exit(1)

    start_idx = count_lines(out_path) if args.recover else 0
    if args.recover and start_idx > 0:
        print(f"[INFO] Resuming at index {start_idx}/{len(truncated_ds)}")

    written = 0
    skipped = 0
    last_flush = time.time()

    with out_path.open("a", encoding="utf-8") as fout:
        for idx in tqdm(range(start_idx, len(truncated_ds))):
            prompt = truncated_ds[idx]
            result, ok = runprompt(
                prompt,
                float(rc["rm_weight"]),
                rc["topk"],
                args.max_new_token,
                rc["mode"],
                rc["sample_temp"],
                noise=args.noise,
                noise_var=args.noise_var,
            )
            if not ok or not result:
                skipped += 1
                if skipped <= 10 or skipped % 100 == 0:
                    print(f"[INFO] idx={idx} skipped (generation failed/empty)")
                continue

            fout.write(json.dumps({"prompt": prompt, "result": result}, ensure_ascii=False) + "\n")
            written += 1

            if (idx + 1) % 10 == 0 or (time.time() - last_flush) > 5.0:
                try:
                    fout.flush()
                    os.fsync(fout.fileno())
                except Exception:
                    pass
                last_flush = time.time()

        try:
            fout.flush()
            os.fsync(fout.fileno())
        except Exception:
            pass

    print(f"[INFO] Done: {out_path} (wrote {written} lines, skipped {skipped})")
