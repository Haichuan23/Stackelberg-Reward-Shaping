from datasets import load_dataset, load_from_disk
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from new_argsearch_cap import ARGS
import math
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
parser.add_argument(
    "--setting",
    type=str,
    default="test",
    choices=["validation", "test"],
    help="validation => last K prompts from TRAIN split; test => first K prompts from TEST split.",
)
parser.add_argument(
    "--K",
    type=int,
    default=0,
    help="Number of prompts to use. If <= 0, use all prompts after selection rule.",
)
parser.add_argument(
    "--dataset_local_dir",
    type=str,
    default=None,
    help=(
        "If set, try to load dataset splits from this local HF 'save_to_disk' directory "
        "or local dataset path. Example: /n/.../datasets/HH-RLHF or /n/.../datasets/SHP"
    ),
)

# ** NEW: optional default cap from CLI (used if config line doesn't specify cap) **
parser.add_argument(
    "--cap",
    type=float,
    default=None,
    help="Reward cap for per-step shifted reward (None disables). Used as default if config lacks 'cap'.",
)

args = parser.parse_args()
print(f"{args=}")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.recover:
    print("[INFO] RECOVERY MODE: same CLI + config expected as the original run.")

if args.max_new_token <= 0:
    print("ERROR: --max_new_token must be > 0", file=sys.stderr)
    sys.exit(1)

cfg_path = Path(args.config)
if not cfg_path.exists():
    print("ERROR: --config not found", file=sys.stderr)
    sys.exit(1)

# -------------------- Load run configs --------------------
with cfg_path.open("r", encoding="utf-8") as f:
    run_configs = [json.loads(line) for line in f if line.strip()]

required_keys = {"rm_weight", "topk", "mode", "sample_temp"}
for i, rc in enumerate(run_configs):
    miss = required_keys - set(rc.keys())
    if miss:
        print(f"ERROR: config line {i} missing keys: {miss}", file=sys.stderr)
        sys.exit(1)

# ** OPTIONAL: allow per-line cap override in config **
# ** e.g., each config JSON can include {"cap": 2.0} **
for i, rc in enumerate(run_configs):
    if "cap" in rc and rc["cap"] is not None:
        try:
            rc["cap"] = float(rc["cap"])
        except Exception:
            print(f"ERROR: config line {i} has invalid cap={rc['cap']}", file=sys.stderr)
            sys.exit(1)

print(f"[INFO] Loaded {len(run_configs)} run configs")

# -------------------- Load dataset --------------------
if args.setting == "validation":
    ds_split = "train"
else:
    ds_split = "test"

print(
    f"[INFO] Loading dataset (dataset={args.dataset}, split={ds_split}, setting={args.setting}, "
    f"dataset_local_dir={args.dataset_local_dir})"
)

raw_ds = None

if args.dataset_local_dir:
    local_path = Path(args.dataset_local_dir)
    if local_path.exists():
        try:
            if args.dataset == "stanfordnlp/SHP":
                print(f"[INFO] Loading SHP from disk at {local_path}")
                ds_dict = load_from_disk(str(local_path))
                raw_ds = ds_dict[ds_split]
            else:
                print(f"[INFO] Loading split={ds_split} locally via load_dataset from {local_path}")
                raw_ds = load_dataset(str(local_path), split=ds_split)
        except Exception as e:
            print(f"[WARN] Failed to load local dataset from {local_path}: {e}", file=sys.stderr)
            raw_ds = None
    else:
        print(f"[WARN] dataset_local_dir does not exist: {local_path} (falling back to Hub)", file=sys.stderr)

if raw_ds is None:
    print(f"[INFO] Loading split={ds_split} from Hub: {args.dataset}")
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

def runprompt(
    prompt: str,
    rm_weight=0.0,
    topk=5,
    new_token=24,
    mode="p_sigmoid_mixing",
    sample_temp=None,
    llm_dev: str = "cuda:0",
    cap: float = None,  
):
    """
    Returns (pure_text, ok_flag). We DO NOT return/store tokens.
    """
    temp = sample_temp if sample_temp is not None else 0.7

    # ** NEW: pass cap into search.generate **
    out = search.generate(
        prompt,
        method=mode,
        topk=topk,
        max_new_token=new_token,
        temperature=temp,
        weight=rm_weight,
        debug=False,
        cap=cap,  
    )

    tokens = _extract_tokens(out)
    if _empty_tokens(tokens):
        return None, False

    try:
        text_out = search.tokens_to_text(tokens)
        if isinstance(text_out, (list, tuple)):
            text = text_out[0]
        else:
            text = str(text_out)
    except Exception as e:
        print(f"[WARN] tokens_to_text failed: {e}", file=sys.stderr)
        return None, False

    try:
        pure = text.removeprefix(prompt)
    except AttributeError:
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

    # ** NEW: resolve cap for this config (config overrides CLI default) **
    cap_this = rc.get("cap", args.cap)  
    if cap_this is not None:
        cap_this = float(cap_this) 
    print(f"[INFO] Using cap={cap_this} for this run-config") 

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
                llm_dev=args.llm_gpu,
                cap=cap_this,  
            )
            if not ok or result is None or result == "":
                skipped += 1
                if skipped <= 10 or skipped % 100 == 0:
                    print(f"[INFO] idx={idx} skipped (generation failed/empty)")
                continue

            line = {"prompt": prompt, "result": result}
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")
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
