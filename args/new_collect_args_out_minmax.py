from datasets import load_dataset, load_from_disk
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from new_argsearch_minmax import ARGS
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
parser.add_argument("--out_file", type=str, required=True, help="Output prefix (we write <prefix>.jsonl)")
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--setting", type=str, default="test", choices=["validation", "test"])
parser.add_argument("--K", type=int, default=0)
parser.add_argument("--dataset_local_dir", type=str, default=None)

parser.add_argument("--B", type=float, required=True, help="Min-max shaping budget B")
parser.add_argument(
    "--stretch",
    type=float,
    default=1.01,
    help="Stretch factor for data-dependent min-max budget: B_eff = min(B, stretch * (r_max - r_min))",
)
parser.add_argument("--noise", action="store_true", default=False,  
                    help="If set, apply Gaussian noise to rewards before shaping.")
parser.add_argument("--noise_var", type=float, default=1.0,         
                    help="Variance of the Gaussian noise added to rewards (default: 1.0).")

args = parser.parse_args()
print(f"{args=}")

# -------------------- Seeding --------------------
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
if args.stretch <= 0:
    print("ERROR: --stretch must be > 0", file=sys.stderr)
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
        except Exception:
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
    ds_list = raw_ds["prompt"]

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

print(f"[INFO] Using {len(selected_ds)} prompts")
truncated_ds = selected_ds

# -------------------- Models --------------------
print(f"[INFO] Loading models (llm={args.llm}, rm={args.rm})")
search = ARGS(
    llm_path=args.llm,
    rm_path=args.rm,
    llm_dev=args.llm_gpu,
    rm_dev=args.rm_gpu,
)
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
        return tok.numel() == 0
    except Exception:
        return False

def runprompt(
    prompt: str,
    rm_weight=0.0,              # lambda
    topk=5,
    new_token=24,
    mode="greedy",
    sample_temp=None,
    noise: bool = False, noise_var: float = 1.0,  
):
    temp = sample_temp if sample_temp is not None else 0.7

    out = search.generate(
        prompt=prompt,
        B=float(args.B),
        lambda_coef=float(rm_weight),
        stretch=float(args.stretch),
        topk=topk,
        max_new_token=new_token,
        method=mode,
        temperature=temp,
        noise=noise, noise_var=noise_var,  
    )

    tokens = _extract_tokens(out)
    if _empty_tokens(tokens):
        return None, False

    text_out = search.tokens_to_text(tokens)
    text = text_out[0] if isinstance(text_out, list) else text_out

    pure = text[len(prompt):] if text.startswith(prompt) else text
    return pure, True

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# -------------------- Main loop --------------------
out_path = Path(f"{args.out_file}.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

start_idx = count_lines(out_path) if args.recover else 0

with out_path.open("a", encoding="utf-8") as fout:
    for idx in tqdm(range(start_idx, len(truncated_ds))):
        prompt = truncated_ds[idx]
        rc = run_configs[0]

        result, ok = runprompt(
            prompt,
            float(rc["rm_weight"]),   # lambda
            rc["topk"],
            args.max_new_token,
            rc["mode"],
            rc["sample_temp"],
            noise=args.noise,         
            noise_var=args.noise_var, 
        )

        if not ok or not result:
            continue

        fout.write(json.dumps({"prompt": prompt, "result": result}, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())

print(f"[INFO] Done: {out_path}")
