from datasets import load_dataset, load_from_disk
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from new_argsearch_raw import ARGS
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
parser.add_argument("--noise", action="store_true", default=False,  
                    help="If set, apply Gaussian noise to rewards before fusion.")
parser.add_argument("--noise_var", type=float, default=1.0,         
                    help="Variance of the Gaussian noise added to rewards (default: 1.0).")
parser.add_argument("--random", action="store_true", default=False,
                    help="If set, replace RM with N(0,5) random rewards.")
parser.add_argument("--malicious", action="store_true", default=False,
                    help="If set, reverse the reward ranking (best becomes worst).")
parser.add_argument(
    "--setting",
    type=str,
    default="test",
    choices=["validation", "test", "speed"],
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
                # SHP is commonly saved as DatasetDict via save_to_disk -> load_from_disk
                print(f"[INFO] Loading SHP from disk at {local_path}")
                ds_dict = load_from_disk(str(local_path))  # DatasetDict
                raw_ds = ds_dict[ds_split]
            else:
                # For HH-RLHF and many others, local_path can be a dataset directory usable by load_dataset
                # (e.g., a local HF dataset script/path or a local clone).
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

if args.setting == "speed":
    # speed mode: fixed 20 random prompts
    rng = np.random.RandomState(args.seed)
    n_speed = min(20, len(ds_list))
    idxs = rng.permutation(len(ds_list))[:n_speed]
    selected_ds = [ds_list[i] for i in idxs]

elif K > 0:
    selected_ds = ds_list[-K:] if args.setting == "validation" else ds_list[:K]

else:
    if args.setting == "validation":
        selected_ds = ds_list[-300:]
    else:
        n = 1000
        rng = np.random.RandomState(args.seed)
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
    """Normalize possible outputs from search.generate into token IDs."""
    if out is None:
        return None
    # common patterns: tokens, (tokens, cache/meta/extra), [tokens], etc.
    if isinstance(out, tuple):
        return out[0]
    return out

def _empty_tokens(tok):
    """Heuristics to detect empty tokens."""
    if tok is None:
        return True
    try:
        # torch tensor?
        if hasattr(tok, "numel"):
            return tok.numel() == 0
        # list/tuple?
        if isinstance(tok, (list, tuple)):
            return len(tok) == 0
    except Exception:
        pass
    return False

def runprompt(prompt: str, rm_weight=0.0, topk=5, new_token=24, mode="p_sigmoid_mixing",
              sample_temp=None, llm_dev: str = "cuda:0",
              noise: bool = False, noise_var: float = 1.0,
              random: bool = False, malicious: bool = False):
    """
    Returns (pure_text, ok_flag). We DO NOT return/store tokens.
    """
    temp = sample_temp if sample_temp is not None else 0.7
    torch.cuda.synchronize(torch.device(args.llm_gpu))
    torch.cuda.synchronize(torch.device(args.rm_gpu))
    start_time = time.time()
    out = search.generate(prompt, method=mode, topk=topk, max_new_token=new_token,
            temperature=temp, weight=rm_weight, debug=False,
            noise=noise, noise_var=noise_var,
            random=random, malicious=malicious)
    torch.cuda.synchronize(torch.device(args.llm_gpu))
    torch.cuda.synchronize(torch.device(args.rm_gpu))
    decode_time = time.time() - start_time
    tokens = _extract_tokens(out)
    if _empty_tokens(tokens):
        return None, False, decode_time

    # Convert to text (expect list[str] or (str, ...) from tokens_to_text)
    try:
        text_out = search.tokens_to_text(tokens)
        if isinstance(text_out, (list, tuple)):
            text = text_out[0]
        else:
            text = str(text_out)
    except Exception as e:
        print(f"[WARN] tokens_to_text failed: {e}", file=sys.stderr)
        return None, False, decode_time

    # Drop the prompt prefix if present
    try:
        pure = text.removeprefix(prompt)
    except AttributeError:
        pure = text[len(prompt):] if text.startswith(prompt) else text

    return pure, True, decode_time

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

    # Ensure directory exists
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
    total_decode_time = 0.0
    num_ok = 0

    # Append mode so we never clobber existing lines
    with out_path.open("a", encoding="utf-8") as fout:
        for idx in tqdm(range(start_idx, len(truncated_ds))):
            prompt = truncated_ds[idx]
            result, ok, decode_time = runprompt(
                prompt,
                float(rc["rm_weight"]),
                rc["topk"],
                args.max_new_token,
                rc["mode"],
                rc["sample_temp"],
                llm_dev=args.llm_gpu,
                noise=args.noise,         
                noise_var=args.noise_var, 
                random=args.random,
                malicious=args.malicious,
            )
            if not ok or result is None or result == "":
                skipped += 1
                # keep a lightweight log for debugging
                if skipped <= 10 or skipped % 100 == 0:
                    print(f"[INFO] idx={idx} skipped (generation failed/empty)")
                continue

            total_decode_time += decode_time
            num_ok += 1

            # --- minimal JSONL: ONLY the fields you asked for ---
            line = {"prompt": prompt, "result": result}
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1

            # Periodic flush & fsync to avoid empty files on crash/NFS buffering
            if (idx + 1) % 10 == 0 or (time.time() - last_flush) > 5.0:
                try:
                    fout.flush()
                    os.fsync(fout.fileno())
                except Exception:
                    pass
                last_flush = time.time()

        # final flush
        try:
            fout.flush()
            os.fsync(fout.fileno())
        except Exception:
            pass

    if num_ok > 0:
        avg_time = total_decode_time / num_ok
        print(f"[TIMING] Average decoding time per prompt: {avg_time:.3f} seconds")
    else:
        print("[TIMING] No successful generations to report timing.")
    print(f"[INFO] Done: {out_path} (wrote {written} lines, skipped {skipped})")

