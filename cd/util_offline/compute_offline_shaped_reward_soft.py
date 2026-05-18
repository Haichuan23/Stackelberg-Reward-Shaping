import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import bisect

# ================== shaping utilities ==================

def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def soft_shape_rewards(
    raw_rewards,
    B=1.0,
    beta=1.0,
    cap=2.0,
    alpha=1.0,
):
    r = np.asarray(raw_rewards, dtype=float)
    stretch = r.max() - r.min()
    B = min(B, 2.0 * stretch)

    gamma = np.exp(min(B / beta, cap))
    def F(m):
        return np.sum((r - m) * np.where(r > m, gamma, 1.0))
    
    lo, hi = r.min() - 10.0, r.max() + 10.0
    try:
        m_star = bisect(F, lo, hi)
        used_default = False
    except ValueError:
        m_star = r.max()
        used_default = True

    shaped =(float(B) * sigmoid(float(alpha) * (r - m_star))).tolist()
    return shaped, m_star, used_default


# ================== main ==================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True,
                   help="Dataset root dir (e.g., datasets/models_llama3-8b-sft-hh or absolute path).")
    p.add_argument("--reward_model", type=str, required=True,
                   help="Subdir name under each prompt (e.g., skywork).")
    p.add_argument("--max_prompts", type=int, required=True,
                   help="Process prompt_0000000 ... prompt_{max_prompts-1}.")
    p.add_argument("--num_responses", type=int, required=True,
                   help="NUM_RESPONSES used in raw_reward filename.")
    p.add_argument("--B", type=float, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--cap", type=float, default=2.0,
                   help="Caps B/beta inside exp in helper_function.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Sigmoid sharpness for soft shaping: B*sigmoid(alpha*(r-m)).")
    return p.parse_args()

def main():
    args = parse_args()

    ROOT_DIR = Path(args.root_dir)
    REWARD_MODEL = args.reward_model
    MAX_PROMPT = int(args.max_prompts)
    NUM_RESPONSES = int(args.num_responses)

    B = float(args.B)
    BETA = float(args.beta)
    CAP = float(args.cap)
    ALPHA = float(args.alpha)

    # preserve exact input formatting (for filenames)
    B_STR = str(args.B)
    BETA_STR = str(args.beta)
    CAP_STR = str(args.cap)
    ALPHA_STR = str(args.alpha)

    shard_dir = ROOT_DIR / "shards"
    print(f"shard_dir = {shard_dir}", flush=True)

    n_done = 0
    n_skipped = 0

    for i in range(MAX_PROMPT):
        pretty = f"[{i+1:06d}/{MAX_PROMPT:06d}]"

        prompt_dir = shard_dir / f"prompt_{i:07d}"
        if not prompt_dir.exists():
            n_skipped += 1
            print(f"{pretty} MISSING {prompt_dir.name}", flush=True)
            continue

        rm_dir = prompt_dir / REWARD_MODEL
        if not rm_dir.exists():
            n_skipped += 1
            print(f"{pretty} SKIP (no {REWARD_MODEL}/) {prompt_dir.name}", flush=True)
            continue

        last5 = i % 100000
        raw_name = f"raw_reward_prompt{last5:05d}_response{NUM_RESPONSES}.json"
        raw_path = rm_dir / raw_name
        if not raw_path.exists():
            n_skipped += 1
            print(f"{pretty} SKIP (no raw json) {raw_path.name}", flush=True)
            continue

        out_name = (
            f"soft_shaped_reward_prompt{last5:05d}_response{NUM_RESPONSES}"
            f"_B_{B_STR}_beta_{BETA_STR}_alpha_{ALPHA_STR}_cap_{CAP_STR}_stretch2.0.json"
        )
        out_path = rm_dir / out_name

        try:
            with open(raw_path, "r") as f:
                data = json.load(f)

            raw_rewards = data.get("collected_rewards", [])
            if not raw_rewards:
                n_skipped += 1
                print(f"{pretty} SKIP (empty rewards) {raw_path.name}", flush=True)
                continue

            shaped, m_star, used_default = soft_shape_rewards(
                raw_rewards, B=B, beta=BETA, cap=CAP, alpha=ALPHA
            )

            with open(out_path, "w") as wf:
                json.dump({"collected_rewards": shaped}, wf, indent=2)

            n_done += 1
            print(
                f"{pretty} OK {prompt_dir.name} -> {out_path.name} "
                f"(m*={m_star:.4f}, default={used_default}, alpha={ALPHA_STR})",
                flush=True,
            )

        except Exception as e:
            print(f"{pretty} WARN Failed on {raw_path}: {e}", flush=True)
            n_skipped += 1

    print(f"Processed prompts (written): {n_done}", flush=True)
    print(f"Skipped prompts:             {n_skipped}", flush=True)

if __name__ == "__main__":
    main()
