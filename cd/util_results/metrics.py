from tqdm import tqdm
import json
import argparse
from nltk import word_tokenize
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
from simcse import SimCSE
import numpy as np
import re


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--response_num", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_prompt", type=int, required=True)
    parser.add_argument("--lambda_value", type=float, required=True)

    parser.add_argument("--mode", type=str, default="greedy")
    parser.add_argument("--base_dir", type=str, default="outputs/llama3-8b-sft/skywork")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--reward_model_name", type=str, required=True)
    parser.add_argument("--evaluation", type=str, required=True)

    args = parser.parse_args()
    return args


def compute_rep_n(text, n):
    tokens = word_tokenize(text, preserve_line=True)
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 100 * (1.0 - len(set(ngrams)) / (len(ngrams) + 1))


def compute_diversity(text):
    d = 1.0
    for n in range(2, 5):
        d *= 1.0 - compute_rep_n(text, n) / 100
    return d


def clean(text, sep="###"):
    return text.split(sep)[0]


def average(xs):
    return sum(xs) / len(xs)


def compute_coherence(prompts, responses):
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    sims = np.array(model.similarity(prompts, responses))
    return sims.trace() / len(sims)


if __name__ == "__main__":
    args = get_args()

    response_num = args.response_num
    beta = args.beta
    seed = args.seed
    max_prompt = args.max_prompt
    lambda_value = args.lambda_value
    mode = args.mode
    base_dir = args.base_dir

    lambda_dir = os.path.join(base_dir, f"LAMBDA_{lambda_value}")
    if not os.path.isdir(lambda_dir):
        raise FileNotFoundError(f"❌ Lambda directory not found: {lambda_dir}")

    target_file = f"ctrl_collect_{mode}_lambda_{lambda_value}.jsonl"


    def load_generations(path):
        with open(path, "r") as f:
            text = f.read().strip()

        try:
            data = json.loads(text)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            pass

        lines = [l for l in text.splitlines() if l.strip()]
        try:
            return [json.loads(l) for l in lines]
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        idx, objs = 0, []
        while idx < len(text):
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            obj, idx = decoder.raw_decode(text, idx)
            objs.append(obj)
        return objs


    def evaluate_and_save(input_path):
        gens = load_generations(input_path)

        entries = []
        for g in tqdm(gens, desc=f"Evaluating {os.path.basename(input_path)}"):
            prompt = g["prompt"]
            raw = g.get("response", g.get("result"))

            answer = raw[len(prompt):] if raw.startswith(prompt) else raw
            response = clean(clean(answer, "###Human:"), "\n\nHuman:") or " "

            entries.append({
                "prompt": prompt,
                "response": response,
                "original_response": answer,
                "rep_2": compute_rep_n(response, 2),
                "rep_3": compute_rep_n(response, 3),
                "rep_4": compute_rep_n(response, 4),
                "diversity": compute_diversity(response),
                "response_length": len(response),
                "elapsed": g.get("elapsed", 0.0),
            })

        evaluations = {
            "rep_2": average([e["rep_2"] for e in entries]),
            "rep_3": average([e["rep_3"] for e in entries]),
            "rep_4": average([e["rep_4"] for e in entries]),
            "diversity": average([e["diversity"] for e in entries]),
            "coherence": compute_coherence(
                [e["prompt"] for e in entries],
                [e["response"] for e in entries],
            ),
            "response_length": average([e["response_length"] for e in entries]),
            "elapsed": average([e["elapsed"] for e in entries]),
            "entries": entries,
        }

        rel = os.path.relpath(input_path, start=base_dir)
        project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
        eval_root = os.path.join(
            project_root,
            "evaluation",
            f"{args.model_name}-{args.evaluation}",
            args.reward_model_name,
        )
        eval_path = os.path.join(eval_root, rel)
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)

        with open(eval_path, "w") as f:
            json.dump(evaluations, f, indent=2)

        print(f"✅ Saved evaluation to: {eval_path}")


    # ================= PATTERNS =================

    soft_pattern = re.compile(
        rf"^soft_resp{response_num}_B[0-9.]+_beta{beta}_alpha[0-9.]+_seed{seed}_prompt_{max_prompt}$"
    )
    hard_pattern = re.compile(
        rf"^hard_resp{response_num}_B[0-9.]+_beta{beta}_seed{seed}_prompt_{max_prompt}$"
    )
    raw_pattern = re.compile(
        rf"^raw_resp{response_num}_seed{seed}_prompt_{max_prompt}$"
    )

    # **NEW: nested mean-std support**
    meanstd_pattern = re.compile(
        rf"^mean_std_resp{response_num}_seed{seed}_prompt_{max_prompt}$"
    )

    # **NEW: nested minmax support**
    minmax_pattern = re.compile(
        rf"^minmax_resp{response_num}_B[0-9.]+_stretch[0-9.]+_seed{seed}_prompt_{max_prompt}$"
    )

    cap_pattern = re.compile(
        rf"^cap_resp{response_num}_B[0-9.]+_beta{beta}_seed{seed}_prompt_{max_prompt}$"
    )

    mean_soft_pattern = re.compile(
        rf"^mean_soft_resp{response_num}_B[0-9.]+_beta{beta}_alpha[0-9.]+_stretch1.5_seed{seed}_prompt_{max_prompt}$"
    )

    matched = 0

    for name in os.listdir(lambda_dir):
        path = os.path.join(lambda_dir, name)

        if not os.path.isdir(path):
            continue

        # **CHANGED: include meanstd + minmax**
        if (
            soft_pattern.match(name)
            or raw_pattern.match(name)
            or meanstd_pattern.match(name)     # **
            or minmax_pattern.match(name)      # **
            or cap_pattern.match(name)
            or mean_soft_pattern.match(name)
            or hard_pattern.match(name)  
        ):
            candidate = os.path.join(path, target_file)
            if os.path.isfile(candidate):
                print(f"✅ Found target file: {candidate}")
                evaluate_and_save(candidate)
                matched += 1
            else:
                print(f"⚠️ Missing target file: {candidate}")

    if matched == 0:
        raise FileNotFoundError(
            f"❌ No matching JSONL found under {lambda_dir}"
        )