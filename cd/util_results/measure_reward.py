from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import PreTrainedModel, LlamaConfig, LlamaModel
import argparse
import torch
import torch.nn as nn
import json
import re
import numpy as np
from tqdm import tqdm
from typing import Optional, List

parser = argparse.ArgumentParser()
parser.add_argument("--out_file", type=str, required=True)
parser.add_argument("--rm", type=str, required=True)
parser.add_argument("--rm_gpu", type=str, default="cuda:0")
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--npout", type=str, default="")
parser.add_argument("--experiment", type=str, default="hhrlhf")
parser.add_argument(
    "--MAX_PROMPT",
    type=int,
    default=None,
    help="If set, only score the first min(MAX_PROMPT, len(data)) entries",
)
parser.add_argument(
    "--rm_type",
    type=str,
    default="standard",
    choices=["standard", "ultrarm", "oasst", "steamshp", "qwen3_32b"],
    help="Reward model architecture / input format",
)
args = parser.parse_args()


# ======================================================================
# UltraRM custom architecture (only used when rm_type=ultrarm)
# ======================================================================
class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden = outs[0]
        tok_scores = self.regression_head(hidden).squeeze(-1)
        assert attention_mask is not None, "attention_mask is required"
        last_idx = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        rewards = torch.gather(tok_scores, 1, last_idx)
        return rewards


# ======================================================================
# Load tokenizer and reward model
# ======================================================================
if args.rm_type == "steamshp":
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rm_model = T5ForConditionalGeneration.from_pretrained(
        args.rm, local_files_only=True
    ).to(args.rm_gpu)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    if args.rm_type == "ultrarm":
        rm_model = LlamaRewardModel.from_pretrained(
            args.rm, torch_dtype=torch.float16, local_files_only=True
        ).to(args.rm_gpu)
    elif args.rm_type == "oasst":
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            args.rm, local_files_only=True
        ).to(args.rm_gpu)
    elif args.rm_type == "qwen3_32b":
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            args.rm, torch_dtype=torch.float16, device_map="auto", local_files_only=True
        )
    else:  # standard
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            args.rm, num_labels=1, torch_dtype=torch.float16,
            local_files_only=True, trust_remote_code=False,
        ).to(args.rm_gpu)

rm_model.eval()

# Max sequence length
if args.rm_type in ("oasst", "steamshp"):
    rm_max_len = 512
else:
    rm_max_len = getattr(rm_model.config, "max_position_embeddings", None)
    if rm_max_len is None or rm_max_len <= 0:
        rm_max_len = getattr(tokenizer, "model_max_length", 2048)
    if rm_max_len is None or rm_max_len > 8192:
        rm_max_len = 2048

TOKEN_A = 71  # SteamSHP: logit index for token 'A'

# ======================================================================
# Load JSON / JSONL
# ======================================================================
with open(args.out_file, "r") as f:
    try:
        data = json.load(f)
        if isinstance(data, dict):
            lines = [data]
        elif isinstance(data, list):
            lines = data
        else:
            raise ValueError("Unsupported JSON structure in --out_file")
        print("✅ Loaded as standard JSON")
    except json.JSONDecodeError:
        print("⚠️ JSON decode failed — trying JSONL format")
        f.seek(0)
        lines = [json.loads(line) for line in f if line.strip()]
        print("✅ Loaded as JSONL (line-by-line)")

print(f"📦 Total entries: {len(lines)}")

if args.MAX_PROMPT is not None:
    orig_len = len(lines)
    lines = lines[: min(args.MAX_PROMPT, orig_len)]
    print(f"✂️ Truncated entries: using {len(lines)} / {orig_len}")


# ======================================================================
# Text extraction
# ======================================================================
def _parse_question(prompt_raw):
    """Extract the human instruction from a HH-style prompt."""
    m = re.search(r'Human:\s*(.*?)\n\s*Assistant:\s*$', prompt_raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    p = prompt_raw
    if "Human:" in p:
        p = p.split("Human:", 1)[1]
    if "Assistant:" in p:
        p = p.split("Assistant:", 1)[0]
    return p.strip()


def _parse_answer(item, prompt_raw):
    """Extract and clean the model completion."""
    resp = item.get("response") or item.get("output") or item.get("result")
    if resp is None:
        return None
    answer = resp.removeprefix(prompt_raw).lstrip()
    if answer.startswith(": "):
        answer = answer[2:]
    return re.split(r'\n\s*Human:\b', answer, flags=re.IGNORECASE)[0].rstrip()


def extract_item(item):
    """
    Returns (text_or_question, answer_or_None).
    standard/ultrarm: answer_or_None is None; the full scored string is in text_or_question.
    oasst/steamshp/qwen3_32b: (question, answer) passed separately to get_rm.
    """
    prompt_raw = item.get("prompt", "") or ""

    if args.rm_type == "standard":
        resp = item.get("response") or item.get("output") or item.get("result")
        if resp is None:
            return None, None
        if args.experiment.lower() == "hhrlhf":
            output_np = resp.removeprefix(prompt_raw)
            if output_np.startswith(": "):
                output_np = output_np[2:]
            output_np = re.split(r"human:", output_np, flags=re.IGNORECASE)[0]
            return prompt_raw + output_np, None
        else:  # shp or default: score raw response
            return resp, None

    elif args.rm_type == "ultrarm":
        resp = item.get("response") or item.get("output") or item.get("result")
        if resp is None:
            return None, None
        question = _parse_question(prompt_raw)
        comp = resp.lstrip()
        comp = re.split(r'\n\s*Human:\b', comp, flags=re.IGNORECASE)[0].rstrip()
        return f"Human: {question}\nAssistant: {comp}", None

    else:  # oasst, steamshp, qwen3_32b
        question = _parse_question(prompt_raw)
        answer = _parse_answer(item, prompt_raw)
        if answer is None:
            return None, None
        if args.rm_type == "steamshp":
            question = question.replace("\n", " ").strip()
            answer = answer.replace("\n", " ").strip()
        return question, answer


# ======================================================================
# Reward scoring
# ======================================================================
def get_rm(text_or_question, answer=None):
    if args.rm_type in ("standard", "ultrarm"):
        enc = tokenizer(
            text_or_question, return_tensors="pt",
            truncation=True, max_length=rm_max_len,
        )
        print(f"len(tokens)={enc.input_ids.shape[1]}")
        if args.rm_type == "ultrarm":
            enc = {k: v.to(args.rm_gpu) for k, v in enc.items()}
            with torch.no_grad():
                return rm_model(**enc).flatten().item()
        else:
            with torch.no_grad():
                return rm_model(enc.input_ids.to(args.rm_gpu)).logits.flatten().item()

    elif args.rm_type == "oasst":
        inputs = tokenizer(
            text_or_question, answer,
            return_tensors="pt", truncation=True, max_length=rm_max_len,
        )
        print(f"len(tokens)={inputs['input_ids'].shape[1]}")
        inputs = {k: v.to(args.rm_gpu) for k, v in inputs.items()}
        with torch.no_grad():
            return rm_model(**inputs).logits[0].item()

    elif args.rm_type == "steamshp":
        input_text = (
            f"POST: {text_or_question}\n\n"
            f"RESPONSE A: {answer}\n\n"
            f"RESPONSE B: .\n\n"
            f"Which response is better? RESPONSE"
        )
        x = tokenizer(
            [input_text], return_tensors="pt",
            truncation=True, max_length=rm_max_len,
        ).input_ids.to(args.rm_gpu)
        print(f"len(tokens)={x.shape[1]}")
        with torch.no_grad():
            outputs = rm_model.generate(
                x, return_dict_in_generate=True,
                output_scores=True, max_new_tokens=1,
            )
        return outputs.scores[0][0, TOKEN_A].item()

    else:  # qwen3_32b
        messages = [
            {"role": "user",      "content": text_or_question + " /nothink"},
            {"role": "assistant", "content": "<think>\n\n</think>\n\n" + answer},
        ]
        tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_tensors="pt", return_dict=True,
        )
        input_ids = tokenized["input_ids"].to(rm_model.device)
        attention_mask = tokenized["attention_mask"].to(rm_model.device)
        print(f"len(tokens)={input_ids.shape[1]}")
        with torch.no_grad():
            return rm_model(input_ids, attention_mask=attention_mask).logits[0][0].item()


# ======================================================================
# Scoring loop
# ======================================================================
rm_scores = []
num_skip = 0
for i, item in enumerate(tqdm(lines)):
    q, a = extract_item(item)
    if q is None:
        rm_scores.append(np.nan)
        continue
    try:
        score = get_rm(q, a)
        rm_scores.append(score)
    except Exception as e:
        print(f"[WARN] Skipping idx={i} due to RM error: {e}")
        rm_scores.append(np.nan)
        num_skip += 1

rm_scores = np.array(rm_scores, dtype=np.float32)

if args.npout:
    np.save(args.npout, rm_scores)
    print(f"💾 Saved per-sample rewards to: {args.npout}")

valid = np.isfinite(rm_scores)
num_valid = int(valid.sum())
if num_valid == 0:
    print("❗ No valid scores computed (all NaN). Check token lengths and RM settings.")
else:
    mean_reward = np.nanmean(rm_scores)
    print(f"✅ Mean reward over {num_valid} valid samples = {mean_reward:.6f}")

print(f"Skipped/NaN samples = {int((~valid).sum())}")