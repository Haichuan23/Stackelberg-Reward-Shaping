import os, json, random, argparse, re, textwrap
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    LlamaConfig,
    LlamaModel,
    StoppingCriteria,
)
from datasets import load_dataset, load_from_disk

DECODE_KW = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
SENTINEL_RE = re.compile(r'(?m)(?:\r?\n){1,3}\s*(?:Human|User)\s*:', flags=re.IGNORECASE)


# ======================================================================
# UltraRM custom architecture (only used when --rm_type ultrarm)
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
        return torch.gather(tok_scores, 1, last_idx)


# ======================================================================
# Generation helpers
# ======================================================================
class StopOnSentinel(StoppingCriteria):
    def __init__(self, tokenizer, sentinels):
        self.pats = [tokenizer.encode(s, add_special_tokens=False) for s in sentinels if s]
        self.maxlen = max((len(p) for p in self.pats), default=0)

    def __call__(self, input_ids, scores, **kwargs):
        if self.maxlen == 0:
            return False
        tail = input_ids[0, -self.maxlen:].tolist()
        return any(len(p) > 0 and len(tail) >= len(p) and tail[-len(p):] == p for p in self.pats)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


PROMPT_DIR_RE = re.compile(r"^prompt_(\d{7})$")


# ======================================================================
# Dataset / filesystem helpers
# ======================================================================
def _prompt_ids_in(dirpath: Path):
    if not dirpath.exists():
        return []
    ids = [int(m.group(1)) for name in os.listdir(dirpath) if (m := PROMPT_DIR_RE.match(name))]
    return sorted(ids)


def _count_samples_for_prompt(shards_dir: Path, pid: int):
    d = shards_dir / f"prompt_{pid:07d}"
    if not d.exists():
        return 0
    return sum(1 for name in os.listdir(d) if name.startswith("sample_") and (d / name).is_dir())


def _contiguous_full_count(shards_dir: Path, M: int):
    i = 0
    while _count_samples_for_prompt(shards_dir, i) == M:
        i += 1
    return i


def _norm_name(name: str) -> str:
    n = name.strip().lower()
    if n in {"hh", "hh-rlhf", "hh_rlhf", "dahoas/full-hh-rlhf"}:
        return "hh-rlhf"
    if n in {"shp", "stanfordnlp/shp"}:
        return "shp"
    if n in {"harmfulqa", "harmful_qa", "harmful-qa", "declare-lab/harmfulqa"}:
        return "harmfulqa"
    if n in {"ultrafeedback", "ultra_feedback", "ultra-feedback", "openbmb/ultrafeedback"}:
        return "ultrafeedback"
    return name


def _hub_id_for(name_norm: str) -> str:
    if name_norm == "hh-rlhf":
        return "Dahoas/full-hh-rlhf"
    if name_norm == "shp":
        return "stanfordnlp/SHP"
    if name_norm == "harmfulqa":
        return "declare-lab/HarmfulQA"
    if name_norm == "ultrafeedback":
        return "openbmb/UltraFeedback"
    return name_norm


def _maybe_local_path(dataset_root: str, name_norm: str, split: str) -> Optional[Path]:
    alias = {
        "hh-rlhf": ["HH-RLHF", "hh-rlhf"],
        "shp": ["shp", "SHP"],
        "harmfulqa": ["harmfulqa", "HarmfulQA", "harmful_qa"],
        "ultrafeedback": ["ultrafeedback", "UltraFeedback", "ultra_feedback"],
    }.get(name_norm, [name_norm])
    for a in alias:
        p = Path(dataset_root) / a / split
        if p.exists():
            return p
    return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def model_tag(model_id: str, dataset_kind: str) -> str:
    base = "_".join(model_id.rstrip("/").split("/")[-2:] if "/" in model_id else [model_id])
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    suffix = {"hh-rlhf": "hh", "shp": "shp", "harmfulqa": "harmfulqa", "ultrafeedback": "ultrafeedback"}.get(dataset_kind, dataset_kind.replace("/", "_"))
    return f"{base}-{suffix}"


def extract_prompt(ex, dataset_kind: str):
    if dataset_kind == "hh-rlhf":
        v = ex.get("prompt")
        if isinstance(v, str) and v.strip():
            return v
        raise KeyError(f"HH-RLHF example missing 'prompt'. Keys: {list(ex.keys())}")
    if dataset_kind == "shp":
        v = ex.get("history") or ex.get("post")
        if isinstance(v, str) and v.strip():
            return v
        raise KeyError(f"SHP example missing 'history'/'post'. Keys: {list(ex.keys())}")
    if dataset_kind == "harmfulqa":
        v = ex.get("question")
        if isinstance(v, str) and v.strip():
            return v
        raise KeyError(f"HarmfulQA example missing 'question'. Keys: {list(ex.keys())}")
    if dataset_kind == "ultrafeedback":
        v = ex.get("instruction")
        if isinstance(v, str) and v.strip():
            return v
        raise KeyError(f"UltraFeedback example missing 'instruction'. Keys: {list(ex.keys())}")
    for k in ("prompt", "history", "context", "post", "question", "instruction"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v
    raise KeyError(f"Could not find a usable prompt field. Keys: {list(ex.keys())}")


def render_chat_prompt(tokenizer, prompt_text: str, for_generation: bool):
    if isinstance(prompt_text, str) and ("Human:" in prompt_text or "Assistant:" in prompt_text):
        return prompt_text, False
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": prompt_text}]
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=for_generation)
        return rendered, True
    return prompt_text, False


# ======================================================================
# LM forward helpers
# ======================================================================
@torch.no_grad()
def get_prompt_last_hidden(model, tokenizer, prompt_text):
    rendered, chat_used = render_chat_prompt(tokenizer, prompt_text, for_generation=True)
    enc = tokenizer(rendered, return_tensors="pt", padding=False, truncation=False)
    lm_device = next(model.parameters()).device
    enc = {k: v.to(lm_device) for k, v in enc.items()}
    attn = torch.ones_like(enc["input_ids"], device=lm_device)
    outs = model(
        input_ids=enc["input_ids"],
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    last_hidden = outs.hidden_states[-1]
    hidden_prompt_last = last_hidden[:, -1, :].contiguous()[0].detach().cpu()
    return enc["input_ids"][0].detach().cpu().numpy(), int(enc["input_ids"].size(1)), hidden_prompt_last, rendered, chat_used


@torch.no_grad()
def generate_many(model, tokenizer, prompt_text, M, max_new_tokens, temperature, top_p, base_seed,
                  stop_on_human=False, stop_sentinels=None):
    model.eval()
    out = []
    rendered, chat_used = render_chat_prompt(tokenizer, prompt_text, for_generation=True)
    sent_list = stop_sentinels or []

    def _norm_nl(s): return s.replace('\r\n', '\n').replace('\r', '\n')

    def _earliest_stop_char(text):
        m = SENTINEL_RE.search(_norm_nl(text))
        return None if m is None else m.start()

    def _char_to_tok(gen_ids, char_idx):
        if char_idx <= 0:
            return 0
        for k in range(1, len(gen_ids) + 1):
            if len(tokenizer.decode(gen_ids[:k], **DECODE_KW)) >= char_idx:
                return k
        return len(gen_ids)

    for m in range(M):
        set_seed(base_seed + m)
        enc = tokenizer(rendered, return_tensors="pt", padding=False, truncation=False)
        lm_device = next(model.parameters()).device
        enc = {k: v.to(lm_device) for k, v in enc.items()}

        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        seq = gen.sequences
        attn_mask = torch.ones_like(seq, device=seq.device)
        outs = model(
            input_ids=seq, attention_mask=attn_mask,
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        last_hidden = outs.hidden_states[-1]
        T_total = int(last_hidden.size(1))
        prompt_len = int(enc["input_ids"].size(1))

        full_ids = seq[0].detach().cpu().numpy()
        gen_ids = full_ids[prompt_len:]
        response_text = tokenizer.decode(full_ids[prompt_len:], **DECODE_KW)

        cut_token_len = len(gen_ids)
        if stop_on_human and sent_list:
            stop_char = _earliest_stop_char(response_text)
            if stop_char is not None:
                cut_token_len = max(0, min(_char_to_tok(gen_ids, stop_char), len(gen_ids)))
                response_text = tokenizer.decode(full_ids[prompt_len:prompt_len + cut_token_len], **DECODE_KW)
                m2 = SENTINEL_RE.search(_norm_nl(response_text))
                if m2:
                    response_text = response_text[:m2.start()].rstrip()
                    tgt_len = len(response_text)
                    k = 0
                    while k < len(gen_ids):
                        k += 1
                        if len(tokenizer.decode(gen_ids[:k], **DECODE_KW)) >= tgt_len:
                            break
                    cut_token_len = k

        gen_len = int(cut_token_len)
        hidden_gen = last_hidden[:, prompt_len:prompt_len + gen_len, :].contiguous()

        out.append({
            "prompt_text": prompt_text,
            "rendered_prompt": rendered,
            "chat_formatted": bool(chat_used),
            "response_text": response_text,
            "input_ids_full": full_ids,
            "attention_mask_full": attn_mask[0].detach().cpu().numpy(),
            "hidden_last_gen": hidden_gen[0].detach().cpu(),
            "prompt_len": int(prompt_len),
            "gen_len": int(gen_len),
            "T_total": int(T_total),
            "reply_end_gen_tokens": int(gen_len),
        })
    return out


# ======================================================================
# Reward model loading (dispatches on rm_type)
# ======================================================================
def load_rm(args):
    rm_device = f"cuda:{args.rm_gpu_id}" if torch.cuda.is_available() else "cpu"

    rm_tok = AutoTokenizer.from_pretrained(
        args.reward_model_name_or_path, use_fast=True, trust_remote_code=True,
    )
    if rm_tok.pad_token_id is None and rm_tok.eos_token_id is not None:
        rm_tok.pad_token = rm_tok.eos_token
    rm_tok.padding_side = "right"

    if args.rm_type == "ultrarm":
        rm_dtype = torch.float16 if "cuda" in rm_device else torch.float32
        rm_model = LlamaRewardModel.from_pretrained(
            args.reward_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=rm_dtype,
        ).to(rm_device)
    else:  # standard: AutoModelForSequenceClassification + chat template
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": args.rm_gpu_id},
            num_labels=1,
            trust_remote_code=True,
        )

    rm_model.eval()
    rm_model.requires_grad_(False)
    return rm_tok, rm_model


# ======================================================================
# Reward scoring (dispatches on rm_type)
# ======================================================================
ULTRARM_TEMPLATE = "Human: {instruction}\n\nAssistant: {completion}"


@torch.no_grad()
def score_with_rm(rm_tok, rm_model, item: dict, rm_type: str, max_length: int = 4096) -> Optional[float]:
    if rm_tok is None or rm_model is None:
        return None

    prompt_raw = item.get("rendered_prompt") or item.get("prompt_text") or ""
    resp_raw = item.get("response_text") or ""

    if not prompt_raw.strip() or not resp_raw.strip():
        return None

    if rm_type == "ultrarm":
        # Extract the last Human turn from the HH-style rendered prompt
        m = re.search(r'Human:\s*(.*?)\s*\n\s*Assistant:\s*$', prompt_raw, flags=re.DOTALL)
        if m:
            instr = m.group(1).strip()
        else:
            last_h = prompt_raw.rfind("Human:")
            last_a = prompt_raw.rfind("Assistant:")
            if last_h != -1 and last_a != -1 and last_h < last_a:
                instr = prompt_raw[last_h + len("Human:"):last_a].strip()
            else:
                instr = prompt_raw.strip()

        comp = re.split(r'\n\s*Human:\b', resp_raw.lstrip(), flags=re.IGNORECASE)[0].rstrip()
        text = ULTRARM_TEMPLATE.format(instruction=instr, completion=comp)

        enc = rm_tok(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        rm_device = next(rm_model.parameters()).device
        enc = {k: v.to(rm_device) for k, v in enc.items()}
        out = rm_model(**enc)

        if isinstance(out, torch.Tensor):
            return float(out.view(-1)[0].item())
        if hasattr(out, "logits") and out.logits is not None:
            return float(out.logits.view(-1)[0].item())
        return None

    else:  # standard: chat-template scoring
        prompt_text = item.get("prompt_text") or prompt_raw
        response_text = resp_raw
        conv = [{"role": "user", "content": prompt_text}, {"role": "assistant", "content": response_text}]
        text = rm_tok.apply_chat_template(conv, tokenize=False)

        enc = rm_tok(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        device = next(rm_model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        out = rm_model(**enc)
        return float(out.logits[0][0].item())


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()

    # --- Language model ---
    ap.add_argument("--model_name_or_path", type=str, required=True,
                    help="CausalLM checkpoint (local path or HF id)")
    ap.add_argument("--lm_dtype", type=str, default=None, choices=["bfloat16", "float16"],
                    help="Dtype for loading the language model. Defaults to float16, "
                         "or bfloat16 automatically when model path contains 'gemma' and '27b'.")
    ap.add_argument("--lm_gpu_id", type=int, default=0)

    # --- Reward model ---
    ap.add_argument("--reward_model_name_or_path", type=str, default=None,
                    help="RM checkpoint (local path or HF id). Omit to skip scoring.")
    ap.add_argument("--rm_type", type=str, default="ultrarm", choices=["ultrarm", "standard"],
                    help="'ultrarm': LlamaRewardModel + Human/Assistant format. "
                         "'standard': AutoModelForSequenceClassification + chat-template format.")
    ap.add_argument("--rm_max_length", type=int, default=4096)
    ap.add_argument("--rm_gpu_id", type=int, default=1)

    # --- Dataset ---
    ap.add_argument("--dataset_name", type=str, default="HH-RLHF",
                    help="One of {HH-RLHF, SHP} or a HF dataset id.")
    ap.add_argument("--dataset_root", type=str, default="datasets",
                    help="Local root; if {root}/{name}/{split} exists, load from disk.")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--num_problems", type=int, default=100)

    # --- Generation ---
    ap.add_argument("--samples_per_prompt", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stop_on_human", action="store_true",
                    help="Hard-trim generations when a new Human/User turn appears.")
    ap.add_argument("--stop_sentinels", nargs="*",
                    default=["\n\nHuman:", "\nHuman:", "\n\nUser:", "\nUser:"])

    # --- Prompt range (for sharded/array-job runs) ---
    ap.add_argument("--start_prompt", type=int, required=True,
                    help="First prompt index to process (inclusive).")
    ap.add_argument("--end_prompt", type=int, required=True,
                    help="Last prompt index to process (exclusive).")

    # --- Output ---
    ap.add_argument("--output_dir", type=str, default="datasets")
    ap.add_argument("--fp16_hidden", action="store_true",
                    help="Store hidden states as float16 on disk (else bfloat16).")
    ap.add_argument("--write_jsonl", action="store_true",
                    help="Also write a compact samples.jsonl per dataset folder.")
    ap.add_argument("--device", type=str, default="cuda:0")

    args = ap.parse_args()
    assert args.samples_per_prompt > 0

    # Auto-detect Gemma-3-27B and force bfloat16 (too large for float16)
    if args.lm_dtype is None:
        model_lower = args.model_name_or_path.lower()
        if "gemma" in model_lower and "27b" in model_lower:
            args.lm_dtype = "bfloat16"
            print("[info] Detected Gemma-3-27B: using bfloat16 for LM loading.")
        else:
            args.lm_dtype = "float16"

    set_seed(args.seed)

    # ---- Dataset ----
    ds_kind = _norm_name(args.dataset_name)
    hub_id = _hub_id_for(ds_kind)
    maybe_local = _maybe_local_path(args.dataset_root, ds_kind, args.split)
    if maybe_local is not None:
        print(f"[info] Loading dataset from disk: {maybe_local}")
        ds = load_from_disk(str(maybe_local))
    else:
        print(f"[info] Loading dataset from hub: {hub_id} [{args.split}]")
        ds = load_dataset(hub_id, split=args.split)

    # ---- Language model ----
    lm_dtype = torch.bfloat16 if args.lm_dtype == "bfloat16" else torch.float16
    print(f"[info] Loading LM {args.model_name_or_path} (dtype={args.lm_dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=lm_dtype,
        device_map={"": args.lm_gpu_id},
        trust_remote_code=True,
    )
    model.eval()
    model.requires_grad_(False)

    # ---- Reward model ----
    rm_tok = rm_model = None
    if args.reward_model_name_or_path:
        print(f"[info] Loading RM {args.reward_model_name_or_path} (rm_type={args.rm_type}) ...")
        rm_tok, rm_model = load_rm(args)

    # ---- Output dirs ----
    tag = model_tag(args.model_name_or_path, ds_kind)
    root = Path(args.output_dir) / tag
    shards_dir = root / "shards"
    prompts_dir = root / "prompts"
    ensure_dir(shards_dir)
    ensure_dir(prompts_dir)
    index_f = open(root / "index.jsonl", "a", buffering=1)
    samples_jsonl_f = open(root / "samples.jsonl", "a", buffering=1) if args.write_jsonl else None

    hidden_dtype = torch.float16 if args.fp16_hidden else torch.bfloat16
    dtype_str = "float16" if args.fp16_hidden else "bfloat16"

    # ---- Resume logic ----
    M = args.samples_per_prompt
    existing_ids = _prompt_ids_in(shards_dir)
    if not existing_ids:
        start_i = 0
        print("[resume] No prior shards found. Starting from 0.")
    else:
        k = existing_ids[-1]
        answered = _contiguous_full_count(shards_dir, M)
        print(f"[resume] Fully-answered prompts from start: {answered}")
        print(f"[resume] Highest prompt dir: {k:07d}")
        k_full = _count_samples_for_prompt(shards_dir, k) == M
        km1_full = (_count_samples_for_prompt(shards_dir, k - 1) == M) if k > 0 else False
        if k_full:
            start_i = k + 1
            print(f"[resume] prompt_{k:07d} complete → resume at {start_i}.")
        elif km1_full:
            start_i = k
            print(f"[resume] prompt_{k:07d} incomplete → resume at {start_i}.")
        else:
            raise RuntimeError(
                f"[resume] Unexpected shard state around tail: "
                f"prompt_{k-1:07d} full={km1_full}, prompt_{k:07d} full={k_full}. "
                f"Fix incomplete shards before resuming."
            )

    N = min(args.num_problems, len(ds))
    # start_prompt = max(start_i, args.start_prompt)
    start_prompt = args.start_prompt
    end_prompt = min(N, args.end_prompt)
    if start_prompt >= end_prompt:
        print(f"[info] Nothing to process: start_prompt={start_prompt}, end_prompt={end_prompt}")
        index_f.close()
        if samples_jsonl_f:
            samples_jsonl_f.close()
        return

    # ---- Main loop ----
    for i in range(start_prompt, end_prompt):
        ex = ds[i]
        prompt = extract_prompt(ex, ds_kind)

        input_ids_prompt, prompt_len_true, hidden_prompt_last, rendered_prompt, chat_used = \
            get_prompt_last_hidden(model, tokenizer, prompt)
        hidden_prompt_last = hidden_prompt_last.to(hidden_dtype)

        pd = prompts_dir / f"prompt_{i:07d}"
        ensure_dir(pd)
        np.save(pd / "input_ids_prompt.npy", input_ids_prompt.astype(np.int64))
        (pd / "prompt.txt").write_text(prompt)
        (pd / "rendered_prompt.txt").write_text(rendered_prompt)
        torch.save(hidden_prompt_last, pd / "hidden_prompt_last.pt")
        with open(pd / "meta.json", "w") as f:
            json.dump({
                "prompt_id": i,
                "prompt_len": int(prompt_len_true),
                "chat_formatted": bool(chat_used),
                "arrays": {
                    "input_ids_prompt": str(pd / "input_ids_prompt.npy"),
                    "hidden_prompt_last": str(pd / "hidden_prompt_last.pt"),
                },
                "dtypes": {"hidden_prompt_last": dtype_str},
                "model_id": args.model_name_or_path,
                "dataset": hub_id,
                "dataset_kind": ds_kind,
                "split": args.split,
            }, f, ensure_ascii=False)

        samples = generate_many(
            model, tokenizer, prompt,
            M=M,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            base_seed=args.seed + i * 100000,
            stop_on_human=args.stop_on_human,
            stop_sentinels=args.stop_sentinels,
        )

        samples_root = shards_dir / f"prompt_{i:07d}"
        ensure_dir(samples_root)

        for m, s in enumerate(samples):
            reward_val = score_with_rm(
                rm_tok, rm_model, s,
                rm_type=args.rm_type,
                max_length=args.rm_max_length,
            ) if rm_model is not None else None

            d = samples_root / f"sample_{m:03d}"
            ensure_dir(d)
            np.save(d / "input_ids.npy", s["input_ids_full"].astype(np.int64))
            np.save(d / "attention_mask.npy", s["attention_mask_full"].astype(np.uint8))
            torch.save(s["hidden_last_gen"].to(hidden_dtype), d / "hidden_last.pt")

            meta = {
                "prompt_id": i,
                "sample_id": m,
                "prompt_ref": str(pd),
                "prompt_text": s["prompt_text"],
                "rendered_prompt": s.get("rendered_prompt", rendered_prompt),
                "chat_formatted": s.get("chat_formatted", chat_used),
                "response_text": s["response_text"],
                "reward": reward_val,
                "reward_model_id": args.reward_model_name_or_path,
                "rm_type": args.rm_type,
                "prompt_len": s["prompt_len"],
                "gen_len": s["gen_len"],
                "T_total": s["T_total"],
                "model_id": args.model_name_or_path,
                "tokenizer_id": args.model_name_or_path,
                "gen_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                },
                "arrays": {
                    "input_ids": str(d / "input_ids.npy"),
                    "attention_mask": str(d / "attention_mask.npy"),
                    "hidden_last": str(d / "hidden_last.pt"),
                    "input_ids_prompt_ref": str(pd / "input_ids_prompt.npy"),
                    "hidden_prompt_last_ref": str(pd / "hidden_prompt_last.pt"),
                },
                "dtypes": {"hidden_last": dtype_str, "hidden_prompt_last_ref": dtype_str},
                "dataset": hub_id,
                "dataset_kind": ds_kind,
                "split": args.split,
                "indices": {"dataset_idx": i, "sample_idx_for_prompt": m},
                "reply_end_gen_tokens": s.get("reply_end_gen_tokens", s["gen_len"]),
            }
            with open(d / "meta.json", "w") as f:
                json.dump(meta, f, ensure_ascii=False)

            if samples_jsonl_f is not None:
                samples_jsonl_f.write(json.dumps({
                    "id": f"{i}:{m}",
                    "prompt": s["prompt_text"],
                    "response": s["response_text"],
                    "reward": reward_val,
                    "lm": args.model_name_or_path,
                    "rm": args.reward_model_name_or_path,
                    "rm_type": args.rm_type,
                    "dataset": hub_id,
                    "dataset_kind": ds_kind,
                    "split": args.split,
                }, ensure_ascii=False) + "\n")

            index_f.write(json.dumps({
                "id": f"{i}:{m}",
                "path": str(d),
                "prompt_id": i,
                "prompt_ref": str(pd),
                "prompt_len": s["prompt_len"],
                "gen_len": s["gen_len"],
                "model_id": args.model_name_or_path,
                "dataset": hub_id,
                "dataset_kind": ds_kind,
                "split": args.split,
            }) + "\n")

        if (i + 1) % 10 == 0:
            print(f"[info] Processed {i + 1}/{end_prompt} prompts ...")

    index_f.close()
    if samples_jsonl_f:
        samples_jsonl_f.close()
    print(f"[done] Wrote samples to {root} (dataset={hub_id}, split={args.split})")


if __name__ == "__main__":
    main()
