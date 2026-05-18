import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== Cache helpers (HF compat) =====================
try:
    from transformers.cache_utils import DynamicCache  # HF >= 4.41
except Exception:
    DynamicCache = None

def pkv_to_legacy_tuple(pkv):
    if pkv is None:
        return None
    if isinstance(pkv, tuple):
        return pkv
    if DynamicCache is not None and hasattr(pkv, "to_legacy_cache"):
        return pkv.to_legacy_cache()
    return pkv

def legacy_tuple_to_cache(pkv_legacy):
    if pkv_legacy is None:
        return None
    if DynamicCache is None:
        return pkv_legacy
    return DynamicCache.from_legacy_cache(pkv_legacy)

def tile_past(past_key_values, k: int):
    if past_key_values is None:
        return None
    pkv_legacy = pkv_to_legacy_tuple(past_key_values)
    new_past = []
    for layer in pkv_legacy:
        new_past.append(tuple(t.repeat_interleave(k, dim=0) for t in layer))
    return tuple(new_past)

# ===================== Value function (unchanged API) =====================
def layer_init(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer

class Agent(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 8192)),
            nn.Tanh(),
            layer_init(nn.Linear(8192, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x.float()).squeeze(-1)

def load_value_agent(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden_size = int(ckpt["args"]["hidden_size"])
    agent = Agent(hidden_size=hidden_size).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent, hidden_size, ckpt.get("args", {})

# ===================== Small utils =====================
def safe_log_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return F.log_softmax(logits / max(temperature, 1e-6), dim=-1)

def pick_topk(logits: torch.Tensor, k: int, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # logits: [1, V]
    logp = safe_log_softmax(logits, temperature=temperature)  # [1, V]
    topk_logp, topk_ids = torch.topk(logp, k=k, dim=-1)      # [1, k]
    return topk_logp.squeeze(0), topk_ids.squeeze(0)          # [k], [k]

# ===================== Controlled Decoder =====================
class ControlledDecoder:
    def __init__(
        self,
        value_ckpt_path: str,
        model_id: str = "meta-llama/Llama-3.1-8B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        lambda_coef: float = 1.0,
        top_k: int = 20,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        debug: bool = False,
        debug_max_steps: int = 10,
    ):
        use_auto_device_map = device is None

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        if use_auto_device_map:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.device = torch.device(device)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)

        self.model.eval()

        # Device for input ids: use the embedding device, which is usually the first model shard.
        self.model_input_device = self.model.get_input_embeddings().weight.device

        # Device for the value head. Put it on the same device as the hidden state before scoring.
        # Usually cuda:0 is fine; this model is small compared to Gemma-27B.
        self.agent_device = self.model_input_device
        self.agent, self.hidden_size, _ = load_value_agent(value_ckpt_path, self.agent_device)

        print("[INFO] model_input_device:", self.model_input_device)
        if hasattr(self.model, "hf_device_map"):
            print("[INFO] hf_device_map:", self.model.hf_device_map)

        self.lambda_coef = float(lambda_coef)
        self.top_k = int(top_k)
        self.temperature = float(temperature)

        # ** Support multiple end-of-turn (EOT/EOS) tokens (e.g., <|eot_id|>, <|end_of_text|>)
        # ** If an explicit eos_token_id is provided, include it; otherwise start with tokenizer.eos_token_id.
        # ** Then add known EOT variants if present in the vocab.
        # ** Keep a primary EOS id for placeholders and a set for fast membership checks.
        self.eos_ids: List[int] = self._collect_eos_ids(eos_token_id)
        self.eos_set = set(self.eos_ids)
        self.primary_eos_id: int = self.eos_ids[0]

        # self.eos_id = int(eos_token_id if eos_token_id is not None else self.tok.eos_token_id)

        self.debug = bool(debug)
        self.debug_max_steps = int(debug_max_steps)

    def _collect_eos_ids(self, explicit_eos_id: Optional[int]) -> List[int]:
        ids: List[int] = []

        # 1) If explicit EOS was passed, use that.
        if explicit_eos_id is not None:
            ids.append(int(explicit_eos_id))

        # 2) Always include tokenizer's default eos_token_id (if any)
        base_eos = getattr(self.tok, "eos_token_id", None)
        if isinstance(base_eos, int) and base_eos not in ids:
            ids.append(base_eos)

        # 3) Try common end-of-turn tokens used by chat/instruct variants.
        for t in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>"):
            try:
                if t in self.tok.get_vocab():
                    tid = self.tok.convert_tokens_to_ids(t)
                    if isinstance(tid, int) and tid not in ids:
                        ids.append(tid)
            except Exception:
                pass

        if not ids:
            raise ValueError("Could not determine any EOS/EOT token ids from tokenizer.")
        return ids

    # ---------- Candidate look-ahead (vectorized) ----------
    @torch.no_grad()
    def _score_candidates(
        self,
        past_key_values,
        topk_ids: torch.Tensor,        # [k] token ids
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        """
        For each candidate id v in topk_ids, do a single-step forward with the same past.
        Returns:
            values: [k] from Agent over last hidden states
            logits: [k, V] (next-token distribution *after* emitting v)
            cand_past: PKV for each candidate (batch k) - not used to advance real state
        """
        k = topk_ids.shape[0]
        pkv_legacy = tile_past(past_key_values, k)
        pkv_cache  = legacy_tuple_to_cache(pkv_legacy)


        out = self.model(
            input_ids=topk_ids.view(k, 1).to(self.model_input_device),
            use_cache=True,
            past_key_values=pkv_cache,
            output_hidden_states=True,
        )

        last_hidden = out.hidden_states[-1][:, -1, :].to(self.agent_device)
        values = self.agent.get_value(last_hidden)
        logits = out.logits[:, -1, :]                  # [k, V]
        return values, logits, out.past_key_values

    # ---------- Pretty debug ----------
    def _token_str(self, tid: int) -> str:
        try:
            return self.tok.decode([tid]).replace("\n", "\\n")
        except Exception:
            return f"<{tid}>"

    @torch.no_grad()
    def _print_step_table(self, t, ids, base_logp, vals, fused, pick_idx):
        if not self.debug or t >= self.debug_max_steps:
            return

        for i in range(min(ids.shape[0], 8)):
            tid = int(ids[i].item())
            V = float(vals[i].item())
            l = float(base_logp[i].item())
            f = float(fused[i].item())
            print(f"  {i:>1}  {tid:<9} {self._token_str(tid)[:18]:<18} {l:+8.3f}  {V:+7.3f}  {(self.lambda_coef*V):+8.3f}  {f:+8.3f}   {'⬅' if i==pick_idx else ''}")

    # ---------- Baseline-faithful greedy ----------
    @torch.no_grad()
    def decode_greedy(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        stop_on_eos: bool = True,
    ) -> str:

        # Step 0: full prompt forward (this is where attention_mask belongs)
        # enc = self.tok(prompt, return_tensors="pt").to(self.device)
        enc = self.tok(prompt, return_tensors="pt").to(self.model_input_device)
        out = self.model(**enc, use_cache=True, output_hidden_states=False)
        past = out.past_key_values

        # We keep input_ids only for final decode/return
        input_ids = enc["input_ids"]  # [1, L]

        for t in range(max_new_tokens):
            # Base next-token logits from REAL state
            base_logits = out.logits[:, -1, :]  # [1, V]
            topk_logp, topk_ids = pick_topk(base_logits, k=self.top_k, temperature=self.temperature)  # [k],[k]

            if self.lambda_coef != 0.0:
                values, _, _ = self._score_candidates(past, topk_ids)          # [k]
                fused = topk_logp + self.lambda_coef * values                   # [k]
            else:
                values = torch.zeros_like(topk_logp)
                fused  = topk_logp

            best_idx = int(torch.argmax(fused).item())
            best_id  = topk_ids[best_idx:best_idx+1].view(1,1)                  # [1,1]

            # Append for final text
            input_ids = torch.cat([input_ids, best_id.to(input_ids.device)], dim=-1)

            out = self.model(
                input_ids=best_id.to(self.model_input_device),
                use_cache=True,
                past_key_values=past,
                output_hidden_states=False,
            )
            past = out.past_key_values

            if stop_on_eos and (best_id.item() in self.eos_set):
                break

        return self.tok.decode(input_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def decode_sample(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        stop_on_eos: bool = True,
        sample_temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> str:
        k = int(top_k or self.top_k)

        enc = self.tok(prompt, return_tensors="pt").to(self.model_input_device)
        out = self.model(**enc, use_cache=True, output_hidden_states=False)
        past = out.past_key_values
        input_ids = enc["input_ids"]

        for t in range(max_new_tokens):
            base_logits = out.logits[:, -1, :]
            topk_logp, topk_ids = pick_topk(base_logits, k=k, temperature=self.temperature)

            if self.lambda_coef != 0.0:
                values, _, _ = self._score_candidates(past, topk_ids)
                fused = topk_logp + self.lambda_coef * values
            else:
                values = torch.zeros_like(topk_logp)
                fused  = topk_logp

            # Sample from fused distribution
            fused_centered = fused - fused.max()
            probs = torch.softmax(
                fused_centered / max(sample_temperature, 1e-6), dim=-1
            )  # [k]
            idx = int(torch.multinomial(probs, num_samples=1).item())
            next_id = topk_ids[idx:idx+1].view(1, 1)

            input_ids = torch.cat(
                [input_ids, next_id.to(input_ids.device)], dim=-1
            )

            out = self.model(
                input_ids=next_id.to(self.model_input_device),
                use_cache=True,
                past_key_values=past,
                output_hidden_states=False,
            )
            past = out.past_key_values

            if stop_on_eos and (next_id.item() in self.eos_set):
                break

        return self.tok.decode(input_ids[0], skip_special_tokens=True)

    # ---------- Lightweight beam with fused increments ----------
    @torch.no_grad()
    def decode_beam(
        self,
        prompt: str,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        stop_on_eos: bool = True,
    ) -> str:
        assert num_beams >= 2

        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        out0 = self.model(**enc, use_cache=True, output_hidden_states=False)
        base_past = out0.past_key_values

        # Beam state
        beam_inputs: List[torch.Tensor] = [enc["input_ids"].clone() for _ in range(num_beams)]
        beam_pasts:  List[object]       = [base_past for _ in range(num_beams)]
        beam_scores = torch.zeros(num_beams, device=self.device)
        finished    = [False] * num_beams

        for _ in range(max_new_tokens):
            all_cands = []
            for b in range(num_beams):
                if finished[b]:
                    # keep EOS beams in the pool
                    all_cands.append((beam_scores[b], self.eos_id, b, None))
                    continue

                # One real step to get base logits for this beam
                last_tok = beam_inputs[b][:, -1:]  # [1,1]
                out_b = self.model(
                    input_ids=last_tok,
                    use_cache=True,
                    past_key_values=beam_pasts[b],
                    output_hidden_states=False,
                )
                base_logits_b = out_b.logits[:, -1, :]  # [1, V]
                topk_logp_b, topk_ids_b = pick_topk(base_logits_b, k=self.top_k, temperature=self.temperature)

                if self.lambda_coef != 0.0:
                    vals_b, _, _ = self._score_candidates(out_b.past_key_values, topk_ids_b)
                    fused_b = topk_logp_b + self.lambda_coef * vals_b
                else:
                    vals_b = torch.zeros_like(topk_logp_b)
                    fused_b = topk_logp_b

                # Propose k continuations from beam b
                for j in range(self.top_k):
                    all_cands.append((
                        beam_scores[b] + fused_b[j],
                        int(topk_ids_b[j].item()),
                        b,
                        vals_b[j].item() if self.lambda_coef != 0.0 else 0.0,
                    ))

            # Select top B overall
            all_cands.sort(key=lambda x: float(x[0]), reverse=True)
            new_inputs, new_pasts, new_scores, new_finished = [], [], [], []
            picks = 0
            for score, tok_id, from_b, _ in all_cands:
                if picks == num_beams:
                    break
                if finished[from_b]:
                    # carry over finished
                    new_inputs.append(beam_inputs[from_b])
                    new_pasts.append(beam_pasts[from_b])
                    new_scores.append(beam_scores[from_b])
                    new_finished.append(True)
                else:
                    appended = torch.cat(
                        [beam_inputs[from_b], torch.tensor([[tok_id]], device=self.device)], dim=1
                    )
                    # REAL step to advance this new beam’s cache
                    out_next = self.model(
                        input_ids=torch.tensor([[tok_id]], device=self.device),
                        use_cache=True,
                        past_key_values=beam_pasts[from_b],
                        output_hidden_states=False,
                    )
                    new_inputs.append(appended)
                    new_pasts.append(out_next.past_key_values)
                    new_scores.append(score)
                    new_finished.append(tok_id == self.eos_id)

                picks += 1

            beam_inputs = new_inputs
            beam_pasts  = new_pasts
            beam_scores = torch.stack([torch.as_tensor(s, device=self.device) for s in new_scores], dim=0)
            finished    = new_finished

            if all(finished):
                break

        best = int(torch.argmax(beam_scores).item())
        return self.tok.decode(beam_inputs[best][0], skip_special_tokens=True)

# ===================== CLI for quick tests =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value_ckpt", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--lambda_coef", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--mode", type=str, default="greedy", choices=["greedy","sample","beam"])
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--sample_temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dec = ControlledDecoder(
        value_ckpt_path=args.value_ckpt,
        model_id=args.model_id,
        device=args.device,
        dtype=dtype_map[args.dtype],
        lambda_coef=args.lambda_coef,
        top_k=args.top_k,
        temperature=args.temperature,
        debug=args.debug,
        debug_max_steps=12,
    )

    if args.mode == "greedy":
        out = dec.decode_greedy(args.prompt, max_new_tokens=args.max_new_tokens, stop_on_eos=True)
    elif args.mode == "sample":
        out = dec.decode_sample(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            stop_on_eos=True,
            sample_temperature=args.sample_temperature,
        )
    else:
        out = dec.decode_beam(
            args.prompt,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            stop_on_eos=True,
        )

    print("\n=== OUTPUT ===\n")
    print(out)

if __name__ == "__main__":
    main()