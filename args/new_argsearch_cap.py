from typing import List, Optional, Tuple, Iterable
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# ===================== Utilities =====================

def create_attention_mask(seq_len: int, bsz: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    HF expects attention_mask as int64 (or bool). We use int64.
    """
    return torch.ones((bsz, seq_len), dtype=torch.long, device=device)

def factors(x: int):
    return [i for i in range(1, x + 1) if x % i == 0]

def auto_size(seq_len: int, topk: int) -> int:
    """
    Heuristic from original code (kept for compatibility).
    """
    estimated = (28672 / (seq_len * 1.5)) - 11.52605
    possible = factors(topk)[::-1]
    if np.all(~(np.array(possible) < estimated)):
        return 1
    return possible[np.argmax(np.array(possible) < estimated)]

def iter_chunks(tensor: torch.Tensor, chunk_size: int) -> Iterable[torch.Tensor]:
    """
    Yield slices of size <= chunk_size without requiring divisibility.
    Works on first dimension of `tensor`.
    """
    n = tensor.shape[0]
    for start in range(0, n, chunk_size):
        yield tensor[start:min(start + chunk_size, n)]

def cap_shift_rewards(rewards: torch.Tensor, cap: Optional[float]) -> torch.Tensor:
    """
    Apply: r <- r - min(r), then clip to [0, cap] if cap is not None.
    rewards: 1D tensor on some device
    """
    r = rewards - rewards.min()
    if cap is not None:
        r = torch.clamp(r, min=0.0, max=float(cap))
    else:
        r = torch.clamp(r, min=0.0)
    return r

# ===================== ARGS Search =====================

class ARGS:
    """
    Reward-guided search:
      - At each step, pre-screen top-k tokens from LM logits.
      - Score [prompt + candidate] with reward model (scalar).
      - Fuse: fused = logit + weight * reward.
      - Pick best (greedy) or sample (topk-sampling).
    """

    def __init__(
        self,
        llm_path: str,
        rm_path: str,
        llm_dev: str = "cuda:0",
        rm_dev: str = "cuda:1",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.llm_dev = torch.device(llm_dev)
        self.rm_dev = torch.device(rm_dev)

        print("Loading LLM...")
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype).to(self.llm_dev)
        self.LLM.eval()

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading RM...")
        self.RM = AutoModelForSequenceClassification.from_pretrained(
            rm_path, num_labels=1, torch_dtype=torch_dtype
        ).to(self.rm_dev)
        self.RM.eval()

        tmax = getattr(self.tokenizer, "model_max_length", 2048)
        self.max_len_cap = 2048 if (isinstance(tmax, int) and tmax > 10_000_000) else int(tmax or 2048)

    # --------------------- helpers ---------------------

    def get_input_ids(self, prompt: str) -> torch.Tensor:
        toks = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.llm_dev)
        return toks

    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    # ----------------- one-step (large) ----------------

    def generate_greedy_step_large(
        self,
        mout,
        input_ids: torch.Tensor,
        pre_screen_beam_width: int = 40,
        weight: float = 0.0,
        rm_cached=None,
        chunk_size: int = 10,
        debug: bool = False,
        _use_cache: bool = True,
        cap: Optional[float] = None,  
    ):
        """
        Memory-friendly version: evaluate RM on pre-screened candidates in chunks.
        Returns:
            best_seq_on_llm_device, None
        """
        del rm_cached

        out_logits = mout.logits[:, -1]  # [B, V]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        Bsz, K, TLp1 = to_rm_eval.shape
        flat_trme = to_rm_eval.view(Bsz * K, TLp1)

        current_best_score: Optional[float] = None
        current_best_tokens: Optional[torch.Tensor] = None

        flat_lm_logits = prescreen_logits.flatten()  # [B*K]

        global_min_reward = None
        offset = 0
        for chunk_cpu in iter_chunks(flat_trme, chunk_size=max(1, int(chunk_size))):
            chunk = chunk_cpu.to(self.rm_dev)
            attn = create_attention_mask(seq_len=chunk.shape[1], bsz=chunk.shape[0], device=self.rm_dev)
            with torch.no_grad():
                rm_out = self.RM(input_ids=chunk, attention_mask=attn)
            chunk_rewards = rm_out.logits.flatten()
            m = chunk_rewards.min().item()
            global_min_reward = m if (global_min_reward is None) else min(global_min_reward, m)

        assert global_min_reward is not None

        offset = 0
        for chunk_cpu in iter_chunks(flat_trme, chunk_size=max(1, int(chunk_size))):
            chunk_len = chunk_cpu.shape[0]
            chunk_logits_lm = flat_lm_logits[offset: offset + chunk_len]
            offset += chunk_len

            chunk = chunk_cpu.to(self.rm_dev)
            attn = create_attention_mask(seq_len=chunk.shape[1], bsz=chunk.shape[0], device=self.rm_dev)

            with torch.no_grad():
                rm_out = self.RM(input_ids=chunk, attention_mask=attn)

            rewards = rm_out.logits.flatten().to(self.llm_dev)  # [chunk_len] on llm_dev

            rewards = rewards - float(global_min_reward)
            if cap is not None:
                rewards = torch.clamp(rewards, min=0.0, max=float(cap))
            else:
                rewards = torch.clamp(rewards, min=0.0)

            fused = chunk_logits_lm.to(self.llm_dev) + weight * rewards  # [chunk_len]

            local_best_val, local_best_idx = torch.max(fused, dim=0)
            local_best_seq = chunk_cpu[local_best_idx.item()].to(self.llm_dev)

            if (current_best_score is None) or (local_best_val.item() > current_best_score):
                current_best_score = local_best_val.item()
                current_best_tokens = local_best_seq

        best_seq = current_best_tokens.unsqueeze(0)
        return best_seq, None

    # ----------------- one-step (standard) ----------------

    def generate_step(
        self,
        mout,
        input_ids: torch.Tensor,
        pre_screen_beam_width: int = 40,
        weight: float = 0.0,
        method: str = "greedy",
        temperature: float = 0.7,
        rm_cached=None,
        debug: bool = False,
        cap: Optional[float] = None,  # ** NEW **
    ):
        """
        Non-chunked version: evaluate all top-k candidates in one RM batch.
        Supports 'greedy' or 'topk'.
        Returns:
            next_tokens_sequence, None
        """
        del rm_cached

        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        Bsz, K, TLp1 = to_rm_eval.shape

        flat_trme = to_rm_eval.view(Bsz * K, TLp1)
        attn = create_attention_mask(seq_len=flat_trme.shape[1], bsz=flat_trme.shape[0], device=self.rm_dev)

        with torch.no_grad():
            rm_out = self.RM(input_ids=flat_trme.to(self.rm_dev), attention_mask=attn)

        rewards = rm_out.logits.flatten().to(self.llm_dev)  # [B*K]

        # shift by min reward in this step, then cap 
        rewards = cap_shift_rewards(rewards, cap)  

        fused = prescreen_logits.flatten().to(self.llm_dev) + weight * rewards

        if method == "greedy":
            top_idx = torch.argmax(fused)
        elif method == "topk":
            assert input_ids.shape[0] == 1, "Sampling method assumes batch size 1."
            scores = F.softmax(fused / max(1e-8, float(temperature)), dim=-1)
            top_idx = torch.multinomial(scores, num_samples=1).squeeze(0)
        else:
            raise ValueError(f"Invalid method '{method}'")

        next_seq = flat_trme[top_idx].unsqueeze(0).to(self.llm_dev)
        return next_seq, None

    # ----------------- main decode loop -----------------

    def generate(
        self,
        prompt: str,
        weight: float = 0.0,
        topk: int = 1,
        max_new_token: int = 128,
        method: str = "greedy",
        temperature: float = 0.7,
        chunk_size: int = 5,
        debug: bool = False,
        cap: Optional[float] = None,  
    ) -> torch.Tensor:
        """
        Returns:
            tokens: torch.LongTensor shape [1, L_final] on llm_dev.
        """
        tokens = self.get_input_ids(prompt)
        initial_len = tokens.shape[-1]

        if chunk_size == "auto":
            chunk_size = auto_size(initial_len + max_new_token, topk)
            if debug:
                print(f"[generate] auto chunk_size={chunk_size}, topk={topk}, initial_len={initial_len}")

        if tokens.shape[-1] >= self.max_len_cap:
            print(f"[ARGS] Prompt too long for model_max_length={self.max_len_cap}. Returning None.")
            return None

        cached = None

        for _ in (range(max_new_token)):
            with torch.no_grad():
                attn = create_attention_mask(seq_len=tokens.shape[1], bsz=tokens.shape[0], device=self.llm_dev)
                if cached is None:
                    mout = self.LLM(
                        **self.LLM.prepare_inputs_for_generation(
                            input_ids=tokens, attention_mask=attn, use_cache=True
                        )
                    )
                else:
                    next_token = tokens[:, -1:].contiguous()
                    mout = self.LLM(
                        input_ids=next_token,
                        attention_mask=attn,
                        past_key_values=cached,
                        use_cache=True,
                    )
                cached = mout.past_key_values

                if method == "greedy_large":
                    tokens, _ = self.generate_greedy_step_large(
                        mout, tokens, pre_screen_beam_width=topk, weight=weight,
                        rm_cached=None, chunk_size=chunk_size, debug=debug,
                        cap=cap, 
                    )
                else:
                    tokens, _ = self.generate_step(
                        mout, tokens, pre_screen_beam_width=topk, weight=weight,
                        method=method, temperature=temperature, rm_cached=None, debug=debug,
                        cap=cap,  
                    )

                if tokens.shape[-1] >= self.max_len_cap:
                    if debug:
                        print(f"[generate] reached model_max_length={self.max_len_cap}, stopping.")
                    break

        return tokens
