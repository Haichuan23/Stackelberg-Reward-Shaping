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
        # Make sure pad token exists for batch eval with classifiers (not strictly required but safer)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading RM...")
        self.RM = AutoModelForSequenceClassification.from_pretrained(
            rm_path, num_labels=1, torch_dtype=torch_dtype
        ).to(self.rm_dev)
        self.RM.eval()

        # Precompute a length cap
        # Prefer tokenizer.model_max_length; fall back to 2048 if it's extremely large (e.g., int(1e30) sentinel)
        tmax = getattr(self.tokenizer, "model_max_length", 2048)
        self.max_len_cap = 2048 if (isinstance(tmax, int) and tmax > 10_000_000) else int(tmax or 2048)

    # --------------------- helpers ---------------------

    def get_input_ids(self, prompt: str) -> torch.Tensor:
        # No truncation here; we’ll do our own length check/early return
        toks = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.llm_dev)
        return toks

    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    # ----------------- one-step (large) ----------------

    def generate_greedy_step_large(
        self,
        mout,  # LM output with logits, last token
        input_ids: torch.Tensor,
        pre_screen_beam_width: int = 40,
        weight: float = 0.0,
        rm_cached=None,  # kept in signature for drop-in compatibility (ignored)
        chunk_size: int = 10,
        debug: bool = False,
        _use_cache: bool = True,  # ignored for RM
        noise: bool = False,       # [ADDED] whether to apply Gaussian noise to rewards
        noise_var: float = 1.0,    # [ADDED] variance of the Gaussian noise
        random: bool = False,
        malicious: bool = False,
    ):
        """
        Memory-friendly version: evaluate RM on pre-screened candidates in chunks.
        No RM KV-caching; no LM cache reordering (bug source removed).
        Returns:
            best_seq_on_llm_device, None
        """
        del rm_cached  # not used (explicit to avoid confusion)

        # 1) LM top-k prescreen
        out_logits = mout.logits[:, -1]  # [B, V]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)  # [B, K], [B, K]

        # 2) Build candidate sequences [input_ids; candidate_token] for each candidate
        # expanded_tis: [B, K, T]
        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)

        # to_rm_eval: [B, K, T+1]
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        B, K, TLp1 = to_rm_eval.shape

        # Flatten to [B*K, T+1]
        flat_trme = to_rm_eval.view(B * K, TLp1)

        # Flatten prescreen logits to align with flat_trme
        flat_lm_logits = prescreen_logits.flatten()  # [B*K]

        if debug:
            print(f"[generate_greedy_step_large] flat_trme: {flat_trme.shape}, flat_lm_logits: {flat_lm_logits.shape}")

        # random/malicious require all rewards before fusion; collect upfront
        if random or malicious:
            if random:
                all_rewards_ns = torch.randn(flat_trme.shape[0], device=self.llm_dev, dtype=flat_lm_logits.dtype) * 5.0
            else:
                collected = []
                for chunk_cpu in iter_chunks(flat_trme, chunk_size=max(1, int(chunk_size))):
                    chunk = chunk_cpu.to(self.rm_dev)
                    attn = create_attention_mask(seq_len=chunk.shape[1], bsz=chunk.shape[0], device=self.rm_dev)
                    with torch.no_grad():
                        rm_out = self.RM(input_ids=chunk, attention_mask=attn)
                    collected.append(rm_out.logits.flatten().to(self.llm_dev))
                all_rewards_ns = torch.cat(collected, dim=0)
            if noise:
                all_rewards_ns = all_rewards_ns + torch.randn_like(all_rewards_ns) * (noise_var ** 0.5)
            if malicious:
                sorted_vals, sort_idx = torch.sort(all_rewards_ns)
                rev = torch.zeros_like(all_rewards_ns)
                rev[sort_idx] = sorted_vals.flip(0)
                all_rewards_ns = rev
            fused_all = flat_lm_logits.to(self.llm_dev) + weight * all_rewards_ns
            best_idx = torch.argmax(fused_all).item()
            best_seq = flat_trme[best_idx].unsqueeze(0).to(self.llm_dev)
            return best_seq, None

        # 3) Evaluate RM in chunks on rm_dev
        current_best_score: Optional[float] = None
        current_best_tokens: Optional[torch.Tensor] = None

        # Iterate chunks
        offset = 0
        for chunk_cpu in iter_chunks(flat_trme, chunk_size=max(1, int(chunk_size))):
            # Align corresponding logits slice
            chunk_len = chunk_cpu.shape[0]
            chunk_logits_lm = flat_lm_logits[offset: offset + chunk_len]  # [chunk_len]
            offset += chunk_len

            # Move to RM device
            chunk = chunk_cpu.to(self.rm_dev)
            attn = create_attention_mask(seq_len=chunk.shape[1], bsz=chunk.shape[0], device=self.rm_dev)

            with torch.no_grad():
                rm_out = self.RM(input_ids=chunk, attention_mask=attn)  # logits: [chunk_len, 1]

            rewards = rm_out.logits.flatten().to(self.llm_dev)  # back to LM device for fusion
            if noise:  # [ADDED] apply Gaussian noise to rewards before fusion
                rewards = rewards + torch.randn_like(rewards) * (noise_var ** 0.5)
            fused = chunk_logits_lm.to(self.llm_dev) + weight * rewards  # [chunk_len]

            # Greedy pick best within this chunk
            local_best_val, local_best_idx = torch.max(fused, dim=0)  # scalar, idx in [0, chunk_len)
            local_best_seq = chunk_cpu[local_best_idx.item()].to(self.llm_dev)  # full sequence with appended token

            if (current_best_score is None) or (local_best_val.item() > current_best_score):
                current_best_score = local_best_val.item()
                current_best_tokens = local_best_seq

        # Shape to [B=1, T+1] since we always keep a single path
        best_seq = current_best_tokens.unsqueeze(0)  # [1, T+1]
        return best_seq, None  # rm_cached is unused

    # ----------------- one-step (standard) ----------------

    def generate_step(
        self,
        mout,  # LM output
        input_ids: torch.Tensor,
        pre_screen_beam_width: int = 40,
        weight: float = 0.0,
        method: str = "greedy",
        temperature: float = 0.7,
        rm_cached=None,   # kept for API compat; ignored
        debug: bool = False,
        noise: bool = False,       # [ADDED] whether to apply Gaussian noise to rewards
        noise_var: float = 1.0,    # [ADDED] variance of the Gaussian noise
        random: bool = False,
        malicious: bool = False,
    ):
        """
        Non-chunked version: evaluate all top-k candidates in one RM batch.
        Supports 'greedy' or 'topk' (sampling within the K candidates).
        Returns:
            next_tokens_sequence, None
        """
        del rm_cached  # not used

        out_logits = mout.logits[:, -1]  # [B, V]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)  # [B, K]

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))  # [B, K, T+1]
        B, K, TLp1 = to_rm_eval.shape

        flat_trme = to_rm_eval.view(B * K, TLp1)  # [B*K, T+1]

        if random:
            rewards = torch.randn(B * K, device=self.llm_dev, dtype=prescreen_logits.dtype) * 5.0
        else:
            attn = create_attention_mask(seq_len=flat_trme.shape[1], bsz=flat_trme.shape[0], device=self.rm_dev)
            with torch.no_grad():
                rm_out = self.RM(input_ids=flat_trme.to(self.rm_dev), attention_mask=attn)
            rewards = rm_out.logits.flatten().to(self.llm_dev)  # [B*K]

        if noise:  # [ADDED] apply Gaussian noise to rewards before fusion
            rewards = rewards + torch.randn_like(rewards) * (noise_var ** 0.5)
        if malicious:
            sorted_vals, sort_idx = torch.sort(rewards)
            rev = torch.zeros_like(rewards)
            rev[sort_idx] = sorted_vals.flip(0)
            rewards = rev
        fused = prescreen_logits.flatten().to(self.llm_dev) + weight * rewards  # [B*K]

        if method == "greedy":
            top_idx = torch.argmax(fused)  # scalar
        elif method == "topk":
            assert input_ids.shape[0] == 1, "Sampling method assumes batch size 1."
            scores = F.softmax(fused / max(1e-8, float(temperature)), dim=-1)
            top_idx = torch.multinomial(scores, num_samples=1).squeeze(0)
        else:
            raise ValueError(f"Invalid method '{method}'")

        next_seq = flat_trme[top_idx].unsqueeze(0).to(self.llm_dev)  # [1, T+1]
        return next_seq, None

    # ----------------- main decode loop -----------------

    def generate(
        self,
        prompt: str,
        weight: float = 0.0,
        topk: int = 1,
        max_new_token: int = 128,
        method: str = "greedy",        # "greedy", "topk", or "greedy_large"
        temperature: float = 0.7,
        chunk_size: int = 5,           # used only by greedy_large
        debug: bool = False,
        noise: bool = False,           # [ADDED] whether to apply Gaussian noise to rewards
        noise_var: float = 1.0,        # [ADDED] variance of the Gaussian noise
        random: bool = False,
        malicious: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            tokens: torch.LongTensor shape [1, L_final] on llm_dev.
        """
        tokens = self.get_input_ids(prompt)  # [1, T0]
        initial_len = tokens.shape[-1]

        # Optional auto chunk size like original
        if chunk_size == "auto":
            chunk_size = auto_size(initial_len + max_new_token, topk)
            if debug:
                print(f"[generate] auto chunk_size={chunk_size}, topk={topk}, initial_len={initial_len}")

        # Length guard (use tokenizer cap)
        if tokens.shape[-1] >= self.max_len_cap:
            print(f"[ARGS] Prompt too long for model_max_length={self.max_len_cap}. Returning None.")
            return None

        # LM KV cache
        cached = None

        # Decode loop
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
                        attention_mask=attn,          # keep full mask; it's fine with KV cache
                        past_key_values=cached,       # DynamicCache or tuple from previous step
                        use_cache=True,
                    )
                cached = mout.past_key_values  # keep LM cache only

                # Choose step function
                if method == "greedy_large":
                    tokens, _ = self.generate_greedy_step_large(
                        mout, tokens, pre_screen_beam_width=topk, weight=weight,
                        rm_cached=None, chunk_size=chunk_size, debug=debug,
                        noise=noise, noise_var=noise_var,  # [ADDED]
                        random=random, malicious=malicious,
                    )
                else:
                    tokens, _ = self.generate_step(
                        mout, tokens, pre_screen_beam_width=topk, weight=weight,
                        method=method, temperature=temperature, rm_cached=None, debug=debug,
                        noise=noise, noise_var=noise_var,  # [ADDED]
                        random=random, malicious=malicious,
                    )

                # Early stop if we hit max length cap (avoid overflow next step)
                if tokens.shape[-1] >= self.max_len_cap:
                    if debug:
                        print(f"[generate] reached model_max_length={self.max_len_cap}, stopping.")
                    break

        return tokens