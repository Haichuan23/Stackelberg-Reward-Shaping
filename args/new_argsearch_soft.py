from typing import List, Optional, Tuple, Iterable
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# ===================== Utilities =====================

def create_attention_mask(seq_len: int, bsz: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.ones((bsz, seq_len), dtype=torch.long, device=device)

def factors(x: int):
    return [i for i in range(1, x + 1) if x % i == 0]

def auto_size(seq_len: int, topk: int) -> int:
    estimated = (28672 / (seq_len * 1.5)) - 11.52605
    possible = factors(topk)[::-1]
    if np.all(~(np.array(possible) < estimated)):
        return 1
    return possible[np.argmax(np.array(possible) < estimated)]

def iter_chunks(tensor: torch.Tensor, chunk_size: int) -> Iterable[torch.Tensor]:
    n = tensor.shape[0]
    for start in range(0, n, chunk_size):
        yield tensor[start:min(start + chunk_size, n)]

# ===================== ARGS Search =====================

class ARGS:
    """
    Reward-guided search:
      - Prescreen top-k tokens by LM logits.
      - Score [prompt + candidate] with RM.
      - Fuse with LM logits.

    SOFT shaping:
      - compute threshold m via PAD equation (same as hard shaping)
      - shaped_reward_i = B * sigmoid(alpha * (r_i - m))    
      - fuse: fused = lm_logits + lambda * shaped_reward
      - lambda = 1/beta if beta provided else lambda=weight
    """

    def __init__(
        self,
        llm_path: str,
        rm_path: str,
        llm_dev: str = "cuda:0",
        rm_dev: str = "cuda:1",
        torch_dtype: torch.dtype = torch.float16,
        seed: Optional[int] = None,  # keep optional
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

        self._torch_gen = torch.Generator(device=self.llm_dev)
        if seed is not None:
            self._torch_gen.manual_seed(int(seed))

    # --------------------- helpers ---------------------

    def set_seed(self, seed: int) -> None:
        self._torch_gen.manual_seed(int(seed))

    def get_input_ids(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.llm_dev)

    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    @staticmethod
    def _resolve_lambda_beta(weight: float, beta: Optional[float]) -> Tuple[float, Optional[float]]:
        if beta is not None:
            beta_eff = float(beta)
            if beta_eff <= 0:
                raise ValueError(f"beta must be > 0, got {beta_eff}")
            lam = 1.0 / beta_eff
            return lam, beta_eff
        lam = float(weight)
        beta_eff = None if lam <= 0 else (1.0 / lam)
        return lam, beta_eff

    @staticmethod
    def _pad_threshold(rewards: torch.Tensor, p: torch.Tensor, c: float, iters: int = 30) -> float:
        lo = float(rewards.min().item())
        hi = float(rewards.max().item())
        if lo == hi:
            return lo
        c_t = torch.tensor(float(c), device=rewards.device, dtype=rewards.dtype)
        for _ in range(int(iters)):
            m = 0.5 * (lo + hi)
            m_t = torch.tensor(m, device=rewards.device, dtype=rewards.dtype)
            w = torch.where(rewards >= m_t, c_t * p, p)
            Fm = torch.sum(w * (rewards - m_t)).item()
            if Fm > 0:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)

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
        reward_shaping: bool = False,
        B: float = 1.0,
        beta: Optional[float] = None,
        alpha: float = 1.0,
        noise: bool = False,
        noise_var: float = 1.0,
        random: bool = False,
        malicious: bool = False,
    ):
        del rm_cached

        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        Bsz, K, TLp1 = to_rm_eval.shape

        flat_trme = to_rm_eval.view(Bsz * K, TLp1)
        flat_lm_logits = prescreen_logits.flatten()

        if debug:
            print(f"[generate_greedy_step_large] flat_trme: {flat_trme.shape}, flat_lm_logits: {flat_lm_logits.shape}")

        if reward_shaping:
            assert Bsz == 1, "reward_shaping assumes batch size 1."
            if alpha <= 0:  # **CHANGED**
                raise ValueError(f"alpha must be > 0, got {alpha}")  # **CHANGED**

            lam, beta_eff = self._resolve_lambda_beta(weight, beta)
            if beta_eff is None:
                beta_eff = 1.0 / max(1e-12, lam)

            if random:
                rewards = torch.randn(K, device=self.llm_dev, dtype=flat_lm_logits.dtype) * 5.0
            else:
                all_rewards = []
                for chunk_cpu in iter_chunks(flat_trme, chunk_size=max(1, int(chunk_size))):
                    chunk = chunk_cpu.to(self.rm_dev)
                    attn = create_attention_mask(seq_len=chunk.shape[1], bsz=chunk.shape[0], device=self.rm_dev)
                    with torch.no_grad():
                        rm_out = self.RM(input_ids=chunk, attention_mask=attn)
                    all_rewards.append(rm_out.logits.flatten().to(self.llm_dev))
                rewards = torch.cat(all_rewards, dim=0)              # [K] on llm_dev

            if noise:
                rewards = rewards + torch.randn_like(rewards) * (noise_var ** 0.5)
            if malicious:
                sorted_vals, sort_idx = torch.sort(rewards)
                rev = torch.zeros_like(rewards)
                rev[sort_idx] = sorted_vals.flip(0)
                rewards = rev
            r_min = rewards.min()
            rewards_shifted = rewards - r_min
            r_max = rewards_shifted.max()
            B = min(1.01 * r_max, B)
            c = min(float(torch.exp(torch.tensor(float(B) / float(beta_eff))).item()), 2)

            lm_logits = flat_lm_logits.to(self.llm_dev)              # [K] on llm_dev
            p = F.softmax(lm_logits, dim=-1)

            m = self._pad_threshold(rewards, p, c)

            # -------- SOFT shaping --------
            # shaped_i = B * sigmoid(alpha * (r_i - m))
            m_t = torch.tensor(m, device=self.llm_dev, dtype=rewards.dtype)  # 
            shaped = float(B) * torch.sigmoid(float(alpha) * (rewards - m_t))  # 
            # -----------------------------

            fused = lm_logits + lam * shaped                          # 
            top_idx = torch.argmax(fused).item()
            best_seq = flat_trme[top_idx].unsqueeze(0).to(self.llm_dev)
            return best_seq, None

        # ---- original non-shaping greedy_large behavior ----
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

        current_best_score: Optional[float] = None
        current_best_tokens: Optional[torch.Tensor] = None

        offset = 0
        for chunk_cpu in iter_chunks(flat_trme, chunk_size=max(1, int(chunk_size))):
            chunk_len = chunk_cpu.shape[0]
            chunk_logits_lm = flat_lm_logits[offset: offset + chunk_len]
            offset += chunk_len

            chunk = chunk_cpu.to(self.rm_dev)
            attn = create_attention_mask(seq_len=chunk.shape[1], bsz=chunk.shape[0], device=self.rm_dev)

            with torch.no_grad():
                rm_out = self.RM(input_ids=chunk, attention_mask=attn)

            rewards = rm_out.logits.flatten().to(self.llm_dev)
            if noise:
                rewards = rewards + torch.randn_like(rewards) * (noise_var ** 0.5)
            fused = chunk_logits_lm.to(self.llm_dev) + weight * rewards

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
        reward_shaping: bool = False,
        B: float = 1.0,
        beta: Optional[float] = None,
        alpha: float = 1.0,
        noise: bool = False,
        noise_var: float = 1.0,
        random: bool = False,
        malicious: bool = False,
    ):
        del rm_cached

        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        Bsz, K, TLp1 = to_rm_eval.shape

        flat_trme = to_rm_eval.view(Bsz * K, TLp1)

        if random:
            rewards = torch.randn(Bsz * K, device=self.llm_dev, dtype=prescreen_logits.dtype) * 5.0
        else:
            attn = create_attention_mask(seq_len=flat_trme.shape[1], bsz=flat_trme.shape[0], device=self.rm_dev)
            with torch.no_grad():
                rm_out = self.RM(input_ids=flat_trme.to(self.rm_dev), attention_mask=attn)
            rewards = rm_out.logits.flatten().to(self.llm_dev)

        if noise:
            rewards = rewards + torch.randn_like(rewards) * (noise_var ** 0.5)
        if malicious:
            sorted_vals, sort_idx = torch.sort(rewards)
            rev = torch.zeros_like(rewards)
            rev[sort_idx] = sorted_vals.flip(0)
            rewards = rev

        lm_logits = prescreen_logits.flatten().to(self.llm_dev)

        if reward_shaping:
            assert Bsz == 1, "reward_shaping assumes batch size 1."
            if alpha <= 0:  # 
                raise ValueError(f"alpha must be > 0, got {alpha}")  # 

            lam, beta_eff = self._resolve_lambda_beta(weight, beta)
            if beta_eff is None:
                beta_eff = 1.0 / max(1e-12, lam)

          
            p = F.softmax(lm_logits, dim=-1)
            r_min = rewards.min()
            rewards_shifted = rewards - r_min
            r_max = rewards_shifted.max()
            B = min(1.01 * r_max, B)

            c = min(float(torch.exp(torch.tensor(float(B) / float(beta_eff))).item()),2)
            m = self._pad_threshold(rewards, p, c)

            # -------- SOFT shaping --------
            m_t = torch.tensor(m, device=self.llm_dev, dtype=rewards.dtype)  # 
            shaped = float(B) * torch.sigmoid(float(alpha) * (rewards - m_t))  #
            fused = lm_logits + lam * shaped  # 
            # -----------------------------
        else:
            fused = lm_logits + weight * rewards

        if method == "greedy":
            top_idx = torch.argmax(fused)
        elif method == "topk":
            assert input_ids.shape[0] == 1, "Sampling assumes batch size 1."
            scores = F.softmax(fused / max(1e-8, float(temperature)), dim=-1)
            top_idx = torch.multinomial(scores, num_samples=1, generator=self._torch_gen).squeeze(0)
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
        reward_shaping: bool = False,
        B: float = 1.0,
        beta: Optional[float] = None,
        alpha: float = 1.0,
        seed: Optional[int] = None,
        noise: bool = False,
        noise_var: float = 1.0,
        random: bool = False,
        malicious: bool = False,
    ) -> torch.Tensor:
        if seed is not None:
            self.set_seed(int(seed))

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
        for _ in range(max_new_token):
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
                        mout, tokens,
                        pre_screen_beam_width=topk,
                        weight=weight,
                        rm_cached=None,
                        chunk_size=chunk_size,
                        debug=debug,
                        reward_shaping=reward_shaping,
                        B=B,
                        beta=beta,
                        alpha=alpha,
                        noise=noise, noise_var=noise_var,
                        random=random, malicious=malicious,
                    )
                else:
                    tokens, _ = self.generate_step(
                        mout, tokens,
                        pre_screen_beam_width=topk,
                        weight=weight,
                        method=method,
                        temperature=temperature,
                        rm_cached=None,
                        debug=debug,
                        reward_shaping=reward_shaping,
                        B=B,
                        beta=beta,
                        alpha=alpha,
                        noise=noise, noise_var=noise_var,
                        random=random, malicious=malicious,
                    )

                if tokens is None:
                    return None
                if tokens.shape[-1] >= self.max_len_cap:
                    if debug:
                        print(f"[generate] reached model_max_length={self.max_len_cap}, stopping.")
                    break

        return tokens
