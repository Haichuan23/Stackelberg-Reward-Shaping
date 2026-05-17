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
    Reward-guided search with mean-std reward shaping.
    """

    def __init__(
        self,
        llm_path: str,
        rm_path: str,
        llm_dev: str = "cuda:0",
        rm_dev: str = "cuda:1",
        torch_dtype: torch.dtype = torch.float16,
        eps: float = 1e-6,                        
    ):
        self.llm_dev = torch.device(llm_dev)
        self.rm_dev = torch.device(rm_dev)
        self.eps = eps                           

        print("Loading LLM...")
        self.LLM = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch_dtype
        ).to(self.llm_dev)
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
        noise: bool = False,       
        noise_var: float = 1.0,    
    ):
        """
        Mean-std shaping across ALL chunks.
        """
        del rm_cached

        # 1) LM top-k prescreen
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        B, K, TLp1 = to_rm_eval.shape
        flat_trme = to_rm_eval.view(B * K, TLp1)

        flat_lm_logits = prescreen_logits.flatten()

        # ---------------- FIRST PASS: collect all rewards ----------------
        all_rewards = []                          
        for chunk_cpu in iter_chunks(flat_trme, chunk_size=max(1, int(chunk_size))):
            chunk = chunk_cpu.to(self.rm_dev)
            attn = create_attention_mask(chunk.shape[1], chunk.shape[0], self.rm_dev)
            with torch.no_grad():
                rm_out = self.RM(input_ids=chunk, attention_mask=attn)
            all_rewards.append(rm_out.logits.flatten().to(self.llm_dev)) 

        all_rewards = torch.cat(all_rewards, dim=0)                          
        if noise:  # [ADDED] apply Gaussian noise to rewards before shaping
            all_rewards = all_rewards + torch.randn_like(all_rewards) * (noise_var ** 0.5)
        mean_r = all_rewards.mean()                                            
        std_r = all_rewards.std(unbiased=False).clamp_min(self.eps)           

        if debug:
            print(f"[meanstd] mean={mean_r.item():.4f}, std={std_r.item():.4f}")

        # ---------------- SECOND PASS: greedy selection ----------------
        current_best_score = None
        current_best_tokens = None
        offset = 0

        for chunk_cpu in iter_chunks(flat_trme, chunk_size=max(1, int(chunk_size))):
            chunk_len = chunk_cpu.shape[0]
            lm_logits = flat_lm_logits[offset: offset + chunk_len].to(self.llm_dev)
            rewards = all_rewards[offset: offset + chunk_len]                 
            offset += chunk_len

            shaped_rewards = (rewards - mean_r) / std_r                       
            fused = lm_logits + weight * shaped_rewards                        

            local_best_val, local_best_idx = torch.max(fused, dim=0)
            local_best_seq = chunk_cpu[local_best_idx.item()].to(self.llm_dev)

            if current_best_score is None or local_best_val.item() > current_best_score:
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
        noise: bool = False,       
        noise_var: float = 1.0,    
    ):
        """
        Mean-std shaping across all B*K candidates.
        """
        del rm_cached

        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        B, K, TLp1 = to_rm_eval.shape

        flat_trme = to_rm_eval.view(B * K, TLp1)
        attn = create_attention_mask(flat_trme.shape[1], flat_trme.shape[0], self.rm_dev)

        with torch.no_grad():
            rm_out = self.RM(input_ids=flat_trme.to(self.rm_dev), attention_mask=attn)

        rewards = rm_out.logits.flatten().to(self.llm_dev)
        if noise:  # [ADDED] apply Gaussian noise to rewards before shaping
            rewards = rewards + torch.randn_like(rewards) * (noise_var ** 0.5)

        # -------- mean-std shaping --------
        mean_r = rewards.mean()                                               
        std_r = rewards.std(unbiased=False).clamp_min(self.eps)               
        shaped_rewards = (rewards - mean_r) / std_r                           

        fused = prescreen_logits.flatten().to(self.llm_dev) + weight * shaped_rewards  

        if method == "greedy":
            top_idx = torch.argmax(fused)
        elif method == "topk":
            assert input_ids.shape[0] == 1
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
        noise: bool = False,           
        noise_var: float = 1.0,        
    ) -> torch.Tensor:

        tokens = self.get_input_ids(prompt)
        initial_len = tokens.shape[-1]

        if chunk_size == "auto":
            chunk_size = auto_size(initial_len + max_new_token, topk)

        if tokens.shape[-1] >= self.max_len_cap:
            print(f"[ARGS] Prompt too long for model_max_length={self.max_len_cap}.")
            return None

        cached = None

        for _ in range(max_new_token):
            with torch.no_grad():
                attn = create_attention_mask(tokens.shape[1], tokens.shape[0], self.llm_dev)
                if cached is None:
                    mout = self.LLM(
                        **self.LLM.prepare_inputs_for_generation(
                            input_ids=tokens, attention_mask=attn, use_cache=True
                        )
                    )
                else:
                    mout = self.LLM(
                        input_ids=tokens[:, -1:],
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
                        chunk_size=chunk_size,
                        debug=debug,
                        noise=noise, noise_var=noise_var,  
                    )
                else:
                    tokens, _ = self.generate_step(
                        mout, tokens,
                        pre_screen_beam_width=topk,
                        weight=weight,
                        method=method,
                        temperature=temperature,
                        debug=debug,
                        noise=noise, noise_var=noise_var,  
                    )

                if tokens.shape[-1] >= self.max_len_cap:
                    break

        return tokens