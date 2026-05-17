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

        self.LLM = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch_dtype
        ).to(self.llm_dev).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.RM = AutoModelForSequenceClassification.from_pretrained(
            rm_path, num_labels=1, torch_dtype=torch_dtype
        ).to(self.rm_dev).eval()

        tmax = getattr(self.tokenizer, "model_max_length", 2048)
        self.max_len_cap = 2048 if (isinstance(tmax, int) and tmax > 10_000_000) else int(tmax or 2048)

    # --------------------- helpers ---------------------

    def get_input_ids(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.llm_dev)
    
    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    # ----------------- one-step (large) ----------------

    def generate_greedy_step_large(
        self,
        mout,
        input_ids: torch.Tensor,
        pre_screen_beam_width: int = 40,
        B: float = 0.0,
        lambda_coef: float = 1.0,                
        stretch: float = 1.01,
        chunk_size: int = 10,
        debug: bool = False,
        noise: bool = False,       
        noise_var: float = 1.0,    
    ):
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded = input_ids.unsqueeze(1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded, prescreen_tokens))
        Bsz, K, TLp1 = to_rm_eval.shape

        flat_trme = to_rm_eval.view(Bsz * K, TLp1)
        flat_lm_logits = prescreen_logits.flatten().to(self.llm_dev)

        # -------- FIRST PASS: compute r_min / r_max --------
        all_rewards = []
        for chunk_cpu in iter_chunks(flat_trme, chunk_size):
            chunk = chunk_cpu.to(self.rm_dev)
            attn = create_attention_mask(chunk.shape[1], chunk.shape[0], self.rm_dev)
            with torch.no_grad():
                r = self.RM(input_ids=chunk, attention_mask=attn).logits.flatten()
            all_rewards.append(r.cpu())

        all_rewards = torch.cat(all_rewards)
        if noise:  # apply Gaussian noise to rewards before shaping
            all_rewards = all_rewards + torch.randn_like(all_rewards) * (noise_var ** 0.5)
        r_min = all_rewards.min()
        r_max = all_rewards.max()

        r_range = (r_max - r_min).item()
        denom = max(r_range, 1e-6)

        #  clip B, then multiply by lambda
        B_eff = min(float(B), float(stretch) * r_range)

        # -------- SECOND PASS: greedy selection --------
        best_score = None
        best_seq = None
        offset = 0

        for chunk_cpu in iter_chunks(flat_trme, chunk_size):
            clen = chunk_cpu.shape[0]
            chunk_logits = flat_lm_logits[offset: offset + clen]
            chunk_rewards = all_rewards[offset: offset + clen].to(self.llm_dev)
            offset += clen

            shaped = lambda_coef * B_eff * (chunk_rewards - r_min) / denom  
            fused = chunk_logits + shaped

            val, idx = fused.max(dim=0)
            seq = chunk_cpu[idx.item()].to(self.llm_dev)

            if best_score is None or val.item() > best_score:
                best_score = val.item()
                best_seq = seq

        return best_seq.unsqueeze(0), None

    # ----------------- one-step (standard) ----------------

    def generate_step(
        self,
        mout,
        input_ids: torch.Tensor,
        pre_screen_beam_width: int = 40,
        B: float = 0.0,
        lambda_coef: float = 1.0,                
        stretch: float = 1.01,
        method: str = "greedy",
        temperature: float = 0.7,
        noise: bool = False,       
        noise_var: float = 1.0,   
    ):
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded = input_ids.unsqueeze(1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded, prescreen_tokens))
        Bsz, K, TLp1 = to_rm_eval.shape

        flat = to_rm_eval.view(Bsz * K, TLp1)
        attn = create_attention_mask(flat.shape[1], flat.shape[0], self.rm_dev)

        with torch.no_grad():
            rewards = self.RM(input_ids=flat.to(self.rm_dev), attention_mask=attn).logits.flatten().to(self.llm_dev)
        if noise:  # [ADDED] apply Gaussian noise to rewards before shaping
            rewards = rewards + torch.randn_like(rewards) * (noise_var ** 0.5)

        r_min = rewards.min()
        r_max = rewards.max()
        r_range = (r_max - r_min).item()
        denom = max(r_range, 1e-6)

        # clip B, then multiply by lambda**
        B_eff = min(float(B), float(stretch) * r_range)

        shaped = lambda_coef * B_eff * (rewards - r_min) / denom   
        fused = prescreen_logits.flatten().to(self.llm_dev) + shaped

        if method == "greedy":
            idx = fused.argmax()
        elif method == "topk":
            probs = F.softmax(fused / max(1e-8, temperature), dim=-1)
            idx = torch.multinomial(probs, 1).item()
        else:
            raise ValueError(method)

        return flat[idx].unsqueeze(0).to(self.llm_dev), None

    # ----------------- main decode loop -----------------

    
    def generate(
        self,
        prompt: str,
        B: float = 0.0,
        lambda_coef: float = 1.0,               
        stretch: float = 1.01,
        topk: int = 1,
        max_new_token: int = 128,
        method: str = "greedy",
        temperature: float = 0.7,
        chunk_size: int = 5,
        debug: bool = False,
        noise: bool = False,          
        noise_var: float = 1.0,       
    ):
        tokens = self.get_input_ids(prompt)
        if tokens.shape[-1] >= self.max_len_cap:
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
                        mout,
                        tokens,
                        pre_screen_beam_width=topk,
                        B=B,
                        lambda_coef=lambda_coef,   
                        stretch=stretch,
                        chunk_size=chunk_size,
                        debug=debug,
                        noise=noise, noise_var=noise_var, 
                    )
                else:
                    tokens, _ = self.generate_step(
                        mout,
                        tokens,
                        pre_screen_beam_width=topk,
                        B=B,
                        lambda_coef=lambda_coef,   
                        stretch=stretch,
                        method=method,
                        temperature=temperature,
                        noise=noise, noise_var=noise_var,  
                    )

                if tokens.shape[-1] >= self.max_len_cap:
                    break

        return tokens