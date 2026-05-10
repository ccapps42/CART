"""
lm-evaluation-harness adapter for CART and DenseBaseline checkpoints.

Implements the lm_eval.api.model.LM interface for:
  - loglikelihood       (HellaSwag, ARC-C, PIQA, LAMBADA)
  - loglikelihood_rolling  (WikiText perplexity)
  - generate_until      (open-ended generation tasks)

Usage (via run_benchmarks.py — don't call this directly):
    from eval.lm_eval_adapter import CARTLMEval
    model = CARTLMEval(checkpoint="checkpoints/<id>/step_61000.pt")
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.cart import CART
from model.config import CARTConfig
from model.dense import DenseBaseline, DenseConfig

TOKENIZER_NAME = "NousResearch/Llama-2-7b-hf"


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, CARTConfig):
        model = CART(config)
    elif isinstance(config, DenseConfig):
        model = DenseBaseline(config)
    else:
        raise ValueError(f"Unknown config type: {type(config)}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model, config


class CARTLMEval(LM):
    def __init__(self, checkpoint: str, batch_size: int = 16, device: str = "cuda"):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = int(batch_size)

        self.model, self.config = load_checkpoint(checkpoint, self._device)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._max_length = self.config.max_seq_len

    # --- required properties ---

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def add_special_tokens(self) -> bool:
        return False

    # --- tokenizer helpers ---

    def tok_encode(self, string: str, add_special_tokens: bool = False) -> list:
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens) -> str:
        return self.tokenizer.decode(tokens)

    # --- internal helpers ---

    def _encode_pair(self, context: str, continuation: str) -> tuple:
        """Encode context+continuation, truncating context from left if needed."""
        ctx_tokens = self.tok_encode(context)
        cont_tokens = self.tok_encode(continuation)
        full = ctx_tokens + cont_tokens
        if len(full) > self._max_length:
            ctx_tokens = ctx_tokens[-(self._max_length - len(cont_tokens)):]
            full = ctx_tokens + cont_tokens
        return full, len(ctx_tokens)

    def _forward_batch(self, token_seqs: list) -> torch.Tensor:
        """Forward pass on right-padded batch. Returns logits [B, T, vocab]."""
        max_len = max(len(s) for s in token_seqs)
        padded = [s + [self.eot_token_id] * (max_len - len(s)) for s in token_seqs]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self._device)
        with torch.no_grad():
            logits, _ = self.model(input_ids)
        return logits

    # --- lm_eval interface ---

    def loglikelihood(self, requests) -> list:
        results = [None] * len(requests)

        encoded = []
        for req in requests:
            context, continuation = req.args
            full_tokens, ctx_len = self._encode_pair(context, continuation)
            encoded.append((full_tokens, ctx_len))

        # Sort by length for efficient batching
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))

        for batch_start in range(0, len(order), self._batch_size):
            batch_idx = order[batch_start:batch_start + self._batch_size]
            batch = [encoded[i] for i in batch_idx]

            # input = full_tokens[:-1], target = full_tokens[1:]
            input_seqs = [tokens[:-1] for tokens, _ in batch]
            logits = self._forward_batch(input_seqs)  # [B, T, vocab]

            for j, (orig_idx, (tokens, ctx_len)) in enumerate(zip(batch_idx, batch)):
                log_probs = F.log_softmax(logits[j], dim=-1)  # [T, vocab]

                # continuation positions in the shifted sequence
                cont_start = max(0, ctx_len - 1)
                cont_end = len(tokens) - 1
                cont_targets = torch.tensor(
                    tokens[ctx_len:], dtype=torch.long, device=self._device
                )

                cont_log_probs = log_probs[cont_start:cont_end]
                ll = cont_log_probs.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1).sum().item()
                is_greedy = (cont_log_probs.argmax(-1) == cont_targets).all().item()
                results[orig_idx] = (ll, bool(is_greedy))

        return results

    def loglikelihood_rolling(self, requests) -> list:
        """Rolling log-likelihood over full strings (used for WikiText perplexity)."""
        results = []
        for req in requests:
            tokens = self.tok_encode(req.args[0])
            total_ll = 0.0
            for start in range(0, max(1, len(tokens) - 1), self._max_length):
                chunk = tokens[start:start + self._max_length + 1]
                if len(chunk) < 2:
                    break
                input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    logits, _ = self.model(input_ids)
                log_probs = F.log_softmax(logits[0], dim=-1)
                targets = torch.tensor(chunk[1:], dtype=torch.long, device=self._device)
                total_ll += log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item()
            results.append(total_ll)
        return results

    def generate_until(self, requests) -> list:
        """Greedy generation until stop strings (no KV cache — fine for short gens)."""
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            tokens = self.tok_encode(context)
            tokens = tokens[-(self._max_length - max_gen):]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device)

            generated = []
            with torch.no_grad():
                for _ in range(max_gen):
                    if input_ids.shape[1] >= self._max_length:
                        break
                    logits, _ = self.model(input_ids)
                    next_tok = logits[0, -1, :].argmax().item()
                    generated.append(next_tok)
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[next_tok]], dtype=torch.long, device=self._device)
                    ], dim=1)
                    decoded = self.tokenizer.decode(generated)
                    if next_tok == self.eot_token_id or any(s in decoded for s in until):
                        break

            results.append(self.tokenizer.decode(generated))
        return results
