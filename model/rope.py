import torch
import torch.nn as nn
from .config import CARTConfig


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings (RoPE).
    Applied to Q and K in attention. Not re-applied per loop iteration.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        d = config.d_head
        base = config.rope_base
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device, dtype):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, seq_len: int):
        # x: [batch, n_heads, seq_len, d_head]
        self._update_cache(seq_len, x.device, x.dtype)
        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]
        return apply_rotary(x, cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # cos/sin: [seq_len, d_head]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_head]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin
