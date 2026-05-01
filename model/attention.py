import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import CARTConfig
from .rope import RotaryEmbedding, apply_rotary


class MLASelfAttention(nn.Module):
    """
    MLA self-attention for prelude and coda layers.
    Compresses K, V through a latent bottleneck of dimension d_kv_latent.
    Q is full-rank. RoPE applied to Q and K.
    Flash Attention via scaled_dot_product_attention (Ampere and later).
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.d_kv_latent = config.d_kv_latent

        # Q: full rank
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        # KV: compress then expand
        self.kv_down = nn.Linear(config.d_model, config.d_kv_latent, bias=False)
        self.k_up    = nn.Linear(config.d_kv_latent, config.n_heads * config.d_head, bias=False)
        self.v_up    = nn.Linear(config.d_kv_latent, config.n_heads * config.d_head, bias=False)
        # Output
        self.o_proj  = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        self.rope = RotaryEmbedding(config)
        self.scale = config.d_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        latent = self.kv_down(x)
        K = self.k_up(latent).view(B, T, H, D).transpose(1, 2)
        V = self.v_up(latent).view(B, T, H, D).transpose(1, 2)

        # Apply RoPE to Q and K
        Q = self.rope(Q, T)
        K = self.rope(K, T)

        # Flash Attention (is_causal=True for autoregressive)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)


class MLACrossAttention(nn.Module):
    """
    MLA cross-attention for the recurrent core block.

    Q is derived from h_t (the current hidden state).
    K and V are derived from e (the prelude output) and are passed in
    as pre-computed constants — computed once before the loop begins
    and reused across all R iterations.

    RoPE is NOT re-applied here. K and V already carry positional
    information from when they were computed in the prelude context.
    Q also does NOT receive RoPE here — the hidden state h_t does
    not have a direct correspondence to token positions.

    Attention is NOT causal here (cross-attention from h_t to e).
    The causal mask was already applied in the prelude when e was built.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Q only — no KV projections here (those are on the prelude side)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        self.scale = config.d_head ** -0.5

    def forward(
        self,
        h: torch.Tensor,     # [B, T, d_model] — current hidden state
        K: torch.Tensor,     # [B, H, T, D] — pre-computed from prelude output e
        V: torch.Tensor,     # [B, H, T, D] — pre-computed from prelude output e
    ) -> torch.Tensor:
        B, T, _ = h.shape
        H, D = self.n_heads, self.d_head

        Q = self.q_proj(h).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        # Non-causal cross-attention: h_t tokens attend to all e tokens
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)


class MLAKVProjection(nn.Module):
    """
    Computes K and V from the prelude output e.
    Called once before the loop begins. Output K, V are passed to
    MLACrossAttention on every loop iteration.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_kv_latent = config.d_kv_latent

        self.kv_down = nn.Linear(config.d_model, config.d_kv_latent, bias=False)
        self.k_up    = nn.Linear(config.d_kv_latent, config.n_heads * config.d_head, bias=False)
        self.v_up    = nn.Linear(config.d_kv_latent, config.n_heads * config.d_head, bias=False)

    def forward(self, e: torch.Tensor):
        # e: [B, T, d_model]
        B, T, _ = e.shape
        H, D = self.n_heads, self.d_head
        latent = self.kv_down(e)
        K = self.k_up(latent).view(B, T, H, D).transpose(1, 2)   # [B, H, T, D]
        V = self.v_up(latent).view(B, T, H, D).transpose(1, 2)   # [B, H, T, D]
        return K, V
