"""
DenseBaseline: parameter-matched standard transformer for CART comparison.

7 uniform layers, full-rank MHA (no KV compression, no weight sharing),
SwiGLU FFN, RMSNorm pre-norm, RoPE applied on every layer, tied embeddings.

Parameter match at each scale (CART with P=6, best-R from Stage 1):
    d=256  7L: ~14.2M  (CART ~14.4M)
    d=512  7L: ~40.2M  (CART ~41.0M)
    d=768  7L: ~74.1M  (CART ~75.3M)
    d=1024 7L: ~122.7M (CART ~125.1M)

Per-layer parameter budget:
    MHA: 4 × d² (Q, K, V, O — full rank, no compression)
    FFN: 3 × d × ffn_intermediate  ≈ 8d²
    Total per layer: ~12d²  (vs CART core 10.75d² + MLA KV compression)
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .ffn import SwiGLUFFN
from .rope import RotaryEmbedding


@dataclass
class DenseConfig:
    d_model: int = 512
    n_layers: int = 7           # Fixed for parameter matching across all CART scales
    vocab_size: int = 32_000
    max_seq_len: int = 1024
    d_head: int = 64
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    rope_base: float = 10_000.0
    dropout: float = 0.0

    @property
    def n_heads(self) -> int:
        assert self.d_model % self.d_head == 0, \
            f"d_model {self.d_model} must be divisible by d_head {self.d_head}"
        return self.d_model // self.d_head

    @property
    def ffn_intermediate(self) -> int:
        nominal = int(8 / 3 * self.d_model)
        return math.ceil(nominal / 256) * 256

    def validate(self):
        assert self.d_model % 64 == 0, "d_model must be divisible by 64"
        assert self.n_layers >= 1, "n_layers must be at least 1"


class StandardMHASelfAttention(nn.Module):
    """
    Standard multi-head self-attention: full-rank Q, K, V, O projections.
    No KV latent compression (unlike CART's MLA).
    RoPE applied to Q and K. Causal. Flash Attention via SDPA.
    """
    def __init__(self, config: DenseConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_head
        d     = config.d_model
        h_dim = config.n_heads * config.d_head

        self.q_proj = nn.Linear(d, h_dim, bias=False)
        self.k_proj = nn.Linear(d, h_dim, bias=False)
        self.v_proj = nn.Linear(d, h_dim, bias=False)
        self.o_proj = nn.Linear(h_dim, d, bias=False)

        self.rope = RotaryEmbedding(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        Q = self.rope(Q, T)
        K = self.rope(K, T)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)


class DenseLayer(nn.Module):
    """Standard pre-norm transformer layer: RMSNorm → Attn → residual, RMSNorm → FFN → residual."""

    def __init__(self, config: DenseConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn  = StandardMHASelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn   = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DenseBaseline(nn.Module):
    """
    Parameter-matched standard transformer for CART architectural comparison.

    Forward pass:
        1. Embed tokens
        2. Run n_layers identical DenseLayers (each with its own weights)
        3. Final RMSNorm
        4. Project to vocab logits (tied embedding weight)

    No weight sharing, no recurrent state, no LTI injection, no hyper-connections.
    This is the standard GPT-2 / LLaMA style architecture (pre-norm, SwiGLU, RoPE).
    """
    def __init__(self, config: DenseConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.embedding  = nn.Embedding(config.vocab_size, config.d_model)
        self.layers     = nn.ModuleList([DenseLayer(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, T]
        targets:   torch.Tensor = None # [B, T] — if provided, compute CE loss
    ):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = x @ self.embedding.weight.T   # [B, T, vocab_size]

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.contiguous().view(-1, self.config.vocab_size),
            targets.contiguous().view(-1),
            ignore_index=-100,
        )
        return logits, loss

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            "total":     total,
            "effective": total,  # No weight sharing — stored == effective
            "leverage":  1.0,
        }
