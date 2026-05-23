import torch
import torch.nn as nn
from .config import CARTConfig
from .norm import RMSNorm
from .attention import MLASelfAttention, MLACrossAttention
from .ffn import SwiGLUFFN


class PreludeLayer(nn.Module):
    """
    Standard transformer layer with unique weights.
    Runs P times sequentially, each with its own parameter set.
    Self-attention: tokens attend to all previous tokens (causal).
    RoPE applied inside MLASelfAttention.
    Pre-norm: RMSNorm before each sub-layer, residual after.
    No dropout.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn  = MLASelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn   = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CoreBlock(nn.Module):
    """
    The shared-weight recurrent block. A single instance of this class
    is looped R times. Weights are shared across all R iterations.

    Processing order per iteration:
        1. Cross-attention: h_t queries prelude output e (K, V pre-computed)
        2. SwiGLU FFN: per-token transformation

    All sub-layers use pre-norm + residual.
    Cross-attention does NOT apply RoPE (see MLACrossAttention docstring).

    LTI injection and LIE signal are applied by the caller (CART.forward)
    before and after this block respectively — not inside CoreBlock.
    CoreBlock receives h_input (already LIE-injected) and returns
    transformer_out (the raw block output, before LTI combination).

    Parameter budget: MLA 2.75d² + SwiGLU 8d² = 10.75d²
    Same as a prelude or coda layer — the leverage comes entirely
    from looping R times, not from the block being larger.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn  = MLACrossAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn   = SwiGLUFFN(config)

    def forward(
        self,
        h: torch.Tensor,    # [B, T, d_model] — h_input from hyper-connection + LIE
        K: torch.Tensor,    # [B, H, T, D] — pre-computed K from prelude output
        V: torch.Tensor,    # [B, H, T, D] — pre-computed V from prelude output
    ) -> torch.Tensor:
        h = h + self.attn(self.norm1(h), K, V)
        h = h + self.ffn(self.norm2(h))
        return h


class CoreBlockSelfAttn(nn.Module):
    """
    Variant of CoreBlock that uses MLA self-attention on h instead of
    cross-attention against the prelude's K, V. Structurally identical
    to PreludeLayer — used as the recurrent core when self_attn_core=True
    to test whether the cross-attention-against-anchor template (and not
    just weight sharing) was capping the architecture.

    Adds ~kv_down + k_up + v_up parameters per block vs CoreBlock, since
    K, V are now produced inside the block from h rather than passed in
    from the prelude's kv_proj.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn  = MLASelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn   = SwiGLUFFN(config)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.norm1(h))
        h = h + self.ffn(self.norm2(h))
        return h


class CodaLayer(nn.Module):
    """
    Single output transformation layer with unique weights.
    Self-attention over the final hidden state h_R.
    Identical structure to PreludeLayer but receives the loop output, not raw embeddings.
    Always exactly 1 coda layer — this is fixed, not swept.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn  = MLASelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn   = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
