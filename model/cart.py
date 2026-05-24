import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util
from .config import CARTConfig
from .norm import RMSNorm
from .attention import MLAKVProjection
from .layers import PreludeLayer, CoreBlock, CoreBlockSelfAttn, CodaLayer
from .hyper import HyperConnection
from .lti import LTIInjection
from .lie import LoopIndexEmbedding


class CART(nn.Module):
    """
    Context-Anchored Recurrent Transformer (CART).

    Forward pass:
        1. Embed tokens
        2. Run P prelude layers (unique weights, causal self-attention)
        3. Store prelude output as e (fixed context for loop)
        4. Compute K, V from e once (KV reuse across all R loops)
        5. Initialize hidden state h = e
        6. Initialize hyper-connection buffer
        7. For r in range(R):
               h_input = hyper.combine(buffer)       — blend previous states
               h_input = lie(h_input, r)             — inject loop-depth signal
               transformer_out = core(h_input, K, V) — shared-weight block
               h = lti(h_input, transformer_out)     — stable recurrent update
               buffer = hyper.update_buffer(buffer, h)
        8. Run 1 coda layer
        9. Project to vocab logits (tied embedding weight)

    Update rule per loop iteration r:
        h_input   = sum_i(w_i * h_{r-i})              [hyper-connection blend]
        h_input   = h_input + LIE(r)                  [loop-depth signal]
        trans_out = CoreBlock(h_input, K, V)           [MLA cross-attn + FFN]
        h_r       = sigmoid(a) * h_input + trans_out  [LTI-stable update]

    Core block parameter budget: MLA 2.75d² + SwiGLU 8d² = 10.75d²
    Effective params = stored params + (R-1) * core_params, giving
    leverage ratio of R at the core block level.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Token embedding (tied with output projection)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Prelude: P layers with unique weights
        self.prelude = nn.ModuleList([
            PreludeLayer(config) for _ in range(config.n_prelude)
        ])

        # KV projection: computes K, V from e once before the loop.
        # Skipped when self_attn_core=True (core uses self-attention, no anchor).
        self.kv_proj = None if config.self_attn_core else MLAKVProjection(config)

        # Core block class depends on self_attn_core flag.
        # CoreBlock: cross-attention against external K, V (CART default).
        # CoreBlockSelfAttn: self-attention on h with RoPE (ablation W).
        block_cls = CoreBlockSelfAttn if config.self_attn_core else CoreBlock

        # Core: one shared block looped R times, OR R unique blocks
        # when unshare_core=True (ablation Z).
        if config.unshare_core:
            self.core = nn.ModuleList(
                [block_cls(config) for _ in range(config.n_loops)]
            )
        else:
            self.core = block_cls(config)

        # Hyper-connections
        self.hyper = HyperConnection(config)

        # LTI stable injection
        self.lti = LTIInjection(config)

        # Loop index embedding
        self.lie = LoopIndexEmbedding(config)

        # Coda: 1 layer
        self.coda = CodaLayer(config)

        # Final norm before output projection
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        # Output projection — tied to embedding weight
        # Use self.embedding.weight.T directly in forward()

        self._use_grad_ckpt = False
        self._init_weights()

    def enable_gradient_checkpointing(self):
        """Wrap core block calls with torch.utils.checkpoint to save VRAM."""
        self._use_grad_ckpt = True

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, T]
        targets: torch.Tensor = None,      # [B, T] — if provided, compute loss
    ):
        B, T = input_ids.shape

        # 1. Embed
        x = self.embedding(input_ids)      # [B, T, d_model]

        # 2. Prelude
        for layer in self.prelude:
            x = layer(x)
        e = x                              # [B, T, d_model] — fixed context

        # 3. Pre-compute K, V from prelude output (reused across all loops)
        #    Skipped when unfreeze_kv=True (recompute per iter) or
        #    self_attn_core=True (core does self-attention, no K/V anchor).
        if not self.config.self_attn_core and not self.config.unfreeze_kv:
            K, V = self.kv_proj(e)         # [B, H, T, D] each

        # 4. Initialize hidden state and hyper-connection buffer
        h = e                              # Start from prelude output
        buffer = self.hyper.init_buffer(h)

        # 5. Recurrent loop
        for r in range(self.config.n_loops):
            # Read prior state. HyperConnection blends last 3 by default;
            # disable_hyper=True falls back to standard residual (most recent only).
            if self.config.disable_hyper:
                h_input = buffer[0]
            else:
                h_input = self.hyper.combine(buffer)
            # Loop-index signal (sinusoidal pe[r]); skipped when disable_lie=True.
            if not self.config.disable_lie:
                h_input = self.lie(h_input, r)
            if self.config.unfreeze_kv:
                K, V = self.kv_proj(h_input)       # recompute from current state
            core_block = self.core[r] if self.config.unshare_core else self.core
            if self.config.self_attn_core:
                # Core does self-attention on h; no K, V passed in.
                if self._use_grad_ckpt and self.training:
                    transformer_out = ckpt_util.checkpoint(
                        core_block, h_input, use_reentrant=False)
                else:
                    transformer_out = core_block(h_input)
            else:
                # Core does cross-attention against external K, V.
                if self._use_grad_ckpt and self.training:
                    transformer_out = ckpt_util.checkpoint(
                        core_block, h_input, K, V, use_reentrant=False)
                else:
                    transformer_out = core_block(h_input, K, V)
            # LTI gate (sigmoid-bounded) by default; standard residual when disabled.
            if self.config.disable_lti:
                h = h_input + transformer_out
            else:
                h = self.lti(h_input, transformer_out)
            buffer = self.hyper.update_buffer(buffer, h)

        # 6. Coda
        h = self.coda(h)

        # 7. Final norm and output projection (tied embeddings)
        h = self.final_norm(h)
        logits = h @ self.embedding.weight.T  # [B, T, vocab_size]

        if targets is None:
            return logits, None

        # FixedOrderDataset pre-shifts: input=tokens[:-1], target=tokens[1:].
        # Spec had logits[:,:-1,:] vs targets[:,1:] — that double-shift produces
        # 2-step-ahead prediction and non-standard perplexity. Corrected to
        # single-shift: logits[t] directly predicts targets[t] = tokens[t+1].
        loss = torch.nn.functional.cross_entropy(
            logits.contiguous().view(-1, self.config.vocab_size),
            targets.contiguous().view(-1),
            ignore_index=-100,
        )
        return logits, loss

    def count_parameters(self) -> dict:
        """Returns total and effective parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        kv_proj_params = (sum(p.numel() for p in self.kv_proj.parameters())
                          if self.kv_proj is not None else 0)
        all_core_params = sum(p.numel() for p in self.core.parameters())
        if self.config.unshare_core:
            # All R blocks already counted in `total`; no leverage multiplier.
            core_params = all_core_params // self.config.n_loops  # per-block
            effective = total
        else:
            # Single shared block; effective counts it R times. KV proj counts
            # once when frozen, R times when unfrozen.
            core_params = all_core_params
            per_iter = core_params + (kv_proj_params if self.config.unfreeze_kv else 0)
            effective = total + (self.config.n_loops - 1) * per_iter
        return {
            "total": total,
            "effective": effective,
            "core_block": core_params,
            "kv_proj": kv_proj_params,
            "leverage": effective / total,
        }
