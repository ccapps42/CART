import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util
from .config import CARTConfig
from .norm import RMSNorm
from .attention import MLAKVProjection
from .layers import PreludeLayer, CoreBlock, CodaLayer
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

        # KV projection: computes K, V from e once before the loop
        self.kv_proj = MLAKVProjection(config)

        # Core: one shared-weight block, looped R times
        self.core = CoreBlock(config)

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
        K, V = self.kv_proj(e)             # [B, H, T, D] each

        # 4. Initialize hidden state and hyper-connection buffer
        h = e                              # Start from prelude output
        buffer = self.hyper.init_buffer(h)

        # 5. Recurrent loop
        for r in range(self.config.n_loops):
            h_input = self.hyper.combine(buffer)   # blend previous states
            h_input = self.lie(h_input, r)         # inject loop-depth signal
            if self._use_grad_ckpt and self.training:
                # Checkpoint the core block to trade VRAM for recomputation.
                # LIE and LTI stay outside — only the heavy cross-attn+FFN is wrapped.
                transformer_out = ckpt_util.checkpoint(
                    self.core, h_input, K, V, use_reentrant=False)
            else:
                transformer_out = self.core(h_input, K, V)
            h = self.lti(h_input, transformer_out) # LTI-stable update
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
        # Effective: count core parameters R times
        core_params = sum(p.numel() for p in self.core.parameters())
        kv_proj_params = sum(p.numel() for p in self.kv_proj.parameters())
        # KV proj is paid once (not R times) due to KV reuse
        effective = total + (self.config.n_loops - 1) * core_params
        return {
            "total": total,
            "effective": effective,
            "core_block": core_params,
            "kv_proj": kv_proj_params,
            "leverage": effective / total,
        }
