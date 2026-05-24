from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CARTConfig:
    # --- Sweep variables (set per run) ---
    d_model: int = 512          # Must be divisible by 64
    n_loops: int = 6            # R: number of recurrent core iterations
    n_prelude: int = 4          # P: number of prelude layers

    # --- Fixed architectural decisions ---
    n_coda: int = 1             # Always 1, do not sweep
    vocab_size: int = 32_000
    max_seq_len: int = 1024     # Allocated; actual seq_len set in training config

    # --- Attention (MLA) ---
    d_head: int = 64            # Fixed; n_heads derived automatically
    mla_compression_ratio: int = 4  # KV latent dim = d_model // 4

    # --- Hyper-connections ---
    n_hyper: int = 3            # Number of previous loop states to combine
    # Weights initialized to [1.0, 0.0, 0.0] — residual baseline

    # --- LTI stability (Parcae, Prairie et al. 2026) ---
    lti_init_value: float = 0.9  # Initial A diagonal; sigmoid_inverse(0.9) used in init
    # A parameterized as sigmoid(a_param): guarantees A_ii in (0,1), rho(A) < 1

    # --- Loop Index Embedding (LIE) ---
    lie_dim: int = 32            # Sinusoidal encoding dim — fixed, not swept

    # --- Normalization ---
    rms_norm_eps: float = 1e-6

    # --- Embedding ---
    tie_embeddings: bool = True  # Input embedding = output projection weight

    # --- Positional encoding ---
    rope_base: float = 10_000.0  # Applied in prelude and coda only, not per loop

    # --- Training ---
    dropout: float = 0.0        # No dropout for sweep runs

    # --- Ablation: unfreeze KV ---
    # When True, K and V are recomputed from the current hidden state h
    # at every core iteration (using the same kv_proj weights), instead of
    # being computed once from the prelude output e and frozen across the loop.
    # Diagnostic for whether the frozen-KV efficiency trick is the architectural
    # ceiling vs. recurrence/shared-weights being the bottleneck.
    unfreeze_kv: bool = False

    # --- Ablation: unshared core weights ---
    # When True, the recurrent core is replaced with n_loops unique CoreBlocks
    # (each iteration uses its own weight set) instead of a single shared block
    # looped n_loops times. Tests whether shared-weight leverage was costing
    # capacity vs being a feature. Stored parameters grow by ~(n_loops - 1) *
    # core_params; effective parameters equal stored (no leverage multiplier).
    unshare_core: bool = False

    # --- Ablation: self-attention core ---
    # When True, the recurrent core uses MLA self-attention on h (with RoPE)
    # instead of cross-attention against frozen K, V from the prelude. The
    # prelude's kv_proj module is skipped entirely. Tests whether the cross-
    # attention-against-anchor template (and not just shared weights) was
    # capping the architecture below a vanilla transformer of equivalent depth.
    # Typically combined with unshare_core=True for a clean comparison vs Dense.
    # Mutually exclusive with unfreeze_kv (no K, V cache exists).
    self_attn_core: bool = False

    # --- Ablation: disable HyperConnection ---
    # When True, the recurrent core skips HyperConnection's weighted blend of
    # prior hidden states; each iteration reads the most recent state directly
    # (h_input = buffer[0], equivalent to standard residual recurrence). The
    # HyperConnection module is still instantiated but its combine() call is
    # bypassed. Tests whether HyperConnection's per-iteration cross-state
    # blending earns its keep in CART's native (shared-weight) configuration.
    disable_hyper: bool = False

    # --- Ablation: disable LTI gate ---
    # When True, the LTI gate is bypassed: h = h_input + transformer_out
    # (standard residual) instead of h = sigmoid(a) * h_input + transformer_out.
    # The LTI module is still instantiated (so spectral_radius() logging
    # still functions) but its parameters are unused at forward time. Tests
    # whether the sigmoid-bounded gate's stability constraint, designed for
    # R -> infinity convergence guarantees, is restricting residual flow at
    # the fixed R = 6 we actually train with.
    disable_lti: bool = False

    # --- Ablation: disable LIE ---
    # When True, the Loop Index Embedding is skipped — h_input goes into the
    # core block without the sinusoidal pe[r] signal added. The LIE module
    # is still instantiated but its forward call is bypassed. In the shared-
    # weight case this is the most architecturally consequential machinery
    # change: LIE is the only mechanism by which the shared block can know
    # which iteration it is processing. Without it, all R iterations are
    # functionally identical apart from the evolving h state. Tests whether
    # the shared block actually uses the loop-index signal for differentiation.
    disable_lie: bool = False

    @property
    def n_heads(self) -> int:
        assert self.d_model % self.d_head == 0, \
            f"d_model {self.d_model} must be divisible by d_head {self.d_head}"
        return self.d_model // self.d_head

    @property
    def d_kv_latent(self) -> int:
        return self.d_model // self.mla_compression_ratio

    @property
    def ffn_intermediate(self) -> int:
        nominal = int(8 / 3 * self.d_model)
        # Round up to multiple of 256
        return math.ceil(nominal / 256) * 256

    def validate(self):
        assert self.d_model % 64 == 0, "d_model must be divisible by 64"
        assert self.n_loops >= 1, "n_loops must be at least 1 (R=1 is the no-recurrence diagnostic)"
        assert self.n_prelude >= 2, "n_prelude must be at least 2"
        assert self.n_coda == 1, "n_coda must be 1 (fixed)"
        assert not (self.self_attn_core and self.unfreeze_kv), \
            "self_attn_core=True makes unfreeze_kv meaningless (no K, V cache exists)"
