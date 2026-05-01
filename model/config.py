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
        assert self.n_loops >= 2, "n_loops must be at least 2"
        assert self.n_prelude >= 2, "n_prelude must be at least 2"
        assert self.n_coda == 1, "n_coda must be 1 (fixed)"
