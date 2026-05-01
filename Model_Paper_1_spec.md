# Model_Paper_1 — Implementation Specification

**Project name:** Model_Paper_1  
**For:** Claude Code  
**Purpose:** Implement the Context-Anchored Recurrent Transformer (CART) for a systematic
parameter sweep across d_model, loop count R, and prelude depth P on consumer GPU
hardware (RTX 3050 / RTX 3090). Results will underpin a research paper characterizing
shared-weight recurrent depth as an efficient axis of model capacity.

**Host environment:** Windows 11, Python 3.14, PyTorch with CUDA (Ampere GPUs).
All code must be Windows-compatible. Use `pathlib.Path` for all file paths — never
hardcode forward-slash separators. Use `os.cpu_count()` for worker counts.

**Critical constraint:** Every design decision in this document is final and
intentional. Do not substitute alternatives without flagging the reason. The
architecture choices were made deliberately for research validity. Sweep
comparability depends on identical implementation across all configs.

---

## 1. Project Structure

```
Model_Paper_1/
  model/
    __init__.py
    config.py          — CARTConfig dataclass
    norm.py            — RMSNorm
    ffn.py             — SwiGLUFFN
    attention.py       — MLAAttention (self-attention and cross-attention variants)
    hyper.py           — HyperConnection
    lti.py             — LTIInjection (spectral radius stability constraint)
    lie.py             — LoopIndexEmbedding (sinusoidal loop-depth signal)
    layers.py          — PreludeLayer, CoreBlock, CodaLayer
    cart.py          — CART (full model)
  data/
    tokenize.py        — Pre-tokenization script (run once before any sweep)
    dataset.py         — FixedOrderDataset
  train/
    train_one.py       — Trains a single config, writes results to DB, exits
    lr_schedule.py     — Cosine schedule with warmup
  sweep/
    schema.sql         — SQLite schema (source of truth)
    generate_configs.py — Populates configs table from sweep parameters
    orchestrate.py     — Runs pending configs sequentially, handles failures
    analyze.py         — Ranks Stage 1 results, proposes Stage 2 configs
  eval/
    perplexity.py      — Computes perplexity on held-out sets
  plot/
    plot_sweep.py      — Generates paper figures from results.db
  data/                — Populated by tokenize.py
    tinystories_train.bin
    tinystories_val.bin
    wikipedia_val.bin
    fineweb_val.bin
  checkpoints/         — Auto-created, one subdir per config_id
  results.db           — Auto-created by generate_configs.py
  requirements.txt
  README.md
```

---

## 2. Environment and Dependencies

**Python:** 3.14 (host environment — do not target an older version)  
**OS:** Windows 11  
**CUDA:** Ampere architecture — RTX 3050 (8GB) and RTX 3090 (24GB)

```
# requirements.txt
torch>=2.1.0          # scaled_dot_product_attention dispatches Flash Attention on Ampere
transformers>=4.35.0  # Tokenizer only
datasets>=2.14.0      # Data loading for pre-tokenization
bitsandbytes>=0.44.0  # 8-bit AdamW — use 0.44.0+ for Windows support
numpy>=1.24.0
tqdm
```

**Windows compatibility notes:**
- Use `pathlib.Path` for all file paths throughout the codebase
- `bitsandbytes` 0.44.0+ has native Windows support — do not use older versions
- `torch.utils.data.DataLoader` with `num_workers > 0` requires `if __name__ == '__main__':` guards on Windows. Use `num_workers=0` for simplicity unless profiling shows it is a bottleneck
- File locking for SQLite on Windows uses a different mechanism than Linux — use WAL mode: `PRAGMA journal_mode=WAL` on DB open

**No external CUDA compilation dependencies.** All components use standard PyTorch
operations and the built-in Flash Attention dispatch via
`scaled_dot_product_attention`. This is a deliberate design decision — sweep
reliability requires that all 64 configs complete without environment-specific
failures. There is no `mamba-ssm` or other Triton-compiled dependency.

---

## 3. Model Configuration

```python
# model/config.py
from dataclasses import dataclass, field
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

    # --- FFN (SwiGLU) ---
    # Intermediate dim rounded to nearest multiple of 256 for hardware efficiency
    # Nominal: int(8/3 * d_model), then round up to multiple of 256

    # --- Hyper-connections ---
    n_hyper: int = 3            # Number of previous loop states to combine
    # Weights initialized to [1.0, 0.0, 0.0] — residual baseline

    # --- LTI stability (Parcae, Prairie et al. 2026) ---
    lti_init_value: float = 0.9  # Initial A diagonal; sigmoid_inverse(0.9) used in init
    # A parameterized as sigmoid(a_param): guarantees A_ii in (0,1), rho(A) < 1

    # --- Loop Index Embedding (LIE) ---
    lie_dim: int = 32            # Sinusoidal encoding dim — fixed, not swept
    # Projected to d_model and added to h before each CoreBlock pass

    # --- Normalization ---
    rms_norm_eps: float = 1e-6

    # --- Embedding ---
    tie_embeddings: bool = True  # Input embedding = output projection weight

    # --- Positional encoding ---
    rope_base: float = 10_000.0  # Applied in prelude and coda only, not per loop

    # --- Training ---
    dropout: float = 0.0        # No dropout for sweep runs

    # --- Derived properties ---
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
```

---

## 4. Component Implementations

### 4.1 RMSNorm

```python
# model/norm.py
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight
```

### 4.2 SwiGLU FFN

```python
# model/ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CARTConfig

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network.
    output = down(silu(gate(x)) * up(x))
    No bias terms anywhere.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        d = config.d_model
        h = config.ffn_intermediate
        self.gate = nn.Linear(d, h, bias=False)
        self.up   = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

### 4.3 Rotary Positional Embedding

```python
# model/rope.py
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
```

### 4.4 MLA Attention — Two Variants

The architecture uses MLA attention in two modes:

- **Self-attention** (prelude and coda): Q, K, V all derived from the same input.
  RoPE applied to Q and K. Standard causal mask.
- **Cross-attention** (core block): Q derived from hidden state `h_t`.
  K and V derived from prelude output `e` — computed **once before the loop**
  and passed in as constants. RoPE is **not re-applied** to K, V inside the loop
  since they were computed with RoPE in their originating prelude layer.

```python
# model/attention.py
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
```

### 4.5 Hyper-connections

```python
# model/hyper.py
import torch
import torch.nn as nn
from .config import CARTConfig

class HyperConnection(nn.Module):
    """
    Hyper-connection mechanism at loop boundaries.
    Maintains a ring buffer of the last n_hyper hidden states.
    Combines them with learned scalar weights initialized to residual baseline:
        weights = [1.0, 0.0, 0.0, ...]  (w0 = most recent, w_{n-1} = oldest)

    At loop iteration r:
        - Buffer holds states [h_{r-1}, h_{r-2}, ..., h_{r-n}]
        - Zero-padded for the first n-1 iterations
        - h_input = sum(w_i * buffer[i])

    With residual initialization, this behaves identically to standard
    residual connections until the training data supports learning non-zero
    weights for older states. The learned weight vector is a paper result.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        n = config.n_hyper
        # Residual initialization: weight all on most recent state
        init = torch.zeros(n)
        init[0] = 1.0
        self.weights = nn.Parameter(init)
        self.n_hyper = n

    def init_buffer(self, h: torch.Tensor) -> list:
        """
        Initialize the ring buffer before the loop starts.
        All slots set to h (the initial hidden state from prelude output).
        With residual init weights [1,0,0] this is safe regardless of
        what fills the non-primary slots.
        """
        return [h.clone() for _ in range(self.n_hyper)]

    def combine(self, buffer: list) -> torch.Tensor:
        """
        buffer[0] = h_{r-1} (most recent)
        buffer[1] = h_{r-2}
        buffer[2] = h_{r-3}
        Returns the weighted combination as h_input for this iteration.
        """
        w = torch.softmax(self.weights, dim=0)  # Normalize weights
        result = sum(w[i] * buffer[i] for i in range(self.n_hyper))
        return result

    def update_buffer(self, buffer: list, h_new: torch.Tensor) -> list:
        """Shift buffer and insert new hidden state at position 0."""
        return [h_new.clone()] + buffer[:-1]
```

**Design note on weight normalization:** Using `softmax` on the weights rather
than raw scalars keeps the combination a convex hull of previous states, which
aids stability. This does not prevent the model from learning to heavily weight
one state — `softmax([10, 0, 0])` ≈ `[1, 0, 0]`. If you prefer unconstrained
scalar weights, remove the softmax and apply it only at analysis time to
interpret learned values.

### 4.6 LTI Injection

```python
# model/lti.py
import math
import torch
import torch.nn as nn
from .config import CARTConfig

class LTIInjection(nn.Module):
    """
    LTI-stable recurrent injection (Parcae, Prairie et al. 2026).

    Replaces the standard residual connection:
        h = h_input + transformer_out
    with:
        h = A * h_input + transformer_out

    A is a learnable diagonal matrix parameterized as sigmoid(a_param),
    which guarantees every diagonal entry is in (0, 1) and therefore
    the spectral radius rho(A) < 1 by construction. This prevents
    residual explosion at high loop counts (R=6, R=8).

    Initialization: a_param = sigmoid_inverse(lti_init_value) so that
    A starts at lti_init_value (default 0.9) — stable but close to
    standard residual behavior (A=1). The model learns to tighten or
    relax A during training.

    The learned A values are a paper result: if A settles near 0.9
    uniformly, the forgetting mechanism is inert. If A is smaller at
    high R than low R, the model has learned to discard stale loop
    states more aggressively when it has more iterations available.
    Log rho(A) periodically during training.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        v = config.lti_init_value
        # sigmoid_inverse(v) = log(v / (1 - v))
        init_val = math.log(v / (1.0 - v))
        self.a_param = nn.Parameter(torch.full((config.d_model,), init_val))

    def forward(
        self,
        h_input: torch.Tensor,        # [B, T, d_model] — from hyper-connection
        transformer_out: torch.Tensor, # [B, T, d_model] — from CoreBlock
    ) -> torch.Tensor:
        A = torch.sigmoid(self.a_param)  # [d_model], all values in (0, 1)
        return A * h_input + transformer_out

    def spectral_radius(self) -> float:
        """Max diagonal value of A. Log this periodically during training."""
        with torch.no_grad():
            return torch.sigmoid(self.a_param).max().item()
```

### 4.7 Loop Index Embedding (LIE)

```python
# model/lie.py
import math
import torch
import torch.nn as nn
from .config import CARTConfig

class LoopIndexEmbedding(nn.Module):
    """
    Sinusoidal loop-index embedding (LIE).

    Injects a signal encoding the current loop iteration r into h_input
    before each CoreBlock pass. This gives the shared-weight block
    positional awareness in the loop dimension — it can learn that
    iteration 1 and iteration 6 are different computational contexts
    warranting different behavior.

    Without LIE, the CoreBlock processes every iteration identically
    except for the changing value of h_t. With LIE, the block can
    learn a genuine computational schedule across loop depth (e.g.,
    coarse pattern matching in early loops, fine-grained refinement
    in late loops).

    Implementation:
        - Pre-compute sinusoidal encodings for loop indices 0..max_loops-1
        - At each loop r, look up pe[r] and project to d_model
        - Add the projected signal to h_input (broadcast across [B, T])

    lie_dim = 32 (fixed, not swept). Parameter cost: 32 * d_model.
    At d=768 that is 24,576 parameters — negligible.

    Note on sweep comparability: LIE embeddings are indexed 0 to R-1.
    A model trained with R=4 never sees indices 4-7. This is expected
    and does not affect within-sweep comparisons (each config trains
    from scratch). Do not compare LIE weight interpretations across
    configs with different R values.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        self.lie_dim = config.lie_dim
        self.proj = nn.Linear(config.lie_dim, config.d_model, bias=False)
        # Pre-compute sinusoidal table for up to 16 loop indices
        # (well beyond current sweep max of R=8)
        pe = self._build_sinusoidal(max_loops=16, lie_dim=config.lie_dim)
        self.register_buffer("pe", pe)  # [16, lie_dim], not a parameter

    def _build_sinusoidal(self, max_loops: int, lie_dim: int) -> torch.Tensor:
        pe = torch.zeros(max_loops, lie_dim)
        pos = torch.arange(max_loops).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, lie_dim, 2).float() * -(math.log(10000.0) / lie_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, h: torch.Tensor, r: int) -> torch.Tensor:
        # h: [B, T, d_model]
        # r: current loop index, 0-based
        signal = self.proj(self.pe[r])  # [d_model]
        return h + signal               # broadcast: [B, T, d_model] + [d_model]
```

### 4.8 Prelude Layer

```python
# model/layers.py  (partial — PreludeLayer)
import torch
import torch.nn as nn
from .config import CARTConfig
from .norm import RMSNorm
from .attention import MLASelfAttention
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
```

### 4.9 Core Block

```python
# model/layers.py  (partial — CoreBlock)
import torch
import torch.nn as nn
from .config import CARTConfig
from .norm import RMSNorm
from .attention import MLACrossAttention
from .ffn import SwiGLUFFN

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
```

### 4.10 Coda Layer

```python
# model/layers.py  (partial — CodaLayer)
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
```

---

## 5. Full Model

```python
# model/cart.py
import torch
import torch.nn as nn
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
        # Note: do NOT register a separate nn.Linear here.
        # Use self.embedding.weight.T directly in forward().

        self._init_weights()

    def _init_weights(self):
        """
        Standard small-model initialization.
        Embedding: normal(0, 0.02)
        Linear weights: normal(0, 0.02)
        Linear biases: zero (none in this model)
        RMSNorm weights: ones (already default)
        Hyper-connection weights: [1, 0, 0] (set in HyperConnection.__init__)
        LTI a_param: sigmoid_inverse(0.9) (set in LTIInjection.__init__)
        LIE proj: normal(0, 0.02) (handled by general Linear init below)
        """
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
            h_input = self.hyper.combine(buffer)       # blend previous states
            h_input = self.lie(h_input, r)             # inject loop-depth signal
            transformer_out = self.core(h_input, K, V) # shared-weight transform
            h = self.lti(h_input, transformer_out)     # LTI-stable update
            buffer = self.hyper.update_buffer(buffer, h)

        # 6. Coda
        h = self.coda(h)

        # 7. Final norm and output projection (tied embeddings)
        h = self.final_norm(h)
        logits = h @ self.embedding.weight.T  # [B, T, vocab_size]

        if targets is None:
            return logits, None

        # Compute cross-entropy loss, shifting for next-token prediction
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
            targets[:, 1:].contiguous().view(-1),
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
        # Correct for KV proj already being outside the loop
        # (KV proj weights are NOT inside core, so already counted once — correct)
        return {
            "total": total,
            "effective": effective,
            "core_block": core_params,
            "kv_proj": kv_proj_params,
            "leverage": effective / total,
        }
```

---

## 6. Data Pipeline

### 6.1 Pre-tokenization

Run once before any sweep. Produces fixed binary files consumed by all training runs.
All four source datasets are already downloaded locally from the OpenHobbs project.

```python
# data/tokenize.py
"""
Run this script ONCE before starting any sweep.
Produces fixed .bin files in the data/ directory.
All sweep runs consume these exact files in the same order.
Source datasets are already downloaded locally — do NOT re-download.

Usage:
    python data/tokenize.py --output-dir data/

Output:
    data/tinystories_train.bin   — Stage 1 training data (primary)
    data/tinystories_val.bin     — held-out eval, never used in training
    data/wikipedia_val.bin       — held-out eval
    data/fineweb_edu_val.bin     — held-out eval
    data/stage2_train.bin        — Stage 2 training blend (see mix below)

Stage 1 training source:
    roneneldan/TinyStories        — local copy from OpenHobbs

Stage 2 training blend (fixed proportions, pre-sampled before any run):
    30% roneneldan/TinyStories
    30% wikimedia/wikipedia (20231101.en)
    40% HuggingFaceFW/fineweb-edu (sample-10BT)
    All three are available locally from OpenHobbs.

Held-out eval sources (sample fixed before any training):
    TinyStories validation split  — 500k tokens
    Wikipedia held-out articles   — 500k tokens (document which articles)
    FineWeb-Edu held-out sample   — 500k tokens (document the random seed)
"""
```

**Tokenizer:** Do NOT reuse the OpenHobbs tokenizer. OpenHobbs uses a custom 50k
vocabulary; this project uses 32k. Use `transformers.AutoTokenizer` from
`microsoft/phi-2` (clean 32k BPE vocabulary). Document the exact tokenizer name
in the `sweep_meta` table before running any sweep.

Each .bin file: raw uint16 token IDs, no headers. Flat 1D array.
At read time, slice into sequences of `seq_len` tokens.

### 6.2 Dataset

```python
# data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class FixedOrderDataset(Dataset):
    """
    Reads a pre-tokenized .bin file in fixed order.
    Every run uses identical token sequences at identical positions.
    No shuffling. This is intentional for sweep comparability.

    Each item is a sequence of seq_len tokens.
    Training: use tinystories_train.bin
    Eval: use *_val.bin files
    """
    def __init__(self, bin_path: str, seq_len: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.n_seqs = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        tokens = torch.from_numpy(
            self.data[start:start + self.seq_len + 1].astype(np.int64)
        )
        return tokens[:-1], tokens[1:]  # input, target
```

---

## 7. Training Script

```python
# train/train_one.py
"""
Trains exactly one config. Reads config from results.db, writes results,
marks status complete or failed. Never loops over configs.

Usage:
    python train/train_one.py --config-id <id> --db results.db

The orchestrator calls this script and waits for it to exit.
"""
import argparse
import sqlite3
import time
import json
import hashlib
import torch
import torch.optim as optim
from bitsandbytes.optim import AdamW8bit
# ... imports

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--db", default="results.db")
    args = parser.parse_args()

    # Load config from DB
    # Mark status = 'running'
    # Build model
    # Build optimizer (AdamW8bit)
    # Build dataloader (FixedOrderDataset)
    # Training loop:
    #   - Tier 1 log to train_log every 50 steps (loss, grad_norm, lr, tokens_seen, spectral_radius)
    #   - Tier 2 eval + log to results at steps 500 and 1500 (perplexity × 3, VRAM, tps)
    #   - See Section 7.4 for exact implementation of both tiers
    # Save checkpoint to checkpoints/{config_id}/step_1500.pt
    # Mark status = 'complete'
    # On exception: mark status = 'failed', log error
    pass
```

### 7.1 Optimizer

```python
optimizer = AdamW8bit(
    model.parameters(),
    lr=3e-4,           # Fixed for all sweep runs
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
```

### 7.2 Learning Rate Schedule

Cosine decay with linear warmup. Fixed for all sweep runs.

```
warmup_steps = 100
min_lr = 3e-5          # 10% of peak lr
peak_lr = 3e-4
total_steps = 1500     # Stage 1
```

### 7.3 Training Loop Detail

```python
# Gradient checkpointing: enabled at the loop boundary level.
# Wrap the recurrent loop body with torch.utils.checkpoint.checkpoint
# to trade activation memory for recomputation. This is important
# for d=768 and d=1024 configs at larger batch sizes.

# Batch configuration — calibrated from OpenHobbs on RTX 3050:
#   OpenHobbs used batch=4, grad_accum=8 → effective batch=32
#   At seq=512, this consumed ~4-5GB VRAM with a comparable architecture.
#   Use the same as a starting point. If VRAM allows, increase batch first
#   before increasing grad_accum.
#
# Starting config (adjust per dim after pilot run):
#   batch_size = 4
#   grad_accum_steps = 8
#   effective_batch_tokens = 4 * 8 * seq_len = 16,384 at seq=512
#
# At d=256 VRAM usage will be much lower — try batch=16, grad_accum=2
# for equivalent effective batch at higher throughput.
# At d=1024 on RTX 3090 — try batch=8, grad_accum=4.
# Log peak VRAM after step 1 and adjust before committing to a full run.

# Mixed precision: torch.autocast(device_type='cuda', dtype=torch.bfloat16)
# Gradient clipping: max_norm=1.0
# Log spectral_radius every 100 steps (model.lti.spectral_radius())

# At each eval step, compute:
#   - eval perplexity on tinystories_val.bin
#   - eval perplexity on wikipedia_val.bin
#   - eval perplexity on fineweb_edu_val.bin
#   - peak VRAM (torch.cuda.max_memory_allocated())
#   - tokens/sec (rolling average over last 100 training steps)
#   - lti_spectral_radius (model.lti.spectral_radius())
```

### 7.4 Metrics Logging Protocol

This section fully specifies what is logged, when, and how. Claude Code must
implement this exactly — it is the difference between having paper-quality loss
curves and having only endpoint numbers.

**Two logging tiers:**

Tier 1 — lightweight training log, every 50 steps:
- Writes to the `train_log` table (see schema)
- Records: step, train_loss, grad_norm, lr, n_tokens_seen, wall_sec
- Must be cheap: no model.eval(), no data loading, no VRAM measurement
- This is the source of the loss curve figures in the paper

Tier 2 — full eval snapshot, at steps 500 and 1500 (Stage 1):
- Writes to the `results` table (see schema)
- Records all metrics: three eval perplexities, peak VRAM, tokens/sec, spectral radius
- Requires: model.eval(), three DataLoader passes over val sets, torch.cuda.reset_peak_memory_stats()
- This is the source of the ranking and comparison tables in the paper

**Rolling tokens/sec implementation:**

```python
# Maintain a deque of (timestamp, n_tokens) tuples for the last 100 steps
from collections import deque
import time

step_times = deque(maxlen=100)  # (wall_time, cumulative_tokens)

# At each step:
t = time.perf_counter()
n_tokens = batch_size * seq_len  # tokens in this micro-batch
step_times.append((t, n_tokens))

# To compute rolling tokens/sec (call at Tier 1 log time):
def rolling_tps(step_times):
    if len(step_times) < 2:
        return 0.0
    elapsed = step_times[-1][0] - step_times[0][0]
    tokens = sum(t[1] for t in step_times)
    return tokens / elapsed if elapsed > 0 else 0.0
```

**Peak VRAM measurement:**

```python
# Reset the peak tracker at the START of each Tier 2 eval interval
# (not at the start of training — you want the peak during that interval)
torch.cuda.reset_peak_memory_stats()

# ... training steps run ...

# At Tier 2 eval time, read the peak:
peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
```

**LTI spectral radius logging:**

```python
# At every Tier 1 log step (every 50 steps):
spectral_radius = model.lti.spectral_radius()  # float in (0, 1)
# Write to train_log table alongside train_loss
# This catches instability early — if it climbs above 0.99, flag immediately
```

**Tier 2 eval perplexity computation:**

```python
def eval_perplexity(model, val_bin_path, seq_len, device, max_batches=50):
    """
    Compute perplexity on a held-out .bin file.
    Caps at max_batches for speed during sweep — full eval is post-sweep.
    Returns perplexity as a float.
    """
    model.eval()
    dataset = FixedOrderDataset(val_bin_path, seq_len)
    total_loss = 0.0
    n_batches = min(max_batches, len(dataset))
    with torch.no_grad():
        for i in range(n_batches):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return math.exp(total_loss / n_batches)
```

**DB write pattern (both tiers):**

Use a context manager to guarantee writes complete even if training is
interrupted immediately after. Never buffer writes — write immediately after
each log event.

```python
def write_train_log(conn, config_id, step, loss, grad_norm, lr,
                    n_tokens_seen, wall_sec, spectral_radius):
    conn.execute("""
        INSERT INTO train_log
        (config_id, step, train_loss, grad_norm, lr,
         n_tokens_seen, lti_spectral_radius, wall_sec, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (config_id, step, loss, grad_norm, lr,
          n_tokens_seen, spectral_radius, wall_sec))
    conn.commit()

def write_results(conn, config_id, step, eval_ppl_tiny, eval_ppl_wiki,
                  eval_ppl_edu, peak_vram_gb, tokens_per_sec,
                  spectral_radius, wall_sec):
    conn.execute("""
        INSERT INTO results
        (config_id, step, eval_ppl_tiny, eval_ppl_wiki, eval_ppl_edu,
         peak_vram_gb, tokens_per_sec, lti_spectral_radius,
         wall_sec, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (config_id, step, eval_ppl_tiny, eval_ppl_wiki, eval_ppl_edu,
          peak_vram_gb, tokens_per_sec, spectral_radius, wall_sec))
    conn.commit()
```

**Training loop skeleton showing both tiers:**

```python
train_start = time.perf_counter()
step_times = deque(maxlen=100)
n_tokens_seen = 0
torch.cuda.reset_peak_memory_stats()

for step in range(1, total_steps + 1):
    # --- forward / backward / optimizer step ---
    loss, grad_norm = train_step(model, optimizer, batch, scaler)
    n_tokens_seen += batch_size * seq_len
    step_times.append((time.perf_counter(), batch_size * seq_len))

    # --- Tier 1: lightweight log every 50 steps ---
    if step % 50 == 0:
        wall_sec = time.perf_counter() - train_start
        write_train_log(conn, config_id, step, loss, grad_norm,
                        get_lr(optimizer), n_tokens_seen, wall_sec,
                        model.lti.spectral_radius())
        # Safety check: flag instability before it becomes divergence
        if model.lti.spectral_radius() > 0.99:
            print(f"WARNING: spectral radius {model.lti.spectral_radius():.4f} at step {step}")

    # --- Tier 2: full eval at checkpoints ---
    if step in (500, 1500):
        ppl_tiny = eval_perplexity(model, 'data/tinystories_val.bin', seq_len, device)
        ppl_wiki = eval_perplexity(model, 'data/wikipedia_val.bin', seq_len, device)
        ppl_edu  = eval_perplexity(model, 'data/fineweb_edu_val.bin', seq_len, device)
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        tps = rolling_tps(step_times)
        wall_sec = time.perf_counter() - train_start
        write_results(conn, config_id, step, ppl_tiny, ppl_wiki, ppl_edu,
                      peak_vram, tps, model.lti.spectral_radius(), wall_sec)
        torch.cuda.reset_peak_memory_stats()  # reset for next interval

# Save checkpoint after final step
save_checkpoint(model, optimizer, step, config_id)
```

---



```sql
-- sweep/schema.sql

CREATE TABLE IF NOT EXISTS sweep_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Populated by generate_configs.py with all fixed hyperparameters
-- Keys include: vocab_size, d_head, mla_compression_ratio,
--               n_hyper, rope_base, ffn_mult,
--               seq_len_stage1, seq_len_stage2, steps_stage1, steps_stage2,
--               tokenizer_name, data_files (json), peak_lr, warmup_steps,
--               weight_decay, grad_clip

CREATE TABLE IF NOT EXISTS configs (
    config_id    TEXT PRIMARY KEY,
    d_model      INTEGER NOT NULL,
    n_loops      INTEGER NOT NULL,
    n_prelude    INTEGER NOT NULL,
    seed         INTEGER NOT NULL DEFAULT 42,
    stage        INTEGER NOT NULL DEFAULT 1,
    status       TEXT NOT NULL DEFAULT 'pending',
    hardware     TEXT NOT NULL,    -- '3050' or '3090'
    retry_count  INTEGER NOT NULL DEFAULT 0,
    error_msg    TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS results (
    result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id      TEXT NOT NULL REFERENCES configs(config_id),
    step           INTEGER NOT NULL,
    train_loss     REAL,
    eval_ppl_tiny  REAL,
    eval_ppl_wiki  REAL,
    eval_ppl_edu   REAL,
    peak_vram_gb   REAL,
    tokens_per_sec REAL,
    lti_spectral_radius REAL,     — model.lti.spectral_radius() at this step
    wall_sec       REAL,
    recorded_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS train_log (
    log_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id      TEXT NOT NULL REFERENCES configs(config_id),
    step           INTEGER NOT NULL,
    train_loss     REAL NOT NULL,   — raw loss at this step (not smoothed)
    grad_norm      REAL,            — gradient norm after clipping
    lr             REAL,            — current learning rate
    n_tokens_seen  INTEGER,         — cumulative tokens processed so far
    lti_spectral_radius REAL,       — logged every 50 steps for early instability detection
    wall_sec       REAL,            — elapsed wall clock since run start
    recorded_at    TEXT NOT NULL
);
-- train_log is the source of loss curve figures.
-- Written every 50 steps. Lightweight — no eval, no VRAM measurement.
-- results table is written only at eval checkpoints (steps 500, 1500, etc.)

CREATE INDEX IF NOT EXISTS idx_results_config   ON results(config_id, step);
CREATE INDEX IF NOT EXISTS idx_train_log_config ON train_log(config_id, step);
CREATE INDEX IF NOT EXISTS idx_configs_status   ON configs(status, stage, hardware);
```

---

## 9. Sweep Infrastructure

### 9.1 Config Generation

```python
# sweep/generate_configs.py
"""
Populates the configs table with all Stage 1 sweep runs.
Run once after schema creation.

Sweep space:
    d_model:  [256, 512, 768]          — RTX 3050
    d_model:  [1024]                   — RTX 3090
    n_loops:  [2, 4, 6, 8]
    n_prelude:[2, 3, 4, 6]
    seed:     [42]                     — Stage 1: single seed
    stage:    1

config_id = sha256(d_model, n_loops, n_prelude, seed, stage)[:16]

Ordering: sort by d_model ASC, then n_loops ASC, then n_prelude ASC.
Small configs run first — get early signal, validate pipeline.
"""

DIMS_3050 = [256, 512, 768]
DIMS_3090 = [1024]
LOOPS = [2, 4, 6, 8]
PRELUDES = [2, 3, 4, 6]
STAGE1_SEED = 42
```

### 9.2 Orchestrator

```python
# sweep/orchestrate.py
"""
Usage:
    python sweep/orchestrate.py --stage 1 --hardware 3050
    python sweep/orchestrate.py --stage 1 --hardware 3090

Behavior:
    1. Query pending configs matching stage and hardware
    2. For each: call train_one.py as subprocess, wait for exit
    3. On non-zero exit: increment retry_count, re-queue if < 3, else fail
    4. After each run: print ETA based on median wall_sec of completed runs
    5. On completion of all pending: print summary stats

Max retries per config: 3
Retry delay: 30 seconds
"""
```

### 9.3 Analyzer

```python
# sweep/analyze.py
"""
Stage 1 → Stage 2 zoom-and-confirm protocol.

For each dim size independently:
    1. Rank all complete Stage 1 configs by eval_ppl_tiny at step 1500
    2. Identify best n_loops and best n_prelude independently
    3. Apply boundary rule: if best is at edge of range, extend outward
       (e.g., if best n_loops = 8, add n_loops = 10)
    4. Propose Stage 2 configs: 3 values around each best
       (best-1_step, best, best+1_step using the original sweep spacing)
    5. Stage 2 uses 3 seeds per config

Outputs:
    - Printed ranking table per dim size
    - Stage 2 configs inserted into DB as pending
    - stage2_configs.json written for reference
"""
```

---

## 10. Evaluation

```python
# eval/perplexity.py
"""
Compute perplexity on a held-out .bin file.
Called by train_one.py at eval steps.
Also callable standalone for post-training evaluation.

Usage:
    python eval/perplexity.py --checkpoint checkpoints/{id}/step_1500.pt
                              --data data/wikipedia_val.bin
                              --seq-len 512
"""
```

For lm-evaluation-harness (post-Stage 2 only, not during sweep):
```
lm_eval --model hf \
        --model_args pretrained=checkpoints/{config_id}/step_final \
        --tasks hellaswag,winogrande,arc_easy,arc_challenge,piqa,boolq,lambada_openai \
        --device cuda \
        --batch_size 8
```

Note: LAMBADA is retained in the eval suite. Although we removed the SSM component,
the MLA cross-attention with KV reuse and iterative loop refinement should still
provide meaningful long-range context capability relative to vanilla transformers.
LAMBADA will tell us whether that capability survives without the SSM.

Implement a thin wrapper that loads Model_Paper_1 checkpoints in the format
lm-eval expects. This is post-sweep work — do not implement during initial
sweep infrastructure build.

---

## 11. Implementation Order

Build in this sequence. Each phase is independently testable before proceeding.

**Phase 1 — Core model (verify forward pass):**
1. `model/config.py`
2. `model/norm.py`
3. `model/ffn.py`
4. `model/rope.py`
5. `model/attention.py` (MLASelfAttention first, then MLACrossAttention + MLAKVProjection)
6. `model/hyper.py`
7. `model/lti.py`
8. `model/lie.py`
9. `model/layers.py`
10. `model/cart.py`

Verification: instantiate CARTConfig(d_model=256, n_loops=2, n_prelude=2),
build CART, run a forward pass with random tokens, confirm output shape and
loss value is a reasonable positive number. Print `count_parameters()`.
Also verify: `model.lti.spectral_radius()` returns approximately 0.9.
Also verify: running the loop with r=0 and r=1 produces different h_input
values even when h is identical (confirms LIE is injecting a distinct signal).

**Phase 2 — Data pipeline:**
1. `data/tokenize.py`
2. `data/dataset.py`

Verification: tokenize a small sample, load with FixedOrderDataset,
confirm token shapes and values.

**Phase 3 — Training:**
1. `train/lr_schedule.py`
2. `train/train_one.py`

Verification: run 50 steps on d=256, n_loops=2, n_prelude=2.
Confirm loss decreases. Confirm DB result rows are written. Confirm checkpoint saved.

**Before running the full sweep — pilot run:**
Run exactly one config: `d_model=256, n_loops=6, n_prelude=4`, 1500 steps.
Record actual sec/step from the training log and compare to the estimate of ~3.1 sec/step.
This calibrates all other wall-clock estimates. If actual time differs by more than
50% from estimate, flag before proceeding — the batch config may need adjustment.**
1. `sweep/schema.sql`
2. `sweep/generate_configs.py`
3. `sweep/orchestrate.py`

Verification: generate configs, confirm count = 64, confirm ordering is correct.
Run orchestrate.py for 2 configs end-to-end.

**Phase 5 — Analysis and plotting:**
1. `eval/perplexity.py`
2. `sweep/analyze.py`
3. `plot/plot_sweep.py`

---

## 12. Known Risks and Mitigations

**Windows DataLoader workers.** On Windows, `num_workers > 0` in DataLoader
requires all training code to be inside `if __name__ == '__main__':` guards,
or it will spawn infinite worker processes and hang. Use `num_workers=0`
throughout. If throughput is bottlenecked on data loading (unlikely given
pre-tokenized binary files), revisit.

**SQLite on Windows.** Enable WAL journal mode immediately on DB open:
`conn.execute("PRAGMA journal_mode=WAL")`. Without this, concurrent reads
during training (e.g., orchestrator checking status while train_one.py writes)
can produce locking errors on Windows.

**LTI A matrix initialization.** If `lti_init_value` is set too low (e.g., 0.5),
early training will strongly discount previous hidden states and the model may
converge slowly. If set too high (e.g., 0.99), the LTI constraint barely differs
from A=I and instability risk remains. The default 0.9 is the recommended starting
point from the Parcae paper. Log `model.lti.spectral_radius()` every 100 steps
for the first 500 steps of each sweep run. If it climbs above 0.99, the A
constraint is not binding and the model may be drifting toward instability.

**LIE sinusoidal dimension mismatch.** The `_build_sinusoidal` method uses
`pe[:, 1::2] = torch.cos(pos * div)` where `div` has length `lie_dim // 2`.
If `lie_dim` is odd, the sin and cos slices will have different lengths — ensure
`lie_dim` is always even. The default of 32 is safe.

**Hyper-connection buffer memory.** At d=768, batch=16, seq=512, storing 3
hidden states costs ~28MB. At d=1024, batch=8, seq=1024, it costs ~48MB.
Both are acceptable. If VRAM is tighter than expected, reduce buffer copies
to n_hyper=2 as a fallback, but flag this change — it is an architectural
decision that affects the sweep.

**Loss NaN at initialization.** If loss goes NaN in the first 10 steps,
the most likely cause is the hyper-connection combine step producing
poorly-conditioned input to the core block. Check that softmax normalization
is applied to the weights and that buffer initialization uses h (not zeros).

**KV reuse correctness.** After implementing MLACrossAttention, write a unit
test that verifies: (1) K and V do not change across loop iterations,
(2) the output of the full forward pass is identical whether K, V are
recomputed each loop or pre-computed once. This is a correctness invariant,
not just an optimization.

**Gradient flow through hyper-connections.** The buffer stores `.clone()`
of the hidden state, which preserves the computation graph. Verify that
`loss.backward()` completes without error and that `hyper.weights.grad`
is non-None after the first backward pass.

---

## 13. What Not To Change

The following decisions are final. Do not modify without explicit instruction:

- Attention type in core: cross-attention (not self-attention over [h_t; e])
- KV reuse: K and V computed once from prelude output, shared across all R loops
- Hyper-connection weights: scalar, n=3, residual initialization [1, 0, 0]
- LTI: always on, sigmoid parameterization, init value 0.9, log spectral radius during training
- LIE: always on, lie_dim=32, sinusoidal encoding, applied before each CoreBlock call
- No SSM of any kind — Mamba, Mamba2, S4, or other state space models
- No MoE anywhere in the model
- Coda layers: exactly 1, not swept
- Tied embeddings: always on
- Vocabulary size: 32,000
- d_head: 64 (n_heads derived automatically)
- RoPE: applied in prelude/coda self-attention only, not re-applied per loop
- No bias terms in any Linear layer
- No dropout
- Normalization: RMSNorm, pre-norm placement, no LayerNorm
