# CART: Context-Anchored Recurrent Transformer

**Chad Capps** · [github.com/ccapps42](https://github.com/ccapps42)

> *A parameter-efficient language model architecture in which a single shared-weight block is looped R times, anchored at each iteration to a fixed contextual representation built once by a dedicated prelude.*

---

## Overview

CART is a novel recurrent-depth transformer architecture designed for systematic study on consumer GPU hardware. The architecture combines MLA cross-attention with KV reuse across loop iterations, LTI stability constraints, loop index embeddings, and hyper-connected residual streams within a shared-weight recurrent core — a combination that has not previously appeared in the literature. Individual components draw on published prior work (see Acknowledgements), but the architectural integration, the cross-attention formulation anchoring the loop to a fixed prelude output, and the empirical characterization of the prelude/core/coda ratio across model scales are original contributions.

This design separates three distinct responsibilities:

- **Prelude** — P unique-weight layers that build a rich contextual representation of the input *once*
- **Core** — one shared-weight block looped R times, performing iterative refinement anchored to the prelude output
- **Coda** — one unique-weight layer that produces the final output representation

The result is a model whose effective computational depth scales with R while its stored parameter count remains compact — enabling systematic exploration of the depth/parameter tradeoff on hardware as small as an 8GB GPU.

---

## Architecture

```
Input Tokens
    ↓
Embedding (32k vocab, tied to output projection)
    ↓
Prelude × P  ─────────────────────────────────────────┐
  MLA self-attention (RoPE, causal)                    │
  SwiGLU FFN                                           │ fixed context e
    ↓                                                  │
KV Projection  ◄──────────────────────────────────────┘
  K, V computed once from e, reused across all R loops
    ↓
┌── Core Block × R (shared weights) ──────────────────┐
│   Hyper-connections (n=3, blend last 3 h states)    │
│   LIE: loop index embedding injected                │
│   MLA cross-attention (Q from h_t, K/V from e)     │
│   SwiGLU FFN                                        │
│   LTI injection: h = sigmoid(A)·h_input + block_out│
└─────────────────────────────────────────────────────┘
    ↓
Coda × 1
  MLA self-attention
  SwiGLU FFN
    ↓
RMSNorm → Output logits (tied embedding weight)
```

### Key Components

| Component | Description | Source |
|---|---|---|
| **MLA cross-attention** | Core block Q from h_t, K/V from prelude output e — computed once, shared across all R loops | DeepSeek-V2 |
| **LTI injection** | Spectral radius < 1 enforced via sigmoid-parameterized A matrix — enables stable training at high R | Parcae |
| **LIE** | Sinusoidal loop-index signal injected before each core pass — allows shared-weight block to learn depth-specific behavior | OpenMythos |
| **Hyper-connections** | Learned weighted blend of last n=3 loop states at each boundary — gradient highway for high-R configs | Hyperloop Transformer |
| **Prelude/Core/Coda** | Three-zone structural separation of contextualization, iterative refinement, and output production | Hyperloop Transformer |

### Fixed Hyperparameters

| Parameter | Value |
|---|---|
| Vocabulary size | 32,000 |
| Head dimension | 64 (n_heads = d_model / 64) |
| KV latent dimension | d_model / 4 |
| FFN width | SwiGLU, 8/3 × d_model |
| Coda layers | 1 (fixed) |
| Hyper-connection states | n = 3 |
| LTI init value | 0.9 |
| LIE dimension | 32 |
| Normalization | RMSNorm, pre-norm |
| Positional encoding | RoPE (prelude and coda only) |

---

## Sweep Design

This repository contains the full sweep across three axes:

| Variable | Values |
|---|---|
| d_model | 256, 512, 768 (RTX 3050) · 1024 (RTX 3090) |
| Loop count R | 2, 4, 6, 8 |
| Prelude depth P | 2, 3, 4, 6 |

**64 total configurations.** Stage 1 (1,500 steps, single seed) screens all configs. Stage 2 (10k–20k steps, 3 seeds) confirms the neighborhood of the optimum.

The central research question: **how does the quality-per-parameter tradeoff of shared-weight recurrent depth scale as a function of model dimension and loop count?**

---

## Results

*Sweep in progress. Results will be posted here upon completion.*

### Benchmarks (post-sweep, winning configs)

| Config | Params (total) | Params (effective) | HellaSwag | ARC-C | LAMBADA | PIQA |
|---|---|---|---|---|---|---|
| CART-256 | — | — | — | — | — | — |
| CART-512 | — | — | — | — | — | — |
| CART-768 | — | — | — | — | — | — |
| CART-1024 | — | — | — | — | — | — |

Baseline comparisons: parameter-matched vanilla transformer, Pythia-160M, Pythia-410M.

---

## Repository Structure

```
Model_Paper_1/
  model/
    config.py       — CARTConfig dataclass
    norm.py         — RMSNorm
    ffn.py          — SwiGLUFFN
    rope.py         — RotaryEmbedding
    attention.py    — MLASelfAttention, MLACrossAttention, MLAKVProjection
    hyper.py        — HyperConnection
    lti.py          — LTIInjection
    lie.py          — LoopIndexEmbedding
    layers.py       — PreludeLayer, CoreBlock, CodaLayer
    cart.py         — CART (full model)
  data/
    tokenize.py     — pre-tokenization (run once)
    dataset.py      — FixedOrderDataset
  train/
    train_one.py    — single-config trainer
    lr_schedule.py  — cosine schedule with warmup
  sweep/
    schema.sql      — SQLite schema
    generate_configs.py
    orchestrate.py
    analyze.py      — Stage 1 → Stage 2 zoom-and-confirm
  eval/
    perplexity.py
  plot/
    plot_sweep.py   — paper figures
```

---

## Training Data

| Dataset | Stage | Proportion |
|---|---|---|
| TinyStories (`roneneldan/TinyStories`) | Stage 1 (screen) | 100% |
| TinyStories | Stage 2 (confirm) | 30% |
| Wikipedia (`wikimedia/wikipedia`, 20231101.en) | Stage 2 | 30% |
| FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, sample-10BT) | Stage 2 | 40% |

Tokenizer: `microsoft/phi-2` BPE, 32k vocabulary.

---

## Hardware

All sweep runs performed on consumer hardware:

- **RTX 3050** (8GB VRAM) — d_model ∈ {256, 512, 768}
- **RTX 3090** (24GB VRAM) — d_model = 1024

No custom CUDA kernels. Flash Attention via `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.1+, Ampere architecture).

---

## Installation

```bash
git clone https://github.com/ccapps42/CART.git
cd CART
pip install -r requirements.txt
```

Requirements: `torch>=2.1.0`, `transformers>=4.35.0`, `datasets>=2.14.0`, `bitsandbytes>=0.44.0`

---

## Running the Sweep

```bash
# 1. Pre-tokenize data (run once)
python data/tokenize.py --output-dir data/

# 2. Generate all 64 sweep configs
python sweep/generate_configs.py --db results.db

# 3. Run Stage 1 sweep (RTX 3050)
python sweep/orchestrate.py --stage 1 --hardware 3050

# 4. Analyze results and propose Stage 2 configs
python sweep/analyze.py --stage 1 --output stage2_configs.json

# 5. Run Stage 2 sweep
python sweep/orchestrate.py --stage 2 --hardware 3050
```

---

## Citation

*Paper forthcoming. Please check back for the arXiv link.*

```bibtex
@article{capps2026cart,
  title   = {CART: Context-Anchored Recurrent Transformers},
  author  = {Capps, Chad},
  year    = {2026},
  url     = {https://github.com/ccapps42/CART}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

CART is an original architecture by Chad Capps. The following published works informed individual components:

- **Parcae** — LTI stability via spectral radius constraint
- **Hyperloop Transformer** (MIT, 2026) — prelude/core/coda structural organization and hyper-connections at loop boundaries
- **OpenMythos** (kyegomez) — loop index embedding (LIE)
- **DeepSeek-V2** — Multi-head Latent Attention (MLA)

The specific combination of these elements — in particular the cross-attention formulation anchoring the recurrent loop to a fixed prelude output with KV reuse, the integration of LTI stability with hyper-connections and LIE within a single shared-weight core, and the systematic empirical characterization of the prelude depth / loop count / model dimension interaction — represents original work not present in any of the above papers individually or in combination.
