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
│   hyper.combine(buffer) → h_input                   │
│     (softmax-weighted blend of last n=3 h states)   │
│   LIE: sinusoidal loop-index signal added to h_input│
│   MLA cross-attention (Q from h_input, K/V from e)  │
│   SwiGLU FFN  →  transformer_out                    │
│   LTI: h = sigmoid(A)·h_input + transformer_out     │
│   hyper.update_buffer(buffer, h)                    │
│     (push h to front of ring buffer, drop oldest)   │
└─────────────────────────────────────────────────────┘
    ↓
Coda × 1
  MLA self-attention (RoPE, causal)
  SwiGLU FFN
    ↓
RMSNorm → Output logits (tied embedding weight)
```

### Key Components

| Component | Description | Source |
|---|---|---|
| **MLA cross-attention** | Core block Q from h_input, K/V from prelude output e — computed once, shared across all R loops | DeepSeek-V2 |
| **LTI injection** | Spectral radius < 1 enforced via sigmoid-parameterized A diagonal — enables stable training at high R | Parcae |
| **LIE** | Sinusoidal loop-index signal projected to model_dim and injected before each core pass — allows shared-weight block to learn depth-specific behavior | OpenMythos |
| **Hyper-connections** | Learned softmax-weighted blend of last n=3 loop states at each boundary; ring buffer updated after LTI | Hyperloop Transformer |
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
| d_model | 256, 512, 768, 1024 (all RTX 3050) |
| Loop count R | 2, 4, 6, 8 |
| Prelude depth P | 2, 3, 4, 6 |

**64 total configurations.** Stage 1 (3,000 steps, single seed, mixed training data) screens all configs. Stage 2 (longer training, 3 seeds) confirms the neighborhood of the optimum.

The central research question: **how does the quality-per-parameter tradeoff of shared-weight recurrent depth scale as a function of model dimension and loop count?**

---

## Results

### Stage 1 — Architecture Screen Complete (all 64 configs, 3,000 steps)

Best configuration per scale on three held-out validation sets:

| Config | Stored params | Effective params | Leverage | ppl_wiki | ppl_tiny | ppl_edu | ρ |
|---|---|---|---|---|---|---|---|
| CART-256 R=6 P=6 | 14.37M | 17.97M | 1.25× | 184.93 | 12.31 | 161.28 | 0.8927 |
| CART-512 R=8 P=6 | 41.05M | 61.24M | 1.49× | 136.42 | 8.404 | 114.52 | 0.8936 |
| CART-768 R=8 P=6 | 75.34M | 116.64M | 1.55× | 114.96 | 7.063 | 95.40 | 0.8956 |
| CART-1024 R=8 P=6 | 125.09M | 200.34M | 1.60× | 97.73 | 6.037 | 82.00 | 0.8966 |

*Perplexity after 3,000 steps (~49M tokens) on mixed training data. Lower is better. Stage 1 is a hyperparameter screen — absolute values are not benchmark-predictive at this token budget.*

**Key findings from Stage 1:**
- P=6 is best at every scale and every R without exception — prelude depth dominates loop count
- The R benefit at P=6 grows with scale: d=256 gains nothing from higher R (−0.25%); d=1024 gains 5.24% (R=2→R=8)
- At d=1024, R=8 beats R=2 at every P value including P=2 — the threshold for R to be universally beneficial falls between d=768 and d=1024
- The spectral radius ρ converges to a narrow band (≈0.893) at every scale, regardless of R or P, drifting upward slowly with d_model
- Scale dominates hyperparameters: d=1024 R=2 P=2 (weakest config, ppl_wiki=113.06) beats d=768 R=8 P=6 (best config, ppl_wiki=114.96)

### Stage 2 — Benchmarks (pending)

Stage 2 will train the best configs for ~1B tokens (61,000 steps, seq_len=1024) on RTX 3090 with 3 seeds per config.

| Config | Stored params | Effective params | HellaSwag | ARC-C | LAMBADA | PIQA |
|---|---|---|---|---|---|---|
| CART-256 R=8 P=6 | 14.37M | 19.42M | — | — | — | — |
| CART-512 R=8 P=6 | 41.05M | 61.24M | — | — | — | — |
| CART-768 R=8 P=6 | 75.34M | 116.64M | — | — | — | — |
| CART-1024 R=8 P=6 | 125.09M | 200.34M | — | — | — | — |

Baseline comparisons: parameter-matched DenseBaseline (7-layer, full-rank MHA, same training data), Pythia-160M.

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
    dense.py        — DenseBaseline, DenseConfig (parameter-matched comparison)
  data/
    build_bins.py   — tokenizes datasets → .bin files (run once)
    dataset.py      — FixedOrderDataset
  train/
    train_one.py    — single CART config trainer
    train_dense.py  — DenseBaseline trainer
    lr_schedule.py  — cosine schedule with warmup
  sweep/
    schema.sql      — SQLite schema
    generate_configs.py    — populates Stage 1 CART configs
    generate_baselines.py  — populates DenseBaseline configs
    orchestrate.py
    analyze.py      — Stage 1 → Stage 2 zoom-and-confirm
  eval/
    perplexity.py
  plot/
    plot_sweep.py   — paper figures
```

---

## Training Data

All sweep configs use a single mixed training bin (`stage2_train.bin`) interleaved in 1024-token chunks:

| Dataset | Proportion | Tokens | HuggingFace ID |
|---|---|---|---|
| TinyStories | 30% | 300M | `roneneldan/TinyStories` |
| Wikipedia | 30% | 300M | `wikimedia/wikipedia`, 20231101.en |
| FineWeb-Edu | 40% | 400M | `HuggingFaceFW/fineweb-edu`, sample-10BT |

**Total training tokens:** ~1B (999,997,440). Stage 1 consumes ~49M tokens (3,000 steps × 16,384 tokens/step); Stage 2 consumes the full ~1B.

Chunks are 1024 tokens so every training window (Stage 1 seq_len=512 or Stage 2 seq_len=1024) is drawn from a single source domain — no cross-domain boundaries within a sequence.

**Validation sets** (held out, never seen during training):

| Val set | Source | Hold-out method |
|---|---|---|
| `tinystories_val.bin` | TinyStories validation split | Official split |
| `wikipedia_val.bin` | Wikipedia shard 40 of 41 | Last shard held out |
| `fineweb_edu_val.bin` | FineWeb-Edu shard 97 of 98 | Last shard, shuffled seed 42 |

**Tokenizer:** `NousResearch/Llama-2-7b-hf` (Llama-2 BPE, 32,000 vocabulary). Note: the architecture spec originally referenced `microsoft/phi-2` but phi-2 uses a ~51k vocabulary; the Llama-2 tokenizer is the correct match for vocab_size=32,000.

---

## Hardware

All sweep runs performed on consumer hardware:

- **RTX 3050** (8GB VRAM) — Stage 1 sweep, all d_model ∈ {256, 512, 768, 1024}
- **RTX 3090** (24GB VRAM) — Stage 2 long runs (61,000 steps per config)

Peak Stage 1 VRAM: 1.88 GB (d=768 R=2 P=2) — all configs fit comfortably on 8GB without gradient checkpointing.

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
# 1. Build training data (run once, ~15 min, produces ~2 GB)
python data/build_bins.py --stage2-only --stage2-out data/stage2

# 2. Generate all 64 Stage 1 configs and 4 DenseBaseline configs
python sweep/generate_configs.py --db results.db
python sweep/generate_baselines.py --db results.db

# 3. Run Stage 1 sweep (RTX 3050, ~3,000 steps per config)
python sweep/orchestrate.py --stage 1 --hardware 3050 --db results.db

# 4. Analyze Stage 1 results and generate Stage 2 candidate configs
python sweep/analyze.py --db results.db --insert-stage2

# 5. Run Stage 2 sweep (RTX 3090, ~61,000 steps per config)
python sweep/orchestrate.py --stage 2 --hardware 3090 --db results.db --ckpt-interval 5000

# 6. Generate paper figures
python plot/plot_sweep.py --db results.db
```

---

## Citation

*Paper forthcoming. Please check back for the arXiv link.*

```bibtex
@article{capps2026cart,
  title   = {CART: Context-Anchored Recurrent Transformer},
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

CART is an original architecture designed and developed by **Chad Capps**. The following published works informed individual components:

- **Parcae** — LTI stability via sigmoid-parameterized spectral radius constraint
- **Hyperloop Transformer** (MIT, 2026) — prelude/core/coda structural organization and hyper-connections at loop boundaries
- **OpenMythos** (kyegomez) — loop index embedding (LIE)
- **DeepSeek-V2** — Multi-head Latent Attention (MLA)

The architectural combination presented here — including the cross-attention formulation anchoring the recurrent loop to a fixed prelude output with KV reuse across loop iterations, the integration of LTI stability, LIE, and hyper-connections within a single shared-weight core, and the empirical characterization of the prelude depth / loop count / model dimension interaction — is the original work of Chad Capps and does not appear in any of the above papers individually or in combination. All sweep infrastructure, experimental results, and trained weights are original work by Chad Capps.
