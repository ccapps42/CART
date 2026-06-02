# CART: Context-Anchored Recurrent Transformer

**Chad Capps** · [github.com/ccapps42](https://github.com/ccapps42)

> *A parameter-efficient language model architecture in which a single shared-weight block is looped R times, anchored at each iteration to a fixed contextual representation built once by a dedicated prelude.*

---

## Headline finding

CART is an architectural study with an **honest negative result**. At the binding parameter-parity test (d=1024, ~1B tokens), CART does **not** beat a parameter-matched Dense transformer:

- vs. **Dense 7L** (matched on *stored* parameters): CART loses by **1–2%** on every perplexity metric.
- vs. **Dense 12L** (matched on *effective* parameters): CART loses by **~10%**.

The predicted "shared-weight leverage" (effective parameters scaling with R at fixed stored parameters) does **not** convert into language-modeling quality at this scale and training budget. Diagnostic ablations split the ~10% effective-parameter gap into ~5% from weight sharing and a residual ~5% from the heterogeneous prelude→anchor→core→coda framing itself; the recurrent-core machinery (HyperConnection, LTI gate, Loop-Index Embedding) turns out to be **individually vestigial**.

Four findings stand on their own, independent of the parameter-efficiency claim:

- **Learned stability timescale.** The LTI gate's spectral radius settles in a narrow band, ϱ ∈ [0.79, 0.83] across all 36 fully-trained configs, rising with d and weakly with R. (Stage 1's "universal ϱ ≈ 0.893" was a mid-training transient.)
- Prelude depth dominates loop count: **P=6 > P=4 > P=3 > P=2** holds at every scale and every R.
- **The Stage-1 → Stage-2 reversal.** A short-budget screen predicted R-benefit grows with width; at full training it reverses, and R=6 is best at every d ≥ 512.
- Running more loops at inference than at training monotonically degrades downstream accuracy, so there is no test-time depth scaling to be had under this recipe.

Full analysis is in the paper ([arXiv:2606.01495](https://arxiv.org/abs/2606.01495)). This README summarizes the architecture and the released artifacts.

---

## Overview

CART is a recurrent-depth ("looped") transformer built for systematic study on consumer GPU hardware. It separates three responsibilities:

- **Prelude:** P unique-weight layers that build a contextual representation of the input *once*.
- **Core:** one shared-weight block looped R times, refining a hidden state that cross-attends to a frozen prelude-derived K/V anchor.
- **Coda:** one unique-weight layer that produces the final output representation.

What sets CART apart from prior looped transformers is **how the prelude enters the loop**. Rather than re-injecting the prelude embedding into a self-attending core that recomputes K/V each iteration (as in Geiping, Hyperloop, Parcae, OpenMythos, and SpiralFormer), CART computes K/V **once** from the prelude and has the loop body **cross-attend** to those frozen tensors. The architectural integration (frozen-KV cross-attending core, learned sigmoid LTI stability, and the causal-cross-attention requirement) is original; its empirical outcome is the negative result above.

The architecture keeps a compact stored parameter count while effective computational depth scales with R, which is what lets the depth/parameter sweep below run on hardware as small as an 8 GB GPU.

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

**Critical invariant:** the core's cross-attention must be **causal**. Prelude and core share the same token sequence, so non-causal cross-attention lets h[t] attend to e[t+1] and leaks the prediction target into the recurrent state (observed empirically as loss collapse with P≥3). See the paper, §3.4.

### Key Components

| Component | Description | Prior work it draws on |
|---|---|---|
| **MLA cross-attention** | Core Q from h_input, K/V from prelude output e, computed once and shared across all R loops | DeepSeek-V2 (MLA) |
| **LTI gate** | Spectral radius < 1 via a learned sigmoid gate `h = σ(A)·h_input + f(h)`; stability is *learned* from data, not structurally imposed | Parcae (stability goal; CART's mechanism differs) |
| **LIE** | Sinusoidal loop-index signal added before each core pass, so the shared block can condition on iteration index | RoPE; OpenMythos |
| **Hyper-connections** | Learned softmax-weighted blend of the last n=3 loop states; ring buffer updated after the gate | Zhang et al. 2024 (Hyper-Connections) |
| **Prelude/Core/Coda** | Three-zone separation of contextualization, iterative refinement, and output | Geiping et al. 2025 (terminology) |

### Fixed Hyperparameters

| Parameter | Value |
|---|---|
| Vocabulary size | 32,000 |
| Head dimension | 64 (n_heads = d_model / 64) |
| KV latent dimension | d_model / 4 |
| FFN width | SwiGLU, 8/3 × d_model |
| Coda layers | 1 (fixed) |
| Hyper-connection states | n = 3 |
| LIE dimension | 32 |
| Normalization | RMSNorm, pre-norm |
| Positional encoding | RoPE (prelude and coda only) |

---

## Experimental Design

The study runs in two stages.

**Stage 1: hyperparameter screen (RTX 3050, 8 GB).** All 64 configs across d_model ∈ {256, 512, 768, 1024}, R ∈ {2, 4, 6, 8}, P ∈ {2, 3, 4, 6}, single seed, 3,000 steps (~49M tokens, seq_len 512). A relative ranking, not a convergence measurement.

**Stage 2: full training (RTX 3090, 24 GB).** P=6 only, R ∈ {6, 8, 10}, all four d_model, **three seeds each = 36 configs**, trained for **30,500 steps (~1B tokens, seq_len 1024)**. Dense baselines are trained at d=1024 in two parameter-matched variants: a **7-layer** model (matched on stored params, ~75M) and a **12-layer** model (matched on CART's effective params at R=6, ~125M), three seeds each.

The central question: **does shared-weight recurrent depth extract more quality per stored parameter than a parameter-matched Dense transformer?** At d=1024 the answer is no, as the results below show.

---

## Results

### Stage 2 perplexity (3-seed means, best R per scale)

| Config | Stored | Effective | ppl_tiny | ppl_wiki | ppl_edu |
|---|---|---|---|---|---|
| CART d=256, R=8 | 14.4M | 19.7M | 4.343 | 40.31 | 40.77 |
| CART d=512, R=6 | 41.1M | 55.5M | 3.360 | 27.10 | 27.44 |
| CART d=768, R=6 | 75.3M | 104.8M | 3.010 | 22.85 | 23.15 |
| CART d=1024, R=6 | 125.1M | 178.8M | 2.798 | 20.35 | 20.67 |
| **Dense 7L** (d=1024, stored-matched) | ~75M | — | **2.746** | **20.13** | **20.26** |
| **Dense 12L** (d=1024, effective-matched) | ~125M | — | **2.592** | **18.32** | **18.50** |

At d=1024, both Dense baselines beat CART. Per-seed standard deviations are tight enough that the gap is signal rather than noise; full per-seed values are in `results.db`.

### Stage 1 → Stage 2 R-benefit reversal

At the 3,000-step screen, R-benefit appeared to grow with width (ppl_wiki, R=2→R=8 at P=6): from −0.25% at d=256 up to **+5.24% at d=1024**. At full training (30,500 steps) this reverses: **R=6 is the best of {6, 8, 10} at every d ≥ 512**, and R=10 at d=1024 *regresses* by ~0.7%. The practical lesson is that short screens are reliable for macro choices such as width and prelude depth, but not for within-scale loop count.

### Spectral radius

The LTI gate's spectral radius settles in **ϱ ∈ [0.79, 0.83]** (3-seed means at step 30,500), rising monotonically with d (+0.033 from d=256 to d=1024 at R=6) and weakly with R (+0.008 to +0.009 per pair of loops). The Stage 1 "universal ϱ ≈ 0.893" was a mid-training transient; ϱ keeps decaying through full training.

### Downstream benchmarks (zero-shot, best R per scale, 1 seed)

| Config | HellaSwag | ARC-C | LAMBADA acc (ppl) | PIQA |
|---|---|---|---|---|
| CART d=256, R=8 | 26.49 | 21.59 | 8.66 (2079) | 54.79 |
| CART d=512, R=6 | 27.10 | 22.87 | 16.15 (507) | 55.60 |
| CART d=768, R=6 | 27.05 | 22.87 | 20.42 (251) | 57.67 |
| CART d=1024, R=6 | 28.04 | 21.84 | 23.02 (164) | 58.11 |

LAMBADA and PIQA scale smoothly with width; HellaSwag and ARC-C remain near the capability threshold at this parameter and token budget. Running the trained d=1024 R=6 model at inference R ∈ {2…16} peaks at the trained R and degrades in both directions, so post-hoc test-time compute scaling does not work under this recipe.

### Diagnostic ablations (d=1024, R=6, P=6, seed 42)

Seven single-seed ablations break the d=1024 gap to Dense into parts:

- **Frozen K/V is exonerated:** recomputing K/V from h each iteration ties baseline within seed noise.
- Recurrence past the first loop adds about **1%**: R=1 raises ppl_wiki by ~1.1% versus R=6.
- **Shared weights cost ~5%:** unrolling the core into R unique blocks improves ppl_wiki by ~5.6%, but only by spending the extra stored parameters.
- Swapping cross-attention for self-attention in the unshared variant changes nothing, so the cross-attention structure is not the residual cap.
- **HyperConnection, the LTI gate, and LIE are each vestigial:** removing any one leaves ppl_wiki within seed noise.

The residual ~5% gap to Dense 12L is therefore a property of the heterogeneous architectural framing itself, not of any single machinery component.

*Total Stage 2 compute: ~22 ExaFLOPs (≈0.25 PetaFLOP-days), roughly one-tenth of a single GPT-3-small training. Every number in this README and the paper is computed from the released `results.db`.*

---

## Repository Structure

```
CART/
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
    build_bins.py   — tokenizes datasets to .bin files (run once)
    dataset.py      — FixedOrderDataset
  train/
    train_one.py    — single CART config trainer
    train_dense.py  — DenseBaseline trainer
    lr_schedule.py  — cosine schedule with warmup
    run_*_ablation.py — seven diagnostic-ablation wrappers (see below)
  sweep/
    schema.sql            — SQLite schema
    generate_configs.py   — populates Stage 1 CART configs
    generate_baselines.py — populates DenseBaseline configs
    orchestrate.py        — runs a stage
    analyze.py            — Stage 1 to Stage 2 zoom-and-confirm
    status.py             — sweep progress report
  eval/
    perplexity.py          — validation perplexity
    flops_calc.py          — FLOPs accounting
    lm_eval_adapter.py     — lm-evaluation-harness adapter
    run_benchmarks.py      — zero/few-shot downstream benchmarks
    variable_r_benchmarks.py — inference-time R sweep
  plot/
    plot_sweep.py   — loss-curve and spectral-radius figures
  figures/
    gen_*.py        — architecture, ppl-vs-R, and Pareto figures
  tools/
    checkpoint_inventory.py — disk-to-results.db checkpoint join
  results.db        — every run's logs, evals, and provenance (released artifact)
```

---

## Training Data

All Stage 2 configs use a single mixed training bin (`stage2_train.bin`) interleaved in 1024-token chunks:

| Dataset | Proportion | Tokens | HuggingFace ID |
|---|---|---|---|
| TinyStories | 30% | 300M | `roneneldan/TinyStories` |
| Wikipedia | 30% | 300M | `wikimedia/wikipedia`, 20231101.en |
| FineWeb-Edu | 40% | 400M | `HuggingFaceFW/fineweb-edu`, sample-10BT |

**Total:** ~1B tokens (999,997,440), consumed in a single pass with no repetition. Stage 1 consumes ~49M tokens (3,000 steps × 16,384 tokens/step); Stage 2 consumes the full ~1B (30,500 steps × 32,768 tokens/step). Chunks are 1024 tokens so every training window is drawn from a single source domain.

**Validation sets** (held out, never seen in training):

| Val set | Source | Hold-out method |
|---|---|---|
| `tinystories_val.bin` | TinyStories validation split | Official split |
| `wikipedia_val.bin` | Wikipedia shard 40 of 41 | Last shard held out |
| `fineweb_edu_val.bin` | FineWeb-Edu shard 97 of 98 | Last shard, shuffled seed 42 |

**Tokenizer:** `NousResearch/Llama-2-7b-hf` (Llama-2 BPE, 32,000 vocab). The `.bin` files are not committed because of their size; they regenerate deterministically from the HuggingFace sources above via `data/build_bins.py`.

---

## Hardware

- **RTX 3050** (8 GB): Stage 1 sweep, all d_model. Peak Stage 1 VRAM ~1.9 GB, so every config fits without gradient checkpointing.
- **RTX 3090** (24 GB): Stage 2 long runs (30,500 steps/config), both Dense baselines, and all seven ablations.

No custom CUDA kernels. Flash Attention runs via `torch.nn.functional.scaled_dot_product_attention` (Ampere). `torch.compile` is not used (Triton is unavailable on Windows 11). No multi-node, no cloud compute.

---

## Installation

```bash
git clone https://github.com/ccapps42/CART.git
cd CART
pip install -r requirements.txt
```

Requirements: `torch>=2.1.0`, `transformers>=4.35.0`, `datasets>=2.14.0`, `bitsandbytes>=0.44.0`.

---

## Running the Sweep

```bash
# 1. Build training data (run once, ~15 min, produces ~2 GB)
python data/build_bins.py --stage2-only --stage2-out data/stage2

# 2. Generate Stage 1 CART configs and the d=1024 Dense baselines
python sweep/generate_configs.py --db results.db
python sweep/generate_baselines.py --db results.db

# 3. Run the Stage 1 screen (RTX 3050, 3,000 steps/config)
python sweep/orchestrate.py --stage 1 --hardware 3050 --db results.db

# 4. Run the Stage 2 sweep (RTX 3090, 30,500 steps/config)
python sweep/orchestrate.py --stage 2 --hardware 3090 --db results.db --ckpt-interval 5000

# 5. Check progress at any time
python sweep/status.py

# 6. Generate figures
python plot/plot_sweep.py --db results.db
python figures/gen_ppl_vs_r.py
```

---

## Diagnostic Ablations

Seven single-seed diagnostic ablations characterize each design choice at `d=1024, R=6, P=6, seed=42`. Each wrapper registers a config row with `hardware='ablation'` (the regular orchestrator skips this tag) and invokes `train/train_one.py` with the corresponding architectural flag.

| Wrapper | Model type | What it tests | Result |
|---|---|---|---|
| `train/run_unfrozen_kv_ablation.py` | `cart_unfrozenkv` | Recompute K, V from h each iteration | Frozen K/V exonerated (within seed noise) |
| `train/run_r1_ablation.py` | `cart_r1` | Set R=1 | Recurrence past first loop ~1% |
| `train/run_unshared_core_ablation.py` | `cart_unshared` | R unique CoreBlocks vs one shared block | Shared weights cost ~5% |
| `train/run_self_attn_unshared_ablation.py` | `cart_selfattn_unshared` | Self-attention in the unshared core | No change vs unshared |
| `train/run_no_hyper_ablation.py` | `cart_no_hyper` | Bypass HyperConnection | Vestigial |
| `train/run_no_lti_ablation.py` | `cart_no_lti` | Bypass the LTI gate (plain residual) | Vestigial |
| `train/run_no_lie_ablation.py` | `cart_no_lie` | Skip the Loop-Index Embedding | Vestigial |

Each wrapper supports `--print-cmd-only`. All ablation results are stored in `results.db` alongside the main sweep.

---

## Citation

Published on arXiv: **[arXiv:2606.01495](https://arxiv.org/abs/2606.01495)**.

```bibtex
@article{capps2026cart,
  title         = {CART: Context-Anchored Recurrent Transformer},
  author        = {Capps, Chad A.},
  year          = {2026},
  eprint        = {2606.01495},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.01495}
}
```

---

## License

This repository holds two distinct artifacts with separate licenses:

- **Code** (everything under `model/`, `train/`, `sweep/`, `data/`, `eval/`, `plot/`, `figures/gen_*.py`, `tools/`): released under the [MIT License](LICENSE). Free to use, modify, and redistribute with attribution.
- **Paper** ([arXiv:2606.01495](https://arxiv.org/abs/2606.01495)): released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Results database** (`results.db`): released under CC BY 4.0 alongside the paper; treat it as the empirical artifact accompanying the manuscript.

Both licenses permit commercial use with attribution.

---

## Acknowledgements

CART is an original architecture designed and developed by **Chad Capps**, and this repository reports a rigorous negative/bounding result on its parameter efficiency (see Headline finding). The following published works informed individual components:

- **DeepSeek-V2:** Multi-head Latent Attention (MLA).
- **Zhang et al. 2024:** Hyper-Connections (the learned blend of recent states).
- **Geiping et al. 2025:** the prelude/core/coda terminology for recurrent-depth models.
- **Parcae:** the spectral-radius < 1 stability goal, which CART reaches via a learned sigmoid gate rather than a structural constraint.
- **OpenMythos:** the loop-index embedding (LIE).
- **Hyperloop Transformer:** a concurrent three-zone looped architecture that recomputes K/V each iteration, the closest comparator to CART's frozen-KV core.

The architectural integration (a unique-layer prelude producing a frozen K/V anchor, a shared-weight cross-attending core, a learned sigmoid LTI gate, and the causal-cross-attention requirement) is original work and does not appear in the papers above individually or in combination. All sweep infrastructure and experimental results are original work by Chad Capps.
