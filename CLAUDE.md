# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project training and evaluating **CART** (Context-Anchored Recurrent Transformer), a parameter-efficient architecture that recycles a shared core block R times through a fixed context anchor. The paper compares CART against a parameter-matched DenseBaseline transformer. All experiments are tracked in `results.db` (SQLite).

## Key Commands

### Check sweep progress
```powershell
python sweep/status.py
```
Always use this when asked for a DB update or progress check — never write ad-hoc queries.

### Launch training sweep (Stage 2, RTX 3090)
```powershell
python sweep/orchestrate.py --stage 2 --hardware 3090 --ckpt-interval 5000
```

### Train a single CART config
```powershell
python train/train_one.py --config-id <id> --db results.db --seq-len 1024
```

### Train a single Dense baseline config
```powershell
python train/train_dense.py --config-id <id> --db results.db
```

### FLOPs analysis
```powershell
python eval/flops_calc.py --d 1024 --seq-len 1024
```

### Build Stage 2 training data
```powershell
python data/build_bins.py --stage2-only --stage2-out data/stage2
```

### Quick smoke test (50 steps)
```powershell
python train/train_one.py --config-id <id> --db results.db --max-steps 50
```

## Architecture

CART has three zones executed sequentially:

```
Embedding → Prelude (P unique layers) → KV projection (once) →
Recurrent Core (R iterations, shared weights) → Coda (1 layer) → Output logits
```

**Prelude** (`P` unique transformer layers): Standard MLA self-attention + SwiGLU FFN. Produces context anchor `e`.

**KV Projection** (`model/attention.py:MLAKVProjection`): Computes K, V from `e` once before the loop. K, V are reused across all R iterations — never recomputed.

**Recurrent Core** (single `CoreBlock`, run R times with shared weights):
1. HyperConnection blends last 3 hidden states (ring buffer, softmax-weighted, residual init [1,0,0])
2. Loop Index Embedding (LIE) adds sinusoidal loop-index signal to h
3. MLA cross-attention: Q from h, K/V from prelude output e
4. LTI gate: `h = sigmoid(a) ⊙ h_input + transformer_out` — guarantees spectral radius < 1

**Coda**: Single unique MLA self-attention + SwiGLU FFN layer. Output goes to tied embedding projection.

**Critical invariants:**
- Cross-attention in the core must be `is_causal=True` — prelude and core share the same token sequence; non-causal leaks future tokens (caused catastrophic collapse in d=768 P=3+ configs)
- Gradient checkpointing is disabled — causes silent gradient corruption with HyperConnection ring buffer + K/V reuse
- RoPE only in prelude/coda, not in the core loop (h is a recurrent state, not a token sequence)
- No bias terms anywhere in the model

## Database Schema

`results.db` is the source of truth for all experiments.

| Table | Purpose |
|-------|---------|
| `configs` | One row per run: `config_id` (sha256[:16]), `d_model`, `n_loops` (R), `n_prelude` (P), `seed`, `stage`, `status`, `hardware`, `model_type` |
| `results` | Eval snapshots every 500 steps: `eval_ppl_tiny/wiki/edu`, `peak_vram_gb`, `tokens_per_sec`, `lti_spectral_radius` |
| `train_log` | Lightweight log every 50 steps: `train_loss`, `grad_norm`, `lr`, `lti_spectral_radius` |
| `sweep_meta` | Fixed hyperparameters (key/value) |

**Status lifecycle:** `pending` → `running` → `complete` / `failed`

**Custom statuses:** `reference` = configs excluded from the active sweep but data preserved (e.g. P=4/P=8 configs from Stage 2 scope reduction).

If a run is killed mid-training, its status stays `running` and must be manually reset:
```powershell
python -c "import sqlite3; c=sqlite3.connect('results.db'); c.execute(\"UPDATE configs SET status='pending' WHERE status='running'\"); c.commit()"
```

## Stage 2 Sweep Configuration

**Active sweep** (36 configs): P=6 only, R ∈ {6, 8, 10}, d_model ∈ {256, 512, 768, 1024}, 3 seeds each.

**Reference configs** (72 configs): P=4 and P=8 — excluded from sweep but data preserved in DB.

**Training setup** (all Stage 2 configs):
- `SEQ_LEN=1024`, `BATCH_SIZE=8`, `GRAD_ACCUM=4` → 32,768 tokens/step effective batch
- `TOTAL_STEPS=30_500` → ~1B tokens
- Training data: `data/stage2/stage2_train.bin` (999M tokens, single pass, no repetition)
- Optimizer: AdamW8bit (bitsandbytes), `PEAK_LR=3e-4`, cosine decay to `MIN_LR=3e-5`, `WARMUP=100`

**Do not change batch_size/grad_accum between d_model scales** — creates hyperparameter inconsistency across the cross-scale comparison even if effective batch is preserved.

## Data Pipeline

Validation sets (held out permanently):
- `data/val/tinystories_val.bin` — from TinyStories shard not used in training
- `data/val/wikipedia_val.bin` — from Wikipedia shard 40
- `data/val/fineweb_edu_val.bin` — from FineWeb-Edu shard 97

Training data is loaded entirely into RAM at startup (`np.fromfile`, not memmap) — required for performance on Windows.

## LaTeX Paper

Paper: `paper/cart_paper.tex`. Macros: `\pplwiki`, `\ppltiny`, `\ppledu` embed `$...$` internally — do not wrap in extra `$...$`. See global CLAUDE.md for LaTeX pitfalls and compile sequence.

## Known Issues / Constraints

- `torch.compile` is unavailable on Windows 11 (no Triton) — do not suggest it
- `lm-evaluation-harness` benchmarks have a hang on the wikitext task — run without it
- Gradient checkpointing causes silent gradient corruption — keep disabled
- `DataLoader num_workers=0` required on Windows (no multiprocessing fork support)
