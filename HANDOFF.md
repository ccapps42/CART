# CART Project — Session Handoff

> Generated 2026-05-03. Feed this file to Claude Code at the start of each new session
> to restore full context without re-reading every source file.

---

## 1. What CART Is

**Context-Anchored Recurrent Transformer** — a parameter-efficient language model architecture.

Forward pass in one sentence: embed tokens → P unique-weight prelude layers (standard causal self-attn) → fix that output as context `e` → compute K,V from `e` once → loop a single shared-weight core block R times (cross-attending to `e` each pass) → 1 coda layer → tied output projection.

Key mechanisms on top of the loop:
- **MLA** (Multi-head Latent Attention): K,V compressed through d/4 bottleneck. Used in prelude/coda as self-attn; in core as cross-attn with pre-computed K,V.
- **LTI Injection**: `h = sigmoid(a) * h_input + transformer_out`. Diagonal A initialized at 0.9 via `sigmoid_inverse(0.9)`. Guarantees spectral radius < 1. Spectral radius is logged every 50 steps as an instability early-warning.
- **HyperConnection**: Softmax-weighted blend of last 3 hidden states (ring buffer). Init weights `[1,0,0]` = pure residual baseline. Learned during training.
- **LIE** (Loop Index Embedding): Sinusoidal 32-dim encoding of loop index r, projected to d_model, added to h before each core pass.

Leverage ratio: CART has R× effective parameters at the core block for the cost of 1× stored — the paper's central claim.

---

## 2. Repository Layout

```
CART/
├── model/
│   ├── config.py          CARTConfig dataclass
│   ├── cart.py            CART model class
│   ├── layers.py          PreludeLayer, CoreBlock, CodaLayer
│   ├── attention.py       MLASelfAttention, MLACrossAttention, MLAKVProjection
│   ├── ffn.py             SwiGLUFFN
│   ├── norm.py            RMSNorm
│   ├── rope.py            RotaryEmbedding
│   ├── hyper.py           HyperConnection
│   ├── lti.py             LTIInjection
│   ├── lie.py             LoopIndexEmbedding
│   └── dense.py           DenseBaseline + DenseConfig  ← NEW
├── train/
│   ├── train_one.py       Single CART config trainer
│   ├── train_dense.py     DenseBaseline trainer         ← NEW
│   └── lr_schedule.py     Cosine LR with warmup
├── data/
│   ├── build_bins.py      Tokenizes datasets → .bin files
│   └── dataset.py         FixedOrderDataset (numpy mmap)
├── sweep/
│   ├── schema.sql         SQLite schema (source of truth)
│   ├── generate_configs.py  Populates Stage 1 CART configs
│   ├── generate_baselines.py  Populates DenseBaseline configs ← NEW
│   ├── orchestrate.py     Runs pending configs sequentially
│   └── analyze.py         Stage 1→Stage 2 zoom-and-confirm
├── eval/
│   └── perplexity.py      Standalone eval on saved checkpoint
└── plot/
    └── plot_sweep.py      Generates 6 figure types from results.db
```

---

## 3. Fixed Hyperparameters (all runs)

| Param | Value | Notes |
|---|---|---|
| Tokenizer | NousResearch/Llama-2-7b-hf | 32,000 vocab. Spec said phi-2 but phi-2 has 51,200 vocab — mismatch caught and corrected. |
| d_head | 64 | n_heads = d_model / 64 |
| MLA compression | 4× | d_kv_latent = d_model / 4 |
| FFN intermediate | ceil(8/3 × d / 256) × 256 | Rounded to 256 multiple |
| n_hyper | 3 | HyperConnection ring buffer size |
| LTI init | 0.9 | sigmoid(a_param) = 0.9 at init |
| RoPE base | 10,000 | Applied in prelude & coda only, NOT per core loop |
| seq_len | 512 | Stage 1 and Stage 2 |
| batch_size | 4 | |
| grad_accum | 8 | Effective batch = 4 × 8 × 512 = 16,384 tokens/step |
| peak_lr | 3e-4 | |
| min_lr | 3e-5 | |
| warmup_steps | 100 | |
| weight_decay | 0.1 | |
| grad_clip | 1.0 | |
| optimizer | AdamW8bit | Falls back to torch AdamW if bitsandbytes missing |
| precision | bfloat16 AMP | CUDA only |
| grad_checkpointing | DISABLED | Corrupts gradients with HyperConnection+LTI aliasing |

---

## 4. Sweep Structure

### Stage 1 — Architecture Search

**64 configs** = 4 d_model × 4 n_loops × 4 n_prelude, seed=42, **3,000 steps each**

| Axis | Values | Hardware |
|---|---|---|
| d_model | 256, 512, 768 | RTX 3050 |
| d_model | 1024 | RTX 3090 |
| n_loops (R) | 2, 4, 6, 8 | |
| n_prelude (P) | 2, 3, 4, 6 | |

Eval: perplexity on 3 val sets every 500 steps. Train log every 50 steps.
**Rank by `eval_ppl_tiny` at step 3000.**

### Stage 2 — Zoom-and-Confirm

Run `sweep/analyze.py` after Stage 1 completes. It:
1. Finds best R and P per d_model
2. Proposes ±1 neighbors around each best
3. Always extends to R=10 if R=8 is in the candidate set (boundary rule)
4. 3 seeds per config (42, 137, 271)
5. **61,000 steps** per run (~1B tokens at 16,384 tokens/step)
6. All Stage 2 runs on RTX 3090

Expected: ~3–4 R candidates × ~3 P candidates × 4 d_model × 3 seeds = ~100–150 configs

---

## 5. Training Data

### Files produced by `data/build_bins.py`

| File | Tokens | Purpose |
|---|---|---|
| `data/tinystories_train.bin` | 100M | Stage 1 training |
| `data/stage2_train.bin` | ~1B | Stage 2 training (30/30/40 blend) |
| `data/val/tinystories_val.bin` | 500k | Val (official split) |
| `data/val/wikipedia_val.bin` | 500k | Val (Wikipedia shard 40, held out) |
| `data/val/fineweb_edu_val.bin` | 500k | Val (FineWeb-Edu shard 97 seed 42, held out) |

### Stage 2 blend (stage2_train.bin)

| Source | Tokens | Shards |
|---|---|---|
| TinyStories | 300M | All train shards |
| Wikipedia (20231101.en) | 300M | Shards 0–39 (shard 40 held out for val) |
| FineWeb-Edu (sample-10BT) | 400M | Shards 0–96 (shard 97 held out for val) |

Interleaved in 512-token chunks: schedule `[tiny, tiny, tiny, wiki, wiki, wiki, fw, fw, fw, fw]` repeating.

**Run once before any training:**
```bash
python data/build_bins.py --output-dir data/
```

---

## 6. Bugs Found and Fixed (important for paper integrity)

### Bug 1: Non-causal cross-attention (critical)
- **File**: `model/attention.py` — `MLACrossAttention`
- **Problem**: `is_causal=False`. With P=3+ deep preludes, the model learned to extract next-token signal from `e[t+1]` through non-causal cross-attention. Result: loss collapse to ppl_tiny ~1.02 (near-perfect) at ~1750 steps — model was cheating.
- **Fix**: `is_causal=True` in `MLACrossAttention.forward()`.
- **Impact**: All d=256/512/768 runs that completed before fix were invalid and reset.

### Bug 2: Gradient checkpointing corrupts gradients
- **File**: `model/cart.py`
- **Problem**: Aliasing between `h_input` used inside and outside `checkpoint()`, and K/V tensors reused across loop iterations. Caused silent gradient corruption.
- **Fix**: Gradient checkpointing disabled entirely. VRAM fits all configs without it (peak d=768: 1.88GB on 8GB VRAM).

### Bug 3: Stage 1 step count inconsistency
- **Problem**: Multiple files referenced 1500 as the canonical Stage 1 step count after the decision was made to use 3000.
- **Files fixed**: `train/train_one.py` constants comment, `sweep/generate_configs.py` SWEEP_META, `sweep/analyze.py` STAGE2_EVAL_STEP, `plot/plot_sweep.py` EVAL_STEP, `eval/perplexity.py` docstring, `test_phase5.py` (×2), `Notes for Writing the Paper.md` (×2), `sweep/schema.sql` comment.

### Bug 4: Stage 2 step count undefined
- **Problem**: `orchestrate.py` had no Stage 2 step count — would have run Stage 2 for only 3000 steps (wrong by 20×).
- **Fix**: Added `STAGE2_TOTAL_STEPS = 61_000` constant; orchestrate.py applies it when `--stage 2` is passed.

### Bug 5: stage2_train.bin too small
- **Problem**: `build_bins.py` only targeted 100M tokens for stage2 — Stage 2 needs ~1B tokens. The model would have cycled the training data 10× times.
- **Fix**: Scaled token targets to 300M/300M/400M and added multi-shard iteration via `iter_shard_texts()`.

### Bug 6: R=10 boundary gap in analyze.py
- **Problem**: `neighbors()` only returned [R-2, R, R+2] if R=8 strictly won. If R=6 won, candidates were [4,6,8] — no R=10 despite R=8 being the grid edge. Can't know where the curve flattens without testing one step past the edge.
- **Fix**: Added explicit R=10 extension after `neighbors()` call: always add R=10 to Stage 2 candidates if R=8 is in the set.

---

## 7. DenseBaseline — Parameter-Matched Comparison

### Why it exists
CART's central claim: "at equal parameter count, recurrent depth beats feed-forward depth." The dense baseline makes this claim testable. It is the primary architectural comparison in the paper.

### Architecture
- 7 uniform layers (parameter-matched to CART at each scale)
- Full-rank MHA: Q, K, V, O each d×d (no KV compression — this is the key difference from MLA)
- Same SwiGLU FFN with identical `ffn_intermediate` formula
- Same RMSNorm pre-norm
- RoPE on every layer
- Same tied embeddings
- No weight sharing, no recurrent state, no LTI, no HyperConnection

### Parameter match (at P=6, best-R expected configs)

| d_model | DenseBaseline 7L | CART (approx) |
|---|---|---|
| 256 | ~14.2M | ~14.4M |
| 512 | ~40.2M | ~41.0M |
| 768 | ~74.1M | ~75.3M |
| 1024 | ~122.7M | ~125.1M |

### New files
- `model/dense.py` — `DenseConfig`, `StandardMHASelfAttention`, `DenseLayer`, `DenseBaseline`
- `train/train_dense.py` — identical training loop to `train_one.py`, 61k steps default, `lti_spectral_radius` stored as NULL
- `sweep/generate_baselines.py` — inserts 4 dense configs into DB, handles `model_type` column migration

### DB integration
`model_type` column added to `configs` table (`DEFAULT 'cart'`). Dense configs stored with:
- `n_loops = 7` (used as n_layers)
- `n_prelude = 0` (sentinel)
- `model_type = 'dense'`

---

## 8. Active Branch

All changes pushed to: **`claude/review-cart-repo-xWteB`**

When you pull on your home machine:
```bash
git fetch origin
git checkout claude/review-cart-repo-xWteB
git pull origin claude/review-cart-repo-xWteB
pip install -r requirements.txt
```

---

## 9. Next Commands to Run (in order)

### Step 1: Build training data (run once, takes ~30–60 min)
```bash
python data/build_bins.py --output-dir data/
```
Verify output:
```
data/tinystories_train.bin     ~200 MB  (100M tokens)
data/stage2_train.bin          ~2 GB    (~1B tokens)
data/val/tinystories_val.bin   ~1 MB    (500k tokens)
data/val/wikipedia_val.bin     ~1 MB    (500k tokens)
data/val/fineweb_edu_val.bin   ~1 MB    (500k tokens)
```

### Step 2: Initialize the database
```bash
python sweep/generate_configs.py --db results.db
```
Expected output: "Inserted 64 new configs" and "All ordering checks passed."

### Step 3: Run Stage 1 sweep
On RTX 3050:
```bash
python sweep/orchestrate.py --stage 1 --hardware 3050 --db results.db
```
On RTX 3090 (for d=1024 configs):
```bash
python sweep/orchestrate.py --stage 1 --hardware 3090 --db results.db
```
- 48 configs on 3050 (d=256/512/768), 16 configs on 3090 (d=1024)
- ~3000 steps each, eval every 500 steps
- Test run first: `--max-configs 2 --max-steps 50`

### Step 4: Analyze Stage 1 results and generate Stage 2 configs
```bash
python sweep/analyze.py --db results.db --insert-stage2
```
Review proposals, then confirm by checking `sweep/stage2_configs.json`.

### Step 5: Generate and run DenseBaseline configs
```bash
python sweep/generate_baselines.py --db results.db
# Then run each baseline (4 total, ~61k steps each):
python train/train_dense.py --config-id <id> --db results.db --ckpt-interval 5000
```
Get the config IDs from generate_baselines.py output.

### Step 6: Run Stage 2 sweep (all on RTX 3090)
```bash
python sweep/orchestrate.py --stage 2 --hardware 3090 --db results.db --ckpt-interval 5000
```

### Step 7: Generate paper figures
```bash
python plot/plot_sweep.py --db results.db
```

---

## 10. Key Design Decisions Recorded

1. **Why 7 layers for dense baseline**: Per-layer dense budget ~12d². CART prelude/coda/core layers ~10.75d². With P=6 CART configs, 7 dense layers gives <2% parameter difference across all four scales — clean match.

2. **Why not Pythia as baseline**: Pythia uses The Pile (different training data), making it impossible to isolate architecture vs data. Build your own dense baseline on the same data.

3. **Why 3050 vs 3090 split**: All d=1024 configs go to 3090. Everything else runs on 3050. Stage 2 should all run on 3090 for faster iteration.

4. **Why is_causal=True in cross-attention**: `e[t]` was computed with causal self-attention, but `e[t]` still saw token `t` as input. Non-causal cross-attention would let `h[t]` attend to `e[t+1]` which encodes token `t+1` — the prediction target. This leaks future tokens.

5. **Why RoPE only in prelude/coda**: The hidden state `h` in the core loop does not correspond to token positions — it's a recurrent state being refined over iterations. Applying RoPE to Q in cross-attention would impose a positional structure on `h` that doesn't make sense semantically.

6. **Why gradient checkpointing is disabled**: The HyperConnection ring buffer holds references to previous `h` tensors. When checkpoint() recomputes the core block, it creates new tensors that are not the same objects the buffer holds. Combined with K/V reuse across loops, this produces silent gradient corruption with no error message.

7. **Why fixed-order dataset (no shuffle)**: All sweep runs must see training data in the same order for cross-architecture comparability. `shuffle=False` in DataLoader.

8. **Why 30/30/40 blend**: TinyStories is high-quality but narrow (children's stories). Wikipedia adds factual breadth. FineWeb-Edu adds educational/instructional diversity. 40% FineWeb to weight toward the most general distribution.

---

## 11. Perplexity Comparability — Dense vs CART

Dense and CART perplexity numbers are **directly comparable**:
- Same tokenizer → same per-token probability definition
- Same three val sets → same test distribution
- Same `eval_perplexity()` function implementation
- Same `max_batches=50` → same 50 sequences evaluated
- `lti_spectral_radius` is NULL for dense runs — filter with `WHERE model_type='cart'` in any query that accesses this column

Console output difference: CART logs `rho=X.XXXX`, dense does not. `tokens_per_sec` will be ~50% higher for dense (less compute per forward pass) — this is informative for the paper's efficiency discussion.

---

## 12. Paper Argument Structure (current thinking)

**Central claim**: At equal parameter count, CART's recurrent depth achieves lower perplexity than a standard 7-layer dense transformer.

**Supporting evidence from sweep**:
- Lower perplexity across 3 val domains at equal params
- Scaling behavior: does leverage ratio improve with R?
- Stable training: spectral radius stays below 0.99 throughout
- VRAM efficiency: CART uses less memory than dense at same effective parameter count

**Evaluation domains**: TinyStories (fluency/coherence), Wikipedia (factual/structured), FineWeb-Edu (educational/reasoning) — tests generalization, not overfitting to one distribution.
