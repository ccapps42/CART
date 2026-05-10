# Notes for Writing the Paper

## Stage 1 Sweep Status (as of 2026-05-03)

All configs use corrected architecture (causal cross-attention), mixed training data (stage2_train.bin), 3000 steps, EVAL_INTERVAL=500.

| d_model | Steps | Training data | Status |
|---------|-------|--------------|--------|
| 256 | 3000 | Mixed (30% TinyStories / 30% Wikipedia / 40% FineWeb-Edu) | **Complete** (16/16) |
| 512 | 3000 | Mixed | **Complete** (16/16) |
| 768 | 3000 | Mixed | **Complete** (16/16) |
| 1024 | 3000 | Mixed | **Complete** (16/16) |

---

## ARCHITECTURAL BUG — Full Sweep Restart Required

### What happened

During the d=768 sweep, R=2 P=3 produced impossible perplexity values at step 3000:
- ppl_tiny=1.02, ppl_wiki=1.31, ppl_edu=1.19

These values are near-perfect prediction across ALL THREE held-out evaluation sets, including Wikipedia and FineWeb-Edu which the model was not training on. Even GPT-4 does not achieve ppl_wiki=1.31. This ruled out memorization and pointed to a code bug.

The training log confirmed catastrophic loss collapse beginning at step 1750 (grad norm spike to 2.834 at step 1950, then loss falling from 4.24 → 0.13 over the next 1,000 steps).

Critically, R=2 P=2 at the same d=768 with identical settings showed NO collapse, running cleanly to ppl_wiki=67.82. The only difference was one additional prelude layer.

### Root cause: non-causal cross-attention

`MLACrossAttention.forward()` was set to `is_causal=False`. The docstring justified this with: *"The causal mask was already applied in the prelude when e was built."* This reasoning is incorrect.

CART is an autoregressive model — the prelude and recurrent core operate on the same token sequence. `h[t]` is trying to predict token `t+1`. With non-causal cross-attention, `h[t]` could attend to `e[t+1]`, `e[t+2]`, etc. But `e[t+1]` was built from token `t+1` as part of its input context. This gives `h[t]` indirect access to the token it is supposed to predict — a target leak.

With a shallow prelude (P=2), `e` does not encode tokens richly enough for the model to easily exploit this. With P=3, after approximately 1,750 training steps on mixed data, the model discovered how to extract next-token information from `e` via the non-causal attention, and the loss collapsed to near zero.

The severity scales with prelude depth because deeper preludes produce richer `e` representations, making the future-token signal easier to extract.

### Fix

Changed `is_causal=False` to `is_causal=True` in `MLACrossAttention.forward()` in `model/attention.py`. This enforces that `h[t]` can only attend to `e[0..t]`, which is correct for an autoregressive model.

```python
# Before (wrong):
out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

# After (correct):
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

Updated docstring to explain the correct causality requirement.

### Why ALL configs were reset

The d=256 and d=512 sweeps (32 configs total) were completed with the non-causal bug. None showed catastrophic collapse at 1500 steps — the leak apparently requires more training or richer representations to fully exploit. However, they trained on a technically incorrect architecture. For a rigorous paper comparing across model scales, all results must come from the same correct architecture. All 48 configs (d=256, 512, 768) were reset and are rerunning.

### Paper note

The causal cross-attention constraint is an architectural contribution worth stating explicitly in the paper. In encoder-decoder models, non-causal cross-attention is correct because encoder and decoder operate on different sequences. In CART, the prelude and core operate on the SAME sequence for next-token prediction, so causal masking is required in the cross-attention. This distinction is non-obvious and the incorrect version is plausible enough that it appeared in the initial implementation.

---

## Gradient Checkpointing Bug (earlier, also fixed)

Gradient checkpointing was enabled for d=768 configs (`if cfg.d_model >= 768`). Two aliasing problems caused corrupted gradients:

1. `h_input` was used both inside `checkpoint(self.core, h_input, K, V)` and outside it in `self.lti(h_input, transformer_out)`.
2. K and V (same tensor objects) were passed to the checkpoint across all R loop iterations.

Fix: gradient checkpointing disabled entirely. VRAM fits all configs without it (d=768 peak observed: 1.88 GB, 8 GB available).

---

## Pre-bug d=256 Results (1500 steps, TinyStories, non-causal architecture — SUPERSEDED)

*These results are from the incorrect architecture and are being rerun. Recorded here for reference only.*

**Key patterns observed (likely valid despite bug, as collapse didn't manifest):**
- P=6 configs were best at every R (ppl_tiny 11.00–11.18 for P=6 vs 12.34–13.56 for P=2)
- R-insensitivity: within P=6, R=2 through R=8 all scored within 0.2 ppl of each other
- P dominates R: higher P beats higher R at matched compute budget
- Spectral radius converged to ~0.896 across all configs — LTI stability working as designed

---

## Pre-bug d=512 Results (1500 steps, TinyStories, non-causal architecture — SUPERSEDED)

*These results are from the incorrect architecture and are being rerun. Recorded here for reference only.*

**Key patterns observed:**
- P=6 best at every R (ppl_tiny 7.38–7.51)
- P=3 consistently worse than P=2 across all R values — previously hypothesized as KV bottleneck; now understood as early-stage exploitation of the non-causal attention leak. P=3's richer prelude output made the leak more exploitable even at 1500 steps.
- R-insensitivity: P=6 configs ranged only 7.38–7.51 across R=2 to R=8

---

## Training Data Decision

Mixed data (`stage2_train.bin`) is used for all configs at all scales. Reason: d=768 (~50M params) trained on TinyStories-only showed training loss collapse to 0.09 by step 1750 — the model memorized the repetitive synthetic dataset. Mixed data (30% TinyStories / 30% Wikipedia / 40% FineWeb-Edu, 100M tokens interleaved) prevents memorization and produces meaningful ppl_wiki and ppl_edu signals at all scales.

**All configs:** 3000 steps, stage2_train.bin, EVAL_INTERVAL=500. Eval checkpoints at steps 500, 1000, 1500, 2000, 2500, 3000 for every config. ppl_tiny, ppl_wiki, and ppl_edu are all directly comparable across d values. Use step-3000 as the primary cross-scale comparison point.

### Stage 2 training data — rebuilt 2026-05-03 (1024-token chunks)

`data/stage2/stage2_train.bin` — **999,997,440 tokens (~1B), 2000 MB**

Built with:
```powershell
python data/build_bins.py --stage2-only --stage2-out data/stage2
```

| Source | Tokens encoded | Shards used | % of source used |
|--------|---------------|-------------|-----------------|
| TinyStories | 300,000,000 | All 4 train shards (58% of docs) | ~58% — near full |
| Wikipedia | 300,000,000 | 40 train shards (shard 40 held out for val) | ~8% |
| FineWeb-Edu | 400,000,000 | 97 train shards (shard 97 held out for val) | ~4% |
| **Interleaved total** | **999,997,440** | 30/30/40 blend in **1024-token chunks** | |

Tokenizer: NousResearch/Llama-2-7b-hf (bos=1, eos=2, vocab=32000). Build time: ~14 min.

**Chunk size change from first build:** Rebuilt with `SEQ_LEN_INTERLEAVE=1024` (was 512) to match Stage 2 training seq_len=1024. With 512-token chunks and 1024-token training windows, ~40% of training sequences would span two different source domains (e.g., first 512 tokens TinyStories, next 512 tokens Wikipedia). 1024-token chunks ensure every training window comes from a single source.

**TinyStories is the binding constraint:** At ~517M total tokens, TinyStories is nearly exhausted at 300M (58%). Wikipedia (~3.6B tokens) and FineWeb-Edu (~9.9B tokens) have vast headroom. If Stage 2 ever needs to scale beyond ~1B tokens, TinyStories proportion must be reduced or it will repeat.

---

## d=256 Clean Results (corrected architecture, mixed data, 3000 steps) — COMPLETE

**ppl_tiny at step 3000:**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 14.21 | 13.50 | 13.36 | 12.73 |
| 4 | 14.28 | 13.63 | 13.08 | 12.38 |
| 6 | 14.31 | 13.65 | 13.07 | 12.31 |
| 8 | 14.35 | 13.70 | 13.03 | 12.34 |

**ppl_wiki at step 3000:**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 202.1 | 195.4 | 194.3 | 185.7 |
| 4 | 204.3 | 198.2 | 191.8 | 185.3 |
| 6 | 206.6 | 197.3 | 193.7 | 184.9 |
| 8 | 207.6 | 197.0 | 194.2 | 186.2 |

**Key patterns:**
- P dominates R: P=6 beats P=4 beats P=3 beats P=2 at every R value without exception
- P=3 anomaly is gone: P=3 correctly outperforms P=2 everywhere (confirms non-causal bug is fixed)
- R sensitivity within P=6: R=2 (12.73 tiny) is noticeably worse; R=4/6/8 are within 0.07 of each other (12.31–12.38)
- Best config at d=256: R=6 P=6 (ppl_tiny=12.31, ppl_wiki=184.9)
- Wall time: 14.3 min (R=2 P=2) to 26.9 min (R=8 P=6) on RTX 3050
- Spectral radius: converged to 0.8908–0.8929 (mean 0.8922) across all 16 configs — very tight, confirms LTI stability working as designed

---

## d=512 Clean Results (corrected architecture, mixed data, 3000 steps) — COMPLETE

**ppl_wiki at step 3000:**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 147.15 | 146.67 | 144.19 | 138.80 |
| 4 | 148.92 | 146.04 | 144.95 | 137.94 |
| 6 | 150.87 | 147.00 | 143.28 | 137.37 |
| 8 | 150.29 | 145.40 | 143.56 | 136.42 |

**Key patterns:**
- P=6 best at every R; P ordering holds cleanly
- At P=2, higher R slightly *hurts* (R=8: 150.29 vs R=2: 147.15) — shallow prelude can't benefit from more loops
- At P=6, higher R consistently helps (R=8: 136.42 best)
- Best config: R=8 P=6 (ppl_wiki=136.42, ppl_tiny=8.40)
- Spectral radius: mean=0.8925, range=[0.8912, 0.8950] across all 16 configs

---

## d=768 Clean Results (corrected architecture, mixed data, 3000 steps) — COMPLETE

**ppl_wiki at step 3000:**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 128.41 | 124.83 | 123.16 | 118.28 |
| 4 | 129.91 | 124.64 | 122.24 | 116.43 |
| 6 | 130.24 | 122.85 | 119.96 | 115.99 |
| 8 | 127.80 | 120.84 | 118.98 | **114.96** |

**Spectral radius (rho) at step 3000:**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 0.8934 | 0.8926 | 0.8937 | 0.8935 |
| 4 | 0.8939 | 0.8934 | 0.8949 | 0.8946 |
| 6 | 0.8946 | 0.8942 | 0.8949 | 0.8954 |
| 8 | 0.8959 | 0.8951 | 0.8956 | 0.8956 |

**Mean rho by R:** R=2: 0.8933 → R=4: 0.8942 → R=6: 0.8948 → R=8: 0.8956. Overall mean=0.8945 — notably higher than d=256 (0.8922) and d=512 (0.8925). Scale trend accelerating.

**R benefit by P (R=8 vs R=2 gain in ppl_wiki):**
- P=2: gain=0.61 (nearly nothing; U-shape — R=4/6 worse than R=2, R=8 barely recovers)
- P=3: gain=3.99
- P=4: gain=4.18 (peak R benefit)
- P=6: gain=3.32

The crossover from "R hurts" to "R helps" falls between P=2 and P=3. P=4 extracts the most from additional loops. At P=6 the prelude is so rich that even R=2 approaches the fixed point, so marginal R gain shrinks slightly.

**Key patterns:**
- P=6 best at every R; P ordering holds
- Best config: R=8 P=6 (ppl_wiki=114.96, ppl_tiny=7.06)
- Best at d=256 was R=6 P=6 (not R=8) — d=768 shows R=8 still improving, confirming larger models benefit more from additional loops
- VRAM peak: 1.88 GB (R=2 P=2) to ~4.7 GB (R=8 P=6 estimated) — all within 8 GB

---

## d=1024 Results (corrected architecture, mixed data, 3000 steps) — COMPLETE

**ppl_wiki at step 3000:**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 113.06 | 109.64 | 105.43 | 103.14 |
| 4 | 113.97 | 109.41 | 105.01 | 101.09 |
| 6 | 113.57 | 106.59 | 102.88 | 98.75 |
| 8 | 111.60 | 104.75 | 100.66 | **97.73** |

**ppl_tiny at step 3000:**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 7.064 | 6.674 | 6.454 | 6.208 |
| 4 | 7.018 | 6.570 | 6.228 | 6.129 |
| 6 | 6.953 | 6.459 | 6.172 | 6.055 |
| 8 | 6.786 | 6.392 | 6.159 | **6.037** |

**ppl_edu at step 3000:**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 96.47 | 92.12 | 89.98 | 85.32 |
| 4 | 96.29 | 91.27 | 86.83 | 84.05 |
| 6 | 96.14 | 88.17 | 84.95 | 82.38 |
| 8 | 93.06 | 86.90 | 84.66 | **82.00** |

**Spectral radius (ρ) at step 3000:**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 0.8945 | 0.8966 | 0.8952 | 0.8957 |
| 4 | 0.8954 | 0.8975 | 0.8956 | 0.8962 |
| 6 | 0.8961 | 0.8982 | 0.8961 | 0.8965 |
| 8 | 0.8967 | 0.8987 | 0.8962 | 0.8966 |

Mean ρ by R: R=2: 0.8955 → R=4: 0.8962 → R=6: 0.8967 → R=8: 0.8971. Overall mean=0.8964 — highest across all scales. Scale trend confirmed (d=256: 0.8922, d=512: 0.8925, d=768: 0.8945, d=1024: 0.8964).

**Wall time at step 3000 (RTX 3050):**

| R | P=2 | P=3 | P=4 | P=6 |
|---|-----|-----|-----|-----|
| 2 | 54.2 min | 62.6 min | 68.7 min | 83.5 min |
| 4 | 67.9 min | 75.1 min | 82.3 min | 97.0 min |
| 6 | 82.1 min | 90.0 min | 96.2 min | 110.5 min |
| 8 | 95.2 min | 102.3 min | 109.7 min | 124.0 min |

**R=8 vs R=2 gain by P (ppl_wiki and ppl_tiny):**

| P | wiki R=2 | wiki R=8 | wiki gain | tiny R=2 | tiny R=8 | tiny gain |
|---|----------|----------|-----------|----------|----------|-----------|
| 2 | 113.06 | 111.60 | +1.28% | 7.064 | 6.786 | +3.94% |
| 3 | 109.64 | 104.75 | +4.46% | 6.674 | 6.392 | +4.23% |
| 4 | 105.43 | 100.66 | +4.52% | 6.454 | 6.159 | +4.58% |
| 6 | 103.14 | 97.73 | +5.24% | 6.208 | 6.037 | +2.75% |

**Key patterns:**
- P=6 best at every R; P ordering holds completely at d=1024
- R=8 consistently best at all P values — including P=2. Critical finding: the "R hurts at P=2" pattern that appeared at d=256/512 is gone at d=1024. At d=768, P=2 showed a U-shape (R=4/6 worse than R=2, R=8 barely recovering). At d=1024, R monotonically helps even at P=2. The crossover from "R hurts" to "R helps at all P" falls between d=768 and d=1024.
- Best config: R=8 P=6 (ppl_wiki=97.73, ppl_tiny=6.037, ppl_edu=82.00, ρ=0.8966)
- Wall time: 54.2 min (R=2 P=2) to 124.0 min (R=8 P=6) on RTX 3050

**Scale dominance confirmed:**
- d=1024 R=2 P=2 (113.06) beats d=768 R=8 P=6 (114.96) — weakest config at larger scale beats best at smaller scale
- d=1024 best (97.73) improves 14.7% over d=768 best (114.96)
- Scaling trend — R=2 P=2 ppl_wiki: d=256: 202.10 → d=512: 147.15 (−27.2%) → d=768: 128.41 (−12.8%) → d=1024: 113.06 (−12.0%)

---

## Cross-Scale Summary — Stage 1 Complete (all 64 configs, step 3000)

### Best config per scale

| d_model | Best R | Best P | ppl_wiki | ppl_tiny | ppl_edu | ρ | Stored params |
|---------|--------|--------|----------|----------|---------|---|--------------|
| 256 | 6 | 6 | 184.93 | 12.31 | 161.28 | 0.8927 | ~14M |
| 512 | 8 | 6 | 136.42 | 8.404 | 114.52 | 0.8936 | ~41M |
| 768 | 8 | 6 | 114.96 | 7.063 | 95.40 | 0.8956 | ~75M |
| 1024 | 8 | 6 | 97.73 | 6.037 | 82.00 | 0.8966 | ~125M |

Note: d=256 peaks at R=6 (R=8 is 1.3 pts worse on ppl_wiki). All other scales peak at R=8.

### P=6 slice — R sensitivity by scale (ppl_wiki)

| d_model | R=2 | R=4 | R=6 | R=8 | R gain (R=2→R=8) |
|---------|-----|-----|-----|-----|-----------------|
| 256 | 185.73 | 185.31 | **184.93** | 186.20 | −0.25% (regresses) |
| 512 | 138.80 | 137.94 | 137.37 | **136.42** | +1.72% |
| 768 | 118.28 | 116.43 | 115.99 | **114.96** | +2.81% |
| 1024 | 103.14 | 101.09 | 98.75 | **97.73** | +5.24% |

R is productive only when d_model provides sufficient capacity. d=256 gains nothing (and slightly regresses at R=8). Benefit increases monotonically with scale: 1.72% → 2.81% → 5.24% from d=512 to d=1024.

### Spectral radius summary — all 64 configs

| d_model | Mean ρ | Min ρ | Max ρ | Δ from d=256 |
|---------|--------|-------|-------|-------------|
| 256 | 0.8922 | 0.8908 | 0.8929 | — |
| 512 | 0.8925 | 0.8912 | 0.8950 | +0.0003 |
| 768 | 0.8945 | 0.8926 | 0.8959 | +0.0023 |
| 1024 | 0.8964 | 0.8945 | 0.8987 | +0.0042 |

ρ is remarkably stable across P and R within a scale (range ≤ 0.0042 at any scale). The LTI gate converges to a narrow band around 0.893. The slow upward drift with scale (+0.0042 d=256→1024) shows the model adapting its memory timescale to increasing representation complexity.

### Ideal ρ vs. observed ρ (across all scales)

| R | Ideal ρ = 0.5^(1/R) | Observed mean (all scales) | Interpretation |
|---|---------------------|---------------------------|---------------|
| 2 | 0.7071 | 0.8929 | Observed >> ideal: model overshoots "optimal" decay |
| 4 | 0.8409 | 0.8937 | Observed > ideal: model uses slower decay than optimal |
| 6 | 0.8909 | 0.8942 | Observed ≈ ideal: near-optimal decay for R=6 |
| 8 | 0.9170 | 0.8947 | Observed < ideal: model undershoots; some fixed-point residual remains |

The ideal ρ = 0.5^(1/R) is defined as the decay where ρ^R = 0.5 (half the initial residual survives after all loops). The model learns ρ ≈ 0.893 regardless of R — approximately the R=6 ideal. For R<6 it overshoots; for R=8 it undershoots. This is consistent with the model finding a stable convergence rate that works adequately across the range of R values encountered in training (not adapting to a specific R, since each model only ever trains with one R). The upward drift with scale means d=1024 approaches the R=8 ideal more closely than d=256 (gap narrows from 0.0241 to 0.0183).

---

## Paper Structure: What Each Stage Contributes

### Stage 1 — CART only, no Dense comparison

Stage 1 is a screening step, not a result section. Only CART configs are run at Stage 1. Dense baselines are not run at Stage 1 settings because there is nothing to screen — Dense has no hyperparameters to rank across (no R, no P). Dense only needs to be run once, at full scale, for the final comparison.

Stage 1 contributes two things to the paper:

1. **Hyperparameter screening:** Identifies which (R, P) combinations are worth running at full Stage 2 cost. Stage 1 perplexity rankings are the evidence for this selection, not the result itself.
2. **Predictive validity:** Shows that Stage 1 rankings are predictive of Stage 2 rankings (measured via Spearman ρ). This is a methodological contribution — it validates the screening approach and lets future researchers use short runs to prune their own config spaces. (TODO: compute Spearman ρ once Stage 2 complete.)

Stage 1 absolute perplexity numbers are not cited in the main results. They trained on ~49M tokens at seq_len=512 — far from convergence and not comparable to any external baseline.

### Stage 2 — CART vs. Dense, the main result

Stage 2 is the primary contribution of the paper. Both CART and Dense are trained under identical conditions: seq_len=1024, 30,500 steps (~1B tokens), same data (stage2_train.bin), same optimizer and schedule. This is the only apples-to-apples comparison.

Stage 2 answers the central question: does CART's parameter efficiency translate into better quality per stored parameter versus a standard Dense transformer?

Four Dense baselines are run after the CART sweep completes (d=256, 512, 768, 1024), one per scale. The comparison is:
- **CART (P=6, R=6/8/10, 3 seeds)** vs. **Dense baseline (same d, same training budget)**
- Secondary comparison: CART vs. FLOPs-matched Dense (Dense 17L at d=1024) — answers "why not just train a bigger dense model with the same compute?"

Dense baseline numbers from earlier runs (different seq_len, different data, recycled tokens) are not used in the paper. Only Stage 2 Dense runs count.

---

## Stage 2 Design

### Why Stage 1 doesn't project to benchmarks

Stage 1 trains for ~49M tokens (3000 steps × 16,384 tokens/step). Chinchilla-optimal compute for a 75M-param model is ~1.5B tokens — Stage 1 is at ~3% of that. At this scale, benchmark tasks (HellaSwag, ARC-C, PIQA, LAMBADA) return near-random accuracy. Stage 1's purpose is hyperparameter ranking, not capability measurement. The rankings are reliable; the absolute numbers are not benchmark-predictive.

Learning curves confirm this: every config is still declining steeply at step 3000 with no plateau. d=768 R=2 P=2 went ppl_wiki 292 → 128 between steps 500 and 3000. Stage 2 with 10–50× more tokens should yield substantially better models.

### Which configs to run in Stage 2

Stage 1 makes this clear. Run P=6, R=6 and R=8 at each scale. No need to benchmark the full grid. Specifically:

| Config | Stored params | Effective params | Leverage | Hardware |
|--------|--------------|-----------------|---------|---------|
| d=256 R=6 P=6 | 14.37M | 17.97M | 1.251× | RTX 3090 |
| d=256 R=8 P=6 | 14.37M | 19.42M | 1.352× | RTX 3090 |
| d=512 R=6 P=6 | 41.05M | 55.47M | 1.351× | RTX 3090 |
| d=512 R=8 P=6 | 41.05M | 61.24M | 1.492× | RTX 3090 |
| d=768 R=6 P=6 | 75.34M | 104.84M | 1.392× | RTX 3090 |
| d=768 R=8 P=6 | 75.34M | 116.64M | 1.548× | RTX 3090 |
| d=1024 R=6 P=6 | 125.09M | 178.84M | 1.430× | RTX 3090 |
| d=1024 R=8 P=6 | 125.09M | 200.34M | 1.602× | RTX 3090 |

All Stage 2 runs on RTX 3090. Consider also running R=4 P=6 at each scale as a cheaper R-sensitivity data point.

### Training budget for Stage 2

**Minimum for meaningful benchmarks:** ~500M tokens. At this level HellaSwag, ARC-C, and PIQA begin to differentiate models reliably.

**Target:** ~1–2B tokens per config. Chinchilla-optimal for 75M params is ~1.5B tokens. At 1B tokens, results should be directly comparable to published models in the 50–130M parameter range (GPT-2, OPT-125M, Pythia-70M/160M, Cerebras-GPT-111M).

**Practical constraint:** Observed tokens/sec (Stage 1, RTX 3050): d=768 R=8 P=6: ~9,876 tok/s; d=1024 R=8 P=6: ~6,725 tok/s. On RTX 3090 expect 2–3× faster. At ~18,000 tok/s (RTX 3090, d=1024 R=8 P=6 estimated), 1B tokens = ~15.5 hours per config. With 8 configs × 3 seeds = 24 runs, total ~370 hours. Run concurrently where possible.

### Stage 2 protocol

- 3 seeds per config (for statistical confidence on benchmarks)
- Eval with `lm-evaluation-harness` (EleutherAI) — the standard tool for few-shot benchmark eval
- Report mean ± std across seeds
- Save checkpoints at regular intervals to plot benchmark score vs. training tokens

### Baselines to compare against

The core claim is efficiency: does CART at N stored parameters match or beat a dense transformer of N stored parameters at equal training compute?

**Primary baselines (same stored parameter count, same training tokens):**

| Baseline | Params | Notes |
|----------|--------|-------|
| GPT-2 Small | 117M | Widely cited; trained on 40B tokens so compare at equal-token checkpoints |
| Pythia-70M | 70M | Trained on The Pile, checkpoints at every 1B tokens — ideal for equal-compute comparison |
| Pythia-160M | 160M | Upper bracket for d=768/d=1024 comparison |
| OPT-125M | 125M | Well-documented; compare against d=1024 configs |
| Cerebras-GPT-111M | 111M | Trained with Chinchilla recipe; clean apples-to-apples |

**Why Pythia is the best baseline:** Pythia releases intermediate checkpoints at fixed token counts, so you can compare CART at 1B tokens against Pythia at 1B tokens directly — same training compute, same evaluation. This is the cleanest possible comparison and pre-empts reviewer concerns about training budget differences.

### Benchmarks to target

| Benchmark | Why |
|-----------|-----|
| HellaSwag | Sentence completion; well-calibrated for 70–200M models; standard |
| ARC-Challenge | Reasoning; harder than ARC-Easy; differentiates architecture quality |
| PIQA | Physical commonsense; tests general world knowledge |
| LAMBADA | Long-range word prediction; directly tests recurrent memory — good fit for CART |
| WikiText-103 perplexity | Standardized; enables direct comparison to published numbers |

**LAMBADA is especially important for CART:** It requires predicting the last word of a passage where the answer depends on context from several sentences back. CART's recurrent core with LTI memory is specifically designed for this kind of long-range dependency. If CART outperforms same-size dense transformers on LAMBADA, that is the clearest possible demonstration of the architectural contribution.

### The key comparison the paper needs to make

Two claims, each needing its own number:

1. **Efficiency claim:** CART R=8 P=6 at N stored parameters achieves better benchmark scores than a dense transformer of N stored parameters at equal training compute. This justifies the architecture.

2. **Compute claim:** CART R=8 P=6 achieves comparable benchmark scores to a dense transformer of N×leverage stored parameters (i.e., the effective parameter count) at lower training cost. This justifies the parameter leverage framing.

Do not claim both simultaneously without clearly separating them. Reviewers will conflate them otherwise.

---

## Parameter Counts — All Stage 2 Candidate Configs

Verified by instantiating models with `CARTConfig` and counting `m.parameters()`.

| Config | Stored params | Effective params | Leverage |
|--------|--------------|-----------------|---------|
| d=256 R=6 P=6 | 14.37M | 17.97M | 1.251× |
| d=256 R=8 P=6 | 14.37M | 19.42M | 1.352× |
| d=512 R=6 P=6 | 41.05M | 55.47M | 1.351× |
| d=512 R=8 P=6 | 41.05M | 61.24M | 1.492× |
| d=768 R=6 P=6 | 75.34M | 104.84M | 1.392× |
| d=768 R=8 P=6 | 75.34M | 116.64M | 1.548× |
| d=1024 R=6 P=6 | 125.09M | 178.84M | 1.430× |
| d=1024 R=8 P=6 | 125.09M | 200.34M | 1.602× |

Effective params = stored params + (R−1) × core block params. All P=6 configs at the same d_model share the same stored parameter count — the prelude/coda/embed/proj layers are identical; only R changes.

**The headline claim:** CART d=1024 R=8 P=6 delivers the compute depth of a 200M-parameter model from 125M stored parameters — a 37% reduction in stored parameters for equivalent effective depth. Trained on a single consumer RTX 3090.

**For the paper abstract/intro:** "A 200M effective-parameter model trained on a single consumer GPU" is concrete and memorable. Most 200M-class dense models require multi-GPU or expensive cloud hardware to train.

**Caveat for reviewers:** The leverage ratio (1.60× at best) is real but modest. The compelling claim isn't the parameter count alone — it's whether quality per stored parameter is better than a dense model of the same stored size. That's what Stage 2 benchmarks (HellaSwag, ARC-C, LAMBADA, PIQA) need to demonstrate. Do not overclaim on parameter efficiency without the benchmark results to support it.

---

## FLOPs Analysis (eval/flops_calc.py, d=1024, T=1024)

### Per-sequence forward-pass FLOPs

| Config | GFLOPs | FLOPs-matched Dense layers | Stored params | Effective params | Leverage |
|--------|--------|---------------------------|--------------|-----------------|---------|
| Dense 7L | 214.21 | — | 122.68M | 122.68M | 1.00× |
| CART R=6 P=4 | 299.04 | 9.8 L | 101.97M | 155.71M | 1.53× |
| CART R=6 P=6 | 354.87 | 11.6 L | 125.04M | 178.78M | 1.43× |
| CART R=6 P=8 | 410.71 | 13.4 L | 148.11M | 201.85M | 1.36× |
| CART R=8 P=4 | 351.65 | 11.5 L | 101.97M | 177.21M | 1.74× |
| CART R=8 P=6 | 407.49 | 13.3 L | 125.04M | 200.28M | 1.60× |
| CART R=8 P=8 | 463.32 | 15.1 L | 148.11M | 223.35M | 1.51× |
| CART R=10 P=4 | 404.26 | 13.2 L | 101.97M | 198.71M | 1.95× |
| CART R=10 P=6 | 460.10 | 15.0 L | 125.04M | 221.77M | 1.77× |
| CART R=10 P=8 | 515.93 | **16.9 L** | 148.11M | 244.84M | 1.65× |

**FLOPs-matched Dense for best CART config (R=10, P=8):** 17 layers (+0.8% overshoot), 251.13M stored params, 520.23 GFLOPs.

CART R=10 P=8 stores **1.70× fewer parameters** than the FLOPs-matched 17-layer Dense (148M vs 251M stored).

### Three comparison points for the paper (d=1024)

| Model | GFLOPs | Stored params | Role |
|-------|--------|--------------|------|
| Dense 7L | 214 | 123M | Param-matched to CART P=6 |
| CART R=10 P=8 | 516 | 148M | Best CART config |
| Dense 17L | 520 | 251M | FLOPs-matched to best CART |

The key reviewer question answered by the FLOPs-matched Dense: "Why not just train a larger dense model with those same FLOPs?" Dense 17L uses 1.70× more stored parameters to achieve the same compute budget. If CART R=10 P=8 matches or beats Dense 17L on benchmarks, that is the clearest possible demonstration of parameter efficiency.

For the sweep's planned configs (R=6/8, P=4/6/8), the FLOPs range is 299–463 GFLOPs per sequence, equivalent to 9.8–15.1 Dense layers. These use **1.50–2.00× fewer stored parameters** than their FLOPs-matched equivalents. The R=6/P=4 config at 101.97M stored params is particularly striking: 9.8 Dense-layer-equivalent FLOPs from a model smaller than Dense 7L in stored params.

### Why training takes so long despite sub-TFLOPs per sequence

The 516 GFLOPs figure is per-sequence forward pass only. The full per-optimizer-step cost is:

```
32 sequences/step (batch=8 × accum=4) × 3 × 516 GFLOPs ≈ 49.5 TFLOPs/step
          ↑ forward + backward (backward ≈ 2× forward)
```

At RTX 3090 BF16 theoretical peak (142 TFLOPS) with ~30% MFU:
```
49.5 TFLOPs ÷ (142 × 0.30) ≈ 1.16 sec/step → 30,500 steps ≈ 9.8 hrs/config
```

MFU is 20–40% in practice because transformer training is often **memory-bandwidth-bound**, not compute-bound. The 3090's 936 GB/s HBM must load weight matrices (d×d = 1M floats at d=1024) for every forward/backward pass, regardless of TFLOP capacity. The "30 TFLOPS" figure commonly cited for the 3090 is FP32 — BF16 tensor cores deliver ~142 TFLOPS theoretical, but the memory wall still limits effective utilization.

**For the paper's compute budget section:** quote wall-clock hours on RTX 3090 and convert to GPU-hours. Do not quote raw TFLOP estimates since MFU variance makes them less credible than measured training times.

---

## Related Work — Key Papers to Cite and Differentiate From

Both papers below appeared April 2026, essentially simultaneous with this work. They validate the research direction and raise the bar for differentiation.

---

### Parcae: Scaling Laws for Stable Looped Language Models (arXiv 2604.12946)
**Prairie, Novack, Berg-Kirkpatrick, Fu — UC San Diego / Together AI**

**Architecture:** Prelude → recurrent block (looped T times) → coda. Nearly identical top-level structure to CART. They formalize the looping as a nonlinear dynamical system over the residual stream:

```
h_{t+1} = Ā h_t + B̄ e + R̄(h_t, e)
```

**Stability mechanism:** Parameterizes the state transition matrix A as a negative diagonal (`A := Diag(−exp(log_A))`), then discretizes via zero-order hold (ZOH). This *architecturally constrains* ρ(Ā) < 1. Prior looped models either set ρ=1 (marginally stable) or left it unconstrained (unstable).

**Training:** Variable recurrence depth per micro-batch (Poisson-sampled). Truncated BPTT limited to ⌈μ_rec/2⌉ steps.

**Results at 1.3B params / 100B tokens:**
- 4.3–9.2% perplexity reduction vs. matched transformer
- +2.99 pts on CORE benchmarks, +1.18 on Core-Extended
- Matches transformers up to 2× its size (23.3–87.5% parameter efficiency)

**Scaling laws:** Optimal training increases looping depth and tokens together following power laws (γ_μ ≈ 0.40, γ_D ≈ 0.78). Test-time compute scales via saturating exponential: L(T) = L_∞ + Z·e^(−z·T).

**No HyperConnections.** Uses LTI system theory for analysis but implements stability via parameterization, not a learned gating mechanism.

**How CART differs from Parcae:**

| Aspect | Parcae | CART |
|--------|--------|------|
| Spectral stability | Architecturally constrained via negative diagonal + ZOH | Learned LTI gate — ρ emerges from training (~0.893 across all scales) |
| Recurrence depth | Variable per batch (Poisson-sampled) | Fixed R per config |
| Backpropagation | Truncated BPTT (⌈r/2⌉ steps) | Full backprop |
| HyperConnections | No | Yes (n_hyper=3) |
| KV compression | Standard injection | MLA (d_kv = d//4) |
| Cross-attention | Linear conditioning matrices W₁, W₂ | Full MLA cross-attention with causal masking |

**Differentiation angle for paper:** CART's LTI gate is a *learned* stability mechanism — the model discovers a convergence rate appropriate to its scale (ρ≈0.893, adapting upward with d_model). Parcae *imposes* stability via parameterization. CART's emergent ρ behavior (converging to the R=6 ideal regardless of actual R; drifting upward with scale) is an empirical finding about how recurrent language models learn memory timescales — not a design choice. Additionally, CART uses MLA compression and HyperConnections, neither of which Parcae employs.

---

### How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped LMs (arXiv 2604.21106)
**Schwethelm, Rückert, Kaissis — TU Munich / Imperial College London / MCML**

**Contribution:** A methodology for measuring the computational value of each recurrence, expressed as the recurrence-equivalence exponent φ. Defines a joint scaling law:

```
L(N_once, N_rec, D, r) = E + A(N_once + r^φ · N_rec)^(−α) + B D^(−β)
```

where N_rec is shared recurrent block parameters, N_once is prelude/coda parameters, r is loop count, D is training tokens. φ=1 would mean each loop equals a unique layer; φ=0 means no benefit from looping.

**Results across 116 runs (30M–1.1B params, up to 2.15×10¹⁹ FLOPs):**

| Configuration | φ | Notes |
|--------------|---|-------|
| Vanilla looped (baseline) | 0.46 (CI: 0.41–0.53) | Each loop ≈ 0.46 unique layers |
| + HyperConnections (K=2) | **0.65** | 41% improvement in loop value |
| + Truncated backprop | 0.38 | Saves ~30% training FLOPs but degrades loop quality |

**Practical consequence at φ=0.46:** A 410M looped model matches 580M non-looped performance but costs 1B FLOPs to train. At φ=0.65 (HyperConnections) this improves meaningfully.

**Architecture:** Decoder-only, 20 effective layers, RMSNorm, RoPE, FlashAttention-2/3, squared-ReLU MLPs. Training data: FineWeb-Edu, Llama 2 tokenizer (32K vocab), 2,049-token sequences — similar to CART's training setup.

**No LTI gating or spectral radius discussion.**

**What this means for CART:**

1. **HyperConnections are externally validated.** CART uses HyperConnections (n_hyper=3). This paper independently shows they raise φ from 0.46 → 0.65. That is the single largest architectural improvement they tested. Cite this as direct validation of CART's design choice.

2. **Full backprop is better than truncated.** CART uses full backprop. Truncated BPTT (used by Parcae) lowers φ to 0.38 — CART's training approach is justified.

3. **Compute φ for CART.** Their methodology can be applied to CART's Stage 2 results to produce a φ value directly comparable to their baseline (0.46) and HyperConnection result (0.65). If CART's LTI gate + HyperConnections + MLA together push φ above 0.65, that is a direct, quantified architectural contribution. This analysis should be included in the paper.

4. **Framing:** CART combines HyperConnections (φ boost confirmed externally) with a learned LTI stability mechanism (validated by Parcae's independent arrival at the same problem) and MLA compression (unique to CART). The combination is the contribution — not any single component.

### Will CART's φ be directly comparable to Schwethelm et al.'s values?

Partially — with caveats that must be stated explicitly in the paper.

**Where comparison is clean:** The framework applies and the spirit is the same — φ measures "how much does each additional loop contribute vs. a unique layer?" CART's R=6/8/10 variation across multiple d values gives exactly the data needed to fit it.

**Where direct comparison breaks down:**

**1. Fixed token budget.** Schwethelm et al. varied D across 116 runs to fit the full scaling law including the D^(−β) term. All CART Stage 2 runs are at a fixed ~1B tokens. We can estimate φ at our training budget, but we're measuring φ(D=1B), not the asymptotic φ they report.

**2. N_rec definition is ambiguous for CART.** Their N_rec = full self-attention block (Q, K, V, O, FFN). CART's CoreBlock only has Q, O, and FFN — K and V come from the prelude's KV projection, computed once and not scaling with R. Two defensible choices:
- N_rec = Q+O+FFN only (what the core actually computes per loop)
- N_rec includes an allocated share of the KV projection (principled but requires a choice)
The definition used must be stated; it directly affects the φ value.

**3. Different attention mechanism.** Their looped block is self-attention; CART's core is cross-attention to a fixed anchor. The φ framework was designed for self-attention loops. Cross-attention may produce a structurally different φ for reasons unrelated to the "value per loop" question.

**Honest framing for the paper:** Report φ using the Schwethelm et al. framework adapted for CART's architecture, state the N_rec definition explicitly, and compare directionally: "our φ = X vs. their vanilla baseline 0.46 and HyperConnection result 0.65." Do not claim exact equivalence. The most meaningful outcome: if CART's φ exceeds 0.65 (HyperConnections alone), that suggests the LTI gate and/or MLA are adding loop value beyond what HyperConnections provide — a direct quantified contribution.

---

---

### Hyperloop Transformers (arXiv 2604.21254)
**Zeitoun, Torroba-Hennigen, Kim — MIT**
**Submitted April 23, 2026**

**Architecture:** Three-zone structure with a fixed 25%/50%/25% parameter split:
- **Begin block** (~25%): standard Transformer layers, run once
- **Middle block** (~50%): standard Transformer layers, looped R times with shared weights
- **End block** (~25%): standard Transformer layers, run once

Concrete example: 136M model = `2L → 4L (×3) → 2L`. Primary experiments use R=3 loops. Ablation sweeps R=2 to R=6; performance peaks at R=3 and degrades past that (partly because the begin/end blocks shrink to maintain the 25/50/25 split as R grows).

**Hyper-connections:** Applied at loop boundaries only (not per-layer). Uses diagonal sigmoid projections H_pre, H_post, H_res on the hidden state — cheap (~200K extra params for 3 loop boundaries). Loop position embedding added after the middle block at each iteration.

**Attention in the middle block:** Standard self-attention over the current hidden state. **Not cross-attention.** No KV anchor, no MLA, no KV compression. KV is recomputed every loop iteration.

**Stability:** No spectral radius constraints, no LTI gating, no eigenvalue bounds. Stability is implicit via the sigmoid diagonal structure of H_res.

**Training:** FineWeb-Edu only. Seq_len=2048. Scales tested: 136M, 580M, 990M params. 12.5B–100B tokens depending on scale.

**Key results:**

| Scale | Model | Params | BF16 PPL |
|-------|-------|--------|----------|
| ~240M | Transformer | 238M | 14.65 |
| ~240M | Looped (no HC) | 135.5M | 14.85 |
| ~240M | **Hyperloop** | **135.7M** | **14.40** |
| ~1B | Transformer | 990.5M | 10.19 |
| ~1B | **Hyperloop** | **579.7M** | **9.65** |
| ~2B | Transformer | 2018M | 8.60 |
| ~2B | **Hyperloop** | **990.8M** | **8.49** |

Hyperloop at ~136M params beats the 238M Transformer. Gains persist through INT4 quantization. At 100B token overtraining, Hyperloop nearly matches full Transformer with 43% fewer parameters.

**Failure mode:** Vanilla looped Transformer (no HyperConnections) underperforms a full Transformer (14.85 vs 14.65). HyperConnections are the critical ingredient — without them, naive weight sharing hurts. This directly supports CART's design decision.

**How CART differs from Hyperloop:**

| Dimension | Hyperloop | CART |
|-----------|-----------|------|
| Loop attention | Self-attention (KV recomputed every loop) | Cross-attention to fixed KV anchor (computed once, reused) |
| KV compression | None | MLA (d_kv = d//4) |
| Stability mechanism | None (implicit sigmoid) | Learned LTI gate (guarantees SR < 1) |
| Spectral radius | Not measured/discussed | Empirically ~0.893 across scales |
| Hyper-connections | Loop-boundary diagonal sigmoid | Ring buffer over last 3 hidden states, softmax-weighted |
| Loop position signal | Learned loop position embedding | LIE (sinusoidal, added before core) |
| Causal masking | Standard self-attn (not discussed as an issue) | Explicitly is_causal=True required; non-causal causes collapse |
| Parameter split | Fixed 25/50/25 | P unique prelude layers / R×shared CoreBlock / 1 coda |
| Data | FineWeb-Edu only | FineWeb-Edu + TinyStories + Wikipedia |

**Differentiation angle for paper:** Hyperloop's middle block recomputes KV every loop — computationally equivalent to running R full Transformer layers with shared weights. CART's cross-attention design computes KV once from the prelude and reuses it, reducing per-loop FLOPs and creating an explicit **information bottleneck**: the recurrent core cannot attend to its own iterative states, only to the stable context anchor from the prelude. This separation of "understanding the context" (prelude) from "iterating on that context" (core) is architecturally distinct from Hyperloop's purely iterative self-attention. Additionally, CART adds an explicit stability guarantee (LTI gate, measured ρ≈0.893) that Hyperloop does not provide.

**What CART gains from the once-computed KV anchor (vs. Hyperloop's self-attention loop):**

**1. FLOPs savings per loop iteration.**
Hyperloop recomputes Q, K, V from the current hidden state h every loop — three full d×d projections per iteration. CART computes K and V once from the prelude before the loop starts and reuses them; each core iteration only needs a Q projection. At R=10, CART skips 2×R = 20 d×d matrix multiplications. MLA compression (d_kv = d//4) makes the once-computed K/V even cheaper. This is the primary reason CART's per-loop FLOPs are lower despite similar architectural depth.

**2. Clean separation of roles.**
In Hyperloop, the middle block self-attends to its own evolving hidden state — "what is the context?" and "what is my current reasoning state?" are the same thing, changing every loop. In CART the prelude output e is a fixed anchor: the prelude encodes the input context, the core iterates on that encoding. The recurrent state h is purely a reasoning variable; what it attends to never changes mid-loop. This is a cleaner computational model and a stronger architectural story for the paper.

**3. Stable attention target.**
Because K and V are frozen during the loop, every core iteration attends to the same representation of the input. In Hyperloop, later iterations attend to h that has already been transformed by earlier iterations — the "context" drifts with the reasoning state. CART's fixed anchor means the core always has a stable ground truth to query against, which may contribute to why the LTI gate converges to a consistent ρ≈0.893 across scales rather than requiring explicit stability constraints.

**Trade-off:** Hyperloop's self-attention can update its attention patterns per loop as h evolves — it can attend to different positions each iteration. CART's Q changes per loop (since Q comes from h) but K is fixed, so the "answers available" don't change, only what the core "asks for." Whether this is a limitation depends on the task.

---

### Loop, Think & Generalize (arXiv 2604.07822)
**Kohli, Parthasarathy, Sun, Yao — April 9, 2026**

Investigates recurrent-depth transformers on compositional generalization and depth extrapolation — tasks where vanilla transformers fail. Finds recurrent-depth models succeed via a sharp three-stage grokking process. Identifies "overthinking" as a failure mode when recurrence steps become excessive. Relevant for framing CART's fixed-R design as deliberate rather than a limitation.

---

### Relational Preference Encoding in Looped Transformer Internal States (arXiv 2604.09870)
**Kirin — April 10, 2026**

Interpretability study of the Ouro-2.6B looped LM. Trains linear probes on iterative hidden states to predict human preference pairs (Anthropic HH-RLHF), achieving 95.2% accuracy. Finds preferences are encoded relationally (pairwise differences) not independently. Shows looped transformer hidden states function as consistency probes. Peripheral to CART but citable for looped model interpretability.

---

### Hierarchical vs. Flat Iteration in Shared-Weight Transformers (arXiv 2604.14442)
**Han — April 15, 2026**

Compares flat iteration (Universal Transformer — same block applied uniformly) vs. hierarchical recurrence (distinct begin/end + looped middle) at 1.2B scale. Finds a sharp empirical gap favoring hierarchical recurrence. Directly supports CART's prelude/coda design — the architectural choice to have distinct unique layers at the boundaries, not just a uniform repeated block, is validated externally.

---

### The Recurrent Transformer (arXiv 2604.21215)
**Oncescu, Morwani, Jelassi, Meterez, Kwun, Kakade — April 23, 2026**

Each layer attends to KV pairs computed from its own activations (layerwise recurrent memory, no extra KV projection parameters). Proves theoretically the architecture emulates both standard Transformers and recurrent token-to-token updates. Introduces a tiling algorithm reducing memory bandwidth from quadratic to logarithmic in sequence length. Outperforms parameter-matched baselines on C4 at 150M and 300M params. Published same day as Hyperloop — the field is converging on this design space simultaneously.

---

### Do Transformers Use Depth Adaptively? (arXiv 2604.12426)
**Curth, Lawrence, Karmalkar, Prasad — April 14, 2026**

Studies whether pretrained transformers allocate layer compute adaptively. Finds pretrained models show only modest adaptive depth behavior; finetuned models show clearer evidence, especially when LM objectives are preserved. Useful for framing depth-recurrent architectures as providing structured adaptive compute that standard transformers lack by default.

---

### RD-ViT (arXiv 2605.03999)
**He — May 5, 2026**

Extends recurrent-depth transformer to vision semantic segmentation (cardiac MRI). Shared block looped T times with stable state injection and adaptive computation allocation. Matches standard ViT at half the parameters on full datasets; outperforms it with 10% training data. Shows the depth-recurrent paradigm transfers to dense prediction and low-data regimes. Citable for generality of the approach across modalities.

---

### Earlier work (just outside April 9 window, still highly relevant)

**Thinking Deeper, Not Longer (arXiv 2603.21676)** — Hung-Hsuan Chen, March 23, 2026
Iterates a shared-weight block in latent space ("vertical chain-of-thought") with LayerScale and identity-biased recurrence to stabilize 20+ steps. Finds a sharp "computational frontier" where performance transitions from chance to near-perfect as recurrence steps scale. Important for framing CART's recurrence as depth rather than token generation.

**Sparse Growing Transformer (arXiv 2603.23998)** — Yao Chen et al., March 25, 2026 (revised April 16)
Training framework that progressively extends recurrence from deep to shallow layers via selective attention looping on informative heads. Reduces the typical 16–20% overhead of block-level looping to 1–3%. Orthogonal to CART but citable for efficiency framing.

**LoopFormer (arXiv 2602.11451, ICLR 2026)** — Jeddi, Ciccone, Taati, February 2026
Budget-conditioned looped transformer with shortcut-consistency training — variable trajectory lengths where shorter loops produce informative intermediate representations. Strong on language modeling and reasoning under compute constraints. Published at ICLR 2026 — gives the research direction conference-level validation.

---

### TODO for paper
- [ ] Add all papers above to the Related Work / Prior Art section — prioritize Hyperloop (2604.21254) and Parcae (2604.12946) as direct competitors
- [ ] Compute φ for CART using Schwethelm et al.'s methodology once d=768 complete (minimum viable dataset); decide N_rec definition (Q+O+FFN only vs. including KV projection share); report as φ(D=1B) not asymptotic φ
- [ ] Frame CART's learned ρ as distinct from Parcae's constrained ρ — both solve the same instability problem differently
- [ ] Frame CART's KV-anchor cross-attention as distinct from Hyperloop's self-attention loop — the once-computed KV reuse is both a FLOPs argument and an architectural argument
- [ ] Check whether Parcae's variable-depth training (Poisson sampling) would apply to CART — could be a future direction
- [ ] Verify: does Parcae use causal cross-attention? Their injection mechanism (linear W₁, W₂) bypasses the issue CART encountered, but worth confirming

---

## How Significant Are Perplexity Improvements in AI Research?

**Typical thresholds in published work:**
- **1–3%** improvement over a strong baseline: solid, publishable result
- **5–10%**: strong result, often highlighted in the abstract
- **>10%**: headline-worthy, draws close reviewer scrutiny

By those standards, the early CART vs. Dense gaps (ppl_wiki −18%, ppl_edu −26% at step 17,500) look extraordinary. Do not report them that way without caveats.

**Why the current numbers require caution:**

1. **The Dense baseline is invalidated.** It trained at seq_len=512 with a recycled 100M token dataset. The corrected Dense rerun at seq_len=1024 with 1B fresh tokens will be better — potentially by a lot. The gap may shrink substantially.

2. **Perplexity gaps don't map linearly to benchmark gaps.** A 26% ppl improvement does not mean 26% better HellaSwag. Benchmark improvements at this model scale are typically single-digit percentage points even when perplexity differences are large.

3. **This is one mid-tier config, one seed.** R=6 P=4 is not the best CART configuration. Full sweep + 3 seeds needed before drawing conclusions.

4. **Compute normalization.** If CART and Dense process different effective information per step due to architectural differences, reviewers will require careful normalization. FLOPs-matched comparison (Dense 17L) is the cleanest answer to this.

**What would constitute genuinely significant results:**
- CART beating corrected Dense by **>5% on ppl_wiki/ppl_edu** after proper apples-to-apples training → strong perplexity story
- CART beating same-parameter Dense by **>3–5% on HellaSwag or LAMBADA** → publishable benchmark claim
- CART matching FLOPs-matched Dense 17L on benchmarks despite storing 1.7× fewer parameters → the efficiency story

**Bottom line for paper framing:** The perplexity numbers support the story but don't tell it. Benchmark results (HellaSwag, ARC-C, LAMBADA, PIQA) against a correctly trained Dense baseline are the actual evidence. Do not lead with perplexity percentages in the abstract or introduction.

---

## Single-Pass vs. Multi-Epoch Training for Language Models

**Modern consensus: single-pass training on large unique datasets is strongly preferred for language model pretraining.**

The Chinchilla paper (Hoffmann et al., 2022) established the key insight: for a fixed compute budget, training on more unique tokens once is always better than training on fewer tokens multiple times. Optimal ratio is ~20 tokens per stored parameter — all unique.

**Why repetition hurts language models:**
- The model memorizes specific token sequences rather than learning generalizable patterns
- Training loss continues falling but held-out perplexity stops improving or regresses
- Generalization to different text domains (ppl_wiki, ppl_edu) degrades noticeably
- The model overfits to corpus-specific quirks — sentence structures, topic distributions, specific phrases

**This was a contributing factor in the invalidated Dense v1 runs.** Those ran 61,000 steps on 100M tokens — ~10 passes through the same data. Perplexity numbers looked plausible but the model had partly memorized the training corpus, making cross-distribution results unreliable.

**Stage 2 is correct:** single pass through 1B unique tokens, no repetition. The plateau at the end of training is a data-supply ceiling (the model has seen all available unique data), not overfitting. These are distinct: overfitting degrades held-out metrics; a data ceiling just stops improving them.

**Contrast with image models:** Multiple epochs (sometimes 100+) are standard for image classification. Images have rich low-level variation (lighting, cropping, augmentation) that makes each pass informative. Text lacks this — a sentence seen twice is simply memorized more deeply, not learned differently.

**When repetition is acceptable for LMs:**
- Fine-tuning on small datasets (instruction tuning, RLHF) — multiple epochs are standard and necessary
- Datasets so large and diverse that any "repetition" is negligible in practice
- When unique data is genuinely exhausted and no alternative exists

**For the paper:** State single-pass training as a one-line methodological note — the AI audience knows Chinchilla. Don't explain it; just cite it and move on. Same for the data ceiling vs. overfitting distinction — one sentence is enough. Over-explaining basics signals unfamiliarity with the field.

---

## Training Plateau at 1B Tokens — Data Ceiling vs. Model Saturation

Models in Stage 2 show apparent convergence in the final ~15% of training steps. This is expected but the cause matters for paper framing.

**Why the plateau is partly an artifact:**

The cosine LR schedule decays to min_lr=3e-5 over 30,500 steps. By step ~26,000, the learning rate is very low and updates are tiny. Any cosine-scheduled run will appear to flatten near the end regardless of whether the model is truly converged. This is schedule-driven, not capacity-driven.

**Why 1B tokens may be a genuine data ceiling:**

Chinchilla scaling laws predict compute-optimal token count ≈ 20× stored parameters. At ~100–150M stored params (Stage 2 configs), Chinchilla-optimal is ~2–3B tokens. Stage 2 trains on ~1B tokens — roughly 33–50% of Chinchilla-optimal. The model should still be improving at this point in principle.

However, 300M of the 1B training tokens are TinyStories — a synthetic, repetitive, child-level dataset. Once the model learns TinyStories' structural patterns, that 30% of the corpus stops providing meaningful new signal. The effective diverse training data is closer to ~700M tokens.

**Implication:** The plateau is a **data-supply ceiling**, not model saturation. The model has not fully converged in the Chinchilla sense — it has run out of fresh data. Extended training on a new 1B token batch (particularly more Wikipedia and FineWeb-Edu) would likely continue improvement, with diminishing but non-zero returns.

**For the paper:**
- Do not claim the model is "fully trained" or "converged" at 1B tokens. Frame it as trained to Chinchilla-50% with a data-supply constraint.
- The gap between CART and Dense is measured at the same token budget, so the comparison is fair regardless of whether either is fully converged.
- A natural future direction: Stage 3 with 2–3B tokens from a richer dataset (reduce TinyStories proportion, add more FineWeb-Edu). TinyStories is the binding constraint — at 300M tokens of 517M total, it is nearly exhausted and should be reduced or replaced if training is extended.

---

## What Perplexity Measures (and Doesn't)

Perplexity is the exponentiated average cross-entropy loss: **ppl = exp(mean −log P(token))**.

Intuitively, it's the model's average "branching factor" at each token — how many equally-likely next tokens the model is effectively choosing between. ppl=3.40 means the model behaves as if it's picking from ~3.4 equally probable options at every step. A perfect model with complete knowledge of the text would have ppl=1.0.

**What it measures:**
- How well the model predicts held-out text from the same distribution it was trained on
- Sensitive to every token, not just hard ones
- A compression metric: lower ppl = better compressor of the text

**What it doesn't tell you:**
- Whether the model can reason or follow instructions — ppl_tiny says nothing about HellaSwag or ARC-C
- Anything about generalization to different distributions (ppl_tiny and ppl_wiki can diverge significantly)
- Whether the model is useful for any downstream task

**Why we still use it:**
- Clean, continuous signal that tracks training progress reliably
- Reproducible and cheap to compute
- Correlates with benchmark performance *within* an architecture family — if CART consistently beats Dense at the same ppl_tiny, something real is happening
- Cross-distribution comparison (ppl_tiny vs ppl_wiki vs ppl_edu) gives a rough sense of generalization beyond training distribution

**The limitation that matters most for this paper:** ppl_tiny=3.40 confirms the model learned TinyStories well. It does *not* confirm whether CART's recurrent structure gives it better long-range reasoning than a Dense model at the same perplexity. That's what LAMBADA and HellaSwag answer in Stage 2 evals. Do not use perplexity alone as the primary evidence for CART's architectural advantage — it supports the story but doesn't tell it.

---

## Stage 2 Early Results vs. Dense Baseline (UPDATE WHEN SWEEP COMPLETE)

*Snapshot at step 13,500 (44% through training). Dense numbers are from the invalidated v1 run (wrong seq_len/data — see notes/dense_baseline_v1_invalidated.md). Update this table with corrected Dense reruns and full CART sweep results when available.*

**CART d=1024 R=6 P=4 seed=137 vs. Dense d=1024 (invalidated reference):**

| Metric | CART step 13,500 (44%) | Dense final (61k steps, wrong settings) | Dense best (42.5k steps) |
|--------|----------------------|----------------------------------------|--------------------------|
| ppl_tiny | **3.40** | 3.44 | 3.37 |
| ppl_wiki | **26.24** | 31.08 | 29.46 |
| ppl_edu | **27.01** | 35.01 | 33.20 |

CART is already ahead on all three metrics at 44% through training, using a mid-tier config (R=6, P=4 — not the best). The ppl_wiki and ppl_edu gaps are the most meaningful: these are held-out distributions the model didn't train on, so they measure generalization. CART is 15–22% better than the Dense final on both out-of-distribution metrics.

**Caveats for paper use:**
- Dense reference numbers come from an invalidated run (seq_len=512, recycled 100M token dataset). The corrected Dense reruns will be better — do not cite these absolute numbers.
- This is one seed of one config. Wait for full sweep + 3 seeds before drawing conclusions.
- The gap will likely narrow once Dense is retrained correctly, but the ppl_wiki/ppl_edu advantage is large enough to expect it to survive.

**Update at step 16,000 (52.5% through training):**

| Metric | CART step 16,000 | Dense best (invalidated) | Gap |
|--------|-----------------|--------------------------|-----|
| ppl_tiny | 3.26 | 3.37 | CART −3.3% |
| ppl_wiki | 24.66 | 29.46 | CART −16.3% |
| ppl_edu | 25.26 | 33.20 | CART −23.9% |

The out-of-distribution scores are pulling away faster than ppl_tiny. ppl_wiki and ppl_edu are improving more steeply than ppl_tiny — the longer context window (1024 vs 512 tokens in the invalidated Dense run) and richer prelude representations appear to matter most for denser, more structured text. The cross-domain generalization gap is widening as training continues.

**What this suggests:** CART's recurrent structure appears to generalize better across text domains, not just fit the training distribution (ppl_tiny). This is consistent with the architecture's design: the fixed-point convergence over a rich prelude output should extract more robust representations than a single-pass dense forward. This framing is worth testing explicitly when full results are in.

---

## Stage 1 as a Predictive Screen — Does It Hold at Stage 2? (UPDATE WHEN SWEEP COMPLETE)

*Partial data as of 6/36 configs complete. d=256 R=6 and R=8 only. Update with full sweep results.*

**The finding:** Stage 1 (3,000 steps, ~49M tokens) rankings are predictive of Stage 2 (30,500 steps, ~1B tokens) rankings — a 10× compute gap. If this holds across all scales and R values, Stage 1 is validated as a reliable cheap hyperparameter screen.

**d=256 R=6 vs R=8 — Stage 1 vs Stage 2:**

| Config | Stage 1 ppl_wiki (step 3000) | Stage 2 ppl_tiny mean (3 seeds) |
|--------|-----------------------------|---------------------------------|
| d=256 R=6 P=6 | 184.93 | 4.34 |
| d=256 R=8 P=6 | 186.20 | 4.34 |

Stage 1 showed R=6 slightly better than R=8 at d=256 (R=8 regressed). Stage 2 confirms: R=6 and R=8 are statistically identical (mean difference = 0.00). The ranking held — d=256 does not benefit from higher R at either compute scale.

**Seed variance at d=256:** Extremely tight — ppl_tiny range 4.31–4.38 across all 6 configs (3 seeds × 2 R values). This gives high confidence that observed differences between configs are real signal, not noise.

**Why this matters for the paper:**

1. **Methodological contribution:** Stage 1 is not just a warm-up — it's a validated screening protocol. A researcher could run Stage 1 (cheap) to select configs, then commit to Stage 2 (expensive) only for the winners. This is practically useful and addresses the reviewer question "why not run Stage 2 on all configs?"

2. **Credibility of the sweep design:** The fact that Stage 2 configs were selected based on Stage 1 rankings could be questioned as cherry-picking. If Stage 1 rankings demonstrably predict Stage 2 rankings across all scales, the selection is justified and transparent.

3. **Framing suggestion:** Add a short section or table in the paper comparing Stage 1 and Stage 2 rankings across all scales. Show rank correlation (Spearman's ρ) between Stage 1 ppl at step 3000 and Stage 2 final ppl. A high correlation is the cleanest possible validation.

**TODO:** Once full sweep is complete, compute Spearman rank correlation between Stage 1 step-3000 ppl_wiki and Stage 2 final ppl_wiki across all 36 configs. Also check whether the P=6 dominance and scale-dependent R benefit observed in Stage 1 hold at Stage 2.

---

## Notes for Paper Framing

- **Architectural correctness:** The causal constraint on cross-attention in autoregressive recurrent models is non-obvious and worth stating explicitly. CART's cross-attention requires `is_causal=True` because prelude and core share the same token sequence. The non-causal version is a plausible mistake — documenting it and the fix strengthens the paper.

- **Stage 1 validity:** 3,000 steps on mixed data is the screen. Rankings stabilize by step 500–1000; gaps are consistent across eval checkpoints. Paper framing: Stage 1 is a hyperparameter screen, Stage 2 (3 seeds, longer) is the quality claim.

- **R-sensitivity and diminishing returns — confirmed across all 64 configs:** Higher R gives diminishing returns, a direct mathematical consequence of the spectral radius. With ρ ≈ 0.893, each loop contributes ~89% of the previous loop. The R benefit at P=6 grows strongly with scale: d=256: −0.25% (R regresses); d=512: +1.72%; d=768: +2.81%; d=1024: +5.24%. Larger models extract far more value from additional loops. Notably, d=256 peaks at R=6 P=6 (R=8 is 1.3 pts worse on ppl_wiki), while d=512/768/1024 all peak at R=8 P=6 — smaller models saturate earlier. The R×P crossover evolves with scale: at d=256/512, P=2 shows R actively *hurting*; at d=768, P=2 shows a U-shape (R=4/6 worse, R=8 barely recovering); at d=1024, R=8 is best at *every* P including P=2 (1.28% wiki gain). **The threshold scale for R to be universally beneficial (positive at all P values) falls between d=768 and d=1024.** At d=1024 the R benefit peaks at P=6 (5.24% wiki), with P=4 close behind (4.52%). The P=4 peak for R benefit seen at d=768 shifts toward P=6 at d=1024, consistent with richer preludes providing more to converge toward at larger scale.

- **Scale dominates hyperparameters:** The most striking finding from all 64 configs is that scale improvement swamps hyperparameter choice. d=1024 R=2 P=2 (ppl_wiki=113.06) beats d=768 R=8 P=6 (114.96) — the absolute weakest configuration at d=1024 beats the best configuration at d=768. d=1024 R=2 P=3 (109.64) beats d=768's best by a further 5 points. This is expected from scaling laws but it's a clean empirical demonstration. **For the paper:** frame Stage 1 results as "within a scale, P and R matter; across scales, d_model dominates." The efficiency claim is then: given a fixed parameter budget, does CART's architecture extract more quality per parameter than a dense transformer? That's the Stage 2 question.

- **P controls the fixed point; R controls convergence to it:** The fixed point h* that the recurrent core converges toward is determined entirely by what the cross-attention can read from e (the prelude output). A deeper prelude (higher P) produces a richer e → better fixed point. More loops (higher R) gets h closer to that fixed point. These are separable contributions: P sets the ceiling, R sets how close you get. **This framing is worth a paragraph in the architecture section.** It also explains why P dominates R in the sweep: the ceiling (P) matters more than how fast you reach it (R) at the loop counts we're testing.

- **Spectral radius creeps upward with scale — and this is meaningful:** Full data across all 64 configs: d=256 mean=0.8922, d=512 mean=0.8925, d=768 mean=0.8945, d=1024 mean=0.8964. The jump accelerates: d=256→512 (+0.0003), d=512→768 (+0.0020), d=768→1024 (+0.0019). This is not noise — it has a direct interpretation. Higher ρ means slower convergence to the fixed point. Larger models prefer slower convergence because their fixed points are richer and harder to reach: at d=1024, the prelude output e encodes more complex representations, so h needs more gradual refinement to fully integrate that information. The model discovers this on its own — it isn't told to use a higher ρ at larger scale. **Discuss in paper:** *the LTI mechanism adapts its convergence rate to the complexity of the representations, naturally allocating more refinement capacity where it's needed.* This also implies that R=8 leaves progressively more residual distance to the fixed point as scale increases — at ρ=0.8971 (d=1024 R=8 mean), ρ^8 = 0.441, meaning 44% of the initial residual remains after 8 loops vs. 41% at d=256 (ρ=0.8947, ρ^8 = 0.408) — which is why larger models extract more benefit from R.

- **The model learns a consistent spectral radius near the R=6 ideal:** For a given R, the ρ where ρ^R = 0.5 (half the residual survives) is ρ = 0.5^(1/R). For R=6 this gives ρ = 0.8909 — essentially exactly what all scales learn (~0.893). Crucially, ρ barely shifts with R: the R-vs-ρ drift at d=512 is only 0.0021 (R=2: 0.8912 mean, R=8: 0.8936 mean); at d=768: 0.0023 (R=2: 0.8933, R=8: 0.8956); at d=1024: 0.0016 (R=2: 0.8955, R=8: 0.8971). The model essentially discovers a single "universal memory timescale" near ρ ≈ 0.893 that it uses regardless of R, with only scale (d_model) having a meaningful influence. For R=8, this means the model falls 0.0183–0.0223 below the theoretical ideal of 0.9170 — it settles for a compromise rather than adapting to its specific R. **Discuss in paper: the LTI formulation lets the model discover a stable memory timescale without supervision. The timescale adapts to model complexity (d_model) but not strongly to loop count (R). This universality suggests the LTI gate learns a global convergence rate rather than a per-R optimal, which is likely a consequence of using shared core weights across all R iterations.**

- **Cross-scale comparison:** All configs use identical training setup (mixed data, 3000 steps, eval at 500/1000/1500/2000/2500/3000). ppl_tiny, ppl_wiki, and ppl_edu are all directly comparable across d values. Use step-3000 as the primary cross-scale comparison point.
