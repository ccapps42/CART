# Dense Baseline Run v1 — Invalidated Results

**Status: INVALIDATED — do not use in paper**

These runs used incorrect training settings and are stored for reference only.
All 4 configs will be rerun under corrected settings.

## Why Invalidated

| Issue | v1 (wrong) | v2 (correct) |
|-------|-----------|--------------|
| SEQ_LEN | 512 | 1024 |
| Training data | `data/stage2_train.bin` (100M tokens) | `data/stage2/stage2_train.bin` (1B tokens) |
| Total steps | 61,000 | 30,500 |
| Tokens/step | 16,384 | 32,768 |
| Total tokens | ~1B (cycling 100M dataset 10×) | ~1B (single pass through 1B dataset) |

## Final-Step Results (step 61,000 except d=768)

| d_model | config_id | step | ppl_tiny | ppl_wiki | ppl_edu | tok/s | VRAM |
|---------|-----------|------|----------|----------|---------|-------|------|
| 256 | 3d04ff17b81e48d4 | 61,000 | 4.033 | 38.406 | 41.873 | 120,515 | 1.20 GB |
| 512 | 360b726d0568fa76 | 61,000 | 3.412 | 29.960 | 32.956 | 85,398 | 1.75 GB |
| 768 | 4fb2e27f81dc94d5 | 11,000 | 4.110 | 37.465 | 42.401 | 21,031 | 2.38 GB |
| 1024 | 6feffa5e66763f35 | 61,000 | 3.438 | 31.076 | 35.014 | 42,931 | 3.18 GB |

*d=768 was stopped at step 11,000 — run had already been reset once due to a CUDA crash at step 14,000 in an earlier attempt.*

## Best ppl_wiki Achieved (any step)

| d_model | best step | ppl_tiny | ppl_wiki | ppl_edu |
|---------|-----------|----------|----------|---------|
| 256 | 61,000 | 4.033 | 38.406 | 41.873 |
| 512 | 61,000 | 3.412 | 29.960 | 32.956 |
| 768 | 11,000 | 4.110 | 37.465 | 42.401 |
| 1024 | 42,500 | 3.368 | 29.460 | 33.200 |

## Training Timeline

| d_model | started | completed | duration |
|---------|---------|-----------|----------|
| 256 | 2026-05-06 18:51 | 2026-05-06 21:12 | ~2.4 hrs |
| 512 | 2026-05-06 21:14 | 2026-05-07 00:28 | ~3.2 hrs |
| 768 | 2026-05-07 11:42 | stopped ~step 11k | ~1.3 hrs (partial) |
| 1024 | 2026-05-07 01:56 | 2026-05-07 08:32 | ~6.6 hrs |

## Notes

- RTX 3090, bfloat16, AdamW8bit, PEAK_LR=3e-4, WARMUP=100 steps, cosine decay to 3e-5
- d=256 and d=1024 show similar final ppl_wiki (~29–31) — d=256 is likely undertrained relative to its capacity at only 100M unique tokens
- d=512 achieved the best ppl_wiki (29.96) possibly due to better capacity/data ratio at this scale
- These numbers are not meaningful for the paper comparison since CART Stage 2 would train on different data at a different seq_len
