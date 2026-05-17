"""
Paper 1.b corpus downloads.

Downloads four corpora to the shared HF cache (K:/data/hf_cache/):
  1. Simple Wikipedia (Tier 2)                  ~300 MB     auto-cached by HF
  2. OpenMathInstruct-2 (Tier 7)                ~5-10 GB    auto-cached by HF
  3. MATH competition (Tier 7, via EleutherAI mirror, 7 configs)  ~50 MB  auto-cached by HF
  4. The Stack v2 Python+JS (Tier 6)            ~30 GB raw  streamed to JSONL shards

The first three use HF's automatic caching. The fourth is streamed and written
as JSONL shards under K:/data/paper_1b/stack_v2_py_js/ so we can cap the
download size and apply the comment-density filter at tier-build time.

Idempotent: HF datasets cache handles re-runs of the first three. The Stack
v2 download writes a checkpoint.json after every shard flush; on re-run it
loads the checkpoint and skips ahead in the stream to resume from where it
left off. Safe across wifi drops, crashes, and Ctrl+C.

Run order: any order, all sequential. Safe to Ctrl+C — written shards stay,
checkpoint reflects last flush, and re-running picks up cleanly.

Usage:
    python data/download_paper_1b_corpora.py
"""

import json
import os
import sys
from pathlib import Path

# Force HF cache to shared location before importing datasets
os.environ["HF_HOME"] = "K:/data/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "K:/data/hf_cache"

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STACK_DATASET_ID = "bigcode/the-stack-v2-dedup"
# Fallback if v2 fails: "bigcode/the-stack-dedup" (v1 dedup, same field shape)

STACK_OUTPUT_DIR = Path("K:/data/paper_1b/stack_v2_py_js")
STACK_LANGUAGES = {"Python", "JavaScript"}
STACK_TOKEN_TARGET = 3_000_000_000   # 3B raw tokens (filter down to ~1B commented at build time)
STACK_SHARD_SIZE   = 100_000          # docs per JSONL shard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Fast token-count proxy without running the tokenizer.
    Llama-2 BPE averages ~1.3 tokens per whitespace-split word for code/prose."""
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Individual downloads
# ---------------------------------------------------------------------------

def download_simple_wikipedia():
    print("\n" + "=" * 60)
    print("[1/4] Simple Wikipedia (~300 MB)")
    print("=" * 60)
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    print(f"  Done. {len(ds):,} docs.")


def download_open_math_instruct():
    print("\n" + "=" * 60)
    print("[2/4] OpenMathInstruct-2 (~5-10 GB)")
    print("=" * 60)
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    print(f"  Done. {len(ds):,} examples.")


def download_math_competition():
    print("\n" + "=" * 60)
    print("[3/4] MATH (Hendrycks competition math via EleutherAI mirror, ~50 MB)")
    print("=" * 60)
    # EleutherAI/hendrycks_math is parquet-based (works with current `datasets`
    # library); the old hendrycks/competition_math is script-based and no
    # longer auto-loads. The dataset has 7 per-subject configs that must be
    # loaded separately and concatenated.
    configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    parts = []
    for cfg in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        print(f"  {cfg:>28s}: {len(ds):>5,} problems")
        parts.append(ds)
    combined = concatenate_datasets(parts)
    print(f"  Done. {len(combined):,} problems total across {len(configs)} subjects.")


def _stack_checkpoint_path() -> Path:
    return STACK_OUTPUT_DIR / "checkpoint.json"


def _load_stack_checkpoint() -> dict:
    """Return checkpoint dict {total_rows_consumed, total_tokens_kept, shard_idx}
    or zeroed defaults if no checkpoint exists."""
    cp_path = _stack_checkpoint_path()
    if cp_path.exists():
        with open(cp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"total_rows_consumed": 0, "total_tokens_kept": 0, "shard_idx": 0}


def _save_stack_checkpoint(total_rows_consumed: int, total_tokens_kept: int, shard_idx: int):
    """Atomic write: temp file + rename, so an interrupted save can't corrupt."""
    cp_path = _stack_checkpoint_path()
    tmp_path = cp_path.with_suffix(".json.tmp")
    payload = {
        "total_rows_consumed": total_rows_consumed,
        "total_tokens_kept":   total_tokens_kept,
        "shard_idx":           shard_idx,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(cp_path)


def download_stack_v2_python_js():
    print("\n" + "=" * 60)
    print(f"[4/4] The Stack v2 (Python+JS), cap {STACK_TOKEN_TARGET / 1e9:.1f}B raw tokens")
    print("=" * 60)

    STACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Resume from checkpoint if it exists ----
    cp = _load_stack_checkpoint()
    rows_skip_target  = cp["total_rows_consumed"]
    total_tokens_kept = cp["total_tokens_kept"]
    shard_idx         = cp["shard_idx"]

    if rows_skip_target > 0:
        print(f"  Resuming from checkpoint:")
        print(f"    rows already consumed: {rows_skip_target:,}")
        print(f"    tokens already kept:   {total_tokens_kept:,}")
        print(f"    next shard index:      {shard_idx}")
        if total_tokens_kept >= STACK_TOKEN_TARGET:
            print(f"  Token target already reached. Done.")
            return

    print(f"  Streaming {STACK_DATASET_ID}, filtering to {sorted(STACK_LANGUAGES)}...")
    try:
        ds = load_dataset(STACK_DATASET_ID, split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR loading {STACK_DATASET_ID}: {e}")
        print(f"  Possible fixes:")
        print(f"    - Accept dataset terms at https://huggingface.co/datasets/{STACK_DATASET_ID}")
        print(f"    - Confirm HF token is set (huggingface-cli whoami)")
        print(f"    - Fall back to v1: edit STACK_DATASET_ID = 'bigcode/the-stack-dedup'")
        raise

    # Fast-forward the stream past already-consumed rows. HF's ds.skip(N) still
    # walks N rows internally; for large N this re-downloads (no efficient
    # mid-stream seek), but the on-disk shards are preserved so we never write
    # duplicates and the final token target is reached eventually.
    if rows_skip_target > 0:
        print(f"  Skipping forward {rows_skip_target:,} rows in the stream...")
        ds = ds.skip(rows_skip_target)

    shard_buf = []
    total_rows_consumed = rows_skip_target

    pbar = tqdm(
        total=STACK_TOKEN_TARGET, initial=total_tokens_kept,
        unit="tok", unit_scale=True,
        desc="Stack v2 tokens", dynamic_ncols=True,
    )

    def flush_shard():
        """Write the in-memory buffer to a new shard file, then update
        the checkpoint. Both file write and checkpoint write are atomic."""
        nonlocal shard_idx, shard_buf
        if not shard_buf:
            return
        shard_path     = STACK_OUTPUT_DIR / f"shard_{shard_idx:05d}.jsonl"
        shard_tmp_path = shard_path.with_suffix(".jsonl.tmp")
        with open(shard_tmp_path, "w", encoding="utf-8") as f:
            for r in shard_buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shard_tmp_path.replace(shard_path)
        shard_idx += 1
        shard_buf = []
        _save_stack_checkpoint(total_rows_consumed, total_tokens_kept, shard_idx)

    try:
        for row in ds:
            total_rows_consumed += 1
            lang = row.get("language")
            if lang not in STACK_LANGUAGES:
                continue
            text = row.get("content") or row.get("text")
            if not text:
                continue

            n_tok = approx_tokens(text)
            shard_buf.append({
                "language": lang,
                "content": text,
                "approx_tokens": n_tok,
            })
            total_tokens_kept += n_tok
            pbar.update(n_tok)

            if len(shard_buf) >= STACK_SHARD_SIZE:
                flush_shard()

            if total_tokens_kept >= STACK_TOKEN_TARGET:
                break
    finally:
        flush_shard()
        # Even if buf was empty, checkpoint should reflect latest total_rows_consumed
        _save_stack_checkpoint(total_rows_consumed, total_tokens_kept, shard_idx)
        pbar.close()

    print(f"  Done. {total_tokens_kept:,} approx tokens across {shard_idx} shards in {STACK_OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"HF cache:           {os.environ['HF_DATASETS_CACHE']}")
    print(f"Stack v2 output:    {STACK_OUTPUT_DIR}")
    print(f"Stack v2 dataset:   {STACK_DATASET_ID}")
    print(f"Stack v2 token cap: {STACK_TOKEN_TARGET / 1e9:.1f}B")

    steps = [
        ("Simple Wikipedia",     download_simple_wikipedia),
        ("OpenMathInstruct-2",   download_open_math_instruct),
        ("MATH (EleutherAI)",    download_math_competition),
        ("The Stack v2 (Py+JS)", download_stack_v2_python_js),
    ]

    failures = []
    for name, fn in steps:
        try:
            fn()
        except KeyboardInterrupt:
            print(f"\n  Interrupted during {name}. Partial output preserved.")
            raise
        except Exception as e:
            print(f"\nERROR downloading {name}: {e}")
            failures.append((name, str(e)))

    print("\n" + "=" * 60)
    print("Download summary")
    print("=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} of {len(steps)} datasets")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print(f"All {len(steps)} datasets downloaded successfully.")


if __name__ == "__main__":
    main()
