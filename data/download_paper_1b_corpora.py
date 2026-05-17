"""
Paper 1.b corpus downloads.

Downloads four corpora to the shared HF cache (K:/data/hf_cache/):
  1. Simple Wikipedia (Tier 2)                  ~300 MB     auto-cached by HF
  2. OpenMathInstruct-2 (Tier 7)                ~5-10 GB    auto-cached by HF
  3. MATH competition (Tier 7)                  ~10 MB      auto-cached by HF
  4. The Stack v2 Python+JS (Tier 6)            ~30 GB raw  streamed to JSONL shards

The first three use HF's automatic caching. The fourth is streamed and written
as JSONL shards under K:/data/paper_1b/stack_v2_py_js/ so we can cap the
download size and apply the comment-density filter at tier-build time.

Idempotent: HF datasets cache handles re-runs of the first three; the Stack
output dir is skipped if already populated.

Run order: any order, all sequential. Safe to Ctrl+C — written shards stay,
only unwritten buffer is lost (~100K rows).

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

from datasets import load_dataset
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
    print("[3/4] MATH (Hendrycks competition_math, ~10 MB)")
    print("=" * 60)
    ds = load_dataset("hendrycks/competition_math", split="train")
    print(f"  Done. {len(ds):,} problems.")


def download_stack_v2_python_js():
    print("\n" + "=" * 60)
    print(f"[4/4] The Stack v2 (Python+JS), cap {STACK_TOKEN_TARGET / 1e9:.1f}B raw tokens")
    print("=" * 60)

    if STACK_OUTPUT_DIR.exists() and any(STACK_OUTPUT_DIR.glob("*.jsonl")):
        n_shards = len(list(STACK_OUTPUT_DIR.glob("*.jsonl")))
        print(f"  Output dir already populated: {STACK_OUTPUT_DIR} ({n_shards} shards)")
        print("  Skipping. Delete the dir manually to re-download.")
        return

    STACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    total_tokens = 0
    shard_idx = 0
    shard_buf = []

    pbar = tqdm(total=STACK_TOKEN_TARGET, unit="tok", unit_scale=True,
                desc="Stack v2 tokens", dynamic_ncols=True)

    def flush_shard():
        nonlocal shard_idx, shard_buf
        if not shard_buf:
            return
        shard_path = STACK_OUTPUT_DIR / f"shard_{shard_idx:05d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for r in shard_buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shard_idx += 1
        shard_buf = []

    try:
        for row in ds:
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
            total_tokens += n_tok
            pbar.update(n_tok)

            if len(shard_buf) >= STACK_SHARD_SIZE:
                flush_shard()

            if total_tokens >= STACK_TOKEN_TARGET:
                break
    finally:
        flush_shard()
        pbar.close()

    print(f"  Done. {total_tokens:,} approx tokens across {shard_idx} shards in {STACK_OUTPUT_DIR}")


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
        ("MATH competition",     download_math_competition),
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
