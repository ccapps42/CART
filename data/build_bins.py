"""
Run this script ONCE before starting any sweep.
Produces fixed .bin files consumed by all training runs.

Usage:
    # Full build (Stage 1 + Stage 2 data):
    python data/build_bins.py --output-dir data/

    # Stage 2 data only — safe to run while Stage 1 is active:
    python data/build_bins.py --stage2-only --stage2-out data/

Output (training bins in --output-dir, val bins in --output-dir/val/):
    data/tinystories_train.bin  — Stage 1 training data
    data/stage2_train.bin       — Stage 2 training blend (30/30/40, ~1B tokens)
    data/val/tinystories_val.bin
    data/val/wikipedia_val.bin  — Wikipedia shard 40 held out
    data/val/fineweb_edu_val.bin — FineWeb-Edu shard 97, seed 42

Tokenizer: NousResearch/Llama-2-7b-hf (public mirror, 32k BPE)
Note: spec named microsoft/phi-2 but phi-2 uses ~51k vocab. Llama-2
      tokenizer has exactly 32,000 tokens and is the correct match.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HF_CACHE_BASE = Path("K:/data/hf_cache")

TINYSTORIES_DIR = (HF_CACHE_BASE /
    "roneneldan___tiny_stories/default/0.0.0"
    "/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64")

WIKIPEDIA_DIR = (HF_CACHE_BASE /
    "wikimedia___wikipedia/20231101.en/0.0.0"
    "/b04c8d1ceb2f5cd4588862100d08de323dccfbaa")

FINEWEB_DIR = (HF_CACHE_BASE /
    "HuggingFaceFW___fineweb-edu/sample-10BT/0.0.0"
    "/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
TOKENIZER_NAME = "NousResearch/Llama-2-7b-hf"

# ---------------------------------------------------------------------------
# Token budget targets
# ---------------------------------------------------------------------------
TARGET_TRAIN_TOKENS = 100_000_000   # 100M tokens for tinystories_train.bin (Stage 1)
TARGET_VAL_TOKENS   =     500_000   # 500k tokens per val set
# Stage 2: 300M + 300M + 400M = ~1B tokens interleaved
STAGE2_TINY_TOKENS     = 300_000_000
STAGE2_WIKI_TOKENS     = 300_000_000
STAGE2_FINEWEB_TOKENS  = 400_000_000

# Wikipedia holdout: shard 40 (last shard, 28,288 docs)
WIKI_VAL_SHARD = 40
# FineWeb holdout: shard 97 (last shard), random seed
FINEWEB_VAL_SHARD = 97
FINEWEB_VAL_SEED  = 42

SEQ_LEN_INTERLEAVE = 1024  # chunk size for stage2 interleaving — must match training seq_len


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_texts(tokenizer, texts, max_tokens: int, desc: str) -> np.ndarray:
    """
    Tokenize an iterable of texts, prepend BOS + append EOS per doc.
    Returns flat uint16 array capped at max_tokens.
    """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    out = []
    total = 0
    for text in tqdm(texts, desc=desc, unit="docs", dynamic_ncols=True):
        if total >= max_tokens:
            break
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        doc = [bos] + ids + [eos]
        remaining = max_tokens - total
        out.append(np.array(doc[:remaining], dtype=np.uint16))
        total += len(doc)
        if total >= max_tokens:
            break
    if not out:
        return np.array([], dtype=np.uint16)
    return np.concatenate(out)


def write_bin(path: Path, tokens: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(np.uint16).tofile(str(path))
    mb = tokens.nbytes / 1e6
    print(f"  -> {path}  [{len(tokens):,} tokens, {mb:.1f} MB]")


def load_shard(shard_path: Path) -> Dataset:
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing shard: {shard_path}")
    return Dataset.from_file(str(shard_path))


def iter_shard_texts(shard_paths: list):
    """Yield texts from multiple HuggingFace arrow shards in order."""
    for path in shard_paths:
        shard = Dataset.from_file(str(path))
        yield from shard["text"]


def interleave_stage2(
    tiny: np.ndarray,
    wiki: np.ndarray,
    fineweb: np.ndarray,
    chunk: int = SEQ_LEN_INTERLEAVE,
) -> np.ndarray:
    """
    Interleave three token arrays in 30/30/40 proportions.
    Every 10 chunks: 3 tiny, 3 wiki, 4 fineweb.
    Stops when any source is exhausted.
    """
    # cycle: 0=tiny, 1=wiki, 2=fineweb
    schedule = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    sources = [tiny, wiki, fineweb]
    pos = [0, 0, 0]
    parts = []
    i = 0
    while True:
        src = schedule[i % 10]
        p = pos[src]
        if p + chunk > len(sources[src]):
            break
        parts.append(sources[src][p: p + chunk])
        pos[src] += chunk
        i += 1
    if not parts:
        return np.array([], dtype=np.uint16)
    result = np.concatenate(parts)
    print(f"  Stage2 total: {len(result):,} tokens  "
          f"({pos[0]:,} tiny / {pos[1]:,} wiki / {pos[2]:,} fineweb)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/",
                        help="Base output directory for full build (val/ subfolder created inside)")
    parser.add_argument("--stage2-only", action="store_true",
                        help="Build only stage2_train.bin; skip Stage 1 and val files")
    parser.add_argument("--stage2-out", default=None,
                        help="Directory for stage2_train.bin when using --stage2-only "
                             "(defaults to --output-dir if not set)")
    args = parser.parse_args()

    out_dir    = Path(args.output_dir)
    stage2_dir = Path(args.stage2_out) if args.stage2_out else out_dir
    val_dir    = out_dir / "val"

    if args.stage2_only:
        stage2_dir.mkdir(parents=True, exist_ok=True)
        print(f"Mode       : stage2-only")
        print(f"Stage2 out : {stage2_dir.resolve()}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir : {out_dir.resolve()}")
        print(f"Val dir    : {val_dir.resolve()}")

    # --- Tokenizer ---
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    assert tokenizer.vocab_size == 32_000, (
        f"vocab_size mismatch: expected 32000, got {tokenizer.vocab_size}")
    print(f"  vocab_size={tokenizer.vocab_size}  "
          f"bos={tokenizer.bos_token_id}  eos={tokenizer.eos_token_id}")

    if not args.stage2_only:
        # ==============================================================
        # 1. TinyStories — train
        # ==============================================================
        print(f"\n[1/5] TinyStories train  (encoding up to {STAGE2_TINY_TOKENS//1_000_000}M tokens "
              f"— first {TARGET_TRAIN_TOKENS//1_000_000}M written to tinystories_train.bin, "
              f"remainder used for stage2 blend)")
        ts_train_paths = sorted(TINYSTORIES_DIR.glob("tiny_stories-train-*.arrow"))
        ts_train = concatenate_datasets([load_shard(p) for p in ts_train_paths])
        print(f"  {len(ts_train):,} docs across {len(ts_train_paths)} shards")
        # Encode up to STAGE2_TINY_TOKENS so the full amount is available for the stage2 blend.
        # tinystories_train.bin (Stage 1) gets only the first TARGET_TRAIN_TOKENS (100M).
        ts_train_tokens = encode_texts(
            tokenizer, ts_train["text"], STAGE2_TINY_TOKENS, "TinyStories train")
        print(f"  Total TinyStories tokens encoded: {len(ts_train_tokens):,}")
        write_bin(out_dir / "tinystories_train.bin", ts_train_tokens[:TARGET_TRAIN_TOKENS])

        # ==============================================================
        # 2. TinyStories — val
        # ==============================================================
        print("\n[2/5] TinyStories val  (target: 500k tokens)")
        ts_val_path = TINYSTORIES_DIR / "tiny_stories-validation.arrow"
        ts_val = load_shard(ts_val_path)
        print(f"  {len(ts_val):,} val docs")
        ts_val_tokens = encode_texts(
            tokenizer, ts_val["text"], TARGET_VAL_TOKENS, "TinyStories val")
        write_bin(val_dir / "tinystories_val.bin", ts_val_tokens)

        # ==============================================================
        # 3. Wikipedia — val (shard 40 held out)
        # ==============================================================
        print(f"\n[3/5] Wikipedia val  (shard {WIKI_VAL_SHARD}, target: 500k tokens)")
        wiki_val_path = WIKIPEDIA_DIR / f"wikipedia-train-{WIKI_VAL_SHARD:05d}-of-00041.arrow"
        wiki_val_shard = load_shard(wiki_val_path)
        print(f"  Shard {WIKI_VAL_SHARD} has {len(wiki_val_shard):,} docs")
        wiki_val_tokens = encode_texts(
            tokenizer, wiki_val_shard["text"], TARGET_VAL_TOKENS, "Wikipedia val")
        sample_titles = [wiki_val_shard[i]["title"] for i in range(min(20, len(wiki_val_shard)))]
        with open(val_dir / "wikipedia_val_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "shard_index": WIKI_VAL_SHARD,
                "shard_file": wiki_val_path.name,
                "total_docs_in_shard": len(wiki_val_shard),
                "selection": "First N docs until 500k tokens reached",
                "first_20_titles": sample_titles,
                "tokenizer": TOKENIZER_NAME,
            }, f, indent=2, ensure_ascii=False)
        write_bin(val_dir / "wikipedia_val.bin", wiki_val_tokens)
        print(f"  Holdout metadata -> {val_dir / 'wikipedia_val_meta.json'}")

        # ==============================================================
        # 4. FineWeb-Edu — val (shard 97, seed 42, held out)
        # ==============================================================
        print(f"\n[4/5] FineWeb-Edu val  (shard {FINEWEB_VAL_SHARD}, seed={FINEWEB_VAL_SEED}, target: 500k tokens)")
        fineweb_val_path = FINEWEB_DIR / f"fineweb-edu-train-{FINEWEB_VAL_SHARD:05d}-of-00098.arrow"
        fineweb_val_shard = load_shard(fineweb_val_path)
        print(f"  Shard {FINEWEB_VAL_SHARD} has {len(fineweb_val_shard):,} docs")
        import random
        rng = random.Random(FINEWEB_VAL_SEED)
        indices = list(range(len(fineweb_val_shard)))
        rng.shuffle(indices)
        fw_val_texts = (fineweb_val_shard[i]["text"] for i in indices)
        fw_val_tokens = encode_texts(
            tokenizer, fw_val_texts, TARGET_VAL_TOKENS, "FineWeb-Edu val")
        with open(val_dir / "fineweb_edu_val_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "shard_index": FINEWEB_VAL_SHARD,
                "shard_file": fineweb_val_path.name,
                "total_docs_in_shard": len(fineweb_val_shard),
                "random_seed": FINEWEB_VAL_SEED,
                "selection": f"Docs shuffled with seed {FINEWEB_VAL_SEED}, first N until 500k tokens",
                "tokenizer": TOKENIZER_NAME,
            }, f, indent=2)
        write_bin(val_dir / "fineweb_edu_val.bin", fw_val_tokens)
        print(f"  Holdout metadata -> {val_dir / 'fineweb_edu_val_meta.json'}")

    # ==================================================================
    # 5. Stage 2 blend (30/30/40 interleaved, ~1B tokens)
    # ==================================================================
    step_label = "[1/1]" if args.stage2_only else "[5/5]"
    print(f"\n{step_label} Stage 2 training blend  "
          f"(30% tiny / 30% wiki / 40% fineweb, "
          f"target ~{(STAGE2_TINY_TOKENS + STAGE2_WIKI_TOKENS + STAGE2_FINEWEB_TOKENS) // 1_000_000}M tokens)")

    # TinyStories — encode fresh in stage2-only mode; reuse from step 1 otherwise
    if args.stage2_only:
        print(f"  Encoding TinyStories (up to {STAGE2_TINY_TOKENS//1_000_000}M tokens)...")
        ts_train_paths = sorted(TINYSTORIES_DIR.glob("tiny_stories-train-*.arrow"))
        ts_train = concatenate_datasets([load_shard(p) for p in ts_train_paths])
        ts_train_tokens = encode_texts(
            tokenizer, ts_train["text"], STAGE2_TINY_TOKENS, "TinyStories")
        print(f"  TinyStories tokens encoded: {len(ts_train_tokens):,}")
    else:
        print(f"  TinyStories: using {len(ts_train_tokens):,} tokens from step 1")
    stage2_tiny = ts_train_tokens

    # Wikipedia — shards 0 to (WIKI_VAL_SHARD - 1); val shard is held out
    wiki_train_shards = sorted(
        p for p in WIKIPEDIA_DIR.glob("wikipedia-train-*-of-00041.arrow")
        if p != WIKIPEDIA_DIR / f"wikipedia-train-{WIKI_VAL_SHARD:05d}-of-00041.arrow"
    )
    print(f"  Wikipedia: {len(wiki_train_shards)} train shards available "
          f"(shard {WIKI_VAL_SHARD} held out for val)")
    stage2_wiki = encode_texts(
        tokenizer, iter_shard_texts(wiki_train_shards),
        STAGE2_WIKI_TOKENS, "Wiki stage2")
    print(f"  Wikipedia tokens encoded: {len(stage2_wiki):,}")

    # FineWeb-Edu — shards 0 to (FINEWEB_VAL_SHARD - 1); val shard is held out
    fw_train_shards = sorted(
        p for p in FINEWEB_DIR.glob("fineweb-edu-train-*-of-00098.arrow")
        if p != FINEWEB_DIR / f"fineweb-edu-train-{FINEWEB_VAL_SHARD:05d}-of-00098.arrow"
    )
    print(f"  FineWeb-Edu: {len(fw_train_shards)} train shards available "
          f"(shard {FINEWEB_VAL_SHARD} held out for val)")
    stage2_fineweb = encode_texts(
        tokenizer, iter_shard_texts(fw_train_shards),
        STAGE2_FINEWEB_TOKENS, "FineWeb stage2")
    print(f"  FineWeb-Edu tokens encoded: {len(stage2_fineweb):,}")

    # Interleave
    print("  Interleaving in 30/30/40 chunks...")
    stage2_tokens = interleave_stage2(stage2_tiny, stage2_wiki, stage2_fineweb)
    write_bin(stage2_dir / "stage2_train.bin", stage2_tokens)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n=== Tokenization complete ===")
    if args.stage2_only:
        files = [stage2_dir / "stage2_train.bin"]
    else:
        files = [
            out_dir / "tinystories_train.bin",
            stage2_dir / "stage2_train.bin",
            val_dir / "tinystories_val.bin",
            val_dir / "wikipedia_val.bin",
            val_dir / "fineweb_edu_val.bin",
        ]
    for f in files:
        if f.exists():
            tokens = np.memmap(str(f), dtype=np.uint16, mode='r')
            print(f"  {f.name:<30s}  {len(tokens):>12,} tokens")
        else:
            print(f"  {f.name:<30s}  MISSING")

    print(f"\nTokenizer: {TOKENIZER_NAME}")
    print("Record this in sweep_meta table before running any sweep.")


if __name__ == "__main__":
    main()
