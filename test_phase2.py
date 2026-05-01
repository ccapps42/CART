"""
Phase 2 verification: dataset loading, tokenizer, FixedOrderDataset.
Does NOT run full tokenization (that's a one-time ~30 min operation).
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "K:/projects/Model_Paper_1")

from datasets import Dataset
from transformers import AutoTokenizer
from data.dataset import FixedOrderDataset

TOKENIZER_NAME = "NousResearch/Llama-2-7b-hf"

TINYSTORIES_DIR = Path("K:/projects/OpenHobbs/data/hf_cache/roneneldan___tiny_stories"
                       "/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64")
FINEWEB_DIR = Path("K:/projects/OpenHobbs/data/hf_cache/HuggingFaceFW___fineweb-edu"
                   "/sample-10BT/0.0.0/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")


def test_tokenizer():
    print("[1] Tokenizer loading and encoding")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=False)
    assert tok.vocab_size == 32_000, f"Expected 32000, got {tok.vocab_size}"
    ids = tok.encode("Once upon a time", add_special_tokens=False)
    assert all(0 <= i < 32_000 for i in ids), "Token IDs out of range"
    print(f"   vocab_size={tok.vocab_size}  bos={tok.bos_token_id}  eos={tok.eos_token_id}")
    print(f"   'Once upon a time' -> {ids}  (len={len(ids)})")
    return tok


def test_arrow_loading():
    print("[2] Dataset.from_file on TinyStories val shard")
    path = TINYSTORIES_DIR / "tiny_stories-validation.arrow"
    assert path.exists(), f"Missing: {path}"
    ds = Dataset.from_file(str(path))
    assert "text" in ds.column_names, f"No 'text' column: {ds.column_names}"
    print(f"   Loaded {len(ds):,} docs, columns: {ds.column_names}")
    print(f"   First doc preview: {ds[0]['text'][:80]!r}...")

    print("[3] Dataset.from_file on FineWeb-Edu shard 0")
    fw_path = FINEWEB_DIR / "fineweb-edu-train-00000-of-00098.arrow"
    assert fw_path.exists(), f"Missing: {fw_path}"
    fw = Dataset.from_file(str(fw_path))
    assert "text" in fw.column_names, f"No 'text' column: {fw.column_names}"
    print(f"   FineWeb shard 0: {len(fw):,} docs, columns: {fw.column_names}")
    print(f"   First doc preview: {fw[0]['text'][:80]!r}...")


def test_fixed_order_dataset(tokenizer):
    print("[4] FixedOrderDataset with synthetic .bin file")
    # Create a small synthetic token array (seq_len=8, 10 sequences = 81 tokens)
    tmp_path = Path("K:/tmp/test_tokens.bin")
    tmp_path.parent.mkdir(exist_ok=True)
    n_tokens = 8 * 10 + 1  # 81 tokens
    tokens = np.arange(n_tokens, dtype=np.uint16) % 32_000
    tokens.tofile(str(tmp_path))

    ds = FixedOrderDataset(tmp_path, seq_len=8)
    assert len(ds) == 10, f"Expected 10 sequences, got {len(ds)}"
    x, y = ds[0]
    assert x.shape == (8,), f"Expected x shape (8,), got {x.shape}"
    assert y.shape == (8,), f"Expected y shape (8,), got {y.shape}"
    # y should be x shifted by 1
    assert torch.equal(y, x + 1), "y != x+1 (shift-by-1 not working)"
    x4, y4 = ds[4]
    assert x4[0].item() == 32, f"Sequence 4 should start at token 32, got {x4[0].item()}"
    print(f"   {len(ds)} sequences, shape x={tuple(x.shape)} y={tuple(y.shape)}")
    print(f"   seq[0]: x[0]={x[0].item()} ... y[-1]={y[-1].item()}")
    print(f"   seq[4]: x[0]={x4[0].item()}  (expected 32)")
    # Close memmap before deleting (Windows holds file locks)
    del ds
    import gc; gc.collect()
    tmp_path.unlink()


def main():
    print("=== Phase 2 Verification ===\n")
    tok = test_tokenizer()
    print()
    test_arrow_loading()
    print()
    test_fixed_order_dataset(tok)
    print("\n=== Phase 2 PASSED ===")


if __name__ == "__main__":
    main()
