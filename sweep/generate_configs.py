"""
Populates the configs table with all Stage 1 sweep runs.
Also populates sweep_meta with fixed hyperparameters.
Run once after schema creation.

Usage:
    python sweep/generate_configs.py --db results.db [--force]
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Sweep space
# ---------------------------------------------------------------------------
DIMS_3050 = [256, 512, 768]
DIMS_3090 = [1024]
LOOPS     = [2, 4, 6, 8]
PRELUDES  = [2, 3, 4, 6]
STAGE1_SEED = 42

# ---------------------------------------------------------------------------
# Fixed hyperparameters (source of truth for sweep_meta)
# ---------------------------------------------------------------------------
SWEEP_META = {
    "vocab_size":             "32000",
    "d_head":                 "64",
    "mla_compression_ratio":  "4",
    "n_hyper":                "3",
    "rope_base":              "10000.0",
    "ffn_mult":               "8/3",
    "seq_len_stage1":         "512",
    "seq_len_stage2":         "1024",
    "steps_stage1":           "3000",
    "steps_stage2":           "61000",     # ~1B tokens at 32,768 tokens/step (seq=1024)
    "tokenizer_name":         "NousResearch/Llama-2-7b-hf",
    "tokenizer_note":         ("Spec named microsoft/phi-2 but phi-2 vocab=51200. "
                               "Llama-2 tokenizer used instead (32000 vocab)."),
    "data_files":             json.dumps({
        "stage1_train":  "data/tinystories_train.bin",
        "stage2_train":  "data/stage2_train.bin",
        "val_tiny":      "data/val/tinystories_val.bin",
        "val_wiki":      "data/val/wikipedia_val.bin",
        "val_fineweb":   "data/val/fineweb_edu_val.bin",
    }),
    "peak_lr":                "3e-4",
    "min_lr":                 "3e-5",
    "warmup_steps":           "100",
    "weight_decay":           "0.1",
    "grad_clip":              "1.0",
    "batch_size":             "4",
    "grad_accum_steps":       "8",
    "effective_batch_tokens": str(4 * 8 * 512),
}


def config_id(d_model: int, n_loops: int, n_prelude: int,
              seed: int, stage: int) -> str:
    """sha256(d_model, n_loops, n_prelude, seed, stage)[:16]"""
    blob = f"{d_model}_{n_loops}_{n_prelude}_{seed}_{stage}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def apply_schema(conn):
    schema_path = Path(__file__).parent / "schema.sql"
    conn.executescript(schema_path.read_text())
    conn.commit()


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="results.db")
    parser.add_argument("--force", action="store_true",
                        help="Drop existing configs and regenerate")
    args = parser.parse_args()

    conn = open_db(args.db)
    apply_schema(conn)

    if args.force:
        conn.execute("DELETE FROM configs WHERE stage=1")
        conn.execute("DELETE FROM sweep_meta")
        conn.commit()
        print("Cleared existing Stage 1 configs and sweep_meta.")

    # Populate sweep_meta
    existing_meta = conn.execute("SELECT COUNT(*) FROM sweep_meta").fetchone()[0]
    if existing_meta == 0:
        for k, v in SWEEP_META.items():
            conn.execute(
                "INSERT OR REPLACE INTO sweep_meta (key, value) VALUES (?, ?)",
                (k, v)
            )
        conn.commit()
        print(f"Populated sweep_meta with {len(SWEEP_META)} entries.")

    # Build config list (ordered: d_model ASC, n_loops ASC, n_prelude ASC)
    configs = []
    for d in sorted(DIMS_3050 + DIMS_3090):
        hw = "3090" if d == 1024 else "3050"
        for r in sorted(LOOPS):
            for p in sorted(PRELUDES):
                cid = config_id(d, r, p, STAGE1_SEED, 1)
                configs.append((cid, d, r, p, STAGE1_SEED, 1, "pending", hw))

    assert len(configs) == 64, f"Expected 64 configs, got {len(configs)}"

    # Insert, skipping any already present
    inserted = 0
    for row in configs:
        try:
            conn.execute(
                """INSERT INTO configs
                   (config_id, d_model, n_loops, n_prelude, seed, stage,
                    status, hardware, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                row,
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # already exists
    conn.commit()

    print(f"Inserted {inserted} new configs ({len(configs) - inserted} already existed).")

    # Verify ordering
    rows = conn.execute(
        "SELECT config_id, d_model, n_loops, n_prelude, hardware "
        "FROM configs WHERE stage=1 ORDER BY d_model, n_loops, n_prelude"
    ).fetchall()

    print(f"\nTotal Stage 1 configs: {len(rows)}")
    print(f"{'config_id':18s}  {'d':>4}  {'R':>2}  {'P':>2}  hw")
    print("-" * 45)
    for r in rows:
        print(f"{r[0]}  {r[1]:4d}  {r[2]:2d}  {r[3]:2d}  {r[4]}")

    # Sanity checks
    d_vals = sorted(set(r[1] for r in rows))
    assert d_vals == [256, 512, 768, 1024], f"Unexpected d_model values: {d_vals}"
    r_vals = sorted(set(r[2] for r in rows))
    assert r_vals == [2, 4, 6, 8], f"Unexpected n_loops values: {r_vals}"
    p_vals = sorted(set(r[3] for r in rows))
    assert p_vals == [2, 3, 4, 6], f"Unexpected n_prelude values: {p_vals}"

    # Confirm ordering: d_model ASC, then n_loops ASC, then n_prelude ASC
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        curr = rows[i]
        ok = (curr[1], curr[2], curr[3]) >= (prev[1], prev[2], prev[3])
        assert ok, f"Ordering violation at row {i}: {prev} vs {curr}"

    print("\nAll ordering checks passed.")
    print(f"\nDB: {args.db}")
    print("Ready to run: python sweep/orchestrate.py --stage 1 --hardware 3050")

    conn.close()


if __name__ == "__main__":
    main()
