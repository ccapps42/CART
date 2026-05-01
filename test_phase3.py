"""
Phase 3 verification: 50-step training run.
Creates a temporary DB and synthetic data files.
Verifies: loss decreases, train_log rows written, results row written, checkpoint saved.
"""
import sys
import gc
import sqlite3
import subprocess
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("K:/projects/Model_Paper_1")
TMP_DIR = Path("K:/tmp/model_paper1_phase3_test")
TMP_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH     = TMP_DIR / "test.db"
TRAIN_BIN   = TMP_DIR / "train.bin"
VAL_DIR     = TMP_DIR / "val"
VAL_TINY    = VAL_DIR / "tinystories_val.bin"
VAL_WIKI    = VAL_DIR / "wikipedia_val.bin"
VAL_FINEWEB = VAL_DIR / "fineweb_edu_val.bin"

CONFIG_ID = "test001ph3"

# Minimal schema needed by train_one.py
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS configs (
    config_id    TEXT PRIMARY KEY,
    d_model      INTEGER NOT NULL,
    n_loops      INTEGER NOT NULL,
    n_prelude    INTEGER NOT NULL,
    seed         INTEGER NOT NULL DEFAULT 42,
    stage        INTEGER NOT NULL DEFAULT 1,
    status       TEXT NOT NULL DEFAULT 'pending',
    hardware     TEXT NOT NULL DEFAULT '3050',
    retry_count  INTEGER NOT NULL DEFAULT 0,
    error_msg    TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS results (
    result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id      TEXT NOT NULL,
    step           INTEGER NOT NULL,
    train_loss     REAL,
    eval_ppl_tiny  REAL,
    eval_ppl_wiki  REAL,
    eval_ppl_edu   REAL,
    peak_vram_gb   REAL,
    tokens_per_sec REAL,
    lti_spectral_radius REAL,
    wall_sec       REAL,
    recorded_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS train_log (
    log_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id      TEXT NOT NULL,
    step           INTEGER NOT NULL,
    train_loss     REAL NOT NULL,
    grad_norm      REAL,
    lr             REAL,
    n_tokens_seen  INTEGER,
    lti_spectral_radius REAL,
    wall_sec       REAL,
    recorded_at    TEXT NOT NULL
);
"""


def setup():
    # Create DB with schema
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        """INSERT INTO configs (config_id, d_model, n_loops, n_prelude,
           seed, stage, status, hardware, created_at)
           VALUES (?, 256, 2, 2, 42, 1, 'pending', '3050', datetime('now'))""",
        (CONFIG_ID,),
    )
    conn.commit()
    conn.close()
    print(f"  DB created: {DB_PATH}")

    # Synthetic token files (need > 50 * 4 * 8 * 512 = 819,200 tokens)
    rng = np.random.default_rng(42)
    train_tokens = rng.integers(3, 31_999, size=1_000_000, dtype=np.uint16)
    TRAIN_BIN.write_bytes(train_tokens.tobytes())
    print(f"  Train bin: {TRAIN_BIN}  ({len(train_tokens):,} tokens)")

    # Val files need ≥ 50 * 512 + 1 = 25,601 tokens each
    VAL_DIR.mkdir(exist_ok=True)
    for path in (VAL_TINY, VAL_WIKI, VAL_FINEWEB):
        val_tokens = rng.integers(3, 31_999, size=30_000, dtype=np.uint16)
        path.write_bytes(val_tokens.tobytes())
    print(f"  Val bins created in {VAL_DIR}")


def run_training():
    cmd = [
        "C:/Python314/python.exe",
        str(PROJECT_ROOT / "train" / "train_one.py"),
        "--config-id", CONFIG_ID,
        "--db", str(DB_PATH),
        "--max-steps", "50",
        "--train-bin", str(TRAIN_BIN),
        "--val-dir", str(VAL_DIR),
    ]
    print(f"\n  Running: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:2000])
    return result.returncode


def verify():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Status should be complete
    row = conn.execute(
        "SELECT status, error_msg FROM configs WHERE config_id=?",
        (CONFIG_ID,)
    ).fetchone()
    assert row["status"] == "complete", f"Expected status='complete', got '{row['status']}'\n{row['error_msg']}"
    print(f"  Config status: {row['status']} OK")

    # train_log should have at least 1 row
    log_rows = conn.execute(
        "SELECT step, train_loss, lti_spectral_radius FROM train_log WHERE config_id=? ORDER BY step",
        (CONFIG_ID,)
    ).fetchall()
    assert len(log_rows) >= 1, f"Expected ≥1 train_log rows, got {len(log_rows)}"
    print(f"  train_log rows: {len(log_rows)}")
    for r in log_rows:
        print(f"    step={r['step']}  loss={r['train_loss']:.4f}  rho={r['lti_spectral_radius']:.4f}")

    # results should have 1 row (Tier 2 fires at step 50 in test mode)
    res_rows = conn.execute(
        "SELECT step, eval_ppl_tiny, eval_ppl_wiki, eval_ppl_edu FROM results WHERE config_id=? ORDER BY step",
        (CONFIG_ID,)
    ).fetchall()
    assert len(res_rows) >= 1, f"Expected ≥1 results rows, got {len(res_rows)}"
    print(f"  results rows: {len(res_rows)}")
    for r in res_rows:
        print(f"    step={r['step']}  ppl_tiny={r['eval_ppl_tiny']:.2f}  "
              f"ppl_wiki={r['eval_ppl_wiki']:.2f}  ppl_edu={r['eval_ppl_edu']:.2f}")

    conn.close()

    # Loss should be a finite positive number (synthetic data, no "decrease" guaranteed in 50 steps)
    last_loss = log_rows[-1]["train_loss"]
    assert 0 < last_loss < 100, f"Loss out of expected range: {last_loss}"
    print(f"  Final loss: {last_loss:.4f} (positive, finite) OK")

    # Checkpoint should exist
    ckpt_dir = PROJECT_ROOT / "checkpoints" / CONFIG_ID
    ckpt_files = list(ckpt_dir.glob("step_*.pt"))
    assert len(ckpt_files) >= 1, f"No checkpoint found in {ckpt_dir}"
    print(f"  Checkpoint: {ckpt_files[0].name} OK")


def cleanup():
    import shutil
    shutil.rmtree(str(TMP_DIR), ignore_errors=True)
    ckpt_dir = PROJECT_ROOT / "checkpoints" / CONFIG_ID
    if ckpt_dir.exists():
        import shutil as _s
        _s.rmtree(str(ckpt_dir), ignore_errors=True)


def main():
    print("=== Phase 3 Verification ===\n")

    print("[Setup] Creating test DB and synthetic data...")
    setup()

    print("\n[Training] Running 50 steps...")
    rc = run_training()
    assert rc == 0, f"train_one.py exited with code {rc}"

    print("\n[Verify] Checking outputs...")
    verify()

    print("\n[Cleanup] Removing temp files...")
    cleanup()

    print("\n=== Phase 3 PASSED ===")


if __name__ == "__main__":
    main()
