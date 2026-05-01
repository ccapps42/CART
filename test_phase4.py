"""
Phase 4 verification:
1. Confirms 64 configs in results.db with correct ordering.
2. Runs orchestrate.py for 2 configs with --max-steps 50 and synthetic data.
3. Verifies both configs complete successfully.
"""
import sqlite3
import subprocess
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("K:/projects/Model_Paper_1")
DB_PATH = PROJECT_ROOT / "results.db"
TMP_DIR = Path("K:/tmp/model_paper1_phase4_test")
TMP_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_BIN   = TMP_DIR / "train.bin"
VAL_DIR     = TMP_DIR / "val"
VAL_TINY    = VAL_DIR / "tinystories_val.bin"
VAL_WIKI    = VAL_DIR / "wikipedia_val.bin"
VAL_FINEWEB = VAL_DIR / "fineweb_edu_val.bin"


def setup_synthetic_data():
    rng = np.random.default_rng(1234)
    TRAIN_BIN.write_bytes(rng.integers(3, 31999, size=1_000_000, dtype=np.uint16).tobytes())
    VAL_DIR.mkdir(exist_ok=True)
    for path in (VAL_TINY, VAL_WIKI, VAL_FINEWEB):
        path.write_bytes(rng.integers(3, 31999, size=30_000, dtype=np.uint16).tobytes())
    print(f"  Synthetic data in {TMP_DIR}")


def verify_configs():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT config_id, d_model, n_loops, n_prelude, hardware "
        "FROM configs WHERE stage=1 ORDER BY d_model, n_loops, n_prelude"
    ).fetchall()
    assert len(rows) == 64, f"Expected 64 configs, got {len(rows)}"
    print(f"  Config count: {len(rows)} (expected 64)")

    # Check ordering
    for i in range(1, len(rows)):
        p, c = rows[i-1], rows[i]
        assert (c["d_model"], c["n_loops"], c["n_prelude"]) >= (p["d_model"], p["n_loops"], p["n_prelude"])

    # Check hardware assignment
    for r in rows:
        if r["d_model"] == 1024:
            assert r["hardware"] == "3090", f"d=1024 should be 3090, got {r['hardware']}"
        else:
            assert r["hardware"] == "3050", f"d<1024 should be 3050, got {r['hardware']}"

    # First config should be d=256, R=2, P=2
    first = rows[0]
    assert first["d_model"] == 256 and first["n_loops"] == 2 and first["n_prelude"] == 2
    print(f"  First config: d={first['d_model']} R={first['n_loops']} P={first['n_prelude']} hw={first['hardware']}")

    # Last config should be d=1024, R=8, P=6
    last = rows[-1]
    assert last["d_model"] == 1024 and last["n_loops"] == 8 and last["n_prelude"] == 6
    print(f"  Last config:  d={last['d_model']} R={last['n_loops']} P={last['n_prelude']} hw={last['hardware']}")

    conn.close()


def run_orchestrator():
    cmd = [
        "C:/Python314/python.exe",
        str(PROJECT_ROOT / "sweep" / "orchestrate.py"),
        "--stage", "1",
        "--hardware", "3050",
        "--db", str(DB_PATH),
        "--max-configs", "2",
        "--max-steps", "50",
        "--train-bin", str(TRAIN_BIN),
        "--val-dir", str(VAL_DIR),
    ]
    print(f"  Running orchestrator for 2 configs...")
    result = subprocess.run(cmd, capture_output=False, timeout=300)
    return result.returncode


def verify_completed():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    completed = conn.execute(
        "SELECT config_id, d_model, n_loops, n_prelude, status "
        "FROM configs WHERE stage=1 AND status='complete' "
        "ORDER BY d_model, n_loops, n_prelude"
    ).fetchall()

    assert len(completed) >= 2, f"Expected >=2 completed configs, got {len(completed)}"
    print(f"  Completed configs: {len(completed)}")
    for r in completed[:4]:
        print(f"    {r['config_id']}  d={r['d_model']} R={r['n_loops']} P={r['n_prelude']}  {r['status']}")

    # Verify results table has entries
    res_count = conn.execute(
        "SELECT COUNT(*) FROM results"
    ).fetchone()[0]
    log_count = conn.execute(
        "SELECT COUNT(*) FROM train_log"
    ).fetchone()[0]
    assert res_count >= 2, f"Expected >=2 results rows, got {res_count}"
    assert log_count >= 2, f"Expected >=2 train_log rows, got {log_count}"
    print(f"  results rows: {res_count}  train_log rows: {log_count}")

    conn.close()


def cleanup():
    import shutil
    shutil.rmtree(str(TMP_DIR), ignore_errors=True)


def main():
    print("=== Phase 4 Verification ===\n")

    print("[1] Verifying config count and ordering...")
    verify_configs()
    print("    Config checks passed.")

    print("\n[2] Setting up synthetic data...")
    setup_synthetic_data()

    print("\n[3] Running orchestrate.py for 2 configs...")
    rc = run_orchestrator()
    assert rc == 0, f"orchestrate.py exited with code {rc}"

    print("\n[4] Verifying completed configs and DB entries...")
    verify_completed()

    print("\n[5] Cleanup...")
    cleanup()

    print("\n=== Phase 4 PASSED ===")


if __name__ == "__main__":
    main()
