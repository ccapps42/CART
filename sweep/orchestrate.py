"""
Runs pending configs sequentially for a given stage and hardware target.

Usage:
    python sweep/orchestrate.py --stage 1 --hardware 3050
    python sweep/orchestrate.py --stage 2 --hardware 3090 --ckpt-interval 5000

    # Testing (short runs):
    python sweep/orchestrate.py --stage 1 --hardware 3050 --max-configs 2 --max-steps 50
"""
import argparse
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
TRAIN_ONE   = _ROOT / "train" / "train_one.py"
TRAIN_DENSE = _ROOT / "train" / "train_dense.py"
PYTHON = sys.executable

MAX_RETRIES        = 3
RETRY_DELAY_S      = 30
STAGE1_TOTAL_STEPS = 3_000
STAGE2_TOTAL_STEPS = 30_500   # ~1B tokens at 32,768 tokens/step (seq_len=1024)


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def get_pending(conn, stage: int, hardware: str):
    return conn.execute(
        """SELECT config_id, d_model, n_loops, n_prelude, seed, model_type
           FROM configs
           WHERE stage=? AND hardware=? AND status='pending'
           ORDER BY model_type, d_model, n_loops, n_prelude, seed""",
        (stage, hardware),
    ).fetchall()


def median_wall_sec(conn, stage: int, hardware: str) -> float | None:
    rows = conn.execute(
        """SELECT CAST(
               (julianday(completed_at) - julianday(started_at)) * 86400
           AS REAL) AS wall_sec
           FROM configs
           WHERE stage=? AND hardware=? AND status='complete'
             AND started_at IS NOT NULL AND completed_at IS NOT NULL""",
        (stage, hardware),
    ).fetchall()
    if not rows:
        return None
    vals = sorted(r[0] for r in rows if r[0] is not None and r[0] > 0)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def run_one(config_id: str, db_path: str, max_steps: int | None,
            train_bin: str | None, val_dir: str | None,
            ckpt_interval: int | None, seq_len: int | None,
            model_type: str = "cart") -> int:
    # Dispatch by model_type: CART uses train_one.py, Dense uses train_dense.py.
    # train_dense.py has SEQ_LEN hardcoded (1024) so --seq-len is silently dropped.
    script = TRAIN_DENSE if model_type == "dense" else TRAIN_ONE
    cmd = [
        PYTHON, str(script),
        "--config-id", config_id,
        "--db", db_path,
    ]
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    if seq_len and model_type != "dense":
        cmd += ["--seq-len", str(seq_len)]
    if ckpt_interval:
        cmd += ["--ckpt-interval", str(ckpt_interval)]
    if train_bin:
        cmd += ["--train-bin", train_bin]
    if val_dir:
        cmd += ["--val-dir", val_dir]
    result = subprocess.run(cmd)
    return result.returncode


def reset_to_pending(conn, config_id: str):
    conn.execute(
        "UPDATE configs SET status='pending' WHERE config_id=?",
        (config_id,)
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",    type=int, required=True, choices=[1, 2])
    parser.add_argument("--hardware", required=True, choices=["3050", "3090"])
    parser.add_argument("--db", default="results.db")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Stop after this many configs (for testing only)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override steps per config. Default: 3000 (stage 1), "
                             "61000 (stage 2). Use for testing or custom runs.")
    parser.add_argument("--ckpt-interval", type=int, default=None,
                        help="Save intermediate checkpoints every N steps. "
                             "Recommended for Stage 2 long runs (e.g. --ckpt-interval 5000).")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override sequence length. Default: 512 (stage 1), 1024 (stage 2).")
    parser.add_argument("--train-bin", default=None,
                        help="Override training .bin path (for testing only)")
    parser.add_argument("--val-dir", default=None,
                        help="Override val/ directory (for testing only)")
    args = parser.parse_args()

    conn = open_db(args.db)
    pending = get_pending(conn, args.stage, args.hardware)

    if not pending:
        print(f"No pending configs for stage={args.stage} hardware={args.hardware}.")
        conn.close()
        return

    total = len(pending)
    if args.max_configs:
        pending = pending[: args.max_configs]

    print(f"Stage {args.stage}  hardware={args.hardware}  "
          f"pending={len(pending)} of {total}")
    print()

    completed = 0
    failed    = 0

    for i, row in enumerate(pending):
        cid = row["config_id"]
        mtype = row["model_type"] if "model_type" in row.keys() else "cart"
        if mtype == "dense":
            print(f"[{i+1}/{len(pending)}] config_id={cid}  "
                  f"Dense d={row['d_model']}  layers={row['n_loops']}  seed={row['seed']}")
        else:
            print(f"[{i+1}/{len(pending)}] config_id={cid}  "
                  f"d={row['d_model']}  R={row['n_loops']}  P={row['n_prelude']}  seed={row['seed']}")

        # All configs use mixed data for cross-scale comparability
        if args.train_bin:
            effective_train_bin = args.train_bin
        elif args.stage == 2:
            effective_train_bin = str(_ROOT / "data" / "stage2" / "stage2_train.bin")
        else:
            effective_train_bin = str(_ROOT / "data" / "stage2_train.bin")

        if args.max_steps:
            effective_max_steps = args.max_steps
        elif args.stage == 2:
            effective_max_steps = STAGE2_TOTAL_STEPS
        else:
            effective_max_steps = STAGE1_TOTAL_STEPS

        if args.seq_len:
            effective_seq_len = args.seq_len
        elif args.stage == 2:
            effective_seq_len = 1024
        else:
            effective_seq_len = None  # train_one.py default (512)

        success = False
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                print(f"  Retry {attempt}/{MAX_RETRIES-1} after {RETRY_DELAY_S}s ...")
                time.sleep(RETRY_DELAY_S)
                reset_to_pending(conn, cid)

            rc = run_one(cid, args.db, effective_max_steps, effective_train_bin, args.val_dir,
                        args.ckpt_interval, effective_seq_len, model_type=mtype)

            if rc == 0:
                success = True
                completed += 1
                break
            else:
                print(f"  FAILED (exit code {rc}, attempt {attempt+1}/{MAX_RETRIES})")

        if not success:
            failed += 1
            print(f"  Giving up on {cid} after {MAX_RETRIES} attempts.")

        # ETA estimate based on completed runs
        med = median_wall_sec(conn, args.stage, args.hardware)
        remaining = len(pending) - (i + 1)
        if med and remaining > 0:
            eta_min = med * remaining / 60
            print(f"  Median wall: {med:.0f}s/run  ETA: ~{eta_min:.0f} min for {remaining} remaining")
        print()

    conn.close()
    print(f"Done.  completed={completed}  failed={failed}  "
          f"skipped={len(pending) - completed - failed}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
