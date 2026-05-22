"""
One-off ablation: train CART with R=1 (single core iteration, no recurrence)
to determine whether the looped core is adding anything at the Stage 2 budget.

If R=1 matches CART R=6 (~tied final ppl), the recurrence machinery
(HyperConnection, LIE, LTI gate, plus the loop itself) is not extracting
additional capacity at d=1024 with 1B tokens — the loss attributed to
"frozen KV" or other architectural choices is not actually about recurrence.

Defaults: d=1024 P=6 seed=42 (the diagnostic corner). R is hardcoded to 1.

Inserts a fresh row with model_type='cart_r1' and hardware='ablation' so the
regular sweep orchestrator will not pick it up, then invokes train_one.py
with full Stage 2 settings. Results land in results.db keyed by the new
config_id and can be compared against the matching d/P/seed baseline at R=6
(config_id c2d449975b4e7d0f for d=1024 P=6 seed=42).

Usage:
    python train/run_r1_ablation.py
    python train/run_r1_ablation.py --d 1024 --P 6 --seed 42
    python train/run_r1_ablation.py --print-cmd-only
"""
import argparse
import hashlib
import sqlite3
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


MODEL_TYPE  = "cart_r1"
HARDWARE    = "ablation"      # orchestrator filters by hardware in {3050, 3090}
TOTAL_STEPS = 30_500          # Stage 2 budget — must match the R=6 baseline
R_VALUE     = 1               # the ablation


def config_id(d: int, R: int, P: int, seed: int) -> str:
    blob = f"r1_d{d}_R{R}_P{P}_s{seed}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def insert_config(db_path: str, d: int, R: int, P: int, seed: int) -> str:
    cid = config_id(d, R, P, seed)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    existing = conn.execute(
        "SELECT status FROM configs WHERE config_id = ?", (cid,)
    ).fetchone()
    if existing is None:
        conn.execute(
            """INSERT INTO configs
               (config_id, d_model, n_loops, n_prelude, seed, stage,
                status, hardware, model_type, created_at)
               VALUES (?, ?, ?, ?, ?, 2, 'pending', ?, ?, datetime('now'))""",
            (cid, d, R, P, seed, HARDWARE, MODEL_TYPE),
        )
        conn.commit()
        print(f"Registered new config: {cid}  "
              f"(d={d} R={R} P={P} seed={seed} model_type={MODEL_TYPE})")
    else:
        print(f"Config already exists: {cid}  status={existing[0]}")
    conn.close()
    return cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--P", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", default="results.db")
    ap.add_argument("--ckpt-interval", type=int, default=2500)
    ap.add_argument("--train-bin", default=str(_ROOT / "data" / "stage2" / "stage2_train.bin"),
                    help="Stage 2 mixed corpus. Must match the corpus the R=6 baseline used.")
    ap.add_argument("--print-cmd-only", action="store_true")
    args = ap.parse_args()

    if not Path(args.train_bin).exists():
        sys.exit(f"ERROR: train_bin not found: {args.train_bin}")

    cid = insert_config(args.db, args.d, R_VALUE, args.P, args.seed)

    cmd = [
        sys.executable, str(_ROOT / "train" / "train_one.py"),
        "--config-id", cid,
        "--db", args.db,
        "--max-steps", str(TOTAL_STEPS),
        "--seq-len", "1024",
        "--ckpt-interval", str(args.ckpt_interval),
        "--train-bin", args.train_bin,
    ]

    if args.print_cmd_only:
        print("\nTo launch manually:")
        print(" ".join(cmd))
        return

    print(f"\nLaunching: {' '.join(cmd)}\n")
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
