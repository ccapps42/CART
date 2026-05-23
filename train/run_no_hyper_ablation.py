"""
Ablation HC: train CART with HyperConnection disabled in the recurrent core.
Each iteration reads buffer[0] (most recent state) directly instead of the
softmax-weighted blend of the last 3 states.

Run against the SHARED-weight baseline (unshare_core=False), not the unshared
variant. Per the diagnostic plan, this tests whether HyperConnection earns
its keep in CART's native deployed configuration. The intervention is one
change vs the well-characterized 3-seed baseline (wiki 20.35 +/- 0.08), so
the result is interpretable on its own without confounding from other
ablations.

If removing HyperConnection improves perplexity, the heterogeneous residual
pattern (HC wrapping only the middle 6 layers, not prelude or coda) was a
real cost. If perplexity is unchanged, HyperConnection is vestigial in CART
(but not actively harmful). If perplexity gets worse, HyperConnection was
earning its keep in the shared-weight setting it was designed for.

Defaults: d=1024 R=6 P=6 seed=42 (the diagnostic corner). All Stage 2
hyperparameters identical to the baseline; only disable_hyper differs.

Inserts a fresh row with model_type='cart_no_hyper' and hardware='ablation'
so the regular sweep orchestrator will not pick it up.

Usage:
    python train/run_no_hyper_ablation.py
    python train/run_no_hyper_ablation.py --d 1024 --R 6 --P 6 --seed 42
    python train/run_no_hyper_ablation.py --print-cmd-only
"""
import argparse
import hashlib
import sqlite3
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


MODEL_TYPE  = "cart_no_hyper"
HARDWARE    = "ablation"     # orchestrator filters by hardware in {3050, 3090}
TOTAL_STEPS = 30_500         # Stage 2 budget — must match the baseline CART


def config_id(d: int, R: int, P: int, seed: int) -> str:
    blob = f"no_hyper_d{d}_R{R}_P{P}_s{seed}".encode()
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
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--P", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", default="results.db")
    ap.add_argument("--ckpt-interval", type=int, default=2500)
    ap.add_argument("--train-bin", default=str(_ROOT / "data" / "stage2" / "stage2_train.bin"),
                    help="Stage 2 mixed corpus. Must match the corpus the baseline used.")
    ap.add_argument("--print-cmd-only", action="store_true")
    args = ap.parse_args()

    if not Path(args.train_bin).exists():
        sys.exit(f"ERROR: train_bin not found: {args.train_bin}")

    cid = insert_config(args.db, args.d, args.R, args.P, args.seed)

    cmd = [
        sys.executable, str(_ROOT / "train" / "train_one.py"),
        "--config-id", cid,
        "--db", args.db,
        "--max-steps", str(TOTAL_STEPS),
        "--seq-len", "1024",
        "--disable-hyper",
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
