"""
Inserts DenseBaseline configs into results.db. Configurable across d_model
scales, n_layers, and seeds, so the same script generates both
stored-matched (e.g., 7-layer) and effective-matched (e.g., 12-layer) Dense
baselines for comparison against CART.

Dense baselines use:
    n_loops    = n_layers  (DenseBaseline reuses this column to store layer count)
    n_prelude  = 0          (sentinel — no prelude in DenseBaseline)
    model_type = 'dense'

config_id is sha256("dense_{d_model}_{n_layers}_{seed}")[:16], so different
(d_model, n_layers, seed) tuples each get unique IDs and coexist in the DB.

Usage:
    # Default: 7L Dense at all four scales, single seed (matches old behavior)
    python sweep/generate_baselines.py --db results.db

    # Both stored-matched and effective-matched Dense at d=1024, three seeds:
    python sweep/generate_baselines.py --db results.db \\
        --scales 1024 --n-layers 7,12 --seeds 42,137,271

    # Delete existing dense configs first (use carefully):
    python sweep/generate_baselines.py --db results.db --force
"""
import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_SCALES    = [256, 512, 768, 1024]
DEFAULT_N_LAYERS  = [7]
DEFAULT_SEEDS     = [42]


def config_id(d_model: int, n_layers: int, seed: int) -> str:
    blob = f"dense_{d_model}_{n_layers}_{seed}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def parse_int_list(s: str) -> list:
    return [int(x) for x in s.split(",")]


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_model_type_column(conn):
    """Add model_type column to configs if it doesn't already exist."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(configs)").fetchall()]
    if "model_type" not in cols:
        conn.execute("ALTER TABLE configs ADD COLUMN model_type TEXT NOT NULL DEFAULT 'cart'")
        conn.commit()
        print("Added model_type column to configs table.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="results.db")
    parser.add_argument("--scales", type=parse_int_list, default=DEFAULT_SCALES,
                        help="Comma-separated d_model values (default: 256,512,768,1024)")
    parser.add_argument("--n-layers", type=parse_int_list, default=DEFAULT_N_LAYERS,
                        help="Comma-separated layer counts (default: 7)")
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS,
                        help="Comma-separated seeds (default: 42)")
    parser.add_argument("--force", action="store_true",
                        help="Delete ALL existing dense baseline configs and re-insert")
    args = parser.parse_args()

    conn = open_db(args.db)
    ensure_model_type_column(conn)

    if args.force:
        conn.execute("DELETE FROM configs WHERE model_type='dense'")
        conn.commit()
        print("Cleared existing dense baseline configs.")

    configs = []
    for d in args.scales:
        hw = "3090" if d == 1024 else "3050"
        for n_layers in args.n_layers:
            for seed in args.seeds:
                cid = config_id(d, n_layers, seed)
                configs.append({
                    "config_id": cid,
                    "d_model":   d,
                    "n_loops":   n_layers,   # stores n_layers for DenseBaseline
                    "n_prelude": 0,          # sentinel — no prelude in DenseBaseline
                    "seed":      seed,
                    "stage":     2,          # baselines run at Stage 2 scale
                    "hardware":  hw,
                    "model_type": "dense",
                })

    inserted = 0
    for cfg in configs:
        try:
            conn.execute(
                """INSERT INTO configs
                   (config_id, d_model, n_loops, n_prelude, seed, stage,
                    status, hardware, model_type, created_at)
                   VALUES (:config_id, :d_model, :n_loops, :n_prelude, :seed, :stage,
                           'pending', :hardware, :model_type, datetime('now'))""",
                cfg,
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()

    print(f"Inserted {inserted} new dense baseline configs "
          f"({len(configs) - inserted} already existed).")

    rows = conn.execute(
        "SELECT config_id, d_model, n_loops, seed, hardware, status "
        "FROM configs WHERE model_type='dense' ORDER BY d_model, n_loops, seed"
    ).fetchall()

    print(f"\n{'config_id':18s}  {'d':>4}  {'layers':>6}  {'seed':>4}  {'hw':>4}  status")
    print("-" * 60)
    for r in rows:
        print(f"{r['config_id']}  {r['d_model']:4d}  {r['n_loops']:6d}  "
              f"{r['seed']:4d}  {r['hardware']:>4}  {r['status']}")

    print(f"\nDB: {args.db}")
    print("Run baselines with:")
    for r in rows:
        print(f"  python train/train_dense.py --config-id {r['config_id']} --db {args.db}")

    conn.close()


if __name__ == "__main__":
    main()
