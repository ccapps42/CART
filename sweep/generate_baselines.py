"""
Inserts DenseBaseline configs into results.db — one per d_model scale.
Also adds model_type column to configs table if not present.

Dense baselines use:
    n_loops  = 7   (stored as n_layers — 7 uniform layers)
    n_prelude = 0  (sentinel — no prelude concept in DenseBaseline)
    model_type = 'dense'

Usage:
    python sweep/generate_baselines.py --db results.db
    python sweep/generate_baselines.py --db results.db --force
"""
import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DENSE_DIMS    = [256, 512, 768, 1024]
N_LAYERS      = 7
BASELINE_SEED = 42


def config_id(d_model: int, n_layers: int, seed: int) -> str:
    blob = f"dense_{d_model}_{n_layers}_{seed}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


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
    parser.add_argument("--force", action="store_true",
                        help="Delete existing dense baseline configs and re-insert")
    args = parser.parse_args()

    conn = open_db(args.db)
    ensure_model_type_column(conn)

    if args.force:
        conn.execute("DELETE FROM configs WHERE model_type='dense'")
        conn.commit()
        print("Cleared existing dense baseline configs.")

    configs = []
    for d in DENSE_DIMS:
        hw = "3090" if d == 1024 else "3050"
        cid = config_id(d, N_LAYERS, BASELINE_SEED)
        configs.append({
            "config_id": cid,
            "d_model":   d,
            "n_loops":   N_LAYERS,   # stores n_layers for DenseBaseline
            "n_prelude": 0,          # sentinel — no prelude in DenseBaseline
            "seed":      BASELINE_SEED,
            "stage":     2,          # baselines run at Stage 2 scale (61k steps)
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
        "SELECT config_id, d_model, n_loops, hardware, status "
        "FROM configs WHERE model_type='dense' ORDER BY d_model"
    ).fetchall()

    print(f"\n{'config_id':18s}  {'d':>4}  {'layers':>6}  {'hw':>4}  status")
    print("-" * 50)
    for r in rows:
        print(f"{r['config_id']}  {r['d_model']:4d}  {r['n_loops']:6d}  "
              f"{r['hardware']:>4}  {r['status']}")

    print(f"\nDB: {args.db}")
    print("Run baselines with:")
    for r in rows:
        print(f"  python train/train_dense.py --config-id {r['config_id']} --db {args.db}")

    conn.close()


if __name__ == "__main__":
    main()
