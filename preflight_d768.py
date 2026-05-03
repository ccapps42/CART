"""
Pre-flight check and cleanup for d=768 sweep restart.
Run this once before starting orchestrate.
"""
import sqlite3
import os
from pathlib import Path

ROOT = Path(__file__).parent
conn = sqlite3.connect("results.db")

print("=== Checking all configs are pending and DB is clean ===")
for d in [256, 512, 768]:
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM configs WHERE d_model=? GROUP BY status", (d,)
    ).fetchall()
    non_pending = [(r[0], r[1]) for r in rows if r[0] != "pending"]
    if non_pending:
        print(f"  d={d}: WARNING non-pending configs: {non_pending}")
    else:
        n = sum(r[1] for r in rows)
        print(f"  d={d}: {n} configs all pending OK")

n_results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
n_logs    = conn.execute("SELECT COUNT(*) FROM train_log").fetchone()[0]
assert n_results == 0 and n_logs == 0, f"Leftover rows: results={n_results} logs={n_logs}"
print(f"  No leftover results or train_log rows.")

conn.close()

print("\n=== Checking data files ===")
files = {
    "stage2_train.bin (mixed training data)": ROOT / "data" / "stage2_train.bin",
    "tinystories_val.bin":                    ROOT / "data" / "val" / "tinystories_val.bin",
    "wikipedia_val.bin":                      ROOT / "data" / "val" / "wikipedia_val.bin",
    "fineweb_edu_val.bin":                    ROOT / "data" / "val" / "fineweb_edu_val.bin",
}
all_ok = True
for label, path in files.items():
    if path.exists():
        mb = path.stat().st_size / 1e6
        print(f"  OK  {label}: {mb:.1f} MB")
    else:
        print(f"  MISSING  {label}: {path}")
        all_ok = False

assert all_ok, "One or more required data files are missing."

print("\n=== Checking train_one.py settings ===")
import re
src = (ROOT / "train" / "train_one.py").read_text()
total = int(re.search(r"TOTAL_STEPS\s*=\s*(\d+)", src).group(1))
eval_interval = int(re.search(r"EVAL_INTERVAL\s*=\s*(\d+)", src).group(1))
has_ckpt = "enable_gradient_checkpointing()" in src

print(f"  TOTAL_STEPS   = {total}   {'OK' if total == 3000 else 'WRONG — expected 3000'}")
print(f"  EVAL_INTERVAL = {eval_interval}   {'OK' if eval_interval == 500 else 'WRONG — expected 500'}")
print(f"  Gradient checkpointing call present: {has_ckpt}   {'WRONG — should be removed' if has_ckpt else 'OK (disabled)'}")

print("\n=== Checking orchestrate.py routes d=768 to stage2_train.bin ===")
orch_src = (ROOT / "sweep" / "orchestrate.py").read_text()
has_routing = "stage2_train.bin" in orch_src and "d_model" in orch_src
print(f"  Mixed-data routing present: {has_routing}   {'OK' if has_routing else 'WRONG — fix orchestrate.py'}")

print("\n=== Checking MLACrossAttention is causal ===")
attn_src = (ROOT / "model" / "attention.py").read_text()
has_causal = "is_causal=True" in attn_src
has_noncausal_cross = "Non-causal cross-attention" in attn_src
print(f"  is_causal=True present: {has_causal}   {'OK' if has_causal else 'WRONG — fix attention.py'}")
print(f"  Old non-causal comment removed: {not has_noncausal_cross}   {'OK' if not has_noncausal_cross else 'WARNING — stale comment present (harmless but confusing)'}")

print("\n=== All checks passed. Ready to run: ===")
print("  python sweep/orchestrate.py --stage 1 --hardware 3050")
