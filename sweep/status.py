"""
Sweep progress report. Shows current state, measured rates, and time estimates.

Usage:
    python sweep/status.py
    python sweep/status.py --db results.db
"""
import argparse
import math
import sqlite3
from pathlib import Path

_ROOT = Path(__file__).parent.parent
TOTAL_STEPS = 30_500


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def get_measured_rates(conn) -> dict:
    """Return {(d_model, n_loops): sec_per_step} from completed/running/reference configs."""
    rates = {}
    rows = conn.execute("""
        SELECT c.config_id, c.d_model, c.n_loops
        FROM configs c
        WHERE c.model_type='cart' AND c.stage=2
        AND c.status IN ('running', 'complete', 'reference')
    """).fetchall()

    for row in rows:
        pts = conn.execute(
            "SELECT step, wall_sec FROM results WHERE config_id=? ORDER BY step",
            (row["config_id"],)
        ).fetchall()
        if len(pts) < 1:
            continue
        if len(pts) >= 2:
            rate = (pts[-1]["wall_sec"] - pts[0]["wall_sec"]) / (pts[-1]["step"] - pts[0]["step"])
        else:
            rate = pts[0]["wall_sec"] / pts[0]["step"] if pts[0]["step"] > 0 else None
        if rate:
            key = (row["d_model"], row["n_loops"])
            if key not in rates:
                rates[key] = []
            rates[key].append(rate)

    return {k: sum(v) / len(v) for k, v in rates.items()}


def estimate_rate(d, R, measured: dict) -> float | None:
    """Return sec/step estimate using measured data or d-scaling."""
    if (d, R) in measured:
        return measured[(d, R)]
    if not measured:
        return None

    # Derive empirical d-scaling exponent if we have two points at the same R
    d_exp = 2.0  # theoretical default
    by_R = {}
    for (md, mR), rate in measured.items():
        by_R.setdefault(mR, []).append((md, rate))
    for pts in by_R.values():
        if len(pts) >= 2:
            pts.sort()
            d1, r1 = pts[0]; d2, r2 = pts[-1]
            if d1 != d2:
                d_exp = math.log(r2 / r1) / math.log(d2 / d1)
            break

    # Scale from closest measured point using empirical exponent
    (md, mR), base = min(measured.items(), key=lambda kv: abs(math.log(d / kv[0][0])))
    return base * (d / md) ** d_exp * (R / mR)


def print_status(db_path: str):
    conn = open_db(db_path)

    # --- Overall counts ---
    print("=" * 60)
    print("CART STAGE 2 SWEEP STATUS")
    print("=" * 60)

    counts = {}
    for status in ["pending", "running", "complete", "failed", "reference"]:
        n = conn.execute(
            "SELECT COUNT(*) FROM configs WHERE model_type='cart' AND stage=2 AND status=?",
            (status,)
        ).fetchone()[0]
        counts[status] = n

    total = counts["pending"] + counts["running"] + counts["complete"] + counts["failed"]
    done = counts["complete"]
    print(f"\nProgress: {done}/{total} complete  |  "
          f"running={counts['running']}  pending={counts['pending']}  "
          f"failed={counts['failed']}  reference={counts['reference']} (excluded)")

    # --- Active configs ---
    active = conn.execute("""
        SELECT c.config_id, c.d_model, c.n_loops, c.n_prelude, c.seed,
               c.status, MAX(r.step) as latest_step, MAX(r.wall_sec) as wall_sec,
               MAX(r.tokens_per_sec) as tps,
               (SELECT r2.eval_ppl_tiny FROM results r2
                WHERE r2.config_id=c.config_id ORDER BY r2.step DESC LIMIT 1) as ppl_tiny
        FROM configs c
        LEFT JOIN results r ON r.config_id = c.config_id
        WHERE c.model_type='cart' AND c.stage=2
        AND c.status IN ('running', 'complete')
        GROUP BY c.config_id
        ORDER BY c.started_at DESC
        LIMIT 10
    """).fetchall()

    if active:
        print(f"\n--- Recent / Active " + "-" * 40)
        print(f"  {'d':>5} {'R':>3} {'P':>3} {'seed':>5}  {'status':>8}  "
              f"{'step':>7}  {'%':>5}  {'tok/s':>7}  {'ppl_tiny':>9}")
        for r in active:
            step = r["latest_step"] or 0
            tps  = r["tps"] or 0
            ppl  = f"{r['ppl_tiny']:.2f}" if r["ppl_tiny"] else "  ---"
            pct  = 100 * step / TOTAL_STEPS
            print(f"  {r['d_model']:>5} {r['n_loops']:>3} {r['n_prelude']:>3} {r['seed']:>5}  "
                  f"{r['status']:>8}  {step:>7,}  {pct:>5.1f}%  {tps:>7.0f}  {ppl:>9}")

    # --- Measured rates ---
    measured = get_measured_rates(conn)

    if measured:
        print(f"\n--- Measured rates " + "-" * 41)
        for (d, R), rate in sorted(measured.items()):
            hrs = rate * TOTAL_STEPS / 3600
            print(f"  d={d:4d} R={R:2d}: {rate:.4f} sec/step  =>  {hrs:.1f} hrs/config")

    # --- Per-config estimates ---
    print(f"\n--- Estimated hours per config " + "-" * 29)
    print(f"  {'d':>5}  {'R=6':>6}  {'R=8':>6}  {'R=10':>6}  {'remaining hrs':>14}")

    # Steps already completed on running configs (to avoid double-counting)
    running_steps_done = {}
    for row in conn.execute("""
        SELECT c.d_model, c.n_loops, MAX(r.step) as steps_done
        FROM configs c JOIN results r ON r.config_id = c.config_id
        WHERE c.model_type='cart' AND c.stage=2 AND c.status='running'
        GROUP BY c.config_id
    """).fetchall():
        key = (row["d_model"], row["n_loops"])
        running_steps_done[key] = running_steps_done.get(key, 0) + (row["steps_done"] or 0)

    # Count remaining (pending + running) per (d, R)
    remaining_counts = {}
    for row in conn.execute("""
        SELECT d_model, n_loops, COUNT(*) as n FROM configs
        WHERE model_type='cart' AND stage=2 AND status IN ('pending','running')
        GROUP BY d_model, n_loops
    """).fetchall():
        remaining_counts[(row["d_model"], row["n_loops"])] = row["n"]

    total_remaining_hrs = 0
    for d in [256, 512, 768, 1024]:
        cells = []
        sub = 0.0
        for R in [6, 8, 10]:
            rate = estimate_rate(d, R, measured)
            if rate:
                h = rate * TOTAL_STEPS / 3600
                n = remaining_counts.get((d, R), 0)
                cells.append(f"{h:6.1f}")
                # Subtract already-completed steps on running configs
                steps_done = running_steps_done.get((d, R), 0)
                sub += h * n - (steps_done * rate / 3600)
            else:
                cells.append("   N/A")
        total_remaining_hrs += sub
        print(f"  {d:>5}  {'  '.join(cells)}  {sub:>12.1f}h")

    print(f"\n  Remaining: {total_remaining_hrs:.0f} hrs = {total_remaining_hrs/24:.1f} days")

    # --- Dense baselines ---
    print(f"\n--- Dense baselines " + "-" * 40)
    dense = conn.execute("""
        SELECT d_model, status,
               (SELECT MAX(step) FROM results WHERE config_id=c.config_id) as last_step
        FROM configs c WHERE model_type='dense' ORDER BY d_model
    """).fetchall()
    for r in dense:
        step_str = f"step {r['last_step']:,}" if r["last_step"] else "no results"
        print(f"  d={r['d_model']:4d}  {r['status']:8s}  {step_str}")

    print()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(_ROOT / "results.db"))
    args = parser.parse_args()
    print_status(args.db)


if __name__ == "__main__":
    main()
