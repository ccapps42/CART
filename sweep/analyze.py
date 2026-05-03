"""
Stage 1 -> Stage 2 zoom-and-confirm protocol.

For each dim size independently:
    1. Rank all complete Stage 1 configs by eval_ppl_tiny at step 3000
    2. Identify best n_loops and best n_prelude independently
    3. Apply boundary rule: if best is at edge of range, extend outward
       (e.g., if best n_loops = 8, add n_loops = 10)
    4. Propose Stage 2 configs: 3 values around each best
       (best-1_step, best, best+1_step using the original sweep spacing)
    5. Stage 2 uses 3 seeds per config

Usage:
    python sweep/analyze.py --db results.db [--insert-stage2] [--dry-run]
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LOOPS_GRID   = [2, 4, 6, 8]    # Stage 1 sweep values
PRELUDES_GRID = [2, 3, 4, 6]   # Stage 1 sweep values
STAGE2_SEEDS = [42, 137, 271]
STAGE2_EVAL_STEP = 3000         # rank by perplexity at this step


def config_id(d_model, n_loops, n_prelude, seed, stage):
    blob = f"{d_model}_{n_loops}_{n_prelude}_{seed}_{stage}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def neighbors(grid: list, best: int) -> list:
    """
    Return 3 values around best in the grid:
    [prev, best, next]. At boundaries, extend outward by the last step size.
    """
    if best not in grid:
        return [best]
    idx = grid.index(best)
    result = []

    # previous value
    if idx > 0:
        result.append(grid[idx - 1])
    else:
        # at left boundary: extend left by the first gap
        step = grid[1] - grid[0]
        result.append(best - step)

    result.append(best)

    # next value
    if idx < len(grid) - 1:
        result.append(grid[idx + 1])
    else:
        # at right boundary: extend right by the last gap
        step = grid[-1] - grid[-2]
        result.append(best + step)

    return result


def analyze_dim(conn, d_model: int) -> dict:
    """Rank Stage 1 results for a given d_model and propose Stage 2 configs."""
    rows = conn.execute(
        """SELECT c.config_id, c.n_loops, c.n_prelude,
                  r.eval_ppl_tiny, r.eval_ppl_wiki, r.eval_ppl_edu,
                  r.lti_spectral_radius, r.tokens_per_sec, r.peak_vram_gb
           FROM configs c
           JOIN results r ON c.config_id = r.config_id
           WHERE c.d_model = ?
             AND c.stage = 1
             AND c.status = 'complete'
             AND r.step = ?
           ORDER BY r.eval_ppl_tiny ASC""",
        (d_model, STAGE2_EVAL_STEP),
    ).fetchall()

    if not rows:
        return {"d_model": d_model, "status": "no_results", "stage2_configs": []}

    print(f"\n--- d_model = {d_model} ({len(rows)} complete at step {STAGE2_EVAL_STEP}) ---")
    print(f"{'rank':>4}  {'config_id':18s}  {'R':>2}  {'P':>2}  "
          f"{'ppl_tiny':>10}  {'ppl_wiki':>10}  {'ppl_edu':>10}")
    print("-" * 70)
    for i, r in enumerate(rows):
        print(f"{i+1:4d}  {r['config_id']}  {r['n_loops']:2d}  {r['n_prelude']:2d}  "
              f"{r['eval_ppl_tiny']:10.2f}  {r['eval_ppl_wiki']:10.2f}  {r['eval_ppl_edu']:10.2f}")

    best = rows[0]
    best_r = best["n_loops"]
    best_p = best["n_prelude"]
    print(f"\nBest: n_loops={best_r}  n_prelude={best_p}  ppl_tiny={best['eval_ppl_tiny']:.2f}")

    # Boundary check
    if best_r == LOOPS_GRID[-1]:
        print(f"  ** n_loops={best_r} is at right edge — extending to {best_r + 2}")
    if best_p == PRELUDES_GRID[-1]:
        print(f"  ** n_prelude={best_p} is at right edge — extending to {best_p + 2}")

    # Stage 2 config proposals
    r_vals = neighbors(LOOPS_GRID, best_r)
    p_vals = neighbors(PRELUDES_GRID, best_p)

    # R=8 is the grid edge — always extend to R=10 if R=8 is in the candidate set,
    # regardless of whether R=8 or R=6 won. We can't know where the curve flattens
    # without testing one step beyond the edge.
    r_max = LOOPS_GRID[-1]
    if r_max in r_vals and (r_max + 2) not in r_vals:
        r_vals.append(r_max + 2)
        print(f"  ** R={r_max} is at grid edge — extending Stage 2 to R={r_max + 2}")

    print(f"  Stage 2 n_loops candidates:   {r_vals}")
    print(f"  Stage 2 n_prelude candidates: {p_vals}")

    stage2_configs = []
    hw = "3090" if d_model == 1024 else "3050"
    for r_val in r_vals:
        for p_val in p_vals:
            for seed in STAGE2_SEEDS:
                cid = config_id(d_model, r_val, p_val, seed, 2)
                stage2_configs.append({
                    "config_id": cid,
                    "d_model":   d_model,
                    "n_loops":   r_val,
                    "n_prelude": p_val,
                    "seed":      seed,
                    "stage":     2,
                    "hardware":  hw,
                })

    print(f"  -> {len(stage2_configs)} Stage 2 configs proposed "
          f"({len(r_vals)} R x {len(p_vals)} P x {len(STAGE2_SEEDS)} seeds)")

    return {
        "d_model":      d_model,
        "best_n_loops": best_r,
        "best_n_prelude": best_p,
        "best_ppl_tiny": best["eval_ppl_tiny"],
        "r_candidates":  r_vals,
        "p_candidates":  p_vals,
        "stage2_configs": stage2_configs,
        "status": "ok",
    }


def insert_stage2(conn, all_configs: list):
    inserted = 0
    for cfg in all_configs:
        try:
            conn.execute(
                """INSERT INTO configs
                   (config_id, d_model, n_loops, n_prelude, seed, stage,
                    status, hardware, created_at)
                   VALUES (:config_id, :d_model, :n_loops, :n_prelude, :seed, :stage,
                           'pending', :hardware, datetime('now'))""",
                cfg,
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="results.db")
    parser.add_argument("--insert-stage2", action="store_true",
                        help="Insert proposed Stage 2 configs into DB")
    parser.add_argument("--out", default="sweep/stage2_configs.json",
                        help="JSON file to write Stage 2 proposals")
    args = parser.parse_args()

    conn = open_db(args.db)

    # Find all d_model values with at least one complete Stage 1 result
    dims = [r[0] for r in conn.execute(
        """SELECT DISTINCT c.d_model FROM configs c
           JOIN results r ON c.config_id = r.config_id
           WHERE c.stage=1 AND c.status='complete' AND r.step=?
           ORDER BY c.d_model""",
        (STAGE2_EVAL_STEP,),
    ).fetchall()]

    if not dims:
        print(f"No complete Stage 1 results at step {STAGE2_EVAL_STEP} found in {args.db}.")
        print("Run the full Stage 1 sweep before analyzing.")
        conn.close()
        return

    print(f"Analyzing dims: {dims}")

    all_proposals = []
    results_by_dim = {}
    for d in dims:
        result = analyze_dim(conn, d)
        results_by_dim[d] = result
        all_proposals.extend(result.get("stage2_configs", []))

    print(f"\n{'='*70}")
    print(f"Total Stage 2 configs proposed: {len(all_proposals)}")

    # Write JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"dims_analyzed": dims, "proposals": results_by_dim}, f, indent=2, default=str)
    print(f"Proposals written to: {out_path}")

    if args.insert_stage2 and all_proposals:
        n = insert_stage2(conn, all_proposals)
        print(f"Inserted {n} Stage 2 configs into DB.")

    conn.close()


if __name__ == "__main__":
    main()
