"""
Variable-R inference benchmarks: evaluate a CART checkpoint at inference-time
R values different from training-time R.

Loads the model once, then for each requested R override:
  - sets model.config.n_loops = R
  - runs all requested lm-eval tasks
  - writes per-(task, metric) rows to results.db with the override R recorded
    in a dedicated inference_n_loops column

Usage:
    python eval/variable_r_benchmarks.py --config-id <id> --step 30500 \\
        --inference-loops 6,8,10,12,14,16 --db results.db

The model architecture's `for r in range(self.config.n_loops)` reads the loop
count dynamically at forward time, so any R is valid without reloading.
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import lm_eval

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.lm_eval_adapter import CARTLMEval
from model.config import CARTConfig

DEFAULT_TASKS = ["hellaswag", "arc_challenge", "piqa", "lambada_openai"]


def find_checkpoint(config_id: str, step: int) -> Path:
    ckpt = Path("checkpoints") / config_id / f"step_{step}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def ensure_variable_r_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS variable_r_results (
            result_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id          TEXT    NOT NULL,
            step               INTEGER NOT NULL,
            train_n_loops      INTEGER NOT NULL,
            inference_n_loops  INTEGER NOT NULL,
            task               TEXT    NOT NULL,
            metric             TEXT    NOT NULL,
            value              REAL    NOT NULL,
            num_fewshot        INTEGER DEFAULT 0,
            recorded_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def write_results(conn, config_id, step, train_R, inference_R, task_results, num_fewshot):
    rows = []
    for task, metrics in task_results.items():
        for metric, value in metrics.items():
            if metric.endswith("_stderr") or metric == "alias":
                continue
            if not isinstance(value, (int, float)):
                continue
            rows.append((config_id, step, train_R, inference_R, task, metric,
                         float(value), num_fewshot))
    conn.executemany(
        "INSERT INTO variable_r_results "
        "(config_id, step, train_n_loops, inference_n_loops, task, metric, value, num_fewshot) "
        "VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--step", type=int, default=30500)
    parser.add_argument("--inference-loops", default="6,8,10,12,14,16",
                        help="Comma-separated R values to evaluate at (default: 6,8,10,12,14,16)")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--db", default="results.db")
    parser.add_argument("--output", help="Optional JSON output of all results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt_path = find_checkpoint(args.config_id, args.step)
    inference_loops = [int(x) for x in args.inference_loops.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"Checkpoint     : {ckpt_path}")
    print(f"Config ID      : {args.config_id}  Step: {args.step}")
    print(f"Inference R    : {inference_loops}")
    print(f"Tasks          : {tasks}")
    print(f"Batch size     : {args.batch_size}  Fewshot: {args.num_fewshot}\n")

    # Load model once
    model = CARTLMEval(
        checkpoint=str(ckpt_path),
        batch_size=args.batch_size,
        device=args.device,
    )
    cfg = model.config
    if not isinstance(cfg, CARTConfig):
        raise ValueError("Variable-R inference only meaningful for CART; "
                         "this checkpoint is not a CART model.")
    train_R = cfg.n_loops
    print(f"Model loaded   : CART d={cfg.d_model} P={cfg.n_prelude} trained at R={train_R}")
    param_info = model.model.count_parameters()
    print(f"Params (trained R): {param_info['total']:,} stored  "
          f"{param_info['effective']:,} effective\n")

    # Prepare DB
    conn = None
    if args.db:
        conn = sqlite3.connect(args.db, timeout=30)
        ensure_variable_r_table(conn)

    all_results = {}
    for R in inference_loops:
        print("=" * 60)
        print(f"  Inference R = {R}  (trained at R = {train_R})")
        print("=" * 60)
        # Override the loop count for inference
        model.config.n_loops = R
        model.model.config.n_loops = R

        t0 = time.time()
        eval_results = lm_eval.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            log_samples=False,
            bootstrap_iters=0,
        )
        elapsed = time.time() - t0
        print(f"\n  Elapsed for R={R}: {elapsed/60:.1f} min")

        # Print + record
        print(f"\n  === Results at inference R={R} ===")
        for task, res in eval_results["results"].items():
            print(f"  {task}:")
            for metric, val in res.items():
                if metric.endswith("_stderr") or metric == "alias":
                    continue
                if not isinstance(val, (int, float)):
                    continue
                print(f"    {metric}: {val:.4f}")

        all_results[R] = eval_results["results"]

        if conn:
            n_rows = write_results(conn, args.config_id, args.step, train_R, R,
                                   eval_results["results"], args.num_fewshot)
            print(f"  Wrote {n_rows} metrics to {args.db}\n")

    if conn:
        conn.close()

    # JSON dump of everything
    if args.output:
        with open(args.output, "w") as f:
            json.dump({str(R): r for R, r in all_results.items()}, f, indent=2, default=str)
        print(f"\nJSON saved to {args.output}")

    print("\n=== Variable-R sweep complete ===")
    print(f"  Trained at R = {train_R}")
    print(f"  Evaluated at R = {inference_loops}")


if __name__ == "__main__":
    main()
