"""
Run lm-evaluation-harness benchmarks on a CART or DenseBaseline checkpoint.

Usage:
    # By checkpoint path:
    python eval/run_benchmarks.py --checkpoint checkpoints/<id>/step_61000.pt --db results.db

    # By config-id (auto-locates checkpoint):
    python eval/run_benchmarks.py --config-id <id> --step 61000 --db results.db

    # Single task, custom batch size:
    python eval/run_benchmarks.py --config-id <id> --step 61000 --db results.db --tasks lambada_openai --batch-size 32
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import lm_eval

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.lm_eval_adapter import CARTLMEval
from model.config import CARTConfig

DEFAULT_TASKS = ["hellaswag", "arc_challenge", "piqa", "lambada_openai", "wikitext"]


def find_checkpoint(config_id: str, step: int) -> Path:
    ckpt = Path("checkpoints") / config_id / f"step_{step}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def ensure_benchmark_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            result_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id   TEXT    NOT NULL,
            step        INTEGER NOT NULL,
            task        TEXT    NOT NULL,
            metric      TEXT    NOT NULL,
            value       REAL    NOT NULL,
            num_fewshot INTEGER DEFAULT 0,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def write_results(conn, config_id, step, task_results, num_fewshot):
    rows = []
    for task, metrics in task_results.items():
        for metric, value in metrics.items():
            if metric.endswith("_stderr") or metric == "alias":
                continue
            if not isinstance(value, (int, float)):
                continue
            rows.append((config_id, step, task, metric, float(value), num_fewshot))
    conn.executemany(
        "INSERT INTO benchmark_results (config_id, step, task, metric, value, num_fewshot) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Direct path to .pt checkpoint file")
    group.add_argument("--config-id", help="Config ID (auto-locates checkpoint under checkpoints/)")

    parser.add_argument("--step", type=int, default=61000,
                        help="Training step (used with --config-id, default: 61000)")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                        help=f"Comma-separated tasks (default: {','.join(DEFAULT_TASKS)})")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--db", help="results.db path — write benchmark results")
    parser.add_argument("--output", help="Optional JSON output file")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Resolve checkpoint path and config_id
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        config_id = ckpt_path.parent.name
        step = int(ckpt_path.stem.split("_")[1]) if ckpt_path.stem.startswith("step_") else args.step
    else:
        ckpt_path = find_checkpoint(args.config_id, args.step)
        config_id = args.config_id
        step = args.step

    print(f"Checkpoint : {ckpt_path}")
    print(f"Config ID  : {config_id}  Step: {step}")

    # Load model
    model = CARTLMEval(
        checkpoint=str(ckpt_path),
        batch_size=args.batch_size,
        device=args.device,
    )
    cfg = model.config
    if isinstance(cfg, CARTConfig):
        print(f"Model      : CART  d={cfg.d_model}  R={cfg.n_loops}  P={cfg.n_prelude}")
    else:
        print(f"Model      : Dense d={cfg.d_model}  n_layers={cfg.n_layers}")

    param_info = model.model.count_parameters()
    print(f"Params     : {param_info['total']:,} stored  "
          f"{param_info['effective']:,} effective  "
          f"{param_info['leverage']:.3f}x leverage")

    # Run evaluation
    tasks = [t.strip() for t in args.tasks.split(",")]
    print(f"\nTasks      : {tasks}")
    print(f"Fewshot    : {args.num_fewshot}  Batch: {args.batch_size}\n")

    print("Running evaluations...", flush=True)
    eval_results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        log_samples=False,
        bootstrap_iters=0,
    )
    print("Evaluations complete. Aggregating results...", flush=True)

    # Print results table
    print("\n=== Results ===", flush=True)
    for task, res in eval_results["results"].items():
        print(f"\n{task}:")
        for metric, val in res.items():
            if metric.endswith("_stderr") or metric == "alias":
                continue
            if not isinstance(val, (int, float)):
                continue
            stderr_key = metric.replace(",none", "_stderr,none")
            stderr = res.get(stderr_key, None)
            stderr_str = f" ± {stderr:.4f}" if isinstance(stderr, float) else ""
            print(f"  {metric}: {val:.4f}{stderr_str}")

    # Write to DB
    if args.db:
        conn = sqlite3.connect(args.db, timeout=30)
        ensure_benchmark_table(conn)
        n = write_results(conn, config_id, step, eval_results["results"], args.num_fewshot)
        conn.close()
        print(f"\nWrote {n} metrics to {args.db}")

    # Write JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        print(f"JSON saved to {args.output}")


if __name__ == "__main__":
    main()
