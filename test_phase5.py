"""
Phase 5 verification:
1. eval/perplexity.py standalone against Phase 3/4 test checkpoint
2. sweep/analyze.py on current results.db (may have few/no step-3000 rows)
3. plot/plot_sweep.py generates figures from available data
"""
import subprocess
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path("K:/projects/Model_Paper_1")
PYTHON = "C:/Python314/python.exe"
DB = PROJECT_ROOT / "results.db"


def run(cmd, **kw):
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        lines = [l for l in result.stderr.splitlines()
                 if "triton" not in l.lower() and "flop" not in l.lower()]
        if lines:
            print("STDERR:", "\n".join(lines))
    return result.returncode


def test_perplexity():
    print("[1] eval/perplexity.py standalone")
    # Find any checkpoint from Phase 4
    ckpt_dirs = list((PROJECT_ROOT / "checkpoints").glob("*/step_50.pt"))
    if not ckpt_dirs:
        print("  No step_50.pt checkpoints found — skipping perplexity test")
        return

    ckpt = ckpt_dirs[0]
    # Create a small synthetic val bin
    import numpy as np
    tmp_val = Path("K:/tmp/phase5_val.bin")
    tmp_val.parent.mkdir(exist_ok=True)
    np.random.default_rng(99).integers(3, 31999, 30_000, dtype=np.uint16).tofile(str(tmp_val))

    rc = run([
        PYTHON, str(PROJECT_ROOT / "eval" / "perplexity.py"),
        "--checkpoint", str(ckpt),
        "--data", str(tmp_val),
        "--seq-len", "512",
        "--max-batches", "5",
        "--batch-size", "1",
    ])
    assert rc == 0, f"perplexity.py failed with code {rc}"
    tmp_val.unlink(missing_ok=True)
    print("  perplexity.py OK")


def test_analyze():
    print("\n[2] sweep/analyze.py (expects step-3000 rows; may show empty)")
    rc = run([
        PYTHON, str(PROJECT_ROOT / "sweep" / "analyze.py"),
        "--db", str(DB),
        "--out", "sweep/stage2_configs.json",
    ])
    # analyze.py exits 0 even with no complete results
    assert rc == 0, f"analyze.py failed with code {rc}"
    print("  analyze.py OK")


def test_plot():
    print("\n[3] plot/plot_sweep.py (generates figures from available data)")
    rc = run([
        PYTHON, str(PROJECT_ROOT / "plot" / "plot_sweep.py"),
        "--db", str(DB),
        "--out", str(PROJECT_ROOT / "figures"),
    ])
    assert rc == 0, f"plot_sweep.py failed with code {rc}"
    figs = list((PROJECT_ROOT / "figures").glob("*.png"))
    print(f"  Generated {len(figs)} figure(s):")
    for f in sorted(figs):
        print(f"    {f.name}")
    print("  plot_sweep.py OK")


def main():
    print("=== Phase 5 Verification ===\n")
    test_perplexity()
    test_analyze()
    test_plot()
    print("\n=== Phase 5 PASSED ===")


if __name__ == "__main__":
    main()
