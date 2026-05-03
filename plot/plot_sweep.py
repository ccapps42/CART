"""
Generates paper figures from results.db.

Usage:
    python plot/plot_sweep.py --db results.db --out figures/

Figures produced:
    loss_curves_{d}.png        — training loss curves per d_model
    ppl_vs_loops_{d}.png       — eval_ppl_tiny vs n_loops, grouped by n_prelude
    ppl_vs_prelude_{d}.png     — eval_ppl_tiny vs n_prelude, grouped by n_loops
    vram_heatmap.png           — peak VRAM (GB) heatmap per (d_model, n_loops)
    tps_vs_d.png               — tokens/sec vs d_model
    spectral_radius_{d}.png    — LTI spectral radius evolution per d_model
"""
import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe on Windows without display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

EVAL_STEP = 3000    # final eval step for main comparison figures
STAGE = 1


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_dims(conn) -> list:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT d_model FROM configs WHERE stage=? ORDER BY d_model",
        (STAGE,)
    ).fetchall()]


# ---------------------------------------------------------------------------
# Figure 1: Loss curves per d_model
# ---------------------------------------------------------------------------

def plot_loss_curves(conn, out_dir: Path, d_model: int):
    rows = conn.execute(
        """SELECT c.n_loops, c.n_prelude, t.step, t.train_loss
           FROM train_log t
           JOIN configs c ON t.config_id = c.config_id
           WHERE c.d_model = ? AND c.stage = ?
           ORDER BY c.n_loops, c.n_prelude, t.step""",
        (d_model, STAGE),
    ).fetchall()

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    configs = {}
    for r in rows:
        key = (r["n_loops"], r["n_prelude"])
        configs.setdefault(key, []).append((r["step"], r["train_loss"]))

    colors = cm.viridis(np.linspace(0, 1, len(configs)))
    for (key, pts), color in zip(sorted(configs.items()), colors):
        steps = [p[0] for p in pts]
        losses = [p[1] for p in pts]
        ax.plot(steps, losses, color=color, linewidth=0.8,
                label=f"R={key[0]} P={key[1]}", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"Loss Curves — d_model={d_model}")
    if len(configs) <= 16:
        ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    path = out_dir / f"loss_curves_{d_model}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ---------------------------------------------------------------------------
# Figure 2 & 3: Perplexity vs n_loops and vs n_prelude
# ---------------------------------------------------------------------------

def plot_ppl_vs_axis(conn, out_dir: Path, d_model: int, x_axis: str):
    """x_axis: 'n_loops' or 'n_prelude'"""
    group_axis = "n_prelude" if x_axis == "n_loops" else "n_loops"
    rows = conn.execute(
        f"""SELECT c.n_loops, c.n_prelude, r.eval_ppl_tiny
            FROM configs c
            JOIN results r ON c.config_id = r.config_id
            WHERE c.d_model = ? AND c.stage = ? AND c.status = 'complete'
              AND r.step = ?
            ORDER BY c.{x_axis}, c.{group_axis}""",
        (d_model, STAGE, EVAL_STEP),
    ).fetchall()

    if not rows:
        return

    groups = {}
    for r in rows:
        key = r[group_axis]
        x = r[x_axis]
        groups.setdefault(key, []).append((x, r["eval_ppl_tiny"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    for (gval, pts), color in zip(sorted(groups.items()), colors):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = f"{'P' if group_axis == 'n_prelude' else 'R'}={gval}"
        ax.plot(xs, ys, marker="o", color=color, label=label)

    ax.set_xlabel(x_axis.replace("_", " ").replace("n ", "").replace("loops", "n_loops").replace("prelude", "n_prelude"))
    ax.set_ylabel("Eval Perplexity (TinyStories)")
    ax.set_title(f"PPL vs {x_axis} — d_model={d_model} (step {EVAL_STEP})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fname = f"ppl_vs_{x_axis.replace('n_', '')}_{d_model}.png"
    path = out_dir / fname
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ---------------------------------------------------------------------------
# Figure 4: VRAM heatmap
# ---------------------------------------------------------------------------

def plot_vram_heatmap(conn, out_dir: Path):
    rows = conn.execute(
        """SELECT c.d_model, c.n_loops, r.peak_vram_gb
           FROM configs c
           JOIN results r ON c.config_id = r.config_id
           WHERE c.stage = ? AND c.status = 'complete' AND r.step = ?
           ORDER BY c.d_model, c.n_loops""",
        (STAGE, EVAL_STEP),
    ).fetchall()

    if not rows:
        return

    dims  = sorted(set(r["d_model"] for r in rows))
    loops = sorted(set(r["n_loops"] for r in rows))
    data  = np.full((len(dims), len(loops)), np.nan)

    vram_lookup = {}
    for r in rows:
        vram_lookup[(r["d_model"], r["n_loops"])] = r["peak_vram_gb"]

    for i, d in enumerate(dims):
        for j, l in enumerate(loops):
            val = vram_lookup.get((d, l))
            if val is not None:
                data[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(loops)))
    ax.set_xticklabels([f"R={l}" for l in loops])
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f"d={d}" for d in dims])
    ax.set_title(f"Peak VRAM (GB) — Stage {STAGE} at step {EVAL_STEP}")
    fig.colorbar(im, ax=ax, label="GB")
    for i in range(len(dims)):
        for j in range(len(loops)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    path = out_dir / "vram_heatmap.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ---------------------------------------------------------------------------
# Figure 5: Tokens/sec vs d_model
# ---------------------------------------------------------------------------

def plot_tps(conn, out_dir: Path):
    rows = conn.execute(
        """SELECT c.d_model, c.n_loops, c.n_prelude, r.tokens_per_sec
           FROM configs c
           JOIN results r ON c.config_id = r.config_id
           WHERE c.stage = ? AND c.status = 'complete' AND r.step = ?
           ORDER BY c.d_model""",
        (STAGE, EVAL_STEP),
    ).fetchall()

    if not rows:
        return

    from collections import defaultdict
    by_dim = defaultdict(list)
    for r in rows:
        if r["tokens_per_sec"]:
            by_dim[r["d_model"]].append(r["tokens_per_sec"])

    dims = sorted(by_dim)
    medians = [float(np.median(by_dim[d])) for d in dims]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(d) for d in dims], medians, color="steelblue")
    ax.set_xlabel("d_model")
    ax.set_ylabel("Tokens / sec (median)")
    ax.set_title(f"Throughput by Model Width — Stage {STAGE} at step {EVAL_STEP}")
    ax.grid(True, axis="y", alpha=0.3)
    path = out_dir / "tps_vs_d.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ---------------------------------------------------------------------------
# Figure 6: LTI spectral radius evolution
# ---------------------------------------------------------------------------

def plot_spectral_radius(conn, out_dir: Path, d_model: int):
    rows = conn.execute(
        """SELECT c.n_loops, c.n_prelude, t.step, t.lti_spectral_radius
           FROM train_log t
           JOIN configs c ON t.config_id = c.config_id
           WHERE c.d_model = ? AND c.stage = ?
             AND t.lti_spectral_radius IS NOT NULL
           ORDER BY c.n_loops, c.n_prelude, t.step""",
        (d_model, STAGE),
    ).fetchall()

    if not rows:
        return

    configs = {}
    for r in rows:
        key = (r["n_loops"], r["n_prelude"])
        configs.setdefault(key, []).append((r["step"], r["lti_spectral_radius"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.plasma(np.linspace(0, 1, len(configs)))
    for (key, pts), color in zip(sorted(configs.items()), colors):
        pts = sorted(pts)
        steps = [p[0] for p in pts]
        rhos  = [p[1] for p in pts]
        ax.plot(steps, rhos, color=color, linewidth=0.8,
                label=f"R={key[0]} P={key[1]}", alpha=0.8)

    ax.axhline(0.99, color="red", linestyle="--", linewidth=0.8, label="instability threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("LTI Spectral Radius rho(A)")
    ax.set_title(f"Spectral Radius Evolution — d_model={d_model}")
    ax.set_ylim(0.85, 1.02)
    if len(configs) <= 16:
        ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    path = out_dir / f"spectral_radius_{d_model}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="results.db")
    parser.add_argument("--out", default="figures/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = open_db(args.db)
    dims = get_dims(conn)

    if not dims:
        print("No configs found in DB.")
        conn.close()
        return

    print(f"Generating figures for dims: {dims}")
    print(f"Output: {out_dir.resolve()}")
    print()

    for d in dims:
        print(f"d_model={d}:")
        plot_loss_curves(conn, out_dir, d)
        plot_ppl_vs_axis(conn, out_dir, d, "n_loops")
        plot_ppl_vs_axis(conn, out_dir, d, "n_prelude")
        plot_spectral_radius(conn, out_dir, d)

    print("Global figures:")
    plot_vram_heatmap(conn, out_dir)
    plot_tps(conn, out_dir)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
