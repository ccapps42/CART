"""
Generate the "ppl_wiki vs. R" figure for the paper.

For each (d_model, P) combination from Stage 1, plot the final eval_ppl_wiki
as a function of R. One subplot per d_model; one line per P within each subplot.

Usage:
    python figures/gen_ppl_vs_r.py
Outputs:
    figures/ppl_vs_r.{png,pdf}
"""
import os
import sqlite3
import sys
import matplotlib.pyplot as plt

DB = "results.db"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_stage1(conn):
    """Final ppl_wiki for each Stage 1 CART config (latest step per config)."""
    return conn.execute("""
        SELECT c.d_model, c.n_loops, c.n_prelude, r.eval_ppl_wiki
        FROM configs c
        JOIN results r ON c.config_id = r.config_id
        WHERE c.stage = 1 AND c.model_type = 'cart' AND c.status = 'complete'
          AND r.step = (SELECT MAX(r2.step) FROM results r2 WHERE r2.config_id = c.config_id)
          AND r.eval_ppl_wiki IS NOT NULL
        ORDER BY c.d_model, c.n_prelude, c.n_loops
    """).fetchall()


def main():
    if not os.path.exists(DB):
        sys.exit(f"results.db not found at {DB}; run from repo root")

    conn = sqlite3.connect(DB)
    rows = fetch_stage1(conn)
    if not rows:
        sys.exit("No complete Stage 1 CART configs in results.db")

    # Bin by (d, P): {(d, P): [(R, ppl), ...]}
    series = {}
    for d, R, P, ppl in rows:
        series.setdefault((d, P), []).append((R, ppl))
    for k in series:
        series[k].sort()

    d_values = sorted({d for d, _ in series.keys()})
    P_values = sorted({P for _, P in series.keys()})

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    axes = axes.flatten()

    P_to_color = {2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c", 6: "#d62728"}
    P_to_marker = {2: "o", 3: "s", 4: "^", 6: "D"}

    for ax, d in zip(axes, d_values):
        for P in P_values:
            pts = series.get((d, P), [])
            if not pts:
                continue
            xs = [r for r, _ in pts]
            ys = [p for _, p in pts]
            ax.plot(xs, ys, marker=P_to_marker[P], color=P_to_color[P],
                    label=f"P={P}", linewidth=1.6, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.set_title(f"$d_\\mathrm{{model}}={d}$", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted({r for pts in series.values() for r, _ in pts}))

    # Shared axis labels
    for ax in axes[:2]:
        ax.set_ylabel(r"PPL$_\mathrm{wiki}$ at step 3{,}000")
    for ax in axes[2:]:
        ax.set_ylabel(r"PPL$_\mathrm{wiki}$ at step 3{,}000")
        ax.set_xlabel("Loop count $R$")

    # Single shared legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(P_values),
               bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=10)

    fig.suptitle("Stage 1: Wikipedia perplexity vs. loop count, by prelude depth and scale",
                 fontsize=11, y=1.06)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ppl_vs_r.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT_DIR, "ppl_vs_r.pdf"),
                bbox_inches="tight", facecolor="white")
    print(f"Saved ppl_vs_r figure. {len(rows)} Stage 1 datapoints across "
          f"{len(d_values)} scales × {len(P_values)} P values.")


if __name__ == "__main__":
    main()
