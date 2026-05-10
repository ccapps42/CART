"""
Pareto frontier plot: stored parameters vs. ppl_wiki for all complete configs.
Highlights configs that lie on the Pareto frontier (no other config has both
fewer params AND lower ppl).

Usage:
    python figures/gen_pareto.py
Outputs:
    figures/pareto_params_pplwiki.{png,pdf}
"""
import sqlite3
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

DB = "results.db"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def cart_param_count(d, P, R, vocab=32000, n_heads=None, d_kv_ratio=4):
    """
    Stored parameter count for a CART config.
    Approximates the formulas in model/cart.py (count_parameters).
    Each MLA self-attention block: ~2.75 d^2  (Q full + KV down/up at d/4)
    Each SwiGLU FFN: ~8 d^2  (3 matrices of d x (8/3)d, rounded)
    Prelude layer = MLA self-attn + SwiGLU = 10.75 d^2
    Coda layer = same as prelude = 10.75 d^2
    Core block = same shape but counted once (loop reuses weights).
    KV projection (once before loop) = 2.25 d^2 (down to d/4 + two d/4 -> d up).
    Embedding = vocab * d (tied with output, counted once).
    Returns (stored, effective).
    """
    layer = 10.75 * d * d
    kv_proj = 2.25 * d * d
    embed = vocab * d
    stored = embed + P * layer + kv_proj + layer + layer  # +core +coda
    effective = stored + (R - 1) * layer
    return int(stored), int(effective)


def dense_param_count(d, n_layers=7, vocab=32000):
    """
    Dense baseline: n_layers transformer layers, each MHA + SwiGLU FFN.
    Standard MHA: 4 d^2 (Q,K,V,O); SwiGLU: 8 d^2.
    Layer total: 12 d^2.
    """
    layer = 12.0 * d * d
    embed = vocab * d
    return int(embed + n_layers * layer)


def fetch_complete_configs(conn):
    """For each complete config, get the latest eval row."""
    rows = conn.execute("""
        SELECT c.config_id, c.d_model, c.n_loops, c.n_prelude, c.seed,
               c.model_type, r.eval_ppl_wiki, r.eval_ppl_tiny, r.eval_ppl_edu
        FROM configs c
        JOIN results r ON c.config_id = r.config_id
        WHERE c.stage = 2 AND c.status = 'complete'
          AND r.step = (SELECT MAX(r2.step) FROM results r2 WHERE r2.config_id = c.config_id)
    """).fetchall()
    return rows


def pareto_frontier(points):
    """Return indices of points on the Pareto frontier (lower x AND lower y)."""
    pts = sorted(enumerate(points), key=lambda kv: (kv[1][0], kv[1][1]))
    frontier = []
    best_y = float("inf")
    for idx, (x, y) in pts:
        if y < best_y:
            frontier.append(idx)
            best_y = y
    return set(frontier)


def main():
    if not os.path.exists(DB):
        sys.exit(f"results.db not found at {DB}; run from repo root")

    conn = sqlite3.connect(DB)
    rows = fetch_complete_configs(conn)
    if not rows:
        sys.exit("No complete Stage 2 configs found in results.db")

    cart_pts = []   # (params_stored, ppl_wiki, label, R, d)
    dense_pts = []
    for cid, d, R, P, seed, mtype, wiki, tiny, edu in rows:
        if wiki is None:
            continue
        if mtype == "cart":
            stored, _ = cart_param_count(d, P, R)
            cart_pts.append((stored / 1e6, wiki, f"d={d} R={R}", R, d))
        elif mtype == "dense":
            stored = dense_param_count(d)
            dense_pts.append((stored / 1e6, wiki, f"Dense d={d}", None, d))

    fig, ax = plt.subplots(figsize=(7, 5))

    # CART points colored by R
    R_to_color = {6: "#1f77b4", 8: "#ff7f0e", 10: "#2ca02c"}
    for params, ppl, label, R, d in cart_pts:
        ax.scatter(params, ppl, c=R_to_color.get(R, "#888888"),
                   s=60, marker="o", edgecolors="black", linewidths=0.6,
                   label=f"CART R={R}" if (R, d) == (R, min(d for *_, dd in cart_pts if _ != _ or dd == d)) else None)

    # De-duplicate legend entries by relabeling cleanly
    seen = set()
    for params, ppl, label, R, d in cart_pts:
        if R not in seen:
            ax.scatter([], [], c=R_to_color[R], s=60, marker="o",
                       edgecolors="black", linewidths=0.6, label=f"CART R={R}")
            seen.add(R)

    # Dense baselines
    if dense_pts:
        for params, ppl, label, _, d in dense_pts:
            ax.scatter(params, ppl, c="#d62728", s=80, marker="s",
                       edgecolors="black", linewidths=0.6,
                       label="Dense baseline" if d == dense_pts[0][4] else None)

    # Pareto frontier across all CART points
    cart_xy = [(p, w) for p, w, *_ in cart_pts]
    if len(cart_xy) >= 2:
        front_idx = pareto_frontier(cart_xy)
        front = sorted([cart_xy[i] for i in front_idx])
        xs, ys = zip(*front)
        ax.plot(xs, ys, "k--", linewidth=1.0, alpha=0.5, label="CART Pareto frontier")

    # Annotate by d_model
    for params, ppl, label, R, d in cart_pts:
        ax.annotate(f"d={d}", (params, ppl), fontsize=7,
                    xytext=(4, 2), textcoords="offset points", color="#333")

    ax.set_xlabel("Stored parameters (M)")
    ax.set_ylabel(r"PPL$_\mathrm{wiki}$ (lower is better)")
    ax.set_title("CART vs. Dense: parameters vs. Wikipedia perplexity")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pareto_params_pplwiki.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT_DIR, "pareto_params_pplwiki.pdf"),
                bbox_inches="tight", facecolor="white")
    print(f"Saved Pareto plot. CART configs plotted: {len(cart_pts)}, "
          f"Dense baselines: {len(dense_pts)}")


if __name__ == "__main__":
    main()
