"""
FLOPs analysis: CART vs DenseBaseline.

Counts forward-pass FLOPs (= 2 * MACs) for one sequence of length T.
Ignores: embedding lookup, RMSNorm, RoPE, LTI, HyperConnection, LIE
  — these are all O(Td) and negligible vs O(Td^2) attention/FFN terms.
Output projection (tied embedding, cost 2*T*d*vocab) is identical in both
models and cancels in comparisons, so it is excluded.

Usage:
    python eval/flops_calc.py
    python eval/flops_calc.py --d 1024 --seq-len 1024
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def ffn_intermediate(d: int) -> int:
    """SwiGLU FFN hidden dim — matches CARTConfig and DenseConfig."""
    return math.ceil(int(8 / 3 * d) / 256) * 256


# ---------------------------------------------------------------------------
# Per-component FLOPs (returns float, units: FLOPs for one sequence of len T)
# ---------------------------------------------------------------------------

def flops_dense_layer(d: int, T: int) -> float:
    """Standard MHA (full-rank Q/K/V/O) + SwiGLU FFN."""
    d_ff = ffn_intermediate(d)
    # Q, K, V, O projections: each d->d applied to T tokens
    proj  = 4 * 2 * T * d * d
    # Attention: QK^T + softmax*V (both T x T x d)
    attn  = 2 * 2 * T * T * d
    # SwiGLU: gate(d->d_ff) + up(d->d_ff) + down(d_ff->d)
    ffn   = 3 * 2 * T * d * d_ff
    return proj + attn + ffn


def flops_mla_self_attn_layer(d: int, T: int) -> float:
    """MLA self-attention (PreludeLayer / CodaLayer): compressed KV + full Q."""
    d_kv  = d // 4           # mla_compression_ratio = 4
    d_ff  = ffn_intermediate(d)
    # Q full-rank, KV compressed: down(d->d_kv), k_up(d_kv->d), v_up(d_kv->d), O(d->d)
    q_proj   = 2 * T * d * d
    kv_down  = 2 * T * d * d_kv
    k_up     = 2 * T * d_kv * d
    v_up     = 2 * T * d_kv * d
    attn     = 2 * 2 * T * T * d         # QK^T + softmax*V
    o_proj   = 2 * T * d * d
    ffn      = 3 * 2 * T * d * d_ff
    return q_proj + kv_down + k_up + v_up + attn + o_proj + ffn


def flops_kv_projection(d: int, T: int) -> float:
    """MLAKVProjection: compute K, V from prelude output e (paid once)."""
    d_kv = d // 4
    kv_down = 2 * T * d * d_kv
    k_up    = 2 * T * d_kv * d
    v_up    = 2 * T * d_kv * d
    return kv_down + k_up + v_up


def flops_core_block(d: int, T: int) -> float:
    """CoreBlock: MLA cross-attention (Q only) + SwiGLU FFN."""
    d_ff  = ffn_intermediate(d)
    # Q only (K, V pre-computed and reused across all R loops)
    q_proj = 2 * T * d * d
    attn   = 2 * 2 * T * T * d           # Q*K^T + softmax*V
    o_proj = 2 * T * d * d
    ffn    = 3 * 2 * T * d * d_ff
    return q_proj + attn + o_proj + ffn


# ---------------------------------------------------------------------------
# Model-level FLOPs
# ---------------------------------------------------------------------------

def flops_cart(d: int, T: int, R: int, P: int) -> float:
    """Total CART forward-pass FLOPs (one sequence)."""
    prelude  = P * flops_mla_self_attn_layer(d, T)
    kv_proj  = flops_kv_projection(d, T)
    core     = R * flops_core_block(d, T)
    coda     = flops_mla_self_attn_layer(d, T)    # n_coda = 1
    return prelude + kv_proj + core + coda


def flops_dense(d: int, T: int, n_layers: int) -> float:
    """Total Dense forward-pass FLOPs (one sequence)."""
    return n_layers * flops_dense_layer(d, T)


def flops_matched_layers(d: int, T: int, R: int, P: int) -> float:
    """Number of Dense layers (at same d) that matches CART FLOPs. May be non-integer."""
    cart   = flops_cart(d, T, R, P)
    layer  = flops_dense_layer(d, T)
    return cart / layer


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------

def params_cart(d: int, R: int, P: int) -> dict:
    """Approximate stored parameter counts for CART (no biases)."""
    d_kv = d // 4
    d_ff = ffn_intermediate(d)
    vocab = 32_000

    # MLA self-attn layer (prelude / coda): Q(d^2) + kv_down(d*d_kv) + k_up(d_kv*d) + v_up(d_kv*d) + O(d^2)
    mla_self = d**2 + d*d_kv + d_kv*d + d_kv*d + d**2
    # SwiGLU FFN: gate + up + down
    ffn_p = d*d_ff + d*d_ff + d_ff*d
    # MLA cross-attn (core): Q(d^2) + O(d^2) only — KV handled by kv_proj
    mla_cross = d**2 + d**2
    # KV projection: kv_down + k_up + v_up
    kv_proj = d*d_kv + d_kv*d + d_kv*d

    prelude_p = P * (mla_self + ffn_p)
    core_p    = mla_cross + ffn_p          # shared, used R times
    coda_p    = mla_self + ffn_p
    embed_p   = vocab * d

    stored    = prelude_p + kv_proj + core_p + coda_p + embed_p
    effective = stored + (R - 1) * core_p
    return {"stored": stored, "effective": effective,
            "leverage": effective / stored, "core": core_p}


def params_dense(d: int, n_layers: int) -> dict:
    """Approximate stored parameter counts for Dense (no biases)."""
    d_ff  = ffn_intermediate(d)
    vocab = 32_000
    # Full MHA: Q, K, V, O each d^2
    mha = 4 * d**2
    ffn = d*d_ff + d*d_ff + d_ff*d
    layer_p = mha + ffn
    embed_p = vocab * d
    total   = n_layers * layer_p + embed_p
    return {"stored": total, "effective": total, "leverage": 1.0}


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",       type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    d, T = args.d, args.seq_len
    d_ff = ffn_intermediate(d)

    print(f"FLOPs Analysis: d={d}  T={T}  d_ff={d_ff}")
    print("=" * 70)

    # --- Dense reference ---
    print(f"\nDense baseline (7 layers, d={d}):")
    d7 = flops_dense(d, T, 7)
    p7 = params_dense(d, 7)
    print(f"  FLOPs : {d7/1e9:>8.2f} GFLOPs")
    print(f"  Params: {p7['stored']/1e6:>8.2f} M stored")

    # --- CART sweep configs ---
    print(f"\nCART configs at d={d}, T={T}:")
    print(f"  {'R':>3} {'P':>3}  {'GFLOPs':>9}  {'Match (layers)':>16}  "
          f"{'Stored M':>10}  {'Effective M':>12}  {'Leverage':>9}")
    print("  " + "-" * 68)

    best_config = None
    best_flops = 0

    for R in [6, 8, 10]:
        for P in [4, 6, 8]:
            f    = flops_cart(d, T, R, P)
            n    = flops_matched_layers(d, T, R, P)
            pc   = params_cart(d, R, P)
            print(f"  {R:>3} {P:>3}  {f/1e9:>9.2f}  {n:>14.1f} L  "
                  f"{pc['stored']/1e6:>10.2f}  {pc['effective']/1e6:>12.2f}  "
                  f"{pc['leverage']:>8.2f}x")
            if f > best_flops:
                best_flops = f
                best_config = (R, P, n, pc)

    # --- Best config detail ---
    R_b, P_b, n_b, pc_b = best_config
    n_b_round = round(n_b)

    print(f"\n{'='*70}")
    print(f"FLOPs-matched Dense for best CART (R={R_b}, P={P_b}):")
    print(f"  CART FLOPs  : {best_flops/1e9:.2f} GFLOPs")
    print(f"  Exact match : {n_b:.2f} Dense layers")
    print(f"  Round to    : {n_b_round} layers")

    dense_match_f = flops_dense(d, T, n_b_round)
    dense_match_p = params_dense(d, n_b_round)
    print(f"  Dense ({n_b_round}L) FLOPs : {dense_match_f/1e9:.2f} GFLOPs  "
          f"({100*(dense_match_f-best_flops)/best_flops:+.1f}%)")

    print(f"\n  Parameter comparison:")
    print(f"  {'Model':<28} {'Stored':>10}  {'Effective':>12}  {'Leverage':>9}")
    print(f"  {'-'*60}")
    print(f"  {'CART R='+str(R_b)+' P='+str(P_b)+' d='+str(d):<28} "
          f"{pc_b['stored']/1e6:>10.2f}M  {pc_b['effective']/1e6:>12.2f}M  "
          f"{pc_b['leverage']:>8.2f}x")
    print(f"  {'Dense '+str(n_b_round)+'L d='+str(d)+' (FLOPs-match)':<28} "
          f"{dense_match_p['stored']/1e6:>10.2f}M  {dense_match_p['effective']/1e6:>12.2f}M  "
          f"{'1.00':>8}x")
    print(f"  {'Dense 7L d='+str(d)+' (param-match)':<28} "
          f"{p7['stored']/1e6:>10.2f}M  {p7['stored']/1e6:>12.2f}M  "
          f"{'1.00':>8}x")

    param_eff = dense_match_p['stored'] / pc_b['stored']
    print(f"\n  CART stores {param_eff:.2f}x fewer params than the FLOPs-matched Dense.")


if __name__ == "__main__":
    main()
