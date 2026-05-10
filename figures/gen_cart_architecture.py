"""
CART Architecture Block Diagram — Context-Anchored Recurrent Transformer
Top to bottom, same style as RTCC diagram.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 19), facecolor='white')
ax.set_xlim(0, 11)
ax.set_ylim(15.0, 40)
ax.set_aspect('equal')
ax.axis('off')

# ── Colors ───────────────────────────────────────────────────────────────────
C_IO      = '#d0e8ff'
C_EMBED   = '#cce5cc'
C_ATTN    = '#dceeff'
C_XATTN   = '#d5e8f5'
C_FFN     = '#dcf5e8'
C_HYPER   = '#ecdcf5'
C_LIE     = '#fde8d0'
C_LTI     = '#fff0cc'
C_KVP     = '#fce8f0'
C_LMH     = '#cce5cc'
C_ANCHOR  = '#fffacc'
EDGE      = '#444444'
ARROW     = '#333333'

CX  = 5.2
BW  = 6.4
BH  = 1.15
BHT = 1.55
GAP = 0.42
pad = 0.62

def box(ax, x, y, w, h, label, color, sublabel=None, fs=11):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h,
                        boxstyle="round,pad=0.0,rounding_size=0.18",
                        facecolor=color, edgecolor=EDGE, linewidth=1.2, zorder=2)
    ax.add_patch(b)
    ty = y + (0.18 if sublabel else 0)
    ax.text(x, ty, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='#1a1a1a', zorder=3)
    if sublabel:
        ax.text(x, y-0.28, sublabel, ha='center', va='center',
                fontsize=9, color='#444444', style='italic', zorder=3)

def arr(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2+0.02), xytext=(x, y1-0.02),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.4), zorder=3)

def section_box(ax, x, y_top, y_bot, w, color, edge, label, label_color):
    lx = x - w/2 - pad
    lw = w + 2*pad
    lh = y_top - y_bot
    r = FancyBboxPatch((lx, y_bot), lw, lh,
                        boxstyle="round,pad=0.0,rounding_size=0.3",
                        facecolor=color, edgecolor=edge,
                        linewidth=1.8, linestyle='--', zorder=0)
    ax.add_patch(r)
    ax.text(lx+0.22, y_bot+lh/2, label,
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=label_color, rotation=90, zorder=1)

# ── Layout ───────────────────────────────────────────────────────────────────
y = 38.5

# 1. Input Tokens
box(ax, CX, y, BW, BH, 'Input Tokens', C_IO)
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# 2. Token Embedding
box(ax, CX, y, BW, BH, 'Token Embedding', C_EMBED,
    sublabel='nn.Embedding(vocab_size=32000, d_model)  →  token vectors')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# ── PRELUDE ×P layers ────────────────────────────────────────────────────────
prelude_top = y + BH/2 + GAP*0.35

box(ax, CX, y, BW, BH, 'Multi-head Latent Attention', C_ATTN,
    sublabel='RMSNorm → MLA Self-Attention + RoPE  (residual add)')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'SwiGLU FFN', C_FFN,
    sublabel='RMSNorm → SwiGLU Feed-Forward  (residual add)')
prelude_bot = y - BH/2 - GAP*0.35

# Anchor e side node
anx = CX + BW/2 + pad + 1.2
anchor_y = y
ax.annotate('', xy=(anx-0.5, anchor_y), xytext=(CX+BW/2+0.02, anchor_y),
            arrowprops=dict(arrowstyle='->', color='#885500', lw=1.3, linestyle='dashed'), zorder=3)
eb = FancyBboxPatch((anx-0.5, anchor_y-0.38), 1.0, 0.76,
                     boxstyle="round,pad=0.0,rounding_size=0.12",
                     facecolor=C_ANCHOR, edgecolor='#885500', linewidth=1.2, zorder=2)
ax.add_patch(eb)
ax.text(anx, anchor_y, 'e\n(anchor)', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#885500', zorder=3)

section_box(ax, CX, prelude_top, prelude_bot, BW, '#f0fff4', '#228833', '×P  Prelude', '#228833')

y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# ── KV Projection (once, between prelude and loop) ───────────────────────────
box(ax, CX, y, BW, BH, 'KV Projection  (computed once)', C_KVP,
    sublabel='MLAKVProjection(e)  →  K, V tensors  (reused across all R iterations)')
kv_y = y



y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# ── RECURRENT CORE ×R ────────────────────────────────────────────────────────
loop_top = y + BH/2 + GAP*0.35

box(ax, CX, y, BW, BH, 'HyperConnection', C_HYPER,
    sublabel='Softmax-weighted blend of last 3 hidden states  →  h_input')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'Loop Index Embedding (LIE)', C_LIE,
    sublabel='Sinusoidal loop-index encoding → projected to model_dim\n→ added to h_input')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BHT, 'MLA Cross-Attention', C_XATTN,
    sublabel='RMSNorm → Q from h_input, K/V from prelude e, no RoPE  (residual add)')
xattn_y = y

# Short horizontal K,V arrow directly into cross-attention from the right
kvx = CX + BW/2 + pad + 1.2
kv_box_y = xattn_y
kv_eb = FancyBboxPatch((kvx-0.5, kv_box_y-0.38), 1.0, 0.76,
                        boxstyle="round,pad=0.0,rounding_size=0.12",
                        facecolor='#fce8f0', edgecolor='#aa3366', linewidth=1.2, zorder=2)
ax.add_patch(kv_eb)
ax.text(kvx, kv_box_y, 'K, V', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#aa3366', zorder=3)
ax.annotate('', xy=(CX+BW/2+0.02, kv_box_y), xytext=(kvx-0.5, kv_box_y),
            arrowprops=dict(arrowstyle='->', color='#aa3366', lw=1.3, linestyle='dashed'), zorder=3)
# Dashed line from KV Projection box down to K,V side node
ax.plot([CX+BW/2+pad+0.62, kvx], [kv_y, kv_y], color='#aa3366', lw=1.2, linestyle='--', zorder=1)
ax.plot([kvx, kvx], [kv_y, kv_box_y+0.38], color='#aa3366', lw=1.2, linestyle='--', zorder=1)

y -= BHT/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'SwiGLU FFN', C_FFN,
    sublabel='RMSNorm → SwiGLU Feed-Forward  (residual add)  →  transformer_out')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'Linear Time-Invariant Injection (LTI)', C_LTI,
    sublabel='h = sigmoid(a) · h_input + transformer_out')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'HyperConnection Buffer Update', C_HYPER,
    sublabel='Push h to front of ring buffer, drop oldest state')
loop_bot = y - BH/2 - GAP*0.35

section_box(ax, CX, loop_top, loop_bot, BW, '#f8f4ff', '#7755aa', '×R  Recurrent Core', '#7755aa')

# Loop back arrow
rx = CX + BW/2 + pad + 0.55
ry_bot = loop_bot + 0.18
ry_top = loop_top - 0.18
ax.annotate('', xy=(rx, ry_top), xytext=(rx, ry_bot),
            arrowprops=dict(arrowstyle='->', color='#7755aa', lw=1.8))
ax.plot([CX+BW/2+pad, rx], [ry_bot, ry_bot], color='#7755aa', lw=1.8)
ax.plot([CX+BW/2+pad, rx], [ry_top, ry_top], color='#7755aa', lw=1.8)
ax.text(rx, (ry_bot+ry_top)/2, '× R loops',
        ha='center', va='center', fontsize=12, fontweight='bold',
        color='#7755aa', rotation=90,
        bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=4))

y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# ── CODA ×1 layer ────────────────────────────────────────────────────────────
coda_top = y + BH/2 + GAP*0.35

box(ax, CX, y, BW, BH, 'Multi-head Latent Attention', C_ATTN,
    sublabel='RMSNorm → MLA Self-Attention + RoPE  (residual add)')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'SwiGLU FFN', C_FFN,
    sublabel='RMSNorm → SwiGLU Feed-Forward  (residual add)')
coda_bot = y - BH/2 - GAP*0.35

section_box(ax, CX, coda_top, coda_bot, BW, '#fff8f0', '#cc6600', '×1  Coda', '#cc6600')

y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# Output Head
box(ax, CX, y, BW, BH, 'Language Model Head', C_LMH,
    sublabel='RMSNorm → Linear projection to vocab logits  (weight-tied to embedding)')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# Logits
box(ax, CX, y, BW, BH, 'Logits  →  Next Token', C_IO)

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(CX, 39.40, 'CART Architecture',
        ha='center', va='center', fontsize=20, fontweight='bold',
        fontfamily='monospace', color='#111111')
ax.text(CX, 38.95, 'Context-Anchored Recurrent Transformer',
        ha='center', va='center', fontsize=12, color='#555555', style='italic')

plt.tight_layout(pad=0.3)
plt.savefig('figures/cart_architecture.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('figures/cart_architecture.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved.")
