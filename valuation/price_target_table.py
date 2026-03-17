"""
=============================================================================
Price Target Scenario Table — Styled like investment memo
=============================================================================
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = "."

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
current_mc    = 0.74   # $B
current_price = 0.94   # $

scenarios = [
    # (Case, Valuation Basis, Implied MC $M, Implied Price, % Change, Probability)
    ("Bull", "1.6x Sui MC / Revenue", 301,  0.81, -14, 0.20),
    ("Base", "1.2x Sui MC / Revenue", 225,  0.61, -35, 0.55),
    ("Bear", "1.0x Sui MC / Revenue", 188,  0.51, -46, 0.25),
]

prob_weighted_price = sum(p * price for _, _, _, price, _, p in scenarios)

# ─────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────
ROW_COLORS = {
    "Bull": "#f4a68a",   # salmon/orange
    "Base": "#a8d5a2",   # light green
    "Bear": "#6fbf73",   # darker green
}
HEADER_BG  = "#1a1a2e"
HEADER_FG  = "white"
CURRENT_BG = "#dce8f7"
FOOTER_BG  = "#f0f0f0"
TEXT_DARK  = "#111111"

# ─────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axis('off')

fig.patch.set_facecolor('white')

# ── Column definitions ───────────────────────────────────────
col_labels = ["Case", "Valuation Basis", "Implied MC\n(circulating)", "Implied Price\n(APT)", "% Change\nvs. current", "Probability"]
col_widths = [0.09, 0.28, 0.16, 0.16, 0.16, 0.13]
n_cols = len(col_labels)

# Layout params
x_starts = []
x = 0.01
for w in col_widths:
    x_starts.append(x)
    x += w

row_height = 0.13
header_y   = 0.75
data_start = header_y - row_height

def draw_cell(ax, x, y, w, h, text, bg, fg=TEXT_DARK, fontsize=10, bold=False, align='center'):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        linewidth=0.5, edgecolor='white',
        facecolor=bg,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        transform=ax.transAxes,
        ha=align, va='center',
        fontsize=fontsize,
        fontweight='bold' if bold else 'normal',
        color=fg,
        wrap=True
    )

# ── Title ────────────────────────────────────────────────────
ax.text(0.5, 0.97, "Price Target Scenario Table",
        transform=ax.transAxes, ha='center', va='top',
        fontsize=14, fontweight='bold', color=TEXT_DARK)

# ── Header row ───────────────────────────────────────────────
for i, (label, w) in enumerate(zip(col_labels, col_widths)):
    draw_cell(ax, x_starts[i], header_y, w, row_height,
              label, HEADER_BG, fg=HEADER_FG, fontsize=9, bold=True)

# ── Scenario rows ─────────────────────────────────────────────
for row_idx, (case, basis, mc_m, price, pct, prob) in enumerate(scenarios):
    y = data_start - row_idx * row_height
    bg = ROW_COLORS[case]
    row_data = [
        case,
        basis,
        f"${mc_m}M",
        f"${price:.2f}",
        f"{pct}%",
        f"{int(prob*100)}%"
    ]
    for i, (val, w) in enumerate(zip(row_data, col_widths)):
        bold = i == 0 or i == 3
        draw_cell(ax, x_starts[i], y, w, row_height, val, bg,
                  fontsize=10, bold=bold)

# ── Current row ───────────────────────────────────────────────
current_y = data_start - 3 * row_height
current_data = ["Current", f"Market Cap = ${current_mc}B", "—", f"${current_price:.2f}", "—", "—"]
for i, (val, w) in enumerate(zip(current_data, col_widths)):
    draw_cell(ax, x_starts[i], current_y, w, row_height, val,
              CURRENT_BG, fontsize=10, bold=(i == 0))

# ── Probability-Adjusted Return row ───────────────────────────
footer_y = data_start - 4 * row_height
footer_texts = ["Prob. Adjusted\nReturn", "", "", f"~${prob_weighted_price:.2f}", "", ""]
total_w_left = sum(col_widths[:3])
total_w_right = sum(col_widths[3:])

draw_cell(ax, x_starts[0], footer_y, total_w_left, row_height,
          "Probability-Weighted Expected Price",
          FOOTER_BG, fontsize=10, bold=True)
draw_cell(ax, x_starts[3], footer_y, total_w_right, row_height,
          f"~${prob_weighted_price:.2f}",
          FOOTER_BG, fontsize=11, bold=True, fg="#c0392b")

# ── Outer border ─────────────────────────────────────────────
total_h = row_height * 6
border = mpatches.FancyBboxPatch(
    (x_starts[0], footer_y), sum(col_widths), total_h,
    boxstyle="square,pad=0",
    linewidth=1.2, edgecolor='#aaaaaa',
    facecolor='none',
    transform=ax.transAxes, clip_on=False
)
ax.add_patch(border)

# ── Footer note ───────────────────────────────────────────────
ax.text(
    0.5, footer_y - 0.06,
    f"Valuation based on peer MC/Revenue multiples (Sui). "
    f"Probability-weighted expected price: ~${prob_weighted_price:.2f}  |  "
    f"Implied downside from current: {((prob_weighted_price / current_price) - 1)*100:.0f}%",
    transform=ax.transAxes, ha='center', va='top',
    fontsize=8.5, color='#555555', style='italic'
)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "price_target_table.png")
plt.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ Saved {out}")
