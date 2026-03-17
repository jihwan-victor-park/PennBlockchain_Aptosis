"""
=============================================================================
Implied Revenue Required to Justify Aptos's Current Valuation
=============================================================================
Chart: bar chart comparing current revenue vs required revenue at peer multiples
Section: Valuation & TAM
=============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR   = "datas"
OUTPUT_DIR = "."
DAYS_30    = 30

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
def load(fname, col):
    df = pd.read_csv(os.path.join(DATA_DIR, fname), encoding='utf-8-sig')
    df.columns = ['date', col]
    df['date'] = pd.to_datetime(df['date'])
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().sort_values('date').set_index('date')

print("📂 Loading data...")

apt_mc      = load("Aptos - Market Cap.csv",              "mc")
apt_rev     = load("Aptos - Revenue.csv",                 "rev")
sui_mc_rev  = load("Sui - MC  Revenue Annualized.csv",    "v")
sol_mc_rev  = load("Solana - MC  Revenue Annualized.csv", "v")

# ─────────────────────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────────────────────
apt_mc_now      = apt_mc["mc"].dropna().iloc[-1]                        # USD
apt_rev_30d_avg = apt_rev["rev"].rolling(DAYS_30).mean().dropna().iloc[-1]  # USD/day
apt_rev_ann     = apt_rev_30d_avg * 365                                 # USD/yr

sui_multiple    = sui_mc_rev["v"].dropna().iloc[-1]                     # x
sol_multiple    = sol_mc_rev["v"].dropna().iloc[-1]                     # x

# Implied required revenue = Market Cap / Peer Multiple
req_rev_sui = apt_mc_now / sui_multiple
req_rev_sol = apt_mc_now / sol_multiple

# Convert to $M for readability
cur_M   = apt_rev_ann   / 1e6
req_sui_M = req_rev_sui / 1e6
req_sol_M = req_rev_sol / 1e6

mult_sui = req_sui_M / cur_M
mult_sol = req_sol_M / cur_M

print(f"  APT Market Cap:              ${apt_mc_now/1e9:.2f}B")
print(f"  APT Ann. Revenue (30D avg):  ${cur_M:.2f}M")
print(f"  SUI MC/Rev multiple:         {sui_multiple:.1f}x")
print(f"  SOL MC/Rev multiple:         {sol_multiple:.1f}x")
print(f"  Required Rev @ SUI multiple: ${req_sui_M:.2f}M  ({mult_sui:.1f}x current)")
print(f"  Required Rev @ SOL multiple: ${req_sol_M:.2f}M  ({mult_sol:.1f}x current)")

# ─────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────
labels = [
    "Current APT\nRevenue",
    f"Required @ SUI\nMultiple ({sui_multiple:.0f}x)",
    f"Required @ SOL\nMultiple ({sol_multiple:.0f}x)",
]
values  = [cur_M, req_sui_M, req_sol_M]
colors  = ["#2ecc71", "#e67e22", "#c0392b"]
alphas  = [1.0, 0.85, 0.85]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor='black',
              linewidth=0.8, width=0.5)

# Value labels on top of bars
for bar, val, mult in zip(bars, values, [1.0, mult_sui, mult_sol]):
    y = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y * 1.03,
        f"${val:.2f}M",
        ha='center', va='bottom',
        fontsize=11, fontweight='bold'
    )
    if mult > 1.0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y * 0.5,
            f"{mult:.1f}x\ncurrent",
            ha='center', va='center',
            fontsize=10, fontweight='bold', color='white'
        )

# Reference line for current revenue
ax.axhline(cur_M, color='#2ecc71', linewidth=1.4, linestyle='--', alpha=0.6,
           label=f'Current Revenue (${cur_M:.2f}M/yr)')

ax.set_title(
    "Revenue Required to Justify Aptos's Current Valuation",
    fontsize=15, fontweight='bold', pad=14
)
ax.set_ylabel("Annualized Revenue (USD Millions)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.1fM'))
ax.grid(axis='y', alpha=0.2)
ax.legend(fontsize=9, frameon=False)

# Subtitle annotation
ax.text(
    0.5, -0.16,
    f"At current market cap of ${apt_mc_now/1e9:.2f}B, Aptos would need "
    f"~{mult_sui:.1f}x current revenue to trade in line with Sui's MC/Revenue multiple,\n"
    f"and ~{mult_sol:.1f}x to match Solana's. Current annualized revenue: ${cur_M:.2f}M.",
    transform=ax.transAxes,
    ha='center', va='top',
    fontsize=9, color='#444444',
    style='italic'
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
out = os.path.join(OUTPUT_DIR, "implied_revenue.png")
plt.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✅ Saved {out}")
