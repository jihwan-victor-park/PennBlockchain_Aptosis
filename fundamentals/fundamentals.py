"""
=============================================================================
Aptos Fundamental Deterioration Analysis (Improved Thesis Version)
=============================================================================
Charts:
  1. Transactions        - smoothed activity trend
  2. TVL                 - capital destruction
  3. Developers          - dev activity deterioration
  4. Valuation Disconnect - Indexed Price vs Fees
  5. Supply Pressure      - Circulating supply growth
=============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
DATA_DIR    = "datas"
OUTPUT_DIR  = "."
ROLL        = 30
DEV_ROLL    = 8   # 8-week MA for weekly developers

# --- FILES ---
CORE_METRICS = {
    'Transactions': 'Aptos - Chain Transactions.csv',
    'TVL':          'Aptos - Chain TVL.csv',
    'Developers':   'Aptos - Weekly Core Active Developers.csv',
}
VALUATION_FILES = {
    'Price': 'Aptos - Price (1).csv',
    'Fees':  'Aptos - Revenue.csv',
}
SUPPLY_FILE = 'Aptos - Circulating Supply.csv'

# --- COLORS ---
NAVY   = '#1a3a5c'
BLUE   = '#aec6e8'
RED    = '#c0392b'
ORANGE = '#e67e22'
GREEN  = '#27ae60'
PURPLE = '#8e44ad'
DARK_PURPLE = '#4a235a'
GRAY   = '#666666'

# --- HELPERS ---
def load_csv(filepath, col_name):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = ['date', col_name]
    df['date'] = pd.to_datetime(df['date'])
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    return df.dropna().sort_values('date').set_index('date')

def winsorize_series(s, low_q=0.01, high_q=0.99):
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    return s.clip(lower=lo, upper=hi)

def pct_change_from_peak(series):
    peak = series.max()
    latest = series.dropna().iloc[-1]
    return (latest / peak - 1) * 100

def pct_change_from_start(series):
    start = series.dropna().iloc[0]
    latest = series.dropna().iloc[-1]
    return (latest / start - 1) * 100

# --- LOAD DATA ---
print("📂 Loading data...")
core_dfs = {}
for name, fname in CORE_METRICS.items():
    core_dfs[name] = load_csv(os.path.join(DATA_DIR, fname), name)

price_df = load_csv(os.path.join(DATA_DIR, VALUATION_FILES['Price']), 'Price')
fee_df   = load_csv(os.path.join(DATA_DIR, VALUATION_FILES['Fees']),  'Fees')
sup_df   = load_csv(os.path.join(DATA_DIR, SUPPLY_FILE), 'Circulating Supply')

print("✅ All files loaded")

# --- FIGURE LAYOUT ---
fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    'Aptos: Weakening Fundamentals, Valuation Disconnect, and Supply Overhang',
    fontsize=20, fontweight='bold', y=0.985
)

ax_tx  = fig.add_subplot(2, 3, 1)
ax_tvl = fig.add_subplot(2, 3, 2)
ax_dev = fig.add_subplot(2, 3, 3)
ax_val = fig.add_subplot(2, 2, 3)
ax_sup = fig.add_subplot(2, 2, 4)

# ── 1. TRANSACTIONS ──────────────────────────────────────────
df = core_dfs['Transactions'].copy()
df['tx_wins'] = winsorize_series(df['Transactions'], 0.01, 0.99)
df['tx_ma'] = df['tx_wins'].rolling(ROLL).mean()

ax_tx.plot(df.index, df['tx_wins'], color=BLUE, alpha=0.35, lw=1.0, label='Daily Tx (winsorized)')
ax_tx.plot(df.index, df['tx_ma'], color=NAVY, lw=2.2, label=f'{ROLL}D MA')
ax_tx.set_title('Transactions Trend', fontsize=13, fontweight='bold')
ax_tx.set_ylabel('Transactions')
ax_tx.legend(fontsize=8, frameon=False)
ax_tx.grid(alpha=0.2)
ax_tx.set_yscale('log')
ax_tx.text(
    0.02, 0.05,
    "Daily activity remains noisy,\nbut smoothed usage shows no sustained breakout",
    transform=ax_tx.transAxes,
    fontsize=8.5, color=GRAY,
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')
)
ax_tx.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax_tx.xaxis.get_majorticklabels(), rotation=35, ha='right')

# ── 2. TVL ───────────────────────────────────────────────────
df = core_dfs['TVL'].copy()
tvl_b = df['TVL'] / 1e9
tvl_drawdown = pct_change_from_peak(tvl_b)

ax_tvl.fill_between(tvl_b.index, tvl_b, alpha=0.25, color=GREEN)
ax_tvl.plot(tvl_b.index, tvl_b, color=GREEN, lw=2.2)
ax_tvl.set_title('TVL Trend', fontsize=13, fontweight='bold')
ax_tvl.set_ylabel('TVL (USD Billions)')
ax_tvl.grid(alpha=0.2)
ax_tvl.text(
    0.02, 0.05,
    f"TVL is down {abs(tvl_drawdown):.0f}% from peak,\nindicating persistent capital exit",
    transform=ax_tvl.transAxes,
    fontsize=8.5, color='darkgreen',
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')
)
ax_tvl.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax_tvl.xaxis.get_majorticklabels(), rotation=35, ha='right')

# ── 3. DEVELOPERS ────────────────────────────────────────────
df = core_dfs['Developers'].copy()
df['dev_ma'] = df['Developers'].rolling(DEV_ROLL).mean()
dev_change = pct_change_from_peak(df['dev_ma'])

ax_dev.bar(df.index, df['Developers'], color=PURPLE, alpha=0.28, width=6, label='Weekly Devs')
ax_dev.plot(df.index, df['dev_ma'], color=DARK_PURPLE, lw=2.2, label=f'{DEV_ROLL}W MA')
ax_dev.set_title('Core Active Developers', fontsize=13, fontweight='bold')
ax_dev.set_ylabel('Developers')
ax_dev.legend(fontsize=8, frameon=False)
ax_dev.grid(alpha=0.2)
ax_dev.text(
    0.02, 0.05,
    f"Core developer activity has fallen ~{abs(dev_change):.0f}% from peak",
    transform=ax_dev.transAxes,
    fontsize=8.5, color=DARK_PURPLE,
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')
)
ax_dev.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax_dev.xaxis.get_majorticklabels(), rotation=35, ha='right')

# ── 4. VALUATION DISCONNECT (indexed, more intuitive) ───────
val = price_df.join(fee_df, how='inner').copy()
val['fees_ma'] = val['Fees'].rolling(30).mean()
val = val.dropna()

val['price_idx'] = val['Price'] / val['Price'].iloc[0] * 100
val['fees_idx']  = val['fees_ma'] / val['fees_ma'].iloc[0] * 100

ax_val.plot(val.index, val['price_idx'], color=NAVY, lw=2.3, label='APT Price Index (100=Start)')
ax_val.plot(val.index, val['fees_idx'], color=RED, lw=2.0, linestyle='--', label='Fees 30D MA Index (100=Start)')

ax_val.set_title(
    'Valuation Disconnect\nPrice Remains Better Supported Than Network Revenue',
    fontsize=13, fontweight='bold'
)
ax_val.set_ylabel('Indexed Level (100 = Start)')
ax_val.grid(alpha=0.18)
ax_val.legend(fontsize=8, loc='upper right', frameon=False)

gap_latest = val['price_idx'].iloc[-1] - val['fees_idx'].iloc[-1]
ax_val.text(
    0.02, 0.05,
    f"Revenue has compressed more sharply than price,\nleaving a valuation-fundamental gap",
    transform=ax_val.transAxes,
    fontsize=8.5, color='darkred',
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')
)
ax_val.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax_val.xaxis.get_majorticklabels(), rotation=35, ha='right')

# ── 5. SUPPLY PRESSURE ───────────────────────────────────────
sup_b = sup_df['Circulating Supply'] / 1e9
supply_growth = pct_change_from_start(sup_b)

ax_sup.fill_between(sup_b.index, sup_b, alpha=0.22, color=ORANGE)
ax_sup.plot(sup_b.index, sup_b, color=ORANGE, lw=2.4)
ax_sup.set_title('Supply Pressure', fontsize=13, fontweight='bold')
ax_sup.set_ylabel('Circulating Supply (Billions APT)')
ax_sup.grid(alpha=0.2)
ax_sup.text(
    0.02, 0.84,
    f"Circulating supply has risen {supply_growth:.0f}%\nsince the start of the sample",
    transform=ax_sup.transAxes,
    fontsize=8.5, color='darkorange',
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')
)
ax_sup.text(
    0.02, 0.06,
    "A growing float creates ongoing sell pressure\nand limits multiple expansion",
    transform=ax_sup.transAxes,
    fontsize=8.5, color='saddlebrown',
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')
)
ax_sup.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax_sup.xaxis.get_majorticklabels(), rotation=35, ha='right')

# --- SAVE FIGURE ---
plt.tight_layout(rect=[0, 0, 1, 0.965])
out = os.path.join(OUTPUT_DIR, 'fundamental_dashboard_improved.png')
plt.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ Saved {out}")

# --- SUMMARY TABLES ---
print("\n📉 Quarter-over-Quarter Summary...")
all_core = pd.concat([v for v in core_dfs.values()], axis=1)
df_q = all_core.resample('QE').mean()
qoq = df_q.pct_change() * 100

print("=" * 60)
print(df_q.iloc[-4:].round(1).to_string())
print("\nQoQ Change (%):")
print(qoq.iloc[-4:].round(1).to_string())
print("=" * 60)

with open(os.path.join(OUTPUT_DIR, 'fundamental_summary.txt'), 'w') as f:
    f.write("Quarterly Averages:\n")
    f.write(df_q.iloc[-4:].round(1).to_string())
    f.write("\n\nQoQ Change (%):\n")
    f.write(qoq.iloc[-4:].round(1).to_string())
    f.write("\n\nKey diagnostic summary:\n")
    f.write(f"\nTVL drawdown from peak: {tvl_drawdown:.1f}%")
    f.write(f"\nDeveloper decline from peak (smoothed): {dev_change:.1f}%")
    f.write(f"\nCirculating supply growth since start: {supply_growth:.1f}%")
print("✅ Saved fundamental_summary.txt")