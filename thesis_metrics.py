"""
=============================================================================
APT Short Thesis — Comprehensive Metrics Report
=============================================================================
Outputs: thesis_metrics.txt
=============================================================================
"""
import pandas as pd
import numpy as np
import os
import requests
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "datas"
OUT_FILE = "thesis_metrics.txt"
DAYS_30  = 30

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def load(fname, col):
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = ['date', col]
    df['date'] = pd.to_datetime(df['date'])
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().sort_values('date').set_index('date')

def latest(df, col):
    return df[col].dropna().iloc[-1]

def ma30(df, col):
    return df[col].rolling(DAYS_30).mean().dropna().iloc[-1]

def fmt(val, prefix='', suffix='', decimals=2):
    if pd.isna(val):
        return 'N/A'
    return f"{prefix}{val:,.{decimals}f}{suffix}"

# ─────────────────────────────────────────────────────────────
# 1. APT SPOT METRICS
# ─────────────────────────────────────────────────────────────
print("📂 Loading spot metrics...")
apt_price  = load("Aptos - Price (1).csv",               "price")
apt_mc     = load("Aptos - Market Cap.csv",              "mc")
apt_fdv    = load("Aptos - Fully Diluted Market Cap.csv","fdv")
apt_supply = load("Aptos - Circulating Supply.csv",      "supply")

apt_price_now  = latest(apt_price,  "price")
apt_mc_now     = latest(apt_mc,     "mc")    / 1e9
apt_fdv_now    = latest(apt_fdv,    "fdv")   / 1e9
apt_supply_now = latest(apt_supply, "supply")/ 1e9

# ─────────────────────────────────────────────────────────────
# 2. MC/REVENUE & MC/FEES (latest value)
# ─────────────────────────────────────────────────────────────
print("📂 Loading multiples...")
apt_mc_rev  = load("Aptos - MC  Revenue Annualized.csv", "v")
apt_mc_fees = load("Aptos - MC  Fees Annualized.csv",    "v")
sui_mc_rev  = load("Sui - MC  Revenue Annualized.csv",   "v")
sui_mc_fees = load("Sui - MC  Fees Annualized.csv",      "v")
sol_mc_rev  = load("Solana - MC  Revenue Annualized.csv","v")
sol_mc_fees = load("Solana - MC  Fees Annualized.csv",   "v")

# ─────────────────────────────────────────────────────────────
# 3. REVENUE & FEES (30D avg)
# ─────────────────────────────────────────────────────────────
print("📂 Loading revenue/fees...")
apt_rev = load("Aptos - Revenue.csv",  "rev")
sui_rev = load("Sui - Revenue.csv",    "rev")
sol_rev = load("Solana - FDMC  Revenue.csv", "rev")   # proxy

apt_rev_30d = ma30(apt_rev, "rev")
sui_rev_30d = ma30(sui_rev, "rev")
sol_rev_30d = ma30(sol_rev, "rev")

apt_rev_ann = apt_rev_30d * 365
sui_rev_ann = sui_rev_30d * 365
sol_rev_ann = sol_rev_30d * 365

# ─────────────────────────────────────────────────────────────
# 4. NVT  (Market Cap / 30D MA Transactions)
# ─────────────────────────────────────────────────────────────
print("📂 Computing NVT...")
apt_tx  = load("Aptos - Chain Transactions.csv", "tx")
sui_tx  = load("Sui - Chain Transactions.csv",   "tx")
sol_tx  = load("Solana - Chain Transactions.csv","tx")

def compute_nvt(mc_df, tx_df):
    merged = mc_df.join(tx_df, how='inner')
    merged['tx_ma'] = merged['tx'].rolling(DAYS_30).mean()
    merged['nvt']   = merged['mc'] / merged['tx_ma']
    current = merged['nvt'].dropna().iloc[-1]
    median  = merged['nvt'].dropna().median()
    return current, median

apt_nvt_cur, apt_nvt_med = compute_nvt(apt_mc.rename(columns={'mc':'mc'}), apt_tx)
sui_nvt_cur, _           = compute_nvt(
    load("Sui - Market Cap.csv", "mc"), sui_tx)
sol_nvt_cur, _           = compute_nvt(
    load("Solana - Market Cap.csv", "mc"), sol_tx)

# ─────────────────────────────────────────────────────────────
# 5. OLS RESULTS  (parse from ols_summary.txt)
# ─────────────────────────────────────────────────────────────
print("📂 Parsing OLS summary...")
ols_alpha = ols_tstat = ols_r2 = None
try:
    with open("ols_summary.txt") as f:
        txt = f.read()
    for line in txt.splitlines():
        if 'const' in line and 'coef' not in line:
            parts = line.split()
            if len(parts) >= 4:
                ols_alpha = float(parts[1])
                ols_tstat = float(parts[3])
        if 'R-squared:' in line and 'Adj' not in line:
            ols_r2 = float(line.split()[-1])
except:
    pass

# ─────────────────────────────────────────────────────────────
# 6. HMM RESULTS  (from hmm_regime_summary.xlsx)
# ─────────────────────────────────────────────────────────────
print("📂 Loading HMM results...")
hmm_bull_ret = hmm_bear_ret = hmm_latest = None
try:
    hmm = pd.read_excel("hmm_regime_summary.xlsx", sheet_name="APT vs Market by Regime")
    bull_row = hmm[hmm['Market Regime'] == 'Bullish'].iloc[0]
    bear_row = hmm[hmm['Market Regime'] == 'Bearish'].iloc[0]
    hmm_bull_ret = bull_row['APT Avg Daily Return (%)']
    hmm_bear_ret = bear_row['APT Avg Daily Return (%)']

    zoomed = pd.read_excel("hmm_regime_summary.xlsx", sheet_name="Zoomed Daily Data")
    hmm_latest = zoomed['mkt_regime_2'].iloc[-1]
except:
    pass

# ─────────────────────────────────────────────────────────────
# 7. EXECUTION: Binance funding rate + 24h volume
# ─────────────────────────────────────────────────────────────
print("🌐 Fetching Binance funding rate & volume...")
funding_rate = daily_volume_usd = None
try:
    r = requests.get(
        "https://fapi.binance.com/fapi/v1/premiumIndex",
        params={"symbol": "APTUSDT"}, timeout=8
    )
    data = r.json()
    funding_rate = float(data.get("lastFundingRate", 0)) * 100   # in %
except:
    pass

try:
    r = requests.get(
        "https://fapi.binance.com/fapi/v1/ticker/24hr",
        params={"symbol": "APTUSDT"}, timeout=8
    )
    data = r.json()
    daily_volume_usd = float(data.get("quoteVolume", 0)) / 1e6   # in $M
except:
    pass

# ─────────────────────────────────────────────────────────────
# BUILD REPORT
# ─────────────────────────────────────────────────────────────
lines = []
def h(title):
    lines.append("")
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)
def row(label, val):
    lines.append(f"  {label:<40} {val}")

h("APT SPOT METRICS")
row("Current Price",           fmt(apt_price_now,  prefix='$'))
row("Market Cap",              fmt(apt_mc_now,      prefix='$', suffix='B'))
row("Fully Diluted Market Cap (FDV)", fmt(apt_fdv_now, prefix='$', suffix='B'))
row("Circulating Supply",      fmt(apt_supply_now,  suffix='B APT'))

h("VALUATION MULTIPLES (latest)")
row("MC / Revenue Annualized — APT", fmt(latest(apt_mc_rev,  'v'), suffix='x', decimals=1))
row("MC / Revenue Annualized — SUI", fmt(latest(sui_mc_rev,  'v'), suffix='x', decimals=1))
row("MC / Revenue Annualized — SOL", fmt(latest(sol_mc_rev,  'v'), suffix='x', decimals=1))
lines.append("")
row("MC / Fees Annualized — APT",    fmt(latest(apt_mc_fees, 'v'), suffix='x', decimals=1))
row("MC / Fees Annualized — SUI",    fmt(latest(sui_mc_fees, 'v'), suffix='x', decimals=1))
row("MC / Fees Annualized — SOL",    fmt(latest(sol_mc_fees, 'v'), suffix='x', decimals=1))

h("REVENUE (30D avg daily / annualized)")
row("APT", f"${apt_rev_30d:,.0f}/day  |  ${apt_rev_ann/1e6:.2f}M/yr")
row("SUI", f"${sui_rev_30d:,.0f}/day  |  ${sui_rev_ann/1e6:.2f}M/yr")
row("SOL", f"${sol_rev_30d:,.0f}/day  |  ${sol_rev_ann/1e6:.2f}M/yr")

h("NVT RATIO  (Market Cap / 30D MA Tx Count)")
row("APT current NVT",         fmt(apt_nvt_cur, decimals=1))
row("APT historical median NVT",fmt(apt_nvt_med, decimals=1))
row("SUI current NVT",         fmt(sui_nvt_cur, decimals=1))
row("SOL current NVT",         fmt(sol_nvt_cur, decimals=1))

h("OLS RETURN ATTRIBUTION")
row("Alpha (daily, annualized)",
    f"{ols_alpha:.4f}  ({ols_alpha*365*100:.1f}% / yr)" if ols_alpha is not None else 'N/A')
row("Alpha t-stat",            fmt(ols_tstat, decimals=3) if ols_tstat else 'N/A')
row("R-squared",               fmt(ols_r2, decimals=3) if ols_r2 else 'N/A')
row("Interpretation",
    "Negative alpha = structural underperformer vs market")

h("HMM REGIME ANALYSIS")
row("APT avg daily return — Bullish regime",
    f"{hmm_bull_ret:.3f}%" if hmm_bull_ret is not None else 'N/A')
row("APT avg daily return — Bearish regime",
    f"{hmm_bear_ret:.3f}%" if hmm_bear_ret is not None else 'N/A')
row("Latest market regime",    hmm_latest if hmm_latest else 'N/A')

h("EXECUTION")
row("Exchange",                "Binance (APTUSDT Perpetual)")
row("Funding rate (latest)",
    f"{funding_rate:.4f}% per 8h  ({funding_rate*3*365:.1f}% annualized)"
    if funding_rate is not None else 'N/A')
row("Funding direction",
    ("Positive → shorts receive funding ✅" if funding_rate and funding_rate > 0
     else "Negative → shorts pay funding ⚠️") if funding_rate is not None else 'N/A')
row("24h Perp Volume",
    f"${daily_volume_usd:.1f}M" if daily_volume_usd is not None else 'N/A')
row("Recommended execution",   "CEX Perp short (Binance/OKX), layered entry")

lines.append("")
lines.append("=" * 60)
lines.append("Generated by thesis_metrics.py")
lines.append("=" * 60)

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
report = "\n".join(lines)
print(report)

with open(OUT_FILE, 'w') as f:
    f.write(report)

print(f"\n✅ Saved {OUT_FILE}")
