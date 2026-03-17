"""
=============================================================================
Smoothed NVT Ratio Comparison Chart for APT Short Thesis
=============================================================================
Goal:
- Keep peer comparison (APT vs SOL vs SUI)
- Remove noise
- Make chart thesis-friendly
- Highlight whether APT trades at richer valuation vs peers
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "datas"
OUTPUT_DIR = "."
START_DATE = "2023-01-01"   # focus on APT live history
END_DATE = None             # or set e.g. "2026-03-01"

# NVT settings
TX_MA_WINDOW = 90           # 90-day MA transaction volume
NVT_SMOOTH_WINDOW = 21      # smooth final NVT with 21D median
CLIP_Q_LOW = 0.01
CLIP_Q_HIGH = 0.99
USE_LOG_SCALE = True

# Assets to compare
ASSETS = {
    "apt": {
        "label": "Aptos",
        "price_file": "Aptos - Price (1).csv",
        "mc_file": "Aptos - Market Cap.csv",
        "tx_file": "Aptos - Chain Transactions.csv",
        "color": "#00529B",
        "lw": 2.8,
        "alpha": 1.0
    },
    "sol": {
        "label": "Solana",
        "price_file": "Solana - Price.csv",
        "mc_file": "Solana - Market Cap.csv",
        "tx_file": "Solana - Chain Transactions.csv",
        "color": "#6AA84F",
        "lw": 1.8,
        "alpha": 0.85
    },
    "sui": {
        "label": "Sui",
        "price_file": "Sui - Price.csv",
        "mc_file": "Sui - Market Cap.csv",
        "tx_file": "Sui - Chain Transactions.csv",
        "color": "#F6B26B",
        "lw": 1.8,
        "alpha": 0.85
    }
}

# -----------------------------
# HELPERS
# -----------------------------
def load_artemis_csv(filepath, value_name):
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = ["date", value_name]
    df["date"] = pd.to_datetime(df["date"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df

def winsorize_series(s, low_q=0.01, high_q=0.99):
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    return s.clip(lower=lo, upper=hi)

# -----------------------------
# LOAD AND MERGE
# -----------------------------
print("📂 Loading NVT data...")

all_dfs = []
for asset, meta in ASSETS.items():
    mc_df = load_artemis_csv(os.path.join(DATA_DIR, meta["mc_file"]), f"{asset}_mc")
    tx_df = load_artemis_csv(os.path.join(DATA_DIR, meta["tx_file"]), f"{asset}_tx")
    all_dfs.extend([mc_df, tx_df])

df = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), all_dfs)
df = df.sort_values("date").set_index("date")

if START_DATE:
    df = df[df.index >= pd.to_datetime(START_DATE)]
if END_DATE:
    df = df[df.index <= pd.to_datetime(END_DATE)]

# -----------------------------
# CALCULATE NVT
# -----------------------------
print("🛠️ Calculating smoothed NVT ratios...")

for asset in ASSETS.keys():
    mc_col = f"{asset}_mc"
    tx_col = f"{asset}_tx"

    # 90D MA of transaction volume
    df[f"{asset}_tx_90dma"] = df[tx_col].rolling(TX_MA_WINDOW, min_periods=TX_MA_WINDOW // 2).mean()

    # raw NVT
    df[f"{asset}_nvt_raw"] = df[mc_col] / df[f"{asset}_tx_90dma"]

    # remove weird zeros / negatives / infinities
    df[f"{asset}_nvt_raw"] = df[f"{asset}_nvt_raw"].replace([np.inf, -np.inf], np.nan)
    df.loc[df[f"{asset}_nvt_raw"] <= 0, f"{asset}_nvt_raw"] = np.nan

    # winsorize per asset to reduce spikes
    valid = df[f"{asset}_nvt_raw"].dropna()
    if len(valid) > 20:
        clipped = winsorize_series(valid, CLIP_Q_LOW, CLIP_Q_HIGH)
        df.loc[clipped.index, f"{asset}_nvt_clipped"] = clipped
    else:
        df[f"{asset}_nvt_clipped"] = df[f"{asset}_nvt_raw"]

    # smooth final NVT with rolling median
    df[f"{asset}_nvt_smooth"] = (
        df[f"{asset}_nvt_clipped"]
        .rolling(NVT_SMOOTH_WINDOW, min_periods=max(5, NVT_SMOOTH_WINDOW // 3))
        .median()
    )

# Optional: drop rows where all are nan
smooth_cols = [f"{a}_nvt_smooth" for a in ASSETS.keys()]
df = df.dropna(subset=smooth_cols, how="all")

# -----------------------------
# SUMMARY STATS
# -----------------------------
print("\n📊 Summary stats (smoothed NVT):")
for asset, meta in ASSETS.items():
    col = f"{asset}_nvt_smooth"
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    print(
        f"{meta['label']}: median={vals.median():.1f}, "
        f"mean={vals.mean():.1f}, latest={vals.iloc[-1]:.1f}"
    )

# -----------------------------
# PLOT 1: Main comparison
# -----------------------------
print("📈 Generating thesis-friendly NVT comparison chart...")

fig, ax = plt.subplots(figsize=(14, 7))

for asset, meta in ASSETS.items():
    col = f"{asset}_nvt_smooth"
    ax.plot(
        df.index,
        df[col],
        label=f"{meta['label']} NVT",
        color=meta["color"],
        linewidth=meta["lw"],
        alpha=meta["alpha"]
    )

# Horizontal median lines for context
for asset, meta in ASSETS.items():
    col = f"{asset}_nvt_smooth"
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    med = vals.median()
    ax.axhline(
        med,
        color=meta["color"],
        linestyle="--",
        linewidth=1,
        alpha=0.25
    )

if USE_LOG_SCALE:
    ax.set_yscale("log")

ax.set_title(
    "Smoothed NVT Ratio Comparison: Aptos vs. Peers\n"
    "(90D MA Transaction Volume, 21D Median-Smoothed NVT)",
    fontsize=16,
    fontweight="bold"
)
ax.set_ylabel("NVT Ratio" + (" (Log Scale)" if USE_LOG_SCALE else ""))
ax.set_xlabel("Date")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=30)

ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
ax.legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "nvt_comparison_smoothed.png"), dpi=220)
plt.close()
print("✅ Saved nvt_comparison_smoothed.png")

# -----------------------------
# PLOT 2: Relative NVT (APT / SOL, APT / SUI)
# -----------------------------
print("📈 Generating relative NVT chart...")

df["apt_vs_sol_nvt"] = df["apt_nvt_smooth"] / df["sol_nvt_smooth"]
df["apt_vs_sui_nvt"] = df["apt_nvt_smooth"] / df["sui_nvt_smooth"]

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df.index, df["apt_vs_sol_nvt"], label="APT / SOL NVT", color="#00529B", linewidth=2.5)
ax.plot(df.index, df["apt_vs_sui_nvt"], label="APT / SUI NVT", color="#F6B26B", linewidth=2.0, alpha=0.9)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

ax.set_title(
    "Relative NVT Premium of Aptos\n"
    "(>1 means Aptos trades richer than peer on NVT basis)",
    fontsize=15,
    fontweight="bold"
)
ax.set_ylabel("Relative NVT")
ax.set_xlabel("Date")

# log not necessary here; keep linear for readability
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
ax.legend(frameon=False, loc="upper right")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "nvt_relative_premium.png"), dpi=220)
plt.close()
print("✅ Saved nvt_relative_premium.png")

# -----------------------------
# PLOT 3: Z-score version (optional but useful)
# -----------------------------
print("📈 Generating standardized NVT chart...")

fig, ax = plt.subplots(figsize=(14, 6))

for asset, meta in ASSETS.items():
    col = f"{asset}_nvt_smooth"
    vals = df[col]
    z = (vals - vals.mean()) / vals.std() if vals.std() != 0 else vals * np.nan
    ax.plot(df.index, z, label=f"{meta['label']} NVT z-score", color=meta["color"], linewidth=meta["lw"], alpha=meta["alpha"])

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title(
    "Standardized NVT Comparison (z-score)\n"
    "Useful for comparing relative richness across chains",
    fontsize=15,
    fontweight="bold"
)
ax.set_ylabel("Z-score")
ax.set_xlabel("Date")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
ax.legend(frameon=False, loc="upper right")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "nvt_zscore_comparison.png"), dpi=220)
plt.close()
print("✅ Saved nvt_zscore_comparison.png")