import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR  = "datas/"
WINDOW    = 90          # days to average for cross-sectional snapshot
PEERS     = ["Avalanche C-Chain", "Near", "Sei Network", "Sui"]
SUBJECT   = "Aptos"
ALL_CHAINS = PEERS + [SUBJECT]

DEFI_LLAMA_NAMES = {
    "Aptos":           "Aptos",
    "Avalanche C-Chain": "Avalanche",
    "Near":            "Near",
    "Sei Network":     "Sei",
    "Sui":             "Sui",
}

# ── FILE MAP ──────────────────────────────────────────────────────────────────
FILE_MAP = {
    chain: {
        "fdv":     f"{DATA_DIR}{chain} - Fully Diluted Market Cap.csv",
        "revenue": f"{DATA_DIR}{chain} - Revenue.csv",
        "dau":     f"{DATA_DIR}{chain} - Daily Active Users (Chain).csv"
                   if chain != "Aptos"
                   else f"{DATA_DIR}{chain} - Transacting Users.csv",
        "dev":     f"{DATA_DIR}{chain} - Weekly Core Active Developers.csv",
    }
    for chain in ALL_CHAINS
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_series(path: str) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["date", "value"]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].sort_index()

def fetch_tvl(llama_name: str) -> pd.Series:
    url = f"https://api.llama.fi/v2/historicalChainTvl/{llama_name}"
    r   = requests.get(url, timeout=15)
    r.raise_for_status()
    df  = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"], unit="s")
    return df.set_index("date")["tvl"].sort_index()

def recent_mean(series: pd.Series, window: int) -> float:
    return series.dropna().iloc[-window:].mean()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("📂 Loading Artemis data...")
records = {}
for chain in ALL_CHAINS:
    fm = FILE_MAP[chain]
    records[chain] = {
        "fdv":     load_series(fm["fdv"]),
        "revenue": load_series(fm["revenue"]),
        "dau":     load_series(fm["dau"]),
        "dev":     load_series(fm["dev"]),
    }

print("🌐 Fetching TVL from DeFiLlama API...")
for chain in ALL_CHAINS:
    records[chain]["tvl"] = fetch_tvl(DEFI_LLAMA_NAMES[chain])

# ── BUILD CROSS-SECTIONAL SNAPSHOT ───────────────────────────────────────────
print(f"\n📊 Computing {WINDOW}-day averages...")
rows = []
for chain in ALL_CHAINS:
    r = records[chain]
    rows.append({
        "chain":   chain,
        "fdv":     recent_mean(r["fdv"],     WINDOW),
        "revenue": recent_mean(r["revenue"], WINDOW),
        "dau":     recent_mean(r["dau"],     WINDOW),
        "dev":     recent_mean(r["dev"],     WINDOW),
        "tvl":     recent_mean(r["tvl"],     WINDOW),
    })

snap = pd.DataFrame(rows).set_index("chain")
print(snap.to_string())

# ── LOG TRANSFORM ─────────────────────────────────────────────────────────────
log_snap = np.log(snap.replace(0, np.nan))

# ── REGRESSION (fit on PEERS only) ───────────────────────────────────────────
print("\n🔬 Running Relative Value Regression (peers only)...")
# With only 4 peers, using all 4 features causes perfect overfitting (n = p).
# Use Ridge regression to regularize, and keep economically sensible features.
features = ["revenue", "dau", "tvl"]   # drop dev: least economic signal

peer_data    = log_snap.loc[PEERS].dropna()
X_peers      = peer_data[features]
y_peers      = peer_data["fdv"]

model = Ridge(alpha=1.0)
model.fit(X_peers, y_peers)
y_pred_peers = model.predict(X_peers)
r2 = r2_score(y_peers, y_pred_peers)

# ── PREDICT APTOS FAIR VALUE ──────────────────────────────────────────────────
aptos_log   = log_snap.loc[SUBJECT][features].values.reshape(1, -1)
fair_log_fdv = model.predict(aptos_log)[0]
fair_fdv     = np.exp(fair_log_fdv)
actual_fdv   = snap.loc[SUBJECT, "fdv"]
premium      = (actual_fdv / fair_fdv - 1) * 100

print(f"\n{'='*60}")
print(f"  Model R²          : {r2:.3f}")
print(f"  Aptos Actual FDV  : ${actual_fdv/1e9:.2f}B")
print(f"  Aptos Fair FDV    : ${fair_fdv/1e9:.2f}B")
print(f"  Overvaluation     : {premium:+.1f}%")
print(f"{'='*60}")

# ── COEFFICIENTS ──────────────────────────────────────────────────────────────
print("\n📐 Regression Coefficients (log-log elasticities):")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:10s}: {coef:+.3f}")

# ── PLOT 1: Actual vs Fair FDV (bar) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Aptos Relative Value Analysis", fontsize=14, fontweight="bold")

chains_plot = PEERS + [SUBJECT]
actual_fdvs = [snap.loc[c, "fdv"] / 1e9 for c in chains_plot]
fair_fdvs   = []
for chain in chains_plot:
    x_in = log_snap.loc[chain][features].values.reshape(1, -1)
    fair_fdvs.append(np.exp(model.predict(x_in)[0]) / 1e9)

x_pos  = np.arange(len(chains_plot))
width  = 0.35
colors = ["#2196F3"] * len(PEERS) + ["#F44336"]

ax1 = axes[0]
bars1 = ax1.bar(x_pos - width/2, actual_fdvs, width, label="Actual FDV",  color=colors, alpha=0.85)
bars2 = ax1.bar(x_pos + width/2, fair_fdvs,   width, label="Model Fair FDV", color="gray", alpha=0.6)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([c.replace(" C-Chain","").replace(" Network","") for c in chains_plot], rotation=15)
ax1.set_ylabel("FDV (USD Billions)")
ax1.set_title("Actual vs Model Fair FDV")
ax1.legend()
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.1fB'))

# highlight Aptos bar
bars1[-1].set_edgecolor("red")
bars1[-1].set_linewidth(2)

# ── PLOT 2: Premium/Discount ──────────────────────────────────────────────────
ax2 = axes[1]
premiums = [(snap.loc[c, "fdv"] / np.exp(model.predict(log_snap.loc[c][features].values.reshape(1,-1))[0]) - 1)*100
            for c in chains_plot]
bar_colors = ["#4CAF50" if p < 0 else "#F44336" for p in premiums]
ax2.bar([c.replace(" C-Chain","").replace(" Network","") for c in chains_plot],
        premiums, color=bar_colors, alpha=0.85)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_ylabel("Premium / Discount to Fair Value (%)")
ax2.set_title("Overvaluation vs Peers")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
for i, (chain, prem) in enumerate(zip(chains_plot, premiums)):
    ax2.text(i, prem + (2 if prem >= 0 else -4),
             f"{prem:+.0f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("relative_value.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved relative_value.png")

# ── SUMMARY TXT ───────────────────────────────────────────────────────────────
with open("relative_value_summary.txt", "w") as f:
    f.write("=== Relative Value Regression Summary ===\n\n")
    f.write(f"Window: last {WINDOW} days average\n")
    f.write(f"Peers:  {', '.join(PEERS)}\n\n")
    f.write(f"Model R²         : {r2:.3f}\n")
    f.write(f"Aptos Actual FDV : ${actual_fdv/1e9:.2f}B\n")
    f.write(f"Aptos Fair FDV   : ${fair_fdv/1e9:.2f}B\n")
    f.write(f"Overvaluation    : {premium:+.1f}%\n\n")
    f.write("Coefficients (log-log):\n")
    for feat, coef in zip(features, model.coef_):
        f.write(f"  {feat:10s}: {coef:+.3f}\n")
    f.write(f"\nIntercept: {model.intercept_:+.3f}\n")
print("✅ Saved relative_value_summary.txt")
