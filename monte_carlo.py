import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
PRICE_FILE   = "datas/Aptos - Price (1).csv"
N_SIMULATIONS = 10_000
HORIZON_DAYS  = 90       # 3-month price target window
SEED          = 42

# ── LOAD PRICE DATA ───────────────────────────────────────────────────────────
print("📂 Loading APT price data...")
df = pd.read_csv(PRICE_FILE, encoding="utf-8-sig")
df.columns = ["date", "price"]
df["date"]  = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()
df["log_ret"] = np.log(df["price"] / df["price"].shift(1))
df = df.dropna()

current_price = df["price"].iloc[-1]
print(f"  Current APT Price : ${current_price:.4f}")
print(f"  Data range        : {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Total days        : {len(df)}")

# ── COMPUTE GBM PARAMETERS ────────────────────────────────────────────────────
# Use recent 180 days for mu/sigma (more relevant than full history)
recent = df["log_ret"].iloc[-180:]
mu_daily    = recent.mean()       # drift
sigma_daily = recent.std()        # volatility

print(f"\n📐 GBM Parameters (last 180 days):")
print(f"  Daily drift (μ)  : {mu_daily:.6f}  ({mu_daily*365*100:.1f}% annualized)")
print(f"  Daily vol   (σ)  : {sigma_daily:.6f}  ({sigma_daily*np.sqrt(365)*100:.1f}% annualized)")

# ── MONTE CARLO SIMULATION ────────────────────────────────────────────────────
print(f"\n🎲 Running {N_SIMULATIONS:,} simulations over {HORIZON_DAYS} days...")
np.random.seed(SEED)

# Shape: (N_SIMULATIONS, HORIZON_DAYS)
shocks = np.random.normal(
    loc   = mu_daily - 0.5 * sigma_daily**2,   # Ito correction
    scale = sigma_daily,
    size  = (N_SIMULATIONS, HORIZON_DAYS)
)
log_paths     = np.cumsum(shocks, axis=1)
price_paths   = current_price * np.exp(log_paths)    # (N_SIMS, HORIZON)
final_prices  = price_paths[:, -1]

# ── STATISTICS ────────────────────────────────────────────────────────────────
p5   = np.percentile(final_prices, 5)
p25  = np.percentile(final_prices, 25)
p50  = np.percentile(final_prices, 50)
p75  = np.percentile(final_prices, 75)
p95  = np.percentile(final_prices, 95)
prob_down = np.mean(final_prices < current_price) * 100

print(f"\n{'='*55}")
print(f"  Current Price          : ${current_price:.4f}")
print(f"  ── {HORIZON_DAYS}-Day Price Target Range ──")
print(f"  Bear  (5th pct)        : ${p5:.4f}   ({(p5/current_price-1)*100:+.1f}%)")
print(f"  Low   (25th pct)       : ${p25:.4f}   ({(p25/current_price-1)*100:+.1f}%)")
print(f"  Base  (50th pct)       : ${p50:.4f}   ({(p50/current_price-1)*100:+.1f}%)")
print(f"  High  (75th pct)       : ${p75:.4f}   ({(p75/current_price-1)*100:+.1f}%)")
print(f"  Bull  (95th pct)       : ${p95:.4f}   ({(p95/current_price-1)*100:+.1f}%)")
print(f"  P(price < current)     : {prob_down:.1f}%")
print(f"{'='*55}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"Aptos (APT) Monte Carlo Simulation — {HORIZON_DAYS}-Day Horizon",
             fontsize=13, fontweight="bold")

# — Plot 1: Price paths —
ax1 = axes[0]
sample_idx = np.random.choice(N_SIMULATIONS, 300, replace=False)
for i in sample_idx:
    ax1.plot(range(HORIZON_DAYS), price_paths[i], alpha=0.04, color="#2196F3", linewidth=0.5)

# Percentile bands
ax1.plot(range(HORIZON_DAYS), np.percentile(price_paths, 5,  axis=0), color="red",    linewidth=1.5, linestyle="--", label="5th / 95th pct")
ax1.plot(range(HORIZON_DAYS), np.percentile(price_paths, 95, axis=0), color="red",    linewidth=1.5, linestyle="--")
ax1.plot(range(HORIZON_DAYS), np.percentile(price_paths, 50, axis=0), color="orange", linewidth=2,   label="Median")
ax1.axhline(current_price, color="white", linewidth=1, linestyle=":", label=f"Current ${current_price:.2f}")

ax1.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#444")
ax1.set_xlabel("Days", color="white")
ax1.set_ylabel("APT Price (USD)", color="white")
ax1.set_title("Simulated Price Paths", color="white")
ax1.legend(fontsize=8, facecolor="#333", labelcolor="white")
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.2f'))

# — Plot 2: Final price distribution —
ax2 = axes[1]
ax2.set_facecolor("#1a1a2e")
n, bins, patches = ax2.hist(final_prices, bins=100, color="#2196F3", alpha=0.7, edgecolor="none")

# Color below current price red
for patch, left in zip(patches, bins[:-1]):
    if left < current_price:
        patch.set_facecolor("#F44336")

ax2.axvline(current_price, color="white",  linewidth=1.5, linestyle=":",  label=f"Current ${current_price:.2f}")
ax2.axvline(p5,            color="red",    linewidth=1.5, linestyle="--", label=f"5th pct  ${p5:.2f}")
ax2.axvline(p50,           color="orange", linewidth=1.5, linestyle="--", label=f"Median   ${p50:.2f}")
ax2.axvline(p95,           color="green",  linewidth=1.5, linestyle="--", label=f"95th pct ${p95:.2f}")

ax2.tick_params(colors="white")
ax2.spines[:].set_color("#444")
ax2.set_xlabel("Final Price (USD)", color="white")
ax2.set_ylabel("Frequency", color="white")
ax2.set_title(f"Distribution of {HORIZON_DAYS}-Day Final Prices\n"
              f"P(below current) = {prob_down:.1f}%  |  Red = downside",
              color="white")
ax2.legend(fontsize=8, facecolor="#333", labelcolor="white")
ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('$%.2f'))

plt.tight_layout()
plt.savefig("monte_carlo.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print("\n✅ Saved monte_carlo.png")

# ── SUMMARY TXT ───────────────────────────────────────────────────────────────
with open("monte_carlo_summary.txt", "w") as f:
    f.write(f"=== APT Monte Carlo Simulation ({HORIZON_DAYS}-Day) ===\n\n")
    f.write(f"Simulations   : {N_SIMULATIONS:,}\n")
    f.write(f"Current Price : ${current_price:.4f}\n")
    f.write(f"Daily μ       : {mu_daily:.6f}\n")
    f.write(f"Daily σ       : {sigma_daily:.6f}\n\n")
    f.write(f"Price Targets:\n")
    f.write(f"  Bear  (5th) : ${p5:.4f}  ({(p5/current_price-1)*100:+.1f}%)\n")
    f.write(f"  Low  (25th) : ${p25:.4f}  ({(p25/current_price-1)*100:+.1f}%)\n")
    f.write(f"  Base (50th) : ${p50:.4f}  ({(p50/current_price-1)*100:+.1f}%)\n")
    f.write(f"  High (75th) : ${p75:.4f}  ({(p75/current_price-1)*100:+.1f}%)\n")
    f.write(f"  Bull (95th) : ${p95:.4f}  ({(p95/current_price-1)*100:+.1f}%)\n\n")
    f.write(f"P(price < current) : {prob_down:.1f}%\n")
print("✅ Saved monte_carlo_summary.txt")
