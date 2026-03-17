# Aptos (APT) Short Thesis — Quantitative Analysis

Short investment thesis on Aptos (APT), a Layer-1 blockchain with FDV > $500M.
Analysis covers on-chain fundamentals, statistical return attribution, regime detection, and valuation.

---

## Repository Structure

```
.
├── quant_models/      # Statistical and ML models (OLS, HMM, Granger)
├── fundamentals/      # On-chain fundamentals (NVT, TVL, Developers, Velocity)
└── valuation/         # Valuation analysis (Relative Value, Monte Carlo, Price Target)
```

---

## quant_models/

### OLS Return Attribution (`ols.py`)
Regresses APT daily log-returns on BTC and ETH factors using OLS.
Estimates alpha (idiosyncratic drift), market beta, and R-squared.

Key results:
- Alpha: -0.0189/day (-690% annualized)
- Alpha t-stat: -3.158 (statistically significant underperformance)
- R-squared: 0.612

Outputs: `ols_return_attribution.png`, `ols_actual_vs_predicted.png`, `ols_rolling_beta.png`, `ols_summary.txt`

---

### HMM Regime Analysis (`hhm.py`)
Fits a 3-state Gaussian Hidden Markov Model on a weighted crypto market index (BTC 77%, ETH 17%, SOL 6%).
Collapses to 2 presentation regimes (Bullish / Bearish) and measures APT performance in each.

Key results:
- Bullish regime APT avg daily return: -0.516%
- Bearish regime APT avg daily return: -0.510%
- APT underperforms in both regimes, confirming structural weakness

Outputs: `hmm_apt_market_relative.png`, `hmm_regime_summary.xlsx`

---

### Granger Causality (`granger.py`)
Tests whether APT circulating supply growth Granger-causes APT price returns.

Key results:
- p-value: 0.31 (not significant)
- Supply shock does not predict price in a statistically meaningful way at the tested lag

Outputs: `granger_irf.png`, `granger_summary.txt`

---

## fundamentals/

### Fundamental Dashboard (`fundamentals.py`)
Five-panel dashboard visualizing APT on-chain deterioration:
1. Transactions — smoothed 30D MA on log scale, winsorized
2. TVL — capital exit with peak-to-latest drawdown annotation
3. Core Active Developers — weekly bars with 8W MA, peak decline annotation
4. Valuation Disconnect — Price vs Fees both indexed to 100 at start
5. Supply Pressure — circulating supply growth from sample start

Outputs: `fundamental_dashboard_improved.png`, `fundamental_summary.txt`

---

### NVT Ratio (`nvt_anal.py`)
Computes Network Value to Transactions ratio (Market Cap / 30D MA daily tx count) for APT, SOL, and SUI.
Normalizes with z-score to compare across chains. A high NVT relative to peers implies the market values each unit of network activity more expensively for APT.

Outputs: `nvt_comparison_smoothed.png`, `nvt_zscore_comparison.png`, `nvt_relative_premium.png`

---

### Token Velocity (`velocity.py`)
Measures token velocity (on-chain transfer volume / market cap) as a proxy for real economic usage.
Lower velocity relative to peers suggests APT tokens are held speculatively rather than used productively.

Outputs: `token_velocity_comparison.png`

---

## valuation/

### Relative Value Regression (`relative_value.py`)
Cross-sectional Ridge regression across L1 peers (Avalanche, Near, Sei, Sui):

```
log(FDV) ~ log(Revenue) + log(DAU) + log(TVL)
```

Key results:
- R-squared: 0.988
- Actual FDV: $2.90B
- Model Fair FDV: $2.05B
- Implied overvaluation: +41.1%

Outputs: `relative_value.png`, `relative_value_summary.txt`

---

### Implied Revenue Analysis (`implied_revenue.py`)
Bar chart showing how much annualized revenue APT would need to generate to justify its current market cap at peer MC/Revenue multiples (Sui, Solana).

Outputs: `implied_revenue.png`

---

### Monte Carlo Simulation (`monte_carlo.py`)
GBM-based 10,000-path simulation over a 90-day horizon calibrated to APT historical returns.

Key results:
- mu: -0.88%/day, sigma: 5.31%/day
- Base case (50th percentile): $0.37 (-60.4% from current)
- Bull case (95th percentile): $0.87 (-8.1%)
- P(price declines from current): 96.6%

Outputs: `monte_carlo.png`, `monte_carlo_summary.txt`

---

### Price Target Scenario Table (`price_target_table.py`)
Styled investment memo table with Bull / Base / Bear scenarios based on Sui MC/Revenue multiples.
Probability-weighted expected price: ~$0.63.

Outputs: `price_target_table.png`

---

## Data

Raw data sourced from Artemis Terminal (CSV exports). Not included in this repository.
Place all CSVs in a `datas/` directory at the project root before running any scripts.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib scipy statsmodels hmmlearn scikit-learn openpyxl requests
```
