"""
=============================================================================
Granger Causality & VAR Analysis for APT Short Thesis
=============================================================================
This script tests the hypothesis that Aptos token unlocks (supply shocks)
Granger-cause a negative price reaction.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_DIR = "datas"
OUTPUT_DIR = "."
PRICE_FILE = "Aptos - Price (1).csv"
SUPPLY_FILE = "Aptos - Circulating Supply.csv"
MAX_LAGS = 14  # Test up to 14 days of lag

# --- DATA LOADING & PREPARATION ---
def load_artemis_csv(filepath, metric_name):
    """Loads and renames a single metric CSV from Artemis."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df.columns = ['date', metric_name]
        df['date'] = pd.to_datetime(df['date'])
        df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')
        return df.dropna().sort_values('date')
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {filepath}. Please download it from Artemis.")
        return None

print("📂 Loading Price and Circulating Supply data...")
price_df = load_artemis_csv(os.path.join(DATA_DIR, PRICE_FILE), 'price')
supply_df = load_artemis_csv(os.path.join(DATA_DIR, SUPPLY_FILE), 'supply')

if price_df is None or supply_df is None:
    exit()

df = pd.merge(price_df, supply_df, on='date', how='inner')

print("🛠️  Preparing time series data...")
# 1. Calculate Price Log Returns
df['price_ret'] = np.log(df['price'] / df['price'].shift(1))

# 2. Calculate Supply Shock (Percentage Change in Circulating Supply)
df['supply_shock'] = df['supply'].pct_change()

# Create the final dataframe for the model
df_model = df[['price_ret', 'supply_shock']].dropna()

# Check for stationarity (ADF test - a common requirement for VAR)
from statsmodels.tsa.stattools import adfuller
print("\n🔍 Checking for stationarity (ADF Test)...")
adf_price = adfuller(df_model['price_ret'])
adf_supply = adfuller(df_model['supply_shock'])
print(f"  Price Return p-value: {adf_price[1]:.4f} -> {'Stationary' if adf_price[1] < 0.05 else 'Non-Stationary'}")
print(f"  Supply Shock p-value: {adf_supply[1]:.4f} -> {'Stationary' if adf_supply[1] < 0.05 else 'Non-Stationary'}")

# --- VAR MODEL & GRANGER CAUSALITY ---
print("\n🤖 Building VAR model to find optimal lag...")
model = VAR(df_model)

# Select optimal lag order based on AIC
lag_order = model.select_order(maxlags=MAX_LAGS)
optimal_lag = max(1, lag_order.aic)
print(f"  Optimal lag selected by AIC: {optimal_lag}")

print("\n🔬 Running Granger Causality Test...")
gc_results = grangercausalitytests(df_model[['price_ret', 'supply_shock']], [optimal_lag], verbose=False)

# Extract and print the key result
p_value = gc_results[optimal_lag][0]['ssr_ftest'][1]
f_statistic = gc_results[optimal_lag][0]['ssr_ftest'][0]

print("\n" + "="*80)
print("Hypothesis: Does Supply Shock Granger-Cause Price Return?")
print(f"  F-Statistic: {f_statistic:.4f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  ✅ Result: Significant (p < 0.05). We REJECT the null hypothesis.")
    print("     This suggests that supply shocks are a statistically useful predictor of price returns.")
else:
    print("  ❌ Result: Not Significant (p >= 0.05). We FAIL to reject the null hypothesis.")
print("="*80)

# --- IMPULSE RESPONSE FUNCTION (IRF) ---
print("\n📈 Generating Impulse Response Function (IRF) plot...")
var_model = model.fit(optimal_lag)
irf = var_model.irf(periods=10) # Response over 10 days

fig = irf.plot(orth=False)
fig.suptitle('Impulse Response: Shock in Supply -> Response in Price', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'granger_irf.png'), dpi=200)
plt.close()
print("✅ Saved granger_irf.png")

with open(os.path.join(OUTPUT_DIR, 'granger_summary.txt'), 'w') as f:
    f.write(f"Granger Causality Test: Supply Shock -> Price Return\n")
    f.write(f"Optimal Lag: {optimal_lag}\n")
    f.write(f"F-Statistic: {f_statistic:.4f}\n")
    f.write(f"p-value: {p_value:.4f}\n")
print("✅ Saved granger_summary.txt")