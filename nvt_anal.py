"""
=============================================================================
NVT Ratio Analysis for APT Short Thesis (Absolute & Relative)
=============================================================================
This script calculates and plots the NVT ratio for Aptos and its peers,
providing both an absolute and relative valuation perspective.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from functools import reduce

# --- CONFIGURATION ---
DATA_DIR = "datas"
OUTPUT_DIR = "."
COMPETITORS = ['Aptos', 'Sui', 'Solana']
DATA_FILES = {
    'Aptos': {
        'mc': 'Aptos - Market Cap.csv',
        'tx_vol': 'Aptos - Chain Transactions.csv'
    },
    'Sui': {
        'mc': 'Sui - Market Cap.csv',
        'tx_vol': 'Sui - Chain Transactions.csv'
    },
    'Solana': {
        'mc': 'Solana - Market Cap.csv',
        'tx_vol': 'Solana - Chain Transactions.csv'
    }
}
ROLLING_WINDOW = 90 # Use 90-day rolling average for transaction volume to smooth it out

# --- DATA LOADING & NVT CALCULATION ---
all_nvt_dfs = []
print("📂 Loading data and calculating NVT ratios...")

for competitor, files in DATA_FILES.items():
    try:
        mc_path = os.path.join(DATA_DIR, files['mc'])
        tx_vol_path = os.path.join(DATA_DIR, files['tx_vol'])
        
        mc_df = pd.read_csv(mc_path, encoding='utf-8-sig', names=['date', 'market_cap'], header=0)
        tx_vol_df = pd.read_csv(tx_vol_path, encoding='utf-8-sig', names=['date', 'tx_volume'], header=0)
        
        df = pd.merge(mc_df, tx_vol_df, on='date', how='inner')
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate NVT Ratio = Market Cap / 90-day rolling average of Tx Volume
        df['tx_volume_ma'] = df['tx_volume'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        df[f'nvt_{competitor.lower()}'] = df['market_cap'] / df['tx_volume_ma']
        
        all_nvt_dfs.append(df[['date', f'nvt_{competitor.lower()}']])
        print(f"  ✅ Calculated NVT for {competitor}")

    except (FileNotFoundError, IndexError) as e:
        print(f"⚠️ Warning: Could not process data for {competitor}. Error: {e}")

# Merge all NVT data into a single dataframe
if not all_nvt_dfs:
    print("❌ No data to plot. Exiting.")
    exit()

df_final = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_nvt_dfs)
df_final = df_final.sort_values('date').set_index('date')
df_final = df_final.dropna(how='all')


# --- PLOTTING ---
print("📊 Generating NVT Ratio Comparison chart...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 7))

for col in df_final.columns:
    competitor_name = col.split('_')[1].capitalize()
    ax.plot(df_final.index, df_final[col], 
            label=f'{competitor_name} NVT Ratio', 
            linewidth=2 if competitor_name == 'Aptos' else 1.5,
            alpha=1 if competitor_name == 'Aptos' else 0.7)

ax.set_yscale('log') # Use log scale to compare assets with very different NVT levels
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_title('NVT Ratio Comparison (90-Day MA Transaction Volume)', fontsize=16, fontweight='bold')
ax.set_ylabel('NVT Ratio (Log Scale)')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'nvt_ratio_comparison.png'), dpi=200)
plt.close()
print("✅ Saved nvt_ratio_comparison.png")
