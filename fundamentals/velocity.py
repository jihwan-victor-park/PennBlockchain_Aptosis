"""
=============================================================================
Token Velocity Analysis for APT Short Thesis
=============================================================================
This script calculates and plots the Token Velocity for Aptos and its peers
to measure the decoupling of valuation from on-chain economic activity.

Velocity = 90-Day Avg Transaction Volume / Market Cap
A declining velocity suggests valuation is not supported by utility.
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

# --- DATA LOADING & VELOCITY CALCULATION ---
all_velocity_dfs = []
print("📂 Loading data and calculating Token Velocity...")

for competitor, files in DATA_FILES.items():
    try:
        mc_path = os.path.join(DATA_DIR, files['mc'])
        tx_vol_path = os.path.join(DATA_DIR, files['tx_vol'])
        
        mc_df = pd.read_csv(mc_path, encoding='utf-8-sig', names=['date', 'market_cap'], header=0)
        tx_vol_df = pd.read_csv(tx_vol_path, encoding='utf-8-sig', names=['date', 'tx_volume'], header=0)
        
        df = pd.merge(mc_df, tx_vol_df, on='date', how='inner')
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate Velocity = 90d MA of Tx Volume / Market Cap
        df['tx_volume_ma'] = df['tx_volume'].rolling(window=ROLLING_WINDOW, min_periods=30).mean()
        # Avoid division by zero
        df = df[df['market_cap'] > 0]
        df[f'velocity_{competitor.lower()}'] = df['tx_volume_ma'] / df['market_cap']
        
        all_velocity_dfs.append(df[['date', f'velocity_{competitor.lower()}']])
        print(f"  ✅ Calculated Velocity for {competitor}")

    except (FileNotFoundError, IndexError) as e:
        print(f"⚠️ Warning: Could not process data for {competitor}. Error: {e}")

# Merge all Velocity data into a single dataframe
if not all_velocity_dfs:
    print("❌ No data to plot. Exiting.")
    exit()

df_final = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_velocity_dfs)
df_final = df_final.sort_values('date').set_index('date')
df_final = df_final.dropna(how='all')


# --- PLOTTING ---
print("📊 Generating Token Velocity Comparison chart...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 7))

for col in df_final.columns:
    competitor_name = col.split('_')[1].capitalize()
    ax.plot(df_final.index, df_final[col], 
            label=f'{competitor_name} Velocity', 
            linewidth=2.5 if competitor_name == 'Aptos' else 1.5,
            alpha=1 if competitor_name == 'Aptos' else 0.7,
            color='red' if competitor_name == 'Aptos' else None)

ax.set_title('Token Velocity: Valuation vs. Economic Activity', fontsize=16, fontweight='bold')
ax.set_ylabel('Velocity (Tx Volume / Market Cap)')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.text(0.01, 0.01, 'Lower velocity implies higher speculative premium or lower utility', 
        transform=ax.transAxes, fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'token_velocity_comparison.png'), dpi=200)
plt.close()
print("✅ Saved token_velocity_comparison.png")
