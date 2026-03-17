"""
=============================================================================
Valuation & Comparables Analysis for APT Short Thesis
=============================================================================
This script creates a bar chart comparing Aptos's valuation multiples
(Price-to-Sales and Price-to-Fees) against its key competitors.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DATA_DIR = "datas"
OUTPUT_DIR = "."
COMPETITORS = ['Aptos', 'Sui', 'Solana']
METRICS = {
    'MC / Revenue (Annualized)': { # P/S Ratio
        'Aptos': 'Aptos - MC  Revenue Annualized.csv',
        'Sui': 'Sui - MC  Revenue Annualized.csv',
        'Solana': 'Solana - MC  Revenue Annualized.csv',
    },
    'MC / Fees (Annualized)': { # P/F Ratio
        'Aptos': 'Aptos - MC  Fees Annualized.csv',
        'Sui': 'Sui - MC  Fees Annualized.csv',
        'Solana': 'Solana - MC  Fees Annualized.csv',
    }
}

# --- DATA LOADING & ANALYSIS ---
latest_multiples = {}
for metric, files in METRICS.items():
    latest_multiples[metric] = {}
    for competitor, filename in files.items():
        try:
            filepath = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            df.columns = ['date', 'value']
            # Get the most recent non-zero value
            latest_value = df[df['value'] > 0].sort_values('date', ascending=False).iloc[0]['value']
            latest_multiples[metric][competitor] = latest_value
        except (FileNotFoundError, IndexError):
            print(f"⚠️ Warning: Could not load or find data for {competitor} - {metric}")
            latest_multiples[metric][competitor] = 0

# --- PLOTTING ---
print("📊 Generating Valuation Comparison chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Aptos: Relative Overvaluation vs. Peers (Economic Multiples)', fontsize=16, fontweight='bold')

for i, (metric, values) in enumerate(latest_multiples.items()):
    ax = axes[i]
    # Filter out competitors with 0 value if any
    valid_competitors = {c: v for c, v in values.items() if v > 0}
    if not valid_competitors:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax.set_title(f'Latest {metric}', fontsize=12)
        continue

    competitors_sorted = sorted(valid_competitors.keys(), key=lambda x: valid_competitors[x], reverse=True)
    values_sorted = [valid_competitors[c] for c in competitors_sorted]
    
    colors = ['#D9534F' if c == 'Aptos' else 'grey' for c in competitors_sorted]
    bars = ax.bar(competitors_sorted, values_sorted, color=colors)
    ax.set_title(f'Latest {metric}', fontsize=12)
    ax.set_ylabel('Multiple (x)')
    ax.bar_label(bars, fmt='%.1fx')
    ax.set_yscale('log') # Use log scale for better visualization of large differences
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'valuation_comparison.png'), dpi=200)
plt.close()
print("✅ Saved valuation_comparison.png")

# Print table for thesis
print("\n📋 Valuation Table for Thesis:")
df_table = pd.DataFrame(latest_multiples)
print(df_table.to_markdown())