"""
=============================================================================
HMM Regime Analysis for APT Short Thesis (Market-Relative Version)
=============================================================================
Goal:
- Use HMM on broader crypto market (BTC+ETH+SOL)
- Collapse 3 regimes into 2 presentation regimes: Bullish vs Bearish
- Show ONE main chart:
    1) APT cumulative return with market regime shading
    2) Market vs APT average daily return by regime
- Export summary tables including regime-relative performance

Requirements:
    pip install hmmlearn matplotlib pandas numpy openpyxl
=============================================================================
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "datas"
OUTPUT_DIR = "."

ASSET_FILES = {
    'apt': "Aptos - Price (1).csv",
    'btc': "Bitcoin - Price (1).csv",
    'eth': "Ethereum - Price.csv",
    'sol': "Solana - Price.csv",
}

MARKET_WEIGHTS = {
    'btc': 0.77,
    'eth': 0.17,
    'sol': 0.06,
}

N_REGIMES = 3
N_TRIES = 100
ZOOM_START = "2025-01-01"

# Presentation colors
PLOT_COLORS = {
    'Bearish': '#F2B8B5',   # light red
    'Bullish': '#BFE3C0',   # light green
}
SERIES_COLORS = {
    'Market': '#4C78A8',
    'APT': '#111111'
}
LINE_COLOR = 'black'

# ============================================================
# HELPERS
# ============================================================
def load_artemis_csv(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna().sort_values('date').reset_index(drop=True)
    return df

def fit_hmm_3state(returns, n_regimes=N_REGIMES, n_tries=N_TRIES):
    """
    Fit 3-state Gaussian HMM.
    Internal classification:
      - highest vol regime => Tail/Extreme
      - remaining lower-return => Bearish
      - remaining higher-return => Bullish
    """
    X = returns.values.reshape(-1, 1)

    best_model = None
    best_score = -np.inf

    for seed in range(n_tries):
        try:
            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=seed,
                tol=0.01
            )
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except:
            continue

    if best_model is None:
        raise RuntimeError("HMM fitting failed for all seeds.")

    states = best_model.predict(X)

    regime_stats = []
    for i in range(n_regimes):
        mask = states == i
        rets = returns[mask]
        regime_stats.append({
            'regime_id': i,
            'ann_ret': rets.mean() * 365,
            'ann_vol': rets.std() * np.sqrt(365),
            'count': int(mask.sum())
        })

    # classify 3 states
    sorted_by_vol = sorted(regime_stats, key=lambda x: x['ann_vol'], reverse=True)
    tail_id = sorted_by_vol[0]['regime_id']
    remaining = [r for r in regime_stats if r['regime_id'] != tail_id]
    remaining.sort(key=lambda x: x['ann_ret'])
    bear_id = remaining[0]['regime_id']
    bull_id = remaining[1]['regime_id']

    regime_map_3 = {
        bear_id: 'Bearish',
        bull_id: 'Bullish',
        tail_id: 'Tail/Extreme'
    }

    mapped_states_3 = np.array([regime_map_3[s] for s in states])
    return best_model, mapped_states_3, regime_stats, regime_map_3

def collapse_regime(label):
    """Collapse Tail/Extreme into Bearish for presentation."""
    return 'Bearish' if label in ['Bearish', 'Tail/Extreme'] else 'Bullish'

def compute_tstat(series):
    series = series.dropna()
    n = len(series)
    if n < 2 or series.std(ddof=1) == 0:
        return np.nan
    return series.mean() / (series.std(ddof=1) / np.sqrt(n))

# ============================================================
# MAIN
# ============================================================
def main():
    print("📂 Loading data...")
    price_data = {}
    for asset, filename in ASSET_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        price_data[asset] = load_artemis_csv(filepath)
        df_asset = price_data[asset]
        print(f"  {asset.upper()}: {df_asset['date'].min().date()} ~ {df_asset['date'].max().date()} ({len(df_asset)} days)")

    # Merge
    df = price_data['apt'].rename(columns={'price': 'apt_price'})
    for asset, df_asset in price_data.items():
        if asset == 'apt':
            continue
        df = df.merge(df_asset.rename(columns={'price': f'{asset}_price'}), on='date', how='inner')

    df = df.sort_values('date').reset_index(drop=True)
    print(f"\n✅ Merged data: {df['date'].min().date()} ~ {df['date'].max().date()} ({len(df)} days)")

    # Returns
    for asset in ASSET_FILES.keys():
        df[f'{asset}_log_ret'] = np.log(df[f'{asset}_price'] / df[f'{asset}_price'].shift(1))

    df['mkt_log_ret'] = sum(MARKET_WEIGHTS[a] * df[f'{a}_log_ret'] for a in MARKET_WEIGHTS)
    df = df.dropna().reset_index(drop=True)

    # HMM on market only
    print("\n🔄 Fitting HMM on market index...")
    mkt_model, mkt_states_3, regime_stats, regime_map_3 = fit_hmm_3state(df['mkt_log_ret'])
    df['mkt_regime_3'] = mkt_states_3
    df['mkt_regime_2'] = df['mkt_regime_3'].map(collapse_regime)

    # Cumulative returns
    df['apt_cum'] = np.exp(df['apt_log_ret'].cumsum())
    df['mkt_cum'] = np.exp(df['mkt_log_ret'].cumsum())

    # Zoomed period
    df_plot = df[df['date'] >= ZOOM_START].copy().reset_index(drop=True)
    if df_plot.empty:
        raise ValueError(f"No data available after {ZOOM_START}")

    df_plot['apt_cum_zoom'] = df_plot['apt_cum'] / df_plot['apt_cum'].iloc[0]
    df_plot['mkt_cum_zoom'] = df_plot['mkt_cum'] / df_plot['mkt_cum'].iloc[0]
    df_plot['apt_excess_log_ret'] = df_plot['apt_log_ret'] - df_plot['mkt_log_ret']

    # ========================================================
    # REGIME SUMMARY TABLE
    # ========================================================
    summary_rows = []
    regime_order = ['Bullish', 'Bearish']

    for regime in regime_order:
        sub = df_plot[df_plot['mkt_regime_2'] == regime].copy()
        if sub.empty:
            continue

        apt_mean = sub['apt_log_ret'].mean() * 100
        mkt_mean = sub['mkt_log_ret'].mean() * 100
        excess_mean = sub['apt_excess_log_ret'].mean() * 100

        apt_ann = sub['apt_log_ret'].mean() * 365 * 100
        mkt_ann = sub['mkt_log_ret'].mean() * 365 * 100
        excess_ann = sub['apt_excess_log_ret'].mean() * 365 * 100

        apt_t = compute_tstat(sub['apt_log_ret'])
        excess_t = compute_tstat(sub['apt_excess_log_ret'])

        # hit ratios
        apt_up_ratio = (sub['apt_log_ret'] > 0).mean() * 100
        apt_underperform_ratio = (sub['apt_excess_log_ret'] < 0).mean() * 100

        summary_rows.append({
            'Market Regime': regime,
            'Count (days)': len(sub),
            'Market Avg Daily Return (%)': mkt_mean,
            'APT Avg Daily Return (%)': apt_mean,
            'APT Excess Daily Return (%)': excess_mean,
            'Market Annualized Return (%)': mkt_ann,
            'APT Annualized Return (%)': apt_ann,
            'APT Excess Annualized Return (%)': excess_ann,
            'APT Return t-stat': apt_t,
            'APT Excess Return t-stat': excess_t,
            'APT Positive Day Ratio (%)': apt_up_ratio,
            'APT Underperform vs Market Ratio (%)': apt_underperform_ratio
        })

    regime_summary = pd.DataFrame(summary_rows)
    regime_summary['Market Regime'] = pd.Categorical(
        regime_summary['Market Regime'],
        categories=regime_order,
        ordered=True
    )
    regime_summary = regime_summary.sort_values('Market Regime').reset_index(drop=True)

    print("\n📊 Regime-relative summary:")
    print(regime_summary.round(3))

    # ========================================================
    # MAIN FIGURE
    # ========================================================
    print("\n🎨 Generating presentation chart...")
    fig, axes = plt.subplots(
        2, 1, figsize=(15, 9),
        gridspec_kw={'height_ratios': [3.1, 1.8], 'hspace': 0.32}
    )
    ax1, ax2 = axes

    # --- Top chart: APT with market regime shading ---
    for i in range(len(df_plot) - 1):
        ax1.axvspan(
            df_plot.loc[i, 'date'],
            df_plot.loc[i + 1, 'date'],
            color=PLOT_COLORS[df_plot.loc[i, 'mkt_regime_2']],
            alpha=0.42,
            linewidth=0
        )

    ax1.plot(
        df_plot['date'],
        df_plot['apt_cum_zoom'],
        color=LINE_COLOR,
        linewidth=1.9,
        label='APT Cumulative Return'
    )

    ax1.set_title(
        'APT Performance Across Crypto Market Regimes (2025–2026)',
        fontsize=16,
        fontweight='bold',
        pad=12
    )
    ax1.set_ylabel('APT Cumulative Return\n(Indexed to 1.0)', fontsize=11)
    ax1.grid(alpha=0.18)

    legend_handles = [
        Patch(facecolor=PLOT_COLORS['Bullish'], alpha=0.42, label='Bullish Market Regime'),
        Patch(facecolor=PLOT_COLORS['Bearish'], alpha=0.42, label='Bearish Market Regime'),
        plt.Line2D([0], [0], color=LINE_COLOR, linewidth=1.8, label='APT Cumulative Return')
    ]
    ax1.legend(handles=legend_handles, loc='upper left', framealpha=0.95, fontsize=9)

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=35, ha='right')

    # helpful annotation
    latest_regime = df_plot['mkt_regime_2'].iloc[-1]
    ax1.text(
        0.99, 0.03,
        f"Latest Market Regime: {latest_regime}",
        transform=ax1.transAxes,
        ha='right', va='bottom',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.78, edgecolor='none')
    )

    # --- Bottom chart: market vs APT by regime ---
    x = np.arange(len(regime_summary))
    width = 0.34

    market_vals = regime_summary['Market Avg Daily Return (%)'].values
    apt_vals = regime_summary['APT Avg Daily Return (%)'].values

    bars1 = ax2.bar(
        x - width/2, market_vals, width,
        label='Market Avg Daily Return',
        color=SERIES_COLORS['Market'],
        alpha=0.85,
        edgecolor='black',
        linewidth=0.8
    )
    bars2 = ax2.bar(
        x + width/2, apt_vals, width,
        label='APT Avg Daily Return',
        color=SERIES_COLORS['APT'],
        alpha=0.9,
        edgecolor='black',
        linewidth=0.8
    )

    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regime_summary['Market Regime'])
    ax2.set_title('Market vs. APT Average Daily Return by Regime', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Avg Daily Return (%)', fontsize=11)
    ax2.grid(axis='y', alpha=0.2)
    ax2.legend(frameon=False, fontsize=9)

    # label bars
    def label_bars(ax, bars):
        for bar in bars:
            y = bar.get_height()
            label = f"{y:.2f}%"
            offset = 0.02 if y >= 0 else -0.02
            va = 'bottom' if y >= 0 else 'top'
            ax.text(
                bar.get_x() + bar.get_width()/2,
                y + offset,
                label,
                ha='center',
                va=va,
                fontsize=9,
                fontweight='bold'
            )

    label_bars(ax2, bars1)
    label_bars(ax2, bars2)

    # add excess-return annotation under each regime
    for i, row in regime_summary.iterrows():
        excess = row['APT Excess Daily Return (%)']
        ax2.text(
            x[i], ax2.get_ylim()[0] * 0.92,
            f"APT - MKT: {excess:.2f}%",
            ha='center',
            va='bottom',
            fontsize=9,
            color='darkred' if excess < 0 else 'darkgreen',
            fontweight='bold'
        )

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'hmm_apt_market_relative.png')
    plt.savefig(out_png, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved: {out_png}")

    # ========================================================
    # SAVE SUMMARY TABLES TO EXCEL
    # ========================================================
    print("\n📁 Saving summary workbook...")
    excel_path = os.path.join(OUTPUT_DIR, 'hmm_regime_summary.xlsx')

    stats_rows = []
    for state_id, label in regime_map_3.items():
        mask = df['mkt_regime_3'] == label
        rets = df.loc[mask, 'mkt_log_ret']
        stats_rows.append({
            'Original 3-State Regime': label,
            'Collapsed Presentation Regime': collapse_regime(label),
            'Annualized Return (%)': rets.mean() * 365 * 100,
            'Annualized Volatility (%)': rets.std() * np.sqrt(365) * 100,
            'Count (days)': int(mask.sum())
        })
    stats_df = pd.DataFrame(stats_rows)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        stats_df.to_excel(writer, sheet_name='Market HMM 3-State', index=False)
        regime_summary.to_excel(writer, sheet_name='APT vs Market by Regime', index=False)

        df_export = df_plot[[
            'date', 'apt_price', 'apt_log_ret', 'mkt_log_ret',
            'apt_excess_log_ret', 'mkt_regime_3', 'mkt_regime_2',
            'apt_cum_zoom', 'mkt_cum_zoom'
        ]].copy()
        df_export.to_excel(writer, sheet_name='Zoomed Daily Data', index=False)

    print(f"  ✅ Saved: {excel_path}")

    print("\n" + "="*70)
    print("DONE")
    print(f"Chart: {out_png}")
    print(f"Excel: {excel_path}")
    print("="*70)

if __name__ == "__main__":
    main()