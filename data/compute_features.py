"""
Step 4 — Feature matrix computation.
Computes 22 underlying-level features per instrument per date and saves
to data/processed/{instrument}_features.parquet.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RSI helper
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Streak helper (vectorized — equivalent to brief's loop spec)
# ---------------------------------------------------------------------------

def _compute_streak(series: pd.Series, direction: str) -> pd.Series:
    """Count consecutive up or down days."""
    condition = (series > series.shift(1)) if direction == 'up' else (series < series.shift(1))
    condition = condition.astype(int)
    groups = (condition == 0).cumsum()
    return condition.groupby(groups).cumsum()


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def compute_underlying_features(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Input: OHLCV dataframe for one instrument.
    Output: dataframe with all 28 features as columns.
    """
    close = df['close']
    open_ = df['open']
    high  = df['high']
    low   = df['low']

    features = pd.DataFrame(index=df.index)

    # 20 underlying-level features
    features['ret_1d']   = np.log(close / close.shift(1))
    features['mom_5d']   = close.pct_change(5)
    features['mom_10d']  = close.pct_change(10)
    features['mom_20d']  = close.pct_change(20)
    features['mom_60d']  = close.pct_change(60)
    features['vol_5d']   = features['ret_1d'].rolling(5).std()
    features['vol_20d']  = features['ret_1d'].rolling(20).std()
    features['vol_ratio_vols'] = features['vol_5d'] / features['vol_20d']
    features['vol_expanding']  = (features['vol_5d'] > features['vol_20d']).astype(int)
    features['z_21d'] = (close - close.rolling(21).mean()) / close.rolling(21).std()
    features['rsi_2']  = _compute_rsi(close, 2)
    features['rsi_14'] = _compute_rsi(close, 14)
    features['rolling_max_252'] = close.rolling(252).max()
    features['rolling_min_252'] = close.rolling(252).min()
    price_range = features['rolling_max_252'] - features['rolling_min_252']
    features['pct_from_hi'] = (close - features['rolling_max_252']) / price_range.replace(0, np.nan)
    features['pct_from_lo'] = (close - features['rolling_min_252']) / price_range.replace(0, np.nan)
    features['ret_zscore']   = (features['ret_1d'] - features['ret_1d'].rolling(20).mean()) / features['vol_20d']
    features['ret_skew_20']  = features['ret_1d'].rolling(20).skew()
    features['up_streak']    = _compute_streak(close, direction='up')
    features['down_streak']  = _compute_streak(close, direction='down')
    features['gap_pct']         = (open_ - close.shift(1)) / close.shift(1)
    features['gap_unfilled_eod'] = ((features['gap_pct'] < -0.01) & (close < close.shift(1))).astype(int)

    features['underlying'] = name
    features['close']      = close

    return features.dropna(subset=['ret_1d', 'z_21d', 'rsi_2'])


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def compute_all(raw_dir: str = 'data/raw',
                out_dir: str = 'data/processed') -> list:
    os.makedirs(out_dir, exist_ok=True)
    parquets = [f for f in os.listdir(raw_dir) if f.endswith('_daily.parquet')]
    summary = []
    for fname in sorted(parquets):
        name = fname.replace('_daily.parquet', '')
        df = pd.read_parquet(os.path.join(raw_dir, fname))
        df.index = pd.to_datetime(df.index)
        feats = compute_underlying_features(df, name)
        out_path = os.path.join(out_dir, f'{name}_features.parquet')
        feats.to_parquet(out_path)
        summary.append({'instrument': name, 'rows': len(feats), 'path': out_path})
    return summary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== Step 4 — Feature Matrix verification ===\n")

    nifty_raw_path = 'data/raw/NIFTY_daily.parquet'
    if not os.path.exists(nifty_raw_path):
        print("  [skip] data/raw/NIFTY_daily.parquet not found. Run Step 2 first.")
        return

    summary = compute_all()
    print(f"  Computed features for {len(summary)} instruments.")

    nifty_feat_path = 'data/processed/NIFTY_features.parquet'
    df = pd.read_parquet(nifty_feat_path)

    # z_21d oscillates around 0
    z_mean = df['z_21d'].mean()
    z_std  = df['z_21d'].std()
    print(f"\n  NIFTY z_21d: mean={z_mean:.3f} (expect ~0), std={z_std:.3f} (expect ~1)")

    # rsi_2 extreme hits
    rsi_lo = (df['rsi_2'] < 10).sum()
    rsi_hi = (df['rsi_2'] > 90).sum()
    years  = len(df) / 252
    print(f"  NIFTY rsi_2: <10 hits={rsi_lo} ({rsi_lo/years:.1f}/yr), >90 hits={rsi_hi} ({rsi_hi/years:.1f}/yr)")

    # 5 most extreme z_21d readings
    extreme = df['z_21d'].abs().nlargest(5)
    print(f"\n  5 most extreme z_21d readings:")
    for date, val in extreme.items():
        row = df.loc[date]
        print(f"    {str(date.date()):<12}  z_21d={row['z_21d']:+.3f}  close={row['close']:.1f}")

    # Column count check
    expected_cols = [
        'ret_1d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d',
        'vol_5d', 'vol_20d', 'vol_ratio_vols', 'vol_expanding',
        'z_21d', 'rsi_2', 'rsi_14', 'rolling_max_252', 'rolling_min_252',
        'pct_from_hi', 'pct_from_lo', 'ret_zscore', 'ret_skew_20',
        'up_streak', 'down_streak', 'gap_pct', 'gap_unfilled_eod',
        'underlying', 'close',
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    print(f"\n  Column check: {len(df.columns)} cols, missing={missing}")
    print(f"  Rows after dropna: {len(df)}")
    print(f"  Nulls in key cols: ret_1d={df['ret_1d'].isna().sum()}, z_21d={df['z_21d'].isna().sum()}, rsi_2={df['rsi_2'].isna().sum()}")

    print("\n=== Step 4 complete ===")


if __name__ == '__main__':
    _verification_check()
