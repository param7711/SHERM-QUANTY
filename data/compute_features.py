"""
Step 3 — Feature matrix computation.
Computes 22 features (20 price + 2 carry) per pair per date and saves
to data/processed/{pair}_features.parquet.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import FX_PAIRS
from feature_definitions import (
    FEATURE_NAMES, carry_differential, carry_direction,
)


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

def compute_pair_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Input: OHLC dataframe for one forex pair.
    Output: dataframe with all 22 features as columns.
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

    # 2 carry features — constant per pair given a policy-rate snapshot
    features['carry_differential'] = carry_differential(pair)
    features['carry_direction']    = carry_direction(pair)

    features['pair']  = pair
    features['close'] = close

    return features.dropna(subset=['ret_1d', 'z_21d', 'rsi_2'])


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def compute_all(raw_dir: str = 'data/raw',
                out_dir: str = 'data/processed') -> list:
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for pair in FX_PAIRS:
        raw_path = os.path.join(raw_dir, f'{pair}_daily.parquet')
        if not os.path.exists(raw_path):
            continue
        df = pd.read_parquet(raw_path)
        df.index = pd.to_datetime(df.index)
        feats = compute_pair_features(df, pair)
        out_path = os.path.join(out_dir, f'{pair}_features.parquet')
        feats.to_parquet(out_path)
        summary.append({'pair': pair, 'rows': len(feats), 'path': out_path})
    return summary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== Step 3 — Feature Matrix verification (forex) ===\n")

    if not os.path.exists('data/raw/EURUSD_daily.parquet'):
        print("  [skip] data/raw/EURUSD_daily.parquet not found. Run Step 2 first.")
        return

    summary = compute_all()
    print(f"  Computed features for {len(summary)} pairs.")

    df = pd.read_parquet('data/processed/EURUSD_features.parquet')

    z_mean = df['z_21d'].mean()
    z_std  = df['z_21d'].std()
    print(f"\n  EURUSD z_21d: mean={z_mean:.3f} (expect ~0), std={z_std:.3f} (expect ~1)")

    rsi_lo = (df['rsi_2'] < 10).sum()
    rsi_hi = (df['rsi_2'] > 90).sum()
    years  = len(df) / 252
    print(f"  EURUSD rsi_2: <10 hits={rsi_lo} ({rsi_lo/years:.1f}/yr), >90 hits={rsi_hi} ({rsi_hi/years:.1f}/yr)")

    print(f"\n  Carry: differential={df['carry_differential'].iloc[-1]:+.4f}, "
          f"direction={int(df['carry_direction'].iloc[-1]):+d}")

    expected_cols = FEATURE_NAMES + ['pair', 'close']
    missing = [c for c in expected_cols if c not in df.columns]
    print(f"\n  Column check: {len(df.columns)} cols, missing={missing}")
    print(f"  Rows after dropna: {len(df)}")

    print("\n=== Step 3 complete ===")


if __name__ == '__main__':
    _verification_check()
