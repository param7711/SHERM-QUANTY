"""
Feature definitions — single source of truth.
Any change to a feature formula here increments FEATURE_SET_VERSION.
"""

import pandas as pd

FEATURE_SET_VERSION = 'v1'

FEATURE_NAMES = [
    'ret_1d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d',
    'vol_5d', 'vol_20d', 'vol_ratio_vols', 'vol_expanding',
    'z_21d', 'rsi_2', 'rsi_14', 'pct_from_hi', 'pct_from_lo',
    'ret_zscore', 'ret_skew_20', 'up_streak', 'down_streak',
    'gap_pct', 'gap_unfilled_eod',
    # Options-specific (computed separately when needed)
    'iv_current', 'iv_rank_252', 'iv_change_5d', 'theta_per_day', 'dte',
    # Futures-specific
    'basis', 'basis_zscore', 'roll_yield',
]


def compute_features_for_instrument(ohlcv_df: pd.DataFrame, instrument_name: str) -> pd.DataFrame:
    """Master feature computation. Import this everywhere."""
    from data.compute_features import compute_underlying_features
    return compute_underlying_features(ohlcv_df, instrument_name)
