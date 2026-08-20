"""
Feature definitions — single source of truth.
Any change to a feature formula here increments FEATURE_SET_VERSION.
"""

import pandas as pd

from config import PAIR_LEGS, POLICY_RATES

FEATURE_SET_VERSION = 'v2-forex'

PRICE_FEATURES = [
    'ret_1d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_60d',
    'vol_5d', 'vol_20d', 'vol_ratio_vols', 'vol_expanding',
    'z_21d', 'rsi_2', 'rsi_14', 'pct_from_hi', 'pct_from_lo',
    'ret_zscore', 'ret_skew_20', 'up_streak', 'down_streak',
    'gap_pct', 'gap_unfilled_eod',
]

CARRY_FEATURES = ['carry_differential', 'carry_direction']

FEATURE_NAMES = PRICE_FEATURES + CARRY_FEATURES


def carry_differential(pair: str, policy_rates: dict = None) -> float:
    """Base-currency policy rate minus quote-currency policy rate."""
    rates = policy_rates or POLICY_RATES
    base, quote = PAIR_LEGS[pair]
    return rates[base] - rates[quote]


def carry_direction(pair: str, policy_rates: dict = None) -> int:
    """Sign of the carry differential: +1 long-base carries positively."""
    diff = carry_differential(pair, policy_rates)
    return (diff > 0) - (diff < 0)


def compute_features_for_pair(ohlcv_df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Master feature computation. Import this everywhere."""
    from data.compute_features import compute_pair_features
    return compute_pair_features(ohlcv_df, pair)
