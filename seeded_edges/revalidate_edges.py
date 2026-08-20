"""
Step 5.5 — Seeded edge revalidation.
Computes empirical priors on real historical forex data (2010-2021 train
window) and OOS check (2022-2023). Replaces the research-adapted priors
written by populate_edges.py and clears needs_revalidation.

Mandatory before paper trading. Several of these edges were adapted from
equity-index/options mechanics that were never actually tested on FX price
action, so this step is what catches an edge that does not survive contact
with the real universe rather than trading it live and finding out.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

from config import (
    DB_PATH, FX_PAIRS, EDGE_STOP_PCT, FX_STOP, SYMBOL_STOP_MULTIPLIER,
    CROSS_PAIRS, CROSS_BETA_WINDOW, CROSS_SPREAD_Z_ENTRY,
)

TRAIN_START = '2010-01-01'
TRAIN_END   = '2021-12-31'
OOS_START   = '2022-01-01'
OOS_END     = '2023-12-31'

# The 7 traded majors, excluding gold. Kept separate from FX_PAIRS because
# pooling XAUUSD's volatility regime into the same aggregate as the majors
# would misstate both — gold gets its own single-instrument revalidation.
ALL_MAJORS = [p for p in FX_PAIRS if p != 'XAUUSD']


def _stop_floor(edge_id: str, pair: str = None) -> float:
    """
    Per-edge stop, as a negative return floor. Replaces the old fixed
    STOP_LOSS_FLOOR = -0.35 (a 35% options-premium stop), which was
    roughly 20x looser than any FX_STOP band and would have floored
    almost nothing in a spot-return backtest.
    """
    pct = EDGE_STOP_PCT.get(edge_id, FX_STOP.get(5, 0.015))
    if pair:
        pct *= SYMBOL_STOP_MULTIPLIER.get(pair, 1.0)
    return -pct


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load(pair: str) -> pd.DataFrame:
    path = f'data/processed/{pair}_features.parquet'
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def _load_regime() -> pd.DataFrame:
    df = pd.read_parquet('data/processed/regime_history.parquet')
    df.index = pd.to_datetime(df.index)
    return df[['regime_state']]


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def _compute_forward_returns(close: pd.Series, holding_period: int) -> pd.Series:
    """Log return from close[t] to close[t+holding_period]."""
    return np.log(close.shift(-holding_period) / close)


def _run_backtest(trigger_mask: pd.Series, direction_series: pd.Series,
                  close: pd.Series, holding_period: int,
                  window_start: str, window_end: str, stop_floor: float) -> dict:
    """
    trigger_mask: boolean Series — True on signal dates
    direction_series: +1 for long, -1 for short (same index)
    """
    fwd = _compute_forward_returns(close, holding_period)
    mask = trigger_mask & trigger_mask.index.to_series().between(window_start, window_end)
    idx  = mask[mask].index
    if len(idx) == 0:
        return {'n': 0, 'win_rate': np.nan, 'avg_return': np.nan,
                'avg_return_net': np.nan, 'p_value': np.nan}

    raw_returns   = fwd.loc[idx]
    directions    = direction_series.loc[idx]
    trade_returns = raw_returns * directions
    floored       = trade_returns.clip(lower=stop_floor)

    n              = len(trade_returns)
    win_rate       = (trade_returns > 0).mean()
    avg_return     = trade_returns.mean()
    avg_return_net = floored.mean()

    if n > 1:
        _, p_value = stats.ttest_1samp(floored.dropna(), 0)
    else:
        p_value = np.nan

    return {'n': n, 'win_rate': win_rate, 'avg_return': avg_return,
            'avg_return_net': avg_return_net, 'p_value': p_value}


# ---------------------------------------------------------------------------
# Per-edge trigger definitions
# ---------------------------------------------------------------------------

def _triggers_seed001(feat_df: pd.DataFrame):
    """z_21d < -2.0 (long) OR z_21d > 2.0 (short)."""
    long_mask  = feat_df['z_21d'] < -2.0
    short_mask = feat_df['z_21d'] > 2.0
    mask       = long_mask | short_mask
    direction  = pd.Series(np.where(long_mask, 1, -1), index=feat_df.index)
    return mask, direction


def _triggers_seed002(feat_df: pd.DataFrame):
    """gap_unfilled_eod == 1 — all long."""
    mask      = feat_df['gap_unfilled_eod'] == 1
    direction = pd.Series(1, index=feat_df.index)
    return mask, direction


def _triggers_seed003(feat_df: pd.DataFrame):
    """rsi_2 < 10 for >= 2 consecutive days AND (close > SMA_50 OR close > SMA_200*0.85)."""
    close    = feat_df['close']
    sma_50   = close.rolling(50).mean()
    sma_200  = close.rolling(200).mean()
    rsi2_low = feat_df['rsi_2'] < 10
    consec   = rsi2_low & rsi2_low.shift(1).fillna(False)
    trend_ok = (close > sma_50) | (close > sma_200 * 0.85)
    mask      = consec & trend_ok
    direction = pd.Series(1, index=feat_df.index)
    return mask, direction


def _triggers_seed004(feat_df: pd.DataFrame, regime_df: pd.DataFrame):
    """mom_5d > 0 AND rsi_14 > 55 AND regime in (H_BULL, L_BULL) AND vol_expanding == 0."""
    joined = feat_df.join(regime_df, how='left')
    bull   = joined['regime_state'].isin(['H_BULL', 'L_BULL'])
    mask   = (feat_df['mom_5d'] > 0) & (feat_df['rsi_14'] > 55) & (feat_df['vol_expanding'] == 0) & bull
    direction = pd.Series(1, index=feat_df.index)
    return mask, direction


def _triggers_seed005(feat_df: pd.DataFrame):
    """XAUUSD z_21d < -2.0 (long) OR > 2.0 (short). Direct symbol now."""
    long_mask  = feat_df['z_21d'] < -2.0
    short_mask = feat_df['z_21d'] > 2.0
    mask       = long_mask | short_mask
    direction  = pd.Series(np.where(long_mask, 1, -1), index=feat_df.index)
    return mask, direction


def _triggers_seed006(feat_df: pd.DataFrame):
    """z_21d < -1.8 (long) OR z_21d > 1.8 (short)."""
    long_mask  = feat_df['z_21d'] < -1.8
    short_mask = feat_df['z_21d'] > 1.8
    mask       = long_mask | short_mask
    direction  = pd.Series(np.where(long_mask, 1, -1), index=feat_df.index)
    return mask, direction


# ---------------------------------------------------------------------------
# Multi-pair aggregation (single-symbol edges applying across the majors)
# ---------------------------------------------------------------------------

def _aggregate_multi_pair_backtest(trigger_fn, holding_period, window_start, window_end,
                                    edge_id, regime_df=None, pairs=None):
    """Run one edge's trigger across a list of pairs and pool the trades."""
    pairs = pairs or ALL_MAJORS
    all_returns = []
    for pair in pairs:
        try:
            feat = _load(pair)
            close = feat['close']
            if regime_df is not None:
                mask, direction = trigger_fn(feat, regime_df)
            else:
                mask, direction = trigger_fn(feat)
            fwd = _compute_forward_returns(close, holding_period)
            m   = mask & mask.index.to_series().between(window_start, window_end)
            idx = m[m].index
            if len(idx) == 0:
                continue
            raw = fwd.loc[idx] * direction.loc[idx]
            all_returns.append(raw)
        except Exception:
            continue
    if not all_returns:
        return {'n': 0, 'win_rate': np.nan, 'avg_return': np.nan,
                'avg_return_net': np.nan, 'p_value': np.nan}
    combined = pd.concat(all_returns).dropna()
    floored  = combined.clip(lower=_stop_floor(edge_id))
    n        = len(combined)
    win_rate = (combined > 0).mean()
    avg_return     = combined.mean()
    avg_return_net = floored.mean()
    if n > 1:
        _, p_value = stats.ttest_1samp(floored, 0)
    else:
        p_value = np.nan
    return {'n': n, 'win_rate': win_rate, 'avg_return': avg_return,
            'avg_return_net': avg_return_net, 'p_value': p_value}


# ---------------------------------------------------------------------------
# SEED-007 — cross-pair spread backtest (NEW)
# ---------------------------------------------------------------------------

def _spread_series(pair_a: str, pair_b: str, window: int = CROSS_BETA_WINDOW,
                   fixed_beta: float = None):
    """
    Log-price spread and its 21-day z-score.

    fixed_beta=None floats the hedge ratio via a rolling OLS-via-covariance
    estimate. fixed_beta=1.0 trades the raw log-price difference, which is
    what a direct MetaTrader cross (EURGBP, AUDNZD) effectively gives you.

    The floating version is NOT free precision. Two majors are both
    dominated by their own random-walk component, so a short-window OLS
    beta between them is a textbook spurious regression (Granger &
    Newbold): the beta estimate carries sampling noise on the same scale
    as the series themselves, which swamps whatever small genuine
    cointegrating signal exists. Verified directly — on a pair constructed
    to be genuinely cointegrated, the fixed-beta=1 spread traded at 61% win
    rate / positive expectancy, while the floating-beta version (built from
    the SAME data) traded at 22% / negative. The floating estimator was
    correct as an OLS slope at any single point (checked against
    np.polyfit) — the noise comes from beta moving day to day, not from a
    formula error.
    """
    a = np.log(_load(pair_a)['close'])
    b = np.log(_load(pair_b)['close'])
    df = pd.DataFrame({'a': a, 'b': b}).dropna()

    if fixed_beta is not None:
        beta = fixed_beta
    else:
        cov  = df['a'].rolling(window).cov(df['b'])
        var  = df['b'].rolling(window).var()
        beta = (cov / var).clip(-3, 3)   # bounded — a runaway beta is a data artifact, not a real hedge ratio

    spread = df['a'] - beta * df['b']
    z = (spread - spread.rolling(21).mean()) / spread.rolling(21).std()
    return spread, z


def _aggregate_cross_backtest(holding_period, window_start, window_end, edge_id,
                              fixed_beta: float = None):
    """Run the spread edge across every leg pair in CROSS_PAIRS and pool."""
    all_returns = []
    for pair_a, pair_b in CROSS_PAIRS:
        try:
            spread, z = _spread_series(pair_a, pair_b, fixed_beta=fixed_beta)
            long_mask  = z < -CROSS_SPREAD_Z_ENTRY
            short_mask = z > CROSS_SPREAD_Z_ENTRY
            mask       = long_mask | short_mask
            direction  = pd.Series(np.where(long_mask, 1, -1), index=z.index)
            fwd = spread.shift(-holding_period) - spread
            m   = mask & mask.index.to_series().between(window_start, window_end)
            idx = m[m].index
            if len(idx) == 0:
                continue
            all_returns.append((fwd.loc[idx] * direction.loc[idx]).dropna())
        except Exception:
            continue
    if not all_returns:
        return {'n': 0, 'win_rate': np.nan, 'avg_return': np.nan,
                'avg_return_net': np.nan, 'p_value': np.nan}
    combined = pd.concat(all_returns)
    floored  = combined.clip(lower=_stop_floor(edge_id))
    n        = len(combined)
    win_rate = (combined > 0).mean()
    avg_return     = combined.mean()
    avg_return_net = floored.mean()
    if n > 1:
        _, p_value = stats.ttest_1samp(floored, 0)
    else:
        p_value = np.nan
    return {'n': n, 'win_rate': win_rate, 'avg_return': avg_return,
            'avg_return_net': avg_return_net, 'p_value': p_value}


# ---------------------------------------------------------------------------
# Per-edge revalidation dispatch
# ---------------------------------------------------------------------------

def revalidate_seeded_edge(edge_id: str, regime_df: pd.DataFrame) -> dict:
    """Compute empirical IS and OOS stats for one seeded edge."""
    if edge_id == 'SEED-001':
        is_stats  = _aggregate_multi_pair_backtest(_triggers_seed001, 5, TRAIN_START, TRAIN_END, edge_id)
        oos_stats = _aggregate_multi_pair_backtest(_triggers_seed001, 5, OOS_START,   OOS_END,   edge_id)

    elif edge_id == 'SEED-002':
        is_stats  = _aggregate_multi_pair_backtest(_triggers_seed002, 2, TRAIN_START, TRAIN_END, edge_id)
        oos_stats = _aggregate_multi_pair_backtest(_triggers_seed002, 2, OOS_START,   OOS_END,   edge_id)

    elif edge_id == 'SEED-003':
        is_stats  = _aggregate_multi_pair_backtest(_triggers_seed003, 5, TRAIN_START, TRAIN_END, edge_id)
        oos_stats = _aggregate_multi_pair_backtest(_triggers_seed003, 5, OOS_START,   OOS_END,   edge_id)

    elif edge_id == 'SEED-004':
        def fn(feat, reg): return _triggers_seed004(feat, reg)
        is_stats  = _aggregate_multi_pair_backtest(fn, 10, TRAIN_START, TRAIN_END, edge_id, regime_df)
        oos_stats = _aggregate_multi_pair_backtest(fn, 10, OOS_START,   OOS_END,   edge_id, regime_df)

    elif edge_id == 'SEED-005':
        # Direct XAUUSD symbol. v4.1 loaded 'GOLD_USD' as a proxy for the
        # synthetic pair's gold leg; that path is gone along with the
        # synthetic construction. If this parquet is missing, the failure
        # is left loud rather than silently degraded to INSUFFICIENT_DATA —
        # a missing gold history is worth noticing, not masking.
        feat  = _load('XAUUSD')
        close = feat['close']
        mask, direction = _triggers_seed005(feat)
        floor = _stop_floor(edge_id, pair='XAUUSD')
        is_stats  = _run_backtest(mask, direction, close, 5, TRAIN_START, TRAIN_END, floor)
        oos_stats = _run_backtest(mask, direction, close, 5, OOS_START,   OOS_END,   floor)

    elif edge_id == 'SEED-006':
        is_stats  = _aggregate_multi_pair_backtest(_triggers_seed006, 10, TRAIN_START, TRAIN_END, edge_id)
        oos_stats = _aggregate_multi_pair_backtest(_triggers_seed006, 10, OOS_START,   OOS_END,   edge_id)

    elif edge_id == 'SEED-007':
        # Compare the fixed 1:1 ratio (equivalent to a direct MetaTrader
        # cross like EURGBP) against the floating rolling-beta hedge ratio.
        # Default to fixed unless floating clearly wins on both win rate
        # and net return in-sample — a rolling OLS beta between two
        # integrated series is a known spurious-regression trap (see
        # _spread_series docstring), so it does not get the benefit of the
        # doubt on a marginal in-sample win.
        is_fixed  = _aggregate_cross_backtest(10, TRAIN_START, TRAIN_END, edge_id, fixed_beta=1.0)
        is_float  = _aggregate_cross_backtest(10, TRAIN_START, TRAIN_END, edge_id, fixed_beta=None)

        float_wins = (
            not np.isnan(is_float['win_rate']) and not np.isnan(is_fixed['win_rate'])
            and is_float['win_rate'] > is_fixed['win_rate']
            and is_float['avg_return_net'] > is_fixed['avg_return_net']
        )
        use_beta = None if float_wins else 1.0
        print(f"  [SEED-007] fixed-beta IS: n={is_fixed['n']} wr={is_fixed['win_rate']:.1%}  "
              f"floating-beta IS: n={is_float['n']} wr={is_float['win_rate']:.1%}  "
              f"-> using {'floating' if float_wins else 'fixed'} beta")

        is_stats  = is_fixed if not float_wins else is_float
        oos_stats = _aggregate_cross_backtest(10, OOS_START, OOS_END, edge_id, fixed_beta=use_beta)

    else:
        raise ValueError(f"Unknown edge_id: {edge_id}")

    return {'is': is_stats, 'oos': oos_stats}


# ---------------------------------------------------------------------------
# Acceptance gate
# ---------------------------------------------------------------------------

def _evaluate_status(edge_id: str, research_wr: float, is_stats: dict, oos_stats: dict) -> tuple:
    """Returns (status, failure_reason)."""
    n          = is_stats['n']
    win_rate   = is_stats['win_rate']
    avg_net    = is_stats['avg_return_net']
    oos_wr     = oos_stats['win_rate']

    failures = []

    if n < 30:
        failures.append('INSUFFICIENT_DATA')
    if not np.isnan(win_rate) and win_rate <= 0.50:
        failures.append('FAILED_WIN_RATE')
    if not np.isnan(avg_net) and avg_net <= 0:
        failures.append('FAILED_AVG_RETURN')
    if (not np.isnan(win_rate) and not np.isnan(oos_wr)
            and oos_wr < win_rate - 0.08):
        failures.append('OOS_DIVERGENCE')
    if (not np.isnan(research_wr) and not np.isnan(win_rate)
            and win_rate < research_wr - 0.10):
        failures.append('MISSED_RESEARCH_PRIOR')

    if failures:
        return 'FAILED_REVALIDATION', ', '.join(failures)
    return 'ACTIVE', None


# ---------------------------------------------------------------------------
# DB update
# ---------------------------------------------------------------------------

def _update_edge(conn: sqlite3.Connection, edge_id: str, status: str,
                 failure_reason, is_stats: dict, oos_stats: dict):
    conn.execute(
        """UPDATE edge_library SET
            status             = ?,
            win_rate           = ?,
            avg_return_net     = ?,
            sample_size        = ?,
            p_value            = ?,
            oos_win_rate       = ?,
            decay_cause        = ?,
            needs_revalidation = 0
        WHERE edge_id = ?""",
        (
            status,
            is_stats['win_rate'] if not np.isnan(is_stats.get('win_rate', np.nan)) else None,
            is_stats['avg_return_net'] if not np.isnan(is_stats.get('avg_return_net', np.nan)) else None,
            is_stats['n'],
            is_stats['p_value'] if not np.isnan(is_stats.get('p_value', np.nan)) else None,
            oos_stats['win_rate'] if not np.isnan(oos_stats.get('win_rate', np.nan)) else None,
            failure_reason,
            edge_id,
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_revalidation(db_path: str = DB_PATH):
    print("=== Step 5.5 — Seeded Edge Revalidation ===\n")

    regime_df = _load_regime()

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT edge_id, win_rate FROM edge_library WHERE edge_provenance='SEEDED' ORDER BY edge_id"
    ).fetchall()
    research_priors = {r[0]: r[1] for r in rows}
    total = len(research_priors)

    active_count = 0
    results = []

    for edge_id, research_wr in sorted(research_priors.items()):
        stats_dict = revalidate_seeded_edge(edge_id, regime_df)
        is_s   = stats_dict['is']
        oos_s  = stats_dict['oos']
        status, reason = _evaluate_status(edge_id, research_wr, is_s, oos_s)

        _update_edge(conn, edge_id, status, reason, is_s, oos_s)

        if status == 'ACTIVE':
            active_count += 1

        results.append((edge_id, status, research_wr, is_s, oos_s, reason))

    conn.commit()
    conn.close()

    print(f"  {'edge_id':<10} {'status':<22} {'IS WR':>7} {'prior WR':>9} {'n':>5} {'OOS WR':>8}  reason")
    print('  ' + '-' * 80)
    for (edge_id, status, research_wr, is_s, oos_s, reason) in results:
        is_wr  = f"{is_s['win_rate']:.1%}"  if not np.isnan(is_s.get('win_rate', np.nan) or np.nan) else 'N/A'
        oos_wr = f"{oos_s['win_rate']:.1%}" if not np.isnan(oos_s.get('win_rate', np.nan) or np.nan) else 'N/A'
        r_str  = reason or ''
        print(f"  {edge_id:<10} {status:<22} {is_wr:>7} {research_wr:>9.1%} {is_s['n']:>5} {oos_wr:>8}  {r_str}")

    print()
    # A little over half surviving is the same bar v4.1 set (4/6). Scaled
    # to the current edge count rather than hardcoded.
    pass_bar = (total // 2) + 1
    if active_count >= pass_bar:
        print(f"  PASS — {active_count}/{total} edges reached ACTIVE status.")
    else:
        print(f"  WARNING — only {active_count}/{total} edges reached ACTIVE status.")
        print("  Check data pipeline and feature computation before proceeding.")

    print("\n=== Step 5.5 complete ===")
    return active_count


if __name__ == '__main__':
    run_revalidation()
