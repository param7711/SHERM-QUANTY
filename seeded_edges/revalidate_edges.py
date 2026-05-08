"""
Step 6.5 — Seeded edge revalidation.
Computes empirical priors on Indian historical data (2010-2021 train window)
and OOS check (2022-2023). Replaces research priors in Edge Library.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

from config import DB_PATH

TRAIN_START = '2010-01-01'
TRAIN_END   = '2021-12-31'
OOS_START   = '2022-01-01'
OOS_END     = '2023-12-31'

STOP_LOSS_FLOOR = -0.35   # options stop at 35% premium loss


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load(instrument: str) -> pd.DataFrame:
    path = f'data/processed/{instrument}_features.parquet'
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def _load_regime() -> pd.DataFrame:
    df = pd.read_parquet('data/processed/regime_history.parquet')
    df.index = pd.to_datetime(df.index)
    return df[['regime_state']]


def _load_raw_close(instrument: str) -> pd.Series:
    path = f'data/raw/{instrument}_daily.parquet'
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df['close']


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def _compute_forward_returns(close: pd.Series, holding_period: int) -> pd.Series:
    """Log return from close[t] to close[t+holding_period]."""
    return np.log(close.shift(-holding_period) / close)


def _run_backtest(trigger_mask: pd.Series,
                  direction_series: pd.Series,
                  close: pd.Series,
                  holding_period: int,
                  window_start: str,
                  window_end: str) -> dict:
    """
    trigger_mask: boolean Series — True on signal dates
    direction_series: +1 for long, -1 for short (same index)
    Returns stats dict.
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

    # Apply options stop-loss floor to simulate 35% cap on loss
    floored = trade_returns.clip(lower=STOP_LOSS_FLOOR)

    n              = len(trade_returns)
    win_rate       = (trade_returns > 0).mean()
    avg_return     = trade_returns.mean()
    avg_return_net = floored.mean()

    if n > 1:
        t_stat, p_value = stats.ttest_1samp(floored.dropna(), 0)
    else:
        p_value = np.nan

    return {
        'n':              n,
        'win_rate':       win_rate,
        'avg_return':     avg_return,
        'avg_return_net': avg_return_net,
        'p_value':        p_value,
    }


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
    """gap_pct < -0.01 AND close < prev_close — all long."""
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
    """synthetic_xauusd z_21d < -2.0 (long) OR > 2.0 (short)."""
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
# Per-edge revalidation
# ---------------------------------------------------------------------------

INDEX_INSTRUMENTS = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']


def _aggregate_index_backtest(trigger_fn, holding_period, window_start, window_end,
                               regime_df=None):
    """Run backtest across all 4 indices and aggregate results."""
    all_returns = []
    for inst in INDEX_INSTRUMENTS:
        try:
            feat = _load(inst)
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
    floored  = combined.clip(lower=STOP_LOSS_FLOOR)
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


def revalidate_seeded_edge(edge_id: str, regime_df: pd.DataFrame) -> dict:
    """Compute empirical IS and OOS stats for one seeded edge."""
    if edge_id == 'SEED-001':
        is_stats  = _aggregate_index_backtest(_triggers_seed001, 5, TRAIN_START, TRAIN_END)
        oos_stats = _aggregate_index_backtest(_triggers_seed001, 5, OOS_START,   OOS_END)

    elif edge_id == 'SEED-002':
        is_stats  = _aggregate_index_backtest(_triggers_seed002, 2, TRAIN_START, TRAIN_END)
        oos_stats = _aggregate_index_backtest(_triggers_seed002, 2, OOS_START,   OOS_END)

    elif edge_id == 'SEED-003':
        is_stats  = _aggregate_index_backtest(_triggers_seed003, 5, TRAIN_START, TRAIN_END)
        oos_stats = _aggregate_index_backtest(_triggers_seed003, 5, OOS_START,   OOS_END)

    elif edge_id == 'SEED-004':
        def fn(feat, reg): return _triggers_seed004(feat, reg)
        is_stats  = _aggregate_index_backtest(fn, 10, TRAIN_START, TRAIN_END, regime_df)
        oos_stats = _aggregate_index_backtest(fn, 10, OOS_START,   OOS_END,   regime_df)

    elif edge_id == 'SEED-005':
        feat  = _load('GOLD_USD')
        close = feat['close']
        mask, direction = _triggers_seed005(feat)
        is_stats  = _run_backtest(mask, direction, close, 5, TRAIN_START, TRAIN_END)
        oos_stats = _run_backtest(mask, direction, close, 5, OOS_START,   OOS_END)

    elif edge_id == 'SEED-006':
        feat  = _load('EURUSD')
        close = feat['close']
        mask, direction = _triggers_seed006(feat)
        is_stats  = _run_backtest(mask, direction, close, 10, TRAIN_START, TRAIN_END)
        oos_stats = _run_backtest(mask, direction, close, 10, OOS_START,   OOS_END)

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
            status        = ?,
            win_rate      = ?,
            avg_return_net= ?,
            sample_size   = ?,
            p_value       = ?,
            oos_win_rate  = ?,
            decay_cause   = ?
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
    print("=== Step 6.5 — Seeded Edge Revalidation ===\n")

    regime_df = _load_regime()

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT edge_id, win_rate FROM edge_library WHERE edge_provenance='SEEDED' ORDER BY edge_id"
    ).fetchall()
    research_priors = {r[0]: r[1] for r in rows}

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

    # Print report
    print(f"  {'edge_id':<10} {'status':<22} {'IS WR':>7} {'prior WR':>9} {'n':>5} {'OOS WR':>8}  reason")
    print('  ' + '-' * 80)
    for (edge_id, status, research_wr, is_s, oos_s, reason) in results:
        is_wr  = f"{is_s['win_rate']:.1%}"  if not np.isnan(is_s.get('win_rate', np.nan) or np.nan) else 'N/A'
        oos_wr = f"{oos_s['win_rate']:.1%}" if not np.isnan(oos_s.get('win_rate', np.nan) or np.nan) else 'N/A'
        r_str  = reason or ''
        print(f"  {edge_id:<10} {status:<22} {is_wr:>7} {research_wr:>9.1%} {is_s['n']:>5} {oos_wr:>8}  {r_str}")

    print()
    if active_count >= 4:
        print(f"  PASS — {active_count}/6 edges reached ACTIVE status.")
    else:
        print(f"  WARNING — only {active_count}/6 edges reached ACTIVE status.")
        print("  Check data pipeline and feature computation before proceeding.")

    print("\n=== Step 6.5 complete ===")
    return active_count


if __name__ == '__main__':
    run_revalidation()
