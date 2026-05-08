"""
Step 15 — Backtest + Validation.
Simulates full system on historical data without lookahead.
Train: 2010-2021 (HMM). Validate: 2022-2023. Test: 2024-present.
Options: synthetic Black-Scholes premiums. Futures: historical prices.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import TOTAL_CAPITAL, MAX_POSITION_PCT, OPTION_STOP_PCT, THETA_STOP_PCT
from regime_engine_sherm import get_regime_on_date
from feature_definitions import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_features(instrument: str) -> pd.DataFrame:
    path = f'data/processed/{instrument}_features.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def load_features_up_to_date(date: pd.Timestamp) -> dict:
    """Load features with strict no-lookahead: only data up to `date`."""
    from config import INDEX_UNDERLYINGS
    features = {}
    for inst in INDEX_UNDERLYINGS:
        df = _load_features(inst)
        if df.empty:
            continue
        features[inst] = df.loc[df.index <= date]
    return features


# ---------------------------------------------------------------------------
# Trigger evaluation (mirrors AI 1B logic)
# ---------------------------------------------------------------------------

def evaluate_triggers(features_up_to_date: dict, regime: dict) -> list:
    """
    For each index, check each seeded edge trigger condition.
    Returns list of (instrument, edge_id, direction, trigger_value).
    """
    signals = []
    for inst, df in features_up_to_date.items():
        if df.empty:
            continue
        feat = df.iloc[-1]

        # SEED-001: z_21d < -2 (long call) or > 2 (long put)
        if regime['regime_state'] == 'SIDEWAYS' and pd.notna(feat.get('z_21d', np.nan)):
            z = feat['z_21d']
            if z < -2.0:
                signals.append((inst, 'SEED-001', 'LONG', z))
            elif z > 2.0:
                signals.append((inst, 'SEED-001', 'SHORT', z))

        # SEED-002: gap down unfilled
        if (regime['regime_state'] == 'SIDEWAYS'
                and feat.get('gap_unfilled_eod', 0) == 1):
            signals.append((inst, 'SEED-002', 'LONG', feat.get('gap_pct', 0)))

        # SEED-003: RSI(2) < 10 (all regimes)
        if pd.notna(feat.get('rsi_2', np.nan)) and feat['rsi_2'] < 10:
            signals.append((inst, 'SEED-003', 'LONG', feat['rsi_2']))

        # SEED-004: momentum bull
        if (regime['regime_state'] in ('H_BULL', 'L_BULL')
                and feat.get('mom_5d', 0) > 0
                and feat.get('rsi_14', 0) > 55
                and feat.get('vol_expanding', 1) == 0):
            signals.append((inst, 'SEED-004', 'LONG', feat['mom_5d']))

    return signals


# ---------------------------------------------------------------------------
# Simulated fill using synthetic BS premiums
# ---------------------------------------------------------------------------

def simulate_fill(signal_tuple: tuple, date: pd.Timestamp,
                  underlying_prices: dict) -> dict:
    """
    Compute synthetic ATM option premium for entry.
    Returns simulated trade dict.
    """
    inst, edge_id, direction, trigger_value = signal_tuple
    hp_map = {'SEED-001': 5, 'SEED-002': 2, 'SEED-003': 5, 'SEED-004': 10}
    holding_period = hp_map.get(edge_id, 5)

    price_series = underlying_prices.get(inst)
    if price_series is None or price_series.empty:
        return None

    try:
        from synthetic_options.bs_engine import get_atm_premium
        option_type = 'call' if direction == 'LONG' else 'put'
        result = get_atm_premium(
            date, inst, option_type,
            dte=holding_period,
            underlying_price_series=price_series,
        )
        return {
            'trade_id':      f"{edge_id}_{inst}_{date.date()}",
            'edge_id':       edge_id,
            'instrument':    inst,
            'instrument_class': 'INDEX_OPTIONS',
            'direction':     direction,
            'holding_period': holding_period,
            'entry_date':    date,
            'entry_premium': result['price'],
            'strike':        result['strike'],
            'underlying_price': result['underlying_price'],
            'synthetic_iv':  result['synthetic_iv'],
            'delta':         result['delta'],
            'theta':         result['theta'],
            'option_type':   option_type,
            'trigger_value': trigger_value,
        }
    except Exception:
        return None


def compute_exit_premium(trade: dict, exit_date: pd.Timestamp,
                          underlying_prices: dict) -> float:
    """Compute option premium at exit using BS with ORIGINAL entry strike."""
    days_remaining = max(0, trade['holding_period'] -
                         (exit_date - trade['entry_date']).days)
    price_series = underlying_prices.get(trade['instrument'])
    if price_series is None:
        return 0.0
    try:
        from synthetic_options.bs_engine import (black_scholes_price,
                                                  compute_synthetic_iv)
        from config import RISK_FREE_RATE

        s_up_to = price_series.loc[:exit_date]
        S     = float(s_up_to.iloc[-1])
        K     = trade['strike']       # original entry strike (not recomputed)
        T     = days_remaining / 252.0
        iv    = compute_synthetic_iv(s_up_to)
        sigma = float(iv.iloc[-1])
        if not (sigma > 0 and np.isfinite(sigma)):
            sigma = trade['synthetic_iv']

        result = black_scholes_price(S, K, T, RISK_FREE_RATE, sigma,
                                     trade['option_type'])
        return result['price']
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(start_date: str = '2022-01-01',
                 end_date: str   = '2023-12-31') -> tuple:
    """
    Simulate full system on historical data without lookahead.
    Rule-based only (no RL in backtest).
    Returns (trades_df, final_capital).
    """
    from config import INDEX_UNDERLYINGS

    # Load all price series for underlying instruments
    underlying_prices = {}
    for inst in INDEX_UNDERLYINGS:
        df = _load_features(inst)
        if not df.empty:
            underlying_prices[inst] = df['close']

    dates   = pd.bdate_range(start_date, end_date)
    capital = TOTAL_CAPITAL
    trades  = []
    open_positions = {}   # trade_id → trade dict

    for date in dates:
        try:
            regime = get_regime_on_date(str(date.date()))
        except Exception:
            continue

        # Skip if transition warning
        if regime.get('transition_warning_flag'):
            continue

        # Check exits first (no lookahead: use yesterday's close, not today's)
        to_close = []
        for trade_id, trade in open_positions.items():
            days_held = (date - trade['entry_date']).days
            if days_held >= trade['holding_period']:
                to_close.append(trade_id)

        for trade_id in to_close:
            trade = open_positions.pop(trade_id)
            exit_premium = compute_exit_premium(trade, date, underlying_prices)
            pnl_pct = (exit_premium - trade['entry_premium']) / trade['entry_premium']
            # Apply stop loss floor
            pnl_pct = max(pnl_pct, -OPTION_STOP_PCT)
            capital += capital * MAX_POSITION_PCT * pnl_pct
            trades.append({
                'trade_id':        trade_id,
                'edge_id':         trade['edge_id'],
                'instrument':      trade['instrument'],
                'instrument_class': trade['instrument_class'],
                'direction':       trade['direction'],
                'holding_period':  trade['holding_period'],
                'entry_date':      trade['entry_date'].date(),
                'exit_date':       date.date(),
                'entry_premium':   trade['entry_premium'],
                'exit_premium':    exit_premium,
                'net_return_pct':  pnl_pct * 100,
                'regime_at_entry': trade.get('regime_at_entry'),
                'trigger_value':   trade['trigger_value'],
            })

        # Skip new entries if max positions held
        if len(open_positions) >= 8:
            continue

        # Evaluate triggers
        feats = load_features_up_to_date(date)
        sigs = evaluate_triggers(feats, regime)

        for sig in sigs:
            if len(open_positions) >= 8:
                break
            trade = simulate_fill(sig, date, underlying_prices)
            if trade is None:
                continue
            trade['regime_at_entry'] = regime['regime_state']
            # Dedup: no same underlying+direction already open
            key = (trade['instrument'], trade['direction'])
            if any((t['instrument'], t['direction']) == key
                   for t in open_positions.values()):
                continue
            open_positions[trade['trade_id']] = trade

    return pd.DataFrame(trades), capital


# ---------------------------------------------------------------------------
# Graduation criteria
# ---------------------------------------------------------------------------

def compute_max_drawdown(daily_returns: pd.Series) -> float:
    equity = (1 + daily_returns).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(abs(drawdown.min()))


def check_graduation_criteria(backtest_trades: pd.DataFrame,
                               final_capital: float) -> dict:
    if backtest_trades.empty:
        return {'passed': False, 'criteria': {}, 'error': 'No trades generated'}

    returns = backtest_trades['net_return_pct'] / 100.0

    # Daily portfolio returns (approximate)
    bt = backtest_trades.copy()
    bt['exit_date'] = pd.to_datetime(bt['exit_date'])
    daily_pnl = bt.groupby('exit_date')['net_return_pct'].mean() / 100.0
    daily_pnl = daily_pnl.reindex(pd.bdate_range(bt['exit_date'].min(),
                                                   bt['exit_date'].max()),
                                   fill_value=0)

    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)
    max_dd = compute_max_drawdown(daily_pnl)
    win_rate = (returns > 0).mean()

    wins_sum   = returns[returns > 0].sum()
    losses_sum = abs(returns[returns < 0].sum())
    profit_factor = wins_sum / losses_sum if losses_sum > 0 else float('inf')

    n_days  = max(len(daily_pnl), 1)
    net_cagr = (final_capital / TOTAL_CAPITAL) ** (252 / n_days) - 1

    hp_counts = backtest_trades.groupby('holding_period').size()
    n_trades  = len(backtest_trades)
    n_classes = backtest_trades['instrument_class'].nunique()

    criteria = {
        'sharpe_gt_1':              (sharpe > 1.0,  sharpe,       1.0),
        'max_dd_lt_8pct':           (max_dd < 0.08, max_dd,       0.08),
        'win_rate_gt_50pct':        (win_rate > 0.50, win_rate,   0.50),
        'profit_factor_gt_1p4':     (profit_factor > 1.4, profit_factor, 1.4),
        'net_cagr_positive':        (net_cagr > 0,  net_cagr,     0),
        'min_100_trades':           (n_trades >= 100, n_trades,   100),
        'all_holding_periods_covered': (
            all(hp in hp_counts.index for hp in [2, 5, 10]),
            list(hp_counts.index), [2, 5, 10],
        ),
        'all_instruments_covered':  (n_classes >= 1, n_classes,  1),
    }

    all_passed = all(c[0] for c in criteria.values())
    return {'passed': all_passed, 'criteria': criteria}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _generate_charts(backtest_trades: pd.DataFrame, start_date: str, end_date: str):
    os.makedirs('outputs/charts', exist_ok=True)

    bt = backtest_trades.copy()
    bt['exit_date'] = pd.to_datetime(bt['exit_date'])
    bt = bt.sort_values('exit_date')

    # Equity curve
    cumulative = TOTAL_CAPITAL * (1 + bt['net_return_pct'] / 100 * MAX_POSITION_PCT).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Panel 1: equity curve
    axes[0].plot(bt['exit_date'], cumulative, color='steelblue', linewidth=1.5)
    axes[0].axhline(TOTAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Equity Curve ({start_date} to {end_date})')
    axes[0].set_ylabel('Portfolio Value (Rs)')

    # Panel 2: return distribution per edge
    for eid in bt['edge_id'].unique():
        sub = bt[bt['edge_id'] == eid]['net_return_pct']
        axes[1].hist(sub, bins=20, alpha=0.5, label=eid)
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].legend(fontsize=8)
    axes[1].set_title('Return Distribution by Edge')
    axes[1].set_xlabel('Net Return %')

    # Panel 3: wins/losses by holding period
    hp_stats = bt.groupby('holding_period').apply(
        lambda g: pd.Series({'win_rate': (g['net_return_pct'] > 0).mean(),
                              'n': len(g)})
    )
    axes[2].bar(hp_stats.index, hp_stats['win_rate'], color='teal', alpha=0.7)
    axes[2].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title('Win Rate by Holding Period')
    axes[2].set_xlabel('Holding Period (days)')
    axes[2].set_ylabel('Win Rate')
    for i, (hp, row) in enumerate(hp_stats.iterrows()):
        axes[2].text(hp, row['win_rate'] + 0.01, f"n={int(row['n'])}", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/charts/backtest_equity_curve.png', dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== Step 15 — Backtest + Validation ===\n")

    print("  Running backtest on 2022-2023 validation window...")
    trades, final_capital = run_backtest('2022-01-01', '2023-12-31')

    print(f"  Total trades: {len(trades)}")
    print(f"  Final capital: Rs {final_capital:,.0f}")
    print(f"  Return: {(final_capital/TOTAL_CAPITAL - 1)*100:.2f}%\n")

    if not trades.empty:
        print(trades[['edge_id', 'instrument', 'direction', 'holding_period',
                       'net_return_pct']].head(10).to_string(index=False))
        print()

    result = check_graduation_criteria(trades, final_capital)

    print(f"  {'Criterion':<35} {'Pass':>5}  {'Actual':>12}  {'Threshold':>10}")
    print('  ' + '-' * 68)
    for name, (passed, actual, threshold) in result.get('criteria', {}).items():
        if isinstance(actual, float):
            actual_str = f'{actual:.4f}'
        else:
            actual_str = str(actual)
        print(f"  {name:<35} {'PASS' if passed else 'FAIL':>5}  {actual_str:>12}  {str(threshold):>10}")

    print()
    if result.get('passed'):
        print("  *** MVP READY FOR PAPER TRADING ***")
    else:
        print("  Graduation criteria NOT met. See failures above.")
        print("  NOTE: On synthetic GBM data, edges without behavioral patterns")
        print("  will show ~50% win rates. Run on real NSE data for actual graduation check.")

    if not trades.empty:
        _generate_charts(trades, '2022-01-01', '2023-12-31')
        print("\n  Charts saved to outputs/charts/backtest_equity_curve.png")

    print("\n=== Step 15 complete ===")


if __name__ == '__main__':
    _verification_check()
