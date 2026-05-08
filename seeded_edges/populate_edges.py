"""
Step 6 — Seeded edge population.
Writes all 6 seeded edges to the Edge Library with research priors.
These are starting priors — Step 6.5 will replace them with empirical priors.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from datetime import datetime

from config import DB_PATH


SEEDED_EDGES = [
    {
        'edge_id': 'SEED-001',
        'instrument_class': 'INDEX_OPTIONS',
        'underlying': 'ALL_INDICES',
        'expiry_type': 'WEEKLY',
        'trigger_feature': 'z_21d',
        'trigger_condition': 'z_21d < -2.0 → BUY ATM CALL; z_21d > 2.0 → BUY ATM PUT',
        'direction': 'BOTH',
        'holding_period': 5,
        'win_rate': 0.58,
        'avg_return_net': 0.22,
        'avg_return_gross': None,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'SIDEWAYS',
        'vix_bucket': None,
        'iv_rank_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'feature_set_version': 'v1',
        'regime_model_version': 'v1',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': 'Avellaneda & Lee (2010), Statistical Arbitrage in US Equities Market; Lo & MacKinlay (1990)',
        'regime_at_discovery': None,
        'theta_window_days': 5,
        'synthetic_pricing_used': 1,
        'trigger_frequency_baseline': None,
        'trigger_frequency_current': None,
        'frequency_decay_flag': 0,
        'win_rate_watch_flag': 0,
        'posterior_win_rate': None,
        'live_wins_in_regime': 0,
        'live_losses_in_regime': 0,
        'resurrection_priority': None,
        'resurrection_attempts': 0,
        'mechanism_story': (
            'Equity indices exhibit short-term mean reversion in sideways and mildly '
            'bearish regimes due to institutional rebalancing flows and overnight '
            'short-covering pressure. When index z-score drops below -2, it is '
            'approximately 2 standard deviations below its 21-day mean — a level '
            'that historically precedes a bounce within 2-7 trading sessions. '
            'Expressed through ATM CALL options for capital efficiency and defined risk. '
            'Mirror logic for z > +2.0 and PUT buying.'
        ),
        'signal_strength': None,
    },
    {
        'edge_id': 'SEED-002',
        'instrument_class': 'INDEX_OPTIONS',
        'underlying': 'ALL_INDICES',
        'expiry_type': 'WEEKLY',
        'trigger_feature': 'gap_unfilled_eod',
        'trigger_condition': 'gap_pct < -0.01 AND close < prev_close → BUY ATM CALL next open',
        'direction': 'LONG',
        'holding_period': 2,
        'win_rate': 0.60,
        'avg_return_net': 0.20,
        'avg_return_gross': None,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'SIDEWAYS',
        'vix_bucket': None,
        'iv_rank_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'feature_set_version': 'v1',
        'regime_model_version': 'v1',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': 'Cooper, Gutierrez, Hameed (2004), Market States and Momentum; Caginalp & Constantine (1995)',
        'regime_at_discovery': None,
        'theta_window_days': 2,
        'synthetic_pricing_used': 1,
        'trigger_frequency_baseline': None,
        'trigger_frequency_current': None,
        'frequency_decay_flag': 0,
        'win_rate_watch_flag': 0,
        'posterior_win_rate': None,
        'live_wins_in_regime': 0,
        'live_losses_in_regime': 0,
        'resurrection_priority': None,
        'resurrection_attempts': 0,
        'mechanism_story': (
            'Overnight gaps in equity indices, when not driven by major news, tend to '
            'fill within 1-3 trading sessions. Driven by market maker repositioning '
            'and short-cover dynamics. Signal fires when index gaps down >1% and '
            'does not fill by EOD. Enter via ATM CALL next morning, targeting gap fill '
            '(previous close price) as exit. News filter critical: no RBI/Fed event '
            'in next 3 days.'
        ),
        'signal_strength': None,
    },
    {
        'edge_id': 'SEED-003',
        'instrument_class': 'INDEX_OPTIONS',
        'underlying': 'ALL_INDICES',
        'expiry_type': 'WEEKLY',
        'trigger_feature': 'rsi_2',
        'trigger_condition': 'rsi_2 < 10 for >= 2 consecutive days AND close > SMA_50 OR close > SMA_200 * 0.85',
        'direction': 'LONG',
        'holding_period': 5,
        'win_rate': 0.67,
        'avg_return_net': 0.25,
        'avg_return_gross': None,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'ALL',
        'vix_bucket': None,
        'iv_rank_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'feature_set_version': 'v1',
        'regime_model_version': 'v1',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': 'Connors & Alvarez (2008), Short Term Trading Strategies That Work',
        'regime_at_discovery': None,
        'theta_window_days': 5,
        'synthetic_pricing_used': 1,
        'trigger_frequency_baseline': None,
        'trigger_frequency_current': None,
        'frequency_decay_flag': 0,
        'win_rate_watch_flag': 0,
        'posterior_win_rate': None,
        'live_wins_in_regime': 0,
        'live_losses_in_regime': 0,
        'resurrection_priority': None,
        'resurrection_attempts': 0,
        'mechanism_story': (
            'RSI(2) below 10 indicates 2-3 day exhaustion selling without relief. '
            'The 2-period RSI is hypersensitive — it captures short-term exhaustion '
            'extremes that the standard 14-period RSI misses. Multi-day RSI(2) < 10 '
            'means sellers have dominated for consecutive days without any recovery. '
            'Statistical reversion to mean typically occurs within 2-5 sessions. '
            'Modified 200-MA filter (85% threshold) prevents buying structural collapses '
            'while remaining applicable in non-bull regimes.'
        ),
        'signal_strength': None,
    },
    {
        'edge_id': 'SEED-004',
        'instrument_class': 'INDEX_OPTIONS',
        'underlying': 'ALL_INDICES',
        'expiry_type': 'WEEKLY',
        'trigger_feature': 'mom_5d',
        'trigger_condition': 'mom_5d > 0 AND rsi_14 > 55 AND regime in H_BULL OR L_BULL AND vol_expanding == 0',
        'direction': 'LONG',
        'holding_period': 10,
        'win_rate': 0.56,
        'avg_return_net': 0.32,
        'avg_return_gross': None,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'H_BULL',
        'vix_bucket': None,
        'iv_rank_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'feature_set_version': 'v1',
        'regime_model_version': 'v1',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': 'Moskowitz, Ooi, Pedersen (2012), Time Series Momentum; Jegadeesh & Titman (1993)',
        'regime_at_discovery': None,
        'theta_window_days': 10,
        'synthetic_pricing_used': 1,
        'trigger_frequency_baseline': None,
        'trigger_frequency_current': None,
        'frequency_decay_flag': 0,
        'win_rate_watch_flag': 0,
        'posterior_win_rate': None,
        'live_wins_in_regime': 0,
        'live_losses_in_regime': 0,
        'resurrection_priority': None,
        'resurrection_attempts': 0,
        'mechanism_story': (
            'In confirmed bull regimes, indices exhibit short-term momentum continuation '
            'due to behavioral underreaction and institutional inflow persistence. '
            'Positive 5-day return combined with RSI(14) above 55 signals an index '
            'in an orderly uptrend with momentum intact. Non-expanding volatility '
            'confirms the move is controlled, not a volatile spike. ATM CALL with '
            '10-day hold captures a trend continuation move before mean reversion '
            'pressure reasserts.'
        ),
        'signal_strength': None,
    },
    {
        'edge_id': 'SEED-005',
        'instrument_class': 'SYNTHETIC_PAIR',
        'underlying': 'XAUUSD_SYNTHETIC',
        'expiry_type': 'MONTHLY',
        'trigger_feature': 'z_21d',
        'trigger_condition': 'synthetic_xauusd_z_21d < -2.0 → long Gold + short USDINR; > 2.0 → short Gold + long USDINR',
        'direction': 'BOTH',
        'holding_period': 5,
        'win_rate': 0.56,
        'avg_return_net': 0.018,
        'avg_return_gross': None,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'SIDEWAYS',
        'vix_bucket': None,
        'iv_rank_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'feature_set_version': 'v1',
        'regime_model_version': 'v1',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': 'Erb & Harvey (2006), Strategic and Tactical Value of Commodity Futures',
        'regime_at_discovery': None,
        'theta_window_days': 5,
        'synthetic_pricing_used': 0,
        'trigger_frequency_baseline': None,
        'trigger_frequency_current': None,
        'frequency_decay_flag': 0,
        'win_rate_watch_flag': 0,
        'posterior_win_rate': None,
        'live_wins_in_regime': 0,
        'live_losses_in_regime': 0,
        'resurrection_priority': None,
        'resurrection_attempts': 0,
        'mechanism_story': (
            'Gold-USD price relationship exhibits mean reversion at short horizons due '
            'to macro flow rebalancing. Both gold and USD function as risk-off proxies, '
            'creating temporary dislocations. The synthetic XAU/USD pair is constructed '
            'as MCX Gold futures (long leg, gold in INR) + NSE USD/INR futures '
            '(short leg, removes INR component). The synthetic pair price z-score '
            'below -2.0 means gold is cheap in USD terms relative to recent history. '
            'Two-leg construction on Zerodha: both legs within same account. '
            'Rollback mechanism if one leg fails to fill.'
        ),
        'signal_strength': None,
    },
    {
        'edge_id': 'SEED-006',
        'instrument_class': 'CURRENCY_FUTURES',
        'underlying': 'EURUSD',
        'expiry_type': 'MONTHLY',
        'trigger_feature': 'z_21d',
        'trigger_condition': 'z_21d < -1.8 → LONG futures; z_21d > 1.8 → SHORT futures; no CB event in 3 days',
        'direction': 'BOTH',
        'holding_period': 10,
        'win_rate': 0.57,
        'avg_return_net': 0.010,
        'avg_return_gross': None,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'SIDEWAYS',
        'vix_bucket': None,
        'iv_rank_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'feature_set_version': 'v1',
        'regime_model_version': 'v1',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': 'Menkhoff, Sarno, Schmeling, Schrimpf (2012), Carry Trades and Global FX Volatility; Burnside et al. (2011)',
        'regime_at_discovery': None,
        'theta_window_days': 10,
        'synthetic_pricing_used': 0,
        'trigger_frequency_baseline': None,
        'trigger_frequency_current': None,
        'frequency_decay_flag': 0,
        'win_rate_watch_flag': 0,
        'posterior_win_rate': None,
        'live_wins_in_regime': 0,
        'live_losses_in_regime': 0,
        'resurrection_priority': None,
        'resurrection_attempts': 0,
        'mechanism_story': (
            'Currency pairs exhibit mean reversion at 5-15 day horizons in absence '
            'of macro events. Driven by central bank intervention thresholds and '
            'carry-trade rebalancing. G10 currency pairs (EUR/USD, GBP/USD, USD/JPY) '
            'are particularly robust because these are the pairs with the most active '
            'institutional flow management. z_21d threshold of 1.8 is lower than the '
            '2.0 used for equity indices — currency pairs are tighter-ranging and '
            'threshold must be relaxed slightly to capture enough signals. '
            'Central bank event filter is critical: do not trade in 3-day window '
            'around FOMC, ECB, BOJ, or BOE meetings.'
        ),
        'signal_strength': None,
    },
]

# Columns in edge_library that are NOT in SEEDED_EDGES (set by other steps)
_SKIP_COLS = {'created_at'}


def populate(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Get actual column order from schema
    cols_info = cur.execute('PRAGMA table_info(edge_library)').fetchall()
    all_cols  = [r[1] for r in cols_info]

    now = datetime.utcnow().isoformat()
    inserted = 0
    skipped  = 0

    for edge in SEEDED_EDGES:
        values = []
        for col in all_cols:
            if col == 'created_at':
                values.append(now)
            else:
                values.append(edge.get(col))

        placeholders = ','.join(['?'] * len(all_cols))
        col_str      = ','.join(all_cols)
        try:
            cur.execute(
                f'INSERT OR IGNORE INTO edge_library ({col_str}) VALUES ({placeholders})',
                values,
            )
            if cur.rowcount:
                inserted += 1
            else:
                skipped += 1
        except sqlite3.Error as e:
            print(f"  [ERROR] {edge['edge_id']}: {e}")

    conn.commit()
    conn.close()
    return inserted, skipped


def _verification_check():
    print("=== Step 6 — Seeded Edge Population verification ===\n")

    inserted, skipped = populate()
    print(f"  Inserted: {inserted}  Already present (skipped): {skipped}")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT edge_id, status, win_rate, holding_period FROM edge_library "
        "WHERE edge_provenance='SEEDED' ORDER BY edge_id"
    ).fetchall()
    conn.close()

    print(f"\n  {'edge_id':<12} {'status':<25} {'win_rate':>9} {'hold':>5}")
    print('  ' + '-' * 58)
    for r in rows:
        print(f"  {r[0]:<12} {r[1]:<25} {r[2]:>9.2f} {r[3]:>5}")

    pending = [r for r in rows if r[1] == 'PENDING_REVALIDATION']
    if len(pending) == 6:
        print(f"\n  PASS — 6 edges with status=PENDING_REVALIDATION.")
    else:
        print(f"\n  FAIL — expected 6 PENDING_REVALIDATION rows, got {len(pending)}.")

    print("\n=== Step 6 complete ===")


if __name__ == '__main__':
    _verification_check()
