"""
Step 5 — Seeded edge population.
Writes all 7 seeded edges to the Edge Library with research-adapted priors.
These are starting priors, adapted from the v4.1 derivatives-universe
figures rather than sourced fresh for forex — every edge carries
needs_revalidation=1 so Step 5.5 must overwrite them with empirical
priors from real historical data before paper trading begins.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from datetime import datetime

from config import DB_PATH
from database.init_db import ensure_schema
from feature_definitions import carry_direction

# Sentinel meaning "applies across the 7 non-gold majors" — expanded by
# revalidate_edges.py and, in Phase 6, by the AI 1B spotter. Gold is
# excluded from this aggregate deliberately: its volatility regime differs
# enough from the majors that pooling them would misstate both.
ALL_MAJORS = 'ALL_MAJORS'


SEEDED_EDGES = [
    {
        'edge_id': 'SEED-001',
        'pair': ALL_MAJORS,
        'pair_type': 'MAJOR',
        'trigger_feature': 'z_21d',
        'trigger_condition': 'z_21d < -2.0 -> LONG; z_21d > 2.0 -> SHORT',
        'direction': 'BOTH',
        'holding_period': 5,
        'win_rate': 0.58,
        'avg_return_net': 0.0021,     # thin by design: unlevered spot, not premium
        'avg_return_gross': None,
        'avg_return_pips': 23,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'SIDEWAYS',
        'vix_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': (
            'Avellaneda & Lee (2010), Statistical Arbitrage in the U.S. Equities '
            'Market; Lo & MacKinlay (1990). Adapted from equity-index options to '
            'direct spot/CFD FX positions.'
        ),
        'regime_at_discovery': None,
        'carry_direction_at_discovery': None,
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
            'Major currency pairs exhibit short-term mean reversion in sideways and '
            'low-conviction regimes, driven by positioning unwinds and overnight '
            'order-flow rebalancing rather than the institutional-rebalancing '
            'mechanism that drove the equity-index version of this edge. A 21-day '
            'z-score below -2 is roughly two standard deviations from the recent '
            'mean, a level that historically precedes reversion within 2-7 sessions. '
            'Direct spot/CFD position, long or short — no options, no strike, no '
            'expiry.'
        ),
        'signal_strength': None,
        'needs_revalidation': 1,
    },
    {
        'edge_id': 'SEED-002',
        'pair': ALL_MAJORS,
        'pair_type': 'MAJOR',
        'trigger_feature': 'gap_unfilled_eod',
        'trigger_condition': 'gap_pct < -0.006 AND close < prev_close -> LONG at next session',
        'direction': 'LONG',
        'holding_period': 2,
        'win_rate': 0.60,
        'avg_return_net': 0.0018,
        'avg_return_gross': None,
        'avg_return_pips': 18,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'SIDEWAYS',
        'vix_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': (
            'Cooper, Gutierrez, Hameed (2004), Market States and Momentum; '
            'Caginalp & Constantine (1995). Adapted from equity-index overnight '
            'gaps to the FX weekly-open gap.'
        ),
        'regime_at_discovery': None,
        'carry_direction_at_discovery': None,
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
            'Spot FX barely gaps within the trading week — this edge is really '
            'about the Sunday 22:00 UTC reopen after the weekend news accumulates '
            'against a closed market. A reopen gap down of more than 0.6% that has '
            'not filled by the first session close tends to fill within 1-2 '
            'sessions, driven by liquidity-provider repositioning. Because the '
            'triggering event is weekly rather than daily, this edge fires far '
            'less often than its index-options ancestor did. News filter: no '
            'major scheduled event (central bank decision, NFP) in the next 3 days.'
        ),
        'signal_strength': None,
        'needs_revalidation': 1,
    },
    {
        'edge_id': 'SEED-003',
        'pair': ALL_MAJORS,
        'pair_type': 'MAJOR',
        'trigger_feature': 'rsi_2',
        'trigger_condition': 'rsi_2 < 10 for >= 2 consecutive days AND (close > SMA_50 OR close > SMA_200 * 0.85) -> LONG',
        'direction': 'LONG',
        'holding_period': 5,
        'win_rate': 0.67,
        'avg_return_net': 0.0028,
        'avg_return_gross': None,
        'avg_return_pips': 30,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'ALL',
        'vix_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': (
            'Connors & Alvarez (2008), Short Term Trading Strategies That Work. '
            'Documented across many liquid instruments including FX; least '
            'changed of the seeded edges by the forex pivot.'
        ),
        'regime_at_discovery': None,
        'carry_direction_at_discovery': None,
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
            'RSI(2) below 10 for two or more consecutive days indicates short-term '
            'exhaustion selling without relief. The 2-period RSI is hypersensitive '
            'and captures extremes the standard 14-period RSI misses. Statistical '
            'reversion typically follows within 2-5 sessions. The relaxed 200-MA '
            'filter (85% threshold) screens out structural collapses while staying '
            'usable outside confirmed uptrends. Highest research win-rate prior of '
            'the five directional edges, which is why it also carries the tightest '
            'stop — the reversion, if it comes, comes fast.'
        ),
        'signal_strength': None,
        'needs_revalidation': 1,
    },
    {
        'edge_id': 'SEED-004',
        'pair': ALL_MAJORS,
        'pair_type': 'MAJOR',
        'trigger_feature': 'mom_5d',
        'trigger_condition': 'mom_5d > 0 AND rsi_14 > 55 AND regime in {H_BULL, L_BULL} AND vol_expanding == 0 -> LONG',
        'direction': 'LONG',
        'holding_period': 10,
        'win_rate': 0.56,
        'avg_return_net': 0.0032,
        'avg_return_gross': None,
        'avg_return_pips': 34,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'H_BULL',
        'vix_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': (
            'Moskowitz, Ooi, Pedersen (2012), Time Series Momentum; Jegadeesh & '
            'Titman (1993). The Moskowitz paper already covers currencies '
            'directly, so this edge needed the least re-grounding of the four '
            'directional edges.'
        ),
        'regime_at_discovery': None,
        'carry_direction_at_discovery': None,
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
            'In confirmed trending regimes, major pairs exhibit short-term '
            'momentum continuation from behavioral underreaction and persistent '
            'order flow. Positive 5-day return combined with RSI(14) above 55 '
            'signals an orderly uptrend with momentum intact; non-expanding '
            'volatility confirms the move is controlled rather than a volatile '
            'spike. Widest stop of the five directional edges by design — it '
            'needs room to run, and sizing shrinks automatically to hold risk '
            'constant.'
        ),
        'signal_strength': None,
        'needs_revalidation': 1,
    },
    {
        'edge_id': 'SEED-005',
        'pair': 'XAUUSD',
        'pair_type': 'MAJOR',
        'trigger_feature': 'z_21d',
        'trigger_condition': 'z_21d < -2.0 -> LONG XAUUSD; z_21d > 2.0 -> SHORT XAUUSD',
        'direction': 'BOTH',
        'holding_period': 5,
        'win_rate': 0.56,
        'avg_return_net': 0.0038,
        'avg_return_gross': None,
        'avg_return_pips': None,   # gold is quoted in dollars, not pips
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'SIDEWAYS',
        'vix_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': (
            'Erb & Harvey (2006), The Strategic and Tactical Value of Commodity '
            'Futures. v4.1 built this as a synthetic pair (long MCX Gold, short '
            'USD/INR) purely because Zerodha had no way to quote gold in dollars '
            '— that construction, its hedge-ratio drift monitor, and its leg-fill '
            'rollback logic are gone. MetaTrader quotes XAUUSD natively as one '
            'symbol, so this is now a single order ticket like any other pair.'
        ),
        'regime_at_discovery': None,
        'carry_direction_at_discovery': carry_direction('XAUUSD'),
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
            'Gold and the dollar both function as risk-off proxies, which creates '
            'temporary dislocations between them that mean-revert at short '
            'horizons as macro flows rebalance. A 21-day z-score on XAUUSD below '
            '-2.0 means gold is cheap in dollar terms relative to its recent '
            'range. This edge never fired in v4.1: the feature it read '
            '(synthetic_xauusd_z_21d) was never computed by anything, so the '
            'trigger silently evaluated to zero on every scan. Trading XAUUSD '
            'directly fixes that as a side effect of the platform change, not '
            'through any logic change to the edge itself.'
        ),
        'signal_strength': None,
        'needs_revalidation': 1,
    },
    {
        'edge_id': 'SEED-006',
        'pair': ALL_MAJORS,
        'pair_type': 'MAJOR',
        'trigger_feature': 'z_21d',
        'trigger_condition': 'z_21d < -1.8 -> LONG; z_21d > 1.8 -> SHORT; no CB event in 3 days',
        'direction': 'BOTH',
        'holding_period': 10,
        'win_rate': 0.57,
        'avg_return_net': 0.0016,
        'avg_return_gross': None,
        'avg_return_pips': 17,
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'SIDEWAYS',
        'vix_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': (
            'Menkhoff, Sarno, Schmeling, Schrimpf (2012), Carry Trades and Global '
            'Foreign Exchange Volatility; Burnside et al. (2011). Already '
            'forex-native in v4.1 — carried forward essentially unchanged, on '
            'MetaTrader instead of NSE cross-currency futures.'
        ),
        'regime_at_discovery': None,
        'carry_direction_at_discovery': None,
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
            'Currency pairs exhibit mean reversion at 5-15 day horizons absent '
            'macro events, driven by central bank intervention thresholds and '
            'carry-trade rebalancing. The z-score threshold (1.8, vs 2.0 for '
            'SEED-001) is tighter because currency pairs are lower-variance and '
            'range more tightly than the mixed instrument set SEED-001 draws '
            'from. Central bank event filter is critical: no trading within 3 '
            'days of an FOMC, ECB, BOJ, or BOE decision.'
        ),
        'signal_strength': None,
        'needs_revalidation': 1,
    },
    {
        'edge_id': 'SEED-007',
        # Two-leg spread edge. `pair` names the default leg pair for
        # reference; the actual leg universe is config.CROSS_PAIRS and the
        # spotter (Phase 6) iterates it.
        'pair': 'EURUSD/GBPUSD',
        'pair_type': 'CROSS',
        'trigger_feature': 'spread_z',
        'trigger_condition': (
            'beta-adjusted spread z-score < -2.0 -> LONG leg A / SHORT leg B; '
            '> 2.0 -> SHORT leg A / LONG leg B; exit at |z| < 0.5'
        ),
        'direction': 'SPREAD',
        'holding_period': 10,
        'win_rate': 0.58,
        'avg_return_net': 0.0012,
        'avg_return_gross': None,
        'avg_return_pips': None,   # spread return, not a single pair's pips
        'sample_size': None,
        'p_value': None,
        'p_value_corrected': None,
        'oos_win_rate': None,
        'regime': 'SIDEWAYS',
        'vix_bucket': None,
        'live_hit_rate': None,
        'decay_flag': 0,
        'decay_cause': None,
        'last_active': None,
        'valid_until': None,
        'status': 'PENDING_REVALIDATION',
        'capacity_ceiling_lots': None,
        'edge_provenance': 'SEEDED',
        'seed_source_citation': (
            'Avellaneda & Lee (2010), Statistical Arbitrage in the U.S. Equities '
            'Market — the same pairs-trading framework SEED-001 draws its mean-'
            'reversion mechanism from, applied here as an actual two-leg pairs '
            'trade rather than a single-series analogy.'
        ),
        'regime_at_discovery': None,
        'carry_direction_at_discovery': None,
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
            'Genuinely new in v5.0. Two correlated pairs sharing a currency leg '
            '(EUR/USD vs GBP/USD, both short USD; AUD/USD vs NZD/USD, both long '
            'the antipodean bloc) drift apart and mean-revert on their spread, '
            'independent of the direction either pair moves outright — this is '
            'why it survives the currency-exposure cap: the two legs offset '
            'rather than concentrate. Revalidation (Step 5.5) compares a floating '
            'rolling-beta hedge ratio against a fixed 1:1 ratio (equivalent to a '
            'direct MetaTrader cross, e.g. EURGBP) and defaults to fixed unless '
            'floating clearly wins on both win rate and net return — a short-'
            'window OLS beta between two majors is a textbook spurious '
            'regression, since both series are dominated by their own random-walk '
            'component and the beta estimate is mostly sampling noise on the '
            'same scale as the thing it is trying to isolate.'
        ),
        'signal_strength': None,
        'needs_revalidation': 1,
    },
]

# Columns in edge_library that are NOT in SEEDED_EDGES (set by other steps)
_SKIP_COLS = {'created_at'}


def populate(db_path: str = DB_PATH):
    ensure_schema(db_path)
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
    print("=== Step 5 — Seeded Edge Population verification ===\n")

    inserted, skipped = populate()
    print(f"  Inserted: {inserted}  Already present (skipped): {skipped}")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT edge_id, status, win_rate, holding_period, needs_revalidation "
        "FROM edge_library WHERE edge_provenance='SEEDED' ORDER BY edge_id"
    ).fetchall()
    conn.close()

    print(f"\n  {'edge_id':<12} {'status':<25} {'win_rate':>9} {'hold':>5} {'needs_reval':>12}")
    print('  ' + '-' * 72)
    for r in rows:
        print(f"  {r[0]:<12} {r[1]:<25} {r[2]:>9.2f} {r[3]:>5} {r[4]:>12}")

    pending = [r for r in rows if r[1] == 'PENDING_REVALIDATION']
    flagged = [r for r in rows if r[4] == 1]
    if len(pending) == 7 and len(flagged) == 7:
        print(f"\n  PASS — 7 edges PENDING_REVALIDATION, all flagged needs_revalidation=1.")
    else:
        print(f"\n  FAIL — expected 7/7, got {len(pending)} pending, {len(flagged)} flagged.")

    print("\n=== Step 5 complete ===")


if __name__ == '__main__':
    _verification_check()
