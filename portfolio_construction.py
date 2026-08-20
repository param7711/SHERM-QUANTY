"""
Step 10 — Portfolio Construction.
Takes approved signals, returns ranked subset to trade.
Applies currency-exposure limits and a correlation filter.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import (
    PAIR_LEGS, MAX_PER_CURRENCY_EXPOSURE, ALLOW_OPPOSING_POSITIONS,
    FX_PAIRS,
)
from slot_system import SlotSystem

# Two signals whose pairs correlate above this are treated as one bet.
CORRELATION_THRESHOLD = 0.50

_CORR_CACHE = None


def _exposure_sign(pair: str, direction: str, currency: str) -> int:
    """+1 if long that currency, -1 if short. Mirrors risk_governor."""
    base, quote = PAIR_LEGS[pair]
    leg_sign = 1 if currency == base else -1
    dir_sign = 1 if direction == 'LONG' else -1
    return leg_sign * dir_sign


def load_correlation_matrix(processed_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Rolling return correlation between traded symbols, computed from the
    feature parquets. Cached — correlations move slowly relative to a
    2-15 day horizon.

    Returns an empty frame if the parquets are absent; callers treat a
    missing entry as uncorrelated rather than guessing a value.
    """
    global _CORR_CACHE
    if _CORR_CACHE is not None:
        return _CORR_CACHE

    returns = {}
    for pair in FX_PAIRS:
        path = os.path.join(processed_dir, f'{pair}_features.parquet')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path, columns=['ret_1d'])
            returns[pair] = df['ret_1d']
        except Exception:
            continue

    if len(returns) < 2:
        _CORR_CACHE = pd.DataFrame()
        return _CORR_CACHE

    _CORR_CACHE = pd.DataFrame(returns).corr()
    return _CORR_CACHE


def signed_correlation(sig_a: dict, sig_b: dict, corr: pd.DataFrame) -> float:
    """
    Correlation between two signals as bets, not as price series.

    Two positively-correlated pairs held in opposite directions are a
    spread, not a doubled bet, so the sign of the correlation flips with
    the relative direction.
    """
    pa, pb = sig_a['pair'], sig_b['pair']
    if corr.empty or pa not in corr.index or pb not in corr.columns:
        return 0.0
    rho = float(corr.loc[pa, pb])
    same_direction = sig_a['direction'] == sig_b['direction']
    return rho if same_direction else -rho


class PortfolioConstruction:
    """
    Takes list of approved signals, returns subset to trade.
    Applies currency-exposure limits and a correlation filter.
    Ranks by adjusted_score.
    """

    def select_trades(self, approved_signals: list, slot_system: SlotSystem) -> list:
        eligible = [s for s in approved_signals
                    if slot_system.can_add(s['holding_period'])]

        eligible = self._remove_duplicates(eligible)

        # Rank before filtering so the survivors of each cap are the
        # highest-scored candidates, not whichever happened to be first.
        scored = [(s, self._compute_adjusted_score(s)) for s in eligible]
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [s for s, _ in scored]

        ranked = self._apply_correlation_filter(ranked)
        ranked = self._apply_currency_limits(ranked)

        # Claim the slot as each trade is selected. Without this the slot
        # count never advances and every signal passes the capacity check.
        selected = []
        for signal in ranked:
            hp = signal['holding_period']
            if slot_system.can_add(hp):
                slot_id = signal.get('signal_id') or (
                    f"{signal.get('edge_id')}_{signal['pair']}_{signal['direction']}_{hp}")
                slot_system.add_slot(slot_id, {
                    'holding_period': hp,
                    'pair':           signal['pair'],
                    'direction':      signal['direction'],
                    'edge_id':        signal.get('edge_id'),
                })
                selected.append(signal)

        return selected

    def _apply_currency_limits(self, signals: list) -> list:
        """
        Cap same-direction exposure per currency. Offsetting exposure does
        not count — long EURUSD and short GBPUSD both touch USD but lean
        opposite ways.
        """
        counts = {}
        result = []
        for s in signals:
            base, quote = PAIR_LEGS[s['pair']]
            keys = [(c, _exposure_sign(s['pair'], s['direction'], c))
                    for c in (base, quote)]
            if any(counts.get(k, 0) >= MAX_PER_CURRENCY_EXPOSURE for k in keys):
                continue
            result.append(s)
            for k in keys:
                counts[k] = counts.get(k, 0) + 1
        return result

    def _apply_correlation_filter(self, signals: list) -> list:
        """
        Drop a signal if it is effectively the same bet as one already kept.
        EURUSD and GBPUSD run ~0.8 correlated, so holding both long is one
        short-USD bet in two tickets.
        """
        corr = load_correlation_matrix()
        if corr.empty:
            return signals

        kept = []
        for s in signals:
            if any(signed_correlation(s, k, corr) > CORRELATION_THRESHOLD
                   for k in kept):
                continue
            kept.append(s)
        return kept

    def _remove_duplicates(self, signals: list) -> list:
        """
        One signal per (pair, direction, holding_period, edge). Different
        edges on the same pair survive under hedging mode — they are
        separate bets with separate attribution.
        """
        seen = set()
        result = []
        for s in signals:
            if ALLOW_OPPOSING_POSITIONS:
                key = (s['pair'], s['direction'], s.get('holding_period'),
                       s.get('edge_id'))
            else:
                key = (s['pair'], s['direction'])
            if key not in seen:
                result.append(s)
                seen.add(key)
        return result

    def _compute_adjusted_score(self, signal: dict) -> float:
        """Higher score = higher priority."""
        base_score   = signal.get('edge_win_rate', 0.5)
        regime_bonus = 0.05 if signal.get('hmm_confidence_at_signal', 0) > 0.70 else 0.0
        strength     = min(abs(signal.get('trigger_value', 0)) / 10.0, 0.05)
        return base_score + regime_bonus + strength


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _make_signal(pair, direction, holding_period=5, edge_id='SEED-001',
                 win_rate=0.60):
    return {
        'pair':            pair,
        'direction':       direction,
        'holding_period':  holding_period,
        'edge_id':         edge_id,
        'edge_win_rate':   win_rate,
        'trigger_value':   -2.0,
    }


def _verification_check():
    print("=== Step 10 — Portfolio Construction verification (5 tests) ===\n")
    passed = 0
    pc = PortfolioConstruction()

    # Test 1: slots are actually consumed. Three 2-day signals against a
    # 3-slot bucket should all fit; a fourth must not.
    slots = SlotSystem()
    sigs = [_make_signal('EURUSD', 'LONG', 2, 'SEED-001'),
            _make_signal('USDJPY', 'LONG', 2, 'SEED-002'),
            _make_signal('XAUUSD', 'LONG', 2, 'SEED-003'),
            _make_signal('USDCAD', 'SHORT', 2, 'SEED-004')]
    selected = pc.select_trades(sigs, slots)
    result = len(selected) <= 3
    print(f"  Test 1 — Slot capacity enforced:          "
          f"{'PASS' if result else 'FAIL'} (selected {len(selected)}, max 3)")
    passed += result

    # Test 2: slot_system state actually advanced (the add_slot bug)
    slots = SlotSystem()
    before = slots.available_slots(5)
    pc.select_trades([_make_signal('EURUSD', 'LONG', 5)], slots)
    after = slots.available_slots(5)
    result = after == before - 1
    print(f"  Test 2 — Slot count decrements:           "
          f"{'PASS' if result else 'FAIL'} ({before} -> {after})")
    passed += result

    # Test 3: currency exposure cap. Three long-USD-quote pairs are all
    # short USD; only two may pass.
    slots = SlotSystem()
    sigs = [_make_signal('EURUSD', 'LONG', 5, 'SEED-001'),
            _make_signal('GBPUSD', 'LONG', 10, 'SEED-002'),
            _make_signal('AUDUSD', 'LONG', 15, 'SEED-003')]
    kept = pc._apply_currency_limits(sigs)
    result = len(kept) == MAX_PER_CURRENCY_EXPOSURE
    print(f"  Test 3 — Currency exposure cap:           "
          f"{'PASS' if result else 'FAIL'} (kept {len(kept)})")
    passed += result

    # Test 4: opposing directions on one pair both survive dedup
    sigs = [_make_signal('EURUSD', 'LONG', 10, 'SEED-004'),
            _make_signal('EURUSD', 'SHORT', 2, 'SEED-001')]
    kept = pc._remove_duplicates(sigs)
    result = len(kept) == 2
    print(f"  Test 4 — Opposing edges survive dedup:    "
          f"{'PASS' if result else 'FAIL'} (kept {len(kept)})")
    passed += result

    # Test 5: signed correlation flips with relative direction
    corr = pd.DataFrame({'EURUSD': [1.0, 0.8], 'GBPUSD': [0.8, 1.0]},
                        index=['EURUSD', 'GBPUSD'])
    same = signed_correlation(_make_signal('EURUSD', 'LONG'),
                              _make_signal('GBPUSD', 'LONG'), corr)
    opp  = signed_correlation(_make_signal('EURUSD', 'LONG'),
                              _make_signal('GBPUSD', 'SHORT'), corr)
    result = same > 0.5 and opp < -0.5
    print(f"  Test 5 — Signed correlation direction:    "
          f"{'PASS' if result else 'FAIL'} (same={same:+.2f}, opposed={opp:+.2f})")
    passed += result

    print(f"\n  {passed}/5 tests passed.")
    print("  PASS — all 5 unit tests passed." if passed == 5
          else "  FAIL — some unit tests failed. See above.")
    print("\n=== Step 10 complete ===")


if __name__ == '__main__':
    _verification_check()
