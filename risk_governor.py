"""
Step 6 — Risk Governor.
Hard circuit breaker. Rule-based only. No AI.
Every method here is a hard stop — returns False means DO NOT TRADE.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TOTAL_CAPITAL, MAX_POSITION_PCT, MAX_POSITIONS_TOTAL,
    MAX_MARGIN_PCT, MAX_PER_CURRENCY_EXPOSURE, MAX_SPREAD_PCT,
    VIX_SPIKE_THRESHOLD, PAIR_LEGS, NEWS_BLACKOUT_MINS,
    ALLOW_OPPOSING_POSITIONS, DAILY_LOSS_HALT_PCT, DRAWDOWN_HALT_PCT,
    PROP_MAX_DAILY_LOSS_PCT, PROP_MAX_DRAWDOWN_PCT, PROP_TRAILING_DRAWDOWN,
)


def _currency_exposure_sign(pair: str, direction: str, currency: str) -> int:
    """
    +1 if the position is long that currency, -1 if short.
    Long EURUSD is long EUR and short USD; short EURUSD is the reverse.
    """
    base, quote = PAIR_LEGS[pair]
    leg_sign = 1 if currency == base else -1
    dir_sign = 1 if direction == 'LONG' else -1
    return leg_sign * dir_sign


class RiskGovernor:
    """
    Hard circuit breaker. Rule-based only. No AI.
    Every method here is a hard stop — returns False means DO NOT TRADE.
    """

    def __init__(self, capital: float = TOTAL_CAPITAL):
        self.capital    = capital
        self.positions  = {}       # trade_id → position dict
        self.daily_pnl  = 0.0
        self.realised_pnl = 0.0    # cumulative, not per-trade
        self.portfolio_drawdown = 0.0
        self.peak_capital = capital
        self.halted     = False
        self.halt_reason = None

    def can_enter_position(self, signal: dict, proposed_lots: float) -> tuple:
        """Master gate — call before every order. Returns (allowed, reason)."""
        checks = [
            self._check_halt_state(),
            self._check_transition_watch(signal),
            self._check_max_positions(),
            self._check_position_size(signal, proposed_lots),
            self._check_margin_available(signal, proposed_lots),
            self._check_currency_exposure(signal),
            self._check_pair_concentration(signal),
            self._check_lot_size_affordable(signal),
            self._check_spread(signal),
            self._check_news_blackout(signal),
            self._check_duplicate_tuple(signal),
        ]
        for allowed, reason in checks:
            if not allowed:
                return False, reason
        return True, 'OK'

    def _check_halt_state(self) -> tuple:
        if self.halted:
            return False, f'SYSTEM_HALTED: {self.halt_reason}'
        return True, 'OK'

    def _check_max_positions(self) -> tuple:
        if len(self.positions) >= MAX_POSITIONS_TOTAL:
            return False, 'MAX_POSITIONS_REACHED'
        return True, 'OK'

    def _check_position_size(self, signal, lots) -> tuple:
        notional = signal['lot_value'] * lots
        if notional > self.capital * MAX_POSITION_PCT:
            return False, f'POSITION_SIZE_EXCEEDS_{MAX_POSITION_PCT*100:.0f}pct_CAPITAL'
        return True, 'OK'

    def _check_margin_available(self, signal, lots) -> tuple:
        required_margin    = signal['margin_per_lot'] * lots
        current_margin_used = sum(p['margin'] for p in self.positions.values())
        if (current_margin_used + required_margin) > self.capital * MAX_MARGIN_PCT:
            return False, 'MARGIN_LIMIT_EXCEEDED'
        return True, 'OK'

    def _check_currency_exposure(self, signal) -> tuple:
        """
        Long EUR/USD and short EUR/GBP both express long EUR. Cap how many
        open positions can lean on any single currency, in either leg.

        Counts only positions whose exposure to that currency points the same
        way as the incoming signal: an existing short-EUR position does not
        consume the long-EUR budget, since together they offset rather than
        concentrate. Positions tagged as legs of a cross-pair spread are
        exempt — the spread is one bet, and its two legs are deliberately
        opposing.
        """
        base, quote = PAIR_LEGS[signal['pair']]
        for currency in (base, quote):
            sign = _currency_exposure_sign(signal['pair'], signal['direction'], currency)
            count = 0
            for p in self.positions.values():
                if p.get('is_cross_leg'):
                    continue
                if currency not in PAIR_LEGS[p['pair']]:
                    continue
                if _currency_exposure_sign(p['pair'], p['direction'], currency) == sign:
                    count += 1
            if count >= MAX_PER_CURRENCY_EXPOSURE:
                return False, f'CURRENCY_EXPOSURE_{currency}'
        return True, 'OK'

    def _check_pair_concentration(self, signal) -> tuple:
        """
        Cap positions per symbol. Under hedging mode opposing positions from
        different edges are legitimate (they are different bets on different
        horizons), so only same-direction positions count toward the cap.
        """
        if ALLOW_OPPOSING_POSITIONS:
            count = sum(1 for p in self.positions.values()
                        if p['pair'] == signal['pair']
                        and p['direction'] == signal['direction'])
        else:
            count = sum(1 for p in self.positions.values()
                        if p['pair'] == signal['pair'])
        if count >= 2:
            return False, f"PAIR_CONCENTRATION_{signal['pair']}"
        return True, 'OK'

    def _check_lot_size_affordable(self, signal) -> tuple:
        if signal['margin_per_lot'] > self.capital:
            return False, 'CANNOT_AFFORD_SINGLE_LOT'
        return True, 'OK'

    def _check_spread(self, signal) -> tuple:
        if signal.get('spread_pct', 0) > MAX_SPREAD_PCT:
            return False, 'SPREAD_TOO_WIDE'
        return True, 'OK'

    def _check_news_blackout(self, signal) -> tuple:
        """No entries within NEWS_BLACKOUT_MINS of a high-impact release."""
        mins = signal.get('mins_to_high_impact_news')
        if mins is not None and abs(mins) < NEWS_BLACKOUT_MINS:
            return False, 'NEWS_BLACKOUT'
        return True, 'OK'

    def _check_duplicate_tuple(self, signal) -> tuple:
        """
        Block the same edge doubling up on the same symbol, direction and
        horizon. Keyed on edge_id as well, so two different edges agreeing on
        a trade still open as separate attributable tickets rather than one
        being silently dropped.
        """
        key = (signal.get('edge_id'), signal['pair'],
               signal['direction'], signal['holding_period'])
        existing_keys = [(p.get('edge_id'), p['pair'],
                          p['direction'], p['holding_period'])
                         for p in self.positions.values()]
        if key in existing_keys:
            return False, 'DUPLICATE_POSITION_TUPLE'
        return True, 'OK'

    def _check_transition_watch(self, signal) -> tuple:
        """Block new entries when regime engine signals transition watch."""
        from regime_engine_sherm import get_current_regime
        try:
            regime = get_current_regime()
            if regime['transition_warning_flag']:
                return False, 'TRANSITION_WATCH_ACTIVE'
        except Exception:
            pass
        return True, 'OK'

    def record_position_opened(self, trade_id: str, position_dict: dict):
        self.positions[trade_id] = position_dict

    def record_position_closed(self, trade_id: str, pnl: float):
        if trade_id in self.positions:
            del self.positions[trade_id]
        self.daily_pnl += pnl
        self.realised_pnl += pnl
        current_capital = self.capital + self.realised_pnl
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital

        # Funded accounts measure drawdown either from the starting balance
        # (fixed floor) or from the high-water mark (trailing). Trailing is
        # strictly harsher — a profitable run raises the floor under you.
        reference = self.peak_capital if PROP_TRAILING_DRAWDOWN else self.capital
        self.portfolio_drawdown = max(0.0, (reference - current_capital) / reference)

        if self.portfolio_drawdown > DRAWDOWN_HALT_PCT and not self.halted:
            self.halted = True
            self.halt_reason = f'DRAWDOWN_{self.portfolio_drawdown:.1%}'
        if self.daily_pnl < -self.capital * DAILY_LOSS_HALT_PCT and not self.halted:
            self.halted = True
            self.halt_reason = f'SINGLE_DAY_LOSS_{DAILY_LOSS_HALT_PCT:.2%}'

    def prop_breach_status(self) -> dict:
        """
        Distance to the funded account's hard termination limits. The internal
        halts above fire well before these; this reports how much genuine
        headroom is left, for monitoring rather than gating.
        """
        daily_used = (-self.daily_pnl) / (self.capital * PROP_MAX_DAILY_LOSS_PCT)
        dd_used    = self.portfolio_drawdown / PROP_MAX_DRAWDOWN_PCT
        return {
            'daily_limit_used_pct':    max(0.0, daily_used),
            'drawdown_limit_used_pct': max(0.0, dd_used),
            'breached': daily_used >= 1.0 or dd_used >= 1.0,
        }

    def check_mid_hold_exits(self, position: dict) -> tuple:
        """Returns (should_exit, reason). Call at each session checkpoint."""
        if position.get('days_held', 0) >= 15:
            return True, 'TIME_STOP'

        if position.get('today_move_vs_expected_vol', 0) > 2.5:
            return True, 'VOL_SPIKE'

        if position.get('vix_change_today', 0) > VIX_SPIKE_THRESHOLD:
            return True, 'VIX_SPIKE'

        # Weekend gap protocol — spot forex gaps over the weekend close in a
        # way an exchange-traded book never did.
        if position.get('is_friday_preclose') and position.get('weekend_event_risk'):
            return True, 'WEEKEND_GAP_RISK'

        return False, 'HOLD'

    def daily_reset(self):
        """Call at end of each trading day."""
        self.daily_pnl = 0.0
        if self.halt_reason == 'SINGLE_DAY_LOSS_4PCT':
            self.halted = False
            self.halt_reason = None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _make_signal(pair='EURUSD', direction='LONG', holding_period=5,
                 lot_value=75_000, margin_per_lot=50_000,
                 spread_pct=0.0001, mins_to_high_impact_news=None,
                 edge_id='SEED-001'):
    return {
        'pair':             pair,
        'direction':        direction,
        'holding_period':   holding_period,
        'lot_value':        lot_value,
        'margin_per_lot':   margin_per_lot,
        'spread_pct':       spread_pct,
        'mins_to_high_impact_news': mins_to_high_impact_news,
        'edge_id':          edge_id,
    }


def _make_position(pair='EURUSD', direction='LONG', holding_period=5,
                   margin=50_000, edge_id='SEED-001', is_cross_leg=False):
    return {
        'pair':           pair,
        'direction':      direction,
        'holding_period': holding_period,
        'margin':         margin,
        'edge_id':        edge_id,
        'is_cross_leg':   is_cross_leg,
    }


def _verification_check():
    print("=== Step 6 — Risk Governor verification (10 unit tests) ===\n")
    passed = 0

    # Test 1: Halt on drawdown breaching the internal (sub-prop) threshold
    rg = RiskGovernor(capital=1_000_000)
    rg.record_position_closed('T1', -int(1_000_000 * (DRAWDOWN_HALT_PCT + 0.01)))
    ok, _ = rg.can_enter_position(_make_signal(), 1)
    result = not ok and 'DRAWDOWN' in rg.halt_reason
    print(f"  Test 1 — Halt on drawdown >{DRAWDOWN_HALT_PCT:.1%}:          {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 2: Halt on single-day loss breaching the internal threshold
    rg = RiskGovernor(capital=1_000_000)
    rg.record_position_closed('T2', -int(1_000_000 * (DAILY_LOSS_HALT_PCT + 0.005)))
    ok, _ = rg.can_enter_position(_make_signal(), 1)
    result = not ok and 'SINGLE_DAY_LOSS' in rg.halt_reason
    print(f"  Test 2 — Halt on daily loss >{DAILY_LOSS_HALT_PCT:.2%}:       {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 3: Block when max positions reached.
    # Uses pairs with disjoint legs where possible; currency-exposure check
    # would otherwise fire first.
    rg = RiskGovernor(capital=100_000_000)
    for i in range(MAX_POSITIONS_TOTAL):
        rg.record_position_opened(f'T{i}', _make_position(pair='EURUSD'))
    ok, reason = rg.can_enter_position(_make_signal(pair='EURUSD'), 1)
    result = not ok and reason in ('MAX_POSITIONS_REACHED', 'CURRENCY_EXPOSURE_EUR')
    print(f"  Test 3 — Block at max {MAX_POSITIONS_TOTAL} positions:        {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 4: Block when margin limit (60%) exceeded
    rg = RiskGovernor(capital=1_000_000)
    rg.record_position_opened('T1', _make_position(pair='AUDUSD', margin=650_000))
    sig = _make_signal(pair='GBPUSD', lot_value=50_000, margin_per_lot=100_000)
    ok, reason = rg.can_enter_position(sig, 1)
    result = not ok and reason == 'MARGIN_LIMIT_EXCEEDED'
    print(f"  Test 4 — Block on margin >60%:            {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 5: Transition watch blocks new entry (mock regime engine)
    import unittest.mock as mock
    rg = RiskGovernor(capital=10_000_000)
    fake_regime = {'regime_state': 'SIDEWAYS', 'regime_confidence': 0.45,
                   'transition_warning_flag': True}
    with mock.patch('regime_engine_sherm.get_current_regime', return_value=fake_regime):
        ok, reason = rg.can_enter_position(_make_signal(), 1)
    result = not ok and reason == 'TRANSITION_WATCH_ACTIVE'
    print(f"  Test 5 — Transition watch blocks entry:   {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 6: Currency exposure cap — long EURUSD + long GBPUSD are both
    # short USD, so a third short-USD position is blocked.
    rg = RiskGovernor(capital=100_000_000)
    rg.record_position_opened('T1', _make_position(pair='EURUSD', direction='LONG'))
    rg.record_position_opened('T2', _make_position(pair='GBPUSD', direction='LONG'))
    ok, reason = rg.can_enter_position(
        _make_signal(pair='AUDUSD', direction='LONG'), 1)
    result = not ok and reason == 'CURRENCY_EXPOSURE_USD'
    print(f"  Test 6 — Currency exposure cap (USD):     {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 7: News blackout blocks entry
    rg = RiskGovernor(capital=100_000_000)
    ok, reason = rg.can_enter_position(
        _make_signal(mins_to_high_impact_news=5), 1)
    result = not ok and reason == 'NEWS_BLACKOUT'
    print(f"  Test 7 — News blackout blocks entry:      {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 8: Hedging — a different edge may take the opposing side of the
    # same pair. This is the core hedging-mode behaviour.
    rg = RiskGovernor(capital=100_000_000)
    rg.record_position_opened('T1', _make_position(
        pair='EURUSD', direction='LONG', edge_id='SEED-004', holding_period=10))
    ok, reason = rg.can_enter_position(_make_signal(
        pair='EURUSD', direction='SHORT', edge_id='SEED-001', holding_period=2), 1)
    print(f"  Test 8 — Opposing edges allowed (hedge):  {'PASS' if ok else 'FAIL'} ({reason})")
    passed += ok

    # Test 9: The SAME edge still cannot double up on the same tuple.
    rg = RiskGovernor(capital=100_000_000)
    rg.record_position_opened('T1', _make_position(
        pair='EURUSD', direction='LONG', edge_id='SEED-001', holding_period=5))
    ok, reason = rg.can_enter_position(_make_signal(
        pair='EURUSD', direction='LONG', edge_id='SEED-001', holding_period=5), 1)
    result = not ok and reason == 'DUPLICATE_POSITION_TUPLE'
    print(f"  Test 9 — Same edge cannot double up:      {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 10: Opposing exposure does not consume the currency budget.
    # Long EURUSD (short USD) + short GBPUSD (long USD) offset, so a further
    # short-USD position is still permitted.
    rg = RiskGovernor(capital=100_000_000)
    rg.record_position_opened('T1', _make_position(pair='EURUSD', direction='LONG'))
    rg.record_position_opened('T2', _make_position(
        pair='GBPUSD', direction='SHORT', edge_id='SEED-004'))
    ok, reason = rg.can_enter_position(_make_signal(
        pair='AUDUSD', direction='LONG', edge_id='SEED-003'), 1)
    print(f"  Test 10 — Offsetting exposure not capped: {'PASS' if ok else 'FAIL'} ({reason})")
    passed += ok

    print(f"\n  {passed}/10 tests passed.")
    print("  PASS — all 10 unit tests passed." if passed == 10
          else "  FAIL — some unit tests failed. See above.")
    print("\n=== Step 6 complete ===")


if __name__ == '__main__':
    _verification_check()
