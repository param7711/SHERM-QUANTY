"""
Step 7 — Risk Governor.
Hard circuit breaker. Rule-based only. No AI.
Every method here is a hard stop — returns False means DO NOT TRADE.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TOTAL_CAPITAL, MAX_POSITION_PCT, MAX_POSITIONS_TOTAL,
    MAX_MARGIN_PCT, THETA_STOP_PCT, VIX_SPIKE_THRESHOLD,
)


class RiskGovernor:
    """
    Hard circuit breaker. Rule-based only. No AI.
    Every method here is a hard stop — returns False means DO NOT TRADE.
    """

    def __init__(self, capital: float = TOTAL_CAPITAL):
        self.capital    = capital
        self.positions  = {}       # trade_id → position dict
        self.daily_pnl  = 0.0
        self.portfolio_drawdown = 0.0
        self.peak_capital = capital
        self.halted     = False
        self.halt_reason = None

    def can_enter_position(self, signal: dict, proposed_lots: int) -> tuple:
        """Master gate — call before every order. Returns (allowed, reason)."""
        checks = [
            self._check_halt_state(),
            self._check_transition_watch(signal),
            self._check_max_positions(),
            self._check_position_size(signal, proposed_lots),
            self._check_margin_available(signal, proposed_lots),
            self._check_asset_class_concentration(signal),
            self._check_underlying_concentration(signal),
            self._check_lot_size_affordable(signal),
            self._check_liquidity(signal),
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

    def _check_asset_class_concentration(self, signal) -> tuple:
        asset_class = signal['instrument_class']
        count = sum(1 for p in self.positions.values()
                    if p['instrument_class'] == asset_class)
        if count >= 2:
            return False, f'ASSET_CLASS_CONCENTRATION_{asset_class}'
        return True, 'OK'

    def _check_underlying_concentration(self, signal) -> tuple:
        underlying = signal['underlying']
        count = sum(1 for p in self.positions.values()
                    if p['underlying'] == underlying)
        if count >= 2:
            return False, f'UNDERLYING_CONCENTRATION_{underlying}'
        return True, 'OK'

    def _check_lot_size_affordable(self, signal) -> tuple:
        if signal['margin_per_lot'] > self.capital:
            return False, 'CANNOT_AFFORD_SINGLE_LOT'
        return True, 'OK'

    def _check_liquidity(self, signal) -> tuple:
        ic = signal['instrument_class']
        if ic == 'INDEX_OPTIONS':
            if signal.get('oi_lots', 0) < 100:
                return False, 'INSUFFICIENT_OI'
            if signal.get('bid_ask_spread_pct', 1) > 0.08:
                return False, 'BID_ASK_TOO_WIDE'
        elif ic == 'COMMODITY_FUTURES':
            if signal.get('daily_volume_lots', 0) < 500:
                return False, 'INSUFFICIENT_VOLUME'
        elif ic == 'CURRENCY_FUTURES':
            if signal.get('daily_volume_lots', 0) < 1000:
                return False, 'INSUFFICIENT_VOLUME'
        return True, 'OK'

    def _check_duplicate_tuple(self, signal) -> tuple:
        """No two positions with same (underlying, direction, holding_period)."""
        key = (signal['underlying'], signal['direction'], signal['holding_period'])
        existing_keys = [(p['underlying'], p['direction'], p['holding_period'])
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
        current_capital = self.capital + pnl
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        self.portfolio_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        if self.portfolio_drawdown > 0.08 and not self.halted:
            self.halted = True
            self.halt_reason = f'DRAWDOWN_{self.portfolio_drawdown:.1%}'
        if self.daily_pnl < -self.capital * 0.04 and not self.halted:
            self.halted = True
            self.halt_reason = 'SINGLE_DAY_LOSS_4PCT'

    def check_mid_hold_exits(self, position: dict) -> tuple:
        """Returns (should_exit, reason). Call at each of 3 daily schedule times."""
        if position.get('is_option'):
            theta_decayed = position.get('theta_decayed_pct', 0)
            if theta_decayed > THETA_STOP_PCT:
                return True, 'THETA_STOP'

        if position.get('days_held', 0) >= 15:
            return True, 'HARD_EXPIRY'

        if position.get('today_move_vs_expected_vol', 0) > 2.5:
            return True, 'VOL_SPIKE'

        if position.get('vix_change_today', 0) > VIX_SPIKE_THRESHOLD:
            if position.get('is_option'):
                return True, 'VIX_SPIKE_OPTION_EXIT'

        if position.get('basis_zscore') and abs(position['basis_zscore']) > 2.0:
            return True, 'BASIS_DISLOCATION'

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

def _make_signal(underlying='NIFTY', instrument_class='INDEX_OPTIONS',
                 direction='LONG', holding_period=5,
                 lot_value=75_000, margin_per_lot=50_000,
                 oi_lots=500, bid_ask_spread_pct=0.02,
                 daily_volume_lots=2000):
    return {
        'underlying':        underlying,
        'instrument_class':  instrument_class,
        'direction':         direction,
        'holding_period':    holding_period,
        'lot_value':         lot_value,
        'margin_per_lot':    margin_per_lot,
        'oi_lots':           oi_lots,
        'bid_ask_spread_pct': bid_ask_spread_pct,
        'daily_volume_lots': daily_volume_lots,
    }


def _verification_check():
    print("=== Step 7 — Risk Governor verification (5 unit tests) ===\n")
    passed = 0

    # Test 1: System halts on drawdown > 8%
    rg = RiskGovernor(capital=1_000_000)
    rg.record_position_closed('T1', -90_000)    # -9% drawdown
    ok, _ = rg.can_enter_position(_make_signal(), 1)
    result = not ok and 'DRAWDOWN' in rg.halt_reason
    print(f"  Test 1 — Halt on drawdown >8%:          {'PASS' if result else 'FAIL'}")
    if result:
        passed += 1

    # Test 2: System halts on single-day loss > 4%
    rg = RiskGovernor(capital=1_000_000)
    rg.record_position_closed('T2', -50_000)    # -5% single-day loss
    ok, _ = rg.can_enter_position(_make_signal(), 1)
    result = not ok and rg.halt_reason == 'SINGLE_DAY_LOSS_4PCT'
    print(f"  Test 2 — Halt on daily loss >4%:         {'PASS' if result else 'FAIL'}")
    if result:
        passed += 1

    # Test 3: Block when max positions reached
    rg = RiskGovernor(capital=10_000_000)
    for i in range(MAX_POSITIONS_TOTAL):
        rg.record_position_opened(f'T{i}', {
            'instrument_class': 'INDEX_OPTIONS',
            'underlying': f'FAKE{i}',
            'direction': 'LONG',
            'holding_period': 5,
            'margin': 50_000,
        })
    ok, reason = rg.can_enter_position(_make_signal(), 1)
    result = not ok and reason == 'MAX_POSITIONS_REACHED'
    print(f"  Test 3 — Block at max {MAX_POSITIONS_TOTAL} positions:      {'PASS' if result else 'FAIL'}")
    if result:
        passed += 1

    # Test 4: Block when margin limit (60%) exceeded
    rg = RiskGovernor(capital=1_000_000)
    rg.record_position_opened('T1', {
        'instrument_class': 'INDEX_OPTIONS',
        'underlying': 'NIFTY',
        'direction': 'LONG',
        'holding_period': 5,
        'margin': 650_000,   # 65% of capital already used
    })
    # lot_value=50000 → notional=50000 < 70000 (7% of 1M) so position-size check passes
    sig = _make_signal(lot_value=50_000, margin_per_lot=100_000)
    ok, reason = rg.can_enter_position(sig, 1)
    result = not ok and reason == 'MARGIN_LIMIT_EXCEEDED'
    print(f"  Test 4 — Block on margin >60%:           {'PASS' if result else 'FAIL'}")
    if result:
        passed += 1

    # Test 5: Transition watch blocks new entry (mock regime engine)
    import unittest.mock as mock
    rg = RiskGovernor(capital=10_000_000)
    fake_regime = {'regime_state': 'SIDEWAYS', 'regime_confidence': 0.45,
                   'transition_warning_flag': True}
    with mock.patch('regime_engine_sherm.get_current_regime', return_value=fake_regime):
        ok, reason = rg.can_enter_position(_make_signal(), 1)
    result = not ok and reason == 'TRANSITION_WATCH_ACTIVE'
    print(f"  Test 5 — Transition watch blocks entry:  {'PASS' if result else 'FAIL'}")
    if result:
        passed += 1

    print(f"\n  {passed}/5 tests passed.")
    if passed == 5:
        print("  PASS — all 5 unit tests passed.")
    else:
        print("  FAIL — some unit tests failed. See above.")
    print("\n=== Step 7 complete ===")


if __name__ == '__main__':
    _verification_check()
