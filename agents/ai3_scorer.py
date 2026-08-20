"""
Step 11 — AI 3 Scorer.
Determines position size in lots for approved signals.
Rule-based (confidence tiers) by default.
Contextual bandit overrides sizing when active (>= 200 trades).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

from config import (
    TOTAL_CAPITAL, MAX_POSITION_PCT, CONTRACT_SIZE, MIN_LOT,
    MAX_LOTS_BY_PERIOD, FX_STOP, SYMBOL_STOP_MULTIPLIER, PAIR_LEGS,
    ACCOUNT_CURRENCY, EDGE_STOP_PCT, EDGE_RISK_PCT, RISK_PER_TRADE_PCT,
    RL_STOP_BOUNDS, BROKER_LEVERAGE, DRAWDOWN_HALT_PCT, MAX_POSITIONS_TOTAL,
)


def stop_distance_pct(pair: str, holding_period: int,
                      edge_id: str = None, rl_multiplier: float = 1.0) -> float:
    """
    Stop as a fraction of price.

    Precedence: the edge's own stop, else the holding-period default. Then
    the symbol multiplier (gold needs more room), then any RL adjustment,
    clamped to RL_STOP_BOUNDS.

    Widening the stop here does not widen risk — size_position divides by
    this, so the position shrinks to compensate.
    """
    base = EDGE_STOP_PCT.get(edge_id) if edge_id else None
    if base is None:
        base = FX_STOP.get(holding_period, 0.015)
    base *= SYMBOL_STOP_MULTIPLIER.get(pair, 1.0)

    lo, hi = RL_STOP_BOUNDS
    return base * max(lo, min(rl_multiplier, hi))


def risk_pct_for(edge_id: str = None) -> float:
    """Risk budget for an edge. The invariant the RL layer must not touch."""
    return EDGE_RISK_PCT.get(edge_id, RISK_PER_TRADE_PCT)


def quote_to_account(pair: str, price: float) -> float:
    """
    Value of one unit of the pair's quote currency, in account currency.

    For USD-quoted pairs (EURUSD, XAUUSD) a quote unit is already a dollar.
    For USD-based pairs (USDJPY) the quote is yen, worth 1/price dollars.
    Crosses with neither leg in USD would need a third rate; the MVP
    universe has none, so that case raises rather than silently mis-sizing.
    """
    base, quote = PAIR_LEGS[pair]
    if quote == ACCOUNT_CURRENCY:
        return 1.0
    if base == ACCOUNT_CURRENCY:
        return 1.0 / price
    raise ValueError(
        f"{pair} has no {ACCOUNT_CURRENCY} leg; cross-rate conversion "
        f"is not implemented for the MVP universe")


def size_position(pair: str, entry_price: float, holding_period: int,
                  capital: float, risk_multiplier: float = 1.0,
                  edge_id: str = None, rl_stop_multiplier: float = 1.0) -> float:
    """
    Lots to trade, from risk rather than notional.

    Risk budget is capital x risk_pct x multiplier. Loss at the stop is
    lots x contract_size x stop_distance, converted to account currency.
    Solving for lots and quantising down to MIN_LOT keeps realised risk at
    or under budget — rounding up would breach it.

    Because stop_distance is in the denominator, an edge with a wider stop
    automatically takes a smaller position for identical dollar risk.
    """
    if entry_price <= 0:
        return 0.0

    risk_amount   = capital * risk_pct_for(edge_id) * risk_multiplier
    stop_distance = entry_price * stop_distance_pct(
        pair, holding_period, edge_id, rl_stop_multiplier)
    if stop_distance <= 0:
        return 0.0

    loss_per_lot = (CONTRACT_SIZE[pair] * stop_distance
                    * quote_to_account(pair, entry_price))
    if loss_per_lot <= 0:
        return 0.0

    raw_lots = risk_amount / loss_per_lot

    # Quantise DOWN to the broker's lot step. int() truncation on a value
    # below 1.0 would yield 0 and the old max(1, ...) floor then forced a
    # full standard lot — a silent over-size of up to 100x on a micro-lot
    # position.
    lots = math.floor(raw_lots / MIN_LOT) * MIN_LOT

    # Margin ceiling. Risk sizing governs the downside; this bounds how much
    # of the account a single position ties up as collateral, which a very
    # tight stop would otherwise let balloon.
    notional_per_lot = (CONTRACT_SIZE[pair] * entry_price
                        * quote_to_account(pair, entry_price))
    margin_per_lot   = notional_per_lot / BROKER_LEVERAGE
    if margin_per_lot > 0:
        margin_cap_lots = (capital * MAX_POSITION_PCT) / margin_per_lot
        lots = min(lots, math.floor(margin_cap_lots / MIN_LOT) * MIN_LOT)

    lots = min(lots, float(MAX_LOTS_BY_PERIOD.get(holding_period, 1)))
    return round(max(lots, 0.0), 2)


class AI3_Scorer:
    """
    Determines position size in lots for approved signals.
    Shadow mode: confidence-based sizing, bandit action logged but not applied.
    Active mode (>= 200 bandit trades): bandit SKIP/SMALL/MEDIUM/LARGE overrides.
    """

    def __init__(self, bandit=None):
        self.bandit = bandit

    def score_and_size(self, signal: dict, capital: float = TOTAL_CAPITAL) -> dict:
        """Returns signal with 'recommended_lots', 'confidence_score', 'size_multiplier'."""
        pair        = signal['pair']
        entry_price = signal.get('entry_price', 0.0)
        holding     = signal.get('holding_period', 5)
        confidence  = signal['hmm_confidence_at_signal']
        edge_id     = signal.get('edge_id')
        # Set by the RL layer when it wants a wider or tighter stop than the
        # edge's default. Risk is unaffected — size absorbs the change.
        rl_stop     = signal.get('rl_stop_multiplier', 1.0)

        if confidence >= 0.70:
            conf_multiplier = 1.0
        elif confidence >= 0.50:
            conf_multiplier = 0.75
        else:
            conf_multiplier = 0.50

        multiplier = conf_multiplier

        if self.bandit is not None:
            meta_score = signal.get('meta_labeler_score', 0.6)
            context    = self.bandit.build_context(signal, meta_score)
            action     = self.bandit.select_action(context, meta_score)
            signal['bandit_action']  = action
            signal['bandit_context'] = context.tolist()

            if self.bandit.trade_count >= self.bandit.MIN_TRADES_ACTIVATE:
                if action == 'SKIP':
                    signal['recommended_lots'] = 0.0
                    signal['size_multiplier']  = 0.0
                    signal['confidence_score'] = 0.0
                    return signal
                # Clamped at 1.0: MAX_POSITION_PCT is a hard risk ceiling, so
                # the bandit may size down but never above it. The LARGE
                # action previously multiplied the cap by 1.5x and breached it.
                multiplier = min(self.bandit.SIZE_MULTIPLIERS[action], 1.0)

        signal['recommended_lots'] = size_position(
            pair, entry_price, holding, capital, multiplier, edge_id, rl_stop)
        signal['size_multiplier']  = multiplier
        signal['confidence_score'] = confidence * multiplier
        signal['stop_distance_pct'] = stop_distance_pct(
            pair, holding, edge_id, rl_stop)
        return signal


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== Step 11 — AI 3 Scorer verification (6 tests) ===\n")
    passed = 0
    capital = 1_350_000

    # Test 1: realised risk stays within budget
    lots = size_position('EURUSD', 1.0850, 5, capital)
    stop = 1.0850 * stop_distance_pct('EURUSD', 5)
    risk = lots * CONTRACT_SIZE['EURUSD'] * stop
    budget = capital * RISK_PER_TRADE_PCT
    result = 0 < risk <= budget
    print(f"  Test 1 — Risk within budget:              "
          f"{'PASS' if result else 'FAIL'} ({lots} lots, risk {risk:,.0f} <= {budget:,.0f})")
    passed += result

    # Test 2: fractional lots survive. Small capital must produce a micro
    # lot, not get floored to 0 and then forced up to 1.0.
    lots = size_position('EURUSD', 1.0850, 2, 10_000)
    result = 0 < lots < 1.0
    print(f"  Test 2 — Fractional lot preserved:        "
          f"{'PASS' if result else 'FAIL'} ({lots} lots)")
    passed += result

    # Test 3: quantised to the broker lot step
    lots = size_position('GBPUSD', 1.2700, 5, capital)
    result = abs(round(lots / MIN_LOT) - (lots / MIN_LOT)) < 1e-6
    print(f"  Test 3 — Quantised to {MIN_LOT} step:         "
          f"{'PASS' if result else 'FAIL'} ({lots} lots)")
    passed += result

    # Test 4: JPY quote conversion. USDJPY loss accrues in yen, so sizing
    # must divide by the rate; skipping that oversizes by ~150x.
    lots_eur = size_position('EURUSD', 1.0850, 5, capital)
    lots_jpy = size_position('USDJPY', 150.00, 5, capital)
    result = 0 < lots_jpy < lots_eur * 5
    print(f"  Test 4 — JPY quote conversion:            "
          f"{'PASS' if result else 'FAIL'} (EURUSD {lots_eur}, USDJPY {lots_jpy})")
    passed += result

    # Test 5: gold uses the 100oz contract and its widened stop
    lots = size_position('XAUUSD', 2000.0, 5, capital)
    stop = 2000.0 * FX_STOP[5] * SYMBOL_STOP_MULTIPLIER['XAUUSD']
    risk = lots * CONTRACT_SIZE['XAUUSD'] * stop
    result = 0 < risk <= capital * MAX_POSITION_PCT
    print(f"  Test 5 — XAUUSD contract + wider stop:    "
          f"{'PASS' if result else 'FAIL'} ({lots} lots, risk {risk:,.0f})")
    passed += result

    # Test 6: bandit LARGE cannot exceed the position cap
    class _FakeBandit:
        ACTIONS = ['SKIP', 'SMALL', 'MEDIUM', 'LARGE']
        SIZE_MULTIPLIERS = {'SKIP': 0, 'SMALL': 0.5, 'MEDIUM': 1.0, 'LARGE': 1.5}
        MIN_TRADES_ACTIVATE = 200
        trade_count = 500
        def build_context(self, signal, meta):
            import numpy as np
            return np.zeros(10)
        def select_action(self, context, meta):
            return 'LARGE'

    scorer = AI3_Scorer(bandit=_FakeBandit())
    sig = scorer.score_and_size({
        'pair': 'EURUSD', 'entry_price': 1.0850, 'holding_period': 5,
        'hmm_confidence_at_signal': 0.80,
    }, capital)
    risk = (sig['recommended_lots'] * CONTRACT_SIZE['EURUSD']
            * 1.0850 * FX_STOP[5])
    result = sig['size_multiplier'] <= 1.0 and risk <= capital * MAX_POSITION_PCT
    print(f"  Test 6 — Bandit LARGE respects cap:       "
          f"{'PASS' if result else 'FAIL'} (mult {sig['size_multiplier']}, risk {risk:,.0f})")
    passed += result

    # Test 7: with the lot cap lifted, risk-based sizing should consume the
    # budget rather than sit far under it. The other tests all clamp at
    # MAX_LOTS_BY_PERIOD, so without this the risk math is never exercised.
    import unittest.mock as mock
    with mock.patch.dict('agents.ai3_scorer.MAX_LOTS_BY_PERIOD', {5: 1000}):
        lots = size_position('EURUSD', 1.0850, 5, capital)
    risk = lots * CONTRACT_SIZE['EURUSD'] * 1.0850 * stop_distance_pct('EURUSD', 5)
    budget = capital * RISK_PER_TRADE_PCT
    utilisation = risk / budget
    result = 0.95 <= utilisation <= 1.0
    print(f"  Test 7 — Budget utilisation (cap lifted): "
          f"{'PASS' if result else 'FAIL'} ({lots} lots, {utilisation:.1%} of budget)")
    passed += result

    # Test 8 — the core property: moving the stop must NOT move risk.
    # This is what lets each edge carry its own stop and lets the RL layer
    # retune one mid-flight without changing exposure.
    with mock.patch.dict('agents.ai3_scorer.MAX_LOTS_BY_PERIOD', {5: 10_000}):
        risks = []
        for mult in (0.5, 1.0, 2.0):
            lots = size_position('EURUSD', 1.0850, 5, capital,
                                 edge_id='SEED-001', rl_stop_multiplier=mult)
            stop = 1.0850 * stop_distance_pct('EURUSD', 5, 'SEED-001', mult)
            risks.append(lots * CONTRACT_SIZE['EURUSD'] * stop)
    spread = (max(risks) - min(risks)) / max(risks)
    result = spread < 0.02      # within quantisation error
    print(f"  Test 8 — Risk invariant to stop width:    "
          f"{'PASS' if result else 'FAIL'} "
          f"(risks {', '.join(f'{r:,.0f}' for r in risks)}; spread {spread:.2%})")
    passed += result

    # Test 9 — per-edge stops actually differ, and the wider-stopped edge
    # takes the smaller position.
    with mock.patch.dict('agents.ai3_scorer.MAX_LOTS_BY_PERIOD', {5: 10_000}):
        tight = size_position('EURUSD', 1.0850, 5, capital, edge_id='SEED-003')
        wide  = size_position('EURUSD', 1.0850, 5, capital, edge_id='SEED-004')
    result = wide < tight
    print(f"  Test 9 — Wider stop => smaller position:  "
          f"{'PASS' if result else 'FAIL'} "
          f"(SEED-003 {tight}, SEED-004 {wide})")
    passed += result

    # Test 10 — worst case across all 8 slots stays inside the halt.
    worst = max(EDGE_RISK_PCT.values()) * 8
    result = worst <= DRAWDOWN_HALT_PCT
    print(f"  Test 10 — 8 concurrent stops < halt:      "
          f"{'PASS' if result else 'FAIL'} "
          f"({worst:.1%} vs {DRAWDOWN_HALT_PCT:.1%} halt)")
    passed += result

    print(f"\n  {passed}/10 tests passed.")
    print("  PASS — all 10 unit tests passed." if passed == 10
          else "  FAIL — some unit tests failed. See above.")
    print("\n  NOTE: tests 1/3/4/5 clamp at MAX_LOTS_BY_PERIOD, so their lot")
    print("  counts reflect that cap rather than the risk calculation.")
    print("\n=== Step 11 complete ===")


if __name__ == '__main__':
    _verification_check()
