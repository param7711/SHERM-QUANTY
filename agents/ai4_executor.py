"""
Step 13 — AI 4 Executor.
Places orders on MetaTrader via the MQL5 bridge.
Single platform. The bridge's lock is the atomic permission queue.

By the time a signal reaches here every decision is made — pair,
direction, lots, stop distance. This agent's only job is to make reality
match that intent, or report honestly that it could not.

Three modes:
  live      — bridge configured and reachable; real orders
  paper     — same code path, bridge pointed at a demo account
  simulated — no bridge; fills at signal price, for BACKTEST ONLY

paper and live are the same path deliberately. If paper trading used the
simulated path, 90 days of it would never exercise the bridge, and the
bridge is where execution bugs live. A demo account runs the real
platform with real spreads and fake money, so the only difference between
paper and live is which account the EA is logged into.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MAX_SLIPPAGE_PIPS, MAX_SPREAD_PCT, PIP_SIZE, MT_ACCOUNT_LOGIN,
)
from mt5_bridge.client import MT5BridgeClient
from mt5_bridge.protocol import BridgeError, BridgeRejected


class AI4_Executor:
    """
    Places orders on MetaTrader through the MQL5 bridge.

    Never opens a position without a broker-side stop attached in the same
    call. The Risk Governor's drawdown limits assume stops execute at the
    distance the sizer used; a stop that lives only in this process does
    nothing between the three daily checkpoints, and every position is
    held across unobserved hours on a 24/5 market.
    """

    def __init__(self, bridge: MT5BridgeClient = None, simulated: bool = False):
        self.bridge    = bridge or MT5BridgeClient()
        self.simulated = simulated
        self._live     = False
        if not simulated:
            self._live = self.bridge.ping()

    # -- status -----------------------------------------------------------

    @property
    def mode(self) -> str:
        if self.simulated:
            return 'simulated'
        if not self._live:
            return 'offline'
        return 'live' if MT_ACCOUNT_LOGIN else 'paper'

    def health_check(self) -> dict:
        """Called before each scan. Bridge down means stop entering."""
        if self.simulated:
            return {'ok': True, 'mode': 'simulated'}
        alive = self.bridge.ping()
        self._live = alive
        return {'ok': alive, 'mode': self.mode,
                'reason': None if alive else 'bridge unreachable'}

    # -- execution --------------------------------------------------------

    def execute_signal(self, signal: dict) -> dict:
        """
        Open the position described by signal.

        Requires stop_price — computed upstream by AI 3 from the edge's
        stop distance, which is the same number position size was derived
        from. Refuses rather than opening unprotected.
        """
        pair   = signal['pair']
        lots   = signal.get('recommended_lots', 0)
        stop   = signal.get('stop_price')

        if lots <= 0:
            return {'status': 'SKIPPED', 'reason': 'zero size'}
        if not stop:
            return {'status': 'REJECTED_NO_STOP',
                    'reason': 'stop_price absent; refusing to open unprotected'}

        if self.simulated:
            return self._simulate_fill(signal)

        if not self._live and not self.bridge.ping():
            # Cannot verify anything through a dead bridge. Existing
            # positions are safe on their broker-side stops; we simply
            # stop adding.
            return {'status': 'BRIDGE_DOWN',
                    'reason': 'cannot place orders, bridge unreachable'}

        # Spread gate. Costs are a large fraction of the edge at 2-day
        # horizons, so a widened quote is a reason not to trade at all.
        try:
            quote = self.bridge.get_quote(pair)
        except (BridgeError, BridgeRejected) as e:
            return {'status': 'ERROR', 'reason': f'quote failed: {e}'}

        if quote['spread_pct'] > MAX_SPREAD_PCT:
            return {'status': 'CANCELLED_SPREAD',
                    'reason': f"spread {quote['spread_pips']:.1f}p exceeds limit",
                    'spread_pips': quote['spread_pips']}

        expected = quote['ask'] if signal['direction'] == 'LONG' else quote['bid']

        try:
            result = self.bridge.open_position(
                symbol=pair,
                direction=signal['direction'],
                lots=lots,
                stop_price=stop,
                comment=str(signal.get('edge_id', ''))[:31],
            )
        except BridgeRejected as e:
            # Definitively did not execute. Safe to report as a clean miss.
            return {'status': 'REJECTED', 'reason': str(e)}
        except BridgeError as e:
            # Ambiguous: may or may not have opened. The caller must
            # reconcile before doing anything else with this pair.
            return {'status': 'UNKNOWN', 'reason': str(e),
                    'requires_reconcile': True}

        fill        = result.get('fill_price', expected)
        pip         = PIP_SIZE.get(pair, 0.0001)
        slippage    = abs(fill - expected) / pip
        max_slip    = MAX_SLIPPAGE_PIPS.get(signal.get('holding_period', 5), 20)

        return {
            'status':               'FILLED',
            'ticket':               result.get('ticket'),
            'fill_price':           fill,
            'stop_price':           result.get('stop_price'),
            'lots':                 result.get('lots', lots),
            'slippage_pips':        slippage,
            'spread_at_entry_pips': quote['spread_pips'],
            # Reported, not enforced: the position is already open, and
            # closing it immediately would realise the slippage as a loss
            # plus another spread. AI 5 tracks this for execution quality.
            'slippage_breach':      slippage > max_slip,
        }

    def close_position(self, ticket: int) -> dict:
        if self.simulated:
            return {'status': 'SIMULATED_CLOSE', 'ticket': ticket}
        try:
            result = self.bridge.close_position(ticket)
            return {'status': 'CLOSED', **result}
        except BridgeRejected as e:
            return {'status': 'REJECTED', 'reason': str(e)}
        except BridgeError as e:
            return {'status': 'UNKNOWN', 'reason': str(e),
                    'requires_reconcile': True}

    def move_stop(self, ticket: int, stop_price: float) -> dict:
        """Used for the 2.5R breakeven move and trailing updates."""
        if self.simulated:
            return {'status': 'SIMULATED_MODIFY', 'ticket': ticket,
                    'stop_price': stop_price}
        try:
            result = self.bridge.modify_stop(ticket, stop_price)
            return {'status': 'MODIFIED', **result}
        except BridgeRejected as e:
            return {'status': 'REJECTED', 'reason': str(e)}
        except BridgeError as e:
            return {'status': 'UNKNOWN', 'reason': str(e),
                    'requires_reconcile': True}

    # -- reconciliation ---------------------------------------------------

    def reconcile(self, expected_positions: dict) -> dict:
        """
        Compare believed state against the broker's. Run at every
        checkpoint: a broker-side stop firing overnight is invisible until
        something asks, and a phantom position corrupts both the slot
        count and the currency-exposure cap.
        """
        if self.simulated:
            return {'in_sync': True, 'matched': list(expected_positions),
                    'closed_at_broker': [], 'unknown_to_us': [],
                    'stop_missing': [], 'live': {}}
        return self.bridge.reconcile(expected_positions)

    # -- backtest only ----------------------------------------------------

    def _simulate_fill(self, signal: dict) -> dict:
        """Instant fill at signal price. BACKTEST ONLY — never paper trading."""
        return {
            'status':      'SIMULATED_FILL',
            'ticket':      f"SIM_{signal.get('signal_id', 'x')}",
            'fill_price':  signal.get('entry_price', 0.0),
            'stop_price':  signal.get('stop_price'),
            'lots':        signal.get('recommended_lots', 0),
            'slippage_pips': 0.0,
        }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== Step 13 — AI 4 Executor verification (7 tests) ===\n")
    from mt5_bridge.fake_server import FakeEAServer

    passed = 0

    def sig(**kw):
        base = {
            'pair': 'EURUSD', 'direction': 'LONG', 'recommended_lots': 0.35,
            'stop_price': 1.0741, 'holding_period': 5, 'edge_id': 'SEED-001',
            'entry_price': 1.0850, 'signal_id': 'S1',
        }
        base.update(kw)
        return base

    with FakeEAServer() as srv:
        bridge = MT5BridgeClient(host=srv.host, port=srv.port, timeout=3)
        ex = AI4_Executor(bridge=bridge)

        # Test 1 — a normal fill, with the stop attached at the broker
        res = ex.execute_signal(sig())
        held = srv.positions.get(res.get('ticket'), {})
        result = res['status'] == 'FILLED' and held.get('stop_price') == 1.0741
        print(f"  Test 1 — Fills with broker-side stop:     "
              f"{'PASS' if result else 'FAIL'} ({res['status']})")
        passed += result

        # Test 2 — a signal with no stop never reaches the broker
        before = srv.count_executions('open')
        res = ex.execute_signal(sig(stop_price=None))
        result = (res['status'] == 'REJECTED_NO_STOP'
                  and srv.count_executions('open') == before)
        print(f"  Test 2 — Stopless signal refused:         "
              f"{'PASS' if result else 'FAIL'} ({res['status']})")
        passed += result

        # Test 3 — zero size is skipped, not sent
        before = srv.count_executions('open')
        res = ex.execute_signal(sig(recommended_lots=0))
        result = (res['status'] == 'SKIPPED'
                  and srv.count_executions('open') == before)
        print(f"  Test 3 — Zero size skipped:               "
              f"{'PASS' if result else 'FAIL'} ({res['status']})")
        passed += result

        # Test 4 — a widened spread cancels before any order is placed
        srv.quotes['EURUSD'] = (1.0800, 1.0900)     # 100 pips
        before = srv.count_executions('open')
        res = ex.execute_signal(sig())
        result = (res['status'] == 'CANCELLED_SPREAD'
                  and srv.count_executions('open') == before)
        print(f"  Test 4 — Wide spread cancels entry:       "
              f"{'PASS' if result else 'FAIL'} ({res['status']})")
        passed += result
        srv.quotes['EURUSD'] = (1.08495, 1.08505)

        # Test 5 — the 2.5R breakeven move reaches the broker
        opened = ex.execute_signal(sig(signal_id='S2'))
        moved  = ex.move_stop(opened['ticket'], 1.0850)
        result = (moved['status'] == 'MODIFIED'
                  and srv.positions[opened['ticket']]['stop_price'] == 1.0850)
        print(f"  Test 5 — Stop move reaches broker:        "
              f"{'PASS' if result else 'FAIL'} ({moved['status']})")
        passed += result

        # Test 6 — an ambiguous write is flagged for reconciliation rather
        # than reported as either success or clean failure
        srv.fail_next = 99
        res = ex.close_position(opened['ticket'])
        result = res['status'] == 'UNKNOWN' and res.get('requires_reconcile')
        print(f"  Test 6 — Ambiguous close flags reconcile: "
              f"{'PASS' if result else 'FAIL'} ({res['status']})")
        passed += result
        srv.fail_next = 0

        # Test 7 — reconciliation notices a broker-side stop-out
        live_now = list(srv.positions)
        srv.force_close(live_now[0])
        diff = ex.reconcile({t: {} for t in live_now})
        result = live_now[0] in diff['closed_at_broker'] and not diff['in_sync']
        print(f"  Test 7 — Detects overnight stop-out:      "
              f"{'PASS' if result else 'FAIL'} "
              f"(closed={diff['closed_at_broker']})")
        passed += result

    print(f"\n  {passed}/7 tests passed.")
    print("  PASS — all 7 unit tests passed." if passed == 7
          else "  FAIL — some unit tests failed. See above.")
    print("\n  NOTE: verified against FakeEAServer. The MQL5 EA is")
    print("  unverified until run against a live terminal.")
    print("\n=== Step 13 complete ===")


if __name__ == '__main__':
    _verification_check()
