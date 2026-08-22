"""
Verification for the MetaTrader bridge client.

Runs against FakeEAServer, so it exercises the real protocol, the real
retry logic and the real reconciliation code without a terminal. What it
cannot verify is the MQL5 side — see ExpertAdvisor.mq5, which is
unverified until run against a live terminal.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt5_bridge.client import MT5BridgeClient
from mt5_bridge.fake_server import FakeEAServer
from mt5_bridge.protocol import BridgeError, BridgeRejected


def _client(server):
    return MT5BridgeClient(host=server.host, port=server.port, timeout=3)


def _verification_check():
    print("=== MT5 Bridge verification (10 tests) ===\n")
    passed = 0

    # Test 1 — round trip
    with FakeEAServer() as srv:
        c = _client(srv)
        result = c.ping()
    print(f"  Test 1 — Ping round trip:                 {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 2 — open attaches the stop atomically
    with FakeEAServer() as srv:
        c = _client(srv)
        res = c.open_position('EURUSD', 'LONG', 0.35, stop_price=1.0741)
        pos = srv.positions[res['ticket']]
        result = pos['stop_price'] == 1.0741 and pos['lots'] == 0.35
    print(f"  Test 2 — Stop attached at open:           "
          f"{'PASS' if result else 'FAIL'} (stop={pos['stop_price']})")
    passed += result

    # Test 3 — a position may never be opened without a stop
    with FakeEAServer() as srv:
        c = _client(srv)
        try:
            c.open_position('EURUSD', 'LONG', 0.35, stop_price=None)
            result = False
        except ValueError:
            result = True
        # and nothing reached the broker
        result = result and srv.count_executions('open') == 0
    print(f"  Test 3 — Stopless open refused locally:   {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 4 — THE important one. A reply lost in transit must not produce
    # two positions when the client retries.
    with FakeEAServer() as srv:
        c = _client(srv)
        srv.drop_replies = 1          # first attempt executes, reply vanishes
        res = c.open_position('EURUSD', 'LONG', 0.35, stop_price=1.0741)
        opens_executed = srv.count_executions('open')
        positions_held = len(srv.positions)
        result = opens_executed == 1 and positions_held == 1 and 'ticket' in res
    print(f"  Test 4 — Lost reply -> no double open:    "
          f"{'PASS' if result else 'FAIL'} "
          f"(executed {opens_executed}, holding {positions_held})")
    passed += result

    # Test 5 — the same request_id is reused across retries (the mechanism
    # test 4 depends on), and the duplicate is visible to the server.
    with FakeEAServer() as srv:
        c = _client(srv)
        srv.drop_replies = 1
        c.open_position('EURUSD', 'LONG', 0.10, stop_price=1.0741)
        open_cmds = [r for r in srv.command_log if r['command'] == 'open']
        ids = {r['request_id'] for r in open_cmds}
        result = len(open_cmds) == 2 and len(ids) == 1
    print(f"  Test 5 — Retry reuses one request_id:     "
          f"{'PASS' if result else 'FAIL'} "
          f"({len(open_cmds)} sends, {len(ids)} distinct id)")
    passed += result

    # Test 6 — a broker rejection is terminal, not retried
    with FakeEAServer() as srv:
        c = _client(srv)
        srv.reject_next = 1
        try:
            c.open_position('EURUSD', 'LONG', 0.35, stop_price=1.0741)
            result, detail = False, 'no exception'
        except BridgeRejected:
            attempts = len([r for r in srv.command_log if r['command'] == 'open'])
            result, detail = attempts == 1, f'{attempts} attempt'
        except BridgeError:
            result, detail = False, 'raised BridgeError, expected BridgeRejected'
    print(f"  Test 6 — Rejection is terminal:           "
          f"{'PASS' if result else 'FAIL'} ({detail})")
    passed += result

    # Test 7 — an unreachable bridge raises rather than failing quiet
    c = MT5BridgeClient(host='127.0.0.1', port=1, timeout=1)
    try:
        c.get_account()
        result = False
    except BridgeError as e:
        result = 'UNKNOWN' not in str(e)     # a read: state is not ambiguous
    print(f"  Test 7 — Dead bridge raises on read:      {'PASS' if result else 'FAIL'}")
    passed += result

    # Test 8 — a mutating command that exhausts retries flags UNKNOWN state,
    # so the caller knows to reconcile rather than assume either outcome.
    with FakeEAServer() as srv:
        c = _client(srv)
        srv.fail_next = 99
        try:
            c.close_position(1234)
            result, detail = False, 'no exception'
        except BridgeError as e:
            result, detail = 'UNKNOWN' in str(e), 'flagged UNKNOWN'
        except BridgeRejected:
            result, detail = False, 'rejected'
    print(f"  Test 8 — Ambiguous write flags UNKNOWN:   "
          f"{'PASS' if result else 'FAIL'} ({detail})")
    passed += result

    # Test 9 — quote derives spread in both pip and percent terms, and the
    # pip size is symbol-aware (JPY and gold quote differently).
    with FakeEAServer() as srv:
        c = _client(srv)
        eur = c.get_quote('EURUSD')
        jpy = c.get_quote('USDJPY')
        result = (abs(eur['spread_pips'] - 1.0) < 0.01
                  and abs(jpy['spread_pips'] - 2.0) < 0.01)
    print(f"  Test 9 — Symbol-aware pip maths:          "
          f"{'PASS' if result else 'FAIL'} "
          f"(EURUSD {eur['spread_pips']:.2f}p, USDJPY {jpy['spread_pips']:.2f}p)")
    passed += result

    # Test 10 — reconciliation catches a broker-side stop firing overnight.
    # We still believe the position is open; the broker knows better.
    with FakeEAServer() as srv:
        c = _client(srv)
        a = c.open_position('EURUSD', 'LONG', 0.35, stop_price=1.0741)
        b = c.open_position('GBPUSD', 'SHORT', 0.20, stop_price=1.2800)
        srv.force_close(a['ticket'])           # stop fired while we slept
        expected = {a['ticket']: {}, b['ticket']: {}}
        diff = c.reconcile(expected)
        result = (diff['closed_at_broker'] == [a['ticket']]
                  and diff['matched'] == [b['ticket']]
                  and not diff['in_sync'])
    print(f"  Test 10 — Reconcile detects stop-out:     "
          f"{'PASS' if result else 'FAIL'} "
          f"(closed={diff['closed_at_broker']}, matched={diff['matched']})")
    passed += result

    print(f"\n  {passed}/10 tests passed.")
    print("  PASS — all 10 unit tests passed." if passed == 10
          else "  FAIL — some unit tests failed. See above.")
    print("\n  NOTE: this verifies the Python side only. ExpertAdvisor.mq5")
    print("  is unverified until run against a live MetaTrader terminal.")


if __name__ == '__main__':
    _verification_check()
