"""
A stand-in for the MQL5 Expert Advisor, for testing the Python client
without a MetaTrader terminal.

This is a test double, not a simulator. It implements the protocol
faithfully — including request_id deduplication, which is the property
most worth testing — and models just enough broker behaviour to exercise
the client's error paths. It does not model spreads, slippage, swaps or
price movement, and must never be mistaken for a backtest.

Fault injection (drop_replies, reject_next, fail_next) exists so tests can
reproduce the failure modes that are hard to trigger against a real
terminal and expensive to discover in production.
"""

import json
import socket
import threading

from mt5_bridge.protocol import (
    CMD_ACCOUNT, CMD_CLOSE, CMD_MODIFY_STOP, CMD_OPEN, CMD_PING,
    CMD_POSITIONS, CMD_QUOTE, MUTATING_COMMANDS, STATUS_ERROR, STATUS_OK,
    STATUS_REJECTED, build_response, decode, encode,
)


class FakeEAServer:
    """Threaded TCP server speaking the bridge protocol."""

    def __init__(self, host='127.0.0.1', port=0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(8)
        self.host, self.port = self._sock.getsockname()

        self._thread  = None
        self._running = False

        # Broker state
        self.positions   = {}      # ticket -> position dict
        self.next_ticket = 1000
        self.account     = {
            'balance': 100_000.0, 'equity': 100_000.0,
            'margin': 0.0, 'margin_free': 100_000.0, 'currency': 'USD',
        }
        self.quotes = {
            'EURUSD': (1.08495, 1.08505),
            'GBPUSD': (1.26990, 1.27010),
            'USDJPY': (149.995, 150.015),
            'XAUUSD': (1999.75, 2000.25),
        }

        # Idempotency ledger — request_id -> stored response data.
        self.processed = {}

        # Observability for assertions
        self.command_log = []       # every request received, including dupes
        self.executions  = []       # mutating commands actually acted on

        # Fault injection
        self.drop_replies = 0       # accept and act, then hang up before replying
        self.reject_next  = 0       # respond REJECTED
        self.fail_next    = 0       # respond ERROR

    # -- lifecycle --------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass
        if self._thread:
            self._thread.join(timeout=2)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _serve(self):
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            try:
                self._handle(conn)
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle(self, conn):
        buf = b''
        conn.settimeout(5)
        while not buf.endswith(b'\n'):
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf += chunk

        request = decode(buf)
        self.command_log.append(request)

        response = self._dispatch(request)

        # Simulate a reply lost in transit: the command HAS been executed,
        # but the caller never learns the outcome. This is the exact
        # scenario request_id dedup exists to make survivable.
        if self.drop_replies > 0:
            self.drop_replies -= 1
            return

        conn.sendall(encode(response))

    # -- dispatch ---------------------------------------------------------

    def _dispatch(self, request: dict) -> dict:
        rid     = request.get('request_id')
        command = request.get('command')
        params  = request.get('params', {})

        if self.fail_next > 0:
            self.fail_next -= 1
            return build_response(rid, STATUS_ERROR, reason='injected failure')

        if self.reject_next > 0:
            self.reject_next -= 1
            return build_response(rid, STATUS_REJECTED, reason='injected rejection')

        # Idempotency: replay the stored result rather than acting twice.
        if command in MUTATING_COMMANDS and rid in self.processed:
            return build_response(rid, STATUS_OK, **self.processed[rid])

        if command == CMD_PING:
            return build_response(rid, STATUS_OK, pong=True)

        if command == CMD_ACCOUNT:
            return build_response(rid, STATUS_OK, **self.account)

        if command == CMD_POSITIONS:
            return build_response(rid, STATUS_OK,
                                  positions=list(self.positions.values()))

        if command == CMD_QUOTE:
            symbol = params.get('symbol')
            if symbol not in self.quotes:
                return build_response(rid, STATUS_ERROR,
                                      reason=f'unknown symbol {symbol}')
            bid, ask = self.quotes[symbol]
            return build_response(rid, STATUS_OK, bid=bid, ask=ask)

        if command == CMD_OPEN:
            return self._do_open(rid, params)

        if command == CMD_CLOSE:
            return self._do_close(rid, params)

        if command == CMD_MODIFY_STOP:
            return self._do_modify(rid, params)

        return build_response(rid, STATUS_ERROR, reason=f'unknown command {command}')

    # -- mutating handlers ------------------------------------------------

    def _do_open(self, rid, params):
        symbol     = params.get('symbol')
        stop_price = params.get('stop_price')

        # A real broker refuses an order with no stop when one is required,
        # and refuses a stop on the wrong side of the market.
        if not stop_price:
            return build_response(rid, STATUS_REJECTED, reason='no stop attached')
        if symbol not in self.quotes:
            return build_response(rid, STATUS_REJECTED, reason=f'unknown symbol {symbol}')

        bid, ask   = self.quotes[symbol]
        direction  = params.get('direction')
        fill_price = ask if direction == 'LONG' else bid

        if direction == 'LONG' and stop_price >= fill_price:
            return build_response(rid, STATUS_REJECTED,
                                  reason='stop above entry for LONG')
        if direction == 'SHORT' and stop_price <= fill_price:
            return build_response(rid, STATUS_REJECTED,
                                  reason='stop below entry for SHORT')

        ticket = self.next_ticket
        self.next_ticket += 1
        position = {
            'ticket':     ticket,
            'symbol':     symbol,
            'direction':  direction,
            'lots':       params.get('lots'),
            'open_price': fill_price,
            'stop_price': stop_price,
            'comment':    params.get('comment', ''),
        }
        self.positions[ticket] = position
        self.executions.append(('open', rid, ticket))

        data = {'ticket': ticket, 'fill_price': fill_price,
                'stop_price': stop_price, 'lots': params.get('lots')}
        self.processed[rid] = data
        return build_response(rid, STATUS_OK, **data)

    def _do_close(self, rid, params):
        ticket = int(params.get('ticket', 0))
        if ticket not in self.positions:
            return build_response(rid, STATUS_REJECTED,
                                  reason=f'no such position {ticket}')
        position = self.positions.pop(ticket)
        self.executions.append(('close', rid, ticket))
        bid, ask = self.quotes[position['symbol']]
        close_price = bid if position['direction'] == 'LONG' else ask
        data = {'ticket': ticket, 'close_price': close_price}
        self.processed[rid] = data
        return build_response(rid, STATUS_OK, **data)

    def _do_modify(self, rid, params):
        ticket = int(params.get('ticket', 0))
        if ticket not in self.positions:
            return build_response(rid, STATUS_REJECTED,
                                  reason=f'no such position {ticket}')
        self.positions[ticket]['stop_price'] = params.get('stop_price')
        self.executions.append(('modify_stop', rid, ticket))
        data = {'ticket': ticket, 'stop_price': params.get('stop_price')}
        self.processed[rid] = data
        return build_response(rid, STATUS_OK, **data)

    # -- helpers for tests ------------------------------------------------

    def force_close(self, ticket: int):
        """Simulate a broker-side stop firing while we were not watching."""
        self.positions.pop(int(ticket), None)

    def count_executions(self, kind: str) -> int:
        return sum(1 for e in self.executions if e[0] == kind)
