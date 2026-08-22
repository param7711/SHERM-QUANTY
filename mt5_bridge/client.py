"""
Python side of the MetaTrader bridge.

Talks to an MQL5 Expert Advisor running inside the terminal (see
ExpertAdvisor.mq5). The EA holds the broker connection; this holds the
policy about retries, idempotency and what "we don't know" means.

Three properties this file exists to guarantee:

1. Retry is safe. Every mutating command carries a request_id the EA
   deduplicates against, so a lost reply can be retried without risking a
   second position.

2. The broker is the source of truth. get_positions() reads live state;
   nothing here caches an authoritative view of what we hold.

3. Failure means stop, not guess. A transport error on a mutating command
   raises BridgeError, and the caller must reconcile rather than assume
   either outcome.
"""

import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MT_BRIDGE_HOST, MT_BRIDGE_PORT, MT_BRIDGE_TIMEOUT, PIP_SIZE,
)
from mt5_bridge.protocol import (
    CMD_ACCOUNT, CMD_CLOSE, CMD_MODIFY_STOP, CMD_OPEN, CMD_PING,
    CMD_POSITIONS, CMD_QUOTE, MUTATING_COMMANDS, STATUS_OK, STATUS_REJECTED,
    BridgeError, BridgeRejected, build_request, decode, encode,
    new_request_id,
)

# Transport-level retries. Only ever safe because of request_id dedup.
MAX_RETRIES   = 3
RETRY_BACKOFF = (0.5, 1.0, 2.0)


class MT5BridgeClient:
    """
    Synchronous client for the MQL5 EA.

    A single lock serialises all traffic. This is the atomic permission
    queue the architecture calls for: two agents cannot interleave orders,
    and the EA sees one command at a time.
    """

    def __init__(self, host: str = None, port: int = None, timeout: int = None):
        self.host    = host or MT_BRIDGE_HOST
        self.port    = port or MT_BRIDGE_PORT
        self.timeout = timeout or MT_BRIDGE_TIMEOUT
        self._lock   = threading.Lock()

    # -- transport --------------------------------------------------------

    def _send_once(self, request: dict) -> dict:
        """One request/response round trip. Connection is not pooled."""
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        try:
            sock.sendall(encode(request))
            buf = b''
            while not buf.endswith(b'\n'):
                chunk = sock.recv(4096)
                if not chunk:
                    raise BridgeError('connection closed before response')
                buf += chunk
            return decode(buf)
        finally:
            sock.close()

    def _call(self, command: str, **params) -> dict:
        """
        Send a command, retrying transport failures.

        Retrying a mutating command is only safe because request_id is
        generated ONCE here and reused across attempts — the EA replays its
        stored result rather than executing again.
        """
        request_id = new_request_id()
        request    = build_request(command, request_id=request_id, **params)
        last_error = None

        with self._lock:
            for attempt in range(MAX_RETRIES):
                try:
                    response = self._send_once(request)
                except (socket.timeout, OSError, BridgeError) as e:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                    continue

                if response.get('request_id') != request_id:
                    last_error = BridgeError(
                        f"response id mismatch: sent {request_id}, "
                        f"got {response.get('request_id')}")
                    continue

                status = response.get('status')
                if status == STATUS_REJECTED:
                    # The broker refused outright. Definitively did not run,
                    # so this is not retryable and not ambiguous.
                    raise BridgeRejected(
                        f"{command} rejected: {response.get('data', {}).get('reason')}")
                if status != STATUS_OK:
                    last_error = BridgeError(
                        f"{command} failed: {response.get('data', {}).get('reason')}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                    continue

                return response.get('data', {})

        raise BridgeError(
            f"{command} failed after {MAX_RETRIES} attempts: {last_error}. "
            f"{'State is UNKNOWN — reconcile before retrying.' if command in MUTATING_COMMANDS else ''}")

    # -- reads ------------------------------------------------------------

    def ping(self) -> bool:
        try:
            self._call(CMD_PING)
            return True
        except (BridgeError, BridgeRejected):
            return False

    def get_account(self) -> dict:
        """balance, equity, margin, margin_free, currency."""
        return self._call(CMD_ACCOUNT)

    def get_positions(self) -> list:
        """
        Live open positions. THE source of truth — never substitute a
        locally cached view for this.
        """
        return self._call(CMD_POSITIONS).get('positions', [])

    def get_quote(self, symbol: str) -> dict:
        """bid, ask, and derived spread in price and pip terms."""
        q = self._call(CMD_QUOTE, symbol=symbol)
        bid, ask = q.get('bid'), q.get('ask')
        if bid is None or ask is None:
            raise BridgeError(f"incomplete quote for {symbol}: {q}")
        spread = ask - bid
        pip    = PIP_SIZE.get(symbol, 0.0001)
        mid    = (ask + bid) / 2.0
        return {
            'symbol':      symbol,
            'bid':         bid,
            'ask':         ask,
            'spread':      spread,
            'spread_pips': spread / pip,
            'spread_pct':  (spread / mid) if mid else 0.0,
            'mid':         mid,
        }

    # -- writes -----------------------------------------------------------

    def open_position(self, symbol: str, direction: str, lots: float,
                      stop_price: float, comment: str = '') -> dict:
        """
        Open a position with its protective stop attached in the same call.

        stop_price is mandatory. Placing first and attaching a stop second
        leaves a window where a live position is unprotected, and on a
        24/5 market held across unobserved hours that window is exactly
        where an account-ending move happens.
        """
        if stop_price is None or stop_price <= 0:
            raise ValueError(
                'stop_price is required — a position may never be opened '
                'without a broker-side stop attached atomically')
        if direction not in ('LONG', 'SHORT'):
            raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")
        if lots <= 0:
            raise ValueError(f"lots must be positive, got {lots}")

        return self._call(
            CMD_OPEN, symbol=symbol, direction=direction,
            lots=round(lots, 2), stop_price=stop_price, comment=comment)

    def close_position(self, ticket: int) -> dict:
        return self._call(CMD_CLOSE, ticket=ticket)

    def modify_stop(self, ticket: int, stop_price: float) -> dict:
        """Move a stop — the 2.5R breakeven shift, or a trailing update."""
        if stop_price is None or stop_price <= 0:
            raise ValueError('stop_price must be positive')
        return self._call(CMD_MODIFY_STOP, ticket=ticket, stop_price=stop_price)

    # -- reconciliation ---------------------------------------------------

    def reconcile(self, expected: dict) -> dict:
        """
        Compare our believed positions against the broker's.

        expected: {ticket: position_dict} as the system understands it.

        Divergence is normal, not exceptional — a broker-side stop fires
        overnight, a process restarts, someone closes a trade by hand. What
        matters is noticing, because a phantom position corrupts the slot
        count and the currency-exposure cap, both of which gate new entries.

        Returns the three-way diff. Callers decide policy; this only reports.
        """
        live = {int(p['ticket']): p for p in self.get_positions()}
        ours = {int(t): v for t, v in expected.items()}

        closed_at_broker = [t for t in ours if t not in live]   # we think open, broker says no
        unknown_to_us    = [t for t in live if t not in ours]   # broker has it, we do not
        matched          = [t for t in ours if t in live]

        stop_missing = [
            t for t in matched
            if not live[t].get('stop_price')
        ]

        return {
            'matched':          matched,
            'closed_at_broker': closed_at_broker,
            'unknown_to_us':    unknown_to_us,
            'stop_missing':     stop_missing,
            'live':             live,
            'in_sync':          not (closed_at_broker or unknown_to_us or stop_missing),
        }
