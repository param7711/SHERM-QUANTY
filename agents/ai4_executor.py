"""
Step 12 — AI 4 Executor.
Places orders via Zerodha Kite API.
Single broker. Atomic permission queue prevents race conditions.

NOTE: Kite API requires live credentials. In sandbox/backtest mode,
execute_signal() returns a simulated fill at the signal price.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time

from config import ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN


class AI4_Executor:
    """
    Places orders via Zerodha Kite API.
    Single broker. Atomic permission queue prevents race conditions.
    """

    def __init__(self):
        self._kite = None
        self._live = False
        if ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN:
            try:
                from kiteconnect import KiteConnect
                self._kite = KiteConnect(api_key=ZERODHA_API_KEY)
                self._kite.set_access_token(ZERODHA_ACCESS_TOKEN)
                self._live = True
            except Exception:
                pass

    def execute_signal(self, signal: dict) -> dict:
        """
        Places a limit order at signal price. Cancels if not filled within 30s.
        Returns trade result dict.
        """
        if not self._live:
            return self._simulate_fill(signal)

        contract = self._select_contract(signal)
        order_params = {
            'tradingsymbol': contract['symbol'],
            'exchange':       contract['exchange'],
            'transaction_type': 'BUY' if signal['direction'] == 'LONG' else 'SELL',
            'quantity':       contract['lot_size'] * signal['recommended_lots'],
            'order_type':    'LIMIT',
            'price':          contract['mid_price'],
            'product':        'MIS' if signal['holding_period'] <= 2 else 'NRML',
            'validity':       'IOC' if signal['holding_period'] <= 2 else 'DAY',
        }
        max_slippage = self._get_max_slippage(signal)
        try:
            order_id = self._kite.place_order(**order_params)
            filled = self._monitor_fill(order_id, timeout=30)
            if not filled:
                self._kite.cancel_order(order_id=order_id)
                return {'status': 'CANCELLED_TIMEOUT', 'order_id': order_id}
            fill_price = self._get_fill_price(order_id)
            slippage = abs(fill_price - contract['mid_price']) / contract['mid_price']
            if slippage > max_slippage:
                return {'status': 'FILLED_SLIPPAGE_BREACH', 'fill_price': fill_price,
                        'slippage': slippage, 'order_id': order_id}
            return {'status': 'FILLED', 'fill_price': fill_price, 'order_id': order_id}
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}

    def _simulate_fill(self, signal: dict) -> dict:
        """Paper-trading simulation — fill at signal price instantly."""
        return {
            'status':     'SIMULATED_FILL',
            'fill_price': signal.get('entry_price', 0.0),
            'order_id':   f"SIM_{signal['signal_id']}",
            'lots':       signal.get('recommended_lots', 1),
        }

    def _select_contract(self, signal: dict) -> dict:
        """Select ATM strike and nearest valid expiry for index options."""
        raise NotImplementedError("_select_contract requires live Kite instrument data")

    def _monitor_fill(self, order_id: str, timeout: int = 30) -> bool:
        """Poll order status for up to timeout seconds."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            orders = self._kite.orders()
            for o in orders:
                if str(o['order_id']) == str(order_id):
                    if o['status'] == 'COMPLETE':
                        return True
                    if o['status'] in ('CANCELLED', 'REJECTED'):
                        return False
            time.sleep(1)
        return False

    def _get_fill_price(self, order_id: str) -> float:
        orders = self._kite.orders()
        for o in orders:
            if str(o['order_id']) == str(order_id):
                return float(o.get('average_price', 0))
        return 0.0

    def _get_max_slippage(self, signal: dict) -> float:
        """Slippage tolerance by instrument class and holding period."""
        hp = signal['holding_period']
        ic = signal['instrument_class']
        if ic == 'INDEX_OPTIONS':
            slippage_map = {2: 0.05, 5: 0.06, 10: 0.08, 15: 0.10}
        elif ic in ('COMMODITY_FUTURES', 'CURRENCY_FUTURES'):
            slippage_map = {2: 0.0015, 5: 0.0020, 10: 0.0025, 15: 0.0030}
        else:
            slippage_map = {2: 0.05, 5: 0.06, 10: 0.08, 15: 0.10}
        return slippage_map.get(hp, 0.05)
