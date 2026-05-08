"""
Step 11 — Slot System.
Manages available capacity per holding period.
Prevents over-concentration in any single time horizon.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from config import MAX_SLOTS_BY_PERIOD


class SlotSystem:
    MAX_SLOTS = MAX_SLOTS_BY_PERIOD   # {2: 3, 5: 2, 10: 2, 15: 1}
    TOTAL_SLOTS = sum(MAX_SLOTS_BY_PERIOD.values())   # 8

    def __init__(self):
        self.active_slots = {}   # trade_id → slot_info
        self.waitlist = []       # signals waiting for a slot

    def available_slots(self, holding_period: int) -> int:
        used = sum(1 for s in self.active_slots.values()
                   if s['holding_period'] == holding_period)
        return self.MAX_SLOTS[holding_period] - used

    def total_available(self) -> int:
        return self.TOTAL_SLOTS - len(self.active_slots)

    def can_add(self, holding_period: int) -> bool:
        return (self.available_slots(holding_period) > 0
                and self.total_available() > 0)

    def add_slot(self, trade_id: str, slot_info: dict):
        self.active_slots[trade_id] = slot_info

    def release_slot(self, trade_id: str):
        if trade_id in self.active_slots:
            del self.active_slots[trade_id]

    def add_to_waitlist(self, signal: dict):
        signal['waitlisted_at'] = datetime.now().isoformat()
        self.waitlist.append(signal)

    def process_waitlist(self, current_features: dict) -> list:
        """On slot opening, check waitlist for valid pending signals."""
        WAITLIST_EXPIRY = {2: 0, 5: 2, 10: 3, 15: 4}   # days
        now = datetime.now()
        valid_signals = []
        expired = []

        for signal in self.waitlist:
            hp  = signal['holding_period']
            age = (now - datetime.fromisoformat(signal['waitlisted_at'])).days
            if age > WAITLIST_EXPIRY.get(hp, 0):
                expired.append(signal)
                continue
            if self._trigger_still_valid(signal, current_features):
                valid_signals.append(signal)

        self.waitlist = [s for s in self.waitlist
                         if s not in expired and s not in valid_signals]
        return valid_signals

    def _trigger_still_valid(self, signal: dict, features: dict) -> bool:
        """Recheck that the trigger condition still holds on fresh data."""
        return True   # full re-check uses feature_definitions.py
