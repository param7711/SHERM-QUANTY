"""
Step 11 — Portfolio Construction.
Takes approved signals, returns ranked subset to trade.
Applies concentration limits and correlation filters.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slot_system import SlotSystem


class PortfolioConstruction:
    """
    Takes list of approved signals, returns subset to trade.
    Applies concentration limits and correlation filters.
    Ranks by adjusted_score.
    """

    def select_trades(self, approved_signals: list, slot_system: SlotSystem) -> list:
        # Filter to signals where slots are available
        eligible = [s for s in approved_signals
                    if slot_system.can_add(s['holding_period'])]

        # Concentration check: max 2 per asset class
        eligible = self._apply_concentration_limits(eligible)

        # Remove duplicative signals (same underlying + direction)
        eligible = self._remove_duplicates(eligible)

        # Score and rank
        scored = [(s, self._compute_adjusted_score(s)) for s in eligible]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top signals fitting available slots
        selected = []
        for signal, score in scored:
            if slot_system.can_add(signal['holding_period']):
                selected.append(signal)

        return selected

    def _apply_concentration_limits(self, signals: list) -> list:
        counts = {}
        result = []
        for s in signals:
            ic = s['instrument_class']
            if counts.get(ic, 0) < 2:
                result.append(s)
                counts[ic] = counts.get(ic, 0) + 1
        return result

    def _remove_duplicates(self, signals: list) -> list:
        seen = set()
        result = []
        for s in signals:
            key = (s['underlying'], s['direction'])
            if key not in seen:
                result.append(s)
                seen.add(key)
        return result

    def _compute_adjusted_score(self, signal: dict) -> float:
        """Higher score = higher priority. Simple version for MVP."""
        base_score  = signal.get('edge_win_rate', 0.5)
        regime_bonus = 0.05 if signal.get('hmm_confidence_at_signal', 0) > 0.70 else 0.0
        return base_score + regime_bonus
