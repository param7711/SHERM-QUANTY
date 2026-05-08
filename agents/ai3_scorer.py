"""
Step 12 — AI 3 Scorer.
Determines position size in lots for approved signals.
In MVP: rule-based (Kelly-adjacent). RL bandit overrides this in Step 14.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOTAL_CAPITAL, MAX_POSITION_PCT


class AI3_Scorer:
    """
    Determines position size in lots for approved signals.
    In MVP: rule-based (Kelly-adjacent). RL bandit overrides this in Step 14.
    """

    def score_and_size(self, signal: dict, capital: float = TOTAL_CAPITAL) -> dict:
        """Returns signal with 'recommended_lots' and 'confidence_score' added."""
        max_notional = capital * MAX_POSITION_PCT

        confidence = signal['hmm_confidence_at_signal']
        if confidence >= 0.70:
            size_multiplier = 1.0
        elif confidence >= 0.50:
            size_multiplier = 0.75
        else:
            size_multiplier = 0.50

        adjusted_notional = max_notional * size_multiplier

        margin_per_lot = signal.get('margin_per_lot', 50_000)
        lots = max(1, int(adjusted_notional / margin_per_lot))

        signal['recommended_lots']  = lots
        signal['confidence_score']  = confidence * size_multiplier
        signal['size_multiplier']   = size_multiplier
        return signal
