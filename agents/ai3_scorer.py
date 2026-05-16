"""
Step 12 — AI 3 Scorer.
Determines position size in lots for approved signals.
Rule-based (confidence tiers) by default.
Contextual bandit overrides sizing when active (>= 200 trades).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOTAL_CAPITAL, MAX_POSITION_PCT


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
        max_notional = capital * MAX_POSITION_PCT
        margin_per_lot = signal.get('margin_per_lot', 50_000)

        # Confidence-based sizing (always computed; used in shadow mode)
        confidence = signal['hmm_confidence_at_signal']
        if confidence >= 0.70:
            conf_multiplier = 1.0
        elif confidence >= 0.50:
            conf_multiplier = 0.75
        else:
            conf_multiplier = 0.50

        if self.bandit is not None:
            meta_score = signal.get('meta_labeler_score', 0.6)
            context    = self.bandit.build_context(signal, meta_score)
            action     = self.bandit.select_action(context, meta_score)
            signal['bandit_action']  = action
            signal['bandit_context'] = context.tolist()

            if self.bandit.trade_count >= self.bandit.MIN_TRADES_ACTIVATE:
                # Active mode: bandit controls sizing
                multiplier = self.bandit.SIZE_MULTIPLIERS[action]
                if action == 'SKIP':
                    signal['recommended_lots'] = 0
                    signal['size_multiplier']  = 0.0
                    signal['confidence_score'] = 0.0
                    return signal
                adjusted_notional          = max_notional * multiplier
                signal['recommended_lots'] = max(1, int(adjusted_notional / margin_per_lot))
                signal['size_multiplier']  = multiplier
                signal['confidence_score'] = confidence * multiplier
                return signal
            # else: shadow mode — bandit action logged, confidence sizing used below

        # Default: confidence-based sizing
        adjusted_notional          = max_notional * conf_multiplier
        signal['recommended_lots'] = max(1, int(adjusted_notional / margin_per_lot))
        signal['confidence_score'] = confidence * conf_multiplier
        signal['size_multiplier']  = conf_multiplier
        return signal
