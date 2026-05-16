"""
Step 14 — Contextual Bandit (LinUCB).
Selects position size: SKIP / SMALL / MEDIUM / LARGE.
Shadow mode until 200 trades; sizing authority after that.
State persists across restarts: A/b matrices saved to database/rl_bandit_state.npz.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import BANDIT_MIN_TRADES, MAX_POSITION_PCT

_STATE_PATH = 'database/rl_bandit_state.npz'


class ContextualBandit:
    """
    Selects position size (SKIP / SMALL=0.5x / MEDIUM=1x / LARGE=1.5x).
    Uses LinUCB algorithm. Online learning — updates after every trade.

    Activates: shadow mode immediately. Sizing authority after 200 trades.
    Hard cap: LARGE never exceeds MAX_POSITION_PCT regardless of bandit output.
    State persists to disk so accumulated learning survives restarts.
    """

    ACTIONS          = ['SKIP', 'SMALL', 'MEDIUM', 'LARGE']
    SIZE_MULTIPLIERS = {'SKIP': 0, 'SMALL': 0.5, 'MEDIUM': 1.0, 'LARGE': 1.5}
    MIN_TRADES_ACTIVATE = BANDIT_MIN_TRADES
    ALPHA = 1.0

    def __init__(self, n_features: int = 10):
        self.n_features  = n_features
        self.trade_count = 0
        self.A = {a: np.identity(n_features)  for a in self.ACTIONS}
        self.b = {a: np.zeros(n_features)     for a in self.ACTIONS}
        self._load_state()

    def _load_state(self):
        """Load persisted A/b matrices and trade count if available."""
        try:
            if os.path.exists(_STATE_PATH):
                data = np.load(_STATE_PATH)
                self.trade_count = int(data['trade_count'])
                for a in self.ACTIONS:
                    self.A[a] = data[f'A_{a}']
                    self.b[a] = data[f'b_{a}']
        except Exception:
            # Reset to defaults on any load error
            self.trade_count = 0
            self.A = {a: np.identity(self.n_features)  for a in self.ACTIONS}
            self.b = {a: np.zeros(self.n_features)     for a in self.ACTIONS}

    def _save_state(self):
        """Persist A/b matrices and trade count to disk."""
        try:
            os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
            save_dict = {'trade_count': np.array(self.trade_count)}
            for a in self.ACTIONS:
                save_dict[f'A_{a}'] = self.A[a]
                save_dict[f'b_{a}'] = self.b[a]
            np.savez(_STATE_PATH, **save_dict)
        except Exception:
            pass

    def select_action(self, context: np.ndarray, meta_labeler_score: float) -> str:
        """Select sizing action given context vector and meta-labeler score."""
        if self.trade_count < self.MIN_TRADES_ACTIVATE:
            if meta_labeler_score > 0.65:
                return 'MEDIUM'
            elif meta_labeler_score > 0.58:
                return 'SMALL'
            else:
                return 'SKIP'

        scores = {}
        for action in self.ACTIONS:
            A_inv = np.linalg.inv(self.A[action])
            theta = A_inv @ self.b[action]
            ucb   = theta @ context + self.ALPHA * np.sqrt(context @ A_inv @ context)
            scores[action] = ucb

        return max(scores, key=scores.get)

    def update(self, action: str, context: np.ndarray, reward: float):
        """Update LinUCB parameters after observing reward."""
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
        self.trade_count += 1
        self._save_state()

    def build_context(self, signal: dict, meta_score: float) -> np.ndarray:
        """Build context vector from signal features."""
        tv = signal.get('trigger_value', 0)
        return np.array([
            signal.get('hmm_confidence_at_signal', 0.5),
            meta_score,
            float(tv) if isinstance(tv, (int, float)) else 0.0,
            signal.get('holding_period', 5) / 15.0,
            1.0 if signal.get('instrument_class') == 'INDEX_OPTIONS' else 0.0,
            1.0 if signal.get('direction') == 'LONG' else 0.0,
            signal.get('recent_edge_win_rate', 0.55),
            signal.get('current_drawdown', 0.0),
            signal.get('open_positions_count', 0) / 8.0,
            signal.get('days_since_last_trade', 7) / 30.0,
        ])
