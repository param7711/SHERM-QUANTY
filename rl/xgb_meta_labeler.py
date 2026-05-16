"""
Step 14 — XGB Meta-Labeler.
Predicts whether a given signal will be profitable.
Shadow mode until 150 trades; filtering authority after that.
State persists across restarts: model saved to database/rl_meta_labeler.pkl.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd

from config import META_LABELER_MIN_TRADES, META_LABELER_THRESHOLD

_MODEL_PATH = 'database/rl_meta_labeler.pkl'
_COUNT_PATH = 'database/rl_meta_labeler_count.json'


class XGBMetaLabeler:
    """
    Predicts whether a given signal will be profitable (1) or not (0).
    Trained on accumulated trade history (shadow ledger + actual outcomes).
    Retrained weekly once minimum trade count reached.

    Activates: shadow mode immediately. Filtering authority after 150 trades.
    State persists to disk so accumulated learning survives restarts.
    """

    MIN_TRADES_SHADOW   = 0
    MIN_TRADES_ACTIVATE = META_LABELER_MIN_TRADES
    THRESHOLD           = META_LABELER_THRESHOLD

    FEATURES = [
        'trigger_value', 'hmm_confidence', 'regime_encoded',
        'vix_level', 'days_to_expiry', 'holding_period',
        'recent_win_rate_edge',
        'vol_expanding', 'z_21d', 'rsi_2', 'mom_5d',
        'day_of_week',
        'time_since_last_trade_edge',
    ]

    def __init__(self):
        self.model       = None
        self.trade_count = 0
        self._load_state()

    def _load_state(self):
        """Load persisted model and trade count if available."""
        try:
            if os.path.exists(_MODEL_PATH) and os.path.exists(_COUNT_PATH):
                import joblib
                self.model = joblib.load(_MODEL_PATH)
                with open(_COUNT_PATH) as f:
                    self.trade_count = json.load(f).get('trade_count', 0)
        except Exception:
            self.model       = None
            self.trade_count = 0

    def _save_state(self):
        """Persist model and trade count to disk."""
        try:
            import joblib
            os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
            joblib.dump(self.model, _MODEL_PATH)
            with open(_COUNT_PATH, 'w') as f:
                json.dump({'trade_count': self.trade_count}, f)
        except Exception:
            pass

    def predict(self, signal: dict) -> float:
        """Returns probability (0–1) that signal will be profitable."""
        if self.model is None:
            return 0.6   # neutral before training
        features = self._extract_features(signal)
        return float(self.model.predict_proba([features])[0][1])

    def retrain(self, trade_history: pd.DataFrame):
        """Called weekly. Trains on all available labeled outcomes."""
        if len(trade_history) < self.MIN_TRADES_ACTIVATE:
            return

        X = trade_history[self.FEATURES].fillna(0)
        y = (trade_history['net_return_pct'] > 0).astype(int)

        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss',
        )
        self.model.fit(X, y)
        self.trade_count = len(trade_history)
        self._save_state()

    def _extract_features(self, signal: dict) -> list:
        regime_map = {
            'H_BULL': 4, 'L_BULL': 3, 'SIDEWAYS': 2,
            'L_BEAR': 1, 'H_BEAR': 0,
        }
        return [
            signal.get('trigger_value', 0) if isinstance(signal.get('trigger_value'), float) else 0,
            signal.get('hmm_confidence_at_signal', 0.5),
            regime_map.get(signal.get('regime_at_signal', 'SIDEWAYS'), 2),
            signal.get('vix_level', 15),
            signal.get('dte', 5),
            signal.get('holding_period', 5),
            signal.get('recent_edge_win_rate', 0.55),
            signal.get('vol_expanding', 0),
            signal.get('z_21d', 0),
            signal.get('rsi_2', 50),
            signal.get('mom_5d', 0),
            signal.get('day_of_week', 2),
            signal.get('days_since_last_trade', 7),
        ]
