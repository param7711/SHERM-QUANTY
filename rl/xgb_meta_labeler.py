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

    # These are SIGNAL-TIME features, so the training frame must come from
    # rl_replay_buffer.state_features_json — not execution_quality_log,
    # which records what happened at fill and stores none of them. Pointing
    # retrain() at the execution log (as the scheduler used to) raises
    # KeyError on nearly every name here.
    # 'days_to_expiry' was also removed: spot FX has no expiry.
    FEATURES = [
        'trigger_value', 'hmm_confidence', 'regime_encoded',
        'vix_level', 'holding_period',
        'recent_win_rate_edge',
        'vol_expanding', 'z_21d', 'rsi_2', 'mom_5d',
        'day_of_week',
        'time_since_last_trade_edge',
        'carry_direction', 'spread_pips_at_signal',
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
        """
        Called weekly. Trains on all available labeled outcomes.

        trade_history must carry every name in FEATURES plus
        net_return_pct — use load_training_frame() to build it from the
        replay buffer rather than passing the execution log.
        """
        if len(trade_history) < self.MIN_TRADES_ACTIVATE:
            return

        missing = [c for c in self.FEATURES + ['net_return_pct']
                   if c not in trade_history.columns]
        if missing:
            raise KeyError(
                f"Training frame is missing {missing}. Signal-time features "
                f"come from rl_replay_buffer.state_features_json; build the "
                f"frame with load_training_frame().")

        X = trade_history[self.FEATURES].fillna(0)
        y = (trade_history['net_return_pct'] > 0).astype(int)

        # A single-class label set trains a model that always predicts that
        # class. Better to keep the previous model (or none) than to install
        # a degenerate one.
        if y.nunique() < 2:
            print(f"[meta-labeler] skipping retrain: all {len(y)} outcomes "
                  f"are {'wins' if y.iloc[0] else 'losses'}")
            return

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
        # Order must match FEATURES exactly.
        values = {
            'trigger_value': (signal.get('trigger_value', 0)
                              if isinstance(signal.get('trigger_value'), float) else 0),
            'hmm_confidence':             signal.get('hmm_confidence_at_signal', 0.5),
            'regime_encoded':             regime_map.get(
                                              signal.get('regime_at_signal', 'SIDEWAYS'), 2),
            'vix_level':                  signal.get('vix_level', 15),
            'holding_period':             signal.get('holding_period', 5),
            'recent_win_rate_edge':       signal.get('recent_edge_win_rate', 0.55),
            'vol_expanding':              signal.get('vol_expanding', 0),
            'z_21d':                      signal.get('z_21d', 0),
            'rsi_2':                      signal.get('rsi_2', 50),
            'mom_5d':                     signal.get('mom_5d', 0),
            'day_of_week':                signal.get('day_of_week', 2),
            'time_since_last_trade_edge': signal.get('days_since_last_trade', 7),
            'carry_direction':            signal.get('carry_direction', 0),
            'spread_pips_at_signal':      signal.get('spread_pips', 1.0),
        }
        return [values[name] for name in self.FEATURES]


def load_training_frame(db_path: str = None) -> pd.DataFrame:
    """
    Build a meta-labeler training frame from rl_replay_buffer.

    Each row's state_features_json holds the signal-time snapshot; reward
    holds the realised net return. Expanding the JSON into columns gives
    exactly the layout retrain() expects. Rows written before a feature was
    added simply come back NaN and are filled with 0 at fit time.
    """
    import json
    import sqlite3
    if db_path is None:
        from config import DB_PATH as db_path

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            "SELECT state_features_json, reward, holding_period FROM rl_replay_buffer",
            conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        try:
            feats = json.loads(r['state_features_json']) or {}
        except Exception:
            feats = {}
        feats.setdefault('holding_period', r['holding_period'])
        feats['net_return_pct'] = r['reward']
        rows.append(feats)

    out = pd.DataFrame(rows)
    for col in XGBMetaLabeler.FEATURES:
        if col not in out.columns:
            out[col] = 0
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== RL — XGBoost Meta-Labeler verification (4 tests) ===\n")
    import numpy as np
    passed = 0
    labeler = XGBMetaLabeler()

    # Test 1: FEATURES and _extract_features stay aligned. They are keyed by
    # name now, so a drift raises rather than silently misaligning columns.
    sig = {'pair': 'EURUSD', 'trigger_value': -2.3, 'holding_period': 5,
           'hmm_confidence_at_signal': 0.7, 'regime_at_signal': 'SIDEWAYS',
           'carry_direction': -1, 'spread_pips': 1.2}
    feats = labeler._extract_features(sig)
    result = len(feats) == len(labeler.FEATURES)
    print(f"  Test 1 — Feature vector matches FEATURES: "
          f"{'PASS' if result else 'FAIL'} ({len(feats)} values)")
    passed += result

    # Test 2: no expiry feature survives anywhere.
    result = not any('expiry' in f or f == 'dte' for f in labeler.FEATURES)
    print(f"  Test 2 — No expiry feature remains:       "
          f"{'PASS' if result else 'FAIL'}")
    passed += result

    # Test 3: retrain on a frame missing signal-time columns must raise a
    # clear error rather than a bare KeyError. This is the failure the
    # scheduler used to hit every Friday.
    exec_log_like = pd.DataFrame({
        'trade_id': range(200), 'pair': ['EURUSD'] * 200,
        'net_return_pct': [0.01] * 200, 'holding_period': [5] * 200,
    })
    try:
        labeler.retrain(exec_log_like)
        result, msg = False, 'no error raised'
    except KeyError as e:
        result = 'load_training_frame' in str(e)
        msg = 'clear error'
    print(f"  Test 3 — Bad frame raises clearly:        "
          f"{'PASS' if result else 'FAIL'} ({msg})")
    passed += result

    # Test 4: a well-formed frame trains, and a single-class frame is
    # refused rather than producing a model that always says the same thing.
    rng = np.random.default_rng(0)
    n = 200
    good = pd.DataFrame({f: rng.normal(size=n) for f in labeler.FEATURES})
    good['net_return_pct'] = rng.normal(size=n)
    try:
        labeler.retrain(good)
        trained = labeler.model is not None
    except Exception as e:
        trained = False
        print(f"      (retrain raised: {e})")

    one_class = good.copy()
    one_class['net_return_pct'] = 0.01
    before = labeler.model
    labeler.retrain(one_class)
    refused = labeler.model is before

    result = trained and refused
    print(f"  Test 4 — Trains; refuses single-class:    "
          f"{'PASS' if result else 'FAIL'} (trained={trained}, refused={refused})")
    passed += result

    print(f"\n  {passed}/4 tests passed.")
    print("  PASS — all 4 unit tests passed." if passed == 4
          else "  FAIL — some unit tests failed. See above.")


if __name__ == '__main__':
    _verification_check()
