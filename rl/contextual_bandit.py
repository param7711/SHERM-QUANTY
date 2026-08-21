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

from config import BANDIT_MIN_TRADES, MAX_POSITION_PCT, PIP_SIZE

_STATE_PATH = 'database/rl_bandit_state.npz'

# Bumped whenever the meaning or ordering of the context vector changes.
# Learned A/b matrices are only meaningful for the feature layout they were
# trained under, so a mismatch discards them rather than reusing weights
# that now point at different quantities.
_STATE_VERSION = 2


class ContextualBandit:
    """
    Selects position size (SKIP / SMALL / MEDIUM / LARGE).
    Uses LinUCB algorithm. Online learning — updates after every trade.

    Activates: shadow mode immediately. Sizing authority after 200 trades.
    LARGE maps to 1.0x the risk budget, never above it — the multipliers
    scale down from the cap rather than up through it.
    State persists to disk so accumulated learning survives restarts.
    """

    ACTIONS = ['SKIP', 'SMALL', 'MEDIUM', 'LARGE']
    # LARGE was 1.5x, which multiplied the position cap and broke the
    # guarantee in this docstring. The ladder now spans up TO the budget:
    # LARGE spends it in full, the others take a fraction.
    SIZE_MULTIPLIERS = {'SKIP': 0.0, 'SMALL': 0.4, 'MEDIUM': 0.7, 'LARGE': 1.0}
    MIN_TRADES_ACTIVATE = BANDIT_MIN_TRADES
    ALPHA = 1.0

    # Self-documenting, and asserted against the built vector so the two
    # cannot drift apart silently.
    FEATURE_NAMES = [
        'hmm_confidence',
        'meta_labeler_score',
        'trigger_value',
        'holding_period_norm',
        'carry_direction',        # was a dead INDEX_OPTIONS flag
        'is_long',
        'recent_edge_win_rate',
        'current_drawdown',
        'open_positions_norm',
        'days_since_last_trade_norm',
        'spread_pips_norm',       # execution cost at signal time
        'is_gold',                # XAUUSD volatility regime differs from the majors
    ]

    def __init__(self, n_features: int = None):
        self.n_features  = n_features or len(self.FEATURE_NAMES)
        self.trade_count = 0
        self.A = {a: np.identity(self.n_features)  for a in self.ACTIONS}
        self.b = {a: np.zeros(self.n_features)     for a in self.ACTIONS}
        self._load_state()

    def _reset(self):
        self.trade_count = 0
        self.A = {a: np.identity(self.n_features)  for a in self.ACTIONS}
        self.b = {a: np.zeros(self.n_features)     for a in self.ACTIONS}

    def _load_state(self):
        """Load persisted A/b matrices and trade count if available."""
        try:
            if not os.path.exists(_STATE_PATH):
                return
            data = np.load(_STATE_PATH)
            version = int(data['state_version']) if 'state_version' in data else 1
            if version != _STATE_VERSION:
                print(f"[bandit] discarding state v{version} "
                      f"(current v{_STATE_VERSION}): context layout changed")
                self._reset()
                return
            if data['A_SKIP'].shape[0] != self.n_features:
                print("[bandit] discarding state: feature count changed")
                self._reset()
                return
            self.trade_count = int(data['trade_count'])
            for a in self.ACTIONS:
                self.A[a] = data[f'A_{a}']
                self.b[a] = data[f'b_{a}']
        except Exception:
            self._reset()

    def _save_state(self):
        """Persist A/b matrices and trade count to disk."""
        try:
            os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
            save_dict = {
                'trade_count':   np.array(self.trade_count),
                'state_version': np.array(_STATE_VERSION),
            }
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
        """Build context vector from signal features. Order matches FEATURE_NAMES."""
        tv = signal.get('trigger_value', 0)
        pair = signal.get('pair', '')

        # Spread in pips, normalised against a ~5-pip reference so a typical
        # major sits near 0.2 and a wide quote pushes toward 1.
        spread_pct = signal.get('spread_pct', 0.0) or 0.0
        price      = signal.get('entry_price', 0.0) or 0.0
        pip        = PIP_SIZE.get(pair, 0.0001)
        spread_pips = (spread_pct * price / pip) if price > 0 else 0.0

        values = {
            'hmm_confidence':             signal.get('hmm_confidence_at_signal', 0.5),
            'meta_labeler_score':         meta_score,
            # Clipped: a runaway z-score would otherwise dominate the linear
            # model and distort every other dimension.
            'trigger_value':              float(np.clip(tv, -5, 5)) if isinstance(tv, (int, float)) else 0.0,
            'holding_period_norm':        signal.get('holding_period', 5) / 15.0,
            'carry_direction':            float(signal.get('carry_direction', 0)),
            'is_long':                    1.0 if signal.get('direction') == 'LONG' else 0.0,
            'recent_edge_win_rate':       signal.get('recent_edge_win_rate', 0.55),
            'current_drawdown':           signal.get('current_drawdown', 0.0),
            'open_positions_norm':        signal.get('open_positions_count', 0) / 8.0,
            'days_since_last_trade_norm': signal.get('days_since_last_trade', 7) / 30.0,
            'spread_pips_norm':           float(np.clip(spread_pips / 5.0, 0, 2)),
            'is_gold':                    1.0 if pair == 'XAUUSD' else 0.0,
        }
        return np.array([values[name] for name in self.FEATURE_NAMES])


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== RL — Contextual Bandit verification (5 tests) ===\n")
    import tempfile
    import shutil
    import unittest.mock as mock

    passed = 0
    tmpdir = tempfile.mkdtemp(prefix='bandit_test_')
    try:
        state = os.path.join(tmpdir, 'bandit.npz')
        with mock.patch(f'{__name__}._STATE_PATH', state):
            sig = {
                'pair': 'EURUSD', 'direction': 'LONG', 'holding_period': 5,
                'hmm_confidence_at_signal': 0.8, 'trigger_value': -2.3,
                'carry_direction': -1, 'spread_pct': 0.0001, 'entry_price': 1.0850,
            }
            b = ContextualBandit()
            ctx = b.build_context(sig, 0.62)

            result = len(ctx) == len(b.FEATURE_NAMES)
            print(f"  Test 1 — Context matches FEATURE_NAMES:  "
                  f"{'PASS' if result else 'FAIL'} ({len(ctx)} dims)")
            passed += result

            # The dead INDEX_OPTIONS flag is gone; carry_direction carries signal.
            idx = b.FEATURE_NAMES.index('carry_direction')
            result = ctx[idx] == -1.0
            print(f"  Test 2 — carry_direction populated:      "
                  f"{'PASS' if result else 'FAIL'} ({ctx[idx]})")
            passed += result

            # No action may exceed the risk budget.
            result = max(b.SIZE_MULTIPLIERS.values()) <= 1.0
            print(f"  Test 3 — No multiplier exceeds budget:   "
                  f"{'PASS' if result else 'FAIL'} "
                  f"(max {max(b.SIZE_MULTIPLIERS.values())})")
            passed += result

            # Gold flag distinguishes XAUUSD.
            gold_ctx = b.build_context({**sig, 'pair': 'XAUUSD', 'entry_price': 2000.0}, 0.62)
            gi = b.FEATURE_NAMES.index('is_gold')
            result = ctx[gi] == 0.0 and gold_ctx[gi] == 1.0
            print(f"  Test 4 — is_gold distinguishes XAUUSD:   "
                  f"{'PASS' if result else 'FAIL'}")
            passed += result

            # State round-trips, and a version bump discards stale weights.
            b.update('MEDIUM', ctx, 0.01)
            saved_count = b.trade_count
            b2 = ContextualBandit()
            reloaded = b2.trade_count == saved_count
            with mock.patch(f'{__name__}._STATE_VERSION', _STATE_VERSION + 1):
                b3 = ContextualBandit()
                discarded = b3.trade_count == 0
            result = reloaded and discarded
            print(f"  Test 5 — State round-trip + version guard: "
                  f"{'PASS' if result else 'FAIL'} "
                  f"(reloaded={reloaded}, stale discarded={discarded})")
            passed += result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n  {passed}/5 tests passed.")
    print("  PASS — all 5 unit tests passed." if passed == 5
          else "  FAIL — some unit tests failed. See above.")


if __name__ == '__main__':
    _verification_check()
