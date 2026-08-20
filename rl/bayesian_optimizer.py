"""
Step 14 — Bayesian Threshold Optimizer.
Adapts signal thresholds based on recent outcomes using Gaussian Process.
Shadow mode until 50 trades; threshold adjustments activate after that.
State persists across restarts: saved to database/rl_bayesian_state.json.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

_STATE_PATH = 'database/rl_bayesian_state.json'


class BayesianThresholdOptimizer:
    """
    Adapts signal thresholds (e.g., z_21d < -2.0) based on recent outcomes.
    Uses Gaussian Process to model expected P&L as function of threshold.
    Updates after each trade.

    Activates: immediately. Runs in shadow mode for first 50 trades,
    then adjusts thresholds within HARD_BOUNDS.
    State persists to disk so accumulated learning survives restarts.
    """

    HARD_BOUNDS = {
        'z_21d_threshold':   (-3.0, -1.5),
        'rsi_2_threshold':   (5, 15),
        # Spot FX barely gaps intraweek; only the Sunday reopen gaps
        # meaningfully, and rarely past 0.5%. The old (-1.5%, -0.7%) band
        # was sized for NSE overnight index gaps and would never trigger.
        'gap_pct_threshold': (-0.006, -0.001),
        'cross_spread_z':    (-3.0, -1.5),
    }
    SHADOW_MODE_UNTIL = 50

    _DEFAULTS = {
        'z_21d_threshold':   -2.0,
        'rsi_2_threshold':   10,
        'gap_pct_threshold': -0.003,
        'cross_spread_z':    -2.0,
    }

    # Callers hand us the feature that fired; the optimizer tunes the
    # threshold applied to it. Without this mapping every observation lands
    # under a key HARD_BOUNDS does not contain, and — because the GP fit is
    # wrapped in a broad except — the failure is silent and the thresholds
    # never move.
    _FEATURE_TO_PARAM = {
        'z_21d':            'z_21d_threshold',
        'rsi_2':            'rsi_2_threshold',
        'gap_pct':          'gap_pct_threshold',
        'gap_unfilled_eod': 'gap_pct_threshold',
        'spread_z':         'cross_spread_z',
    }

    @classmethod
    def resolve_param(cls, name: str) -> str:
        """Accept either a feature name or a param name; return the param."""
        if name in cls.HARD_BOUNDS:
            return name
        return cls._FEATURE_TO_PARAM.get(name, 'z_21d_threshold')

    def __init__(self):
        self.trade_count = 0
        self.observations = []
        self.current_thresholds = dict(self._DEFAULTS)
        self._load_state()

    def _load_state(self):
        """Load persisted observations and thresholds if available."""
        try:
            if os.path.exists(_STATE_PATH):
                with open(_STATE_PATH) as f:
                    data = json.load(f)
                self.trade_count        = data.get('trade_count', 0)
                self.observations       = data.get('observations', [])
                self.current_thresholds = {
                    **self._DEFAULTS,
                    **data.get('current_thresholds', {}),
                }
        except Exception:
            self.trade_count        = 0
            self.observations       = []
            self.current_thresholds = dict(self._DEFAULTS)

    def _save_state(self):
        """Persist observations and thresholds to disk."""
        try:
            os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
            with open(_STATE_PATH, 'w') as f:
                json.dump({
                    'trade_count':        self.trade_count,
                    'observations':       self.observations,
                    'current_thresholds': self.current_thresholds,
                }, f)
        except Exception:
            pass

    def record_outcome(self, threshold_used: float, param_name: str, pnl_pct: float):
        param_name = self.resolve_param(param_name)
        self.observations.append({
            'param':     param_name,
            'threshold': threshold_used,
            'pnl':       pnl_pct,
        })
        self.trade_count += 1
        if self.trade_count >= self.SHADOW_MODE_UNTIL:
            self._update_thresholds(param_name)
        self._save_state()

    def get_current_threshold(self, param_name: str) -> float:
        return self.current_thresholds.get(param_name, self._DEFAULTS.get(param_name, -2.0))

    def _update_thresholds(self, param_name: str):
        """Fit GP and find threshold maximising expected P&L."""
        relevant = [o for o in self.observations if o['param'] == param_name]
        if len(relevant) < 20:
            return

        # Resolved before the try block: an unknown param is a wiring bug,
        # and the broad except below would otherwise bury it forever.
        if param_name not in self.HARD_BOUNDS:
            raise KeyError(
                f"Unknown optimizer param {param_name!r}; "
                f"expected one of {sorted(self.HARD_BOUNDS)}")

        thresholds = np.array([o['threshold'] for o in relevant])
        pnls       = np.array([o['pnl']       for o in relevant])

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF

            gp = GaussianProcessRegressor(kernel=RBF(), normalize_y=True)
            gp.fit(thresholds.reshape(-1, 1), pnls)

            lo, hi = self.HARD_BOUNDS[param_name]
            candidates = np.linspace(lo, hi, 50).reshape(-1, 1)
            pred_mean, _ = gp.predict(candidates, return_std=True)
            optimal = float(candidates[np.argmax(pred_mean)][0])
            self.current_thresholds[param_name] = optimal
        except Exception:
            pass
