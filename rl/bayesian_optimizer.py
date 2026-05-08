"""
Step 14 — Bayesian Threshold Optimizer.
Adapts signal thresholds based on recent outcomes using Gaussian Process.
Shadow mode until 50 trades; threshold adjustments activate after that.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class BayesianThresholdOptimizer:
    """
    Adapts signal thresholds (e.g., z_21d < -2.0) based on recent outcomes.
    Uses Gaussian Process to model expected P&L as function of threshold.
    Updates after each trade.

    Activates: immediately. Runs in shadow mode for first 50 trades,
    then adjusts thresholds within HARD_BOUNDS.
    """

    HARD_BOUNDS = {
        'z_21d_threshold':   (-3.0, -1.5),
        'rsi_2_threshold':   (5, 15),
        'gap_pct_threshold': (-0.015, -0.007),
        'z_21d_currency':    (-2.5, -1.2),
    }
    SHADOW_MODE_UNTIL = 50

    def __init__(self):
        self.trade_count = 0
        self.observations = []   # {param, threshold, pnl}
        self.current_thresholds = {
            'z_21d_threshold':   -2.0,
            'rsi_2_threshold':   10,
            'gap_pct_threshold': -0.01,
            'z_21d_currency':    -1.8,
        }

    def record_outcome(self, threshold_used: float, param_name: str, pnl_pct: float):
        self.observations.append({
            'param':     param_name,
            'threshold': threshold_used,
            'pnl':       pnl_pct,
        })
        self.trade_count += 1
        if self.trade_count >= self.SHADOW_MODE_UNTIL:
            self._update_thresholds(param_name)

    def get_current_threshold(self, param_name: str) -> float:
        return self.current_thresholds.get(param_name, -2.0)

    def _update_thresholds(self, param_name: str):
        """Fit GP and find threshold maximising expected P&L."""
        relevant = [o for o in self.observations if o['param'] == param_name]
        if len(relevant) < 20:
            return

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
