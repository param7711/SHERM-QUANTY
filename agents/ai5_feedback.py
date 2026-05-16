"""
Step 13 — AI 5 Feedback Loop.
Monitors positions, records trade outcomes, updates Edge Library,
populates RL replay buffer, appends to KAIROS log.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import uuid
import pandas as pd
from datetime import datetime

from config import DB_PATH, KAIROS_DB_PATH
from risk_governor import RiskGovernor


class AI5_FeedbackLoop:
    """
    Monitors positions, records outcomes, updates edge metrics, feeds RL.
    Bandit and Bayesian optimizer are updated after every closed trade.
    """

    def __init__(self, bandit=None, bayesian_optimizer=None):
        self.risk_governor      = RiskGovernor()
        self.bandit             = bandit
        self.bayesian_optimizer = bayesian_optimizer
        self._ensure_kairos_schema()

    def monitor_positions(self, open_positions: list, current_market: dict):
        """Called at each of 3 daily schedule times."""
        for position in open_positions:
            should_exit, reason = self.risk_governor.check_mid_hold_exits(position)
            if should_exit:
                self._execute_exit(position, reason)

    def record_trade_outcome(self, trade_id: str, outcome: dict):
        """Called when position closes."""
        self._write_execution_log(trade_id, outcome)
        self._update_edge_library(outcome)
        self._check_decay(outcome['edge_id'])
        self._write_replay_buffer(outcome)
        self._update_shadow_ledger_outcome(outcome)
        self._update_rl(outcome)

    def _update_rl(self, outcome: dict):
        """Update bandit and Bayesian optimizer with closed-trade outcome."""
        import numpy as np

        if self.bandit is not None and 'bandit_action' in outcome:
            action  = outcome['bandit_action']
            context = np.array(outcome.get('bandit_context') or [0.0] * 10)
            reward  = float(outcome.get('net_return_pct', 0.0))
            self.bandit.update(action, context, reward)

        if self.bayesian_optimizer is not None:
            param   = outcome.get('trigger_feature', 'z_21d_threshold')
            trigger = float(outcome.get('trigger_value', 0.0))
            pnl     = float(outcome.get('net_return_pct', 0.0))
            self.bayesian_optimizer.record_outcome(trigger, param, pnl)

    def _execute_exit(self, position: dict, reason: str):
        """Trigger exit order. In live: calls AI4_Executor. Here: logs the event."""
        pass

    def _write_execution_log(self, trade_id: str, outcome: dict):
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT OR IGNORE INTO execution_quality_log
               (trade_id, edge_id, instrument_class, underlying, contract_spec,
                signal_price, fill_price, slippage_bps, fill_status, fill_pct,
                cost_budgeted_bps, cost_realised_bps, signal_date, entry_date,
                exit_date, holding_period, iv_at_entry, theta_at_entry, basis_at_entry,
                net_return_pct, gross_return_pct, regime_at_entry, regime_at_exit,
                synthetic_pair_leg_consistency, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trade_id,
                outcome.get('edge_id'),
                outcome.get('instrument_class'),
                outcome.get('underlying'),
                outcome.get('contract_spec'),
                outcome.get('signal_price'),
                outcome.get('fill_price'),
                outcome.get('slippage_bps'),
                outcome.get('fill_status', 'FILLED'),
                outcome.get('fill_pct', 1.0),
                outcome.get('cost_budgeted_bps'),
                outcome.get('cost_realised_bps'),
                outcome.get('signal_date'),
                outcome.get('entry_date'),
                outcome.get('exit_date'),
                outcome.get('holding_period'),
                outcome.get('iv_at_entry'),
                outcome.get('theta_at_entry'),
                outcome.get('basis_at_entry'),
                outcome.get('net_return_pct'),
                outcome.get('gross_return_pct'),
                outcome.get('regime_at_entry'),
                outcome.get('regime_at_exit'),
                outcome.get('synthetic_pair_leg_consistency'),
                datetime.now().isoformat(),
            )
        )
        conn.commit()
        conn.close()

    def _update_edge_library(self, outcome: dict):
        """Update rolling live hit rate and win/loss counts."""
        edge_id = outcome['edge_id']
        is_win  = int(outcome.get('net_return_pct', 0) > 0)
        conn = sqlite3.connect(DB_PATH)
        if is_win:
            conn.execute(
                "UPDATE edge_library SET live_wins_in_regime = live_wins_in_regime + 1 WHERE edge_id=?",
                (edge_id,)
            )
        else:
            conn.execute(
                "UPDATE edge_library SET live_losses_in_regime = live_losses_in_regime + 1 WHERE edge_id=?",
                (edge_id,)
            )

        row = conn.execute(
            "SELECT live_wins_in_regime, live_losses_in_regime FROM edge_library WHERE edge_id=?",
            (edge_id,)
        ).fetchone()
        if row:
            w, l = row
            total = w + l
            if total > 0:
                conn.execute(
                    "UPDATE edge_library SET live_hit_rate=?, last_active=? WHERE edge_id=?",
                    (w / total, datetime.now().date().isoformat(), edge_id)
                )
        conn.commit()
        conn.close()

    def _check_decay(self, edge_id: str):
        """Bayesian posterior update on win rate. Flag edge if decaying."""
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            """SELECT win_rate, live_wins_in_regime, live_losses_in_regime
               FROM edge_library WHERE edge_id=?""",
            (edge_id,)
        ).fetchone()
        if not row:
            conn.close()
            return

        historical_wr, wins, losses = row[0] or 0.5, row[1] or 0, row[2] or 0
        prior_alpha = historical_wr * 20
        prior_beta  = (1 - historical_wr) * 20
        posterior_wr = ((prior_alpha + wins)
                        / (prior_alpha + wins + prior_beta + losses))

        if posterior_wr < (historical_wr - 0.05) and (wins + losses) >= 20:
            conn.execute(
                "UPDATE edge_library SET win_rate_watch_flag=1, posterior_win_rate=? WHERE edge_id=?",
                (posterior_wr, edge_id)
            )
        if posterior_wr < (historical_wr - 0.08) and (wins + losses) >= 30:
            conn.execute(
                """UPDATE edge_library SET decay_flag=1, decay_cause='WIN_RATE_DECAY',
                   status='DECAYED' WHERE edge_id=?""",
                (edge_id,)
            )
        conn.commit()
        conn.close()

    def _write_replay_buffer(self, outcome: dict):
        """Add trade to RL replay buffer."""
        conn = sqlite3.connect(DB_PATH)
        state_features = {k: outcome.get(k) for k in [
            'z_21d', 'rsi_2', 'mom_5d', 'vol_expanding',
            'hmm_confidence', 'regime_encoded',
        ]}
        conn.execute(
            """INSERT OR IGNORE INTO rl_replay_buffer
               (transition_id, edge_id, instrument_class, holding_period,
                regime_at_trade, hmm_confidence_at_trade, regime_model_version,
                state_features_json, action, reward, synthetic_pricing_source,
                trade_date, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                outcome.get('edge_id'),
                outcome.get('instrument_class'),
                outcome.get('holding_period'),
                outcome.get('regime_at_entry'),
                outcome.get('hmm_confidence'),
                'v1',
                json.dumps(state_features),
                outcome.get('direction', 'LONG'),
                outcome.get('net_return_pct', 0),
                int(outcome.get('synthetic_pricing_used', 0)),
                outcome.get('signal_date'),
                datetime.now().isoformat(),
            )
        )
        conn.commit()
        conn.close()

    def _update_shadow_ledger_outcome(self, outcome: dict):
        """Mark the shadow ledger entry with the actual outcome."""
        signal_id = outcome.get('signal_id')
        if not signal_id:
            return
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """UPDATE shadow_ledger SET
               actual_return_pct=?, exit_date=?, exit_reason=?
               WHERE signal_id=?""",
            (
                outcome.get('net_return_pct'),
                outcome.get('exit_date'),
                outcome.get('exit_reason', 'HOLDING_PERIOD_COMPLETE'),
                signal_id,
            )
        )
        conn.commit()
        conn.close()

    def append_kairos_log(self, regime_data: dict):
        """Append-only. Never update existing rows."""
        conn = sqlite3.connect(KAIROS_DB_PATH)
        conn.execute("""
            INSERT INTO kairos_log VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            str(uuid.uuid4()),
            regime_data['date'],
            regime_data['regime_state'],
            regime_data['regime_confidence'],
            json.dumps(regime_data.get('hmm_probs', {})),
            int(regime_data['transition_warning_flag']),
            regime_data.get('vix_level', 0),
            regime_data.get('nifty_close', 0),
            'DERIVATIVES_MVP',
            'v1',
            datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()

    def _ensure_kairos_schema(self):
        """Create KAIROS log table if it doesn't exist."""
        conn = sqlite3.connect(KAIROS_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kairos_log (
                observation_id           TEXT PRIMARY KEY,
                date                     TEXT NOT NULL,
                regime_state             TEXT,
                regime_confidence        REAL,
                hmm_probs_json           TEXT,
                transition_warning_flag  INTEGER,
                vix_level                REAL,
                nifty_close              REAL,
                instrument_class_context TEXT,
                regime_model_version     TEXT,
                created_at               TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
