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
            # Fallback dimensionality follows the live bandit's own feature
            # count, not a literal — a mismatch here shape-errors against
            # its A matrix rather than silently producing garbage.
            fallback = [0.0] * self.bandit.n_features
            context  = np.array(outcome.get('bandit_context') or fallback)
            reward   = float(outcome.get('net_return_pct', 0.0))
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
        """
        Record execution quality for one closed trade.

        Columns follow the v2-forex schema: pair/lot_size/direction and
        pip-denominated costs replace the options and futures fields
        (instrument_class, contract_spec, iv/theta/basis_at_entry,
        synthetic_pair_leg_consistency) that the derivatives schema carried.
        """
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT OR IGNORE INTO execution_quality_log
               (trade_id, edge_id, pair, lot_size, direction,
                signal_price, fill_price, slippage_pips, spread_at_entry_pips,
                swap_charged, session_at_entry, fill_status, fill_pct,
                cost_budgeted_bps, cost_realised_bps, signal_date, entry_date,
                exit_date, holding_period,
                net_return_pct, gross_return_pct, regime_at_entry, regime_at_exit,
                created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trade_id,
                outcome.get('edge_id'),
                outcome.get('pair'),
                outcome.get('lot_size'),
                outcome.get('direction'),
                outcome.get('signal_price'),
                outcome.get('fill_price'),
                outcome.get('slippage_pips'),
                outcome.get('spread_at_entry_pips'),
                outcome.get('swap_charged'),
                outcome.get('session_at_entry'),
                outcome.get('fill_status', 'FILLED'),
                outcome.get('fill_pct', 1.0),
                outcome.get('cost_budgeted_bps'),
                outcome.get('cost_realised_bps'),
                outcome.get('signal_date'),
                outcome.get('entry_date'),
                outcome.get('exit_date'),
                outcome.get('holding_period'),
                outcome.get('net_return_pct'),
                outcome.get('gross_return_pct'),
                outcome.get('regime_at_entry'),
                outcome.get('regime_at_exit'),
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
        """
        Add trade to RL replay buffer.

        Keys here must match rl.xgb_meta_labeler.XGBMetaLabeler.FEATURES
        exactly — load_training_frame() expands state_features_json
        straight into training columns by name. The original version
        snapshotted 6 of 14 names, so retraining silently trained on mostly
        missing (zero-filled) columns.

        The snapshot is derived from FEATURES rather than restated, so a
        feature added to the labeler cannot go uncaptured here.
        """
        from rl.xgb_meta_labeler import XGBMetaLabeler

        # A few features are known upstream under a different key.
        aliases = {
            'hmm_confidence':             'hmm_confidence_at_signal',
            'recent_win_rate_edge':       'recent_edge_win_rate',
            'time_since_last_trade_edge': 'days_since_last_trade',
            'spread_pips_at_signal':      'spread_at_entry_pips',
        }

        conn = sqlite3.connect(DB_PATH)
        state_features = {}
        for name in XGBMetaLabeler.FEATURES:
            value = outcome.get(name)
            if value is None and name in aliases:
                value = outcome.get(aliases[name])
            state_features[name] = value
        conn.execute(
            """INSERT OR IGNORE INTO rl_replay_buffer
               (transition_id, edge_id, pair, holding_period,
                regime_at_trade, hmm_confidence_at_trade, regime_model_version,
                state_features_json, action, reward,
                trade_date, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                outcome.get('edge_id'),
                outcome.get('pair'),
                outcome.get('holding_period'),
                outcome.get('regime_at_entry'),
                outcome.get('hmm_confidence'),
                'v1',
                json.dumps(state_features),
                outcome.get('direction', 'LONG'),
                outcome.get('net_return_pct', 0),
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
        """
        Append-only. Never update existing rows.

        vix_level and nifty_close stay: the regime engine still reads India
        VIX and Nifty as its macro risk-on/risk-off inputs, and those are
        what this row records. Only the scope tag changes.

        Columns are named explicitly rather than relying on positional
        VALUES — the insert must stay aligned with _ensure_kairos_schema
        below, and a positional insert breaks silently if either moves.
        """
        conn = sqlite3.connect(KAIROS_DB_PATH)
        conn.execute("""
            INSERT INTO kairos_log
                (observation_id, date, regime_state, regime_confidence,
                 hmm_probs_json, transition_warning_flag, vix_level,
                 nifty_close, instrument_class_context, regime_model_version,
                 created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            str(uuid.uuid4()),
            regime_data['date'],
            regime_data['regime_state'],
            regime_data['regime_confidence'],
            json.dumps(regime_data.get('hmm_probs', {})),
            int(regime_data['transition_warning_flag']),
            regime_data.get('vix_level', 0),
            regime_data.get('nifty_close', 0),
            'FOREX_MVP',
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


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== Step 12 — AI 5 Feedback Loop verification (6 tests) ===\n")
    import tempfile
    import shutil
    import unittest.mock as mock
    from database.init_db import ensure_schema
    from rl.xgb_meta_labeler import XGBMetaLabeler, load_training_frame

    passed = 0
    tmpdir = tempfile.mkdtemp(prefix='ai5_test_')
    try:
        test_db     = os.path.join(tmpdir, 'test.db')
        test_kairos = os.path.join(tmpdir, 'kairos.db')
        ensure_schema(test_db)

        # Seed one edge so the edge-library update has a row to touch.
        conn = sqlite3.connect(test_db)
        conn.execute(
            """INSERT INTO edge_library
               (edge_id, pair, pair_type, trigger_feature, trigger_condition,
                direction, holding_period, win_rate, regime, status,
                created_at, edge_provenance)
               VALUES ('SEED-001','EURUSD','MAJOR','z_21d','z<-2','LONG',5,
                       0.58,'SIDEWAYS','ACTIVE',datetime('now'),'SEEDED')""")
        conn.commit()
        conn.close()

        mod = sys.modules[__name__]
        with mock.patch.object(mod, 'DB_PATH', test_db), \
             mock.patch.object(mod, 'KAIROS_DB_PATH', test_kairos):

            ai5 = AI5_FeedbackLoop()

            outcome = {
                'edge_id': 'SEED-001', 'pair': 'EURUSD', 'direction': 'LONG',
                'lot_size': 0.35, 'holding_period': 5,
                'signal_price': 1.0850, 'fill_price': 1.0851,
                'slippage_pips': 1.0, 'spread_at_entry_pips': 1.2,
                'swap_charged': -0.45, 'session_at_entry': 'LONDON',
                'net_return_pct': 0.012, 'gross_return_pct': 0.014,
                'regime_at_entry': 'SIDEWAYS', 'regime_at_exit': 'SIDEWAYS',
                'signal_date': '2026-01-05', 'entry_date': '2026-01-05',
                'exit_date': '2026-01-10',
                'trigger_feature': 'z_21d', 'trigger_value': -2.3,
                'hmm_confidence_at_signal': 0.72, 'regime_encoded': 2,
                'vix_level': 14.2, 'recent_edge_win_rate': 0.6,
                'vol_expanding': 0, 'z_21d': -2.3, 'rsi_2': 8.0,
                'mom_5d': -0.004, 'day_of_week': 1,
                'days_since_last_trade': 3, 'carry_direction': -1,
            }

            # Test 1: the whole close path runs. Previously this raised
            # "no such column: instrument_class" on the first real trade.
            try:
                ai5.record_trade_outcome('T-001', outcome)
                ok, err = True, ''
            except Exception as e:
                ok, err = False, f'{type(e).__name__}: {e}'
            print(f"  Test 1 — record_trade_outcome runs:       "
                  f"{'PASS' if ok else 'FAIL'} {err}")
            passed += ok

            conn = sqlite3.connect(test_db)

            # Test 2: execution log wrote forex columns.
            row = conn.execute(
                """SELECT pair, lot_size, slippage_pips, swap_charged
                   FROM execution_quality_log WHERE trade_id='T-001'""").fetchone()
            result = row is not None and row[0] == 'EURUSD' and row[1] == 0.35
            print(f"  Test 2 — Execution log has forex cols:    "
                  f"{'PASS' if result else 'FAIL'} ({row})")
            passed += result

            # Test 3: edge library counted the win.
            row = conn.execute(
                "SELECT live_wins_in_regime, live_hit_rate FROM edge_library "
                "WHERE edge_id='SEED-001'").fetchone()
            result = row is not None and row[0] == 1
            print(f"  Test 3 — Edge library win recorded:       "
                  f"{'PASS' if result else 'FAIL'} (wins={row[0] if row else None})")
            passed += result
            conn.close()

            # Test 4: the replay-buffer snapshot covers every labeler
            # feature. A partial snapshot trains the model on zeros.
            frame = load_training_frame(test_db)
            missing = [f for f in XGBMetaLabeler.FEATURES if f not in frame.columns]
            populated = [f for f in XGBMetaLabeler.FEATURES
                         if f in frame.columns and frame[f].notna().all()]
            result = not missing and len(populated) == len(XGBMetaLabeler.FEATURES)
            print(f"  Test 4 — Replay snapshot complete:        "
                  f"{'PASS' if result else 'FAIL'} "
                  f"({len(populated)}/{len(XGBMetaLabeler.FEATURES)} populated, "
                  f"missing={missing})")
            passed += result

            # Test 5: KAIROS append works and carries the forex tag.
            ai5.append_kairos_log({
                'date': '2026-01-05', 'regime_state': 'SIDEWAYS',
                'regime_confidence': 0.55, 'hmm_probs': {'SIDEWAYS': 0.55},
                'transition_warning_flag': False,
                'vix_level': 14.2, 'nifty_close': 24000.0,
            })
            kconn = sqlite3.connect(test_kairos)
            krow = kconn.execute(
                "SELECT instrument_class_context, vix_level FROM kairos_log").fetchone()
            kconn.close()
            result = krow is not None and krow[0] == 'FOREX_MVP'
            print(f"  Test 5 — KAIROS tagged FOREX_MVP:         "
                  f"{'PASS' if result else 'FAIL'} ({krow})")
            passed += result

            # Test 6: the RL wiring accepts a feature name as the optimizer
            # param — the mapping that used to KeyError into a silent except.
            from rl.bayesian_optimizer import BayesianThresholdOptimizer
            state = os.path.join(tmpdir, 'bayes.json')
            with mock.patch('rl.bayesian_optimizer._STATE_PATH', state):
                opt = BayesianThresholdOptimizer()
                ai5_rl = AI5_FeedbackLoop(bayesian_optimizer=opt)
                ai5_rl._update_rl(outcome)
                params = {o['param'] for o in opt.observations}
            result = params == {'z_21d_threshold'}
            print(f"  Test 6 — RL param resolved from feature:  "
                  f"{'PASS' if result else 'FAIL'} ({params})")
            passed += result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n  {passed}/6 tests passed.")
    print("  PASS — all 6 unit tests passed." if passed == 6
          else "  FAIL — some unit tests failed. See above.")
    print("\n=== Step 12 complete ===")


if __name__ == '__main__':
    _verification_check()
