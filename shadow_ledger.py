"""
Step 9 — Shadow Ledger.
Records every signal AI 1B fires, regardless of whether AI 2 approves it.
At holding period end, AI 5 computes the counterfactual outcome.
Critical for separating edge decay from AI 2 miscalibration.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

from config import DB_PATH, MAX_SPREAD_PCT


class ShadowLedger:
    """
    Records every signal AI 1B fires, regardless of whether AI 2 approves it.
    At holding period end, AI 5 computes what the outcome would have been.
    Critical for separating edge decay from AI 2 miscalibration.
    """

    def log_signal(self, signal: dict, ai2_decision: str, rejection_reason: str = None):
        """Call immediately after AI 2 makes its decision."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT OR IGNORE INTO shadow_ledger
               (signal_id, edge_id, pair, direction, signal_date,
                trigger_condition, regime_at_signal, hmm_confidence_at_signal,
                ai2_decision, ai2_rejection_reason, holding_period,
                feature_values_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal['signal_id'], signal['edge_id'], signal['pair'],
                signal['direction'], signal['signal_time'][:10],
                str(signal['trigger_value']), signal['regime_at_signal'],
                signal['hmm_confidence_at_signal'], ai2_decision, rejection_reason,
                signal['holding_period'], json.dumps(signal.get('feature_snapshot', {})),
                datetime.now().isoformat(),
            )
        )
        conn.commit()
        conn.close()

    def update_ai2_decision(self, signal_id: str, ai2_decision: str,
                             rejection_reason: str = None):
        """Update the AI 2 decision for an already-logged signal."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """UPDATE shadow_ledger SET ai2_decision=?, ai2_rejection_reason=?
               WHERE signal_id=?""",
            (ai2_decision, rejection_reason, signal_id)
        )
        conn.commit()
        conn.close()

    def compute_shadow_outcomes(self, as_of_date: str):
        """
        AI 5 calls this at end of each holding period.
        For all signals: compute what the return would have been on the pair.
        Update shadow_outcome and shadow_return_pct.
        """
        conn = sqlite3.connect(DB_PATH)
        pending = pd.read_sql(
            """SELECT * FROM shadow_ledger
               WHERE shadow_outcome IS NULL AND signal_date <= ?""",
            conn, params=(as_of_date,)
        )
        conn.close()

        if pending.empty:
            return 0

        updated = 0
        for _, row in pending.iterrows():
            result = self._compute_one_outcome(row, as_of_date)
            if result is None:
                continue
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                """UPDATE shadow_ledger
                   SET shadow_outcome=?, shadow_return_pct=?, exit_date=?, exit_reason=?
                   WHERE signal_id=?""",
                (result['outcome'], result['return_pct'], result['exit_date'],
                 result['exit_reason'], row['signal_id'])
            )
            conn.commit()
            conn.close()
            updated += 1

        return updated

    def _compute_one_outcome(self, row: pd.Series, as_of_date: str) -> dict:
        """
        Compute hypothetical outcome for one shadow signal, as the pair's
        forward return over the holding period, net of the round-trip spread.
        """
        signal_date   = pd.Timestamp(row['signal_date'])
        holding       = int(row['holding_period'])
        # Holding periods are in sessions; x1.5 approximates the calendar
        # span including weekends.
        exit_target   = signal_date + timedelta(days=int(holding * 1.5))
        exit_date_ts  = min(pd.Timestamp(as_of_date), exit_target)

        pair = row['pair']
        path = f'data/processed/{pair}_features.parquet'
        if not os.path.exists(path):
            return None

        try:
            feat = pd.read_parquet(path)[['close']]
            feat.index = pd.to_datetime(feat.index)
        except Exception:
            return None

        if signal_date not in feat.index:
            idx = feat.index.get_indexer([signal_date], method='ffill')[0]
            if idx < 0:
                return None
            signal_date = feat.index[idx]

        if exit_date_ts not in feat.index:
            idx = feat.index.get_indexer([exit_date_ts], method='ffill')[0]
            if idx < 0:
                return None
            exit_date_ts = feat.index[idx]

        entry_price = float(feat.loc[signal_date, 'close'])
        exit_price  = float(feat.loc[exit_date_ts, 'close'])

        if entry_price == 0:
            return None

        raw_return = (exit_price - entry_price) / entry_price

        # Direction comes from its own column. Reading it from ai2_decision
        # (APPROVED/REJECTED) silently scored every short as a long.
        direction = str(row.get('direction') or 'LONG').upper()
        if direction == 'SHORT':
            raw_return = -raw_return

        # Net of the round-trip spread. On a 2-day hold this is a large
        # fraction of the edge, and omitting it makes every shadow outcome
        # look better than the live trade would have been.
        raw_return -= MAX_SPREAD_PCT

        outcome    = 'WIN' if raw_return > 0 else 'LOSS'
        return_pct = raw_return * 100

        return {
            'outcome':    outcome,
            'return_pct': return_pct,
            'exit_date':  str(exit_date_ts.date()),
            'exit_reason': 'HOLDING_PERIOD_COMPLETE',
        }

    def get_rejection_analysis(self) -> pd.DataFrame:
        """Return summary of AI 2 rejection rates and shadow outcomes by edge."""
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM shadow_ledger", conn)
        conn.close()
        if df.empty:
            return df
        summary = df.groupby(['edge_id', 'ai2_decision']).agg(
            count=('signal_id', 'count'),
            win_rate=('shadow_outcome', lambda x: (x == 'WIN').mean()),
        ).reset_index()
        return summary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== Step 8 — Shadow Ledger verification (4 tests) ===\n")

    import tempfile
    import shutil
    import unittest.mock as mock

    passed = 0
    tmpdir = tempfile.mkdtemp(prefix='shadow_ledger_test_')
    test_db = os.path.join(tmpdir, 'test.db')

    try:
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'database', 'schema.sql')
        conn = sqlite3.connect(test_db)
        with open(schema_path) as f:
            conn.executescript(f.read())
        conn.commit()
        conn.close()

        # Patch the module-level DB_PATH so the test never touches the
        # production database. The previous version left a permanent
        # TEST_SL_001 row behind on every run.
        with mock.patch.object(sys.modules[__name__], 'DB_PATH', test_db):
            sl = ShadowLedger()

            mock_signal = {
                'signal_id':                'TEST_SL_001',
                'edge_id':                  'SEED-001',
                'pair':                     'EURUSD',
                'direction':                'SHORT',
                'regime_at_signal':         'SIDEWAYS',
                'hmm_confidence_at_signal': 0.62,
                'trigger_feature':          'z_21d',
                'trigger_value':            2.3,
                'holding_period':           5,
                'signal_time':              datetime.now().isoformat(),
                'feature_snapshot':         {'z_21d': 2.3, 'rsi_2': 91.5},
            }
            sl.log_signal(mock_signal, 'APPROVED')

            conn = sqlite3.connect(test_db)
            row = conn.execute(
                """SELECT signal_id, pair, direction, ai2_decision
                   FROM shadow_ledger WHERE signal_id='TEST_SL_001'"""
            ).fetchone()
            conn.close()

            result = row is not None and row[1] == 'EURUSD'
            print(f"  Test 1 — Signal logs with pair:           "
                  f"{'PASS' if result else 'FAIL'} ({row[1] if row else None})")
            passed += result

            # Direction must round-trip as its own column, not be inferred
            # from ai2_decision.
            result = row is not None and row[2] == 'SHORT'
            print(f"  Test 2 — Direction stored separately:     "
                  f"{'PASS' if result else 'FAIL'} ({row[2] if row else None})")
            passed += result

            sl.update_ai2_decision('TEST_SL_001', 'REJECTED', 'TRANSITION_WATCH_ACTIVE')
            conn = sqlite3.connect(test_db)
            row2 = conn.execute(
                """SELECT ai2_decision, direction FROM shadow_ledger
                   WHERE signal_id='TEST_SL_001'"""
            ).fetchone()
            conn.close()

            result = row2[0] == 'REJECTED' and row2[1] == 'SHORT'
            print(f"  Test 3 — Decision update preserves dir:   "
                  f"{'PASS' if result else 'FAIL'} ({row2})")
            passed += result

        # A short signal on a rising market must score a LOSS. Under the old
        # code direction came from ai2_decision and this returned WIN.
        rising = pd.Series({
            'signal_date':    '2024-01-01',
            'holding_period': 5,
            'pair':           'EURUSD',
            'direction':      'SHORT',
        })
        feat = pd.DataFrame(
            {'close': [1.00, 1.05]},
            index=pd.to_datetime(['2024-01-01', '2024-01-08']))
        sl2 = ShadowLedger()
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('pandas.read_parquet', return_value=feat):
            outcome = sl2._compute_one_outcome(rising, '2024-01-08')

        result = outcome is not None and outcome['outcome'] == 'LOSS'
        print(f"  Test 4 — Short on rising market = LOSS:   "
              f"{'PASS' if result else 'FAIL'} "
              f"({outcome['outcome'] if outcome else None}, "
              f"{outcome['return_pct']:.2f}%)" if outcome else "(None)")
        passed += result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n  {passed}/4 tests passed.")
    print("  PASS — all 4 unit tests passed." if passed == 4
          else "  FAIL — some unit tests failed. See above.")
    print("\n=== Step 8 complete ===")


if __name__ == '__main__':
    _verification_check()
