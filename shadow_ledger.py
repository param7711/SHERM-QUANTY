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

from config import DB_PATH


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
               (signal_id, edge_id, instrument_class, underlying, signal_date,
                trigger_condition, regime_at_signal, hmm_confidence_at_signal,
                ai2_decision, ai2_rejection_reason, holding_period,
                feature_values_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal['signal_id'], signal['edge_id'], signal['instrument_class'],
                signal['underlying'], signal['signal_time'][:10],
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
        For all signals: compute what the return would have been from the underlying.
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
        Compute hypothetical outcome for one shadow signal.
        Uses underlying's forward return over holding_period as proxy for option P&L.
        """
        signal_date   = pd.Timestamp(row['signal_date'])
        holding       = int(row['holding_period'])
        exit_target   = signal_date + timedelta(days=int(holding * 1.5))
        exit_date_ts  = min(pd.Timestamp(as_of_date), exit_target)

        # Determine raw instrument name for price lookup
        underlying = row['underlying']
        if underlying == 'ALL_INDICES':
            underlying = 'NIFTY'

        raw_map = {
            'GOLD': 'GOLD_USD', 'SILVER': 'SILVER_USD',
            'CRUDEOIL': 'CRUDE_USD', 'COPPER': 'COPPER_USD',
        }
        inst = raw_map.get(underlying, underlying)
        path = f'data/processed/{inst}_features.parquet'
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

        # Apply direction
        direction = row.get('ai2_decision', 'LONG')
        if direction in ('REJECTED', 'PENDING', None):
            direction = 'LONG'
        if 'SHORT' in str(direction):
            raw_return = -raw_return

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
    print("=== Step 9 — Shadow Ledger verification ===\n")

    sl = ShadowLedger()

    # Log a mock signal
    mock_signal = {
        'signal_id':             'TEST_SL_001',
        'edge_id':               'SEED-001',
        'instrument_class':      'INDEX_OPTIONS',
        'underlying':            'NIFTY',
        'regime_at_signal':      'SIDEWAYS',
        'hmm_confidence_at_signal': 0.62,
        'transition_warning_flag':  False,
        'trigger_feature':       'z_21d',
        'trigger_value':         -2.3,
        'holding_period':        5,
        'signal_time':           datetime.now().isoformat(),
        'feature_snapshot':      {'z_21d': -2.3, 'rsi_2': 8.5},
    }

    sl.log_signal(mock_signal, 'APPROVED')

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT signal_id, ai2_decision FROM shadow_ledger WHERE signal_id='TEST_SL_001'"
    ).fetchone()
    count = conn.execute("SELECT COUNT(*) FROM shadow_ledger").fetchone()[0]
    conn.close()

    print(f"  Signal logged: {row}")
    print(f"  Total shadow ledger rows: {count}")

    # Update decision
    sl.update_ai2_decision('TEST_SL_001', 'REJECTED', 'TRANSITION_WATCH_ACTIVE')
    conn = sqlite3.connect(DB_PATH)
    row2 = conn.execute(
        "SELECT ai2_decision, ai2_rejection_reason FROM shadow_ledger WHERE signal_id='TEST_SL_001'"
    ).fetchone()
    conn.close()
    print(f"  After decision update: {row2}")

    if row and row[1] == 'APPROVED' and row2[0] == 'REJECTED':
        print("\n  PASS — Shadow Ledger log and update work correctly.")
    else:
        print("\n  FAIL — Shadow Ledger operation failed.")

    print("\n=== Step 9 complete ===")


if __name__ == '__main__':
    _verification_check()
