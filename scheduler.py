"""
Sherm Quanty — Main Scheduler.
Run this file to start live trading.
Executes the 3x daily pipeline + commodity evening check.
Requires real Zerodha credentials in .env.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import schedule
from datetime import datetime

from config import (
    TOTAL_CAPITAL, DB_PATH,
    ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN,
)
from regime_engine_sherm import get_current_regime, update_regime
from agents.ai1b_spotter import AI1B_Spotter
from agents.ai2_validator import AI2_Validator
from agents.ai3_scorer import AI3_Scorer
from agents.ai4_executor import AI4_Executor
from agents.ai5_feedback import AI5_FeedbackLoop
from portfolio_construction import PortfolioConstruction
from slot_system import SlotSystem
from risk_governor import RiskGovernor
from rl.xgb_meta_labeler import XGBMetaLabeler
from rl.contextual_bandit import ContextualBandit

# ---------------------------------------------------------------------------
# Shared state (process-level singletons)
# ---------------------------------------------------------------------------

slot_system    = SlotSystem()
risk_governor  = RiskGovernor()
ai1b           = AI1B_Spotter()
ai2            = AI2_Validator()
ai3            = AI3_Scorer()
ai4            = AI4_Executor()
ai5            = AI5_FeedbackLoop()
portfolio_ctor = PortfolioConstruction()
meta_labeler   = XGBMetaLabeler()
bandit         = ContextualBandit()


def _get_trade_history():
    import pandas as pd, sqlite3
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM execution_quality_log", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Scheduled jobs
# ---------------------------------------------------------------------------

def morning_scan():
    """9:30 AM IST — post-open."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] morning_scan start")
    update_regime()
    regime = get_current_regime()
    ai5.append_kairos_log({
        'date':                  datetime.now().date().isoformat(),
        'regime_state':          regime['regime_state'],
        'regime_confidence':     regime['regime_confidence'],
        'transition_warning_flag': regime['transition_warning_flag'],
    })

    signals = ai1b.scan()
    approved = ai2.validate(signals)
    selected = portfolio_ctor.select_trades(approved, slot_system)
    sized    = [ai3.score_and_size(s, TOTAL_CAPITAL) for s in selected]

    for signal in sized:
        result = ai4.execute_signal(signal)
        if result.get('status') in ('FILLED', 'SIMULATED_FILL'):
            slot_system.add_slot(result['order_id'], signal)
            risk_governor.record_position_opened(result['order_id'], signal)

    print(f"[{datetime.now():%H:%M}] morning_scan: {len(signals)} signals, "
          f"{len(approved)} approved, {len(sized)} sized")


def midday_scan():
    """1:00 PM IST — midday check."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] midday_scan start")
    ai5.monitor_positions(list(slot_system.active_slots.values()), {})

    signals  = ai1b.scan()
    approved = ai2.validate(signals)
    selected = portfolio_ctor.select_trades(approved, slot_system)
    sized    = [ai3.score_and_size(s, TOTAL_CAPITAL) for s in selected]

    for signal in sized:
        ai4.execute_signal(signal)

    print(f"[{datetime.now():%H:%M}] midday_scan done")


def preclose_scan():
    """3:20 PM IST — pre-close."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] preclose_scan start")
    ai5.monitor_positions(list(slot_system.active_slots.values()), {})

    # Retrain meta-labeler weekly (Friday only)
    if datetime.now().weekday() == 4:
        trade_history = _get_trade_history()
        if not trade_history.empty:
            meta_labeler.retrain(trade_history)
            print(f"  Meta-labeler retrained on {len(trade_history)} trades")

    risk_governor.daily_reset()
    print(f"[{datetime.now():%H:%M}] preclose_scan done")


def commodity_evening_check():
    """11:00 PM IST — MCX evening session check."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] commodity_evening_check start")
    commodity_positions = [
        p for p in slot_system.active_slots.values()
        if p.get('instrument_class') == 'COMMODITY_FUTURES'
    ]
    for pos in commodity_positions:
        ai5.monitor_positions([pos], {'source': 'commodity_evening'})


# ---------------------------------------------------------------------------
# Register schedules
# ---------------------------------------------------------------------------

schedule.every().day.at('09:30').do(morning_scan)
schedule.every().day.at('13:00').do(midday_scan)
schedule.every().day.at('15:20').do(preclose_scan)
schedule.every().day.at('23:00').do(commodity_evening_check)


if __name__ == '__main__':
    print("Sherm Quanty scheduler started.")
    print(f"Zerodha API configured: {bool(ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN)}")
    print("Waiting for schedule times: 09:30, 13:00, 15:20, 23:00 IST")
    while True:
        schedule.run_pending()
        time.sleep(30)
