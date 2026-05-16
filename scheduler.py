"""
Sherm Quanty — Live Scheduler.
AI 1B scans continuously during market hours (every 5 minutes by default).
Signals are deduplicated: same edge+underlying+direction fires at most once per day.
Daily-once tasks (regime update, meta-labeler retrain, daily reset) are gated by flags.

NSE session:  09:15–15:30 IST, Mon–Fri
MCX evening:  09:00–23:30 IST, Mon–Fri (commodity positions monitored throughout)
Requires real Zerodha credentials in .env for live order routing.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime, timezone, timedelta

from config import (
    TOTAL_CAPITAL, DB_PATH,
    ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN,
    LIVE_SCAN_INTERVAL_SECS,
    NSE_OPEN_TIME, NSE_CLOSE_TIME,
    MCX_OPEN_TIME, MCX_CLOSE_TIME,
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
from rl.bayesian_optimizer import BayesianThresholdOptimizer

# ---------------------------------------------------------------------------
# IST timezone (UTC+5:30 — no external tz library needed)
# ---------------------------------------------------------------------------

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(_IST)

def _parse_time(hhmm: str):
    h, m = map(int, hhmm.split(':'))
    return h * 60 + m   # minutes since midnight

_NSE_OPEN  = _parse_time(NSE_OPEN_TIME)
_NSE_CLOSE = _parse_time(NSE_CLOSE_TIME)
_MCX_OPEN  = _parse_time(MCX_OPEN_TIME)
_MCX_CLOSE = _parse_time(MCX_CLOSE_TIME)


def _minutes_since_midnight(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def _is_nse_open(dt: datetime) -> bool:
    if dt.weekday() >= 5:   # Saturday / Sunday
        return False
    m = _minutes_since_midnight(dt)
    return _NSE_OPEN <= m <= _NSE_CLOSE


def _is_mcx_open(dt: datetime) -> bool:
    if dt.weekday() >= 5:
        return False
    m = _minutes_since_midnight(dt)
    return _MCX_OPEN <= m <= _MCX_CLOSE


# ---------------------------------------------------------------------------
# Shared state — process-level singletons
# ---------------------------------------------------------------------------

slot_system    = SlotSystem()
risk_governor  = RiskGovernor()

# RL singletons — load persisted state from disk if available
bayesian_opt   = BayesianThresholdOptimizer()
meta_labeler   = XGBMetaLabeler()
bandit         = ContextualBandit()

# Agents — RL objects injected so all components share the same instances
ai1b           = AI1B_Spotter(live_mode=True, bayesian_optimizer=bayesian_opt)
ai2            = AI2_Validator(meta_labeler=meta_labeler)
ai3            = AI3_Scorer(bandit=bandit)
ai4            = AI4_Executor()
ai5            = AI5_FeedbackLoop(bandit=bandit, bayesian_optimizer=bayesian_opt)
portfolio_ctor = PortfolioConstruction()


# ---------------------------------------------------------------------------
# Signal deduplication — prevents re-entering same position intraday
# ---------------------------------------------------------------------------

_signals_today: set = set()         # {(edge_id, underlying, direction)}
_dedup_date: str    = ''            # ISO date string of last reset


def _dedup_signals(signals: list) -> list:
    """
    Remove any signal whose (edge_id, underlying, direction) tuple has already
    been seen today (either in a prior scan or earlier in this batch).
    Keys are marked immediately so duplicates within the same batch are also
    filtered.  Resets automatically at midnight IST.
    """
    global _signals_today, _dedup_date
    today = _now_ist().date().isoformat()
    if _dedup_date != today:
        _signals_today = set()
        _dedup_date    = today

    new_signals = []
    for sig in signals:
        key = (sig['edge_id'], sig['underlying'], sig['direction'])
        if key not in _signals_today:
            new_signals.append(sig)
            _signals_today.add(key)   # mark immediately — same key won't pass again today
    return new_signals


# ---------------------------------------------------------------------------
# Daily-once task flags (keyed by ISO date string)
# ---------------------------------------------------------------------------

_regime_updated_for:  str = ''
_meta_retrained_for:  str = ''
_daily_reset_done_for: str = ''


def _get_trade_history():
    import pandas as pd, sqlite3
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM execution_quality_log", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def _run_morning_regime(today: str):
    """Run once per day on first scan after NSE open."""
    global _regime_updated_for
    if _regime_updated_for == today:
        return
    update_regime()
    regime = get_current_regime()
    ai5.append_kairos_log({
        'date':                    today,
        'regime_state':            regime['regime_state'],
        'regime_confidence':       regime['regime_confidence'],
        'transition_warning_flag': regime['transition_warning_flag'],
    })
    _regime_updated_for = today
    print(f"  [regime] {regime['regime_state']} "
          f"({regime['regime_confidence']:.2%} conf, TWF={regime['transition_warning_flag']})")


def _run_friday_retrain(today: str, weekday: int):
    """Retrain meta-labeler once on Friday."""
    global _meta_retrained_for
    if weekday != 4 or _meta_retrained_for == today:
        return
    trade_history = _get_trade_history()
    if not trade_history.empty:
        meta_labeler.retrain(trade_history)
        print(f"  [meta-labeler] retrained on {len(trade_history)} trades")
    _meta_retrained_for = today


def _run_daily_reset(today: str):
    """Reset risk governor counters once after NSE close."""
    global _daily_reset_done_for
    if _daily_reset_done_for == today:
        return
    risk_governor.daily_reset()
    _daily_reset_done_for = today
    print(f"  [risk] daily reset done")


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _execute_pipeline(signals: list, tag: str):
    """
    Run AI 2 → portfolio construction → AI 3 → AI 4 for a batch of signals.
    bandit_action and bandit_context are stored in the slot so AI 5 can update
    the bandit when the trade closes.
    Returns count of fills.
    """
    if not signals:
        return 0

    approved = ai2.validate(signals)
    selected = portfolio_ctor.select_trades(approved, slot_system)
    sized    = [ai3.score_and_size(s, TOTAL_CAPITAL) for s in selected]

    fills = 0
    for signal in sized:
        if signal.get('recommended_lots', 1) == 0:
            continue   # bandit said SKIP
        result = ai4.execute_signal(signal)
        status = result.get('status', '')
        if status in ('FILLED', 'SIMULATED_FILL'):
            # Carry bandit context into the slot for later RL update
            slot_data = {
                **signal,
                'bandit_action':  signal.get('bandit_action', 'MEDIUM'),
                'bandit_context': signal.get('bandit_context', []),
            }
            slot_system.add_slot(result['order_id'], slot_data)
            risk_governor.record_position_opened(result['order_id'], signal)
            fills += 1

    if sized:
        print(f"  [{tag}] {len(signals)} raw → {len(approved)} approved → "
              f"{len(sized)} sized → {fills} filled")
    return fills


# ---------------------------------------------------------------------------
# Main live loop
# ---------------------------------------------------------------------------

def run_live():
    print("Sherm Quanty live scanner started.")
    print(f"Zerodha API configured: {bool(ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN)}")
    print(f"Scan interval: {LIVE_SCAN_INTERVAL_SECS}s  "
          f"NSE: {NSE_OPEN_TIME}–{NSE_CLOSE_TIME}  "
          f"MCX: {MCX_OPEN_TIME}–{MCX_CLOSE_TIME} IST")

    while True:
        now   = _now_ist()
        today = now.date().isoformat()
        ts    = now.strftime('%Y-%m-%d %H:%M')

        # ── NSE session ───────────────────────────────────────────────────
        if _is_nse_open(now):
            print(f"[{ts}] NSE scan")

            _run_morning_regime(today)

            # Monitor open positions first
            try:
                open_positions = list(slot_system.active_slots.values())
                if open_positions:
                    ai5.monitor_positions(open_positions, {'source': 'live_scan'})
            except Exception as e:
                print(f"[{ts}] WARN monitor_positions: {type(e).__name__}: {e}")

            # Scan → deduplicate → execute
            try:
                signals  = ai1b.scan()
                new_sigs = _dedup_signals(signals)
                _execute_pipeline(new_sigs, 'nse')
            except Exception as e:
                print(f"[{ts}] WARN pipeline: {type(e).__name__}: {e}")

            # Pre-close: Friday meta-labeler retrain (only after 15:00)
            if _minutes_since_midnight(now) >= _parse_time('15:00'):
                _run_friday_retrain(today, now.weekday())

        # ── Post-close daily reset (once, between 15:30 and 23:30) ────────
        elif (_minutes_since_midnight(now) > _NSE_CLOSE
              and _minutes_since_midnight(now) < _parse_time('23:30')
              and now.weekday() < 5):
            _run_daily_reset(today)

            # MCX evening: commodity positions still open
            if _is_mcx_open(now):
                commodity_positions = [
                    p for p in slot_system.active_slots.values()
                    if p.get('instrument_class') == 'COMMODITY_FUTURES'
                ]
                if commodity_positions:
                    print(f"[{ts}] MCX evening check — {len(commodity_positions)} positions")
                    try:
                        ai5.monitor_positions(commodity_positions, {'source': 'mcx_evening'})
                    except Exception as e:
                        print(f"[{ts}] WARN mcx_monitor: {type(e).__name__}: {e}")

        time.sleep(LIVE_SCAN_INTERVAL_SECS)


if __name__ == '__main__':
    run_live()
