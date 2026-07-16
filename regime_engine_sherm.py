"""
Sherm Quanty — Regime Engine (facade)

Unified 2-tier regime API. This file is a thin orchestrator; the two engines
it merges live in their own modules:

  * regime_engine_macro.py    — DAILY macro climate (BULL / NON_BULL),
                                 ported from VortexMomentum. Slow and stable.
  * regime_engine_tactical.py — 2-HOUR tactical swing regime (H_BULL / L_BULL /
                                 SIDEWAYS / L_BEAR / H_BEAR). Fast; the actionable
                                 engine for 2–15 day trades. (Repurposed from the
                                 original daily Sherm HMM that used to live here.)

Design principle:
  Daily regime = macro climate (slow/stable, from VortexMomentum).
  2h regime    = actionable swing-trading regime (fast, Sherm HMM adapted).
  30m          = entry/execution refinement only, NEVER regime detection.

Backward compatibility:
  Every caller keeps `from regime_engine_sherm import get_current_regime`.
  The returned dict still contains regime_state / regime_confidence /
  transition_warning_flag. When 2h data is available those aliases reflect the
  TACTICAL regime; otherwise they fall back to the DAILY macro regime.
  get_regime_on_date() also still returns hmm_state_int / prob_H_BULL /
  prob_L_BULL / prob_SIDEWAYS / prob_L_BEAR / prob_H_BEAR (as in the original
  standalone HMM engine) — sourced from the tactical HMM row when available,
  else None.
"""

import pandas as pd

from regime_engine_macro import (
    get_daily_regime,
    get_daily_regime_on_date,
    update_daily_regime,
    build_macro_regime_history,
)
from regime_engine_tactical import (
    get_tactical_regime,
    get_tactical_regime_on_date,
    update_tactical_regime,
    build_tactical_regime_history,
    get_tactical_features,          # exposed for future AI 1A
    _load_parquet as _load_tactical_parquet,
)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge_regimes(daily: dict, tactical: dict | None) -> dict:
    """
    Combine daily macro + 2h tactical into the unified output contract.

    Backward-compatible aliases (regime_state / regime_confidence /
    transition_warning_flag) point at the tactical regime when 2h data is
    available, else at the daily macro regime. regime_timeframe records which.
    """
    if tactical is not None:
        alias_state = tactical['tactical_regime_state']
        alias_conf  = tactical['tactical_regime_confidence']
        alias_twf   = tactical['tactical_transition_warning_flag']
        timeframe   = '2h'
    else:
        alias_state = daily['daily_regime_state']
        alias_conf  = daily['daily_regime_confidence']
        alias_twf   = daily['daily_transition_warning_flag']
        timeframe   = '1d'

    return {
        # --- backward-compatible aliases (never None) ---
        'regime_state':                     alias_state,
        'regime_confidence':                alias_conf,
        'transition_warning_flag':          alias_twf,
        # --- daily macro (always present) ---
        'daily_regime_state':               daily['daily_regime_state'],
        'daily_regime_confidence':          daily['daily_regime_confidence'],
        'daily_transition_warning_flag':    daily['daily_transition_warning_flag'],
        # --- 2h tactical (None when unavailable) ---
        'tactical_regime_state':            tactical['tactical_regime_state']            if tactical else None,
        'tactical_regime_confidence':       tactical['tactical_regime_confidence']       if tactical else None,
        'tactical_transition_warning_flag': tactical['tactical_transition_warning_flag'] if tactical else None,
        # --- metadata ---
        'regime_timeframe':                 timeframe,
    }


def _tactical_hmm_detail(date: str | None = None) -> dict:
    """
    hmm_state_int / prob_* columns from the tactical HMM row, preserved for
    backward compatibility with the original standalone HMM engine's
    get_regime_on_date() contract. None values when tactical data is absent.
    """
    empty = {
        'hmm_state_int': None,
        'prob_H_BULL':   None, 'prob_L_BULL':   None, 'prob_SIDEWAYS': None,
        'prob_L_BEAR':   None, 'prob_H_BEAR':   None,
    }
    try:
        df = _load_tactical_parquet()
    except FileNotFoundError:
        return empty
    if len(df) == 0:
        return empty

    if date is None:
        row = df.iloc[-1]
    else:
        idx = pd.Timestamp(date)
        prior = df.index[df.index <= idx] if idx not in df.index else [idx]
        if len(prior) == 0:
            return empty
        row = df.loc[prior[-1]] if idx not in df.index else df.loc[idx]

    return {
        'hmm_state_int': int(row['hmm_state_int']),
        'prob_H_BULL':   float(row['prob_H_BULL']),
        'prob_L_BULL':   float(row['prob_L_BULL']),
        'prob_SIDEWAYS': float(row['prob_SIDEWAYS']),
        'prob_L_BEAR':   float(row['prob_L_BEAR']),
        'prob_H_BEAR':   float(row['prob_H_BEAR']),
    }


# ---------------------------------------------------------------------------
# Public API — signatures unchanged from the original engine
# ---------------------------------------------------------------------------

def get_current_regime() -> dict:
    """Today's regime. Used by AI 1B, Risk Governor, Scheduler."""
    tactical = get_tactical_regime()          # None if 2h unavailable
    daily    = get_daily_regime()
    return _merge_regimes(daily, tactical)


def get_regime_on_date(date: str) -> dict:
    """Regime for a specific historical date. Used by the backtest."""
    daily    = get_daily_regime_on_date(date)
    tactical = get_tactical_regime_on_date(date)   # None if that date absent
    merged   = _merge_regimes(daily, tactical)
    merged.update(_tactical_hmm_detail(date))
    return merged


def update_regime():
    """
    Daily refresh (called ~09:30 IST by the scheduler).
    Updates both engines; the tactical update is a graceful no-op when 2h data
    is unavailable.
    """
    update_daily_regime()
    update_tactical_regime()


def build_regime_history():
    """Rebuild both engines from scratch."""
    print("=== Sherm Quanty Regime Engine — Build (2-tier) ===\n")
    print("[1/2] Daily macro engine ...")
    build_macro_regime_history()
    print("\n[2/2] 2h tactical engine ...")
    tactical = build_tactical_regime_history()
    if tactical is None:
        print("\n  2h tactical unavailable — system will run on daily macro regime.")
    print("\n=== Build complete ===")


# ---------------------------------------------------------------------------
# Diagnostics — daily vs tactical detection lag on high-opportunity phases
# ---------------------------------------------------------------------------

BENCHMARK_EVENTS = [
    ('2020 COVID crash',    '2020-02-15', '2020-03-23', 'H_BEAR'),
    ('2020 recovery',       '2020-03-24', '2020-12-31', 'H_BULL'),
    ('2021 bull',           '2021-01-01', '2021-10-31', 'H_BULL'),
    ('2022 bear/rate-hike', '2021-11-01', '2022-06-30', 'H_BEAR'),
    ('2024 rally',          '2024-01-01', '2024-05-31', 'H_BULL'),
]


def print_regime_diagnostics():
    """
    Print daily and tactical regime distributions separately, plus a detection
    lag comparison over documented high-opportunity phases. Goal: the tactical
    (2h) engine should flag H_BULL/H_BEAR EARLIER than the daily macro engine.
    """
    from regime_engine_macro    import _load_parquet as _load_macro
    from regime_engine_tactical import _load_parquet as _load_tactical
    import os

    print("=" * 68)
    print("REGIME DIAGNOSTICS")
    print("=" * 68)

    # --- distributions ---
    macro = _load_macro()
    print("\nDaily macro distribution (BULL / NON_BULL):")
    print(macro['daily_regime_state'].value_counts().to_string())

    if os.path.exists('data/processed/tactical_regime_history.parquet'):
        tac = _load_tactical()
        print("\nTactical 2h distribution (5-state):")
        order = ['H_BULL', 'L_BULL', 'SIDEWAYS', 'L_BEAR', 'H_BEAR']
        print(tac['tactical_regime_state'].value_counts().reindex(order).to_string())
    else:
        tac = None
        print("\nTactical 2h: unavailable (no parquet).")

    # --- detection-lag table ---
    print("\n" + "-" * 68)
    print(f"{'Event':<22}{'target':<9}{'daily 1st':<13}{'tactical 1st':<15}{'lag(d)':<7}")
    print("-" * 68)
    for name, start, end, target in BENCHMARK_EVENTS:
        # Daily macro is BULL/NON_BULL; map an H_BULL/H_BEAR target onto it:
        #   H_BULL  → first BULL day in window
        #   H_BEAR  → first NON_BULL day in window
        daily_target = 'BULL' if target == 'H_BULL' else 'NON_BULL'
        d_first = _first_hit(macro, 'daily_regime_state', daily_target, start, end)

        if tac is not None:
            t_first = _first_hit(tac, 'tactical_regime_state', target, start, end)
        else:
            t_first = None

        lag = ''
        if d_first is not None and t_first is not None:
            lag = f"{(d_first - t_first).days:+d}"   # positive = tactical earlier
        print(f"{name:<22}{target:<9}"
              f"{str(d_first.date()) if d_first is not None else '—':<13}"
              f"{str(t_first.date()) if t_first is not None else '—':<15}"
              f"{lag:<7}")
    print("-" * 68)
    print("lag(d) > 0 means the tactical engine detected the phase earlier.")


def _first_hit(df: pd.DataFrame, col: str, target: str,
               start: str, end: str):
    """First index in [start, end] where df[col] == target, else None."""
    window = df.loc[(df.index >= pd.Timestamp(start)) &
                    (df.index <= pd.Timestamp(end))]
    hits = window.index[window[col] == target]
    return hits[0] if len(hits) > 0 else None


if __name__ == '__main__':
    build_regime_history()
    print()
    print("Current regime:", get_current_regime())
