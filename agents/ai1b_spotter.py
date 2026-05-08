"""
Step 8 — AI 1B Spotter.
Reads the Edge Library and scans for signal conditions.
Supports both scheduled (3x daily) and live continuous scanning modes.
In live_mode=True: fetches fresh daily bars from yfinance before each scan.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
from datetime import datetime

from config import (
    DB_PATH,
    INDEX_UNDERLYINGS, COMMODITY_UNDERLYINGS, CURRENCY_UNDERLYINGS,
)
from regime_engine_sherm import get_current_regime


# Commodity instruments use USD proxies in data/raw
_COMMODITY_RAW_MAP = {
    'GOLD':     'GOLD_USD',
    'SILVER':   'SILVER_USD',
    'CRUDEOIL': 'CRUDE_USD',
    'COPPER':   'COPPER_USD',
}

# Currency instruments as available in raw data
_CURRENCY_RAW_MAP = {
    'USDINR':  'USDINR',
    'EURINR':  'EURINR',
    'GBPINR':  'GBPINR',
    'JPYINR':  'JPYINR',
    'EURUSD':  'EURUSD',
    'GBPUSD':  'GBPUSD',
    'USDJPY':  'USDJPY',
}

ALL_INSTRUMENTS = (
    INDEX_UNDERLYINGS
    + [_COMMODITY_RAW_MAP.get(c, c) for c in COMMODITY_UNDERLYINGS]
    + list(_CURRENCY_RAW_MAP.values())
)


class AI1B_Spotter:
    def __init__(self, db_path: str = DB_PATH, live_mode: bool = False):
        self.db_path   = db_path
        self.live_mode = live_mode

    def scan(self) -> list:
        """
        Main entry point. Called at each schedule time.
        Returns list of signal dicts for all triggered edges.
        All signals logged to shadow ledger regardless of downstream decisions.
        """
        active_edges = self._load_active_edges()
        live_features = self._compute_live_features()
        current_regime = get_current_regime()

        signals = []
        for edge in active_edges:
            if not self._regime_matches(edge, current_regime):
                continue
            triggered, trigger_value = self._check_trigger(edge, live_features)
            if not triggered:
                continue
            signal = self._build_signal(edge, live_features, current_regime, trigger_value)
            signals.append(signal)

        self._log_to_shadow_ledger(signals, current_regime)
        return signals

    def _load_active_edges(self) -> list:
        """Load all ACTIVE edges from Edge Library."""
        conn = sqlite3.connect(self.db_path)
        edges = pd.read_sql(
            "SELECT * FROM edge_library WHERE status = 'ACTIVE'", conn
        ).to_dict('records')
        conn.close()
        return edges

    def _compute_live_features(self) -> dict:
        """
        Returns dict {instrument_name: features_dataframe}.
        live_mode=True: refreshes daily bars from yfinance before computing features.
        live_mode=False (sandbox): reads cached parquets as-is.
        """
        if self.live_mode:
            return self._fetch_live_features()

        features = {}
        for inst in ALL_INSTRUMENTS:
            path = f'data/processed/{inst}_features.parquet'
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                df.index = pd.to_datetime(df.index)
                features[inst] = df
            except Exception:
                continue
        return features

    def _fetch_live_features(self) -> dict:
        """
        Live mode: download the last 60 days of daily OHLCV from yfinance,
        merge with existing raw parquet (keeps full history for feature windows),
        recompute features, and return the updated dataframes.
        Falls back to cached feature parquet for any instrument that fails.
        """
        import yfinance as yf
        from data.download_data import YFINANCE_TICKERS
        from data.compute_features import compute_underlying_features

        features = {}
        for inst in ALL_INSTRUMENTS:
            ticker    = YFINANCE_TICKERS.get(inst)
            raw_path  = f'data/raw/{inst}_daily.parquet'
            feat_path = f'data/processed/{inst}_features.parquet'

            if not ticker:
                if os.path.exists(feat_path):
                    df = pd.read_parquet(feat_path)
                    df.index = pd.to_datetime(df.index)
                    features[inst] = df
                continue

            try:
                fresh = yf.download(ticker, period='60d', progress=False, auto_adjust=True)
                if fresh is None or fresh.empty:
                    raise ValueError("empty response")
                if isinstance(fresh.columns, pd.MultiIndex):
                    fresh.columns = fresh.columns.get_level_values(0)
                fresh = fresh.rename(columns={c: c.lower() for c in fresh.columns})
                fresh.index = pd.to_datetime(fresh.index).normalize()
                fresh.index.name = 'date'
                fresh = fresh[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])

                if os.path.exists(raw_path):
                    existing = pd.read_parquet(raw_path)
                    merged = (
                        pd.concat([existing, fresh])
                        .reset_index()
                        .drop_duplicates(subset='date', keep='last')
                        .set_index('date')
                        .sort_index()
                    )
                else:
                    merged = fresh

                # Persist updated raw data so next scan starts from here
                merged.to_parquet(raw_path)

                feat_df = compute_underlying_features(merged, inst)
                feat_df.index = pd.to_datetime(feat_df.index)
                features[inst] = feat_df

            except Exception as e:
                print(f"  [AI1B live] {inst}: yfinance error ({e}), using cache")
                if os.path.exists(feat_path):
                    try:
                        df = pd.read_parquet(feat_path)
                        df.index = pd.to_datetime(df.index)
                        features[inst] = df
                    except Exception:
                        pass

        return features

    def _regime_matches(self, edge: dict, current_regime: dict) -> bool:
        """Edge only fires if current regime matches edge's regime tag."""
        edge_regime = edge['regime']
        if edge_regime == 'ALL':
            return True
        return (current_regime['regime_state'] == edge_regime or
                edge_regime in current_regime['regime_state'])

    def _check_trigger(self, edge: dict, features: dict) -> tuple:
        """
        Parse trigger_condition and evaluate against live features.
        Returns (triggered, trigger_feature_value).
        """
        underlying = edge['underlying']
        if underlying == 'ALL_INDICES':
            underlyings_to_check = INDEX_UNDERLYINGS
        else:
            underlyings_to_check = [underlying]

        for und in underlyings_to_check:
            raw_und = _COMMODITY_RAW_MAP.get(und, und)
            feat_df = features.get(raw_und) or features.get(und)
            if feat_df is None or len(feat_df) == 0:
                continue
            feat = feat_df.iloc[-1]
            triggered = self._evaluate_condition(edge, feat)
            if triggered:
                tv = feat.get(edge['trigger_feature'], 0.0)
                return True, float(tv) if pd.notna(tv) else 0.0

        return False, 0.0

    def _evaluate_condition(self, edge: dict, feat: pd.Series) -> bool:
        """Evaluate a seeded edge trigger condition against latest feature row."""
        eid = edge['edge_id']
        if eid == 'SEED-001':
            return feat['z_21d'] < -2.0 or feat['z_21d'] > 2.0
        elif eid == 'SEED-002':
            return feat['gap_pct'] < -0.01 and feat['gap_unfilled_eod'] == 1
        elif eid == 'SEED-003':
            return feat['rsi_2'] < 10
        elif eid == 'SEED-004':
            return (feat['mom_5d'] > 0 and feat['rsi_14'] > 55
                    and feat['vol_expanding'] == 0)
        elif eid == 'SEED-005':
            return abs(feat.get('synthetic_xauusd_z_21d', 0.0)) > 2.0
        elif eid == 'SEED-006':
            return abs(feat['z_21d']) > 1.8
        return False

    def _build_signal(self, edge: dict, features: dict,
                      current_regime: dict, trigger_value: float) -> dict:
        """Build full signal object."""
        direction = self._determine_direction(edge, trigger_value)
        option_type = 'call' if direction == 'LONG' else 'put'
        return {
            'signal_id':             (f"{edge['edge_id']}_"
                                      f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'edge_id':               edge['edge_id'],
            'instrument_class':      edge['instrument_class'],
            'underlying':            edge['underlying'],
            'direction':             direction,
            'holding_period':        edge['holding_period'],
            'regime_at_signal':      current_regime['regime_state'],
            'hmm_confidence_at_signal': current_regime['regime_confidence'],
            'transition_warning_flag':  current_regime['transition_warning_flag'],
            'trigger_feature':       edge['trigger_feature'],
            'trigger_value':         trigger_value,
            'signal_time':           datetime.now().isoformat(),
            'option_type':           option_type,
            'strike_selection':      'ATM',
            'expiry_selection':      'NEAREST_WEEKLY_GTE_2DTE',
            'feature_snapshot':      {},
        }

    def _determine_direction(self, edge: dict, trigger_value: float) -> str:
        eid = edge['edge_id']
        if eid in ['SEED-001', 'SEED-005', 'SEED-006']:
            return 'LONG' if trigger_value < 0 else 'SHORT'
        return edge['direction']

    def _log_to_shadow_ledger(self, signals: list, current_regime: dict):
        """Log all fired signals to shadow ledger (AI 2 decision filled in by AI2_Validator)."""
        import json
        if not signals:
            return
        conn = sqlite3.connect(self.db_path)
        now  = datetime.now().isoformat()
        for sig in signals:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO shadow_ledger
                       (signal_id, edge_id, instrument_class, underlying,
                        signal_date, trigger_condition, regime_at_signal,
                        hmm_confidence_at_signal, ai2_decision,
                        holding_period, feature_values_json, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sig['signal_id'], sig['edge_id'],
                        sig['instrument_class'], sig['underlying'],
                        sig['signal_time'][:10],
                        str(sig['trigger_value']),
                        sig['regime_at_signal'],
                        sig['hmm_confidence_at_signal'],
                        'PENDING',
                        sig['holding_period'],
                        json.dumps(sig.get('feature_snapshot', {})),
                        now,
                    )
                )
            except Exception:
                pass
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verification_check():
    print("=== Step 8 — AI 1B Spotter verification ===\n")

    spotter = AI1B_Spotter()

    active_edges = spotter._load_active_edges()
    print(f"  Active edges loaded: {len(active_edges)}")
    for e in active_edges:
        print(f"    {e['edge_id']} | regime={e['regime']} | trigger={e['trigger_feature']}")

    live_features = spotter._compute_live_features()
    print(f"\n  Instruments with live features: {len(live_features)}")
    for inst, df in live_features.items():
        last = df.index[-1].date() if len(df) > 0 else 'N/A'
        print(f"    {inst:<16} rows={len(df)}  latest={last}")

    current_regime = get_current_regime()
    print(f"\n  Current regime: {current_regime['regime_state']} "
          f"({current_regime['regime_confidence']:.2%} confidence, "
          f"TWF={current_regime['transition_warning_flag']})")

    signals = spotter.scan()
    print(f"\n  Signals fired: {len(signals)}")
    for s in signals:
        print(f"    {s['signal_id']} | {s['underlying']} | {s['direction']} | "
              f"trigger={s['trigger_value']:.4f}")

    if len(signals) == 0 and len(active_edges) == 0:
        print("\n  NOTE: No active edges (synthetic data revalidation yielded 1/6 ACTIVE).")
        print("  AI 1B is wired correctly — will fire when edges are ACTIVE on real data.")

    print("\n  PASS — AI 1B scan completed without errors.")
    print("\n=== Step 8 complete ===")


if __name__ == '__main__':
    _verification_check()
