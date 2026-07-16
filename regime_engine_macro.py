"""
Sherm Quanty — Daily Macro Regime Engine
Ported from VortexMomentum's daily India-regime detector.

Role: slow, stable macro climate filter. BULL / NON_BULL only.
This is NOT a swing detector — it provides broad market context.
The actionable swing regime lives in regime_engine_tactical.py.

Self-contained: no cross-repo import of any VortexMomentum file.
No dependency on any other Sherm Quanty module.

Detection rule (VortexMomentum V14 — all three must hold for BULL):
  1. price > MA200   (on 3-business-day resampled close, 67-bar lookback)
  2. HH/HL swing structure   (20-bar window split at its midpoint)
  3. ADX proxy > 0.10        (net_move / (ATR * period), 14-bar)
Else NON_BULL.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_START           = '2010-01-01'
MACRO_PARQUET        = 'data/processed/macro_regime_history.parquet'
NIFTY_CSV            = 'data/raw/NSEI_daily.csv'

# Detector parameters (VortexMomentum V14 — do not tune without backtest)
SWING_LOOKBACK       = 20
ADX_PERIOD           = 14
MA200_BARS_3D        = 67        # ~200 daily bars ≈ 67 three-day bars
ADX_PROXY_THRESHOLD  = 0.10
MIN_3D_BARS          = 75        # need at least this many 3B bars to classify

# yfinance India-index fallbacks, in priority order (VortexMomentum Cell 4)
NIFTY_TICKERS        = ['^CRSLDX', '^NSEI', 'NIFTYBEES.NS', '^CNX500']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_nifty_close(start: str = DATA_START) -> pd.Series:
    """
    Load Nifty daily close.
    Priority: local CSV (data/raw/NSEI_daily.csv) → yfinance fallbacks.
    Returns a tz-naive, date-normalized close series.
    """
    if os.path.exists(NIFTY_CSV):
        print("  [macro] Loading Nifty from local CSV ...")
        df = pd.read_csv(NIFTY_CSV, parse_dates=['date'], index_col='date')
        col = 'close' if 'close' in df.columns else 'Close'
        s = df[col].dropna()
        s.index = pd.to_datetime(s.index).normalize()
        s.name = 'nifty'
        return s.loc[start:]

    print("  [macro] Attempting yfinance download ...")
    for ticker in NIFTY_TICKERS:
        try:
            raw = yf.download(ticker, start=start, end=None,
                              auto_adjust=True, progress=False)
            close = raw['Close'].squeeze()
            if len(close) > 100:
                close.index = pd.to_datetime(close.index).normalize()
                close.name = 'nifty'
                print(f"  [macro] India index → using {ticker} ({len(close)} bars)")
                return close.dropna().loc[start:]
        except Exception:
            continue

    print("  [macro] yfinance unavailable — generating synthetic daily data "
          "(offline build only).")
    return _generate_synthetic_daily(start)


def _generate_synthetic_daily(start: str) -> pd.Series:
    """
    Synthetic daily Nifty close for offline validation only. Regime-blocked GBM
    calibrated to documented NSE history (mirrors the tactical engine's offline
    generator). Replace with real data (CSV or yfinance) for production.
    """
    rng = np.random.default_rng(42)
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start=start, end=end)

    # (start, end, daily_mu, daily_sigma) — trending/choppy blocks
    REGIMES = [
        ('2010-01-01', '2010-10-31',  0.0008, 0.0095),
        ('2010-11-01', '2011-12-31', -0.0003, 0.0135),
        ('2012-01-01', '2013-08-31',  0.0002, 0.0085),
        ('2013-09-01', '2014-04-30',  0.0003, 0.0090),
        ('2014-05-01', '2015-12-31',  0.0010, 0.0075),
        ('2016-01-01', '2016-02-28', -0.0007, 0.0120),
        ('2016-03-01', '2016-10-31',  0.0008, 0.0070),
        ('2016-11-01', '2016-12-31', -0.0008, 0.0130),
        ('2017-01-01', '2017-12-31',  0.0009, 0.0065),
        ('2018-01-01', '2019-03-31', -0.0002, 0.0100),
        ('2019-04-01', '2020-01-31',  0.0005, 0.0085),
        ('2020-02-01', '2020-03-23', -0.0055, 0.0250),
        ('2020-03-24', '2020-12-31',  0.0015, 0.0130),
        ('2021-01-01', '2021-10-31',  0.0010, 0.0072),
        ('2021-11-01', '2022-06-30', -0.0003, 0.0105),
        ('2022-07-01', '2023-12-31',  0.0007, 0.0082),
        ('2024-01-01', '2024-05-31',  0.0008, 0.0075),
        ('2024-06-01', '2026-12-31',  0.0005, 0.0090),
    ]
    regime_map = {}
    for s_str, e_str, mu, sigma in REGIMES:
        s, e = pd.Timestamp(s_str), pd.Timestamp(e_str)
        for d in dates:
            if s <= d <= e:
                regime_map[d] = (mu, sigma)

    level = 5200.0
    vals = []
    for d in dates:
        mu, sigma = regime_map.get(d, (0.0004, 0.0090))
        level *= np.exp(rng.normal(mu, sigma))
        vals.append(level)

    s = pd.Series(vals, index=dates, name='nifty')
    print("  [macro] NOTE: using synthetic daily data.")
    return s.loc[start:]


# ---------------------------------------------------------------------------
# Core detector (ported from VortexMomentum detect_india_regime)
# ---------------------------------------------------------------------------

def _detect_india_regime_on_date(nifty_close: pd.Series,
                                  date: pd.Timestamp
                                  ) -> tuple[str, float]:
    """
    VortexMomentum V14 detector, returning (label, confidence).

    label      : "BULL" if all 3 conditions hold, else "NON_BULL"
    confidence : fraction of the 3 conditions satisfied (0.0, 0.33, 0.67, 1.0)
                 — an addition over the original binary detector so downstream
                 consumers get a graded signal.
    """
    try:
        raw     = nifty_close.loc[:date]
        data_3d = raw.resample('3B').last().dropna()
        if len(data_3d) < MIN_3D_BARS:
            return 'NON_BULL', 0.0

        close_arr = data_3d.values
        current   = close_arr[-1]

        # Condition 1: above MA200 (67 three-day bars ≈ 200 daily bars)
        above_ma200 = current > close_arr[-MA200_BARS_3D:].mean()

        # Condition 2: HH/HL swing structure over the last 20 bars
        window = close_arr[-SWING_LOOKBACK:]
        mid    = SWING_LOOKBACK // 2
        hh_hl  = (window[mid:].max() > window[:mid].max()) and \
                 (window[mid:].min() > window[:mid].min())

        # Condition 3: ADX proxy (directional strength) above threshold
        ranges    = np.abs(np.diff(close_arr[-(ADX_PERIOD + 2):]))
        atr       = ranges.mean()
        net_move  = abs(close_arr[-1] - close_arr[-(ADX_PERIOD + 1)])
        adx_proxy = net_move / (atr * ADX_PERIOD) if atr > 0 else 0.0
        trending  = adx_proxy > ADX_PROXY_THRESHOLD

        conditions = [above_ma200, hh_hl, trending]
        n_met      = sum(bool(c) for c in conditions)
        confidence = n_met / 3.0
        label      = 'BULL' if n_met == 3 else 'NON_BULL'
        return label, round(confidence, 4)
    except Exception:
        return 'NON_BULL', 0.0


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame):
    os.makedirs(os.path.dirname(MACRO_PARQUET), exist_ok=True)
    df.to_parquet(MACRO_PARQUET)
    print(f"  [macro] Saved {len(df)} rows → {MACRO_PARQUET}")


def _load_parquet() -> pd.DataFrame:
    return pd.read_parquet(MACRO_PARQUET)


# ---------------------------------------------------------------------------
# History builder
# ---------------------------------------------------------------------------

def _classify_series(nifty_close: pd.Series) -> pd.DataFrame:
    """Run the detector across every date and attach transition flags."""
    labels, confs = [], []
    for d in nifty_close.index:
        lbl, cf = _detect_india_regime_on_date(nifty_close, d)
        labels.append(lbl)
        confs.append(cf)

    df = pd.DataFrame({
        'daily_regime_state':      labels,
        'daily_regime_confidence': confs,
        'nifty_close':             nifty_close.values,
    }, index=nifty_close.index)
    df.index.name = 'date'

    # Transition warning: label flipped versus the previous bar
    flipped = df['daily_regime_state'] != df['daily_regime_state'].shift(1)
    flipped.iloc[0] = False
    df['daily_transition_warning_flag'] = flipped.values.astype(bool)
    return df


def build_macro_regime_history(start: str = DATA_START) -> pd.DataFrame:
    """Build the full daily macro regime history from scratch and persist it."""
    print("=== Sherm Quanty — Daily Macro Regime Engine (build) ===")
    nifty_close = _load_nifty_close(start)
    print(f"  [macro] Nifty rows: {len(nifty_close)}  "
          f"({nifty_close.index[0].date()} → {nifty_close.index[-1].date()})")

    df = _classify_series(nifty_close)

    print("\n  --- Daily macro regime distribution ---")
    print(df['daily_regime_state'].value_counts().to_string())
    bull_pct = (df['daily_regime_state'] == 'BULL').mean()
    print(f"  BULL: {bull_pct:.1%} of days   "
          f"Transitions: {int(df['daily_transition_warning_flag'].sum())}")

    _save_parquet(df)
    return df


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def update_daily_regime() -> None:
    """
    Append newly available dates to the macro parquet.
    Full re-classification of the appended tail (the detector is stateless
    per-date, so recomputing only new dates is exact — no refit needed).
    """
    if not os.path.exists(MACRO_PARQUET):
        build_macro_regime_history()
        return

    existing  = _load_parquet()
    last_date = existing.index[-1]

    nifty_close = _load_nifty_close(DATA_START)
    new_dates   = nifty_close.index[nifty_close.index > last_date]
    if len(new_dates) == 0:
        print("  [macro] Already up to date.")
        return

    # Reclassify from a small warmup window before last_date so swing/MA
    # lookbacks are fully populated, then keep only the genuinely new rows.
    fresh = _classify_series(nifty_close)
    new_rows = fresh[fresh.index > last_date]

    combined = pd.concat([existing, new_rows])
    combined = combined[~combined.index.duplicated(keep='last')]
    _save_parquet(combined)
    print(f"  [macro] Appended {len(new_rows)} new rows.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_daily_regime() -> dict:
    """Latest daily macro regime. Consumed by the regime_engine_sherm facade."""
    df  = _load_parquet()
    row = df.iloc[-1]
    return {
        'daily_regime_state':            str(row['daily_regime_state']),
        'daily_regime_confidence':       float(row['daily_regime_confidence']),
        'daily_transition_warning_flag': bool(row['daily_transition_warning_flag']),
    }


def get_daily_regime_on_date(date: str) -> dict:
    """Daily macro regime for a specific historical date (used by backtest)."""
    df  = _load_parquet()
    idx = pd.Timestamp(date).normalize()
    if idx not in df.index:
        # nearest prior available date (macro is slow-moving; ffill is safe)
        prior = df.index[df.index <= idx]
        if len(prior) == 0:
            raise KeyError(f"No macro regime on/before {date}.")
        idx = prior[-1]
    row = df.loc[idx]
    return {
        'daily_regime_state':            str(row['daily_regime_state']),
        'daily_regime_confidence':       float(row['daily_regime_confidence']),
        'daily_transition_warning_flag': bool(row['daily_transition_warning_flag']),
    }


if __name__ == '__main__':
    build_macro_regime_history()
    print("\n  Latest:", get_daily_regime())
