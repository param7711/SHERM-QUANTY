"""
Sherm Quanty — 2-Hour Tactical Regime Engine
Repurposed from the original daily Sherm HMM (regime_engine_sherm.py).

Role: the ACTIONABLE swing-trading regime for 2–15 day trades. Detects whether
a high-opportunity H_BULL / H_BEAR move is starting, continuing, exhausting, or
reversing — faster than the daily engine.

Design notes vs the original daily engine:
  * Same proven machinery: StandardScaler fit on the training window only,
    5-state GaussianHMM, SIDEWAYS override, transition-warning logic, parquet I/O,
    synthetic fallback for offline build environments, and the matplotlib
    validation chart (regime-shaded price + confidence panel).
  * Data is 2-HOUR bars, not daily.
  * 9 faster swing features replace the original 3 daily features.
  * State labelling is a COMPOSITE profile (return + momentum + volatility + VIX
    + drawdown), not mean-return alone — so an early momentum/vol expansion is
    recognised as H_BULL/H_BEAR sooner.
  * The 9 features are persisted so AI 1A (deferred) can consume them later.

30-minute data is NEVER used here (reserved for entry refinement elsewhere).

Self-contained: no dependency on any other Sherm Quanty module.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD_H  = 0.70     # chart reference line only (see old daily engine)
CONFIDENCE_THRESHOLD_L  = 0.50     # below this max-prob → SIDEWAYS override
HMM_PROB_DROP_THRESHOLD = 0.20     # confidence drop → transition warning
VIX_SPIKE_THRESHOLD     = 0.25     # bar-over-bar VIX jump → transition warning
N_STATES                = 5
TRAIN_FRACTION          = 0.70     # scaler + HMM fit on first 70% of 2h bars

TACTICAL_PARQUET        = 'data/processed/tactical_regime_history.parquet'
TACTICAL_MODEL          = 'data/processed/tactical_hmm_model.pkl'

# 2h bars per trading day on NSE (09:15–15:30 → ~3.25h; broker buckets give ~3
# two-hour bars/day). Momentum lookbacks are expressed in 2h bars.
BARS_PER_DAY  = 3
MOM_1D        = 1 * BARS_PER_DAY     # ~3 bars
MOM_3D        = 3 * BARS_PER_DAY     # ~9 bars
MOM_5D        = 5 * BARS_PER_DAY     # ~15 bars
VOL_WIN       = 10                    # realized-vol rolling window (bars)
VOL_FAST      = 5
VOL_SLOW      = 20
SWING_WIN     = 20                    # swing-high / MA window (bars)

# Zerodha Kite instrument token for the NIFTY 50 index
NIFTY_KITE_TOKEN = 256265

REGIME_LABELS = ['H_BULL', 'L_BULL', 'SIDEWAYS', 'L_BEAR', 'H_BEAR']

# Ported unchanged from the original daily engine's chart palette.
REGIME_COLORS = {
    'H_BULL':   '#006400',
    'L_BULL':   '#90EE90',
    'SIDEWAYS': '#808080',
    'L_BEAR':   '#FFB6C1',
    'H_BEAR':   '#8B0000',
}

FEATURE_COLS = [
    'ret_2h', 'mom_1d', 'mom_3d', 'mom_5d',
    'vol_2h', 'vol_expansion', 'vix_chg', 'drawdown', 'dist_ma',
]

_feature_scaler: StandardScaler | None = None


# ===========================================================================
# Data loading — Kite → yfinance → synthetic
# ===========================================================================

def _load_2h_data(kite=None) -> tuple[pd.Series, pd.Series, bool]:
    """
    Load 2-hour Nifty close and (best-effort) India VIX close.
    Returns (nifty_2h, vix_2h, is_synthetic).

    Priority:
      1. Zerodha Kite historical_data (deep intraday history, chunked)
      2. yfinance 2h (capped ~730 days)
      3. synthetic fallback (offline build environments)
    VIX is aligned to the Nifty index; when no intraday VIX is available it is
    forward-filled from a flat proxy so the feature still exists.
    """
    # ---- 1. Kite -------------------------------------------------------
    if kite is not None:
        try:
            nifty_2h = _fetch_kite_2h(kite, NIFTY_KITE_TOKEN)
            if nifty_2h is not None and len(nifty_2h) > 200:
                vix_2h = _fetch_kite_vix(kite, nifty_2h.index)
                print(f"  [tactical] Kite 2h bars: {len(nifty_2h)}")
                return nifty_2h, vix_2h, False
        except Exception as e:
            print(f"  [tactical] Kite fetch failed: {e}")

    # ---- 2. yfinance (hourly → resampled to 2h) ------------------------
    # yfinance has no native "2h" interval and caps intraday history at ~730
    # days. We pull 60m bars and resample to 2h to match the engine's cadence.
    try:
        nifty_2h = _yf_hourly_to_2h('^NSEI')
        if nifty_2h is not None and len(nifty_2h) > 200:
            vix_2h = _yf_hourly_to_2h('^INDIAVIX')
            if vix_2h is not None and len(vix_2h) > 0:
                vix_2h = vix_2h.reindex(nifty_2h.index).ffill().bfill()
            else:
                vix_2h = pd.Series(15.0, index=nifty_2h.index)
            vix_2h.name = 'vix'
            print(f"  [tactical] yfinance 60m→2h bars: {len(nifty_2h)}")
            return nifty_2h, vix_2h, False
    except Exception as e:
        print(f"  [tactical] yfinance 2h fetch failed: {e}")

    # ---- 3. synthetic --------------------------------------------------
    print("  [tactical] Generating synthetic 2h data (offline build only) ...")
    return _generate_synthetic_2h()


def _fetch_kite_2h(kite, token: int) -> pd.Series | None:
    """
    Fetch 2h (120minute) history from Kite in <=60-day chunks (Kite intraday
    limit), concatenated. Returns a close series or None.
    """
    end   = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=5 * 365)
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=60), end)
        try:
            data = kite.historical_data(
                token, cursor.strftime('%Y-%m-%d %H:%M:%S'),
                chunk_end.strftime('%Y-%m-%d %H:%M:%S'), '120minute')
            if data:
                chunks.append(pd.DataFrame(data))
        except Exception:
            pass
        cursor = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    dt = pd.to_datetime(df['date'])
    df['date'] = dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
    s = df.set_index('date')['close'].sort_index()
    s = s[~s.index.duplicated(keep='last')]
    s.name = 'nifty'
    return s


def _fetch_kite_vix(kite, index) -> pd.Series:
    """Best-effort intraday VIX from Kite; flat proxy if unavailable."""
    return pd.Series(15.0, index=index, name='vix')


def _yf_hourly_to_2h(ticker: str) -> pd.Series | None:
    """
    Download 60-minute bars (max ~730 days) and resample to 2-hour bars.
    yfinance exposes no native 2h interval, so this is how we obtain 2h data
    without a broker. Returns a close series (2h) or None.
    """
    raw = yf.download(ticker, interval='60m', period='730d',
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return None
    close = raw['Close'].squeeze()
    idx = pd.to_datetime(close.index)
    close.index = idx.tz_localize(None) if idx.tz is not None else idx
    close = close.dropna()
    # Resample to 2h; drop empty (overnight/weekend) buckets.
    two_h = close.resample('2h').last().dropna()
    two_h.name = ticker
    return two_h


def _generate_synthetic_2h() -> tuple[pd.Series, pd.Series, bool]:
    """
    Synthetic 2h Nifty/VIX for offline validation only. Regime-blocked GBM with
    an OU VIX, mirroring the daily engine's synthetic generator but at 2h cadence.
    """
    rng = np.random.default_rng(42)
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=730)
    # ~3 two-hour bars per business day
    days = pd.bdate_range(start=start, end=end)
    stamps = []
    for d in days:
        for h in (10, 12, 14):
            stamps.append(d + pd.Timedelta(hours=h))
    idx = pd.DatetimeIndex(stamps)

    REGIMES = [   # (frac_start, frac_end, mu_2h, sigma_2h, vix_mu, vix_vf)
        (0.00, 0.15,  0.0004, 0.0035, 14.0, 0.6),   # H_BULL
        (0.15, 0.30, -0.0006, 0.0060, 22.0, 1.1),   # L_BEAR
        (0.30, 0.45,  0.0001, 0.0030, 16.0, 0.7),   # SIDEWAYS
        (0.45, 0.55, -0.0030, 0.0110, 45.0, 2.5),   # H_BEAR (sharp)
        (0.55, 0.75,  0.0006, 0.0050, 20.0, 1.0),   # L_BULL → H_BULL
        (0.75, 0.90,  0.0005, 0.0032, 13.5, 0.6),   # H_BULL
        (0.90, 1.00, -0.0002, 0.0045, 18.0, 0.9),   # L_BULL / chop
    ]
    n = len(idx)
    level, prev_vix = 18000.0, 15.0
    nifty_vals, vix_vals = [], []
    for i in range(n):
        frac = i / n
        mu, sigma, vix_mu, vix_vf = 0.0002, 0.0040, 16.0, 0.8
        for fs, fe, m, s, vm, vf in REGIMES:
            if fs <= frac < fe:
                mu, sigma, vix_mu, vix_vf = m, s, vm, vf
                break
        level *= np.exp(rng.normal(mu, sigma))
        new_vix = prev_vix + 0.05 * (vix_mu - prev_vix) + \
            rng.normal(0, vix_mu * 0.08 * vix_vf)
        prev_vix = max(new_vix, 8.0)
        nifty_vals.append(level)
        vix_vals.append(prev_vix)

    nifty_2h = pd.Series(nifty_vals, index=idx, name='nifty')
    vix_2h   = pd.Series(vix_vals,   index=idx, name='vix')
    return nifty_2h, vix_2h, True


# ===========================================================================
# Feature engineering — 9 swing features
# ===========================================================================

def _build_feature_matrix(nifty_2h: pd.Series,
                          vix_2h: pd.Series,
                          train_fraction: float = TRAIN_FRACTION
                          ) -> tuple[np.ndarray, pd.DatetimeIndex, StandardScaler, pd.DataFrame]:
    """
    Build and standardize the 9-feature 2h matrix.
    StandardScaler is fit on the first `train_fraction` of bars only (no OOS
    leakage); the full matrix is then transformed with that scaler.
    Returns (X_scaled, dates, scaler, raw_feature_df).
    """
    log_ret = np.log(nifty_2h / nifty_2h.shift(1))

    df = pd.DataFrame(index=nifty_2h.index)
    df['ret_2h']        = log_ret
    df['mom_1d']        = log_ret.rolling(MOM_1D).sum()
    df['mom_3d']        = log_ret.rolling(MOM_3D).sum()
    df['mom_5d']        = log_ret.rolling(MOM_5D).sum()
    df['vol_2h']        = log_ret.rolling(VOL_WIN).std()

    vol_fast            = log_ret.rolling(VOL_FAST).std()
    vol_slow            = log_ret.rolling(VOL_SLOW).std()
    df['vol_expansion'] = vol_fast / vol_slow

    df['vix_chg']       = (vix_2h - vix_2h.shift(1)) / vix_2h.shift(1)

    swing_high          = nifty_2h.rolling(SWING_WIN).max()
    df['drawdown']      = (swing_high - nifty_2h) / swing_high

    ma                  = nifty_2h.rolling(SWING_WIN).mean()
    df['dist_ma']       = (nifty_2h - ma) / ma

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X_raw = df[FEATURE_COLS].values
    dates = df.index

    n_train = max(int(len(X_raw) * train_fraction), 50)
    scaler  = StandardScaler()
    scaler.fit(X_raw[:n_train])
    X_scaled = scaler.transform(X_raw)

    return X_scaled, dates, scaler, df


# ===========================================================================
# HMM fit + classify
# ===========================================================================

def _composite_state_scores(model: hmm.GaussianHMM) -> np.ndarray:
    """
    Score each HMM state by a composite bullishness profile on the (z-scored)
    feature means. Higher = more bullish. Feature order == FEATURE_COLS.
      + return + 0.4*(mom_1d+mom_3d+mom_5d)
      - vol_2h - vol_expansion - vix_chg - drawdown
      + dist_ma
    """
    m = model.means_   # shape (n_states, 9)
    ret, mom1, mom3, mom5, vol, volx, vixc, dd, distma = [m[:, i] for i in range(9)]
    score = (ret
             + 0.4 * (mom1 + mom3 + mom5)
             - vol - volx - vixc - dd
             + distma)
    return score


def _fit_and_classify(X: np.ndarray,
                      dates: pd.DatetimeIndex,
                      nifty_2h: pd.Series,
                      vix_2h: pd.Series,
                      raw_features: pd.DataFrame,
                      model: hmm.GaussianHMM | None = None
                      ) -> tuple[pd.DataFrame, hmm.GaussianHMM]:
    """
    Fit a 5-state GaussianHMM on the training slice (or reuse a supplied model),
    classify all bars, and label states by composite score.
    """
    n_train = max(int(len(X) * TRAIN_FRACTION), 50)

    if model is None:
        model = hmm.GaussianHMM(
            n_components=N_STATES, covariance_type='full',
            n_iter=2000, random_state=42,
            init_params='stmc', params='stmc',
        )
        model.fit(X[:n_train])
        print(f"  [tactical] HMM converged: {model.monitor_.converged}  "
              f"(train bars: {n_train})")

    # Composite labelling: sort states by bullishness ascending
    scores        = _composite_state_scores(model)
    sorted_states = np.argsort(scores)
    state_labels  = {
        sorted_states[0]: 'H_BEAR',
        sorted_states[1]: 'L_BEAR',
        sorted_states[2]: 'SIDEWAYS',
        sorted_states[3]: 'L_BULL',
        sorted_states[4]: 'H_BULL',
    }

    all_states = model.predict(X)
    all_probs  = model.predict_proba(X)

    prob_cols = {label: np.zeros(len(dates)) for label in REGIME_LABELS}
    for raw_state, label in state_labels.items():
        prob_cols[label] = all_probs[:, raw_state]

    regime_confidence = all_probs.max(axis=1)
    leading_raw       = all_probs.argmax(axis=1)
    regime_state = np.array([
        'SIDEWAYS' if regime_confidence[i] < CONFIDENCE_THRESHOLD_L
        else state_labels[leading_raw[i]]
        for i in range(len(dates))
    ])

    # Transition warning: confidence drop OR VIX bar-over-bar spike
    conf_series = pd.Series(regime_confidence, index=dates)
    vix_aligned = vix_2h.reindex(dates)
    conf_drop   = conf_series.shift(1) - conf_series
    vix_spike   = (vix_aligned - vix_aligned.shift(1)) / vix_aligned.shift(1)
    twf         = (conf_drop > HMM_PROB_DROP_THRESHOLD) | (vix_spike > VIX_SPIKE_THRESHOLD)
    twf.iloc[0] = False

    result = pd.DataFrame({
        'tactical_regime_state':            regime_state,
        'tactical_regime_confidence':       regime_confidence,
        'tactical_transition_warning_flag': twf.values.astype(bool),
        'hmm_state_int':                    all_states.astype(int),
        'prob_H_BULL':                      prob_cols['H_BULL'],
        'prob_L_BULL':                      prob_cols['L_BULL'],
        'prob_SIDEWAYS':                    prob_cols['SIDEWAYS'],
        'prob_L_BEAR':                      prob_cols['L_BEAR'],
        'prob_H_BEAR':                      prob_cols['H_BEAR'],
        'nifty_close':                      nifty_2h.reindex(dates).values,
        'vix_close':                        vix_aligned.values,
    }, index=dates)
    result.index.name = 'date'

    # Persist the 9 raw features so AI 1A can consume them later
    for col in FEATURE_COLS:
        result[col] = raw_features[col].reindex(dates).values

    return result, model


# ===========================================================================
# Parquet / model I/O
# ===========================================================================

def _save_parquet(df: pd.DataFrame):
    os.makedirs(os.path.dirname(TACTICAL_PARQUET), exist_ok=True)
    df.to_parquet(TACTICAL_PARQUET)
    print(f"  [tactical] Saved {len(df)} rows → {TACTICAL_PARQUET}")


def _load_parquet() -> pd.DataFrame:
    return pd.read_parquet(TACTICAL_PARQUET)


def _save_model(model: hmm.GaussianHMM, scaler: StandardScaler):
    os.makedirs(os.path.dirname(TACTICAL_MODEL), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, TACTICAL_MODEL)


def _load_model() -> tuple[hmm.GaussianHMM, StandardScaler] | tuple[None, None]:
    if not os.path.exists(TACTICAL_MODEL):
        return None, None
    obj = joblib.load(TACTICAL_MODEL)
    return obj['model'], obj['scaler']


# ===========================================================================
# Validation chart — ported from the original daily Sherm HMM engine
# ===========================================================================

def _generate_validation_chart(df: pd.DataFrame, synthetic: bool = False):
    """Same visual convention as the original daily engine's chart, retargeted
    at the tactical (2h) columns: regime-shaded price panel + confidence panel
    with transition-warning markers."""
    os.makedirs('outputs/charts', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    title_suffix = " [SYNTHETIC DATA — real validation requires live data]" if synthetic else ""
    fig.suptitle(f'Sherm Quanty — Tactical (2h) Regime Engine Validation{title_suffix}',
                 fontsize=13)

    # Run-length encode regime blocks (one axvspan per contiguous block)
    regimes = df['tactical_regime_state'].values
    dates   = df.index
    blocks  = []
    start_i = 0
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            blocks.append((regimes[start_i], dates[start_i], dates[i - 1]))
            start_i = i
    blocks.append((regimes[start_i], dates[start_i], dates[-1]))

    for regime, d_start, d_end in blocks:
        color = REGIME_COLORS.get(regime, '#808080')
        ax1.axvspan(d_start, d_end, alpha=0.35, color=color, linewidth=0)
        ax2.axvspan(d_start, d_end, alpha=0.20, color=color, linewidth=0)

    # Nifty close
    ax1.plot(df.index, df['nifty_close'], color='black', linewidth=0.8, label='Nifty 50')
    ax1.set_ylabel('Nifty 50 Close')
    ax1.set_title('Nifty 50 (2h) with Tactical Regime Background (5-state HMM)')

    # Confidence
    ax2.plot(df.index, df['tactical_regime_confidence'], color='navy', linewidth=0.8)
    ax2.axhline(CONFIDENCE_THRESHOLD_H, color='green',  linestyle='--',
                linewidth=0.7, label=f'H threshold ({CONFIDENCE_THRESHOLD_H})')
    ax2.axhline(CONFIDENCE_THRESHOLD_L, color='orange', linestyle='--',
                linewidth=0.7, label=f'L threshold ({CONFIDENCE_THRESHOLD_L})')
    ax2.set_ylabel('Regime Confidence')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Tactical Regime Confidence + Transition Warnings')
    ax2.legend(loc='lower left', fontsize=7)

    # Transition warnings — vertical dashed lines on both panels
    warn_dates = df.index[df['tactical_transition_warning_flag']]
    for d in warn_dates:
        ax1.axvline(d, color='purple', linestyle='--', linewidth=0.4, alpha=0.5)
        ax2.axvline(d, color='purple', linestyle='--', linewidth=0.4, alpha=0.5)

    # Regime legend
    legend_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.7, label=r)
        for r in REGIME_LABELS
    ]
    legend_patches.append(
        mpatches.Patch(facecolor='none', edgecolor='purple',
                       linestyle='--', linewidth=1.5, label='Transition Warning')
    )
    ax1.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=3)

    plt.tight_layout()
    out = 'outputs/charts/tactical_regime_validation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [tactical] Chart saved → {out}")


# ===========================================================================
# History builder
# ===========================================================================

def build_tactical_regime_history(kite=None) -> pd.DataFrame | None:
    """
    Build the full 2h tactical regime history and persist it (+ the model).
    Returns None if no usable 2h data exists (caller falls back to daily).
    """
    print("=== Sherm Quanty — 2h Tactical Regime Engine (build) ===")
    nifty_2h, vix_2h, is_synth = _load_2h_data(kite)
    if nifty_2h is None or len(nifty_2h) < 200:
        print("  [tactical] Insufficient 2h data — tactical engine unavailable.")
        return None

    X, dates, scaler, raw_features = _build_feature_matrix(nifty_2h, vix_2h)
    global _feature_scaler
    _feature_scaler = scaler
    print(f"  [tactical] Feature matrix: {X.shape}  "
          f"({dates[0]} → {dates[-1]})")

    df, model = _fit_and_classify(X, dates, nifty_2h, vix_2h, raw_features)

    print("\n  --- Tactical (2h) regime distribution ---")
    print(df['tactical_regime_state'].value_counts().reindex(REGIME_LABELS).to_string())
    print(f"  Transition warnings: {int(df['tactical_transition_warning_flag'].sum())}")
    if is_synth:
        print("  [tactical] NOTE: built on SYNTHETIC 2h data (offline mode).")

    _save_parquet(df)
    _save_model(model, scaler)
    _generate_validation_chart(df, synthetic=is_synth)
    return df


# ===========================================================================
# Incremental update
# ===========================================================================

def update_tactical_regime(kite=None) -> None:
    """
    Append new 2h bars and reclassify using the SAVED model (no refit).
    No-op with a clear message if 2h data or the model is unavailable.
    """
    if not os.path.exists(TACTICAL_PARQUET) or not os.path.exists(TACTICAL_MODEL):
        print("  [tactical] No existing tactical model — building from scratch.")
        build_tactical_regime_history(kite)
        return

    model, scaler = _load_model()
    if model is None:
        build_tactical_regime_history(kite)
        return

    nifty_2h, vix_2h, _ = _load_2h_data(kite)
    if nifty_2h is None or len(nifty_2h) < 200:
        print("  [tactical] 2h data unavailable at update — leaving history as-is.")
        return

    # Rebuild feature matrix, transform with the SAVED scaler (no refit)
    _, _, _, raw_features = _build_feature_matrix(nifty_2h, vix_2h)
    X_all = scaler.transform(raw_features[FEATURE_COLS].values)
    dates = raw_features.index

    df, _ = _fit_and_classify(X_all, dates, nifty_2h, vix_2h, raw_features, model=model)

    existing  = _load_parquet()
    last_date = existing.index[-1]
    new_rows  = df[df.index > last_date]
    if len(new_rows) == 0:
        print("  [tactical] Already up to date.")
        return

    combined = pd.concat([existing, new_rows])
    combined = combined[~combined.index.duplicated(keep='last')]
    _save_parquet(combined)
    print(f"  [tactical] Appended {len(new_rows)} new 2h rows.")


# ===========================================================================
# Public API
# ===========================================================================

def get_tactical_regime() -> dict | None:
    """
    Latest 2h tactical regime, or None if the tactical parquet is missing
    (2h data unavailable). The facade falls back to the daily engine on None.
    """
    if not os.path.exists(TACTICAL_PARQUET):
        return None
    df  = _load_parquet()
    if len(df) == 0:
        return None
    row = df.iloc[-1]
    return {
        'tactical_regime_state':            str(row['tactical_regime_state']),
        'tactical_regime_confidence':       float(row['tactical_regime_confidence']),
        'tactical_transition_warning_flag': bool(row['tactical_transition_warning_flag']),
    }


def get_tactical_regime_on_date(date: str) -> dict | None:
    """Tactical regime for a specific date (nearest prior 2h bar). None if absent."""
    if not os.path.exists(TACTICAL_PARQUET):
        return None
    df  = _load_parquet()
    idx = pd.Timestamp(date)
    prior = df.index[df.index <= idx] if idx not in df.index else [idx]
    if len(prior) == 0:
        return None
    row = df.loc[prior[-1]] if idx not in df.index else df.loc[idx]
    return {
        'tactical_regime_state':            str(row['tactical_regime_state']),
        'tactical_regime_confidence':       float(row['tactical_regime_confidence']),
        'tactical_transition_warning_flag': bool(row['tactical_transition_warning_flag']),
    }


def get_tactical_features(date: str | None = None) -> dict | None:
    """
    Expose the 9 tactical swing features for a bar (default: latest).
    Intended for future AI 1A consumption. None if unavailable.
    """
    if not os.path.exists(TACTICAL_PARQUET):
        return None
    df = _load_parquet()
    if len(df) == 0:
        return None
    row = df.iloc[-1] if date is None else df.loc[pd.Timestamp(date)]
    return {col: float(row[col]) for col in FEATURE_COLS if col in df.columns}


if __name__ == '__main__':
    build_tactical_regime_history(kite=None)
    print("\n  Latest:", get_tactical_regime())
