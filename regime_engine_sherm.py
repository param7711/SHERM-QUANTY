"""
Sherm Quanty — Regime Engine (Step 0)
Standalone 5-state HMM regime classifier.
No dependencies on any other Sherm Quanty file.
"""

import os
import warnings
import numpy as np
import pandas as pd
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
CONFIDENCE_THRESHOLD_H  = 0.70
CONFIDENCE_THRESHOLD_L  = 0.50
HMM_PROB_DROP_THRESHOLD = 0.20
VIX_SPIKE_THRESHOLD     = 0.25
N_STATES                = 5
TRAIN_END               = '2021-12-31'
DATA_START              = '2010-01-01'
REGIME_PARQUET          = 'data/processed/regime_history.parquet'

REGIME_COLORS = {
    'H_BULL':   '#006400',
    'L_BULL':   '#90EE90',
    'SIDEWAYS': '#808080',
    'L_BEAR':   '#FFB6C1',
    'H_BEAR':   '#8B0000',
}

# Set once when HMM is fitted; reused by update_regime() for OOS rows
_feature_scaler: StandardScaler | None = None

# ---------------------------------------------------------------------------
# Data loading — tries yfinance, falls back to local CSV
# ---------------------------------------------------------------------------

def _download_yfinance(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = df['Close'].squeeze()
    close.index = pd.to_datetime(close.index).normalize()
    close.name = ticker
    return close.dropna()


def _load_raw_data(start: str = DATA_START) -> tuple[pd.Series, pd.Series]:
    """
    Load Nifty 50 and India VIX daily closes.
    Priority:
      1. Local CSV files at data/raw/NSEI_daily.csv and data/raw/INDIAVIX_daily.csv
         (columns: date, close — or any OHLCV with a 'close'/'Close' column)
      2. yfinance live download (requires internet access)
    Returns (nifty_close, vix_close) aligned on an inner join of dates.
    """
    nifty_csv = 'data/raw/NSEI_daily.csv'
    vix_csv   = 'data/raw/INDIAVIX_daily.csv'

    if os.path.exists(nifty_csv) and os.path.exists(vix_csv):
        print("  [data] Loading from local CSV files ...")
        nifty = _read_local_csv(nifty_csv, 'nifty')
        vix   = _read_local_csv(vix_csv,   'vix')
    else:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
        print("  [data] Attempting yfinance download ...")
        try:
            nifty = _download_yfinance('^NSEI',     start, end)
            vix   = _download_yfinance('^INDIAVIX', start, end)
            if len(nifty) == 0 or len(vix) == 0:
                raise ValueError("yfinance returned empty data")
        except Exception as e:
            print(f"  [data] yfinance failed: {e}")
            print("  [data] Generating calibrated synthetic data for validation ...")
            nifty, vix = _generate_synthetic_data(start)

    nifty = nifty.loc[start:]
    vix   = vix.loc[start:]

    aligned = pd.concat([nifty, vix], axis=1, join='inner').dropna()
    aligned.columns = ['nifty', 'vix']
    dropped = len(nifty) + len(vix) - 2 * len(aligned)
    print(f"  Nifty rows: {len(nifty)}  VIX rows: {len(vix)}  "
          f"After inner join: {len(aligned)}  (dropped {dropped} mismatched dates)")
    return aligned['nifty'], aligned['vix']


def _read_local_csv(path: str, name: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=['date'], index_col='date')
    col = 'close' if 'close' in df.columns else 'Close'
    s = df[col].dropna()
    s.name = name
    return s


# ---------------------------------------------------------------------------
# Synthetic data generator (calibrated to actual NSE statistics)
# Used only when yfinance is unavailable (e.g. sandboxed build environment).
# ---------------------------------------------------------------------------

def _generate_synthetic_data(start: str) -> tuple[pd.Series, pd.Series]:
    """
    Generate synthetic NSE/VIX data calibrated to actual statistical properties.
    Uses a known regime sequence based on documented historical market history.

    NOTE: This is for build-environment testing only.
    Replace with real yfinance data (or local CSVs) for production validation.
    """
    rng = np.random.default_rng(42)
    end = pd.Timestamp.today()
    dates = pd.bdate_range(start=start, end=end)

    # Documented historical regime periods
    # Each entry: (start_date, end_date, daily_mu, daily_sigma, vix_mean, vix_vol_factor)
    # Parameters calibrated to actual NSE data (2010–2024)
    REGIMES = [
        # --- Recovery from 2008 crisis ---
        ('2010-01-01', '2010-10-31', 0.0008,  0.0095,  18.0, 0.8),  # L_BULL
        # --- Euro crisis + India slowdown ---
        ('2010-11-01', '2011-12-31', -0.0003, 0.0135,  24.0, 1.2),  # L_BEAR
        # --- Sideways consolidation ---
        ('2012-01-01', '2013-08-31',  0.0002, 0.0085,  17.0, 0.7),  # SIDEWAYS
        # --- Pre-election uncertainty ---
        ('2013-09-01', '2014-04-30',  0.0003, 0.0090,  15.0, 0.7),  # SIDEWAYS → L_BULL
        # --- Modi bull market 2014–2017 ---
        ('2014-05-01', '2015-12-31',  0.0010, 0.0075,  14.0, 0.6),  # H_BULL
        # --- China devaluation/commodity worry ---
        ('2016-01-01', '2016-02-28', -0.0007, 0.0120,  22.0, 1.1),  # L_BEAR
        # --- Demonetization shock Nov 2016 ---
        ('2016-03-01', '2016-10-31',  0.0008, 0.0070,  13.0, 0.6),  # H_BULL
        ('2016-11-01', '2016-12-31', -0.0008, 0.0130,  21.0, 1.0),  # L_BEAR (demo)
        # --- 2017 bull ---
        ('2017-01-01', '2017-12-31',  0.0009, 0.0065,  12.0, 0.5),  # H_BULL
        # --- 2018 correction (IL&FS + global EM selloff) ---
        ('2018-01-01', '2019-03-31', -0.0002, 0.0100,  20.0, 0.9),  # L_BEAR
        # --- 2019 election rally + consolidation ---
        ('2019-04-01', '2020-01-31',  0.0005, 0.0085,  14.5, 0.7),  # L_BULL
        # --- COVID crash Feb–Mar 2020 ---
        ('2020-02-01', '2020-03-23', -0.0055, 0.0250,  55.0, 3.0),  # H_BEAR (sharp)
        # --- COVID bottom and recovery ---
        ('2020-03-24', '2020-12-31',  0.0015, 0.0130,  22.0, 1.2),  # L_BULL → H_BULL
        # --- 2021 bull market ---
        ('2021-01-01', '2021-10-31',  0.0010, 0.0072,  13.0, 0.6),  # H_BULL
        # --- Rate hike fears + FII selling ---
        ('2021-11-01', '2022-06-30', -0.0003, 0.0105,  22.0, 1.0),  # L_BEAR / H_BEAR
        # --- Recovery rally ---
        ('2022-07-01', '2023-12-31',  0.0007, 0.0082,  13.5, 0.7),  # L_BULL
        # --- 2024 election year bull ---
        ('2024-01-01', '2024-05-31',  0.0008, 0.0075,  13.0, 0.6),  # H_BULL
        # --- Post-election (onwards) ---
        ('2024-06-01', '2026-12-31',  0.0005, 0.0090,  15.0, 0.8),  # L_BULL
    ]

    nifty_level = 5200.0
    nifty_vals  = []
    vix_vals    = []
    used_dates  = []

    # Build a per-date regime lookup
    regime_map = {}
    for (s_str, e_str, mu, sigma, vix_mu, vix_vf) in REGIMES:
        s = pd.Timestamp(s_str)
        e = pd.Timestamp(e_str)
        for d in dates:
            if s <= d <= e:
                regime_map[d] = (mu, sigma, vix_mu, vix_vf)

    # Default for dates outside the map
    DEFAULT = (0.0004, 0.0090, 16.0, 0.8)

    prev_vix = 16.0
    for d in dates:
        mu, sigma, vix_mu, vix_vf = regime_map.get(d, DEFAULT)
        ret = rng.normal(mu, sigma)
        nifty_level = nifty_level * np.exp(ret)

        # VIX: mean-reverting Ornstein-Uhlenbeck with regime-dependent level
        kappa = 0.05
        vix_noise = rng.normal(0, vix_mu * 0.08 * vix_vf)
        new_vix = prev_vix + kappa * (vix_mu - prev_vix) + vix_noise
        new_vix = max(new_vix, 8.0)  # VIX floor
        prev_vix = new_vix

        nifty_vals.append(nifty_level)
        vix_vals.append(new_vix)
        used_dates.append(d)

    nifty_s = pd.Series(nifty_vals, index=used_dates, name='nifty')
    vix_s   = pd.Series(vix_vals,   index=used_dates, name='vix')
    print("  [synthetic] NOTE: Using calibrated synthetic data. "
          "Regime chart will show recovered HMM states on synthetic series.")
    print("  [synthetic] Run with real data (or provide data/raw/ CSVs) for historical validation.")
    return nifty_s, vix_s


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def _build_feature_matrix(nifty_close: pd.Series,
                           vix_close: pd.Series) -> tuple[np.ndarray, pd.DatetimeIndex]:
    nifty_log_ret = np.log(nifty_close / nifty_close.shift(1))
    nifty_vol_5d  = nifty_log_ret.rolling(5).std()
    vix_level     = vix_close / 100

    df = pd.DataFrame({
        'log_ret':   nifty_log_ret,
        'vol_5d':    nifty_vol_5d,
        'vix_level': vix_level,
    })
    df = df.dropna()

    X     = df[['log_ret', 'vol_5d', 'vix_level']].values
    dates = df.index
    return X, dates


# ---------------------------------------------------------------------------
# HMM fit + classify
# ---------------------------------------------------------------------------

def _fit_and_classify(X: np.ndarray,
                      dates: pd.DatetimeIndex,
                      nifty_close: pd.Series,
                      vix_close: pd.Series) -> pd.DataFrame:
    global _feature_scaler

    train_mask = dates <= pd.Timestamp(TRAIN_END)
    X_train    = X[train_mask]
    X_oos      = X[~train_mask]

    print(f"  Training on {train_mask.sum()} rows (up to {TRAIN_END})")
    print(f"  OOS rows: {(~train_mask).sum()}")

    # Scale: fit on training data only so OOS cannot leak into the scaler
    _feature_scaler = StandardScaler()
    X_train_s = _feature_scaler.fit_transform(X_train)
    X_oos_s   = _feature_scaler.transform(X_oos) if len(X_oos) > 0 else X_oos

    print(f"  Feature scales (train means): "
          f"log_ret={_feature_scaler.mean_[0]:.5f}  "
          f"vol_5d={_feature_scaler.mean_[1]:.5f}  "
          f"vix={_feature_scaler.mean_[2]:.4f}")

    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type='full',
        n_iter=2000,
        random_state=42,
        init_params='stmc',
        params='stmc',
    )
    model.fit(X_train_s)
    print(f"  HMM converged: {model.monitor_.converged}")

    # State labeling: sort states by mean log return (ascending)
    state_means   = model.means_[:, 0]
    sorted_states = np.argsort(state_means)
    state_labels  = {
        sorted_states[0]: 'H_BEAR',
        sorted_states[1]: 'L_BEAR',
        sorted_states[2]: 'SIDEWAYS',
        sorted_states[3]: 'L_BULL',
        sorted_states[4]: 'H_BULL',
    }

    print("  Learned state means (log_ret, vol_5d, vix_level):")
    for raw, label in sorted(state_labels.items()):
        print(f"    State {raw} ({label}): "
              f"ret={model.means_[raw, 0]:.5f}  "
              f"vol={model.means_[raw, 1]:.5f}  "
              f"vix={model.means_[raw, 2]:.4f}")

    # Classify: in-sample, then OOS (both use scaled features)
    states_train = model.predict(X_train_s)
    probs_train  = model.predict_proba(X_train_s)
    if len(X_oos_s) > 0:
        states_oos = model.predict(X_oos_s)
        probs_oos  = model.predict_proba(X_oos_s)
    else:
        states_oos = np.array([], dtype=int)
        probs_oos  = np.empty((0, N_STATES))

    all_states = np.concatenate([states_train, states_oos])
    all_probs  = (np.vstack([probs_train, probs_oos])
                  if len(probs_oos) > 0 else probs_train)

    # Map raw HMM state → label probability columns
    prob_label_cols = {label: np.zeros(len(dates)) for label in REGIME_COLORS}
    for raw_state, label in state_labels.items():
        prob_label_cols[label] = all_probs[:, raw_state]

    regime_confidence = all_probs.max(axis=1)
    leading_state_raw = all_probs.argmax(axis=1)

    # regime_state: SIDEWAYS override if no state ≥ 0.50
    regime_state = np.array([
        'SIDEWAYS' if regime_confidence[i] < CONFIDENCE_THRESHOLD_L
        else state_labels[leading_state_raw[i]]
        for i in range(len(dates))
    ])

    # Transition warning flag
    conf_series = pd.Series(regime_confidence, index=dates)
    vix_aligned = vix_close.reindex(dates)
    conf_drop   = conf_series.shift(1) - conf_series
    vix_spike   = (vix_aligned - vix_aligned.shift(1)) / vix_aligned.shift(1)
    twf         = (conf_drop > HMM_PROB_DROP_THRESHOLD) | (vix_spike > VIX_SPIKE_THRESHOLD)
    twf.iloc[0] = False

    result = pd.DataFrame({
        'date':                    dates,
        'regime_state':            regime_state,
        'regime_confidence':       regime_confidence,
        'hmm_state_int':           all_states.astype(int),
        'prob_H_BULL':             prob_label_cols['H_BULL'],
        'prob_L_BULL':             prob_label_cols['L_BULL'],
        'prob_SIDEWAYS':           prob_label_cols['SIDEWAYS'],
        'prob_L_BEAR':             prob_label_cols['L_BEAR'],
        'prob_H_BEAR':             prob_label_cols['H_BEAR'],
        'transition_warning_flag': twf.values.astype(bool),
        'nifty_close':             nifty_close.reindex(dates).values,
        'vix_close':               vix_aligned.values,
    })
    result = result.set_index('date')
    return result


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame):
    os.makedirs(os.path.dirname(REGIME_PARQUET), exist_ok=True)
    df.to_parquet(REGIME_PARQUET)
    print(f"  Saved {len(df)} rows → {REGIME_PARQUET}")


def _load_parquet() -> pd.DataFrame:
    return pd.read_parquet(REGIME_PARQUET)


# ---------------------------------------------------------------------------
# Validation chart
# ---------------------------------------------------------------------------

def _generate_validation_chart(df: pd.DataFrame, synthetic: bool = False):
    os.makedirs('outputs/charts', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    title_suffix = " [SYNTHETIC DATA — real validation requires live data]" if synthetic else ""
    fig.suptitle(f'Sherm Quanty — Regime Engine Validation{title_suffix}', fontsize=13)

    # Run-length encode regime blocks (one axvspan per contiguous block)
    regimes = df['regime_state'].values
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
    ax1.set_title('Nifty 50 with Regime Background (shaded by 5-state HMM)')

    # Confidence
    ax2.plot(df.index, df['regime_confidence'], color='navy', linewidth=0.8)
    ax2.axhline(CONFIDENCE_THRESHOLD_H, color='green',  linestyle='--',
                linewidth=0.7, label=f'H threshold ({CONFIDENCE_THRESHOLD_H})')
    ax2.axhline(CONFIDENCE_THRESHOLD_L, color='orange', linestyle='--',
                linewidth=0.7, label=f'L threshold ({CONFIDENCE_THRESHOLD_L})')
    ax2.set_ylabel('Regime Confidence')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Regime Confidence + Transition Warnings')
    ax2.legend(loc='lower left', fontsize=7)

    # Transition warnings — vertical dashed lines on both panels
    warn_dates = df.index[df['transition_warning_flag']]
    for d in warn_dates:
        ax1.axvline(d, color='purple', linestyle='--', linewidth=0.4, alpha=0.5)
        ax2.axvline(d, color='purple', linestyle='--', linewidth=0.4, alpha=0.5)

    # Regime legend
    legend_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.7, label=r)
        for r in ['H_BULL', 'L_BULL', 'SIDEWAYS', 'L_BEAR', 'H_BEAR']
    ]
    legend_patches.append(
        mpatches.Patch(facecolor='none', edgecolor='purple',
                       linestyle='--', linewidth=1.5, label='Transition Warning')
    )
    ax1.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=3)

    plt.tight_layout()
    out = 'outputs/charts/regime_validation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved → {out}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_current_regime() -> dict:
    """Return today's regime state. Used by AI 1B and AI 2."""
    df  = _load_parquet()
    row = df.iloc[-1]
    return {
        'regime_state':            str(row['regime_state']),
        'regime_confidence':       float(row['regime_confidence']),
        'transition_warning_flag': bool(row['transition_warning_flag']),
    }


def get_regime_on_date(date: str) -> dict:
    """Return regime for a specific historical date. Used by backtest."""
    df  = _load_parquet()
    idx = pd.Timestamp(date)
    if idx not in df.index:
        raise KeyError(f"Date {date} not found in regime history.")
    row = df.loc[idx]
    return {
        'regime_state':            str(row['regime_state']),
        'regime_confidence':       float(row['regime_confidence']),
        'transition_warning_flag': bool(row['transition_warning_flag']),
        'hmm_state_int':           int(row['hmm_state_int']),
        'prob_H_BULL':             float(row['prob_H_BULL']),
        'prob_L_BULL':             float(row['prob_L_BULL']),
        'prob_SIDEWAYS':           float(row['prob_SIDEWAYS']),
        'prob_L_BEAR':             float(row['prob_L_BEAR']),
        'prob_H_BEAR':             float(row['prob_H_BEAR']),
    }


def update_regime():
    """
    Download latest data and append new rows to regime_history.parquet.
    Called at 9:30 AM IST daily by scheduler.
    Uses read-concat-rewrite (no append mode).
    """
    existing  = _load_parquet()
    last_date = existing.index[-1]
    new_start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end       = pd.Timestamp.today().strftime('%Y-%m-%d')

    if new_start > end:
        print("  Regime history already up to date.")
        return

    # Full rebuild: re-fits HMM + scaler on 2010-TRAIN_END, classifies to today
    print(f"  Fetching from {DATA_START} to rebuild with latest data ...")
    nifty_full, vix_full  = _load_raw_data(start=DATA_START)
    X_full, dates_full    = _build_feature_matrix(nifty_full, vix_full)
    new_df                = _fit_and_classify(X_full, dates_full, nifty_full, vix_full)
    # _feature_scaler is refreshed inside _fit_and_classify (module-level)

    new_rows = new_df[new_df.index > last_date]
    if len(new_rows) == 0:
        print("  No new rows to append.")
        return

    combined = pd.concat([existing, new_rows])
    combined = combined[~combined.index.duplicated(keep='last')]
    _save_parquet(combined)
    print(f"  Appended {len(new_rows)} new rows.")


# ---------------------------------------------------------------------------
# Main — build from scratch and generate validation chart
# ---------------------------------------------------------------------------

def build_regime_history():
    print("=== Sherm Quanty Regime Engine — Build (Step 0) ===\n")
    print("[1/4] Loading market data ...")
    nifty_close, vix_close = _load_raw_data()
    using_synthetic = not (os.path.exists('data/raw/NSEI_daily.csv'))

    print("\n[2/4] Building feature matrix ...")
    X, dates = _build_feature_matrix(nifty_close, vix_close)
    print(f"  Feature matrix: {X.shape}  "
          f"Range: {dates[0].date()} → {dates[-1].date()}")

    print("\n[3/4] Fitting HMM and classifying regimes ...")
    df = _fit_and_classify(X, dates, nifty_close, vix_close)

    print("\n--- Regime distribution ---")
    print(df['regime_state'].value_counts().sort_index().to_string())
    print(f"\n--- Transition warning events: {df['transition_warning_flag'].sum()} ---")
    print("\n--- Last 5 rows ---")
    print(df[['regime_state', 'regime_confidence', 'transition_warning_flag',
              'nifty_close', 'vix_close']].tail().to_string())

    print("\n[4/4] Saving parquet and generating chart ...")
    _save_parquet(df)
    _generate_validation_chart(df, synthetic=using_synthetic)

    print("\n=== Step 0 complete ===")
    if using_synthetic:
        print("\n*** SYNTHETIC DATA MODE ***")
        print("    The chart shows HMM regime recovery on synthetic NSE/VIX data.")
        print("    To validate against real history:")
        print("    Option A: Provide real data CSVs at data/raw/NSEI_daily.csv")
        print("              and data/raw/INDIAVIX_daily.csv (columns: date, close)")
        print("    Option B: Run on a machine with unrestricted internet access.")
        print("    Real validation checks: 2014 H_BULL, Mar 2020 H_BEAR, 2022 L_BEAR/H_BEAR")
    else:
        print("\n>>> Review outputs/charts/regime_validation.png before Step 1.")


if __name__ == '__main__':
    build_regime_history()
