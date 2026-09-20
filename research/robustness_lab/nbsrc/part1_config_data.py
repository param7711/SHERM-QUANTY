# %% [markdown]
# # Time-Series Overfitting & Robustness Laboratory
#
# **What this notebook is for.** A backtest tells you what happened on *one*
# realisation of history. That is a sample of size one. The question that
# actually matters before risking money is:
#
# > *If I had observed a different but statistically similar history of the
# > market, would my strategy probably still have appeared profitable?*
#
# This notebook answers that question with evidence rather than with a score.
# It measures the statistical character of real market data, builds a family of
# generators calibrated **to that data and to nothing else**, freezes them,
# produces many alternative market histories, and then runs an unmodified
# strategy across all of them. It layers on the standard overfitting controls
# (walk-forward, parameter perturbation, placebo strategies, execution
# perturbation, multiple-testing corrections, PBO, Deflated Sharpe).
#
# **The rule that keeps this honest:**
#
# ```
# REAL MARKET DATA -> ESTIMATE MARKET PROPERTIES -> BUILD GENERATORS
#   -> VALIDATE GENERATORS AGAINST REAL DATA -> FREEZE -> RUN STRATEGY -> COMPARE
# ```
#
# Generators are calibrated against *market properties*. They are never tuned
# against *strategy performance*. Section 09 freezes them before Section 10 ever
# defines a strategy, and that ordering is the point, not an accident of layout.
#
# **What this notebook cannot do.** It cannot prove an edge is real. Every
# generator encodes assumptions, and a strategy that exploits structure no
# generator reproduces will be under-credited, while one that exploits an
# artefact every generator shares will be over-credited. Section 22-H states the
# limitations explicitly. Treat the output as evidence that shifts a prior, not
# as a verdict.
#
# **Structure.** Sections 00-23, in dependency order. Restart Kernel -> Run All
# works from a clean kernel with no hidden state.

# %% [markdown]
# ---
# ## 00 - Configuration
#
# Every tunable lives here. Nothing below this cell hard-codes a path, a column
# name, a threshold, or a simulation count; everything reads from these
# constants, and `CONFIG` at the bottom is the serialisable snapshot that gets
# written to `results/experiment_config.json` so any result can be traced back
# to the settings that produced it.
#
# `MODE` picks a preset. `QUICK` is for iterating (small simulation counts, runs
# in a couple of minutes). `FULL` is for the real experiment. Override from the
# shell with `ROBUSTLAB_MODE=FULL` without editing the notebook.

# %%
import os

# --------------------------------------------------------------------------
# Run mode
# --------------------------------------------------------------------------
MODE = os.environ.get("ROBUSTLAB_MODE", "QUICK").upper()
if MODE not in ("QUICK", "FULL"):
    raise ValueError(f"MODE must be 'QUICK' or 'FULL', got {MODE!r}")

_PRESETS = {
    # n_simulations is per generator, not in total.
    "QUICK": dict(n_simulations=120, n_mc_trade_resamples=800, cscv_n_splits=10,
                  block_lengths=(5, 20, 60), iaaft_iters=60),
    "FULL": dict(n_simulations=1000, n_mc_trade_resamples=5000, cscv_n_splits=14,
                 block_lengths=(5, 10, 20, 50, 100), iaaft_iters=200),
}
_P = _PRESETS[MODE]

N_SIMULATIONS = _P["n_simulations"]
N_MC_TRADE_RESAMPLES = _P["n_mc_trade_resamples"]
CSCV_N_SPLITS = _P["cscv_n_splits"]
BLOCK_LENGTHS = _P["block_lengths"]
IAAFT_ITERS = _P["iaaft_iters"]

# Set to None for a genuinely independent (non-reproducible) run.
RANDOM_SEED = 20260920

# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------
# DATA_FORMAT: "auto" | "parquet_dir" | "parquet" | "csv" | "demo"
#   parquet_dir - a directory of <ASSET>.parquet OHLCV files (the layout this
#                 repo already uses). parquet/csv - a single file. demo - a
#                 calibrated synthetic series, so the notebook is runnable with
#                 no data present at all.
DATA_FORMAT = "auto"
# Relative to the notebook's working directory. Point ROBUSTLAB_DATA at your own
# file or directory to analyse something else without editing this cell.
DATA_PATH = os.environ.get("ROBUSTLAB_DATA",
                           os.path.join("..", "sma18_trend_forex", "data", "1d"))

# Column names in the source file. Ignored by parquet_dir, which already uses
# the canonical names; used when loading a user CSV with its own conventions.
DATE_COLUMN = "Date"
PRICE_COLUMN = "close"       # the column treated as the settlement/close price
OPEN_COLUMN = "open"
HIGH_COLUMN = "high"
LOW_COLUMN = "low"
VOLUME_COLUMN = "volume"     # may be absent; absence is reported, not fatal
ASSET_COLUMN = None          # set for long-format multi-asset files

# Which instruments to load. None -> every asset found at DATA_PATH.
ASSETS = ["GOLD", "CRUDE", "SUGAR", "BTCUSD", "ETHUSD", "SOLUSD"]
PRIMARY_ASSET = "GOLD"       # the instrument the deep single-asset analysis uses

START_DATE = None            # e.g. "2005-01-01"; None -> earliest available
END_DATE = None              # e.g. "2024-12-31"; None -> latest available

MIN_OBSERVATIONS = 750       # refuse to analyse a series shorter than this
# True applies the minimal conservative repairs in Section 03 and prints every
# single change; it never repairs silently. The default is True because real
# vendor OHLC data routinely contains a small number of internally inconsistent
# bars, and halting on them would make the notebook unrunnable on exactly the
# kind of data it is meant to analyse. Set False to halt instead and inspect.
AUTO_REPAIR = True

# --------------------------------------------------------------------------
# Chronological splits (never shuffled - see Section 06)
# --------------------------------------------------------------------------
TRAIN_FRAC = 0.50
VALIDATION_FRAC = 0.25
TEST_FRAC = 0.25

WALK_FORWARD_N_FOLDS = 6
WALK_FORWARD_MIN_TRAIN = 500     # bars
WALK_FORWARD_EMBARGO = 5         # bars dropped between train and test

# Which slice calibrates the synthetic generators. "train" keeps the test set
# genuinely untouched; "all" uses the full history and is the honest default
# ONLY for descriptive work. Section 34's leakage audit checks this.
GENERATOR_CALIBRATION_SLICE = "train"

# --------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------
N_REGIMES = 4
REGIME_METHODS = ("quantile", "gmm", "hmm")   # all are fitted and compared
REGIME_PRIMARY_METHOD = "hmm"                 # the one the generators consume
REGIME_FEATURE_WINDOW = 21                    # bars for rolling trend/vol features

# --------------------------------------------------------------------------
# Shocks
# --------------------------------------------------------------------------
SHOCK_METHOD = "quantile"        # "sd" or "quantile"
SHOCK_THRESHOLD_SD = 3.0
SHOCK_QUANTILES = (0.95, 0.99, 0.995)
SHOCK_PRIMARY_QUANTILE = 0.99
SHOCK_RECOVERY_WINDOW = 20       # bars examined after a shock

# --------------------------------------------------------------------------
# Strategy + execution
# --------------------------------------------------------------------------
STRATEGY_NAME = "sma_trend"
STRATEGY_PARAMS = {"lookback": 18, "confirm_bars": 2, "direction": "long", "target_vol": None}

COST_BPS = 5.0                   # round-trip cost charged per unit turnover
SLIPPAGE_BPS = 2.0
EXECUTION_DELAY_BARS = 0         # extra bars between signal and fill
COST_GRID_BPS = (0.0, 5.0, 10.0, 20.0, 50.0)
EXECUTION_DELAY_GRID = (0, 1, 2)
SIGNAL_NOISE_GRID = (0.0, 0.05, 0.10, 0.25)

# --------------------------------------------------------------------------
# Parameter robustness
# --------------------------------------------------------------------------
PARAM_GRID = {"lookback": list(range(5, 61, 1))}
PARAM_PERTURB_PCT = (-0.10, -0.05, 0.0, 0.05, 0.10)
ROBUST_PLATEAU_TOLERANCE = 0.25   # fraction of peak Sharpe defining the plateau

# --------------------------------------------------------------------------
# Multiple testing
# --------------------------------------------------------------------------
ALPHA = 0.05
FDR_ALPHA = 0.05

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
RESULTS_DIR = os.environ.get("ROBUSTLAB_RESULTS", "results")
SAVE_REPRESENTATIVE_PATHS = 5    # synthetic series written to disk per generator
FIGSIZE = (11.0, 4.2)
DPI = 110

CONFIG = {
    "mode": MODE,
    "random_seed": RANDOM_SEED,
    "n_simulations": N_SIMULATIONS,
    "n_mc_trade_resamples": N_MC_TRADE_RESAMPLES,
    "cscv_n_splits": CSCV_N_SPLITS,
    "block_lengths": list(BLOCK_LENGTHS),
    "iaaft_iters": IAAFT_ITERS,
    "data": {
        "format": DATA_FORMAT, "path": DATA_PATH, "assets": ASSETS,
        "primary_asset": PRIMARY_ASSET, "start_date": START_DATE, "end_date": END_DATE,
        "date_column": DATE_COLUMN, "price_column": PRICE_COLUMN,
        "volume_column": VOLUME_COLUMN, "asset_column": ASSET_COLUMN,
        "min_observations": MIN_OBSERVATIONS, "auto_repair": AUTO_REPAIR,
    },
    "splits": {
        "train_frac": TRAIN_FRAC, "validation_frac": VALIDATION_FRAC,
        "test_frac": TEST_FRAC, "walk_forward_n_folds": WALK_FORWARD_N_FOLDS,
        "walk_forward_min_train": WALK_FORWARD_MIN_TRAIN,
        "walk_forward_embargo": WALK_FORWARD_EMBARGO,
        "generator_calibration_slice": GENERATOR_CALIBRATION_SLICE,
    },
    "regimes": {
        "n_regimes": N_REGIMES, "methods": list(REGIME_METHODS),
        "primary_method": REGIME_PRIMARY_METHOD, "feature_window": REGIME_FEATURE_WINDOW,
    },
    "shocks": {
        "method": SHOCK_METHOD, "threshold_sd": SHOCK_THRESHOLD_SD,
        "quantiles": list(SHOCK_QUANTILES), "primary_quantile": SHOCK_PRIMARY_QUANTILE,
        "recovery_window": SHOCK_RECOVERY_WINDOW,
    },
    "strategy": {"name": STRATEGY_NAME, "params": dict(STRATEGY_PARAMS)},
    "execution": {
        "cost_bps": COST_BPS, "slippage_bps": SLIPPAGE_BPS,
        "execution_delay_bars": EXECUTION_DELAY_BARS,
        "cost_grid_bps": list(COST_GRID_BPS),
        "execution_delay_grid": list(EXECUTION_DELAY_GRID),
        "signal_noise_grid": list(SIGNAL_NOISE_GRID),
    },
    "param_robustness": {
        "grid": {k: [int(x) for x in v] for k, v in PARAM_GRID.items()},
        "perturb_pct": list(PARAM_PERTURB_PCT),
        "plateau_tolerance": ROBUST_PLATEAU_TOLERANCE,
    },
    "multiple_testing": {"alpha": ALPHA, "fdr_alpha": FDR_ALPHA},
}

print(f"MODE={MODE}  N_SIMULATIONS={N_SIMULATIONS}  seed={RANDOM_SEED}")
print(f"results -> {os.path.abspath(RESULTS_DIR)}")

# %% [markdown]
# ---
# ## 01 - Imports and dependency check
#
# Required packages are fatal if missing. Optional ones gate individual
# generators or tests: if `arch` is absent the GARCH family is skipped and
# everything else still runs, and the skip is reported rather than silently
# swallowed. The rule throughout this notebook is that a missing capability
# produces a visible, named gap in the results, never a quiet substitution.

# %%
import importlib
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path

REQUIRED = ["numpy", "pandas", "scipy", "matplotlib"]
OPTIONAL = {
    "statsmodels": "ACF/PACF, Ljung-Box, Jarque-Bera",
    "sklearn": "Gaussian-mixture regime detection",
    "arch": "GARCH / EGARCH / GJR-GARCH generators",
    "joblib": "parallel simulation",
    "tqdm": "progress bars",
}

_missing_required, _have = [], {}
for _m in REQUIRED:
    try:
        importlib.import_module(_m)
        _have[_m] = True
    except ImportError:
        _missing_required.append(_m)
        _have[_m] = False

if _missing_required:
    raise ImportError(
        "missing required packages: " + ", ".join(_missing_required)
        + "\ninstall with:  pip install " + " ".join(_missing_required)
    )

CAPABILITIES = {}
for _m, _why in OPTIONAL.items():
    try:
        importlib.import_module(_m)
        CAPABILITIES[_m] = True
    except ImportError:
        CAPABILITIES[_m] = False

HAVE_STATSMODELS = CAPABILITIES["statsmodels"]
HAVE_SKLEARN = CAPABILITIES["sklearn"]
HAVE_ARCH = CAPABILITIES["arch"]
HAVE_JOBLIB = CAPABILITIES["joblib"]
HAVE_TQDM = CAPABILITIES["tqdm"]

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats, signal as sp_signal
from scipy.spatial.distance import jensenshannon

if HAVE_STATSMODELS:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox
if HAVE_SKLEARN:
    from sklearn.mixture import GaussianMixture
if HAVE_ARCH:
    from arch import arch_model
if HAVE_TQDM:
    from tqdm.auto import tqdm
else:
    def tqdm(x, **kw):  # noqa: D103 - transparent no-op fallback
        return x

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

print(f"python      {sys.version.split()[0]}")
print(f"numpy       {np.__version__}")
print(f"pandas      {pd.__version__}")
print(f"matplotlib  {mpl.__version__}")
print("\noptional capabilities:")
for _m, _ok in CAPABILITIES.items():
    print(f"  [{'x' if _ok else ' '}] {_m:<13} {OPTIONAL[_m]}"
          + ("" if _ok else f"   -> pip install {_m}"))

_skipped = [m for m, ok in CAPABILITIES.items() if not ok]
if _skipped:
    print(f"\nNOTE: {len(_skipped)} optional package(s) missing; the dependent "
          f"sections will report themselves as SKIPPED rather than silently "
          f"substituting another method.")

# %% [markdown]
# ### Plot theme
#
# One theme, applied once, so every figure in the notebook reads as part of the
# same system. Colours come from a categorical palette whose slot ordering has
# been validated for colour-vision-deficiency separation; slots are assigned in
# fixed order and never cycled. Sequential magnitude uses a single blue ramp;
# polarity uses a blue-red diverging pair with a neutral grey midpoint. Grid and
# axis lines are deliberately recessive so the data carries the contrast.

# %%
# Categorical slots, in fixed assignment order (never cycled past slot 8).
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"],
          C["magenta"], C["green"], C["violet"], C["red"]]
# Status colours are reserved; they never stand in for "series N".
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

mpl.rcParams.update({
    "figure.figsize": FIGSIZE, "figure.dpi": DPI, "savefig.dpi": DPI,
    "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
    "savefig.facecolor": INK["surface"],
    "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
    "axes.labelcolor": INK["secondary"], "axes.titlecolor": INK["primary"],
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.7, "grid.alpha": 1.0,
    "xtick.color": INK["muted"], "ytick.color": INK["muted"],
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK["secondary"],
    "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
    "font.size": 9.5, "font.family": "sans-serif",
    "text.color": INK["primary"], "axes.axisbelow": True,
})


def finish(ax, title=None, xlabel=None, ylabel=None, legend=None, subtitle=None):
    """Apply the shared chart chrome. `legend` forces the legend on/off; the
    default shows one whenever the axes carry two or more labelled series,
    because identity must never be carried by colour alone."""
    if title:
        ax.set_title(title, loc="left")
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=INK["muted"], ha="left", va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    show = (len(labels) >= 2) if legend is None else legend
    if show and labels:
        ax.legend(loc="best")
    ax.grid(axis="both")
    return ax


def pct(x, digits=2):
    """Format a fraction as a percentage string, NaN-safe."""
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x*100:.{digits}f}%"


def num(x, digits=3):
    """Format a float for a table cell, NaN-safe."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


print("plot theme applied")

# %% [markdown]
# ### Reproducibility
#
# All randomness flows from one seeded generator factory. Nothing in the
# notebook calls the legacy global `np.random.*` API, so a rerun with the same
# seed reproduces every number, and each named experiment gets its own
# independent stream (so adding a new experiment cannot shift the draws of an
# existing one).

# %%
_SEED_REGISTRY: dict[str, int] = {}


def get_rng(name: str) -> np.random.Generator:
    """Return an independent, reproducible Generator for a named experiment.

    Deriving each stream from (RANDOM_SEED, name) rather than from a single
    advancing global stream means experiments are order-independent: inserting a
    new experiment does not perturb the draws any other experiment receives.

    If RANDOM_SEED is None the stream is seeded from OS entropy and the run is
    deliberately non-reproducible.
    """
    if RANDOM_SEED is None:
        return np.random.default_rng()
    entropy = abs(hash((RANDOM_SEED, name))) % (2**63)
    _SEED_REGISTRY[name] = entropy
    return np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, entropy]))


RESULTS_PATH = Path(RESULTS_DIR)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
(RESULTS_PATH / "synthetic_samples").mkdir(exist_ok=True)
(RESULTS_PATH / "figures").mkdir(exist_ok=True)

print(f"results dir ready: {RESULTS_PATH.resolve()}")

# %% [markdown]
# ---
# ## 02 - Data loading
#
# One canonical in-memory representation, whatever the source:
#
# | field | contract |
# |---|---|
# | index | `DatetimeIndex`, tz-naive, sorted ascending, unique, named `date` |
# | `open` `high` `low` `close` | float64, strictly positive |
# | `volume` | float64, may be entirely NaN (absence is reported, not fatal) |
#
# Loaders normalise into that contract and do nothing else - no filling, no
# clipping, no dropping. Everything questionable is left in place for Section 03
# to find and report, because a loader that quietly fixes data is a loader that
# quietly hides problems.
#
# If `DATA_PATH` does not exist the notebook falls back to a calibrated
# demonstration series so it remains runnable end-to-end with no data on disk.
# That fallback is loud: it prints a banner and sets `USING_DEMO_DATA`, which
# the final report reads and states.

# %%
CANON_COLS = ["open", "high", "low", "close", "volume"]


def _finalise_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame onto the canonical contract.

    Assumptions: `df` already has a DatetimeIndex or a parseable date column has
    been set as the index by the caller. Rows are NOT dropped or filled here.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name}: index must be a DatetimeIndex, got {type(df.index).__name__}")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "date"
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "close" not in df.columns:
        raise KeyError(f"{name}: no 'close' column after normalisation; have {list(df.columns)}")
    # A close-only series is legal; OHL are mirrored from close and Section 03
    # flags the substitution so no downstream reader mistakes it for real range.
    for c in ("open", "high", "low"):
        if c not in df.columns:
            df[c] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df[CANON_COLS].astype("float64")
    df = df.sort_index()
    return df


def load_parquet_dir(path, assets=None) -> dict[str, pd.DataFrame]:
    """Load <ASSET>.parquet OHLCV files from a directory."""
    p = Path(path)
    files = sorted(p.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no .parquet files in {p}")
    wanted = set(assets) if assets else None
    out = {}
    for f in files:
        name = f.stem
        if wanted is not None and name not in wanted:
            continue
        out[name] = _finalise_frame(pd.read_parquet(f), name)
    if wanted:
        for miss in sorted(wanted - set(out)):
            print(f"  WARNING: requested asset {miss!r} not found in {p}")
    return out


def load_tabular(path, date_column, asset_column=None, rename=None) -> dict[str, pd.DataFrame]:
    """Load a single CSV/Parquet file, wide (one asset) or long (asset column).

    `rename` maps source column names onto canonical ones.
    """
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        raw = pd.read_parquet(p)
    elif p.suffix.lower() in (".csv", ".txt", ".gz"):
        raw = pd.read_csv(p)
    else:
        raise ValueError(f"unsupported extension {p.suffix!r}; use .csv or .parquet")

    if rename:
        raw = raw.rename(columns=rename)
    if date_column and date_column in raw.columns:
        raw[date_column] = pd.to_datetime(raw[date_column], errors="coerce", utc=False)
        raw = raw.set_index(date_column)
    elif not isinstance(raw.index, pd.DatetimeIndex):
        raw.index = pd.to_datetime(raw.index, errors="coerce")

    if asset_column and asset_column in raw.columns:
        return {str(a): _finalise_frame(g.drop(columns=[asset_column]), str(a))
                for a, g in raw.groupby(asset_column)}
    return {p.stem: _finalise_frame(raw, p.stem)}


def make_demo_ohlcv(n=4000, seed=7, start="2008-01-02") -> pd.DataFrame:
    """A demonstration series with the stylised facts of a real market.

    Deliberately NOT a Gaussian random walk. It layers a persistent two-state
    drift regime (so trends exist and reverse), a GARCH-like variance recursion
    (so volatility clusters), Student-t innovations (so tails are fat) and rare
    Poisson jumps (so shocks occur). It exists purely so the notebook runs with
    no data on disk; it is never used when real data is available.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta, nu = 2.0e-6, 0.09, 0.89, 5.0
    p_stay_bull, p_stay_bear = 0.995, 0.985
    mu_bull, mu_bear = 0.0006, -0.0005

    r = np.empty(n)
    sig2 = omega / max(1e-12, 1 - alpha - beta)
    bull = True
    eps = 0.0
    for t in range(n):
        u = rng.random()
        bull = (u < p_stay_bull) if bull else (u > p_stay_bear)
        sig2 = omega + alpha * eps**2 + beta * sig2
        z = rng.standard_t(nu) / math.sqrt(nu / (nu - 2.0))
        eps = math.sqrt(sig2) * z
        jump = rng.normal(-0.02, 0.05) if rng.random() < 0.0025 else 0.0
        r[t] = (mu_bull if bull else mu_bear) + eps + jump

    close = 100.0 * np.exp(np.cumsum(r))
    # Bar shapes scaled by that bar's own move, so ranges look plausible.
    scale = np.abs(r) + 0.004
    high = close * (1 + np.abs(rng.normal(0, 0.5, n)) * scale)
    low = close * (1 - np.abs(rng.normal(0, 0.5, n)) * scale)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.15, n - 1) * scale[1:])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = np.exp(rng.normal(13.0, 0.45, n)) * (1 + 3 * np.abs(r))

    idx = pd.bdate_range(start=start, periods=n, freq="B")
    return _finalise_frame(pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx), "DEMO")


def load_market_data() -> tuple[dict[str, pd.DataFrame], bool]:
    """Resolve DATA_FORMAT/DATA_PATH into {asset: canonical OHLCV frame}.

    Returns (data, using_demo). Applies the START_DATE/END_DATE window and drops
    assets shorter than MIN_OBSERVATIONS, reporting each drop.
    """
    fmt, path = DATA_FORMAT, Path(DATA_PATH)
    using_demo = False

    if fmt == "demo" or not path.exists():
        if fmt != "demo":
            print("=" * 72)
            print(f"DATA_PATH does not exist: {path}")
            print("Falling back to the built-in demonstration series so the notebook")
            print("runs end to end. Results describe SIMULATED data, not a real market.")
            print("=" * 72)
        data, using_demo = {"DEMO": make_demo_ohlcv()}, True
    elif fmt in ("auto", "parquet_dir") and path.is_dir():
        data = load_parquet_dir(path, ASSETS)
    elif fmt in ("auto", "parquet", "csv") and path.is_file():
        data = load_tabular(path, DATE_COLUMN, ASSET_COLUMN, rename={
            OPEN_COLUMN: "open", HIGH_COLUMN: "high", LOW_COLUMN: "low",
            PRICE_COLUMN: "close", VOLUME_COLUMN: "volume"})
    else:
        raise ValueError(f"cannot resolve DATA_FORMAT={fmt!r} against path {path}")

    lo = pd.Timestamp(START_DATE) if START_DATE else None
    hi = pd.Timestamp(END_DATE) if END_DATE else None
    kept = {}
    for name, df in data.items():
        if lo is not None:
            df = df[df.index >= lo]
        if hi is not None:
            df = df[df.index <= hi]
        if len(df) < MIN_OBSERVATIONS:
            print(f"  DROPPED {name}: {len(df)} bars < MIN_OBSERVATIONS={MIN_OBSERVATIONS}")
            continue
        kept[name] = df
    if not kept:
        raise ValueError("no asset survived the date window and MIN_OBSERVATIONS filter")
    return kept, using_demo


DATA, USING_DEMO_DATA = load_market_data()

if PRIMARY_ASSET not in DATA:
    _fallback = sorted(DATA, key=lambda k: -len(DATA[k]))[0]
    print(f"\nPRIMARY_ASSET={PRIMARY_ASSET!r} unavailable; using {_fallback!r} "
          f"(longest loaded series) instead.")
    PRIMARY_ASSET = _fallback
    CONFIG["data"]["primary_asset"] = PRIMARY_ASSET

print(f"\nloaded {len(DATA)} asset(s); primary = {PRIMARY_ASSET}")
_summary = pd.DataFrame([
    {"asset": k, "bars": len(v), "start": v.index[0].date(), "end": v.index[-1].date(),
     "has_volume": bool(v["volume"].notna().any()),
     "has_real_ohlc": bool((v["high"] > v["low"]).any())}
    for k, v in sorted(DATA.items())
])
print(_summary.to_string(index=False))

# %% [markdown]
# ---
# ## 03 - Data validation
#
# Run before anything else touches the data. The report is explicit about
# severity:
#
# - **FAIL** - makes the analysis invalid (non-monotonic index, non-positive
#   prices, impossible OHLC). Execution stops.
# - **WARN** - worth knowing and may bias results (calendar gaps, missing
#   volume, mirrored OHLC, extreme returns).
# - **OK** - checked and clean.
#
# Nothing is repaired unless `AUTO_REPAIR=True`, and when it is, every single
# change is itemised. Silent repair is how a subtly broken dataset produces a
# confident, wrong backtest.

# %%
def validate_ohlcv(df: pd.DataFrame, name: str) -> dict:
    """Run every structural check against one OHLCV frame.

    Returns a report dict with a `checks` list of (severity, check, detail) and
    `n_fail`/`n_warn` counts. Pure: never mutates `df`.
    """
    checks: list[tuple[str, str, str]] = []

    def add(sev, check, detail):
        checks.append((sev, check, detail))

    n = len(df)
    add("OK" if n >= MIN_OBSERVATIONS else "FAIL", "observations",
        f"{n} bars (minimum {MIN_OBSERVATIONS})")

    # --- index integrity -------------------------------------------------
    add("OK" if df.index.is_monotonic_increasing else "FAIL", "chronological order",
        "sorted ascending" if df.index.is_monotonic_increasing else "index NOT sorted")

    n_dup = int(df.index.duplicated().sum())
    add("OK" if n_dup == 0 else "FAIL", "duplicate timestamps",
        "none" if n_dup == 0 else f"{n_dup} duplicated index entries")

    add("OK" if getattr(df.index, "tz", None) is None else "WARN", "timezone",
        "tz-naive (normalised)" if getattr(df.index, "tz", None) is None else str(df.index.tz))

    n_nat = int(df.index.isna().sum())
    add("OK" if n_nat == 0 else "FAIL", "unparseable dates",
        "none" if n_nat == 0 else f"{n_nat} NaT index entries")

    # --- calendar gaps ---------------------------------------------------
    # Financial series are not calendar-continuous; weekends and holidays are
    # expected. So the check is against the series' OWN modal spacing rather
    # than an assumed daily frequency, and only unusually long gaps are flagged.
    if n > 10:
        deltas = pd.Series(df.index).diff().dropna()
        modal = deltas.mode()
        modal_gap = modal.iloc[0] if len(modal) else deltas.median()
        big = deltas[deltas > modal_gap * 5]
        pct_big = len(big) / max(1, len(deltas))
        add("OK" if pct_big < 0.01 else "WARN", "calendar gaps",
            f"modal spacing {modal_gap}; {len(big)} gaps > 5x modal ({pct_big:.2%})"
            + (f"; largest {big.max()}" if len(big) else ""))

    # --- price integrity -------------------------------------------------
    for col in ("open", "high", "low", "close"):
        n_nan = int(df[col].isna().sum())
        n_nonpos = int((df[col] <= 0).sum())
        n_inf = int(np.isinf(df[col].to_numpy()).sum())
        sev = "FAIL" if (n_nan or n_nonpos or n_inf) else "OK"
        add(sev, f"{col} values",
            "clean" if sev == "OK" else f"{n_nan} NaN, {n_nonpos} <= 0, {n_inf} inf")

    hi, lo, op, cl = (df[c].to_numpy() for c in ("high", "low", "open", "close"))
    n_hl = int((hi < lo).sum())
    n_ho = int((hi < np.maximum(op, cl) - 1e-12).sum())
    n_lo = int((lo > np.minimum(op, cl) + 1e-12).sum())
    bad_ohlc = n_hl + n_ho + n_lo
    add("OK" if bad_ohlc == 0 else "FAIL", "OHLC relationships",
        "high >= max(o,c) >= min(o,c) >= low holds everywhere" if bad_ohlc == 0
        else f"{n_hl} high<low, {n_ho} high<max(o,c), {n_lo} low>min(o,c)")

    mirrored = bool(np.allclose(hi, lo))
    add("WARN" if mirrored else "OK", "OHLC realism",
        "high==low everywhere: OHL were mirrored from close (close-only source). "
        "Any intrabar rule will be unusable on this series." if mirrored
        else "genuine intrabar range present")

    # --- volume ----------------------------------------------------------
    if df["volume"].notna().any():
        n_negv = int((df["volume"] < 0).sum())
        n_zerov = int((df["volume"] == 0).sum())
        n_nanv = int(df["volume"].isna().sum())
        sev = "FAIL" if n_negv else ("WARN" if (n_zerov or n_nanv) else "OK")
        add(sev, "volume", f"{n_negv} negative, {n_zerov} zero, {n_nanv} missing")
    else:
        add("WARN", "volume", "absent; volume-dependent tests will be skipped")

    # --- return sanity ---------------------------------------------------
    r = np.diff(np.log(np.where(cl > 0, cl, np.nan)))
    r = r[np.isfinite(r)]
    if len(r) > 30:
        sd = r.std()
        n_extreme = int((np.abs(r) > 10 * sd).sum())
        add("OK" if n_extreme == 0 else "WARN", "extreme returns",
            f"{n_extreme} bars beyond 10 sd (max |log-ret| {np.abs(r).max():.3f})")
        n_flat = int((r == 0).sum())
        add("OK" if n_flat / len(r) < 0.05 else "WARN", "flat bars",
            f"{n_flat} zero-return bars ({n_flat/len(r):.2%})")

    n_fail = sum(1 for s, _, _ in checks if s == "FAIL")
    n_warn = sum(1 for s, _, _ in checks if s == "WARN")
    return {"asset": name, "n_bars": n, "checks": checks, "n_fail": n_fail, "n_warn": n_warn}


def print_validation(report: dict) -> None:
    """Render one validation report as an aligned table."""
    mark = {"OK": "  ok ", "WARN": "WARN ", "FAIL": "FAIL "}
    print(f"\n{report['asset']}  ({report['n_bars']} bars)")
    print("-" * 78)
    for sev, check, detail in report["checks"]:
        print(f"  {mark[sev]} {check:<22} {detail}")
    verdict = ("FAILED" if report["n_fail"] else
               ("passed with warnings" if report["n_warn"] else "passed clean"))
    print(f"  -> {verdict}: {report['n_fail']} fail, {report['n_warn']} warn")


def repair_ohlcv(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list[str]]:
    """Apply the minimal repairs that make a frame analysable, itemising each.

    Only ever called when AUTO_REPAIR is True. Returns (frame, changelog).
    """
    log, out = [], df.copy()
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
        log.append("sorted index ascending")
    if out.index.duplicated().any():
        k = int(out.index.duplicated().sum())
        out = out[~out.index.duplicated(keep="last")]
        log.append(f"dropped {k} duplicate timestamps (kept last)")
    bad = (out[["open", "high", "low", "close"]] <= 0).any(axis=1) | out["close"].isna()
    if bad.any():
        log.append(f"dropped {int(bad.sum())} rows with non-positive or missing prices")
        out = out[~bad]
    hi = out[["high", "open", "close"]].max(axis=1)
    lo = out[["low", "open", "close"]].min(axis=1)
    n_fix = int(((hi > out["high"]) | (lo < out["low"])).sum())
    if n_fix:
        out["high"], out["low"] = hi, lo
        log.append(f"widened {n_fix} bars so high/low bracket open/close")
    return out, log


VALIDATION_REPORTS = {}
for _name in sorted(DATA):
    _rep = validate_ohlcv(DATA[_name], _name)
    VALIDATION_REPORTS[_name] = _rep
    print_validation(_rep)

# Every repair is recorded here so Section 22-H can state what was changed
# rather than leaving it buried in cell output.
REPAIR_LOG: dict[str, list[str]] = {}

_failed = [k for k, v in VALIDATION_REPORTS.items() if v["n_fail"]]
if _failed:
    if AUTO_REPAIR:
        print("\n" + "=" * 78)
        print("AUTO_REPAIR=True - applying minimal repairs. Every change is listed.")
        print("=" * 78)
        for k in _failed:
            DATA[k], _log = repair_ohlcv(DATA[k], k)
            REPAIR_LOG[k] = _log
            for line in _log:
                print(f"  {k}: {line}")
            VALIDATION_REPORTS[k] = validate_ohlcv(DATA[k], k)
        _still = [k for k, v in VALIDATION_REPORTS.items() if v["n_fail"]]
        if _still:
            raise ValueError(f"assets still failing after repair: {_still}")
        print("\nWhat this means: the affected bars had an internally inconsistent")
        print("range (high below the open/close, or low above them) in the source")
        print("feed. The true intrabar path is unknowable, so the repair is the")
        print("minimal one - widen the range just enough to bracket the open and")
        print("close. Close-to-close returns are untouched, so any close-based")
        print("strategy is unaffected; an intrabar rule reading these bars would")
        print("see a slightly wider range than the vendor reported.")
    else:
        raise ValueError(
            f"validation FAILED for {_failed}. Inspect the report above. "
            f"Set AUTO_REPAIR=True to apply itemised repairs, or fix the source data."
        )

print(f"\nall {len(DATA)} asset(s) passed validation "
      f"({sum(v['n_warn'] for v in VALIDATION_REPORTS.values())} warnings total)")

# %% [markdown]
# ---
# ## 04 - Exploratory analysis
#
# A first look at the primary asset before any modelling. The point of this
# section is to build intuition for what the generators in Section 08 will have
# to reproduce: the price path, the clustered bursts of volatility, the
# persistent underwater stretches, and tails far heavier than a normal
# distribution would permit.

# %%
def log_returns(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Log returns, indexed to the bar they are earned on. First bar dropped."""
    s = np.log(df[col]).diff().dropna()
    s.name = "log_return"
    return s


def simple_returns(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Simple (arithmetic) returns - the correct basis for P&L aggregation."""
    s = df[col].pct_change().dropna()
    s.name = "return"
    return s


def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Empirical bar frequency measured from the index itself.

    Assuming 252 would be wrong for crypto (24/7) and wrong again for intraday
    bars, so the annualisation factor is derived from the data's own span.
    """
    if len(index) < 3:
        return float(len(index))
    span_days = (index[-1] - index[0]).days
    if span_days <= 0:
        return float(len(index))
    return len(index) / (span_days / 365.25)


def drawdown_series(equity: np.ndarray | pd.Series) -> np.ndarray:
    """Fractional drawdown at every point (0 at a new high, negative below)."""
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    return eq / peak - 1.0


PX = DATA[PRIMARY_ASSET]
RET = simple_returns(PX)
LOGRET = log_returns(PX)
BPY = infer_bars_per_year(PX.index)

print(f"{PRIMARY_ASSET}: {len(PX)} bars, {PX.index[0].date()} -> {PX.index[-1].date()}")
print(f"inferred bars/year = {BPY:.1f}  ({len(PX)/BPY:.1f} years)")
print(f"annualised volatility = {RET.std()*np.sqrt(BPY):.2%}")
print(f"total return = {PX['close'].iloc[-1]/PX['close'].iloc[0]-1:.1%}")

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1, 1]})

ax = axes[0]
ax.plot(PX.index, PX["close"], color=C["blue"], lw=1.4)
ax.set_yscale("log")
finish(ax, f"{PRIMARY_ASSET} - price, volatility and drawdown",
       ylabel="close (log scale)", legend=False,
       subtitle="log scale, so equal vertical distances are equal percentage moves")

ax = axes[1]
roll_vol = RET.rolling(63).std() * np.sqrt(BPY)
ax.plot(roll_vol.index, roll_vol, color=C["orange"], lw=1.3)
ax.axhline(RET.std() * np.sqrt(BPY), color=INK["muted"], lw=1.0, ls="--")
ax.text(roll_vol.index[int(len(roll_vol) * 0.02)], RET.std() * np.sqrt(BPY),
        "  full-sample average", va="bottom", ha="left", fontsize=8, color=INK["muted"])
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
finish(ax, None, ylabel="63-bar realised vol (ann.)", legend=False)

ax = axes[2]
dd = drawdown_series(PX["close"])
ax.fill_between(PX.index, dd, 0, color=STATUS["critical"], alpha=0.22, lw=0)
ax.plot(PX.index, dd, color=STATUS["critical"], lw=1.0)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
finish(ax, None, xlabel="date", ylabel="drawdown", legend=False)

fig.tight_layout()
plt.show()

print(f"max drawdown (buy & hold): {dd.min():.1%}")
print("Note the clustering in the middle panel: high-volatility bars arrive next")
print("to other high-volatility bars. Any generator that samples returns")
print("independently will fail to reproduce that, and Section 09 will catch it.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

ax = axes[0]
r = RET.to_numpy()
ax.hist(r, bins=120, color=C["blue"], alpha=0.75, density=True, label="observed returns")
xs = np.linspace(r.min(), r.max(), 400)
ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color=C["orange"], lw=1.8,
        label="normal with same mean/sd")
ax.set_yscale("log")
finish(ax, "Return distribution vs. a normal", xlabel="simple return",
       ylabel="density (log)",
       subtitle="log density makes the tail discrepancy visible")

ax = axes[1]
osm, osr = stats.probplot(r, dist="norm", fit=False)
ax.scatter(osm, osr, s=5, color=C["blue"], alpha=0.5, label="observed quantiles")
lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
ax.plot(lim, lim, color=INK["muted"], lw=1.2, ls="--", label="normal reference")
finish(ax, "Normal Q-Q plot", xlabel="theoretical quantile", ylabel="observed quantile")

fig.tight_layout()
plt.show()

print(f"excess kurtosis = {stats.kurtosis(r, fisher=True):.2f}  (normal = 0)")
print(f"skewness        = {stats.skew(r):.2f}  (normal = 0)")
print("Points bending away from the dashed line at both ends are the fat tails:")
print("real extreme moves are far larger and far more frequent than normal.")
