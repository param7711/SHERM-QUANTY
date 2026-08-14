"""
Builds intraday_fx.ipynb -- the INTRADAY counterpart to generalization.ipynb.

The sibling `generalization.ipynb` could only run its intraday grid (Grid A) on
SYNTHETIC data, because yfinance is firewalled in this sandbox. Its only
decision-grade real-data result was Grid B: DAILY/WEEKLY FX, where 4 of 6 runs
showed the forward-return contrast INVERTING sign between in-sample and
out-of-sample -- the classic overfitting signature, which REFUTED its
pre-registered prediction P3.

This notebook runs the SAME test on REAL LONG-HISTORY INTRADAY FX -- 3 pairs x
{1h, 2h}, ~57,600 bars per pair spanning 2012-11-16 -> 2022-03-04 -- reachable
from this sandbox via raw.githubusercontent.com. It is therefore MUCH better
powered per run than the daily grid (~57,600 bars vs ~13,900) and it answers
whether the daily inversion finding reproduces intraday.

Every metric definition, guard, fixed-constants rule and the IS-vs-OOS
overfitting test are REUSED VERBATIM from GENERALIZATION_PROTOCOL.md and its
implementation in build_generalization.py, so the numbers are directly
comparable across the two notebooks.

The engine is INLINED VERBATIM out of build_master_notebook_v2.py; the S / L / W
metrics, the ZigZag and guards G1-G4 are INLINED VERBATIM out of
build_stability_lag.py. Nothing here imports or modifies the master, its
generator, regime_scorecard.py or any sibling generator.
"""
import json
import os

HERE = '/tmp/claude-0/-home-user-SHERM-QUANTY/f1a44349-dfe0-5e0f-9b1f-b0b39e930e1b/scratchpad'
OUT = f'{HERE}/intraday_fx.ipynb'
MASTER_GEN = f'{HERE}/build_master_notebook_v2.py'
SL_GEN = f'{HERE}/build_stability_lag.py'

# ---------------------------------------------------------------------------
# HARVEST 1 -- the engine cells from the master generator, VERBATIM.
# Identical technique to build_generalization.py: the generator's cell bodies
# are Python string literals with runtime concatenation, so they are obtained
# by EXECUTING the generator in a private namespace (output path redirected to
# a throwaway file) and reading its assembled `cells` list.
# ---------------------------------------------------------------------------
_gsrc = open(MASTER_GEN).read()
assert "regdet_v11_master.ipynb'" in _gsrc
_gsrc = _gsrc.replace("regdet_v11_master.ipynb'", "_intraday_fx_engine_harvest.ipynb'", 1)
_ns = {'__name__': '_master_gen_harvest'}
exec(compile(_gsrc, MASTER_GEN, 'exec'), _ns)
_mcells = _ns['cells']


def _src(i):
    return ''.join(_mcells[i]['source'])


IDX_CONST, IDX_LOAD, IDX_ENGINE, IDX_PLOT = 3, 5, 7, 9
SRC_CONST, SRC_LOAD = _src(IDX_CONST), _src(IDX_LOAD)
SRC_ENGINE, SRC_PLOT = _src(IDX_ENGINE), _src(IDX_PLOT)

# Cell identity is asserted by CONTENT, never trusted by number.
assert 'REGIME_COLORS' in SRC_CONST and 'import numpy as np' in SRC_CONST \
    and 'CONFIGS = [' in SRC_CONST and 'BASE_WIN = dict(' in SRC_CONST
assert 'def load_2h' in SRC_LOAD and 'def _synth' in SRC_LOAD
assert 'def build_features' in SRC_ENGINE and 'def label_bars' in SRC_ENGINE \
    and 'def fit_hmm_ensemble' in SRC_ENGINE and 'def confirm_delay' in SRC_ENGINE \
    and 'def n_params' in SRC_ENGINE \
    and 'def ensemble_direction_masses_by_mode' in SRC_ENGINE
assert 'def shade_bands' in SRC_PLOT and 'def regime_legend' in SRC_PLOT \
    and 'def set_price_ylim' in SRC_PLOT and 'def shade_regimes' in SRC_PLOT \
    and 'def regime_blocks' in SRC_PLOT and 'def mark_split' in SRC_PLOT \
    and 'YLIM_CHECKS' in SRC_PLOT

# ---------------------------------------------------------------------------
# HARVEST 2 -- the HAC contrast machinery, VERBATIM, out of the master's
# BAR_DIR_WEIGHT-sweep section (the two function definitions only).
# ---------------------------------------------------------------------------
_i0 = _gsrc.index('def _w_hac_contrast(')
_i1 = _gsrc.index('def W_SECONDARY_PERBAR(')
SRC_HAC = _gsrc[_i0:_i1].rstrip() + '\n'
assert 'def _w_hac_contrast' in SRC_HAC and 'def W_PRIMARY_METRIC' in SRC_HAC
assert 'Bartlett' in SRC_HAC and 'np.linalg.pinv' in SRC_HAC
assert 'W_GRID' not in SRC_HAC and 'print(' not in SRC_HAC   # definitions only

# ---------------------------------------------------------------------------
# HARVEST 3 -- the ZigZag and the S / L / W + guard metrics, VERBATIM, out of
# build_stability_lag.py (sliced by content marker; that generator cannot be
# exec'd here because it asserts BAR_DIR_WEIGHT == 0.0, now 0.5).
# ---------------------------------------------------------------------------
_slsrc = open(SL_GEN).read()


def _sl_cell(marker):
    """Return the body of the build_stability_lag.py `co(r\"\"\"...\"\"\")` cell
    that contains `marker`. Verbatim, not one character altered."""
    hits = []
    for chunk in _slsrc.split('co(r"""')[1:]:
        body = chunk.split('\n""")')[0]
        if marker in body:
            hits.append(body)
    assert len(hits) == 1, f'{marker!r} matched {len(hits)} cells in {SL_GEN}'
    return hits[0].strip('\n') + '\n'


SRC_ZIGZAG = _sl_cell('THE ZIGZAG -- LABEL-BLIND')
# That cell also carries build_stability_lag.py's OWN leakage tripwire, which
# hard-codes `assert _bw == 0.0`. This notebook installs its own tripwire (same
# four checks, anchored to the frozen BAR_DIR_WEIGHT instead of a stale literal),
# so the trailing tripwire is cut off here rather than double-installed.
_cut = SRC_ZIGZAG.index('# THE LEAKAGE TRIPWIRE.')
_cut = SRC_ZIGZAG.rindex('# ---', 0, _cut)
SRC_ZIGZAG = SRC_ZIGZAG[:_cut].rstrip() + '\n'
assert '_label_bars_raw' not in SRC_ZIGZAG and 'assert _bw == 0.0' not in SRC_ZIGZAG
SRC_SLW = _sl_cell('S / L / W  -- the three metrics')
assert 'def zigzag_pivots' in SRC_ZIGZAG and 'def zigzag_swings' in SRC_ZIGZAG
assert 'def metric_S' in SRC_SLW and 'def metric_L' in SRC_SLW \
    and 'def metric_W' in SRC_SLW and 'def evaluate_guards' in SRC_SLW \
    and 'def occupancy_pct' in SRC_SLW and 'def guards_pass' in SRC_SLW

PROVENANCE = (
    f'build_master_notebook_v2.py cell {IDX_CONST} (constants/imports/CONFIGS), '
    f'cell {IDX_LOAD} (load_2h/_synth), cell {IDX_ENGINE} (feature + labeling '
    f'engine), cell {IDX_PLOT} (regime-background plot helpers), plus '
    f'_w_hac_contrast / W_PRIMARY_METRIC sliced verbatim from the same '
    f'generator; ZigZag + S/L/W + guards G1-G4 sliced verbatim from '
    f'build_stability_lag.py'
)
try:
    os.remove(f'{HERE}/_intraday_fx_engine_harvest.ipynb')
except OSError:
    pass

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip('\n').splitlines(keepends=True)})


def co(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip('\n').splitlines(keepends=True)})


BANNER = '# ' + '=' * 74 + '\n'

# ===========================================================================
md(r"""
# RegDet V1.1 — INTRADAY FX GENERALIZATION (REAL DATA)

The **intraday counterpart** to `generalization.ipynb`. Same frozen protocol
(`GENERALIZATION_PROTOCOL.md`), same metric definitions, same guards, same
fixed-constants rule, same IS-vs-OOS overfitting test — so every number here is
directly comparable with that notebook's.

## Why this notebook exists

`generalization.ipynb` had two grids. Its **Grid B** (daily/weekly FX, Fed H.10)
ran on real data and was decision-grade. Its **Grid A** — the *intraday*,
multi-market grid — could only run on **synthetic GBM**, because yfinance is
firewalled in this sandbox. So the architecture had never been tested on **real
intraday** data locally.

Real long-history intraday FX **is** reachable from here. This notebook uses it.
There is **no synthetic fallback** for this grid: if the data does not load, the
runs appear in the table as errors and nothing is substituted.

## The headline question

Grid B of the sibling notebook found the forward-return contrast **inverting
sign between in-sample and out-of-sample on 4 of 6 real daily FX runs**
(e.g. JPY/USD daily: IS HAC *t* = +2.06, OOS *t* = −0.38). That **REFUTED** its
pre-registered prediction P3.

This notebook asks the same question with **~57,600 bars per run instead of
~13,900** — roughly 4× the sample, and therefore far better powered. Section 9
states explicitly, in plain words, whether the intraday result **AGREES** or
**DISAGREES** with that daily finding.

## The data — COMMUNITY GITHUB DATA, verified live, NOT an official feed

| | |
|---|---|
| source | `raw.githubusercontent.com/ejtraderLabs/historical-data` |
| status | **community GitHub repository — NOT an official exchange or central-bank feed** |
| pairs | EURUSD, GBPUSD, USDJPY |
| base file | `{SYM}h1.csv` — hourly OHLC + tick volume |
| span | **2012-11-16 → 2022-03-04**, 57,600 bars per pair |
| verification | re-run live below against **Brexit** and **COVID**, plus structural checks |

> ### THE DATA ENDS 2022-03-04.
> This notebook says **nothing whatsoever** about 2022–2026 — not the inflation
> shock, not the rate-hiking cycle, not the 2022 EURUSD parity break, not
> anything after the first week of March 2022. Every conclusion below is scoped
> to a **2012–2022 sample**.

Prices in the source files are **integer-scaled, and the scale differs by pair**
(EURUSD `127801.0` → `1.27801`; USDJPY `81121.0` → `81.121`). The loader
**detects** the divisor per pair and asserts the rescaled series lands in a sane
band. Note: rescaling by a constant **does not change log returns**, so it
cannot change a single label or statistic — it matters only for readable charts
and for the sanity assert. That is stated here rather than implied.

## The grid — 3 pairs × 2 timeframes = 6 runs

* **1h** — the repo's `h1` file used **directly**.
* **2h** — the `h1` file resampled **downward** (`open=first, high=max,
  low=min, close=last, tick_volume=sum`), asserted downward-only.

Finer bars are **never** fabricated from coarser ones. 4h is not run: the
request was 1h and 2h.

## What is FIXED and what may re-derive

**Fixed at the shipped Nifty-derived values for all 6 runs:** `N_STATES=5`,
`cov='diag'`, the 9 features, `CONF_L=0.50`, `CONFIRM_BARS=2`,
`BAR_DIR_WEIGHT=0.5`, `ENSEMBLE_K=4`, `Z_HI=0.5`, `EFF_HI=0.35`,
`Z_HI_EXIT=0.35`, `EFF_HI_EXIT=0.25`, `EFF_WIN=9`, momentum ladder 1/3/5 *days*,
`CONTEXT_DAYS=12`, `INTENSITY_MODE='frozen_z'`,
`ESCALATION_DURING_HOLD='allow'`, `DIRECTION_MODE='rank'`,
`TRAIN_FRACTION=0.70`, `N_FOLDS=4`.

**May re-derive per run:** the `StandardScaler` and any fit-window baseline the
engine already computes internally. **Nothing else**, asserted after every run.

> If the architecture only works when knobs are re-tuned per market,
> **that is the finding** — it is reported, not engineered around.
""")

co(r"""
%pip install -q hmmlearn
""")

# --- engine ----------------------------------------------------------------
md(f"""
## 1. Engine, inlined verbatim

Inlined out of `build_master_notebook_v2.py` rather than imported, because a
standalone `.ipynb` cannot import a local `.py` sitting next to it.
Source: `{PROVENANCE}`.

Not one line of the engine is modified — same constants, same 9 features, same
gates, same hysteresis, same causality proofs. The master notebook, its
generator, `regime_scorecard.py` and every sibling generator are untouched.

The **S / L / W metrics, the ZigZag and guards G1–G4** are inlined verbatim out
of `build_stability_lag.py` in section 2, so the `S`, `L`, `W` and guard numbers
here are directly comparable with that notebook's **and** with
`generalization.ipynb`'s.

The master's own `load_2h()` runs at the bottom of the loader cell. Its Nifty
series is **not used** by any run in this notebook — only its `TAC_SYNTH` flag
and helper definitions come along for the ride.
""")

co(BANNER
   + f'# CONSTANTS + IMPORTS -- INLINED VERBATIM from {MASTER_GEN.split("/")[-1]}\n'
   + f'# (master notebook code cell {IDX_CONST}). Unmodified.\n'
   + BANNER + SRC_CONST)

co(BANNER
   + f'# DATA LOADER -- INLINED VERBATIM (master code cell {IDX_LOAD}).\n'
   + '# Supplies TAC_SYNTH and _synth(). Its Nifty series is NOT used below.\n'
   + BANNER + SRC_LOAD)

co(BANNER
   + f'# FEATURE + LABELING ENGINE -- INLINED VERBATIM (master code cell {IDX_ENGINE}).\n'
   + '# build_features / fit_hmm_ensemble / confirm_delay / label_bars / n_params.\n'
   + BANNER + SRC_ENGINE)

co(BANNER
   + f'# PLOT HELPERS -- INLINED VERBATIM (master code cell {IDX_PLOT}).\n'
   + '# regime_blocks / shade_bands / shade_regimes / set_price_ylim /\n'
   + '# regime_legend / mark_split / YLIM_CHECKS.\n'
   + '# This is the MASTER NOTEBOOK regime-background style (full-height bands,\n'
   + '# each block extended to the START of the next so there are no white stripes,\n'
   + '# explicit non-zero-anchored y-limits).\n'
   + BANNER + SRC_PLOT)

co(BANNER
   + '# HAC (Newey-West) CONTRAST -- INLINED VERBATIM from the master generator\n'
   + '# (the two function definitions only; the sweep that surrounds them is not\n'
   + '# run here). This is the SAME estimator the master, the w-sweep and\n'
   + '# generalization.ipynb use, so every t-stat here is comparable with theirs.\n'
   + BANNER + SRC_HAC)

# ===========================================================================
md(r"""
## 2. Frozen parameters, the fixed-knob invariant, and the leakage tripwire

Everything in this section is declared **before any run exists**.

### `BARS_PER_DAY` for intraday FX

Spot FX trades ~24h on weekdays, so the assumption is **1h → 24** and
**2h → 12**. Every *day*-denominated window (the 1/3/5-day momentum ladder, the
12-day context window) is `days × BARS_PER_DAY`, rebuilt per run by the
**master's own formula** with the **same day counts**. That is the protocol's
instruction, not a re-tune — holding raw *bar counts* fixed across timeframes
would silently change the horizon the detector looks at.

The **realised** median bars per session is measured from each series and
printed beside the assumption, with a warning on material disagreement.
Nothing is silently rescaled.

Two honest consequences, stated before the numbers:

* **`EFF_WIN = 9` is bar-denominated and frozen by decree.** It covers 9 hours
  at 1h and 18 hours at 2h. The protocol fixes it, so it stays fixed — but the
  chop filter's real-time horizon is therefore **not** constant across the
  timeframe axis. A property of the frozen spec, flagged rather than fixed.
* `VOL_WIN=10` and `VOL_FAST=5` are likewise raw bar counts in the master and
  are left alone, with the same caveat.

### The VIX question — a proxy, never a silent substitute

Spot FX has **no natural VIX**. `^INDIAVIX` pairs with Nifty and only with
Nifty. Every run here feeds the `vix_chg` feature the **same causal trailing
realised-volatility proxy** the sibling notebooks use: the rolling standard
deviation of log returns over `VOL_SLOW` bars, annualised. It reads bars ≤ *t*
only — no shift, no centring, **no bfill** (that was audited bug 3). Every run
prints `realised-vol PROXY (causal)` as its volatility source, and the stamp is
carried through every table.

### The ZigZag leakage tripwire

`label_bars` is shadowed by a guard that asserts, on **every** call: phase
ordering (`ZZ_CALLS == 0`), object identity against every ZigZag output, memory
aliasing via `np.shares_memory`, and `bar_dir_weight == BAR_DIR_WEIGHT`.
""")

co(r"""
# ===========================================================================
# METRIC + GRID PARAMETERS -- ALL DECLARED HERE, BEFORE ANY NUMBER EXISTS.
# Values are IDENTICAL to build_stability_lag.py and build_generalization.py so
# the three notebooks compare directly.
# ===========================================================================
import itertools, time, contextlib, io, sys, urllib.request
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap

T_START = time.time()

R_SEED_SETS   = 4       # protocol: R = 4 DISJOINT seed sets. NEVER cut.
ZZ_PCT        = 2.0     # ZigZag reversal threshold, %
W_GUARD_MAX   = 25.0    # G3
OCC_MIN_PCT   = 3.0     # G1 / G2
OCC_MAX_PCT   = 50.0    # G1
SIDEWAYS_MAX  = 50.0    # G4

# Pre-registered P2 band (protocol text: "roughly 20-45%").
P2_LO, P2_HI  = 20.0, 45.0

# Day-denominated windows. Same day counts as the master; only the bars-per-day
# multiplier moves.
MOM_DAYS      = (1, 3, 5)
FWD_DAYS      = (1, 3, 5)

print(f'R = {R_SEED_SETS} disjoint seed sets   ZigZag = {ZZ_PCT}%   '
      f'G3 W<={W_GUARD_MAX}   G1 occ in [{OCC_MIN_PCT},{OCC_MAX_PCT}]%   '
      f'G4 SIDEWAYS<={SIDEWAYS_MAX}%')
print(f'momentum ladder {MOM_DAYS} days   CONTEXT_DAYS={CONTEXT_DAYS}   '
      f'forward horizons {FWD_DAYS} days')
print('All declared before a single run exists.')
""")

# The two harvested metric cells sit HERE, after their parameters exist:
# zigzag_pivots' signature captures ZZ_PCT as a default argument, and
# evaluate_guards reads OCC_MIN_PCT / W_GUARD_MAX / SIDEWAYS_MAX.
co(BANNER
   + f'# ZIGZAG -- INLINED VERBATIM from {SL_GEN.split("/")[-1]}.\n'
   + '# Label-blind, retrospective. Registers itself in ZZ_CALLS/ZZ_IDS/ZZ_ARRAYS\n'
   + '# so the leakage tripwire below can prove it never reached a label.\n'
   + '# That generator ships its own tripwire in the SAME cell, hard-coded to\n'
   + '# bar_dir_weight == 0.0; it is cut off here because this notebook installs\n'
   + '# the same four checks anchored to the frozen BAR_DIR_WEIGHT (0.5) instead.\n'
   + BANNER + SRC_ZIGZAG)

co(BANNER
   + f'# S / L / W + GUARDS G1-G4 -- INLINED VERBATIM from {SL_GEN.split("/")[-1]}.\n'
   + '# Identical definitions to the stability/lag and generalization notebooks,\n'
   + '# so the numbers in all three are directly comparable.\n'
   + BANNER + SRC_SLW)

co(r"""
# ===========================================================================
# THE FIXED-KNOB INVARIANT.
# Every constant GENERALIZATION_PROTOCOL.md forbids re-tuning is captured here
# and re-asserted after EVERY run. If any of them moves, the whole test is void
# and this assert says so.
# ===========================================================================
FIXED_KNOBS = dict(
    N_STATES               = 5,
    COVARIANCE             = 'diag',
    N_FEATURES             = len(FEATURE_COLS),
    FEATURE_COLS           = tuple(FEATURE_COLS),
    CONF_L                 = CONF_L,
    CONFIRM_BARS           = CONFIRM_BARS,
    BAR_DIR_WEIGHT         = BAR_DIR_WEIGHT,
    ENSEMBLE_K             = ENSEMBLE_K,
    Z_HI                   = Z_HI,
    EFF_HI                 = EFF_HI,
    Z_HI_EXIT              = Z_HI_EXIT,
    EFF_HI_EXIT            = EFF_HI_EXIT,
    EFF_WIN                = EFF_WIN,
    MOM_DAYS               = MOM_DAYS,
    CONTEXT_DAYS           = CONTEXT_DAYS,
    INTENSITY_MODE         = INTENSITY_MODE,
    ESCALATION_DURING_HOLD = ESCALATION_DURING_HOLD,
    DIRECTION_MODE         = DIRECTION_MODE,
    TRAIN_FRACTION         = TRAIN_FRACTION,
    N_FOLDS                = N_FOLDS,
    DIRECTION_EXCLUDE      = tuple(DIRECTION_EXCLUDE),
    H_TARGET_RATE          = H_TARGET_RATE,
    H_EXIT_SLACK           = H_EXIT_SLACK,
    BASE_SEED              = BASE_SEED,
)

# The protocol's literal values, hard-coded, so this notebook fails loudly if
# the harvested master ever drifts away from the frozen spec.
PROTOCOL_VALUES = dict(
    N_STATES=5, COVARIANCE='diag', N_FEATURES=9, CONF_L=0.50, CONFIRM_BARS=2,
    BAR_DIR_WEIGHT=0.5, ENSEMBLE_K=4, Z_HI=0.5, EFF_HI=0.35, Z_HI_EXIT=0.35,
    EFF_HI_EXIT=0.25, EFF_WIN=9, CONTEXT_DAYS=12, INTENSITY_MODE='frozen_z',
    ESCALATION_DURING_HOLD='allow', DIRECTION_MODE='rank',
    TRAIN_FRACTION=0.70, N_FOLDS=4,
)
_bad = {k: (FIXED_KNOBS[k], v) for k, v in PROTOCOL_VALUES.items()
        if FIXED_KNOBS[k] != v}
assert not _bad, ('the harvested engine DISAGREES with GENERALIZATION_PROTOCOL.md '
                  f'on {_bad} -- the generalization test is VOID until resolved')
print('ASSERT OK: every frozen constant in the harvested engine matches '
      'GENERALIZATION_PROTOCOL.md exactly.')


def assert_knobs_unchanged(where):
    '''Re-read every frozen knob out of the live globals and compare.'''
    g = globals()
    now = dict(FIXED_KNOBS)
    for k in ('CONF_L', 'CONFIRM_BARS', 'BAR_DIR_WEIGHT', 'ENSEMBLE_K', 'Z_HI',
              'EFF_HI', 'Z_HI_EXIT', 'EFF_HI_EXIT', 'EFF_WIN', 'CONTEXT_DAYS',
              'INTENSITY_MODE', 'ESCALATION_DURING_HOLD', 'DIRECTION_MODE',
              'TRAIN_FRACTION', 'N_FOLDS', 'H_TARGET_RATE', 'H_EXIT_SLACK',
              'BASE_SEED'):
        now[k] = g[k]
    now['FEATURE_COLS'] = tuple(g['FEATURE_COLS'])
    now['N_FEATURES'] = len(g['FEATURE_COLS'])
    now['DIRECTION_EXCLUDE'] = tuple(g['DIRECTION_EXCLUDE'])
    diff = {k: (FIXED_KNOBS[k], now[k]) for k in FIXED_KNOBS if FIXED_KNOBS[k] != now[k]}
    assert not diff, (f'FIXED KNOB DRIFTED at {where}: {diff}. A knob was '
                      're-tuned per market -- the generalization test is VOID.')


assert_knobs_unchanged('declaration')
print(f'fixed-knob invariant armed over {len(FIXED_KNOBS)} constants; '
      'checked after every run.')
""")

co(r"""
# ===========================================================================
# BARS_PER_DAY: window rebuilding. The ONLY thing that legitimately changes
# per run, and it changes by the master's OWN formula with the SAME day counts.
# ===========================================================================
BASE_WIN_MASTER = dict(BASE_WIN)     # the master's 2h Nifty windows, for reference


def windows_for(bpd):
    '''The master's BASE_WIN formula, evaluated at a different bars-per-day.'''
    return dict(MOM_1D=MOM_DAYS[0] * bpd, MOM_3D=MOM_DAYS[1] * bpd,
                MOM_5D=MOM_DAYS[2] * bpd,
                VOL_WIN=BASE_WIN_MASTER['VOL_WIN'],      # raw bar counts in the
                VOL_FAST=BASE_WIN_MASTER['VOL_FAST'],    # master; left alone
                VOL_SLOW=CONTEXT_DAYS * bpd,
                SWING_WIN=CONTEXT_DAYS * bpd)


assert windows_for(3) == BASE_WIN_MASTER, \
    'windows_for(3) must reproduce the master BASE_WIN bit for bit'
print('ASSERT OK: windows_for(BARS_PER_DAY=3) reproduces the master BASE_WIN '
      f'exactly -> {BASE_WIN_MASTER}')


@contextlib.contextmanager
def bars_per_day(bpd):
    '''Install this run's day-denominated windows. Restored on exit.'''
    g = globals()
    old_win, old_bpd = g['BASE_WIN'], g['BARS_PER_DAY']
    g['BASE_WIN'], g['BARS_PER_DAY'] = windows_for(bpd), bpd
    try:
        yield g['BASE_WIN']
    finally:
        g['BASE_WIN'], g['BARS_PER_DAY'] = old_win, old_bpd


def fwd_horizons_for(bpd):
    return [max(1, d * bpd) for d in FWD_DAYS]


assert fwd_horizons_for(3) == FWD_HORIZONS, \
    'fwd_horizons_for(3) must reproduce the master FWD_HORIZONS'
print(f'ASSERT OK: fwd_horizons_for(3) == master FWD_HORIZONS == {FWD_HORIZONS}')
print()
print('INTRADAY FX windows that will actually be used:')
for _tf, _b in (('1h', 24), ('2h', 12)):
    _w = windows_for(_b)
    print(f"  {_tf}  BARS_PER_DAY={_b:>3}  momentum {_w['MOM_1D']}/{_w['MOM_3D']}/"
          f"{_w['MOM_5D']} bars   context {_w['SWING_WIN']} bars   "
          f'fwd horizons {fwd_horizons_for(_b)} bars')
print()
print(f'EFF_WIN stays FROZEN at {EFF_WIN} BARS on every run (protocol decree), '
      'i.e. 9h at')
print('1h and 18h at 2h. The chop filter\'s REAL-TIME horizon therefore VARIES '
      'across')
print('the timeframe axis. Same for VOL_WIN=%d and VOL_FAST=%d. Flagged, not fixed.'
      % (BASE_WIN_MASTER['VOL_WIN'], BASE_WIN_MASTER['VOL_FAST']))
""")

co(r"""
# ===========================================================================
# THE LEAKAGE TRIPWIRE. `label_bars` is SHADOWED by a guard; every later call
# in this notebook goes through it. Four independent runtime checks.
# ===========================================================================
_label_bars_raw = label_bars
LABEL_CALLS = 0
LABEL_PHASE_OPEN = True


@contextlib.contextmanager
def reopen_label_phase(why):
    '''EXPLICITLY and LOUDLY reopen the label phase after the ZigZag exists.

    Used only by the truncation probes, whose whole job is to re-label a cut
    series. Only the PHASE-ORDERING check is relaxed; identity, aliasing and
    the bar_dir_weight decree stay armed, and the reopening prints itself so it
    can never be a silent hole in the proof.
    '''
    global LABEL_PHASE_OPEN
    print('!' * 78)
    print(f'!! LABEL PHASE EXPLICITLY REOPENED: {why}')
    print('!! ordering check relaxed; identity / aliasing / w-decree STAY ARMED')
    print('!' * 78)
    LABEL_PHASE_OPEN = True
    try:
        yield
    finally:
        LABEL_PHASE_OPEN = False
        print('!! LABEL PHASE CLOSED AGAIN -- ordering check re-armed.')


def label_bars(*args, **kwargs):
    '''Guarded shadow of the engine's label_bars. Adds asserts, changes nothing.'''
    global LABEL_CALLS
    assert ZZ_CALLS == 0 or LABEL_PHASE_OPEN, (
        'LEAKAGE: a label was produced AFTER a ZigZag had been computed.')
    _bw = kwargs.get('bar_dir_weight', BAR_DIR_WEIGHT)
    assert _bw == BAR_DIR_WEIGHT, \
        f'BAR_DIR_WEIGHT drifted to {_bw} inside a labelling call'
    for v in list(args) + list(kwargs.values()):
        assert id(v) not in ZZ_IDS, 'LEAKAGE: a ZigZag output was passed to label_bars'
        if isinstance(v, np.ndarray):
            for z in ZZ_ARRAYS:
                assert not np.shares_memory(v, z), \
                    'LEAKAGE: a label_bars argument aliases ZigZag memory'
    LABEL_CALLS += 1
    return _label_bars_raw(*args, **kwargs)


print('leakage tripwire ready.')
print('  check 1  phase ordering  : label_bars asserts ZZ_CALLS == 0')
print('  check 2  object identity : every argument checked against ZZ_IDS')
print('  check 3  memory aliasing : np.shares_memory vs every ZigZag array')
print(f'  check 4  decree          : bar_dir_weight == BAR_DIR_WEIGHT == {BAR_DIR_WEIGHT}')
""")

# ===========================================================================
md(r"""
## 3. PRE-REGISTERED PREDICTIONS — printed **before** a single number exists

These are written down here, graded in section 9, and **not reinterpreted after
seeing their numbers**. A refutation is a result; it is not softened.

P4 needs a precise rule, so one is fixed here before any run: each of the 6 runs
is described by the vector `[SIDEWAYS%, S, W, L, IS_t, OOS_t]`, **z-scored
column-wise across the 6 runs** so no unit dominates. P4 compares the mean
Euclidean distance of the **3 same-pair pairings** (EURUSD 1h↔2h, GBPUSD 1h↔2h,
USDJPY 1h↔2h) against the mean distance of the **6 same-timeframe cross-pair
pairings** (3 pair-combinations × 2 timeframes). If pair identity matters more
than bar size, the same-pair distance is the smaller one.
""")

co(r"""
# ===========================================================================
# PRE-REGISTERED PREDICTIONS. Printed BEFORE any data is loaded or fitted.
# ===========================================================================
P4_METRICS = ('SIDEWAYS%', 'S', 'W', 'L', 'IS_t', 'OOS_t')

PREDICTIONS = {
 'P1': ('Guards G1-G4 pass on a MAJORITY of the 6 runs.',
        'CONFIRMED if >= 4 of 6 runs pass ALL FOUR guards. Broad failure means '
        'the architecture is Nifty-specific.'),
 'P2': ('SIDEWAYS occupancy lands roughly 20-45% across runs.',
        f'CONFIRMED if a MAJORITY (>= 4 of 6) of runs have SIDEWAYS in '
        f'[{P2_LO:.0f}%, {P2_HI:.0f}%]. Wild swings mean the thresholds are '
        'scale-dependent.'),
 'P3': ('OOS forward-return ordering DEGRADES vs IS but does NOT INVERT sign.',
        'CONFIRMED if the OOS (BULL - BEAR) mean-forward-return contrast keeps '
        'the SAME SIGN as IS on a MAJORITY (>= 4 of 6) of runs. Inversion on a '
        'majority = overfitting.'),
 'P4': ('The 1h and 2h results for the SAME pair agree with each other MORE '
        'than different pairs at the same timeframe do (pair identity matters '
        'more than bar size).',
        f'Each run -> the vector {list(P4_METRICS)}, z-scored COLUMN-WISE across '
        'the 6 runs. CONFIRMED if the MEAN Euclidean distance over the 3 '
        'SAME-PAIR pairings (1h vs 2h) is STRICTLY LESS than the MEAN distance '
        'over the 6 SAME-TIMEFRAME cross-pair pairings.'),
}
print('=' * 110)
print('PRE-REGISTERED PREDICTIONS  (frozen; graded in section 9; not softened)')
print('=' * 110)
for k, (claim, rule) in PREDICTIONS.items():
    print(f'\n{k}.  {claim}')
    print(f'     GRADING RULE: {rule}')
print('\n' + '=' * 110)
print('ALSO PRE-REGISTERED, as a plain-words comparison rather than a pass/fail:')
print('  generalization.ipynb Grid B found 4 of 6 REAL DAILY FX runs INVERTING')
print('  sign between IS and OOS. Section 9 states explicitly whether this')
print('  INTRADAY grid AGREES or DISAGREES with that daily finding.')
print()
print('NOT PREDICTED, and therefore not gradeable: absolute return, Sharpe, or')
print('any tradeability claim. This notebook measures the DETECTOR, not a strategy.')
print('=' * 110)
""")

# ===========================================================================
md(r"""
## 4. Data layer — REAL intraday FX, verified live

### Provenance, stated plainly

> **COMMUNITY GITHUB DATA — verified against Brexit/COVID, NOT an official feed.**
>
> `raw.githubusercontent.com/ejtraderLabs/historical-data` is a community
> repository, not an exchange, a broker of record, or a central bank. It is used
> here because it is the only real long-history intraday source reachable from
> this sandbox, and because its content **can be and is** checked live against
> events whose magnitudes are independently known. It is **not** a substitute
> for a licensed feed.

### Integer price scaling — detected, not assumed

The files store prices as integers at a **per-pair** scale. The loader picks the
divisor from `{1, 10, 10², 10³, 10⁴, 10⁵, 10⁶}` by testing the **median close**
against a plausible band per pair, prints the detected divisor, and asserts the
whole rescaled series stays inside a generous version of that band.

**Scaling cannot change any result.** Every feature the engine builds is a
function of log returns, ratios or z-scores, all of which are invariant to
multiplying the series by a constant. The divisor matters for the *charts* and
for the *sanity assert*, and for nothing else.

### Verification re-run live in this notebook

Structural: 57,600 bars per pair, `2012-11-16 → 2022-03-04`, monotone index, 0
duplicate timestamps, all prices positive, OHLC internally consistent
(`high ≥ max(open,close)`, `low ≤ min(open,close)`).

Event: **Brexit** (GBPUSD 2016-06-23/24, ≈ 1.5006 → 1.3379, ≈ −10.8%) and
**COVID** (EURUSD March 2020, range ≈ 1.0654–1.1470, ≈ 7.7% swing). Hourly
realised volatility is printed per pair.
""")

co(r"""
# ===========================================================================
# INTRADAY FX LOADER -- REAL DATA, COMMUNITY GITHUB SOURCE.
# NO SYNTHETIC FALLBACK. If the fetch fails, the runs appear as errors.
# ===========================================================================
FX_BASE = ('https://raw.githubusercontent.com/ejtraderLabs/historical-data/'
           'main/{sym}/{sym}{tf}.csv')
FX_SOURCE_LABEL = ('COMMUNITY GITHUB DATA (ejtraderLabs/historical-data) '
                   '- verified against Brexit/COVID, NOT an official feed')

# Plausible MEDIAN band per pair, used to DETECT the integer scaling. These are
# wide, coarse, textbook ranges -- they are a scale detector, not a fit.
FX_PAIRS = [
    dict(sym='EURUSD', name='EUR/USD', med_band=(0.9, 1.7)),
    dict(sym='GBPUSD', name='GBP/USD', med_band=(0.9, 1.7)),
    dict(sym='USDJPY', name='USD/JPY', med_band=(75.0, 160.0)),
]
FX_TFS = [('1h', 1, 24), ('2h', 2, 12)]     # (name, hours, assumed BARS_PER_DAY)

SCALE_CANDIDATES = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
OHLC_AGG = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'tick_volume': 'sum'}


def detect_divisor(close_raw, med_band, sym):
    '''Pick the power-of-ten divisor that puts the MEDIAN close in `med_band`.'''
    med = float(np.median(close_raw))
    hits = [d for d in SCALE_CANDIDATES if med_band[0] <= med / d <= med_band[1]]
    assert len(hits) == 1, (
        f'{sym}: scale detection is AMBIGUOUS -- median {med} lands in '
        f'{med_band} for divisors {hits}. Refusing to guess.')
    return hits[0]


def load_fx_h1(p):
    '''Fetch one pair's hourly OHLC, detect + apply the scale, verify.'''
    url = FX_BASE.format(sym=p['sym'], tf='h1')
    df = pd.read_csv(url, parse_dates=['Date'])
    assert list(df.columns) == ['Date', 'open', 'high', 'low', 'close',
                                'tick_volume'], \
        f"{p['sym']}: unexpected columns {list(df.columns)}"
    df = df.set_index('Date').sort_index()
    div = detect_divisor(df['close'].values, p['med_band'], p['sym'])
    px = df[['open', 'high', 'low', 'close']] / div
    # generous sanity band -- the DETECTOR used the median; this checks the
    # WHOLE series did not land somewhere absurd.
    lo, hi = p['med_band'][0] / 1.5, p['med_band'][1] * 1.5
    assert lo <= px.values.min() and px.values.max() <= hi, (
        f"{p['sym']}: rescaled series [{px.values.min():.4f}, "
        f"{px.values.max():.4f}] escapes the sanity band [{lo:.4f}, {hi:.4f}] "
        f'at divisor {div:g}')
    out = px.copy()
    out['tick_volume'] = df['tick_volume'].values
    return out, div


FX_OK, FX_FAIL = True, ''
FX_H1, FX_DIV = {}, {}
try:
    _t = time.time()
    for _p in FX_PAIRS:
        FX_H1[_p['sym']], FX_DIV[_p['sym']] = load_fx_h1(_p)
    print(f'fetched 3 hourly FX files in {time.time() - _t:.1f}s')
except Exception as e:
    FX_OK, FX_FAIL = False, f'{type(e).__name__}: {e}'
    print(f'FX FETCH FAILED: {FX_FAIL}')
    print('The 6 runs will appear in the tables with this reason.')
    print('NOTHING IS SUBSTITUTED. There is no synthetic fallback in this notebook.')

print()
print('=' * 108)
print('PROVENANCE:  ' + FX_SOURCE_LABEL)
print('=' * 108)
print('*** THE DATA ENDS 2022-03-04. This notebook says NOTHING about 2022-2026. ***')
print()
if FX_OK:
    print(f"{'pair':<9} {'divisor':>9} {'bars':>7}  {'span':<44} "
          f"{'median':>9} {'min':>9} {'max':>9}")
    for p in FX_PAIRS:
        d = FX_H1[p['sym']]
        print(f"{p['name']:<9} {FX_DIV[p['sym']]:>9.0f} {len(d):>7}  "
              f"{d.index[0]:%Y-%m-%d %H:%M} -> {d.index[-1]:%Y-%m-%d %H:%M}"
              f"{'':<8}"
              f"{d['close'].median():>9.4f} {d['close'].min():>9.4f} "
              f"{d['close'].max():>9.4f}")
    print()
    print('NOTE: dividing every price by a constant leaves LOG RETURNS unchanged,')
    print('so the divisor cannot alter one feature, one label or one statistic.')
    print('It exists so the charts are readable and so the sanity assert can run.')
""")

co(r"""
# ===========================================================================
# LIVE DATA VERIFICATION. Structural checks + known-event checks.
# Nothing is trusted on the strength of somebody having checked it before.
# ===========================================================================
FX_CHECKS = []
if FX_OK:
    def _chk(name, ok, detail):
        FX_CHECKS.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")

    print('STRUCTURAL CHECKS')
    for p in FX_PAIRS:
        d = FX_H1[p['sym']]
        _chk(f"{p['name']} 57,600 hourly bars", len(d) == 57600, f'{len(d)} bars')
        _chk(f"{p['name']} index monotone, no duplicate timestamps",
             d.index.is_monotonic_increasing and not d.index.duplicated().any(),
             f'{d.index[0]:%Y-%m-%d} -> {d.index[-1]:%Y-%m-%d}')
        _chk(f"{p['name']} all prices strictly positive and finite",
             bool((d[['open', 'high', 'low', 'close']] > 0).values.all())
             and bool(np.isfinite(d[['open', 'high', 'low', 'close']].values).all()),
             f"min {d[['open', 'high', 'low', 'close']].values.min():.4f}")
        _hi_ok = (d['high'].values >= np.maximum(d['open'].values, d['close'].values) - 1e-12).all()
        _lo_ok = (d['low'].values <= np.minimum(d['open'].values, d['close'].values) + 1e-12).all()
        _chk(f"{p['name']} OHLC internally consistent",
             bool(_hi_ok and _lo_ok and (d['high'] >= d['low']).all()),
             'high >= max(o,c), low <= min(o,c), high >= low')

    print('\nHOURLY REALISED VOLATILITY (sd of hourly log returns)')
    for p in FX_PAIRS:
        _r = np.diff(np.log(FX_H1[p['sym']]['close'].values))
        print(f"  {p['name']:<9} {_r.std() * 100:.3f}%")

    print('\nEVENT CHECKS against independently known magnitudes')
    _g = FX_H1['GBPUSD']['close'].loc['2016-06-23':'2016-06-24']
    _mv = (_g.min() / _g.max() - 1.0) * 100.0
    _chk('BREXIT GBPUSD 2016-06-23/24 crash',
         abs(_mv - (-10.84)) < 0.30 and abs(_g.max() - 1.5006) < 0.01
         and abs(_g.min() - 1.3379) < 0.01,
         f'{_g.max():.4f} -> {_g.min():.4f} = {_mv:.2f}%  (expected '
         f'1.5006 -> 1.3379 = -10.84%)')

    _e = FX_H1['EURUSD']['close'].loc['2020-03']
    _sw = (_e.max() / _e.min() - 1.0) * 100.0
    _chk('COVID EURUSD March-2020 range',
         abs(_e.min() - 1.0654) < 0.01 and abs(_e.max() - 1.1470) < 0.01
         and abs(_sw - 7.7) < 0.5,
         f'{_e.min():.4f} - {_e.max():.4f} ({_sw:.1f}% swing)  '
         f'(expected 1.0654 - 1.1470, 7.7%)')

    _np_ = sum(1 for _, o, _ in FX_CHECKS if o)
    print(f'\nDATA VERIFICATION: {_np_}/{len(FX_CHECKS)} checks PASS')
    assert _np_ == len(FX_CHECKS), (
        'a data verification check FAILED -- the source is not what it was '
        'verified to be. STOP: do not proceed on data you cannot confirm.')
    print('ASSERT OK: every structural and event check passed on data fetched '
          'in THIS run.')
    print()
    print('This is a COMMUNITY repository. What these checks establish is that '
          'its')
    print('content reproduces two large, independently-known FX moves to within '
          'a few')
    print('basis points, and that it is structurally clean. They do NOT make it '
          'an')
    print('official feed, and it is not labelled as one anywhere in this notebook.')
else:
    print('data unavailable -- verification skipped, runs will be error rows.')
""")

co(r"""
# ===========================================================================
# DOWNWARD-ONLY RESAMPLING + the causal realised-vol VIX proxy.
# ===========================================================================
def resample_ohlc_down(df, hours):
    '''Downward OHLC aggregation from hourly bars. NEVER upward.

    open=first, high=max, low=min, close=last, tick_volume=sum.
    Asserts (a) the output cadence is COARSER than the input, and (b) the bar
    count falls by roughly the expected factor. Finer bars are never
    manufactured from coarser ones.
    '''
    assert hours >= 2, 'resample_ohlc_down is for COARSENING only'
    step_in = pd.Series(df.index).diff().median()
    out = df.resample(f'{hours}h').agg(OHLC_AGG).dropna(how='any')
    step_out = pd.Series(out.index).diff().median()
    assert step_out > step_in, (f'UPWARD resampling attempted: {step_in} -> '
                                f'{step_out}. Only downward is allowed.')
    ratio = len(out) / len(df)
    assert 0.35 <= ratio * hours <= 1.15, (
        f'{hours}h resample produced {len(out)} bars from {len(df)} '
        f'(x{ratio:.3f}); expected roughly x{1 / hours:.3f}')
    assert (out['high'].values >= np.maximum(out['open'].values, out['close'].values) - 1e-12).all()
    assert (out['low'].values <= np.minimum(out['open'].values, out['close'].values) + 1e-12).all()
    return out, ratio


def realised_vol_proxy(close, bpd, win=None):
    '''CAUSAL trailing realised-volatility stand-in for a VIX.

    ann.vol(t) = sd(log returns over the last `win` bars, inclusive of t)
                 * sqrt(bpd * 252) * 100

    Reads bars <= t only: the return at bar t is log(C[t]/C[t-1]) and the
    rolling window ends at t. There is no shift, no centring and no bfill.
    This is NOT a VIX. It is labelled `vol-proxy` everywhere it is used.
    '''
    win = int(BASE_WIN['VOL_SLOW'] if win is None else win)
    r = np.log(close / close.shift(1))
    v = r.rolling(win).std() * np.sqrt(max(bpd, 1) * 252.0) * 100.0
    v = v.replace([np.inf, -np.inf], np.nan)
    # leading NaNs are left as NaN -- build_features' dropna() removes exactly
    # those bars. NEVER bfill: that is the look-ahead the master already fixed.
    return v.rename('vix')


def realised_bpd(index):
    '''Median number of bars per calendar SESSION actually present in the data.'''
    if len(index) == 0:
        return np.nan
    per_day = pd.Series(1, index=index).groupby(index.normalize()).sum()
    return float(per_day.median())


if FX_OK:
    print('DOWNWARD RESAMPLE CHECK (1h source -> 2h)')
    print(f"{'pair':<9} {'1h bars':>8} {'2h bars':>8} {'ratio':>7}  cadence")
    FX_H2 = {}
    for p in FX_PAIRS:
        d2, rr = resample_ohlc_down(FX_H1[p['sym']], 2)
        FX_H2[p['sym']] = d2
        _s1 = pd.Series(FX_H1[p['sym']].index).diff().median()
        _s2 = pd.Series(d2.index).diff().median()
        print(f"{p['name']:<9} {len(FX_H1[p['sym']]):>8} {len(d2):>8} "
              f"{rr:>7.3f}  {_s1} -> {_s2}")
    print('\nASSERT OK: every 2h series is a DOWNWARD aggregation of its own 1h')
    print('file, the cadence strictly coarsens, and the bar count roughly halves.')
    print('No finer bar is fabricated from a coarser one anywhere in this notebook.')

print()
print('VOLATILITY INPUT for every run: realised-vol PROXY (causal).')
print('Spot FX has NO natural VIX. The vix_chg feature is fed a trailing')
print('realised-volatility proxy that reads bars <= t only. It is NOT a VIX and')
print('is stamped as a proxy on every run and in every table.')
""")

# ===========================================================================
md(r"""
## 5. Runtime measurement — measured first, then extrapolated, then run

The grid is 6 runs × R=4 disjoint seed sets × K=4 = **96 HMM fits** on up to
~40,000 training rows. That is the real risk in this notebook, so it is
**measured before it is launched**: one seed set is fitted on the *largest* run,
the total is extrapolated, and the estimate is printed.

**The only sanctioned reduction** if the projection is excessive is a cap on the
training-window length (fit on the most recent *N* bars, applied forward).
`R = 4` is never cut — `S` is meaningless without it — and no pair or timeframe
is ever dropped. Whether the cap was applied is printed either way.
""")

co(r"""
# ===========================================================================
# THE RUN HARNESS. Fits + labels ONE (pair, timeframe) run.
# ===========================================================================
N_STATES_FIXED = FIXED_KNOBS['N_STATES']
COV_FIXED = FIXED_KNOBS['COVARIANCE']

FIT_COUNT = 0

# TRAIN_CAP_BARS: None = no cap (fit on the full leading TRAIN_FRACTION window).
# Set only by the runtime probe below, and only if the projection is excessive.
TRAIN_CAP_BARS = None
RUNTIME_BUDGET_S = 15 * 60


def seed_sets(K=ENSEMBLE_K, R=R_SEED_SETS, base=BASE_SEED):
    ss = [list(range(base + r * K, base + r * K + K)) for r in range(R)]
    flat = [s for st in ss for s in st]
    assert len(set(flat)) == len(flat), 'seed sets must be DISJOINT'
    return ss


SEED_SETS = seed_sets()
print(f'R={R_SEED_SETS} DISJOINT seed sets, K={ENSEMBLE_K}: {SEED_SETS}')


def prepare_run(close, volser, bpd):
    '''Features + scaler + fit window for one run. Causal throughout.'''
    with bars_per_day(bpd):
        feat = build_features(close, volser, LOOKBACK_SCALE)
    if len(feat) < 200:
        raise ValueError(f'only {len(feat)} feature bars after warmup (need >= 200)')
    dates = feat.index
    n_bars = len(dates)
    n_fit = max(int(n_bars * TRAIN_FRACTION), 50)
    raw = feat[FEATURE_COLS].values
    # The fit window is the LEADING n_fit bars. A cap, if armed, shortens it
    # from the LEFT (most recent bars kept), and is applied FORWARD identically
    # on every run. n_fit itself -- the IS/OOS boundary -- never moves.
    fit_lo = 0 if TRAIN_CAP_BARS is None else max(0, n_fit - int(TRAIN_CAP_BARS))
    sc = StandardScaler().fit(raw[fit_lo:n_fit])
    Xs = sc.transform(raw)
    return dict(feat=feat, dates=dates, n_bars=n_bars, n_fit=n_fit, fit_lo=fit_lo,
                Xs=Xs, scaler=sc,
                close=close.reindex(dates), trend_raw=feat[TREND_FEATURE].values)


def label_run(prep, close, bpd, models_by_set=None):
    '''Label the full series once per seed set. Returns (label_sets, models).'''
    global FIT_COUNT
    out, mods = [], []
    for si, sd in enumerate(SEED_SETS):
        if models_by_set is None:
            ms, _conv = fit_hmm_ensemble(prep['Xs'][prep['fit_lo']:prep['n_fit']],
                                         N_STATES_FIXED, COV_FIXED,
                                         K=ENSEMBLE_K, base_seed=sd[0])
            FIT_COUNT += ENSEMBLE_K
        else:
            ms = models_by_set[si]
        mods.append(ms)
        with bars_per_day(bpd):
            lab = label_bars(ms, prep['Xs'], prep['dates'], close,
                             prep['trend_raw'], prep['n_fit'], FEATURE_COLS,
                             dir_feats=prep['feat'])
        out.append(lab['tactical_regime_state'].values.copy())
    return out, mods


def build_run(run_id, pair, tf, close, volser, vix_kind, bpd_assumed,
              source, note=''):
    '''Full LABEL-PHASE work for one run. Returns a run dict (or an error row).'''
    rec = dict(run_id=run_id, market=pair, tf=tf, vix_kind=vix_kind,
               source=source, note=note, bpd_assumed=bpd_assumed,
               bpd_realised=realised_bpd(close.index), error=None)
    rec['_raw_close'] = close
    rec['_raw_vol'] = volser
    try:
        prep = prepare_run(close, volser, bpd_assumed)
        labels, models = label_run(prep, close, bpd_assumed)
        rec.update(labels=labels, models=models,
                   dates=prep['dates'], close_arr=prep['close'].values,
                   close_ser=prep['close'], feat=prep['feat'],
                   Xs=prep['Xs'], scaler=prep['scaler'],
                   n_bars=prep['n_bars'], n_fit=prep['n_fit'],
                   fit_lo=prep['fit_lo'],
                   span=(prep['dates'][0], prep['dates'][-1]),
                   win=windows_for(bpd_assumed),
                   fwd_h=fwd_horizons_for(bpd_assumed))
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
    assert_knobs_unchanged(f'after run {run_id}')
    return rec


print('run harness ready: prepare_run / label_run / build_run.')
print('Per-run inputs are ONLY (close, volatility, BARS_PER_DAY). No knob is a '
      'function of the pair.')
""")

co(r"""
# ===========================================================================
# RUNTIME PROBE. Measure ONE seed set on the LARGEST run, extrapolate the whole
# grid, PRINT the estimate -- before launching anything.
# ===========================================================================
RUNTIME_EST = None
if FX_OK:
    _p0 = FX_PAIRS[0]
    _c0 = FX_H1[_p0['sym']]['close'].astype(float)
    _c0.name = _p0['sym']
    with bars_per_day(24):
        _v0 = realised_vol_proxy(_c0, 24)
    _t = time.time()
    _prep0 = prepare_run(_c0, _v0, 24)
    _t_prep = time.time() - _t

    _t = time.time()
    _m0, _ = fit_hmm_ensemble(_prep0['Xs'][_prep0['fit_lo']:_prep0['n_fit']],
                              N_STATES_FIXED, COV_FIXED, K=ENSEMBLE_K,
                              base_seed=SEED_SETS[0][0])
    _t_set = time.time() - _t
    FIT_COUNT += ENSEMBLE_K

    _t = time.time()
    with bars_per_day(24):
        _l0 = label_bars(_m0, _prep0['Xs'], _prep0['dates'], _c0,
                         _prep0['trend_raw'], _prep0['n_fit'], FEATURE_COLS,
                         dir_feats=_prep0['feat'])
    _t_lab = time.time() - _t

    _n_fit_1h = _prep0['n_fit']
    _per_run_1h = R_SEED_SETS * (_t_set + _t_lab) + _t_prep
    # EM cost is ~linear in training rows; a 2h run has ~half of them.
    _per_run_2h = 0.5 * R_SEED_SETS * _t_set + 0.5 * R_SEED_SETS * _t_lab + _t_prep
    RUNTIME_EST = 3 * _per_run_1h + 3 * _per_run_2h

    print('=' * 100)
    print('RUNTIME PROBE -- measured on the LARGEST run before anything is launched')
    print('=' * 100)
    print(f"  probe run              : {_p0['name']} @ 1h")
    print(f"  feature bars           : {_prep0['n_bars']}   training rows: "
          f"{_n_fit_1h - _prep0['fit_lo']}")
    print(f'  prepare_run            : {_t_prep:.1f}s')
    print(f'  ONE seed set (K={ENSEMBLE_K} fits): {_t_set:.1f}s   '
          f'({_t_set / ENSEMBLE_K:.1f}s per HMM fit)')
    print(f'  label_bars full series : {_t_lab:.1f}s')
    print()
    print(f'  projected per 1h run   : {_per_run_1h:6.0f}s  '
          f'(R={R_SEED_SETS} seed sets)')
    print(f'  projected per 2h run   : {_per_run_2h:6.0f}s  '
          '(~half the training rows)')
    print(f'  PROJECTED GRID TOTAL   : {RUNTIME_EST:6.0f}s  '
          f'= {RUNTIME_EST / 60:.1f} min   '
          f'({R_SEED_SETS * ENSEMBLE_K * 6} HMM fits)')
    print(f'  budget                 : {RUNTIME_BUDGET_S / 60:.0f} min')
    print()
    if RUNTIME_EST > RUNTIME_BUDGET_S:
        TRAIN_CAP_BARS = 20000
        print(f'  >>> PROJECTION EXCEEDS BUDGET. Applying the ONLY sanctioned')
        print(f'  >>> reduction: TRAIN_CAP_BARS = {TRAIN_CAP_BARS}. Each run fits')
        print(f'  >>> on the most recent {TRAIN_CAP_BARS} bars of its own leading')
        print(f'  >>> {TRAIN_FRACTION:.0%} window, applied FORWARD. The IS/OOS boundary')
        print(f'  >>> (n_fit) does NOT move. R={R_SEED_SETS} is NOT cut and no pair')
        print(f'  >>> or timeframe is dropped.')
    else:
        print(f'  >>> PROJECTION IS WITHIN BUDGET. TRAIN_CAP_BARS stays None:')
        print(f'  >>> every run fits on its FULL leading {TRAIN_FRACTION:.0%} window.')
        print(f'  >>> Nothing is capped, nothing is cut, nothing is dropped.')
    print('=' * 100)
    del _m0, _l0, _prep0
else:
    print('data unavailable -- runtime probe skipped.')
""")

# ===========================================================================
md(r"""
## 6. Label phase — 6 runs

Every label in this notebook is produced in the **LABEL PHASE**, which closes
before the first ZigZag is computed. The tripwire proves it. All 6 runs are
labelled first; only then does evaluation begin.
""")

co(r"""
# ===========================================================================
# LABEL PHASE -- 3 pairs x {1h, 2h}. REAL DATA, NO SYNTHETIC FALLBACK.
# ===========================================================================
RUNS = []
t0 = time.time()
if not FX_OK:
    for p in FX_PAIRS:
        for tf_name, _h, _b in FX_TFS:
            RUNS.append(dict(run_id=f"{p['name']}@{tf_name}", market=p['name'],
                             sym=p['sym'], tf=tf_name, cls='FX',
                             vix_kind='realised-vol PROXY (causal)',
                             source=FX_SOURCE_LABEL, note='', bpd_assumed=_b,
                             bpd_realised=np.nan,
                             error=f'FX fetch failed: {FX_FAIL}'))
    print('FX data unavailable; 6 error rows recorded. NOTHING was substituted.')
else:
    print(f"{'run':<16} {'bars':>7} {'fit rows':>9} {'BPD':>4} "
          f"{'span':<26} {'windows (mom/ctx)':<22} {'secs':>6}")
    print('-' * 104)
    for p in FX_PAIRS:
        for tf_name, hours, bpd in FX_TFS:
            tt = time.time()
            rid = f"{p['name']}@{tf_name}"
            bars = FX_H1[p['sym']] if hours == 1 else FX_H2[p['sym']]
            close = bars['close'].astype(float)
            close.name = p['sym']
            with bars_per_day(bpd):
                vol = realised_vol_proxy(close, bpd)
            r = build_run(rid, p['name'], tf_name, close, vol,
                          'realised-vol PROXY (causal)', bpd,
                          FX_SOURCE_LABEL, note='FX')
            r['cls'] = 'FX'
            r['sym'] = p['sym']
            r['pair'] = p['name']
            RUNS.append(r)
            if r['error']:
                print(f'{rid:<16} ERROR: {r["error"]}')
            else:
                w = r['win']
                print(f"{rid:<16} {r['n_bars']:>7} "
                      f"{r['n_fit'] - r['fit_lo']:>9} {bpd:>4} "
                      f"{r['span'][0]:%Y-%m-%d} -> {r['span'][1]:%Y-%m-%d}  "
                      f"{str(w['MOM_1D']) + '/' + str(w['MOM_3D']) + '/' + str(w['MOM_5D']) + '  ctx ' + str(w['SWING_WIN']):<22} "
                      f"{time.time() - tt:>6.1f}")

print(f'\nLabel phase: {time.time() - t0:.1f}s   {FIT_COUNT} HMM fits, '
      f'{LABEL_CALLS} label_bars calls, {len(RUNS)} runs.')
if RUNTIME_EST is not None:
    print(f'(runtime probe projected {RUNTIME_EST:.0f}s for the grid; the probe '
          f'itself is included in the fit count.)')
print(f'TRAIN_CAP_BARS = {TRAIN_CAP_BARS}  '
      + ('(no cap: every run used its full leading window)' if TRAIN_CAP_BARS is None
         else f'(capped to the most recent {TRAIN_CAP_BARS} training bars)'))
assert ZZ_CALLS == 0, 'LEAKAGE: a ZigZag existed during the label phase'
print('ASSERT OK: ZZ_CALLS == 0 -- every label was produced BEFORE any ZigZag '
      'was computed.')
""")

co(r"""
# ===========================================================================
# BARS_PER_DAY: ASSUMED vs REALISED. Printed, warned on, never silently rescaled.
# ===========================================================================
print('=' * 104)
print('BARS PER SESSION -- ASSUMED vs REALISED (measured from the data itself)')
print('=' * 104)
print('Spot FX runs ~24h on WEEKDAYS, so the assumption is 1h -> 24 and 2h -> 12.')
print('Weekend days carry no bars at all and are simply absent from the index, so')
print('the realised median is taken over days that HAVE bars.')
print()
print(f"{'run':<16} {'assumed':>8} {'realised':>9} {'ratio':>7}  verdict")
BPD_WARNINGS = []
for r in RUNS:
    a, b = r['bpd_assumed'], r['bpd_realised']
    ratio = b / a if a else np.nan
    if not np.isfinite(ratio):
        v = 'UNMEASURABLE'
    elif 0.8 <= ratio <= 1.25:
        v = 'ok'
    else:
        v = ('WARNING: day-denominated windows are %.2fx the intended horizon'
             % ratio)
        BPD_WARNINGS.append((r['run_id'], a, b, ratio))
    print(f"{r['run_id']:<16} {a:>8} {b:>9.2f} {ratio:>7.2f}  {v}")

print()
if BPD_WARNINGS:
    print(f'{len(BPD_WARNINGS)} run(s) disagree materially with their assumed '
          'BARS_PER_DAY.')
    print('THE ASSUMPTION IS NOT CHANGED. Doing so would re-tune a window per')
    print('run, which the protocol forbids. The discrepancy is reported and')
    print('carried into the tables as a caveat on those rows.')
else:
    print('No material disagreement: every run realises within [0.80, 1.25]x its '
          'assumed bars per session.')
""")

# ===========================================================================
md(r"""
## 7. EVAL PHASE — S / L / W, the guards, and **the overfitting test**

The label phase is now closed. From here on any `label_bars` call fails the
tripwire unless the phase is explicitly and loudly reopened (only the truncation
probe does that).

### The overfitting test, precisely — identical to `generalization.ipynb`

For each run and each forward horizon *h* ∈ {1, 3, 5} days:

* `fwd_h[t] = C[t+h]/C[t] − 1` — a *retrospective grade*, exactly like the
  ZigZag. It grades a label that was already emitted; it never enters one.
* `BULL = {H_BULL, L_BULL}`, `BEAR = {H_BEAR, L_BEAR}`, everything else is the
  control group (kept in the design so the Newey–West lag structure sees the
  real bar spacing).
* The HAC (Newey–West, Bartlett, lag = *h*−1) *t* of (mean fwd | BULL) −
  (mean fwd | BEAR) is computed on the **IS span** and the **OOS span**
  **separately**.
* The IS span is the leading `n_fit` bars **minus an embargo of max(h) bars**,
  so no in-sample forward return can read an out-of-sample price. At 1h that
  embargo is 120 bars; at 2h, 60.
* **No pooled number is reported anywhere.**

The reported per-run figure is the mean over *h* of the HAC *t* — the master's
`W_PRIMARY_METRIC` — evaluated once per seed set and averaged, with the seed
spread carried alongside.

**Ordering** is graded on the 5-label monotonicity of mean forward return:
`H_BULL ≥ L_BULL ≥ SIDEWAYS ≥ L_BEAR ≥ H_BEAR`. `HOLDS` / `PARTIAL` (BULL block
above BEAR block but not fully monotone) / `BROKEN`.

**Inversion** — the thing P3 is about — is `sign(OOS BULL−BEAR) ≠ sign(IS
BULL−BEAR)`, evaluated on the *h*-averaged contrast.
""")

co(r"""
# ===========================================================================
# EVAL PHASE BEGINS.
# ===========================================================================
LABEL_PHASE_OPEN = False
print('LABEL PHASE CLOSED. Any label_bars call from here on fails the ordering '
      'check unless the phase is EXPLICITLY reopened.')

BULL_SET = ('H_BULL', 'L_BULL')
BEAR_SET = ('H_BEAR', 'L_BEAR')


def fwd_returns(close_arr, h):
    f = np.full(len(close_arr), np.nan)
    if h < len(close_arr):
        f[:len(close_arr) - h] = (close_arr[h:] / close_arr[:len(close_arr) - h]
                                  - 1.0) * 100.0
    return f


def span_contrast(labels, close_arr, horizons, lo, hi):
    '''HAC contrast over bars [lo, hi). Returns dict per h + the h-mean.

    The forward return is computed on the FULL close array first and then
    sliced, so a bar near the right edge of the span uses the real next price
    rather than a truncated one. The EMBARGO applied by the caller is what
    stops an IS bar from reading an OOS price.
    '''
    labels = np.asarray(labels, dtype=object)
    grp = np.where(np.isin(labels, BULL_SET), 1.0,
                   np.where(np.isin(labels, BEAR_SET), -1.0, 0.0))
    per_h, ts, ds = {}, [], []
    for h in horizons:
        f = fwd_returns(close_arr, h)
        d, se, t, npos, nneg = _w_hac_contrast(f[lo:hi], grp[lo:hi], lag=h - 1)
        per_h[h] = dict(diff_pct=d, hac_se=se, hac_t=t, n_bull=npos, n_bear=nneg)
        if np.isfinite(t):
            ts.append(t)
        if np.isfinite(d):
            ds.append(d)
    return dict(per_h=per_h,
                t_mean=float(np.mean(ts)) if ts else np.nan,
                d_mean=float(np.mean(ds)) if ds else np.nan,
                n=int(hi - lo))


ORDER = ['H_BULL', 'L_BULL', 'SIDEWAYS', 'L_BEAR', 'H_BEAR']


def ordering_status(labels, close_arr, horizons, lo, hi):
    '''5-label monotonicity of mean forward return, averaged over horizons.'''
    labels = np.asarray(labels, dtype=object)[lo:hi]
    means = {}
    for lb in ORDER:
        m = (labels == lb)
        if m.sum() < 5:
            means[lb] = np.nan
            continue
        vs = []
        for h in horizons:
            f = fwd_returns(close_arr, h)[lo:hi]
            v = f[m]
            v = v[np.isfinite(v)]
            if len(v):
                vs.append(float(np.mean(v)))
        means[lb] = float(np.mean(vs)) if vs else np.nan
    seq = [means[lb] for lb in ORDER]
    ok = [s for s in seq if np.isfinite(s)]
    if len(ok) < 4:
        return 'THIN', means
    fully = all(a >= b - 1e-12 for a, b in zip(ok, ok[1:]))
    bull = [means[l] for l in BULL_SET if np.isfinite(means[l])]
    bear = [means[l] for l in BEAR_SET if np.isfinite(means[l])]
    block = bool(bull and bear and min(bull) >= max(bear))
    return ('HOLDS' if fully else ('PARTIAL' if block else 'BROKEN')), means


def evaluate_run(r):
    '''S / L / W, occupancy, guards, and the IS-vs-OOS overfitting test.'''
    if r['error']:
        return r
    labels, close_arr = r['labels'], r['close_arr']
    horizons = r['fwd_h']
    n, n_fit = r['n_bars'], r['n_fit']
    embargo = max(horizons)
    is_lo, is_hi = 0, max(50, n_fit - embargo)
    oos_lo, oos_hi = n_fit, n

    # ---- S / L / W / occupancy / guards -- verbatim metrics ----------------
    S, Smat, pair_vals = metric_S(labels)
    swings = zigzag_swings(close_arr, ZZ_PCT)
    Lm = [metric_L(lb, swings)[0] for lb in labels]
    L75 = [metric_L(lb, swings)[1] for lb in labels]
    unm = [metric_L(lb, swings)[3] for lb in labels]
    Ws = [metric_W(lb) for lb in labels]
    occs = [occupancy_pct(lb) for lb in labels]
    occ = {lb: float(np.mean([o[lb] for o in occs])) for lb in REGIME_LABELS}
    W = float(np.mean(Ws))
    g = evaluate_guards(occ, W)

    # ---- the overfitting test, PER SEED SET, IS and OOS SEPARATELY ---------
    IS, OOS = [], []
    for lb in labels:
        IS.append(span_contrast(lb, close_arr, horizons, is_lo, is_hi))
        OOS.append(span_contrast(lb, close_arr, horizons, oos_lo, oos_hi))
    is_t = [x['t_mean'] for x in IS]
    oos_t = [x['t_mean'] for x in OOS]
    is_d = [x['d_mean'] for x in IS]
    oos_d = [x['d_mean'] for x in OOS]

    ord_is, means_is = ordering_status(labels[0], close_arr, horizons, is_lo, is_hi)
    ord_oos, means_oos = ordering_status(labels[0], close_arr, horizons, oos_lo, oos_hi)

    def _mn(v):
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else np.nan

    def _sp(v):
        v = [x for x in v if np.isfinite(x)]
        return (float(np.max(v) - np.min(v)) if len(v) > 1 else np.nan)

    IS_T, OOS_T = _mn(is_t), _mn(oos_t)
    IS_D, OOS_D = _mn(is_d), _mn(oos_d)
    inverted = bool(np.isfinite(IS_D) and np.isfinite(OOS_D)
                    and np.sign(IS_D) != np.sign(OOS_D) and IS_D != 0 and OOS_D != 0)

    r.update(S=S, S_matrix=Smat, S_min=min(pair_vals), S_max=max(pair_vals),
             L=float(np.mean(Lm)), L75=float(np.mean(L75)),
             unmatched=float(np.mean(unm)), n_swings=len(swings),
             W=W, W_lo=float(np.min(Ws)), W_hi=float(np.max(Ws)),
             occ=occ, guards=g, guards_ok=guards_pass(g),
             IS_T=IS_T, OOS_T=OOS_T, IS_D=IS_D, OOS_D=OOS_D,
             IS_T_spread=_sp(is_t), OOS_T_spread=_sp(oos_t),
             IS_n=IS[0]['n'], OOS_n=OOS[0]['n'], embargo=embargo,
             ord_is=ord_is, ord_oos=ord_oos,
             means_is=means_is, means_oos=means_oos,
             inverted=inverted, IS_perh=IS[0]['per_h'], OOS_perh=OOS[0]['per_h'])
    return r


t0 = time.time()
for r in RUNS:
    evaluate_run(r)
    assert_knobs_unchanged(f"eval {r['run_id']}")
print(f'evaluated {len(RUNS)} runs in {time.time() - t0:.1f}s   '
      f'ZZ_CALLS={ZZ_CALLS}')
print('ASSERT OK: fixed-knob invariant held across every run and every eval.')
""")

co(r"""
# ===========================================================================
# RESULTS TABLE. Every run appears. Nothing is dropped for looking bad.
# ===========================================================================
def guard_str(r):
    if r['error']:
        return '----'
    return ''.join(('.' if r['guards'][k][0] else 'X') for k in ('G1', 'G2', 'G3', 'G4'))


def fmt(v, w=6, p=2):
    return f'{v:>{w}.{p}f}' if (v is not None and np.isfinite(v)) else ' ' * (w - 3) + 'n/a'


print('=' * 138)
print('INTRADAY FX GRID -- 3 pairs x 2 timeframes = 6 runs   [REAL DATA, '
      'community GitHub source, 2012-11-16 -> 2022-03-04]')
print('=' * 138)
print(f"{'run':<16} {'bars':>7} {'H_BULL':>7} {'L_BULL':>7} {'SIDE':>7} "
      f"{'L_BEAR':>7} {'H_BEAR':>7} {'S%':>6} {'W':>6} {'L':>6} "
      f"{'guards':>7} {'IS t':>7} {'OOS t':>7} {'IS ord':>8} {'OOS ord':>8} {'INV':>4}")
print('-' * 138)
for r in RUNS:
    if r['error']:
        print(f"{r['run_id']:<16} {'':>7}   ERROR: {r['error']}")
        continue
    o = r['occ']
    print(f"{r['run_id']:<16} {r['n_bars']:>7} "
          + ' '.join(fmt(o[l], 7, 2) for l in ORDER) + ' '
          + fmt(r['S'], 6, 2) + ' ' + fmt(r['W'], 6, 2) + ' '
          + fmt(r['L'], 6, 1) + ' '
          + f"{guard_str(r):>7} "
          + fmt(r['IS_T'], 7, 2) + ' ' + fmt(r['OOS_T'], 7, 2) + ' '
          + f"{r['ord_is']:>8} {r['ord_oos']:>8} "
          + f"{'YES' if r['inverted'] else 'no':>4}")
print('-' * 138)
print("guards: '.' = pass, 'X' = fail, order G1 G2 G3 G4  "
      "(G1 occ in [3,50]% | G2 no collapse | G3 W<=25 | G4 SIDEWAYS<=50%)")
print("INV = the OOS BULL-BEAR contrast FLIPPED SIGN vs IS. That is the "
      "overfit signature.")
print("Every run carries a realised-vol PROXY, not a VIX. Every run ends "
      "2022-03-04.")
""")

co(r"""
# ===========================================================================
# THE OVERFITTING TABLE. IS and OOS SIDE BY SIDE, per run. NEVER POOLED.
# ===========================================================================
print('=' * 132)
print('IN-SAMPLE vs OUT-OF-SAMPLE  --  (mean fwd return | BULL) - (mean fwd | BEAR)')
print('HAC (Newey-West, Bartlett) t. Spans are DISJOINT; the IS span is embargoed')
print('by max(h) bars so no in-sample forward return can read an out-of-sample price.')
print('=' * 132)
print(f"{'run':<16} {'IS n':>7} {'OOS n':>7} {'emb':>4} | "
      f"{'IS d%':>7} {'IS t':>7} {'seed sprd':>9} | "
      f"{'OOS d%':>7} {'OOS t':>7} {'seed sprd':>9} | {'t change':>9} {'verdict':<22}")
print('-' * 132)
for r in RUNS:
    if r['error']:
        print(f"{r['run_id']:<16}   ERROR: {r['error']}")
        continue
    dt = (r['OOS_T'] - r['IS_T']) if (np.isfinite(r['OOS_T']) and np.isfinite(r['IS_T'])) else np.nan
    if r['inverted']:
        vd = 'INVERTED (overfit sig.)'
    elif np.isfinite(dt) and abs(r['OOS_T']) < abs(r['IS_T']):
        vd = 'degraded, same sign'
    elif np.isfinite(dt):
        vd = 'held or improved'
    else:
        vd = 'unmeasurable'
    print(f"{r['run_id']:<16} {r['IS_n']:>7} {r['OOS_n']:>7} {r['embargo']:>4} | "
          + fmt(r['IS_D'], 7, 3) + ' ' + fmt(r['IS_T'], 7, 2) + ' '
          + fmt(r['IS_T_spread'], 9, 2) + ' | '
          + fmt(r['OOS_D'], 7, 3) + ' ' + fmt(r['OOS_T'], 7, 2) + ' '
          + fmt(r['OOS_T_spread'], 9, 2) + ' | '
          + fmt(dt, 9, 2) + f' {vd:<22}')
print('-' * 132)
_ok = [r for r in RUNS if not r['error']]
_inv = [r for r in _ok if r['inverted']]
print(f'{len(_inv)} of {len(_ok)} runs INVERTED out of sample.')
print('A pooled IS+OOS number is NOT computed anywhere in this notebook, by design.')
print()
print('IS/OOS SPLIT DATES (TRAIN_FRACTION = %.2f):' % TRAIN_FRACTION)
for r in _ok:
    print(f"  {r['run_id']:<16} IS {r['span'][0]:%Y-%m-%d} -> "
          f"{r['dates'][r['n_fit'] - 1]:%Y-%m-%d}   "
          f"OOS {r['dates'][r['n_fit']]:%Y-%m-%d} -> {r['span'][1]:%Y-%m-%d}")
""")

co(r"""
# ===========================================================================
# PER-LABEL FORWARD RETURN, IS vs OOS, for every run. The ordering, in full.
# ===========================================================================
print('=' * 120)
print('MEAN FORWARD RETURN (%) BY LABEL -- averaged over the 1/3/5-day horizons')
print('IS row then OOS row for each run. Correct ordering is monotone DECREASING.')
print('=' * 120)
print(f"{'run':<16} {'span':<4} " + ' '.join(f'{l:>9}' for l in ORDER) + '  status')
print('-' * 120)
for r in RUNS:
    if r['error']:
        continue
    for tag, mm, st in (('IS', r['means_is'], r['ord_is']),
                        ('OOS', r['means_oos'], r['ord_oos'])):
        print(f"{(r['run_id'] if tag == 'IS' else ''):<16} {tag:<4} "
              + ' '.join(fmt(mm[l], 9, 4) for l in ORDER) + f'  {st}')
print('-' * 120)
print('HOLDS   = fully monotone across all five labels')
print('PARTIAL = the BULL block still sits above the BEAR block, but not monotone')
print('BROKEN  = the BULL block does NOT sit above the BEAR block')
print('THIN    = too few bars in too many labels to grade')
""")

# ===========================================================================
md(r"""
## 8. Machinery proofs

Three things are *driven*, not asserted in prose:

1. **G4 bites.** A degenerate all-`SIDEWAYS` labelling is pushed through the
   **same** `evaluate_guards` the real runs use. It scores a perfect `S = 100%`
   — exactly how `S` is gamed — and must be VOIDED.
2. **Truncation probe.** Cutting the raw series at *T* and rebuilding features,
   scaling with the same scaler and labelling with the same models must leave
   every label at *t* ≤ *T* **bit-identical**.
3. **No ZigZag leakage.** `ZZ_CALLS` was 0 for the whole label phase, and every
   `label_bars` argument was checked by identity and by `np.shares_memory`.
""")

co(r"""
# ===========================================================================
# PROOF 1 -- G4 (and G1) MUST BITE on a degenerate all-SIDEWAYS labelling.
# Driven through the REAL guard code, not a copy of it.
# ===========================================================================
_ref = [r for r in RUNS if not r['error']][0]
_n = _ref['n_bars']
_stub = [np.array(['SIDEWAYS'] * _n, dtype=object) for _ in range(R_SEED_SETS)]
_S_stub, _, _ = metric_S(_stub)
_occ_stub = occupancy_pct(_stub[0])
_W_stub = metric_W(_stub[0])
_g_stub = evaluate_guards(_occ_stub, _W_stub)

print('DEGENERATE STUB: every bar labelled SIDEWAYS, on the real run '
      f"{_ref['run_id']} ({_n} bars)")
print(f'  S = {_S_stub:.2f}%   W = {_W_stub:.2f}   SIDEWAYS = '
      f"{_occ_stub['SIDEWAYS']:.1f}%")
for k in ('G1', 'G2', 'G3', 'G4'):
    ok, why = _g_stub[k]
    print(f"  {k}: {'PASS' if ok else 'FAIL'}  {why}")
assert _S_stub == 100.0, 'the stub should trivially maximise S'
assert not _g_stub['G4'][0], 'G4 FAILED TO BITE on an all-SIDEWAYS labelling'
assert not _g_stub['G1'][0], 'G1 FAILED TO BITE on an all-SIDEWAYS labelling'
assert not guards_pass(_g_stub), 'the degenerate stub was not VOIDED'
print('\nPROOF 1 PASS: S = 100.00% (perfectly gamed) and the labelling is VOIDED '
      'by G4 and G1.')
print('The guard is TESTED, not trusted.')

# and a high-whipsaw stub, to show G3 bites too
_alt = np.array([REGIME_LABELS[i % 5] for i in range(_n)], dtype=object)
_g_alt = evaluate_guards(occupancy_pct(_alt), metric_W(_alt))
assert not _g_alt['G3'][0], 'G3 FAILED TO BITE on a 100-switches-per-100-bars labelling'
print(f"PROOF 1b PASS: a switch-every-bar labelling is VOIDED by G3 "
      f"({_g_alt['G3'][1]}).")
""")

co(r"""
# ===========================================================================
# PROOF 2 -- TRUNCATION PROBE, run on BOTH timeframes.
#
# Cut the RAW price/volatility series at bar T_raw, rebuild features from
# scratch, scale with the SAME scaler, label with the SAME fitted models and
# the SAME n_fit. Every label at t <= T must be bit-identical.
#
# WHAT THIS PROVES: the feature construction and the labelling path use bars
# <= t only. WHAT IT DOES NOT PROVE: anything about refitting -- a model refit
# on a shorter window is a DIFFERENT model by design, so the probe deliberately
# holds the fit fixed. Same scope as the master's probe.
# ===========================================================================
def truncation_probe(r, frac=0.85):
    if r['error']:
        return None
    dates_full, labels_full = r['dates'], r['labels'][0]
    ser_raw, vol_raw = r['_raw_close'], r['_raw_vol']
    bpd = r['bpd_assumed']
    T_raw = int(len(ser_raw) * frac)
    cut_close = ser_raw.iloc[:T_raw]
    cut_vol = vol_raw.iloc[:T_raw] if vol_raw is not None else None

    with bars_per_day(bpd):
        feat_cut = build_features(cut_close, cut_vol, LOOKBACK_SCALE)
    Xs_cut = r['scaler'].transform(feat_cut[FEATURE_COLS].values)
    with reopen_label_phase(f"truncation probe on {r['run_id']}"):
        with bars_per_day(bpd):
            lab_cut = label_bars(r['models'][0], Xs_cut, feat_cut.index,
                                 cut_close, feat_cut[TREND_FEATURE].values,
                                 r['n_fit'], FEATURE_COLS, dir_feats=feat_cut)
    got = lab_cut['tactical_regime_state']
    common = feat_cut.index.intersection(dates_full)
    ref = pd.Series(labels_full, index=dates_full).reindex(common)
    new = got.reindex(common)
    n_diff = int((ref.values != new.values).sum())
    return dict(run=r['run_id'], T_raw=T_raw, n_common=len(common),
                n_diff=n_diff, cut_at=cut_close.index[-1])


_probes = []
for _tf in ('1h', '2h'):
    _c = [r for r in RUNS if r['tf'] == _tf and not r['error']]
    if _c:
        _probes.append(truncation_probe(_c[0]))

print('=' * 104)
print('TRUNCATION PROBE')
print('=' * 104)
for p in _probes:
    if p is None:
        continue
    print(f"  {p['run']:<16} cut at {p['cut_at']:%Y-%m-%d %H:%M} "
          f"({p['T_raw']} raw bars) -> {p['n_common']} overlapping labels, "
          f"{p['n_diff']} differ")
    assert p['n_common'] > 100, 'probe overlap too small to be meaningful'
    assert p['n_diff'] == 0, (f"LOOK-AHEAD: {p['n_diff']} labels changed when the "
                              f"series was cut at {p['cut_at']}")
print('\nPROOF 2 PASS: every label at t <= T is BIT-IDENTICAL after truncation, '
      'on one run per timeframe.')
print('Features and labels at bar t therefore read bars <= t only.')
""")

co(r"""
# ===========================================================================
# PROOF 3 -- ZigZag leakage. Restate the tripwire's evidence.
# ===========================================================================
print(f'label_bars calls          : {LABEL_CALLS}')
print(f'ZigZag computations       : {ZZ_CALLS}')
print(f'ZigZag objects registered : {len(ZZ_IDS)}')
print('  check 1  ordering : every label in the LABEL PHASE was produced with '
      'ZZ_CALLS == 0 (asserted at the phase boundary).')
print('  check 2  identity : every label_bars argument was checked against '
      'ZZ_IDS on every call.')
print('  check 3  aliasing : every ndarray argument was checked with '
      'np.shares_memory against every ZigZag array.')
print('  check 4  decree   : every call asserted bar_dir_weight == '
      f'{BAR_DIR_WEIGHT}.')
print('The only calls after the phase closed were the truncation probes, which '
      'reopened it LOUDLY (banner above) with checks 2-4 still armed.')
print('\nPROOF 3 PASS: the ZigZag is used to GRADE labels (metric L) and never '
      'to PRODUCE one.')
assert LABEL_CALLS > 0 and ZZ_CALLS > 0
""")

# ===========================================================================
md(r"""
## 9. Figures

Three figures, each of which decides something.

1. **IS vs OOS HAC *t* per run** — the overfitting view, with inversions marked.
2. **Occupancy + guard pass/fail across the 6 runs.**
3. **Regime-background chart, Brexit week on GBPUSD** — the master's chart
   style: black price line, full-height bands, tight **equal** y-limits that are
   **not** zero-anchored, and each band extended to the start of the next so
   overnight/weekend gaps never render as white stripes.
""")

co(r"""
# ===========================================================================
# FIGURE 1 -- THE OVERFITTING VIEW. IS vs OOS, every run.
# ===========================================================================
FIGS = []
_ok_all = [r for r in RUNS if not r['error']]
if _ok_all:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6),
                                   gridspec_kw={'width_ratios': [1.35, 1]})
    y = np.arange(len(_ok_all))
    lab = [r['run_id'] for r in _ok_all]
    isv = np.array([r['IS_T'] for r in _ok_all], dtype=float)
    oov = np.array([r['OOS_T'] for r in _ok_all], dtype=float)
    inv = np.array([r['inverted'] for r in _ok_all])

    axL.barh(y - 0.2, isv, height=0.38, color='#4C72B0', label='IN-SAMPLE')
    axL.barh(y + 0.2, oov, height=0.38, color='#DD8452', label='OUT-OF-SAMPLE')
    axL.axvline(0, color='black', lw=1)
    for x in (-2, 2):
        axL.axvline(x, color='grey', ls=':', lw=1)
    axL.set_yticks(y)
    axL.set_yticklabels(lab, fontsize=9)
    axL.invert_yaxis()
    axL.set_xlabel('HAC t of (mean fwd | BULL) - (mean fwd | BEAR)')
    axL.set_title('IS vs OOS direction contrast, per run\n'
                  'dotted lines = |t| = 2. NEVER POOLED.', fontsize=10)
    axL.legend(fontsize=8, loc='lower right')
    _all = np.concatenate([isv, oov])
    _xlo = min(0.0, float(np.nanmin(_all))) * 1.10 - 0.30
    _xhi = max(0.0, float(np.nanmax(_all))) * 1.45 + 0.30
    axL.set_xlim(_xlo, _xhi)
    for i, r in enumerate(_ok_all):
        if r['inverted']:
            axL.text(_xhi, i, 'INVERTED ', fontsize=8, color='crimson',
                     va='center', ha='right', fontweight='bold')
    axL.grid(axis='x', alpha=0.2)

    fin = np.isfinite(isv) & np.isfinite(oov)
    axR.axhline(0, color='black', lw=1)
    axR.axvline(0, color='black', lw=1)
    lim = float(np.nanmax(np.abs(_all[np.isfinite(_all)]))) * 1.25 + 0.3
    axR.plot([-lim, lim], [-lim, lim], color='grey', ls='--', lw=1,
             label='no degradation (y = x)')
    axR.fill_between([-lim, 0], 0, lim, color='crimson', alpha=0.07)
    axR.fill_between([0, lim], -lim, 0, color='crimson', alpha=0.07)
    for i, r in enumerate(_ok_all):
        if not fin[i]:
            continue
        mk = 'o' if r['tf'] == '1h' else 's'
        axR.scatter(isv[i], oov[i], marker=mk, s=70,
                    color=('crimson' if inv[i] else '#1a7f37'),
                    edgecolor='black', linewidth=0.6, zorder=3)
        axR.annotate(r['run_id'], (isv[i], oov[i]), fontsize=7,
                     xytext=(5, 4), textcoords='offset points')
    axR.set_xlim(-lim, lim); axR.set_ylim(-lim, lim)
    axR.set_xlabel('IN-SAMPLE HAC t')
    axR.set_ylabel('OUT-OF-SAMPLE HAC t')
    axR.set_title('shaded quadrants = SIGN INVERSION (the overfit signature)\n'
                  'circles = 1h, squares = 2h', fontsize=10)
    axR.legend(fontsize=8, loc='lower right')
    axR.grid(alpha=0.25)
    fig.suptitle('INTRADAY FX -- THE OVERFITTING TEST   '
                 f'[{len(_inv)} of {len(_ok_all)} runs INVERTED]   '
                 'REAL DATA 2012-11-16 -> 2022-03-04, community GitHub source',
                 fontsize=11)
    fig.tight_layout()
    FIGS.append('fig1_is_vs_oos')
    plt.show()
""")

co(r"""
# ===========================================================================
# FIGURE 2 -- OCCUPANCY + GUARD PASS/FAIL across the 6 runs.
# ===========================================================================
if _ok_all:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.0),
                                   gridspec_kw={'width_ratios': [1.5, 1]})
    x = np.arange(len(_ok_all))
    names = [r['run_id'] for r in _ok_all]

    bot = np.zeros(len(_ok_all))
    for lb in ORDER:
        v = np.array([r['occ'][lb] for r in _ok_all])
        axA.bar(x, v, 0.62, bottom=bot, color=REGIME_COLORS[lb],
                edgecolor='white', lw=0.6)
        if lb == 'SIDEWAYS':
            for xi in range(len(x)):
                axA.text(xi, bot[xi] + v[xi] / 2, f'{v[xi]:.1f}%', ha='center',
                         va='center', fontsize=8, fontweight='bold')
        bot = bot + v
    # the pre-registered P2 band, drawn against the SIDEWAYS SEGMENT's own
    # height -- shown as a separate marker row rather than a horizontal line
    # across a STACKED bar, which would read against the wrong quantity.
    axA.set_ylim(0, 122)
    axA.set_xticks(x); axA.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    axA.set_ylabel('occupancy %')
    axA.set_title('Label occupancy, all 5 labels, averaged over the '
                  f'{R_SEED_SETS} seed sets\n'
                  'SIDEWAYS % is written on its OWN segment (a line across a '
                  'stacked bar\nwould read against the wrong quantity). '
                  f'Pre-registered P2 band = [{P2_LO:.0f}%, {P2_HI:.0f}%].',
                  fontsize=9.5)
    for xi, r in enumerate(_ok_all):
        s = r['occ']['SIDEWAYS']
        axA.text(xi, 112, 'in P2 band' if P2_LO <= s <= P2_HI else 'OUT of band',
                 ha='center', fontsize=7.5,
                 color=('#1a7f37' if P2_LO <= s <= P2_HI else 'crimson'),
                 fontweight='bold')
    regime_legend(axA, loc='lower center')

    GK = ('G1', 'G2', 'G3', 'G4')
    GT = {'G1': f'G1  occ in [{OCC_MIN_PCT:.0f},{OCC_MAX_PCT:.0f}]%',
          'G2': 'G2  no label collapse',
          'G3': f'G3  W <= {W_GUARD_MAX:.0f} / 100 bars',
          'G4': f'G4  SIDEWAYS <= {SIDEWAYS_MAX:.0f}%'}
    Gm = np.array([[1.0 if r['guards'][k][0] else 0.0 for k in GK]
                   for r in _ok_all])
    axB.imshow(Gm, cmap=ListedColormap(['#d62728', '#2ca02c']),
               aspect='auto', vmin=0, vmax=1)
    axB.set_xticks(range(4))
    axB.set_xticklabels([GT[k] for k in GK], rotation=25, ha='right', fontsize=8)
    axB.set_yticks(range(len(_ok_all)))
    axB.set_yticklabels(names, fontsize=9)
    for i in range(Gm.shape[0]):
        for j in range(Gm.shape[1]):
            axB.text(j, i, 'PASS' if Gm[i, j] > 0.5 else 'FAIL', ha='center',
                     va='center', fontsize=8, color='white', fontweight='bold')
    axB.set_xticks(np.arange(-.5, 4, 1), minor=True)
    axB.set_yticks(np.arange(-.5, len(_ok_all), 1), minor=True)
    axB.grid(which='minor', color='white', linewidth=1.6)
    axB.tick_params(which='minor', bottom=False, left=False)
    _np2 = sum(1 for r in _ok_all if r['guards_ok'])
    axB.set_title(f'Guards G1-G4  --  {_np2} of {len(_ok_all)} runs pass ALL FOUR\n'
                  'green = PASS, red = FAIL. No threshold was moved.', fontsize=9.5)
    fig.suptitle('INTRADAY FX -- OCCUPANCY AND GUARDS   [REAL DATA, '
                 'community GitHub source, ends 2022-03-04]', fontsize=11)
    fig.tight_layout()
    FIGS.append('fig2_occupancy_guards')
    plt.show()
""")

co(r"""
# ===========================================================================
# FIGURE 3 -- REGIME-BACKGROUND CHART, BREXIT WEEK ON GBPUSD.
# Master style, using the master's OWN harvested helpers:
#   regime_blocks  -- each band runs to the START of the next, so an overnight
#                     or weekend gap never renders as a white stripe.
#   shade_bands    -- ONE PolyCollection per label, full-height.
#   set_price_ylim -- EXPLICIT, TIGHT, EQUAL-margin y-limits, NOT zero-anchored.
# ===========================================================================
_gb = [r for r in RUNS if r['market'] == 'GBP/USD' and r['tf'] == '1h'
       and not r['error']]
if _gb:
    _r = _gb[0]
    _lab = pd.Series(_r['labels'][0], index=_r['dates'])
    _px = _r['close_ser']
    _split = _r['dates'][_r['n_fit'] - 1]

    WINDOWS = [
        (pd.Timestamp('2016-06-13'), pd.Timestamp('2016-07-08'),
         'BREXIT FORTNIGHT -- GBP/USD 1h, 2016-06-13 -> 2016-07-08'),
        (pd.Timestamp('2016-01-01'), pd.Timestamp('2016-12-31'),
         'CONTEXT -- GBP/USD 1h, calendar 2016'),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    for ax, (t0_, t1_, ttl) in zip(axes, WINDOWS):
        m = (_r['dates'] >= t0_) & (_r['dates'] <= t1_)
        sub_lab, sub_px = _lab[m], _px[m]
        shade_regimes(ax, sub_lab, alpha=0.38)
        ax.plot(sub_px.index, sub_px.values, color='black', lw=1.1, zorder=4)
        set_price_ylim(ax, sub_px, pad=0.03, tag=f"brexit {ttl[:18]}")
        ax.set_xlim(sub_px.index[0], sub_px.index[-1])
        h = mark_split(ax, sub_lab.index, _split)
        regime_legend(ax, loc='lower left', extra=[h] if h else None)
        ax.axvline(pd.Timestamp('2016-06-24'), color='#0033aa', lw=1.6, ls='--',
                   zorder=5)
        ax.text(pd.Timestamp('2016-06-24'), ax.get_ylim()[1], ' referendum result',
                color='#0033aa', fontsize=8.5, va='top', fontweight='bold',
                zorder=6)
        ax.set_title(ttl, fontsize=11)
        ax.set_ylabel('USD per GBP')
        ax.grid(alpha=0.18, zorder=0)
    _gs = ', '.join(k for k, v in _r['guards'].items() if not v[0]) or 'none'
    fig.suptitle(
        'RegDet V1.1 applied UNCHANGED to real hourly GBP/USD. Every knob is '
        'the Nifty-2h value.\n'
        f"Whole-run context: W = {_r['W']:.1f} switches / 100 bars "
        f"(a switch every {100 / max(_r['W'], 1e-9):.1f} bars), "
        f"S = {_r['S']:.1f}%, SIDEWAYS = {_r['occ']['SIDEWAYS']:.1f}%, "
        f"guards failed: {_gs}.\n"
        'This window is IN-SAMPLE (the IS/OOS split is later); it is shown to '
        'inspect BEHAVIOUR through a known shock, not to claim skill.',
        fontsize=10.5, y=0.999)
    fig.tight_layout()
    FIGS.append('fig3_brexit_regimes')
    plt.show()

    # the two rendering defects this figure has hit before, ASSERTED
    _blocks = regime_blocks(_lab)
    for _a, _b in zip(_blocks, _blocks[1:]):
        assert _a[2] == _b[1], 'regime bands do not abut -- white-stripe defect'
    print(f'ASSERT OK: {len(_blocks)} regime bands abut exactly (no white stripes).')
    for tag, ax, lo, hi in YLIM_CHECKS:
        y0, y1 = ax.get_ylim()
        assert y0 < lo and y1 > hi, f'y-limits do not bracket the series ({tag})'
        assert not (y0 <= 0 <= y1), f'y-axis is zero-anchored ({tag})'
    print(f'ASSERT OK: {len(YLIM_CHECKS)} shaded panel(s) have tight, '
          'non-zero-anchored y-limits that bracket their own series.')
    _bw = _lab.loc['2016-06-23':'2016-06-27']
    print('\nLabels emitted through the Brexit shock (GBP/USD 1h):')
    print('  ' + '  '.join(f'{k}={v}' for k, v in _bw.value_counts().items()))
else:
    print('GBP/USD 1h unavailable; Brexit regime chart skipped.')
print(f'\nfigures produced: {len(FIGS)} -> {FIGS}')
""")

# ===========================================================================
md(r"""
## 10. Grading the pre-registered predictions

Graded by the rules printed in section 3, **before** any of these numbers
existed. A refutation is stated plainly and is not softened, reinterpreted or
explained away.
""")

co(r"""
# ===========================================================================
# GRADING. The rules are the ones printed in section 3, unchanged.
# ===========================================================================
def verdict(ok):
    return 'CONFIRMED' if ok else 'REFUTED'


GRADES = {}
_okR = [r for r in RUNS if not r['error']]
_N = len(_okR)
_MAJ = _N // 2 + 1

print('=' * 116)
print('GRADING THE PRE-REGISTERED PREDICTIONS')
print(f'{_N} of {len(RUNS)} runs completed; a MAJORITY is >= {_MAJ}.')
print('=' * 116)

# ---- P1 -------------------------------------------------------------------
print(f"\nP1  {PREDICTIONS['P1'][0]}")
_pass = [r for r in _okR if r['guards_ok']]
for r in _okR:
    _f = ', '.join(f'{k}: {v[1]}' for k, v in r['guards'].items() if not v[0])
    print(f"    {r['run_id']:<16} {'PASS' if r['guards_ok'] else 'FAIL'}"
          + (f'   {_f}' if _f else ''))
_p1 = len(_pass) >= _MAJ
GRADES['P1'] = _p1
print(f'    {len(_pass)} of {_N} runs pass ALL FOUR guards.')
print(f'    -> P1 {verdict(_p1)}')

# ---- P2 -------------------------------------------------------------------
print(f"\nP2  {PREDICTIONS['P2'][0]}")
_inb = [r for r in _okR if P2_LO <= r['occ']['SIDEWAYS'] <= P2_HI]
for r in _okR:
    s = r['occ']['SIDEWAYS']
    print(f"    {r['run_id']:<16} SIDEWAYS {s:5.1f}%   "
          + ('in band' if P2_LO <= s <= P2_HI else 'OUT of band'))
_p2 = len(_inb) >= _MAJ
GRADES['P2'] = _p2
print(f'    {len(_inb)} of {_N} runs land in [{P2_LO:.0f}%, {P2_HI:.0f}%].')
print(f'    -> P2 {verdict(_p2)}')

# ---- P3 -- THE HEADLINE ---------------------------------------------------
print(f"\nP3  {PREDICTIONS['P3'][0]}")
_invd = [r for r in _okR if r['inverted']]
_same = [r for r in _okR if not r['inverted']]
for r in _okR:
    print(f"    {r['run_id']:<16} IS d {r['IS_D']:+.4f} (t {r['IS_T']:+.2f})  ->  "
          f"OOS d {r['OOS_D']:+.4f} (t {r['OOS_T']:+.2f})   "
          + ('INVERTED' if r['inverted'] else 'same sign'))
_p3 = len(_same) >= _MAJ
GRADES['P3'] = _p3
print(f'    {len(_same)} of {_N} runs KEPT their sign; {len(_invd)} INVERTED.')
print(f'    -> P3 {verdict(_p3)}')

# ---- P4 -------------------------------------------------------------------
print(f"\nP4  {PREDICTIONS['P4'][0]}")
_p4 = False
if _N == 6:
    _V = np.array([[r['occ']['SIDEWAYS'], r['S'], r['W'], r['L'],
                    r['IS_T'], r['OOS_T']] for r in _okR], dtype=float)
    _mu, _sd = np.nanmean(_V, axis=0), np.nanstd(_V, axis=0)
    _sd = np.where(_sd > 0, _sd, 1.0)
    _Z = (_V - _mu) / _sd
    _idx = {r['run_id']: i for i, r in enumerate(_okR)}

    def _dist(a, b):
        return float(np.linalg.norm(_Z[_idx[a]] - _Z[_idx[b]]))

    _pairs = [r['market'] for r in _okR if r['tf'] == '1h']
    _samepair = [(f'{p}@1h', f'{p}@2h') for p in _pairs]
    _sametf = []
    for _tf in ('1h', '2h'):
        for _a, _b in itertools.combinations(_pairs, 2):
            _sametf.append((f'{_a}@{_tf}', f'{_b}@{_tf}'))

    print(f'    z-scored over {list(P4_METRICS)}')
    print('    SAME PAIR, different bar size (1h vs 2h):')
    _ds = []
    for a, b in _samepair:
        d = _dist(a, b); _ds.append(d)
        print(f'      {a:<16} <-> {b:<16} distance {d:6.3f}')
    print('    SAME TIMEFRAME, different pair:')
    _dt = []
    for a, b in _sametf:
        d = _dist(a, b); _dt.append(d)
        print(f'      {a:<16} <-> {b:<16} distance {d:6.3f}')
    _msp, _mst = float(np.mean(_ds)), float(np.mean(_dt))
    _p4 = _msp < _mst
    print(f'    mean SAME-PAIR distance      {_msp:6.3f}  (n={len(_ds)})')
    print(f'    mean SAME-TIMEFRAME distance {_mst:6.3f}  (n={len(_dt)})')
    print(f'    pair identity matters more than bar size: '
          f'{"YES" if _p4 else "NO"}')
else:
    print(f'    only {_N} of 6 runs completed -- the pairing design is '
          'incomplete. UNMEASURABLE, graded REFUTED rather than assumed.')
GRADES['P4'] = _p4
print(f'    -> P4 {verdict(_p4)}')

print('\n' + '=' * 116)
print('SUMMARY:  ' + '   '.join(f'{k} {verdict(v)}' for k, v in GRADES.items()))
print('=' * 116)
""")

co(r"""
# ===========================================================================
# THE HEADLINE COMPARISON -- does INTRADAY agree with the sibling's DAILY
# finding? Stated in plain words, from the numbers, not from a narrative.
# ===========================================================================
DAILY_INV, DAILY_N = 4, 6      # generalization.ipynb Grid B, REAL daily FX
print('=' * 116)
print('INTRADAY vs DAILY -- does this notebook AGREE with generalization.ipynb?')
print('=' * 116)
print(f'  generalization.ipynb Grid B  (REAL DAILY/WEEKLY FX, ~13,900 bars/run):')
print(f'      {DAILY_INV} of {DAILY_N} runs INVERTED sign between IS and OOS.')
print(f'      Its pre-registered P3 was REFUTED.')
print(f'      Worked example quoted there: JPY/USD daily IS t = +2.06 -> '
      f'OOS t = -0.38.')
print()
_ni = len(_invd)
print(f'  THIS notebook  (REAL INTRADAY FX, ~{int(np.mean([r["n_bars"] for r in _okR])):,} '
      f'bars/run, ~{np.mean([r["n_bars"] for r in _okR]) / 13900:.1f}x the sample):')
print(f'      {_ni} of {_N} runs INVERTED sign between IS and OOS.')
print(f'      Its pre-registered P3 is {verdict(GRADES["P3"])}.')
print()
_daily_majority_inverted = DAILY_INV > DAILY_N / 2
_intra_majority_inverted = _ni > _N / 2
print('  PLAIN WORDS:')
if _daily_majority_inverted and _intra_majority_inverted:
    print('  The intraday result AGREES with the daily finding. A MAJORITY of runs')
    print('  invert on BOTH cadences. On roughly four times the sample per run, the')
    print('  IS-to-OOS sign inversion REPRODUCES. This is not a small-sample')
    print('  artefact of the daily grid: it survives a much better-powered test,')
    print('  which makes the overfitting reading STRONGER, not weaker.')
elif (not _daily_majority_inverted) and (not _intra_majority_inverted):
    print('  The intraday result AGREES with the daily finding in the sense that')
    print('  neither cadence inverts on a majority of runs.')
elif _daily_majority_inverted and not _intra_majority_inverted:
    print('  The intraday result DISAGREES with the daily finding. The daily grid')
    print('  inverted on a majority of runs; this intraday grid does NOT, on '
          'roughly')
    print('  four times the sample per run. The daily inversion therefore does not')
    print('  generalise down the timeframe axis. Two readings are open and this')
    print('  notebook does not choose between them: either the daily inversion was')
    print('  a small-sample artefact, or intraday and daily FX are genuinely')
    print('  different problems for this detector. Deciding that needs work this')
    print('  notebook does not do.')
else:
    print('  The intraday result DISAGREES with the daily finding in the other')
    print('  direction: this grid inverts on a majority of runs where the daily')
    print('  grid did not.')
print()
print('  CAVEAT ON THE COMPARISON, stated rather than buried: the two grids do '
      'NOT')
print('  cover the same span. Grid B ran 1971/1999 -> 2026. This grid runs')
print('  2012-11-16 -> 2022-03-04 ONLY. A difference between them may be a')
print('  CADENCE effect, a SAMPLE-PERIOD effect, or both, and this design cannot')
print('  separate the two.')
print('=' * 116)
""")

co(r"""
# ===========================================================================
# THE HONEST READING. A template that fills itself from the numbers, so it
# cannot drift away from what was actually measured.
# ===========================================================================
print('=' * 116)
print('WHAT THIS RUN DOES AND DOES NOT ESTABLISH')
print('=' * 116)
print('* THE DATA IS REAL and was verified live in section 4 against Brexit and')
print('  COVID. It is COMMUNITY GITHUB DATA, not an official feed, and it is')
print('  labelled that way everywhere.')
print('* THE DATA ENDS 2022-03-04. Nothing here speaks to 2022-2026 -- not the')
print('  inflation shock, not the hiking cycle, not EURUSD parity.')
print('* THE VOLATILITY INPUT IS A PROXY. Spot FX has no VIX; every run used a')
print('  causal trailing realised-vol stand-in, stamped as such on every row.')
print('* NOTHING WAS RE-TUNED. Every constant is the shipped Nifty-2h value; the')
print('  fixed-knob invariant was re-asserted after every run and every eval.')
print('* NO RUN WAS DROPPED and no threshold was moved.')
print()
print('PER-RUN, in plain words:')
for r in _okR:
    gp = 'PASS' if r['guards_ok'] else ('FAIL ' + ','.join(
        k for k, v in r['guards'].items() if not v[0]))
    print(f"  {r['run_id']:<16} {r['n_bars']:>6} bars   "
          f"SIDEWAYS {r['occ']['SIDEWAYS']:5.1f}%   S {r['S']:5.1f}%   "
          f"W {r['W']:5.2f}   guards {gp:<14} "
          f"IS t {r['IS_T']:+5.2f} -> OOS t {r['OOS_T']:+5.2f}"
          + ('   INVERTED' if r['inverted'] else ''))
print()
print('WHAT IS NOT CLAIMED: no return, no Sharpe, no tradeability. This notebook')
print('measures a DETECTOR, not a strategy. It also does NOT test whether')
print('re-tuning the knobs per pair would help -- doing that would void the')
print('architecture-level test this notebook exists to run.')
print('=' * 116)
""")

# ===========================================================================
md(r"""
## 11. Config hand-off

A pasteable block recording **exactly** what was held fixed for every run, and
the short list of things that were allowed to re-derive.
""")

co(r"""
# ===========================================================================
# CONFIG HAND-OFF -- pasteable Python. Generated from the LIVE globals, so it
# cannot drift from what actually ran.
# ===========================================================================
assert_knobs_unchanged('hand-off')

_ho = f'''# ---------------------------------------------------------------------------
# RegDet V1.1 INTRADAY FX RUN -- EXACTLY WHAT WAS HELD FIXED
# generated from the live notebook globals after every run completed
# data source : {FX_SOURCE_LABEL}
# span        : 2012-11-16 -> 2022-03-04  (SAYS NOTHING ABOUT 2022-2026)
# grid        : 3 pairs x {{1h, 2h}} = 6 runs; 2h is a DOWNWARD resample of 1h
# volatility  : causal trailing realised-vol PROXY on every run (FX has no VIX)
# ---------------------------------------------------------------------------
REGDET_INTRADAY_FX_FIXED = dict(
    # --- model ---------------------------------------------------------
    N_STATES               = {N_STATES_FIXED},
    covariance_type        = {COV_FIXED!r},
    FEATURE_COLS           = {list(FEATURE_COLS)!r},
    ENSEMBLE_K             = {ENSEMBLE_K},
    BASE_SEED              = {BASE_SEED},
    R_SEED_SETS            = {R_SEED_SETS},
    # --- labeling ------------------------------------------------------
    CONF_L                 = {CONF_L},
    CONFIRM_BARS           = {CONFIRM_BARS},
    BAR_DIR_WEIGHT         = {BAR_DIR_WEIGHT},
    BAR_DIR_FEATURES       = {list(BAR_DIR_FEATURES)!r},
    BAR_DIR_TAU            = {BAR_DIR_TAU},
    DIRECTION_MODE         = {DIRECTION_MODE!r},
    DIRECTION_EXCLUDE      = {tuple(DIRECTION_EXCLUDE)!r},
    INTENSITY_MODE         = {INTENSITY_MODE!r},
    ESCALATION_DURING_HOLD = {ESCALATION_DURING_HOLD!r},
    Z_HI                   = {Z_HI},
    Z_HI_EXIT              = {Z_HI_EXIT},
    EFF_HI                 = {EFF_HI},
    EFF_HI_EXIT            = {EFF_HI_EXIT},
    EFF_WIN                = {EFF_WIN},      # BARS, frozen; real-time horizon varies by timeframe
    H_TARGET_RATE          = {H_TARGET_RATE},
    H_EXIT_SLACK           = {H_EXIT_SLACK},
    TREND_FEATURE          = {TREND_FEATURE!r},
    # --- windows: DAY-denominated, rebuilt per run from BARS_PER_DAY ----
    MOM_DAYS               = {MOM_DAYS!r},
    CONTEXT_DAYS           = {CONTEXT_DAYS},
    VOL_WIN_BARS           = {BASE_WIN_MASTER['VOL_WIN']},   # raw bar counts in the master
    VOL_FAST_BARS          = {BASE_WIN_MASTER['VOL_FAST']},
    FWD_DAYS               = {FWD_DAYS!r},
    BARS_PER_DAY_1H        = 24,      # spot FX ~24h weekday
    BARS_PER_DAY_2H        = 12,
    # --- harness -------------------------------------------------------
    TRAIN_FRACTION         = {TRAIN_FRACTION},
    N_FOLDS                = {N_FOLDS},
    HMM_ITER               = {HMM_ITER},
    TRAIN_CAP_BARS         = {TRAIN_CAP_BARS!r},   # None = no cap applied
    # --- guards / metrics ----------------------------------------------
    ZZ_PCT                 = {ZZ_PCT},
    G1_OCC_BAND_PCT        = ({OCC_MIN_PCT}, {OCC_MAX_PCT}),
    G3_W_MAX               = {W_GUARD_MAX},
    G4_SIDEWAYS_MAX        = {SIDEWAYS_MAX},
    P2_SIDEWAYS_BAND       = ({P2_LO}, {P2_HI}),
)

REGDET_INTRADAY_FX_SCALING = {{
'''
for p in FX_PAIRS:
    _d = FX_DIV.get(p['sym'])
    _ho += (f"    {p['sym']!r:<10}: dict(divisor={_d!r}, "
            f"median_band={p['med_band']!r}),\n")
_ho += "}\n\n# ALLOWED to re-derive per run (data-dependent by design):\n"
_ho += (f"#   * StandardScaler, fit on that run's own LEADING "
        f"{TRAIN_FRACTION:.0%} window\n")
_ho += "#   * trend_z's mu/sd baseline, frozen on the same window by the engine\n"
_ho += "#   * the H-gate band, derived from fit-window occupancy (H_TARGET_RATE)\n"
_ho += ("#   * BASE_WIN, rebuilt from BARS_PER_DAY by the master's OWN formula\n"
        "#     with the SAME day counts -- this keeps the HORIZON fixed, it does\n"
        "#     not tune a knob\n")
_ho += "# NOTHING ELSE. The notebook asserts this after every single run.\n\n"
_ho += 'REGDET_INTRADAY_FX_BPD = {\n'
for r in RUNS:
    _ho += (f"    {r['run_id']!r:<18}: dict(assumed={r['bpd_assumed']}, "
            f"realised={r['bpd_realised']:.2f}, "
            f"vol_source={r.get('vix_kind', 'n/a')!r}),\n")
_ho += '}\n'
print(_ho)
""")

co(r"""
# ===========================================================================
# RUN SUMMARY.
# ===========================================================================
_el = time.time() - T_START
print('=' * 100)
print('RUN SUMMARY')
print('=' * 100)
print(f'  runs                : {len(RUNS)}  (3 pairs x 2 timeframes)')
print(f'  completed / errored : {len([r for r in RUNS if not r["error"]])} / '
      f'{len([r for r in RUNS if r["error"]])}')
print(f'  HMM fits            : {FIT_COUNT}   '
      f'({R_SEED_SETS} disjoint seed sets x K={ENSEMBLE_K} per run, '
      'plus the runtime probe)')
print(f'  label_bars calls    : {LABEL_CALLS}')
print(f'  ZigZag computations : {ZZ_CALLS}  (all in the EVAL phase)')
print(f'  figures             : {len(FIGS)}  {FIGS}')
print(f'  TRAIN_CAP_BARS      : {TRAIN_CAP_BARS}')
if RUNTIME_EST is not None:
    print(f'  projected / actual  : {RUNTIME_EST:.0f}s projected for the grid, '
          f'{_el:.0f}s total wall clock')
print()
print('  NOTHING WAS CUT. R=4 seed sets, 3 pairs, 2 timeframes -- the full grid.')
print('=' * 100)
assert_knobs_unchanged('end of notebook')
print('FINAL ASSERT OK: every frozen constant is exactly what it was at '
      'declaration.')
""")

# ===========================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)

n_code = sum(1 for c in cells if c['cell_type'] == 'code')
print(f'wrote {OUT} - {len(cells)} cells ({n_code} code, '
      f'{len(cells) - n_code} markdown)')
