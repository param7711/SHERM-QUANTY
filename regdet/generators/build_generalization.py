"""
Builds generalization.ipynb -- a standalone notebook implementing
GENERALIZATION_PROTOCOL.md EXACTLY.

The question: RegDet V1.1 was developed entirely on Nifty 2h. Does the
ARCHITECTURE generalise across markets and timeframes, or is it tuned to that
one series? This notebook MEASURES; it never re-tunes.

Two grids:
  A  5 markets x 3 timeframes via yfinance   (real on Kaggle, synthetic here)
  B  3 FX pairs x {daily, weekly} via the Fed H.10 GitHub mirror (REAL here)

The headline is the OVERFITTING TEST: forward-return direction ordering and
HAC t computed SEPARATELY on the in-sample and out-of-sample spans of every
run, reported side by side, never pooled.

The engine is INLINED VERBATIM out of build_master_notebook_v2.py; the S / L / W
metrics, the ZigZag and guards G1-G4 are INLINED VERBATIM out of
build_stability_lag.py, so numbers are comparable across notebooks. Nothing
here imports or modifies the master, its generator, regime_scorecard.py or any
sibling generator.
"""
import json
import os
import re

HERE = '/tmp/claude-0/-home-user-SHERM-QUANTY/f1a44349-dfe0-5e0f-9b1f-b0b39e930e1b/scratchpad'
OUT = f'{HERE}/generalization.ipynb'
MASTER_GEN = f'{HERE}/build_master_notebook_v2.py'
SL_GEN = f'{HERE}/build_stability_lag.py'

# ---------------------------------------------------------------------------
# HARVEST 1 -- the engine cells from the master generator, VERBATIM.
# Same technique as build_stability_lag.py / build_hmm_share_charts.py: the
# generator's cell bodies are Python string literals with runtime
# concatenation, so they are obtained by EXECUTING the generator in a private
# namespace (output path redirected to a throwaway file) and reading its
# assembled `cells` list.
# ---------------------------------------------------------------------------
_gsrc = open(MASTER_GEN).read()
assert "regdet_v11_master.ipynb'" in _gsrc
_gsrc = _gsrc.replace("regdet_v11_master.ipynb'", "_generalization_engine_harvest.ipynb'", 1)
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
    and 'def regime_blocks' in SRC_PLOT

# ---------------------------------------------------------------------------
# HARVEST 2 -- the HAC contrast machinery, VERBATIM, out of the master's
# BAR_DIR_WEIGHT-sweep section. Sliced from the generator source by content
# markers (this code lives in a cell that also RUNS the sweep, which this
# notebook must not do, so the two function definitions are taken alone).
# ---------------------------------------------------------------------------
_i0 = _gsrc.index('def _w_hac_contrast(')
_i1 = _gsrc.index('def W_SECONDARY_PERBAR(')
SRC_HAC = _gsrc[_i0:_i1].rstrip() + '\n'
assert 'def _w_hac_contrast' in SRC_HAC and 'def W_PRIMARY_METRIC' in SRC_HAC
assert 'Bartlett' in SRC_HAC and 'np.linalg.pinv' in SRC_HAC
assert 'W_GRID' not in SRC_HAC and 'print(' not in SRC_HAC   # definitions only

# ---------------------------------------------------------------------------
# HARVEST 3 -- the ZigZag and the S / L / W + guard metrics, VERBATIM, out of
# build_stability_lag.py. That generator cannot be exec'd here (it asserts
# BAR_DIR_WEIGHT == 0.0 in the master constants, which is now 0.5), so its
# code-cell string literals are sliced out by content marker instead.
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
    os.remove(f'{HERE}/_generalization_engine_harvest.ipynb')
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
# RegDet V1.1 — CROSS-MARKET / CROSS-TIMEFRAME GENERALIZATION

Implements `GENERALIZATION_PROTOCOL.md` **exactly**. The protocol was frozen
before any number in this notebook existed.

## The question

RegDet V1.1 was developed end to end on **Nifty 2h over ~2.9 years**. Does the
**architecture** work anywhere else, or is it tuned to that one series?

This is an **architecture-level overfitting test, not a parameter search.**
Nothing here is allowed to re-tune RegDet. If a market fails, that is the
result — it is reported, not engineered around.

## Two grids

| grid | what | source | status here |
|---|---|---|---|
| **A** | 5 markets × 3 timeframes = 15 runs | yfinance 60m, resampled | **synthetic fallback in this sandbox** (yfinance is firewalled); real on Kaggle |
| **B** | 3 FX pairs × {daily, weekly} = 6 runs | US Federal Reserve H.10 via the `datasets` GitHub mirror | **REAL DATA, decision-grade for FX** |

## The headline — the overfitting test

For **every** run, the forward-return direction ordering and its HAC *t* are
computed **separately** on the in-sample (train) span and the out-of-sample
span, and reported **side by side**. They are **never pooled**.

> A detector that orders correctly in-sample and **inverts** out-of-sample is
> the classic overfit signature. That comparison is the point of this notebook.

## What is FIXED and what may re-derive

**Fixed everywhere, never re-tuned per market:** `N_STATES=5`, `cov='diag'`,
the 9 features, `CONF_L=0.50`, `CONFIRM_BARS=2`, `BAR_DIR_WEIGHT=0.5`,
`ENSEMBLE_K=4`, `Z_HI=0.5`, `EFF_HI=0.35`, `Z_HI_EXIT=0.35`, `EFF_HI_EXIT=0.25`,
`EFF_WIN=9`, momentum ladder 1/3/5 *days*, `CONTEXT_DAYS=12`,
`INTENSITY_MODE='frozen_z'`, `ESCALATION_DURING_HOLD='allow'`,
`DIRECTION_MODE='rank'`, `TRAIN_FRACTION=0.70`, `N_FOLDS=4`.

**May re-derive per market (data-dependent by design):** the `StandardScaler`,
and any fit-window baseline or quantile the engine already computes internally
(`trend_z`'s frozen μ/σ, the H-gate band).

**Nothing else.** Section 2 captures every fixed knob into a frozen dict and
re-asserts it after **every single run**. If a knob is re-tuned per market the
test is void, and the assert says so rather than the prose.
""")

co(r"""
%pip install -q hmmlearn yfinance
""")

# --- engine ----------------------------------------------------------------
md(f"""
## 1. Engine, inlined verbatim

Inlined out of `build_master_notebook_v2.py` rather than imported, because a
standalone `.ipynb` on Kaggle cannot import a local `.py` sitting next to it.
Source: `{PROVENANCE}`.

Not one line of the engine is modified — same constants, same 9 features, same
gates, same hysteresis, same causality proofs. The master notebook, its
generator, `regime_scorecard.py` and every sibling generator are untouched.

The **S / L / W metrics, the ZigZag and guards G1–G4** are inlined verbatim out
of `build_stability_lag.py` in section 2 (they have to follow their parameter
block), so the `S`, `L`, `W` and guard numbers here are directly comparable with
that notebook's.

The master's own `load_2h()` runs at the bottom of the loader cell and gives
this notebook its `TAC_SYNTH` flag and the `_synth()` generator that the Grid-A
fallback reuses. Its Nifty series is **not** used for any Grid-A or Grid-B run.
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
   + '# regime_blocks / shade_bands / shade_regimes / set_price_ylim / regime_legend.\n'
   + '# This is the MASTER NOTEBOOK regime-background style (full-height bands,\n'
   + '# each block extended to the START of the next so there are no white stripes,\n'
   + '# explicit non-zero-anchored y-limits).\n'
   + BANNER + SRC_PLOT)

co(BANNER
   + '# HAC (Newey-West) CONTRAST -- INLINED VERBATIM from the master generator\n'
   + '# (the two function definitions only; the sweep that surrounds them is not\n'
   + '# run here). This is the SAME estimator the master and the w-sweep use, so\n'
   + '# every t-stat in this notebook is comparable with theirs.\n'
   + BANNER + SRC_HAC)

# ===========================================================================
md(r"""
## 2. Frozen parameters, the fixed-knob invariant, and the leakage tripwire

Everything in this section is declared **before any run exists**.

### `BARS_PER_DAY` is the one thing that legitimately changes

Every *day*-denominated window (the 1/3/5-day momentum ladder, the 12-day
context window) is expressed as `days × BARS_PER_DAY`. A 1h bar and a 4h bar do
not contain the same amount of day, so holding the *bar count* fixed across
timeframes would silently change the horizon the detector looks at.

So `BASE_WIN` is recomputed per run from that run's `BARS_PER_DAY` — using the
**same formula** the master uses, with the **same day counts**. That is the
protocol's instruction, not a re-tune.

Two honest consequences, stated before the numbers:

* **`EFF_WIN = 9` is bar-denominated and is frozen by decree.** It therefore
  covers 9 hours at 1h and 36 hours at 4h. The protocol fixes it, so it stays
  fixed; but it means the chop filter's real-time horizon is *not* constant
  across the timeframe axis. This is a property of the frozen spec, flagged
  rather than fixed.
* `VOL_WIN=10` and `VOL_FAST=5` are likewise raw bar counts in the master and
  are left alone.

The **assumed** `BARS_PER_DAY` per market/timeframe is declared below; the
**realised** median bars per session is computed from each series and printed
next to it, with a warning when they disagree materially. Nothing is silently
rescaled.

### Forward horizons

`FWD_HORIZONS` is `[1, 3, 5] × BARS_PER_DAY`, i.e. ~1 / 3 / 5 **days** on every
run. At Nifty 2h that is exactly the master's `[3, 9, 15]`. Holding the raw bar
counts fixed instead would compare a 3-hour return on one row against a 12-hour
return on the next, which is not a comparison.

### The ZigZag leakage tripwire

`label_bars` is shadowed by a guard that asserts, on **every** call:
phase ordering (`ZZ_CALLS == 0`), object identity against every ZigZag output,
memory aliasing via `np.shares_memory`, and `bar_dir_weight == BAR_DIR_WEIGHT`.

> **One documented change from `build_stability_lag.py`.** That notebook's
> tripwire hard-codes `bar_dir_weight == 0.0`, because `BAR_DIR_WEIGHT` was
> `0.0` in the master at the time. The master now ships `0.5`, which is also
> what `GENERALIZATION_PROTOCOL.md` freezes. The check here is therefore
> `_bw == BAR_DIR_WEIGHT`, i.e. *the frozen master value*, whatever it is. The
> check is not weakened — it is anchored to the frozen constant instead of to a
> literal that has since moved.
""")

co(r"""
# ===========================================================================
# METRIC + GRID PARAMETERS -- ALL DECLARED HERE, BEFORE ANY NUMBER EXISTS.
# Values are IDENTICAL to build_stability_lag.py so the two notebooks compare.
# ===========================================================================
import itertools, time, contextlib, io, sys
import matplotlib.dates as mdates

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
# evaluate_guards reads OCC_MIN_PCT / W_GUARD_MAX / SIDEWAYS_MAX, so both cells
# must be defined after the parameter block above. (Same order as
# build_stability_lag.py, where the parameter cell also precedes them.)
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
   + '# Identical definitions to the stability/lag notebook so the numbers in the\n'
   + '# two notebooks are directly comparable.\n'
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
print('EFF_WIN stays FROZEN at %d BARS on every run (protocol decree). Real-time '
      'horizon therefore VARIES across the timeframe axis -- flagged, not fixed.'
      % EFF_WIN)
""")

co(r"""
# ===========================================================================
# THE LEAKAGE TRIPWIRE. `label_bars` is SHADOWED by a guard; every later call
# in this notebook goes through it. Four independent runtime checks.
#
# Structure taken from build_stability_lag.py. The ONE documented change: the
# decree check is `_bw == BAR_DIR_WEIGHT` (the frozen master/protocol value,
# 0.5) rather than the literal 0.0 that generator hard-codes.
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
""")

co(r"""
# ===========================================================================
# PRE-REGISTERED PREDICTIONS. Printed BEFORE any data is loaded or fitted.
# ===========================================================================
PREDICTIONS = {
 'P1': ('Guards G1-G4 pass on a MAJORITY of the 15 Grid-A runs.',
        'CONFIRMED if >= 8 of 15 Grid-A runs pass ALL FOUR guards. '
        'Broad failure means the architecture is Nifty-specific.'),
 'P2': ('SIDEWAYS occupancy lands in a similar band, roughly 20-45%, across markets.',
        f'CONFIRMED if a MAJORITY of Grid-A runs have SIDEWAYS in '
        f'[{P2_LO:.0f}%, {P2_HI:.0f}%]. Wild swings mean the thresholds are '
        'scale-dependent.'),
 'P3': ('OOS direction ordering DEGRADES vs IS but does not INVERT.',
        'CONFIRMED if the OOS (BULL - BEAR) mean-forward-return contrast keeps '
        'the SAME SIGN as IS on a MAJORITY of runs. Inversion on a majority = '
        'overfitting. Graded SEPARATELY on Grid A and Grid B.'),
 'P4': ('FX and equities behave DIFFERENTLY. A difference is EXPECTED and is '
        'NOT a failure.',
        'CONFIRMED if the FX and equity Grid-A groups separate on SIDEWAYS '
        'occupancy by more than the larger within-group spread. REFUTED here '
        'means they behaved ALIKE -- also informative, also not a failure.'),
}
print('=' * 110)
print('PRE-REGISTERED PREDICTIONS  (frozen; graded in section 9; not softened)')
print('=' * 110)
for k, (claim, rule) in PREDICTIONS.items():
    print(f'\n{k}.  {claim}')
    print(f'     GRADING RULE: {rule}')
print('\n' + '=' * 110)
print('NOT PREDICTED, and therefore not gradeable: absolute return, Sharpe, or')
print('any tradeability claim. This notebook measures the DETECTOR, not a strategy.')
print('=' * 110)
""")

# ===========================================================================
md(r"""
## 4. Data layer

### Grid A — yfinance (real on Kaggle, **synthetic here**)

One 60-minute download per market, then **OHLC aggregation** down to 2h and 4h
(`open=first, high=max, low=min, close=last`). Downward resampling only;
1h is never manufactured from anything coarser.

### The VIX question — answered explicitly, never silently substituted

`^INDIAVIX` pairs with Nifty and **only** with Nifty. For this grid:

| market | volatility input | kind |
|---|---|---|
| `^GSPC`, `^IXIC` | `^VIX` | **real implied-vol index** |
| `EURUSD=X`, `GBPUSD=X`, `JPY=X` | trailing realised vol | **PROXY — not a VIX** |
| Grid B (all) | trailing realised vol | **PROXY — not a VIX** |

There is no natural VIX for a spot FX cross, so the `vix_chg` feature is fed a
**causal trailing realised-volatility proxy**: the rolling standard deviation of
log returns over `VOL_SLOW` bars, annualised. It reads bars ≤ *t* only. Every
run prints its volatility source and its kind, and the proxy runs carry a
`vol-proxy` stamp through every table.

### Grid B — REAL FED DATA via GitHub mirror

`raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv` — the
US Federal Reserve **H.10** release, redistributed by the `datasets` org.
Long format (`Date, Country, Exchange rate`), filtered to Japan / United
Kingdom / Euro. Quoted **foreign currency per USD** (JPY ≈ 163, GBP ≈ 0.75,
EUR ≈ 0.88 at the right edge), so a *rise* in the series is USD strength.

Five sanity checks are re-run here, live, against known events. Daily is also
resampled to **weekly** for a cadence check at the long-horizon end.

> **Grid B is CLOSE-ONLY.** There is no high/low/open. This turns out to cost
> nothing: `build_features` in the master reads **only the close series and the
> volatility series** — all nine features are close-only by construction. So
> **no feature is skipped and none is degraded**, and no OHLC is fabricated from
> closes. The only place OHLC matters at all is Grid-A's 60m→2h/4h aggregation.
""")

co(r"""
# ===========================================================================
# GRID A -- market definitions and the assumed BARS_PER_DAY grid.
# ===========================================================================
GRID_A_MARKETS = [
    dict(sym='^GSPC',    name='S&P 500',  cls='EQ', vix='^VIX',
         session_h=6.5, tz_note='NYSE 09:30-16:00 = 6.5h'),
    dict(sym='^IXIC',    name='NASDAQ',   cls='EQ', vix='^VIX',
         session_h=6.5, tz_note='NASDAQ 09:30-16:00 = 6.5h'),
    dict(sym='EURUSD=X', name='EUR/USD',  cls='FX', vix=None,
         session_h=24.0, tz_note='spot FX ~24h weekday'),
    dict(sym='GBPUSD=X', name='GBP/USD',  cls='FX', vix=None,
         session_h=24.0, tz_note='spot FX ~24h weekday'),
    dict(sym='JPY=X',    name='USD/JPY',  cls='FX', vix=None,
         session_h=24.0, tz_note='spot FX ~24h weekday'),
]
GRID_A_TFS = [('1h', 1), ('2h', 2), ('4h', 4)]

# ASSUMED bars per session. The protocol says to STATE the choice per market:
# an equity session is ~6.5h, a spot FX weekday is ~24h, so a single number
# across both asset classes would be wrong for one of them. The realised value
# is measured from the data and printed beside the assumption.
def assumed_bpd(mkt, hours):
    return max(1, int(round(mkt['session_h'] / hours)))


print('GRID A -- 5 markets x 3 timeframes = 15 runs')
print(f"{'market':<10} {'class':<6} {'vol source':<16} {'session':<22} "
      f"{'BPD 1h':>7} {'BPD 2h':>7} {'BPD 4h':>7}")
for m in GRID_A_MARKETS:
    print(f"{m['sym']:<10} {m['cls']:<6} {(m['vix'] or 'realised-vol PROXY'):<16} "
          f"{m['tz_note']:<22} "
          + ''.join(f'{assumed_bpd(m, h):>7d}' for _, h in GRID_A_TFS))
print('\nThese are ASSUMPTIONS. Every run prints the REALISED median bars per '
      'session next to them and warns on a material disagreement.')
""")

co(r"""
# ===========================================================================
# OHLC RESAMPLING + the causal realised-vol VIX proxy.
# ===========================================================================
OHLC_AGG = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}


def resample_ohlc(df, rule):
    '''Downward OHLC aggregation. open=first, high=max, low=min, close=last.

    Downward only: `rule` must be coarser than the input cadence. Asserted.
    '''
    assert set(OHLC_AGG).issubset(df.columns), f'need OHLC columns, got {list(df.columns)}'
    step_in = pd.Series(df.index).diff().median()
    out = df.resample(rule).agg(OHLC_AGG).dropna(how='any')
    step_out = pd.Series(out.index).diff().median()
    assert step_out >= step_in, (f'UPWARD resampling attempted: {step_in} -> '
                                 f'{step_out}. Only downward resampling is allowed.')
    # internal consistency of the aggregation itself
    assert (out['High'] >= out[['Open', 'Close']].max(axis=1) - 1e-9).all()
    assert (out['Low'] <= out[['Open', 'Close']].min(axis=1) + 1e-9).all()
    return out


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


print('resample_ohlc (downward-only, asserted), realised_vol_proxy (causal), '
      'realised_bpd ready.')
""")

co(r"""
# ===========================================================================
# GRID A LOADER. Tries yfinance; falls back to synthetic with a LOUD banner.
# ===========================================================================
GRID_A_SYNTH = None      # set once the first market is loaded


def _yf_ohlc_60m(tk):
    import yfinance as yf
    raw = yf.download(tk, interval='60m', period='730d',
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    need = ['Open', 'High', 'Low', 'Close']
    if not set(need).issubset(raw.columns):
        return None
    out = raw[need].dropna(how='any')
    idx = pd.to_datetime(out.index)
    out.index = idx.tz_localize(None) if idx.tz is not None else idx
    return out if len(out) > 200 else None


def _synth_ohlc_60m(sym, bars_per_session, seed):
    '''Synthetic 60m OHLC with the market's OWN session structure.

    Deliberately built at the right cadence (6-7 bars/session for equities,
    24 for FX) so the resampling path, the realised-BPD check and the
    day-denominated windows are all exercised on realistic shapes. It is a
    PLUMBING TEST. It says nothing about any real market.
    '''
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
    if bars_per_session >= 20:                      # FX: 24 hourly bars a day
        hours = list(range(24))
    else:                                           # equities: 09:00..15:00
        hours = list(range(9, 9 + bars_per_session))
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in hours])
    n = len(idx)
    # regime-blocked GBM, same spirit as the master's _synth
    REG = [(0.00, 0.15, 6e-5, 0.0018), (0.15, 0.30, -1e-4, 0.0032),
           (0.30, 0.45, 2e-5, 0.0014), (0.45, 0.55, -5e-4, 0.0058),
           (0.55, 0.75, 1e-4, 0.0026), (0.75, 0.90, 9e-5, 0.0016),
           (0.90, 1.00, -3e-5, 0.0023)]
    lvl, o, h_, l_, c_ = 100.0, [], [], [], []
    for i in range(n):
        f = i / n
        mu, sg = 3e-5, 0.0020
        for fs, fe, m_, s_ in REG:
            if fs <= f < fe:
                mu, sg = m_, s_
                break
        op = lvl
        lvl *= np.exp(rng.normal(mu, sg))
        wick = abs(rng.normal(0, sg * 0.6))
        o.append(op); c_.append(lvl)
        h_.append(max(op, lvl) * (1 + wick)); l_.append(min(op, lvl) * (1 - wick))
    return pd.DataFrame({'Open': o, 'High': h_, 'Low': l_, 'Close': c_}, index=idx)


def load_grid_a(mkt):
    '''-> (ohlc_60m, vix_60m_or_None, is_synth, vix_kind, source_note)'''
    global GRID_A_SYNTH
    try:
        df = _yf_ohlc_60m(mkt['sym'])
        if df is None:
            raise ValueError('yfinance returned nothing usable')
        vx, kind = None, 'realised-vol PROXY (causal)'
        if mkt['vix']:
            raw = _yf_ohlc_60m(mkt['vix'])
            if raw is not None and len(raw):
                # ffill ONLY. bfill would take a value from the future -- the
                # exact look-ahead the master's loader already fixes.
                vx = raw['Close'].reindex(df.index).ffill()
                kind = f"REAL {mkt['vix']}"
        print(f"  {mkt['sym']:<10} yfinance 60m: {len(df)} bars   vol={kind}")
        return df, vx, False, kind, f"yfinance 60m period=730d ({mkt['sym']})"
    except Exception as e:
        print(f"  {mkt['sym']:<10} yfinance unavailable ({type(e).__name__}: "
              f"{str(e)[:70]}) -> SYNTHETIC")
        bps = 7 if mkt['cls'] == 'EQ' else 24
        # DETERMINISTIC seed. Python's builtin hash() is salted per process
        # (PYTHONHASHSEED), so deriving the seed from it made the synthetic
        # fallback -- and therefore every Grid-A number -- change between runs
        # of the SAME notebook. zlib.crc32 is stable across processes.
        import zlib
        seed = 1000 + zlib.crc32(mkt['sym'].encode()) % 10000
        df = _synth_ohlc_60m(mkt['sym'], bps, seed)
        return (df, None, True, 'realised-vol PROXY (causal)',
                f"SYNTHETIC 60m OHLC ({bps} bars/session, seed {seed})")


t0 = time.time()
GRID_A_RAW = {}
print('Loading Grid A ...')
for _m in GRID_A_MARKETS:
    GRID_A_RAW[_m['sym']] = load_grid_a(_m)
GRID_A_SYNTH = any(v[2] for v in GRID_A_RAW.values())
print(f'Grid A raw load done in {time.time() - t0:.1f}s')

if GRID_A_SYNTH:
    print('#' * 100)
    for _ in range(3):
        print('#   GRID A IS SYNTHETIC - ILLUSTRATIVE ONLY   ' * 2)
    print('#' * 100)
    print('#  yfinance is FIREWALLED in this sandbox. Every Grid-A number below is')
    print('#  generated from regime-blocked GBM. It is a PLUMBING TEST ONLY. It says')
    print('#  NOTHING about real S&P/NASDAQ/FX regimes and CANNOT confirm or refute')
    print('#  P1, P2, P3 or P4 as claims about real markets.')
    print('#  GRID B BELOW IS REAL DATA and IS decision-grade for FX.')
    print('#  Re-run on Kaggle, where yfinance works, before believing Grid A.')
    print('#' * 100)
else:
    print('=' * 100)
    print('REAL yfinance data across all Grid-A markets. Grid A is decision-grade.')
    print('=' * 100)
""")

co(r"""
# ===========================================================================
# GRID B LOADER -- REAL FED DATA via GitHub mirror, with LIVE sanity checks.
# ===========================================================================
GRID_B_URL = ('https://raw.githubusercontent.com/datasets/exchange-rates/'
              'main/data/daily.csv')
GRID_B_PAIRS = [
    dict(country='Japan',          name='JPY/USD', code='JPY'),
    dict(country='United Kingdom', name='GBP/USD', code='GBP'),
    dict(country='Euro',           name='EUR/USD', code='EUR'),
]

GRID_B_OK = True
GRID_B_FAIL = ''
try:
    _t = time.time()
    _raw = pd.read_csv(GRID_B_URL, parse_dates=['Date'])
    print(f'fetched {GRID_B_URL}')
    print(f'  {len(_raw)} rows, {_raw["Country"].nunique()} countries, '
          f'{time.time() - _t:.1f}s')
except Exception as e:
    GRID_B_OK, GRID_B_FAIL = False, f'{type(e).__name__}: {e}'
    _raw = None
    print(f'GRID B FETCH FAILED: {GRID_B_FAIL}')
    print('Grid B rows will appear in the tables with this reason. Nothing is '
          'dropped and nothing is substituted.')

GRID_B_SERIES = {}
if GRID_B_OK:
    for p in GRID_B_PAIRS:
        s = _raw[_raw['Country'] == p['country']].dropna(subset=['Exchange rate'])
        s = s[s['Exchange rate'] > 0].set_index('Date')['Exchange rate'].sort_index()
        s.name = p['code']
        GRID_B_SERIES[p['code']] = s

    print('\n' + '=' * 100)
    print('GRID B PROVENANCE:  REAL FED DATA via GitHub mirror')
    print('  US Federal Reserve H.10 release, redistributed by the `datasets` org.')
    print('  Quoted FOREIGN CURRENCY PER USD -- a RISE in the series is USD strength.')
    print('  CLOSE-ONLY: no open/high/low exists and none is fabricated.')
    print('=' * 100)
    print(f"{'pair':<9} {'bars':>7}  {'span':<26} {'first':>10} {'last':>10}")
    for p in GRID_B_PAIRS:
        s = GRID_B_SERIES[p['code']]
        print(f"{p['name']:<9} {len(s):>7}  "
              f"{s.index[0]:%Y-%m-%d} -> {s.index[-1]:%Y-%m-%d}   "
              f"{s.iloc[0]:>10.4f} {s.iloc[-1]:>10.4f}")
""")

co(r"""
# ===========================================================================
# GRID B SANITY CHECKS -- re-run LIVE against known events. Nothing is trusted
# on the strength of somebody having checked it before.
#
# The event magnitudes in GENERALIZATION_PROTOCOL.md are LOG returns
# (ln(1+r)); both are printed so the check is unambiguous.
# ===========================================================================
GRID_B_CHECKS = []
if GRID_B_OK:
    def _chk(name, ok, detail):
        GRID_B_CHECKS.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")

    print('Grid B structural checks')
    for p in GRID_B_PAIRS:
        s = GRID_B_SERIES[p['code']]
        _chk(f"{p['name']} dates strictly increasing, no duplicates",
             s.index.is_monotonic_increasing and not s.index.duplicated().any(),
             f'{len(s)} bars')
        _chk(f"{p['name']} all prices strictly positive",
             bool((s > 0).all()) and bool(np.isfinite(s).all()),
             f'min {s.min():.4f}  max {s.max():.4f}')

    print('\nGrid B event checks (protocol figures are LOG returns)')
    _ev = [('GBP', '2016-06-24', +8.17, 'Brexit referendum result'),
           ('JPY', '1973-02-13', -9.50, 'Bretton Woods collapse / yen float')]
    for code, day, exp_log, why in _ev:
        s = GRID_B_SERIES[code]
        d = pd.Timestamp(day)
        if d in s.index:
            i = s.index.get_loc(d)
            simple = 100.0 * (s.iloc[i] / s.iloc[i - 1] - 1.0)
            logr = 100.0 * np.log(s.iloc[i] / s.iloc[i - 1])
            _chk(f'{code} {day} {why}', abs(logr - exp_log) < 0.15,
                 f'log {logr:+.2f}% (expected {exp_log:+.2f}%) | simple {simple:+.2f}%')
        else:
            _chk(f'{code} {day} {why}', False, 'date absent from series')

    for code in ('GBP', 'JPY'):
        s = GRID_B_SERIES[code]
        r = (np.log(s / s.shift(1)) * 100.0).dropna()
        oct08 = r.loc['2008-10']
        _chk(f'{code} GFC Oct-2008 volatility spike',
             oct08.std() > 2.0 * r.std(),
             f'Oct-08 sd {oct08.std():.2f}% vs full-sample {r.std():.2f}%  '
             f'(max |move| {oct08.abs().max():.2f}% on '
             f'{oct08.abs().idxmax():%Y-%m-%d})')

    _npass = sum(1 for _, o, _ in GRID_B_CHECKS if o)
    print(f'\nGRID B SANITY: {_npass}/{len(GRID_B_CHECKS)} checks PASS')
    assert _npass == len(GRID_B_CHECKS), \
        'a Grid B sanity check FAILED -- the source data is not what the protocol says'
    print('ASSERT OK: every Grid B sanity check passed on data fetched in THIS run.')

    print('\nNOTE ON ROW COUNTS. The protocol lists Japan 13,926 / United Kingdom')
    print('13,932 / Euro 7,190. The first two match exactly after dropping missing')
    print('and non-positive quotes. The Euro figure 7,190 is the RAW row count')
    print('INCLUDING blank holiday rows; after the same cleaning it is')
    print(f"{len(GRID_B_SERIES['EUR'])}. Reported, not reconciled away.")
""")

# ===========================================================================
md(r"""
## 5. The run harness

One function builds features, fits **R = 4 disjoint seed sets** of `ENSEMBLE_K`
HMMs on the leading 70% of bars, and labels the full series with each. Nothing
is tuned; the only per-run inputs are the price series, the volatility series
and `BARS_PER_DAY`.

**R = 4 is never cut.** `S` — seed-set label agreement — is meaningless without
it, and the protocol says so. If runtime has to be cut, the timeframe axis of
Grid A goes first. (It did not need to be cut: see the runtime line in section 8.)

### Phase discipline

Every label in this notebook is produced in the **LABEL PHASE**, which closes
before the first ZigZag is computed. The tripwire proves it. All Grid-A and
Grid-B runs are labelled first; only then does evaluation begin.
""")

co(r"""
# ===========================================================================
# THE RUN HARNESS. Fits + labels ONE (market, timeframe) run.
# ===========================================================================
N_STATES_FIXED = FIXED_KNOBS['N_STATES']
COV_FIXED = FIXED_KNOBS['COVARIANCE']

FIT_COUNT = 0


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
    sc = StandardScaler().fit(raw[:n_fit])       # fit on the LEADING window only
    Xs = sc.transform(raw)
    return dict(feat=feat, dates=dates, n_bars=n_bars, n_fit=n_fit,
                Xs=Xs, scaler=sc,
                close=close.reindex(dates), trend_raw=feat[TREND_FEATURE].values)


def label_run(prep, close, bpd, models_by_set=None):
    '''Label the full series once per seed set. Returns (label_sets, models).'''
    global FIT_COUNT
    out, mods = [], []
    for si, sd in enumerate(SEED_SETS):
        if models_by_set is None:
            ms, _conv = fit_hmm_ensemble(prep['Xs'][:prep['n_fit']],
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


def build_run(run_id, grid, market, tf, close, volser, vix_kind, bpd_assumed,
              source, note=''):
    '''Full LABEL-PHASE work for one run. Returns a run dict (or an error row).'''
    rec = dict(run_id=run_id, grid=grid, market=market, tf=tf,
               vix_kind=vix_kind, source=source, note=note,
               bpd_assumed=bpd_assumed,
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
                   span=(prep['dates'][0], prep['dates'][-1]),
                   win=windows_for(bpd_assumed),
                   fwd_h=fwd_horizons_for(bpd_assumed))
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
    assert_knobs_unchanged(f'after run {run_id}')
    return rec


print('run harness ready: prepare_run / label_run / build_run.')
print('Per-run inputs are ONLY (close, volatility, BARS_PER_DAY). No knob is a '
      'function of the market.')
""")

co(r"""
# ===========================================================================
# LABEL PHASE -- GRID A. 15 runs. No ZigZag exists yet; the tripwire proves it.
# ===========================================================================
RUNS = []
t0 = time.time()
print(f"{'run':<22} {'bars':>6} {'fit':>6} {'BPD asm':>8} {'BPD real':>9} "
      f"{'windows (mom/ctx)':<20} {'vol source':<24} {'secs':>6}")
print('-' * 118)

for m in GRID_A_MARKETS:
    df60, vx60, is_syn, vix_kind, source = GRID_A_RAW[m['sym']]
    for tf_name, hours in GRID_A_TFS:
        tt = time.time()
        rid = f"{m['sym']}@{tf_name}"
        bpd = assumed_bpd(m, hours)
        if hours == 1:
            bars = df60.copy()
        else:
            bars = resample_ohlc(df60, f'{hours}h')
        close = bars['Close'].astype(float)
        close.name = m['sym']
        if vx60 is not None:
            vol = vx60.resample(f'{hours}h').last().reindex(close.index).ffill() \
                  if hours != 1 else vx60.reindex(close.index).ffill()
            vol.name = 'vix'
            vk = vix_kind
        else:
            with bars_per_day(bpd):
                vol = realised_vol_proxy(close, bpd)
            vk = 'realised-vol PROXY (causal)'
        r = build_run(rid, 'A', m['name'], tf_name, close, vol, vk, bpd,
                      source, note=m['cls'])
        r['cls'] = m['cls']
        r['sym'] = m['sym']
        RUNS.append(r)
        if r['error']:
            print(f"{rid:<22} {'':>6} {'':>6} {bpd:>8} "
                  f"{r['bpd_realised']:>9.1f}  ERROR: {r['error']}")
        else:
            w = r['win']
            print(f"{rid:<22} {r['n_bars']:>6} {r['n_fit']:>6} {bpd:>8} "
                  f"{r['bpd_realised']:>9.1f} "
                  f"{str(w['MOM_1D']) + '/' + str(w['MOM_3D']) + '/' + str(w['MOM_5D']) + '  ctx ' + str(w['SWING_WIN']):<20} "
                  f"{vk:<24} {time.time() - tt:>6.1f}")

print(f'\nGrid A label phase: {time.time() - t0:.1f}s, {FIT_COUNT} HMM fits so far.')
""")

co(r"""
# ===========================================================================
# BARS_PER_DAY: ASSUMED vs REALISED. Printed, warned on, never silently rescaled.
# ===========================================================================
print('=' * 110)
print('BARS PER SESSION -- ASSUMED vs REALISED (measured from the data itself)')
print('=' * 110)
print(f"{'run':<22} {'assumed':>8} {'realised':>9} {'ratio':>7}  verdict")
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
    print(f"{r['run_id']:<22} {a:>8} {b:>9.2f} {ratio:>7.2f}  {v}")

print()
if BPD_WARNINGS:
    print(f'{len(BPD_WARNINGS)} run(s) disagree materially with their assumed '
          'BARS_PER_DAY.')
    print('THE ASSUMPTION IS NOT CHANGED. Doing so would re-tune a window per')
    print('market, which the protocol forbids. The discrepancy is reported and')
    print('carried into the tables as a caveat on those rows.')
else:
    print('No material disagreement: every run realises within [0.80, 1.25]x its '
          'assumed bars per session.')
""")

co(r"""
# ===========================================================================
# LABEL PHASE -- GRID B. 3 pairs x {daily, weekly}. REAL DATA.
# Daily -> weekly only. Daily -> intraday is IMPOSSIBLE and is not attempted.
# ===========================================================================
t0 = time.time()
if not GRID_B_OK:
    for p in GRID_B_PAIRS:
        for cad in ('daily', 'weekly'):
            RUNS.append(dict(run_id=f"{p['name']}@{cad}", grid='B', cls='FX',
                             sym=p['code'], market=p['name'], tf=cad,
                             vix_kind='n/a', source=GRID_B_URL,
                             note='', bpd_assumed=1, bpd_realised=np.nan,
                             error=f'Grid B fetch failed: {GRID_B_FAIL}'))
    print('Grid B unavailable; 6 error rows recorded (nothing dropped).')
else:
    print(f"{'run':<22} {'bars':>6} {'fit':>6} {'span':<26} "
          f"{'windows (mom/ctx)':<20} {'secs':>6}")
    print('-' * 96)
    for p in GRID_B_PAIRS:
        s_daily = GRID_B_SERIES[p['code']]
        # weekly: CLOSE-ONLY series -> last observation of each week. There is no
        # high/low to aggregate, and none is invented.
        s_weekly = s_daily.resample('W-FRI').last().dropna()
        for cad, ser, bpd in (('daily', s_daily, 1), ('weekly', s_weekly, 1)):
            tt = time.time()
            rid = f"{p['name']}@{cad}"
            ser = ser.astype(float)
            ser.name = p['code']
            with bars_per_day(bpd):
                vol = realised_vol_proxy(ser, 1 if cad == 'daily' else 1)
            r = build_run(rid, 'B', p['name'], cad, ser, vol,
                          'realised-vol PROXY (causal)', bpd,
                          f'FED H.10 via GitHub mirror ({p["country"]})',
                          note='FX')
            r['cls'] = 'FX'
            r['sym'] = p['code']
            RUNS.append(r)
            if r['error']:
                print(f'{rid:<22} ERROR: {r["error"]}')
            else:
                w = r['win']
                print(f"{rid:<22} {r['n_bars']:>6} {r['n_fit']:>6} "
                      f"{r['span'][0]:%Y-%m-%d} -> {r['span'][1]:%Y-%m-%d}   "
                      f"{str(w['MOM_1D']) + '/' + str(w['MOM_3D']) + '/' + str(w['MOM_5D']) + '  ctx ' + str(w['SWING_WIN']):<20} "
                      f"{time.time() - tt:>6.1f}")

print(f'\nGrid B label phase: {time.time() - t0:.1f}s')
print(f'TOTAL: {FIT_COUNT} HMM fits, {LABEL_CALLS} label_bars calls, '
      f'{len(RUNS)} runs.')
assert ZZ_CALLS == 0, 'LEAKAGE: a ZigZag existed during the label phase'
print('ASSERT OK: ZZ_CALLS == 0 -- every label in this notebook was produced '
      'BEFORE any ZigZag was computed.')
print()
print('NOTE on Grid-B weekly BARS_PER_DAY. At daily and weekly cadence there is')
print('at most one bar per session, so BARS_PER_DAY = 1 and the day-denominated')
print('windows become 1/3/5 BARS with a 12-BAR context. At weekly that is')
print('1/3/5 WEEKS and a 12-WEEK context -- a genuinely longer real horizon.')
print('That is the cadence robustness this grid is testing, and it is stated')
print('rather than hidden inside a rescale.')
""")

# ===========================================================================
md(r"""
## 6. EVAL PHASE — S / L / W, the guards, and **the overfitting test**

The label phase is now closed. From here on any `label_bars` call fails the
tripwire unless the phase is explicitly and loudly reopened (only the truncation
probes do that).

### The overfitting test, precisely

For each run and each forward horizon *h* ∈ {1, 3, 5} days:

* `fwd_h[t] = C[t+h]/C[t] - 1` — a *retrospective grade*, exactly like the
  ZigZag. It grades a label that was already emitted; it never enters one.
* `BULL = {H_BULL, L_BULL}`, `BEAR = {H_BEAR, L_BEAR}`, everything else is the
  control group (kept in the design so the Newey–West lag structure sees the
  real bar spacing — the master's reasoning, verbatim with the estimator).
* The HAC (Newey–West, Bartlett, lag = *h*−1) *t* of (mean fwd | BULL) −
  (mean fwd | BEAR) is computed on the **IS span** and the **OOS span**
  **separately**.
* The IS span is the leading `n_fit` bars **minus an embargo of max(h) bars**,
  so no in-sample forward return can read an out-of-sample price.
* **No pooled number is reported anywhere.**

The reported per-run figure is the mean over *h* of the HAC *t* — the master's
`W_PRIMARY_METRIC`, so it is directly comparable with the w-sweep — evaluated
once per seed set and averaged, with the seed spread carried alongside.

**Ordering** is graded on the 5-label monotonicity of mean forward return:
`H_BULL ≥ L_BULL ≥ SIDEWAYS ≥ L_BEAR ≥ H_BEAR`. `HOLDS` / `PARTIAL` (BULL block
above BEAR block but not fully monotone) / `BROKEN`.

**Inversion** — the thing P3 is about — is `sign(OOS BULL−BEAR) ≠ sign(IS
BULL−BEAR)`, evaluated on the h-averaged contrast.
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
# GRID A RESULTS TABLE. Every run appears. Nothing is dropped for looking bad.
# ===========================================================================
def guard_str(r):
    if r['error']:
        return '----'
    return ''.join(('.' if r['guards'][k][0] else 'X') for k in ('G1', 'G2', 'G3', 'G4'))


def fmt(v, w=6, p=2):
    return f'{v:>{w}.{p}f}' if (v is not None and np.isfinite(v)) else ' ' * (w - 3) + 'n/a'


def print_table(rows, title):
    print('=' * 138)
    print(title)
    print('=' * 138)
    print(f"{'run':<20} {'bars':>6} {'H_BULL':>7} {'L_BULL':>7} {'SIDE':>7} "
          f"{'L_BEAR':>7} {'H_BEAR':>7} {'S%':>6} {'W':>6} {'L':>6} "
          f"{'guards':>7} {'IS t':>7} {'OOS t':>7} {'IS ord':>8} {'OOS ord':>8} {'INV':>4}")
    print('-' * 138)
    for r in rows:
        if r['error']:
            print(f"{r['run_id']:<20} {'':>6}   ERROR: {r['error']}")
            continue
        o = r['occ']
        print(f"{r['run_id']:<20} {r['n_bars']:>6} "
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


A_RUNS = [r for r in RUNS if r['grid'] == 'A']
B_RUNS = [r for r in RUNS if r['grid'] == 'B']

_tag = '  [SYNTHETIC - PLUMBING ONLY]' if GRID_A_SYNTH else '  [REAL yfinance]'
print_table(A_RUNS, 'GRID A -- 5 markets x 3 timeframes' + _tag)
print()
print_table(B_RUNS, 'GRID B -- FED H.10 daily FX via GitHub mirror  [REAL DATA]')
""")

co(r"""
# ===========================================================================
# THE OVERFITTING TABLE. IS and OOS SIDE BY SIDE, per horizon. NEVER POOLED.
# ===========================================================================
print('=' * 132)
print('IN-SAMPLE vs OUT-OF-SAMPLE  --  (mean fwd return | BULL) - (mean fwd | BEAR)')
print('HAC (Newey-West, Bartlett) t. Spans are DISJOINT; the IS span is embargoed')
print('by max(h) bars so no in-sample forward return can read an out-of-sample price.')
print('=' * 132)
print(f"{'run':<20} {'IS n':>6} {'OOS n':>6} {'emb':>4} | "
      f"{'IS d%':>7} {'IS t':>7} {'seed sprd':>9} | "
      f"{'OOS d%':>7} {'OOS t':>7} {'seed sprd':>9} | {'t change':>9} {'verdict':<22}")
print('-' * 132)
for r in RUNS:
    if r['error']:
        print(f"{r['run_id']:<20}   ERROR: {r['error']}")
        continue
    dt = (r['OOS_T'] - r['IS_T']) if (np.isfinite(r['OOS_T']) and np.isfinite(r['IS_T'])) else np.nan
    if r['inverted']:
        verdict = 'INVERTED (overfit sig.)'
    elif np.isfinite(dt) and abs(r['OOS_T']) < abs(r['IS_T']):
        verdict = 'degraded, same sign'
    elif np.isfinite(dt):
        verdict = 'held or improved'
    else:
        verdict = 'unmeasurable'
    print(f"{r['run_id']:<20} {r['IS_n']:>6} {r['OOS_n']:>6} {r['embargo']:>4} | "
          + fmt(r['IS_D'], 7, 3) + ' ' + fmt(r['IS_T'], 7, 2) + ' '
          + fmt(r['IS_T_spread'], 9, 2) + ' | '
          + fmt(r['OOS_D'], 7, 3) + ' ' + fmt(r['OOS_T'], 7, 2) + ' '
          + fmt(r['OOS_T_spread'], 9, 2) + ' | '
          + fmt(dt, 9, 2) + f' {verdict:<22}')
print('-' * 132)
_ok = [r for r in RUNS if not r['error']]
_inv = [r for r in _ok if r['inverted']]
print(f'{len(_inv)} of {len(_ok)} runs INVERTED out of sample.')
print('A pooled IS+OOS number is NOT computed anywhere in this notebook, by design.')
""")

co(r"""
# ===========================================================================
# PER-LABEL FORWARD RETURN, IS vs OOS, for every run. The ordering, in full.
# ===========================================================================
print('=' * 120)
print('MEAN FORWARD RETURN (%) BY LABEL -- averaged over the 1/3/5-day horizons')
print('IS row then OOS row for each run. Correct ordering is monotone DECREASING.')
print('=' * 120)
print(f"{'run':<20} {'span':<4} " + ' '.join(f'{l:>9}' for l in ORDER) + '  status')
print('-' * 120)
for r in RUNS:
    if r['error']:
        continue
    for tag, mm, st in (('IS', r['means_is'], r['ord_is']),
                        ('OOS', r['means_oos'], r['ord_oos'])):
        print(f"{(r['run_id'] if tag == 'IS' else ''):<20} {tag:<4} "
              + ' '.join(fmt(mm[l], 9, 4) for l in ORDER) + f'  {st}')
print('-' * 120)
print('HOLDS   = fully monotone across all five labels')
print('PARTIAL = the BULL block still sits above the BEAR block, but not monotone')
print('BROKEN  = the BULL block does NOT sit above the BEAR block')
print('THIN    = too few bars in too many labels to grade')
""")

# ===========================================================================
md(r"""
## 7. Machinery proofs

Three things are *driven*, not asserted in prose:

1. **G4 bites.** A degenerate all-`SIDEWAYS` labelling is pushed through the
   **same** `evaluate_guards` the real runs use. It scores a perfect `S = 100%`
   — which is exactly how `S` is gamed — and must be VOIDED.
2. **Truncation probe**, one run per grid. Cutting the raw series at *T* and
   rebuilding features, scaling with the same scaler and labelling with the same
   models must leave every label at *t* ≤ *T* **bit-identical**.
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
# PROOF 2 -- TRUNCATION PROBE, one run per grid.
#
# Cut the RAW price/volatility series at bar T_raw, rebuild features from
# scratch, scale with the SAME scaler, label with the SAME fitted models and
# the SAME n_fit. Every label at t <= T must be bit-identical.
#
# WHAT THIS PROVES: the feature construction and the labelling path use bars
# <= t only. WHAT IT DOES NOT PROVE: anything about refitting -- a model refit
# on a shorter window is a DIFFERENT model by design, so the probe deliberately
# holds the fit fixed. That is the same scope the master's probe has.
# ===========================================================================
def truncation_probe(r, frac=0.85):
    if r['error']:
        return None
    close_full = r['close_ser']
    dates_full, labels_full = r['dates'], r['labels'][0]
    ser_raw = r['_raw_close']
    vol_raw = r['_raw_vol']
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
for _grid in ('A', 'B'):
    _cands = [r for r in RUNS if r['grid'] == _grid and not r['error']]
    if _cands:
        _probes.append(truncation_probe(_cands[0]))

print('=' * 100)
print('TRUNCATION PROBE')
print('=' * 100)
for p in _probes:
    if p is None:
        continue
    print(f"  {p['run']:<20} cut at {p['cut_at']:%Y-%m-%d %H:%M} "
          f"({p['T_raw']} raw bars) -> {p['n_common']} overlapping labels, "
          f"{p['n_diff']} differ")
    assert p['n_common'] > 100, 'probe overlap too small to be meaningful'
    assert p['n_diff'] == 0, (f"LOOK-AHEAD: {p['n_diff']} labels changed when the "
                              f"series was cut at {p['cut_at']}")
print('\nPROOF 2 PASS: every label at t <= T is BIT-IDENTICAL after truncation, '
      'on one run per grid.')
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
## 8. Figures

Four figures, each of which decides something:

1. **Grid-A heatmap** — 15 runs × the metrics that drive the guards.
2. **IS vs OOS** — the overfitting view, every run on one axis.
3. **Grid-B long-history regime chart** — 55 years of JPY/USD in the master's
   regime-background style, with a recent zoom.
4. **Grid-B cadence** — daily vs weekly occupancy, the long-horizon robustness
   check.
""")

co(r"""
# ===========================================================================
# FIGURE 1 -- GRID A HEATMAP.
# ===========================================================================
FIGS = []
_ok_a = [r for r in A_RUNS if not r['error']]
if _ok_a:
    # Each column declares which DIRECTION is better, so green always means
    # "better on this metric". A min-max ramp with no direction would paint a
    # high SIDEWAYS%, a high whipsaw W and a long lag L green, and would paint
    # an all-pass 'guards' column solid red because the column is constant.
    #   +1  higher is better        -1  lower is better
    #   'band'   good = inside the pre-registered P2 band
    #   'binary' good = 1, bad = 0, no scaling
    cols = [('SIDEWAYS %', lambda r: r['occ']['SIDEWAYS'], '%.1f', 'band'),
            ('S (agree %)', lambda r: r['S'], '%.1f', +1),
            ('W /100 bars', lambda r: r['W'], '%.1f', -1),
            ('L (bars)', lambda r: r['L'], '%.0f', -1),
            ('min occ %', lambda r: min(r['occ'].values()), '%.1f', +1),
            ('IS t', lambda r: r['IS_T'], '%.2f', +1),
            ('OOS t', lambda r: r['OOS_T'], '%.2f', +1),
            ('guards pass', lambda r: 1.0 if r['guards_ok'] else 0.0, '%.0f',
             'binary')]
    M = np.array([[c[1](r) for c in cols] for r in _ok_a], dtype=float)
    Z = np.full_like(M, np.nan)
    for j, (_, _, _, direction) in enumerate(cols):
        col = M[:, j]
        f = np.isfinite(col)
        if direction == 'binary':
            Z[:, j] = np.where(col > 0.5, 1.0, 0.0)
        elif direction == 'band':
            Z[:, j] = np.where((col >= P2_LO) & (col <= P2_HI), 1.0, 0.0)
        elif f.sum() and np.nanmax(col) > np.nanmin(col):
            z = (col - np.nanmin(col)) / (np.nanmax(col) - np.nanmin(col))
            Z[:, j] = z if direction > 0 else 1.0 - z
        else:
            Z[f, j] = 0.5            # constant column -> NEUTRAL, never red
        Z[~f, j] = np.nan

    fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(_ok_a) + 3.4))
    im = ax.imshow(Z, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] + ('  (in band)' if c[3] == 'band' else
                                '  (pass/fail)' if c[3] == 'binary' else
                                '  (high=good)' if c[3] > 0 else '  (low=good)')
                        for c in cols], rotation=30, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(_ok_a)))
    ax.set_yticklabels([r['run_id'] for r in _ok_a], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = ('n/a' if not np.isfinite(v)
                   else ('PASS' if v > 0.5 else 'FAIL') if cols[j][3] == 'binary'
                   else cols[j][2] % v)
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.5,
                    color='black')
    ax.set_title('GRID A -- 15 runs x key metrics'
                 + ('   [SYNTHETIC - PLUMBING ONLY]' if GRID_A_SYNTH
                    else '   [REAL yfinance]')
                 + '\nGREEN = BETTER on that column. Colour is a WITHIN-COLUMN '
                   'rank only -- there is NO composite score and columns are '
                   'not comparable to each other.\n'
                   f'SIDEWAYS is coloured by the pre-registered P2 band '
                   f'[{P2_LO:.0f}%, {P2_HI:.0f}%]; guards is pass/fail, not scaled.',
                 fontsize=9.5)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(_ok_a), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label='within-column position (min -> max)')
    fig.tight_layout()
    FIGS.append('fig1_grid_a_heatmap')
    plt.show()
else:
    print('no Grid-A run completed; heatmap skipped (rows appear in the table '
          'with their error).')
""")

co(r"""
# ===========================================================================
# FIGURE 2 -- THE OVERFITTING VIEW. IS vs OOS, every run.
# ===========================================================================
_ok_all = [r for r in RUNS if not r['error']]
if _ok_all:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 0.30 * len(_ok_all) + 3.6),
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
    axL.set_yticklabels(lab, fontsize=7.5)
    axL.invert_yaxis()
    axL.set_xlabel('HAC t of (mean fwd | BULL) - (mean fwd | BEAR)')
    axL.set_title('IS vs OOS direction contrast, per run\n'
                  'dotted lines = |t| = 2. NEVER POOLED.', fontsize=10)
    axL.legend(fontsize=8, loc='lower right')
    _xlo = min(0.0, float(np.nanmin(np.concatenate([isv, oov])))) * 1.05 - 0.15
    _xhi = max(0.0, float(np.nanmax(np.concatenate([isv, oov])))) * 1.32 + 0.15
    axL.set_xlim(_xlo, _xhi)
    for i, r in enumerate(_ok_all):
        if r['inverted']:
            # placed hard against the RIGHT margin, clear of every bar
            axL.text(_xhi, i, 'INVERTED ', fontsize=7, color='crimson',
                     va='center', ha='right', fontweight='bold')
        if r['grid'] == 'B':
            axL.get_yticklabels()[i].set_color('#1a7f37')

    fin = np.isfinite(isv) & np.isfinite(oov)
    axR.axhline(0, color='black', lw=1)
    axR.axvline(0, color='black', lw=1)
    lim = np.nanmax(np.abs(np.concatenate([isv[fin], oov[fin]]))) * 1.15 + 0.2
    axR.plot([-lim, lim], [-lim, lim], color='grey', ls='--', lw=1,
             label='no degradation (y = x)')
    axR.fill_between([-lim, 0], 0, lim, color='crimson', alpha=0.07)
    axR.fill_between([0, lim], -lim, 0, color='crimson', alpha=0.07)
    for i, r in enumerate(_ok_all):
        if not fin[i]:
            continue
        mk = 'o' if r['grid'] == 'A' else 's'
        axR.scatter(isv[i], oov[i], marker=mk, s=52,
                    color=('crimson' if inv[i] else
                           ('#1a7f37' if r['grid'] == 'B' else '#4C72B0')),
                    edgecolor='black', linewidth=0.5, zorder=3)
    axR.set_xlim(-lim, lim); axR.set_ylim(-lim, lim)
    axR.set_xlabel('IN-SAMPLE HAC t')
    axR.set_ylabel('OUT-OF-SAMPLE HAC t')
    axR.set_title('shaded quadrants = SIGN INVERSION (the overfit signature)\n'
                  'circles = Grid A, squares = Grid B (real data)', fontsize=10)
    axR.legend(fontsize=8, loc='lower right')
    axR.grid(alpha=0.25)
    fig.tight_layout()
    FIGS.append('fig2_is_vs_oos')
    plt.show()
""")

co(r"""
# ===========================================================================
# FIGURE 3 -- GRID B LONG-HISTORY REGIME CHART, master regime-background style.
#
# Uses the master's OWN helpers, harvested verbatim:
#   regime_blocks  -- each band runs to the START of the next one, so a gap in
#                     the index never renders as a white stripe (a defect that
#                     hit this project twice).
#   shade_bands    -- ONE PolyCollection per label, full-height via the blended
#                     transform, added with autolim=False.
#   set_price_ylim -- EXPLICIT, TIGHT, EQUAL-MARGIN y-limits that are NOT
#                     zero-anchored (the squashed-price-panel defect).
# ===========================================================================
_b_daily = [r for r in B_RUNS if r['tf'] == 'daily' and not r['error']]
if _b_daily:
    _r = max(_b_daily, key=lambda r: r['n_bars'])      # longest history
    _lab = pd.Series(_r['labels'][0], index=_r['dates'])
    _px = _r['close_ser']
    _split = _r['dates'][_r['n_fit'] - 1]

    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    for ax, (t0_, t1_, ttl) in zip(axes, [
            (_r['dates'][0], _r['dates'][-1],
             f"{_r['market']} -- FULL HISTORY  "
             f"{_r['dates'][0]:%Y-%m-%d} -> {_r['dates'][-1]:%Y-%m-%d}"),
            (max(_r['dates'][0], _r['dates'][-1] - pd.Timedelta(days=5 * 365)),
             _r['dates'][-1], f"{_r['market']} -- last 5 years (zoom)")]):
        m = (_r['dates'] >= t0_) & (_r['dates'] <= t1_)
        sub_lab, sub_px = _lab[m], _px[m]
        shade_regimes(ax, sub_lab, alpha=0.38)
        ax.plot(sub_px.index, sub_px.values, color='black', lw=1.1, zorder=4)
        set_price_ylim(ax, sub_px, pad=0.03, tag=f"gridB {_r['run_id']} {ttl[:18]}")
        ax.set_xlim(sub_px.index[0], sub_px.index[-1])
        h = mark_split(ax, sub_lab.index, _split)
        regime_legend(ax, loc='upper left', extra=[h] if h else None)
        ax.set_title(ttl + '   [REAL FED DATA via GitHub mirror]', fontsize=11)
        ax.set_ylabel(f"{_r['sym']} per USD")
        ax.grid(alpha=0.18, zorder=0)
    _gs = ', '.join(k for k, v in _r['guards'].items() if not v[0]) or 'none'
    fig.suptitle(
        'GRID B -- RegDet V1.1 applied UNCHANGED to daily FX. '
        'Every knob is the Nifty-2h value.\n'
        f"THE FRAGMENTATION IS THE RESULT, NOT A RENDERING FAULT: "
        f"W = {_r['W']:.1f} switches / 100 bars (a switch every "
        f"{100 / max(_r['W'], 1e-9):.1f} bars), S = {_r['S']:.1f}%, "
        f"SIDEWAYS = {_r['occ']['SIDEWAYS']:.1f}%, guards failed: {_gs}.",
        fontsize=11, y=0.998)
    fig.tight_layout()
    FIGS.append('fig3_grid_b_regimes')
    plt.show()

    # the two defects this figure has hit before, asserted rather than eyeballed
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
else:
    print('Grid B daily unavailable; regime chart skipped.')
""")

co(r"""
# ===========================================================================
# FIGURE 4 -- GRID B CADENCE CHECK. daily vs weekly, the long-horizon end.
# ===========================================================================
_okb = [r for r in B_RUNS if not r['error']]
if _okb:
    pairs = sorted({r['market'] for r in _okb})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(pairs))
    wdt = 0.38
    for ax, (key, lbl) in zip(axes, [('occ', 'occupancy by label (%)'),
                                     ('t', 'HAC t of BULL - BEAR')]):
        if key == 'occ':
            bot_d = np.zeros(len(pairs)); bot_w = np.zeros(len(pairs))
            for lb in ORDER:
                vd = [next((r['occ'][lb] for r in _okb
                            if r['market'] == p and r['tf'] == 'daily'), 0) for p in pairs]
                vw = [next((r['occ'][lb] for r in _okb
                            if r['market'] == p and r['tf'] == 'weekly'), 0) for p in pairs]
                ax.bar(x - wdt / 2, vd, wdt, bottom=bot_d,
                       color=REGIME_COLORS[lb], edgecolor='white', lw=0.4)
                ax.bar(x + wdt / 2, vw, wdt, bottom=bot_w,
                       color=REGIME_COLORS[lb], edgecolor='white', lw=0.4,
                       hatch='//')
                if lb == 'SIDEWAYS':
                    # label the SIDEWAYS segment explicitly. A 50% guide line
                    # across a STACKED bar would NOT test SIDEWAYS -- SIDEWAYS
                    # is the middle segment, not the bottom one -- so the G4
                    # ceiling is written on the segment instead of drawn as a
                    # line that reads against the wrong quantity.
                    for xi, (b0, h0) in enumerate(zip(bot_d, vd)):
                        ax.text(xi - wdt / 2, b0 + h0 / 2, f'{h0:.0f}%',
                                ha='center', va='center', fontsize=8)
                    for xi, (b0, h0) in enumerate(zip(bot_w, vw)):
                        ax.text(xi + wdt / 2, b0 + h0 / 2, f'{h0:.0f}%',
                                ha='center', va='center', fontsize=8)
                bot_d = bot_d + np.array(vd); bot_w = bot_w + np.array(vw)
            ax.set_ylim(0, 118)
            ax.set_ylabel('occupancy %')
            ax.set_title('daily (solid) vs weekly (hatched)\n'
                         f'SIDEWAYS % is written on its own segment; '
                         f'G4 fails above {SIDEWAYS_MAX:.0f}%', fontsize=10)
            regime_legend(ax, loc='upper center')
        else:
            bw = 0.20
            for k, (cad, tag, col, hat) in enumerate([
                    ('daily', 'IS_T', '#4C72B0', ''),
                    ('daily', 'OOS_T', '#4C72B0', '//'),
                    ('weekly', 'IS_T', '#DD8452', ''),
                    ('weekly', 'OOS_T', '#DD8452', '//')]):
                v = [next((r[tag] for r in _okb
                           if r['market'] == p and r['tf'] == cad), np.nan)
                     for p in pairs]
                ax.bar(x + (k - 1.5) * bw, v, bw, color=col, hatch=hat,
                       edgecolor='black', lw=0.4,
                       label=f'{cad} {"IS" if tag == "IS_T" else "OOS"}')
            ax.axhline(0, color='black', lw=1)
            for yy in (-2, 2):
                ax.axhline(yy, color='grey', ls=':', lw=1)
            ax.set_ylabel('HAC t of (BULL - BEAR) fwd return')
            ax.set_title('IS (solid) vs OOS (hatched), daily vs weekly\n'
                         'dotted grey = |t| = 2', fontsize=10)
            ax.legend(fontsize=7, ncol=2)
        ax.set_xticks(x); ax.set_xticklabels(pairs)
    fig.suptitle('GRID B CADENCE CHECK -- daily vs weekly   '
                 '[REAL FED DATA via GitHub mirror]', fontsize=11)
    fig.tight_layout()
    FIGS.append('fig4_grid_b_cadence')
    plt.show()
print(f'figures produced: {len(FIGS)} -> {FIGS}')
""")

# ===========================================================================
md(r"""
## 9. Grading the pre-registered predictions

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
_okA = [r for r in A_RUNS if not r['error']]
_okB = [r for r in B_RUNS if not r['error']]

print('=' * 116)
print('GRADING THE PRE-REGISTERED PREDICTIONS')
if GRID_A_SYNTH:
    print('!' * 116)
    print('!! GRID A IS SYNTHETIC IN THIS RUN. P1, P2 and P4 are graded on GBM,')
    print('!! so they are MACHINERY EVIDENCE ONLY and are stamped SYNTHETIC-VOID.')
    print('!! They say nothing about real S&P/NASDAQ/FX behaviour. P3 is ALSO')
    print('!! graded on GRID B, which is REAL DATA, and that grade IS evidence.')
    print('!' * 116)
print('=' * 116)

# ---- P1 -------------------------------------------------------------------
_p1n = sum(1 for r in _okA if r['guards_ok'])
_p1 = _p1n >= 8
GRADES['P1'] = _p1
print(f"\nP1  {PREDICTIONS['P1'][0]}")
print(f"    {_p1n} of 15 Grid-A runs pass ALL FOUR guards "
      f"({len(A_RUNS) - len(_okA)} errored).")
for r in _okA:
    if not r['guards_ok']:
        bad = [f'{k}({v[1]})' for k, v in r['guards'].items() if not v[0]]
        print(f"      FAIL {r['run_id']:<20} " + '; '.join(bad))
print(f'    -> P1 {verdict(_p1)}'
      + ('   [SYNTHETIC-VOID as market evidence]' if GRID_A_SYNTH else ''))

# ---- P2 -------------------------------------------------------------------
_sw = [r['occ']['SIDEWAYS'] for r in _okA]
_p2n = sum(1 for v in _sw if P2_LO <= v <= P2_HI)
_p2 = _p2n > len(_okA) / 2 if _okA else False
GRADES['P2'] = _p2
print(f"\nP2  {PREDICTIONS['P2'][0]}")
if _sw:
    print(f'    SIDEWAYS across Grid A: min {min(_sw):.1f}%  median '
          f'{np.median(_sw):.1f}%  max {max(_sw):.1f}%  '
          f'(spread {max(_sw) - min(_sw):.1f}pp)')
    print(f'    {_p2n} of {len(_okA)} runs land in [{P2_LO:.0f}%, {P2_HI:.0f}%].')
    _out = [(r['run_id'], r['occ']['SIDEWAYS']) for r in _okA
            if not (P2_LO <= r['occ']['SIDEWAYS'] <= P2_HI)]
    for rid, v in _out:
        print(f'      OUTSIDE BAND {rid:<20} {v:.1f}%')
print(f'    -> P2 {verdict(_p2)}'
      + ('   [SYNTHETIC-VOID as market evidence]' if GRID_A_SYNTH else ''))

# ---- P3 -- graded SEPARATELY on each grid ---------------------------------
print(f"\nP3  {PREDICTIONS['P3'][0]}")
_p3parts = {}
for gname, rows, real in (('GRID A', _okA, not GRID_A_SYNTH),
                          ('GRID B', _okB, True)):
    if not rows:
        print(f'    {gname}: no completed run.')
        continue
    ninv = sum(1 for r in rows if r['inverted'])
    ok = ninv <= len(rows) / 2
    _p3parts[gname] = ok
    _deg = [abs(r['OOS_T']) - abs(r['IS_T']) for r in rows
            if np.isfinite(r['OOS_T']) and np.isfinite(r['IS_T'])]
    print(f'    {gname}: {ninv} of {len(rows)} runs INVERTED sign out of sample. '
          f'mean |t| change {np.mean(_deg):+.2f}.'
          + ('' if real else '   [SYNTHETIC-VOID as market evidence]'))
    for r in rows:
        if r['inverted']:
            print(f"      INVERTED {r['run_id']:<20} IS d {r['IS_D']:+.4f} "
                  f"(t {r['IS_T']:+.2f})  ->  OOS d {r['OOS_D']:+.4f} "
                  f"(t {r['OOS_T']:+.2f})")
_p3 = all(_p3parts.values()) if _p3parts else False
GRADES['P3'] = _p3
print(f'    -> P3 {verdict(_p3)}  (requires a majority NON-inverted on '
      f'EVERY graded grid)')
if 'GRID B' in _p3parts:
    print(f"    -> P3 on GRID B ALONE (REAL DATA): "
          f"{verdict(_p3parts['GRID B'])}")

# ---- P4 -------------------------------------------------------------------
print(f"\nP4  {PREDICTIONS['P4'][0]}")
_fx = [r['occ']['SIDEWAYS'] for r in _okA if r['cls'] == 'FX']
_eq = [r['occ']['SIDEWAYS'] for r in _okA if r['cls'] == 'EQ']
if _fx and _eq:
    gap = abs(np.mean(_fx) - np.mean(_eq))
    within = max(np.ptp(_fx), np.ptp(_eq))
    _p4 = gap > within
    print(f'    FX  SIDEWAYS mean {np.mean(_fx):.1f}%  (n={len(_fx)}, '
          f'spread {np.ptp(_fx):.1f}pp)')
    print(f'    EQ  SIDEWAYS mean {np.mean(_eq):.1f}%  (n={len(_eq)}, '
          f'spread {np.ptp(_eq):.1f}pp)')
    print(f'    between-group gap {gap:.1f}pp vs largest within-group spread '
          f'{within:.1f}pp')
    print('    NOTE: REFUTED here means the two asset classes behaved ALIKE. '
          'That is')
    print('    informative and is NOT a failure of the architecture, exactly as '
          'the')
    print('    protocol says. Nothing is re-tuned either way.')
else:
    _p4 = False
    print('    one of the two groups has no completed run -- UNMEASURABLE.')
GRADES['P4'] = _p4
print(f'    -> P4 {verdict(_p4)}'
      + ('   [SYNTHETIC-VOID as market evidence]' if GRID_A_SYNTH else ''))

print('\n' + '=' * 116)
print('SUMMARY:  ' + '   '.join(f'{k} {verdict(v)}' for k, v in GRADES.items()))
if GRID_A_SYNTH:
    print('GRID A GRADES ARE SYNTHETIC-VOID. The only decision-grade grade in '
          'this run is P3 on Grid B.')
print('=' * 116)
""")

co(r"""
# ===========================================================================
# THE HONEST READING. Written as a template that fills itself from the numbers,
# so it cannot drift away from what was actually measured.
# ===========================================================================
print('=' * 116)
print('WHAT THIS RUN DOES AND DOES NOT ESTABLISH')
print('=' * 116)
if GRID_A_SYNTH:
    print('* GRID A: yfinance is firewalled here, so all 15 runs used synthetic')
    print('  GBM. Grid A in THIS notebook is a MACHINERY PROOF: it shows the')
    print('  architecture RUNS unchanged at 1h/2h/4h across five bar-count and')
    print('  session-length regimes, that the day-denominated windows rebuild')
    print('  correctly, and that the guards and the IS/OOS split behave. It')
    print('  establishes NOTHING about real S&P, NASDAQ or FX regimes.')
    _bigis = max((r['IS_T'] for r in _okA if np.isfinite(r['IS_T'])), default=np.nan)
    if np.isfinite(_bigis):
        print()
        print('  ONE THING GRID A DOES ESTABLISH, and it is worth stating plainly:')
        print(f'  the synthetic series are GEOMETRIC BROWNIAN MOTION. There is NO')
        print(f'  regime structure and NO predictable direction in them at all. The')
        print(f'  detector nevertheless produced an IN-SAMPLE BULL-BEAR contrast as')
        print(f'  large as HAC t = {_bigis:+.2f} on that data. An in-sample t of that')
        print('  size is therefore NOT evidence of skill even on real data -- it is')
        print('  what this architecture returns on noise it was fitted to. Only the')
        print('  OUT-OF-SAMPLE column carries information, which is exactly why the')
        print('  two spans are never pooled.')
else:
    print('* GRID A: real yfinance data. The table above is decision-grade for')
    print('  those markets over the ~730d 60m window yfinance serves.')
print('* GRID B: REAL Federal Reserve H.10 data via the GitHub mirror, verified')
print('  live against Brexit, the Bretton Woods collapse and the GFC. The')
print('  Grid-B rows ARE decision-grade for daily/weekly FX.')
print()
_okB2 = [r for r in B_RUNS if not r['error']]
if _okB2:
    print('GRID B, in plain words:')
    for r in _okB2:
        sw = r['occ']['SIDEWAYS']
        gp = 'PASS' if r['guards_ok'] else ('FAIL ' + ','.join(
            k for k, v in r['guards'].items() if not v[0]))
        print(f"  {r['run_id']:<18} {r['n_bars']:>6} bars   SIDEWAYS {sw:5.1f}%   "
              f"S {r['S']:5.1f}%   W {r['W']:5.2f}   guards {gp:<14} "
              f"IS t {r['IS_T']:+5.2f} -> OOS t {r['OOS_T']:+5.2f}"
              + ('   INVERTED' if r['inverted'] else ''))
print()
print('WHAT IS NOT CLAIMED: no return, no Sharpe, no tradeability. This')
print('notebook measures a DETECTOR. Nothing here was re-tuned to make a market')
print('pass, no threshold was moved, and no run was dropped for looking bad --')
print('every run in both grids appears in the tables above, errors included.')
print('=' * 116)
""")

# ===========================================================================
md(r"""
## 10. Config hand-off

A pasteable block recording **exactly** what was held fixed for every run in
both grids, and the short list of things that were allowed to re-derive.
""")

co(r"""
# ===========================================================================
# CONFIG HAND-OFF -- pasteable Python. Generated from the LIVE globals, so it
# cannot drift from what actually ran.
# ===========================================================================
assert_knobs_unchanged('hand-off')

_ho = f'''# ---------------------------------------------------------------------------
# RegDet V1.1 GENERALIZATION RUN -- EXACTLY WHAT WAS HELD FIXED
# generated from the live notebook globals after every run completed
# grid A source : {'SYNTHETIC (yfinance firewalled)' if GRID_A_SYNTH else 'REAL yfinance 60m period=730d'}
# grid B source : REAL FED DATA via GitHub mirror ({GRID_B_URL})
# ---------------------------------------------------------------------------
REGDET_GENERALIZATION_FIXED = dict(
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
    # --- harness -------------------------------------------------------
    TRAIN_FRACTION         = {TRAIN_FRACTION},
    N_FOLDS                = {N_FOLDS},
    HMM_ITER               = {HMM_ITER},
    # --- guards / metrics ----------------------------------------------
    ZZ_PCT                 = {ZZ_PCT},
    G1_OCC_BAND_PCT        = ({OCC_MIN_PCT}, {OCC_MAX_PCT}),
    G3_W_MAX               = {W_GUARD_MAX},
    G4_SIDEWAYS_MAX        = {SIDEWAYS_MAX},
)

# ALLOWED to re-derive per run (data-dependent by design, protocol section
# "WHAT MAY RE-DERIVE PER MARKET"):
#   * StandardScaler, fit on that run's own LEADING {TRAIN_FRACTION:.0%} window
#   * trend_z's mu/sd baseline, frozen on the same window by the engine
#   * the H-gate band, derived from fit-window occupancy (H_TARGET_RATE)
#   * BASE_WIN, rebuilt from BARS_PER_DAY by the master's OWN formula with the
#     SAME day counts -- this keeps the HORIZON fixed, it does not tune a knob
# NOTHING ELSE. The notebook asserts this after every single run.

REGDET_GENERALIZATION_BPD = {{
'''
for r in RUNS:
    _ho += (f"    {r['run_id']!r:<24}: dict(assumed={r['bpd_assumed']}, "
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
print(f'  runs                : {len(RUNS)}  '
      f'({len(A_RUNS)} Grid A + {len(B_RUNS)} Grid B)')
print(f'  completed / errored : {len([r for r in RUNS if not r["error"]])} / '
      f'{len([r for r in RUNS if r["error"]])}')
print(f'  HMM fits            : {FIT_COUNT}   '
      f'({R_SEED_SETS} disjoint seed sets x K={ENSEMBLE_K} per run)')
print(f'  label_bars calls    : {LABEL_CALLS}')
print(f'  ZigZag computations : {ZZ_CALLS}  (all in the EVAL phase)')
print(f'  figures             : {len(FIGS)}  {FIGS}')
print(f'  wall clock          : {_el:.1f}s')
print()
print('  NOTHING WAS CUT. R=4 seed sets, 5 markets, 3 timeframes, 3 FX pairs,')
print('  2 cadences -- the full grid the protocol specifies.')
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
