"""
Builds daily_fit.ipynb -- a standalone notebook testing the "learn slow, apply
fast" architecture declared in DAILY_FIT_PROTOCOL.md:

    fit the DIRECTION model on DAILY bars with long history, project it causally
    onto 2h bars, and leave the INTENSITY (H vs L) axis on 2h, untouched.

Three arms, ONE KNOB AT A TIME:
    1. BASELINE          - direction fitted on 2h, ~2y, the existing 9 features.
    2. DAILY-9FEAT       - direction fitted on DAILY bars, long history, the SAME
                           9 features (same window lengths in bars).
    3. DAILY-3FEAT       - as arm 2 plus the decollinearized feature set
                           (one momentum, one volatility, one dispersion).

S / L / W and guards G1-G4 are REUSED VERBATIM from build_stability_lag.py so the
numbers are comparable across the two notebooks. The engine is INLINED VERBATIM
out of build_master_notebook_v2.py, same practice as build_stability_lag.py and
build_hmm_share_charts.py. Nothing here imports or modifies the master, its
generator, regime_scorecard.py, build_hmm_share_charts.py or build_stability_lag.py.

THE CRITICAL RISK this notebook is built around: the daily -> 2h projection must
be strictly causal. Every 2h bar's direction state comes from the last FULLY
CLOSED daily bar (strictly earlier session date). Enforced by a per-bar assert AND
by a truncation probe that re-runs the whole pipeline on a cut series and demands
bit-identical labels.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(
    os.environ.get('REGDET_OUT_DIR',
                   os.path.join(os.path.dirname(HERE), 'notebooks')),
    'daily_fit.ipynb')
MASTER_GEN = f'{HERE}/build_master_notebook_v2.py'

# ---------------------------------------------------------------------------
# Harvest the engine cells from the master generator, VERBATIM, by executing it
# in a private namespace with its output path redirected to a throwaway file.
# ---------------------------------------------------------------------------
_gsrc = open(MASTER_GEN).read()
assert "regdet_v11_master.ipynb'" in _gsrc
_gsrc = _gsrc.replace("regdet_v11_master.ipynb'", "_daily_fit_engine_harvest.ipynb'", 1)
_ns = {'__name__': '_master_gen_harvest', '__file__': MASTER_GEN}
exec(compile(_gsrc, MASTER_GEN, 'exec'), _ns)
_mcells = _ns['cells']


def _src(i):
    return ''.join(_mcells[i]['source'])


IDX_CONST, IDX_LOAD, IDX_ENGINE, IDX_PLOT = 3, 5, 7, 9
SRC_CONST, SRC_LOAD = _src(IDX_CONST), _src(IDX_LOAD)
SRC_ENGINE, SRC_PLOT = _src(IDX_ENGINE), _src(IDX_PLOT)

# Cell identity asserted by CONTENT, never trusted by number.
assert 'REGIME_COLORS' in SRC_CONST and 'BAR_DIR_WEIGHT   = 0.0' in SRC_CONST
assert 'import numpy as np' in SRC_CONST and 'CONFIGS = [' in SRC_CONST
assert 'BARS_PER_DAY' in SRC_CONST and 'BASE_WIN = dict(' in SRC_CONST
assert 'def load_2h' in SRC_LOAD and 'def _synth' in SRC_LOAD
assert 'def build_features' in SRC_ENGINE and 'def label_bars' in SRC_ENGINE \
    and 'def fit_hmm_ensemble' in SRC_ENGINE and 'def confirm_delay' in SRC_ENGINE \
    and 'def n_params' in SRC_ENGINE \
    and 'def ensemble_direction_masses_by_mode' in SRC_ENGINE
assert 'def shade_bands' in SRC_PLOT and 'def set_price_ylim' in SRC_PLOT

PROVENANCE = (
    f'build_master_notebook_v2.py cell {IDX_CONST} (constants/imports/CONFIGS), '
    f'cell {IDX_LOAD} (load_2h/_synth), cell {IDX_ENGINE} (feature + labeling engine), '
    f'cell {IDX_PLOT} (regime-background plot helpers)'
)
try:
    os.remove(f'{HERE}/_daily_fit_engine_harvest.ipynb')
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
# RegDet V1.1 — DAILY-FIT DIRECTION ("learn slow, apply fast")

Implements `DAILY_FIT_PROTOCOL.md` exactly. That protocol, and the four
predictions in section 2, were frozen before any number here existed.

**The change.** Fit the DIRECTION model on **daily** bars with the longest
available history, project the resulting direction state **causally** onto the 2h
bars, and leave the **intensity** (H vs L) axis on 2h, completely untouched.
Direction is slow, weak and needs many regimes to estimate; intensity is fast,
strong and needs fine resolution. Fit each where its information lives.

**Three arms, one knob at a time.**

| arm | direction fitted on | features |
|---|---|---|
| 1 BASELINE | 2h, ~2y | the existing 9 |
| 2 DAILY-9FEAT | daily, long history | the same 9 |
| 3 DAILY-3FEAT | daily, long history | decollinearized: one momentum, one vol, one dispersion |

Arm 2 isolates the **data** change. Arm 3 adds the **feature** change.

**The one thing that can silently invalidate all of this** is the daily→2h
projection. At a 2h bar on calendar day *d*, the direction state must come from
the last **fully closed** daily bar — day *d−1* or earlier. Section 4 enforces
that with a per-bar assert *and* a truncation probe that re-runs the entire
pipeline on a cut series and demands bit-identical labels. A failure there is a
hard stop, not something to work around.

`S`, `L`, `W` and guards `G1`–`G4` are **reused verbatim** from
`STABILITY_LAG_PROTOCOL.md` / `stability_lag.ipynb`, so rows here are directly
comparable with rows there. `BAR_DIR_WEIGHT = 0.0` throughout, asserted.
""")

co(r"""
%pip install -q hmmlearn yfinance
""")

# --- engine ----------------------------------------------------------------
md(f"""
## 1. Engine, inlined verbatim

Inlined out of `build_master_notebook_v2.py` rather than imported, because a
standalone `.ipynb` on Kaggle cannot import a local `.py` next to it. Source:
`{PROVENANCE}`. Not one line is modified.
""")

co(BANNER
   + f'# CONSTANTS + IMPORTS -- INLINED VERBATIM from {MASTER_GEN.split("/")[-1]}\n'
   + f'# (master notebook code cell {IDX_CONST}). Unmodified.\n'
   + BANNER + SRC_CONST)

co(BANNER
   + f'# DATA LOADER -- INLINED VERBATIM (master code cell {IDX_LOAD}).\n'
   + '# yfinance 60m -> 2h resample, with a synthetic fallback. Unmodified.\n'
   + BANNER + SRC_LOAD)

co(BANNER
   + f'# FEATURE + LABELING ENGINE -- INLINED VERBATIM (master code cell {IDX_ENGINE}).\n'
   + BANNER + SRC_ENGINE)

co(BANNER
   + f'# PLOT HELPERS -- INLINED VERBATIM (master code cell {IDX_PLOT}).\n'
   + BANNER + SRC_PLOT)

# ===========================================================================
md(r"""
## 2. The four PRE-REGISTERED predictions

Printed **before** any number in this notebook exists, together with the exact
arithmetic that will grade them. Section 8 grades itself against this block and
prints, in plain words, whether the small-sample diagnosis is CONFIRMED or
REFUTED. **If P1 and P2 both fail, the diagnosis is WRONG and the notebook says
so.** No prediction is reinterpreted after its number is seen.
""")

co(r"""
# ===========================================================================
# PRE-REGISTERED PREDICTIONS + THE GRADING ARITHMETIC. Declared here, first,
# before a single fit is run.
# ===========================================================================
import itertools, time, contextlib
import matplotlib.dates as mdates

T_START = time.time()

P1_S_MIN      = 95.0   # P1: S must exceed this
BIMODAL_SPREAD = 10.0  # a pairwise matrix is BIMODAL if (max-min) >= this pp ...
BIMODAL_GAP    = 5.0   # ... AND the largest gap between sorted pairwise values >= this
BASIN_LOW      = 75.0  # P2: a pairwise value below this is an ACROSS-BASIN pair

# These three thresholds are round numbers taken from the ALREADY-PUBLISHED
# baseline description in DAILY_FIT_PROTOCOL.md (S = 88.29%, spread 14.94pp,
# ~98% within-basin vs ~63% across-basin). They are NOT read off this run's
# numbers, and they are NOT moved afterwards.

PREDICTIONS = [
 ('P1', 'S rises to > %.0f%% AND the pairwise matrix stops being bimodal' % P1_S_MIN),
 ('P2', 'the two-basin structure (~98% within / ~63% across) disappears or weakens'),
 ('P3', 'L does NOT improve much and may worsen -- EXPECTED, not a failure'),
 ('P4', 'occupancy stabilises across seed sets'),
]
print('=' * 96)
print('PRE-REGISTERED PREDICTIONS  (frozen in DAILY_FIT_PROTOCOL.md before any number existed)')
print('=' * 96)
for k, t in PREDICTIONS:
    print(f'  {k}: {t}')
print('-' * 96)
print('GRADING ARITHMETIC, declared now:')
print(f'  BIMODAL(matrix) := (max-min of the 6 pairwise values) >= {BIMODAL_SPREAD:.1f} pp')
print(f'                     AND largest gap between sorted values >= {BIMODAL_GAP:.1f} pp')
print(f'  P1 PASS := S > {P1_S_MIN:.1f}%  AND  not BIMODAL')
print(f'  P2 PASS := count of pairwise values < {BASIN_LOW:.1f}% is STRICTLY LOWER than the')
print('             baseline count (0 vs 0 is reported as UNTESTABLE, not as a pass)')
print('  P3      := reported as consistent/inconsistent. A WORSE L is EXPECTED and is')
print('             not counted against the architecture. Only a large IMPROVEMENT in L')
print('             would be inconsistent with P3.')
print('  P4 PASS := max across-seed-set occupancy spread (pp, over the 5 labels) is')
print("             LOWER than the baseline's")
print('-' * 96)
print("ARM 3's FEATURE SET, declared as a RULE before the correlation matrix is seen:")
print('  momentum slot   = TREND_FEATURE (the engine\'s own pre-existing choice)')
print('  volatility slot = vol_2h        (the engine\'s own primary volatility feature)')
print('  dispersion slot = whichever of {dist_ma, drawdown, vix_chg} has the LOWEST')
print('                    MAXIMUM |r| against the other two, on the DAILY matrix.')
print('  This is LABEL-BLIND: no label, no metric and no forward return enters it. It is')
print('  a rule, not a pick -- "decollinearized" has to mean decollinearized, and section')
print('  3 reports whether the chosen triple actually achieves it.')
print('-' * 96)
print('IF P1 AND P2 BOTH FAIL -> the small-sample diagnosis is WRONG. Section 8 prints')
print('that verdict without softening it.')
print('=' * 96)

assert BAR_DIR_WEIGHT == 0.0, 'BAR_DIR_WEIGHT must be 0.0 for this notebook'
W_FIXED = 0.0

# Metric parameters, reused VERBATIM from STABILITY_LAG_PROTOCOL.md.
R_SEED_SETS   = 4       # R = 4 DISJOINT seed sets. NOT reducible -- S is meaningless without it.
ZZ_PCT        = 2.0     # ZigZag reversal threshold for L
MACRO_ZZ_PCT  = 10.0    # ZigZag threshold used ONLY to COUNT macro regimes in a fit window
W_GUARD_MAX   = 25.0    # G3
OCC_MIN_PCT   = 3.0     # G1 / G2
OCC_MAX_PCT   = 50.0    # G1
SIDEWAYS_MAX  = 50.0    # G4
COLLINEAR_HI  = 0.70    # declared NOW: |r| >= this between two features is "high"

print(f'\nR={R_SEED_SETS} disjoint seed sets   L-ZigZag={ZZ_PCT}%   '
      f'macro-regime ZigZag={MACRO_ZZ_PCT}%   BAR_DIR_WEIGHT={W_FIXED} (asserted)')
print(f'guards: G1 occ in [{OCC_MIN_PCT},{OCC_MAX_PCT}]%  G2 no collapse  '
      f'G3 W<={W_GUARD_MAX}  G4 SIDEWAYS<={SIDEWAYS_MAX}%')
print(f'collinearity is called HIGH at |r| >= {COLLINEAR_HI} -- declared before the matrix '
      'is printed')
""")

# ===========================================================================
md(r"""
## 3. Data — daily ^NSEI (long history) and the 2h series

`^NSEI` daily reaches ~2007 on yfinance; India VIX starts later. **VIX is not
forward-filled across years in which it does not exist.** The fit window is
restricted to the span where *both* series exist, and the cost of that
restriction is printed. VIX is kept rather than dropped so that arm 2 really is
"the same nine features" — dropping it would confound the data change with a
feature change, which the protocol forbids.

`N_REGIMES_OBSERVED` — distinct macro price legs (a 10% ZigZag) inside each arm's
**fit window** — is the headline number. The entire argument rests on the daily
arms seeing materially more regimes than the ~6–10 the 2h fit sees.
""")

co(r"""
# ===========================================================================
# DAILY DATA. Real yfinance if reachable; otherwise a LONG synthetic daily
# series from which the 2h series is REBUILT so the two are mutually consistent
# (an inconsistent pair would make the causality asserts meaningless theatre).
# ===========================================================================
def load_daily():
    '''(daily_close, daily_vix, is_synth). Real ^NSEI period=max if reachable.'''
    try:
        import yfinance as yf

        def d1(tk):
            raw = yf.download(tk, interval='1d', period='max',
                              auto_adjust=True, progress=False)
            if raw is None or len(raw) == 0:
                return None
            c = raw['Close'].squeeze().dropna()
            idx = pd.to_datetime(c.index)
            c.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
            return c[~c.index.duplicated(keep='last')]

        nd = d1('^NSEI')
        if nd is None or len(nd) < 500:
            raise ValueError('too few daily bars')
        vd = d1('^INDIAVIX')
        if vd is None or len(vd) < 200:
            raise ValueError('no India VIX daily history')
        nd.name, vd.name = 'nifty_d', 'vix_d'
        print(f'yfinance daily: ^NSEI {len(nd)} bars {nd.index[0]:%Y-%m-%d} -> '
              f'{nd.index[-1]:%Y-%m-%d}')
        print(f'yfinance daily: ^INDIAVIX {len(vd)} bars {vd.index[0]:%Y-%m-%d} -> '
              f'{vd.index[-1]:%Y-%m-%d}')
        return nd, vd, False
    except Exception as e:
        print(f'daily yfinance unavailable ({e}); falling back to SYNTHETIC daily.')
        return _synth_daily()


def _synth_daily(years=18, seed=20260730):
    '''Regime-blocked GBM daily series with MANY macro regimes, plus an OU VIX
    that deliberately STARTS LATE, so the short-VIX-history handling is
    exercised rather than assumed.'''
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(years * 252))
    n = len(days)
    # 16 macro regimes, alternating in character, so N_REGIMES_OBSERVED on the
    # synthetic daily fit window is realistically LARGE.
    blocks = [(0.00050, 0.0075), (-0.00090, 0.0140), (0.00035, 0.0060),
              (0.00080, 0.0070), (-0.00160, 0.0210), (0.00060, 0.0090),
              (0.00020, 0.0055), (-0.00070, 0.0130), (0.00095, 0.0080),
              (0.00010, 0.0065), (-0.00120, 0.0175), (0.00070, 0.0085),
              (0.00040, 0.0060), (-0.00050, 0.0115), (0.00085, 0.0075),
              (0.00030, 0.0068)]
    edges = np.linspace(0, n, len(blocks) + 1).astype(int)
    lvl, out = 2500.0, []
    for bi, (mu, sg) in enumerate(blocks):
        for _ in range(edges[bi + 1] - edges[bi]):
            lvl *= np.exp(rng.normal(mu, sg))
            out.append(lvl)
    nd = pd.Series(out, index=days, name='nifty_d')
    # VIX starts LATE ON PURPOSE (2 years in), mimicking India VIX vs ^NSEI.
    vstart = int(2 * 252)
    vv, p = [], 15.0
    for i in range(vstart, n):
        tgt = 12.0 + 900.0 * blocks[min(int(i / n * len(blocks)), len(blocks) - 1)][1]
        p = max(p + 0.06 * (tgt - p) + rng.normal(0, 1.2), 8.0)
        vv.append(p)
    vd = pd.Series(vv, index=days[vstart:], name='vix_d')
    return nd, vd, True


def rebuild_2h_from_daily(nd, vd, n_days=730, seed=7, sigma_intra=0.35):
    '''Expand the LAST n_days daily bars into 3 x 2h bars/day by a Brownian
    bridge whose LAST bar of day d equals the daily close of day d.

    Used only on the synthetic path. It exists so the daily and 2h series in the
    synthetic run describe the SAME market -- otherwise the daily->2h projection
    would be projecting one random walk onto an unrelated one and every causality
    assert would pass while measuring nothing.
    '''
    rng = np.random.default_rng(seed)
    d = nd.iloc[-(n_days + 1):]
    lg = np.log(d.values)
    idx, val = [], []
    for i in range(1, len(d)):
        lp, lc = lg[i - 1], lg[i]
        step = (lc - lp)
        for k, h in enumerate((10, 12, 14), start=1):
            base = lp + step * k / 3.0
            noise = 0.0 if k == 3 else rng.normal(
                0, abs(step) * sigma_intra + 1e-4) * np.sqrt(k * (3 - k) / 3.0)
            idx.append(d.index[i] + pd.Timedelta(hours=h))
            val.append(float(np.exp(base + noise)))
    px = pd.Series(val, index=pd.DatetimeIndex(idx), name='nifty')
    vx = (vd.reindex(d.index).ffill().bfill()
            .reindex(pd.DatetimeIndex(idx).normalize()).values)
    vx = pd.Series(vx * (1.0 + rng.normal(0, 0.01, len(idx))),
                   index=px.index, name='vix')
    return px, vx


NIFTY_D, VIX_D, DAILY_SYNTH = load_daily()

if DAILY_SYNTH:
    # Consistency decree: if the daily series is synthetic, the 2h series is
    # REBUILT from it. A real 2h series paired with a synthetic daily one would
    # make every projection assert vacuous.
    nifty, vix = rebuild_2h_from_daily(NIFTY_D, VIX_D)
    TAC_SYNTH = True
    print('\nSYNTHETIC daily -> the 2h series has been REBUILT from it so that the two '
          'describe the same market.')

if TAC_SYNTH or DAILY_SYNTH:
    print('#' * 100)
    for _ in range(3):
        print('#   SYNTHETIC - ILLUSTRATIVE ONLY   ' * 2)
    print('#' * 100)
    print('#  yfinance is firewalled in this sandbox. Every number and chart below comes')
    print('#  from a synthetic series. This run is a PLUMBING TEST of the daily-fit')
    print('#  machinery and the causality proofs ONLY. It says NOTHING about real Nifty')
    print('#  regimes, real fit stability, or whether the predictions hold. The REAL')
    print('#  verdict requires the Kaggle run, where yfinance works.')
    print('#' * 100)
else:
    print('\n' + '=' * 100)
    print('REAL yfinance data on BOTH cadences. Verdicts below are decision-grade.')
    print('=' * 100)

# ---- THE VIX SPAN PROBLEM, handled explicitly -----------------------------
_d0_px, _d0_vx = NIFTY_D.index[0], VIX_D.index[0]
DAILY_START = max(_d0_px, _d0_vx)
NIFTY_D = NIFTY_D[NIFTY_D.index >= DAILY_START]
VIX_D = VIX_D.reindex(NIFTY_D.index).ffill()
assert VIX_D.notna().all(), 'VIX has holes INSIDE its own span -- investigate before using'
_lost_yrs = (DAILY_START - _d0_px).days / 365.25
print(f'\nVIX SPAN HANDLING (explicit, per protocol)')
print(f'  ^NSEI daily starts     : {_d0_px:%Y-%m-%d}')
print(f'  India VIX daily starts : {_d0_vx:%Y-%m-%d}')
print(f'  DECISION: RESTRICT the daily window to {DAILY_START:%Y-%m-%d}, the first date '
      f'both series exist.')
print(f'  WHY: keeping vix_chg makes arm 2 genuinely "the SAME nine features", so the')
print(f'       data change is not confounded with a feature change. VIX is NOT')
print(f'       back-filled across years it does not exist -- those years are DROPPED.')
print(f'  COST: {_lost_yrs:.1f} years of price history discarded '
      f'({len(NIFTY_D)} daily bars retained).')
print(f'  (ffill is used only INSIDE the common span, for NSE sessions with no VIX '
      f'print; that is a same-day hole, not a missing era.)')
""")

# ===========================================================================
md(r"""
## 4. The daily→2h projection — the causality proof

Two independent proofs, both runtime asserts:

1. **Per-bar assert.** Every 2h bar's source daily timestamp is *strictly* earlier
   than that bar's own session date.
2. **Truncation probe.** The whole pipeline — features, scaler, HMM fit,
   projection, labelling — is re-run on a series cut at `T` (with the daily series
   cut to sessions strictly before `T`'s session, i.e. exactly what a live engine
   would have had) and every label at `t ≤ T` must come out **bit-identical**.
   The assert alone is not the test; this is.

The daily fit is **anchored**: it is fit on the leading `TRAIN_FRACTION` of the
daily history and applied forward. It is never fit on all data and applied
backwards.
""")

co(r"""
# ===========================================================================
# FEATURES ON BOTH CADENCES.
#
# The DAILY features use `build_features` UNMODIFIED, with BASE_WIN unchanged.
# That means the window lengths are IDENTICAL IN BARS on both cadences
# (1 / 3 / 9 / 15 / 10 / 5 / 20 / 20), which is the cleanest reading of "the same
# features expressed as daily equivalents": same function, same geometry, same
# number of observations behind every feature.
#
# The consequence, stated plainly rather than buried: on daily bars those windows
# span 3x more CALENDAR time (mom_3d is a 9-DAY momentum, not a 9-BAR one). That
# is inherent to "learn slow" -- it is the change, not a side effect of it.
#
# The alternative -- matching calendar horizons -- would make mom_1d IDENTICAL to
# ret_2h on daily bars (both 1-day returns), a perfectly collinear duplicate
# column. That is a worse experiment, so it was not used.
# ===========================================================================
FEAT_2H = build_features(nifty, vix, LOOKBACK_SCALE)
FEAT_D = build_features(NIFTY_D, VIX_D, LOOKBACK_SCALE)

FEATURES_9 = list(FEATURE_COLS)
# Arm 3's slots, per the rule declared in section 2. The dispersion/context slot
# is RESOLVED in the next cell, from the daily correlation matrix only.
F3_MOM, F3_VOL = TREND_FEATURE, 'vol_2h'
F3_DISP_CANDIDATES = ['dist_ma', 'drawdown', 'vix_chg']

BASE_CFG = [c for c in CONFIGS if c['name'] == ADOPTED_CONFIG_NAME][0]
BASE_N, BASE_COV = BASE_CFG['N'], BASE_CFG['cov']
assert list(BASE_CFG['features']) == FEATURES_9, \
    'the adopted config is expected to carry all 9 features'

print(f'2h feature bars   : {len(FEAT_2H)}   {FEAT_2H.index[0]:%Y-%m-%d} -> '
      f'{FEAT_2H.index[-1]:%Y-%m-%d}')
print(f'daily feature bars: {len(FEAT_D)}   {FEAT_D.index[0]:%Y-%m-%d} -> '
      f'{FEAT_D.index[-1]:%Y-%m-%d}')
print(f'window lengths IN BARS (identical on both cadences): {BASE_WIN}')
print(f'model            : N={BASE_N} states, cov={BASE_COV!r} (the adopted config, '
      f'unchanged in every arm)')
print(f'params  9 feats  : {int(n_params(BASE_N, 9, BASE_COV))}   '
      f'3 feats: {int(n_params(BASE_N, 3, BASE_COV))}')
""")

co(r"""
# ===========================================================================
# THE FEATURE CORRELATION MATRICES -- the collinearity claim is CHECKED here,
# not assumed. If the correlations are LOW, the "nested momenta create likelihood
# ridges" explanation for the bimodality is WRONG, and this cell says so.
# ===========================================================================
def corr_report(df, feats, tag):
    C = df[feats].corr()
    print(f'\n{tag}   ({len(df)} bars, {len(feats)} features)')
    print(C.round(2).to_string())
    A = C.abs().values.copy()
    np.fill_diagonal(A, 0.0)
    _cn = np.linalg.cond(C.values)
    print(f'  max |r| off-diagonal = {A.max():.3f}   '
          f'pairs with |r| >= {COLLINEAR_HI}: '
          f'{int((np.triu(A, 1) >= COLLINEAR_HI).sum())} of '
          f'{len(feats) * (len(feats) - 1) // 2}')
    print(f'  condition number of the correlation matrix = {_cn:,.1f}   '
          '(a likelihood-ridge proxy: large = flat directions in the objective)')
    return C, float(A.max()), float(_cn)


MOM = ['mom_1d', 'mom_3d', 'mom_5d']
C2H, MX2H, CN2H = corr_report(FEAT_2H, FEATURES_9, 'A. 2h, 9 features  [BASELINE fit input]')
CD9, MXD9, CND9 = corr_report(FEAT_D, FEATURES_9, 'B. DAILY, 9 features  [arm 2 fit input]')

# ---- RESOLVE arm 3's dispersion slot by the rule declared in section 2 ------
print(f'\nARM 3 dispersion/context slot -- applying the pre-declared rule '
      f'(momentum={F3_MOM}, volatility={F3_VOL}):')
_sc = {d: max(abs(CD9.loc[d, F3_MOM]), abs(CD9.loc[d, F3_VOL]))
       for d in F3_DISP_CANDIDATES}
for d, v in _sc.items():
    print(f'    {d:<9} max |r| against the other two slots = {v:.3f}')
F3_DISP = min(_sc, key=_sc.get)
FEATURES_3 = [F3_MOM, F3_VOL, F3_DISP]
assert all(f in FEATURES_9 for f in FEATURES_3)
print(f'  -> dispersion slot = {F3_DISP!r}')
if F3_DISP != 'dist_ma':
    print(f'  NOTE: the "obvious" role-based pick, dist_ma, scores {_sc["dist_ma"]:.3f} and is')
    print('        REJECTED by the rule. Hard-coding it would have produced an arm labelled')
    print('        "decollinearized" that was not -- which is exactly the assumption this')
    print('        section exists to stop.')

CD3, MXD3, CND3 = corr_report(FEAT_D, FEATURES_3, 'C. DAILY, 3 features  [arm 3 fit input]')
if MXD3 >= COLLINEAR_HI:
    print(f'  *** ARM 3 IS NOT ACTUALLY DECOLLINEARIZED: max |r| = {MXD3:.3f} >= '
          f'{COLLINEAR_HI}. ***')
    print('  Whatever arm 3 measures, it is NOT "the collinearity removed". Read its row')
    print('  with that in mind.')
else:
    print(f'  arm 3 IS decollinearized: max |r| = {MXD3:.3f} < {COLLINEAR_HI}, condition '
          f'number {CND3:.1f} vs {CND9:.1f} for the 9-feature daily set.')

print('\n' + '=' * 96)
print('THE COLLINEARITY CLAIM, CHECKED')
print('=' * 96)
for _tag, _C in (('2h   ', C2H), ('daily', CD9)):
    _tri = [(a, b, abs(_C.loc[a, b])) for a, b in itertools.combinations(MOM, 2)]
    print(f'  {_tag} nested momentum trio: ' +
          '   '.join(f'|r({a},{b})| = {v:.3f}' for a, b, v in _tri))
_mx2 = max(abs(C2H.loc[a, b]) for a, b in itertools.combinations(MOM, 2))
_mxd = max(abs(CD9.loc[a, b]) for a, b in itertools.combinations(MOM, 2))
print('-' * 96)
if _mx2 >= COLLINEAR_HI:
    print(f'  VERDICT: the 2h momentum trio IS highly collinear (max |r| = {_mx2:.3f} '
          f'>= {COLLINEAR_HI}).')
    print('  The likelihood-ridge explanation for the bimodality is CONSISTENT with the')
    print('  data. Consistent is not proven -- collinearity is necessary for that story,')
    print('  not sufficient. Arm 3 is the actual test of it.')
else:
    print(f'  VERDICT: the 2h momentum trio is NOT highly collinear '
          f'(max |r| = {_mx2:.3f} < {COLLINEAR_HI}).')
    print('  *** THE COLLINEARITY EXPLANATION FOR THE BIMODALITY IS THEREFORE WRONG. ***')
    print('  The nested mom_1d/mom_3d/mom_5d features are not creating a likelihood')
    print('  ridge here, so whatever produces the two basins is something else. This is')
    print('  reported as a refutation, not softened.')
print(f'  (daily trio max |r| = {_mxd:.3f}; condition number 2h {CN2H:,.0f} vs '
      f'daily {CND9:,.0f} vs daily-3feat {CND3:,.0f})')
print('=' * 96)
""")

co(r"""
# ===========================================================================
# THE CAUSAL DAILY -> 2h PROJECTION.
# ===========================================================================
def daily_source_index(daily_index, bar_index):
    '''For every bar in `bar_index`, the position in `daily_index` of the last
    daily bar STRICTLY BEFORE that bar's session date.

    searchsorted(..., 'left') - 1 gives the last daily date < session date, so a
    bar on day d can only ever see day d-1 or earlier. Returns -1 where no such
    daily bar exists (those bars are dropped, loudly).
    '''
    sess = pd.DatetimeIndex(bar_index).normalize().values
    di = pd.DatetimeIndex(daily_index).normalize().values
    assert (np.diff(di.astype('int64')) > 0).all(), 'daily index must be strictly increasing'
    return np.searchsorted(di, sess, side='left') - 1, sess, di


SRC_IDX, _SESS, _DI = daily_source_index(FEAT_D.index, FEAT_2H.index)
_have = SRC_IDX >= 0
if not _have.all():
    print(f'dropping {int((~_have).sum())} leading 2h bars with no fully-closed daily '
          f'bar behind them')

# EVERY arm is restricted to the SAME bar set, so S / L / W stay comparable.
FEAT_DF = FEAT_2H[_have]
DATES = FEAT_DF.index
SRC_IDX = SRC_IDX[_have]
CLOSE = nifty.reindex(DATES).values
TREND_RAW = FEAT_DF[TREND_FEATURE].values
N_BARS = len(DATES)
N_FIT = max(int(N_BARS * TRAIN_FRACTION), 50)          # 2h anchored fit window
N_FIT_D = max(int(len(FEAT_D) * TRAIN_FRACTION), 50)   # DAILY anchored fit window

# ---- PROOF 1: the per-bar causality assert --------------------------------
_src_ts = _DI[SRC_IDX]
_bar_sess = pd.DatetimeIndex(DATES).normalize().values
assert (_src_ts < _bar_sess).all(), \
    'CAUSALITY VIOLATION: a 2h bar is reading a daily bar from its own session or later'
_gap_days = (_bar_sess - _src_ts).astype('timedelta64[D]').astype(int)
print('ASSERT OK (proof 1): for all %d 2h bars, source daily date < bar session date.' % N_BARS)
print(f'  source-to-bar gap in calendar days: min {_gap_days.min()}  median '
      f'{int(np.median(_gap_days))}  max {_gap_days.max()}  '
      '(min must be >= 1, and it is)')
assert _gap_days.min() >= 1

print(f'\n2h  : {N_BARS} bars   {DATES[0]:%Y-%m-%d} -> {DATES[-1]:%Y-%m-%d}   '
      f'anchored fit window = {N_FIT} bars')
print(f'daily: {len(FEAT_D)} bars   {FEAT_D.index[0]:%Y-%m-%d} -> '
      f'{FEAT_D.index[-1]:%Y-%m-%d}   anchored fit window = {N_FIT_D} bars '
      f'({FEAT_D.index[0]:%Y-%m-%d} -> {FEAT_D.index[N_FIT_D - 1]:%Y-%m-%d})')
if FEAT_D.index[N_FIT_D - 1] < DATES[0]:
    print('  NOTE: the DAILY fit window CLOSES BEFORE the 2h evaluation span even begins,')
    print('        so arms 2 and 3 are scored entirely out of sample on the daily fit.')
""")

co(r"""
# ===========================================================================
# THE DAILY DIRECTION MODEL + its projection onto 2h bars.
# ===========================================================================
def seed_sets(K, R=R_SEED_SETS, base=BASE_SEED):
    ss = [list(range(base + r * K, base + r * K + K)) for r in range(R)]
    flat = [s for st in ss for s in st]
    assert len(set(flat)) == len(flat), 'seed sets must be DISJOINT'
    return ss


def scaled(df, feats, n_fit):
    raw = df[list(feats)].values
    return StandardScaler().fit(raw[:n_fit]).transform(raw)


def daily_masses(feat_d, feats, n_fit_d, seeds, N=None, cov=None):
    '''ANCHORED daily fit -> causal filtered direction masses per DAILY bar.

    The models see ONLY feat_d[:n_fit_d]; the masses come from the forward-only
    filtered posteriors, so daily bar j uses daily bars <= j and nothing later.
    '''
    N = BASE_N if N is None else N
    cov = BASE_COV if cov is None else cov
    Xd = scaled(feat_d, feats, n_fit_d)
    models, _ = fit_hmm_ensemble(Xd[:n_fit_d], N, cov, K=len(seeds), base_seed=seeds[0])
    b, s, r, probs = ensemble_direction_masses_by_mode(
        models, Xd, list(feats), DIRECTION_EXCLUDE, DIRECTION_MODE, None, None, None)
    return models, np.stack([b, s, r], axis=1), probs


def project(mass_d, probs_d, src_idx):
    '''Daily -> 2h. Row t of the output is the daily row for the last FULLY
    CLOSED session before bar t. Because `src_idx` is constant within a calendar
    day, this is exactly "shift one full day, then forward-fill".'''
    return mass_d[src_idx], probs_d[src_idx]


@contextlib.contextmanager
def projected_direction(mass_2h, probs_2h):
    '''Route label_bars' direction masses through the PROJECTED DAILY masses.

    label_bars resolves `ensemble_direction_masses_by_mode` from the notebook
    globals at call time, so rebinding that name substitutes the direction axis
    WITHOUT editing a line of the inlined engine. The INTENSITY axis
    (trend_raw / efficiency / gate_band) is untouched and stays on 2h -- that is
    the whole architecture in one context manager.
    '''
    g = globals()
    old = g['ensemble_direction_masses_by_mode']

    def _proj(models, Xs, feature_subset, exclude=None, direction_mode=None,
              tau=None, c=None, scale=None):
        assert len(Xs) == len(mass_2h), 'projected masses are misaligned with the bars'
        return mass_2h[:, 0], mass_2h[:, 1], mass_2h[:, 2], probs_2h

    g['ensemble_direction_masses_by_mode'] = _proj
    try:
        yield
    finally:
        g['ensemble_direction_masses_by_mode'] = old


print('projection machinery ready: daily fit -> filtered daily masses -> shift one '
      'full session -> forward-fill onto 2h bars -> injected as the DIRECTION axis.')
print('The INTENSITY axis (trend_z, efficiency, the H/L gate) remains 100% 2h in every arm.')
""")

co(r"""
# ===========================================================================
# PROOF 2 -- THE TRUNCATION PROBE. The real test.
#
# Re-run the ENTIRE pipeline on a series cut at T, with the daily series cut to
# sessions STRICTLY BEFORE T's session -- exactly the information a live engine
# would have held at T. Every label at t <= T must be BIT-IDENTICAL.
#
# The anchored fit windows are held at their ABSOLUTE bar counts, because that is
# what "anchored" means: the model is fit on a fixed leading window and applied
# forward. A probe that also shrank the fit window would be testing a different
# model, not the causality of this one.
# ===========================================================================
def run_arm(px, vx, feat_d_px, feat_d_vx, feats, seeds, n_fit_2h, n_fit_d,
            daily=True, end=None):
    '''One full end-to-end labelling. `end` truncates the 2h series at that
    timestamp and the daily series at the last session strictly before it.'''
    f2 = build_features(px, vx, LOOKBACK_SCALE)
    fd = build_features(feat_d_px, feat_d_vx, LOOKBACK_SCALE)
    if end is not None:
        f2 = f2[f2.index <= end]
        fd = fd[fd.index < pd.Timestamp(end).normalize()]
    src, _, di = daily_source_index(fd.index, f2.index)
    keep = src >= 0
    f2, src = f2[keep], src[keep]
    dates = f2.index
    assert (di[src] < pd.DatetimeIndex(dates).normalize().values).all(), \
        'CAUSALITY VIOLATION inside run_arm'
    X2 = scaled(f2, FEATURES_9, n_fit_2h)
    tr = f2[TREND_FEATURE].values
    if daily:
        dmodels, md_, pd_ = daily_masses(fd, feats, n_fit_d, seeds)
        m2, p2 = project(md_, pd_, src)
        with projected_direction(m2, p2):
            out = label_bars(dmodels, X2, dates, px, tr, n_fit_2h, FEATURES_9,
                             dir_feats=f2)
    else:
        Xf = scaled(f2, feats, n_fit_2h)
        models, _ = fit_hmm_ensemble(Xf[:n_fit_2h], BASE_N, BASE_COV,
                                     K=len(seeds), base_seed=seeds[0])
        out = label_bars(models, Xf, dates, px, tr, n_fit_2h, list(feats),
                         dir_feats=f2)
    return dates, out['tactical_regime_state'].values.copy()


_pseeds = seed_sets(ENSEMBLE_K)[0]
_t0 = time.time()
_d_full, _l_full = run_arm(nifty, vix, NIFTY_D, VIX_D, FEATURES_9, _pseeds,
                           N_FIT, N_FIT_D, daily=True)
_T = _d_full[-40]
_d_cut, _l_cut = run_arm(nifty[nifty.index <= _T], vix[vix.index <= _T],
                         NIFTY_D, VIX_D, FEATURES_9, _pseeds,
                         N_FIT, N_FIT_D, daily=True, end=_T)

_common = _d_full <= _T
assert len(_l_cut) == int(_common.sum()), \
    f'truncation changed the BAR COUNT: {len(_l_cut)} vs {int(_common.sum())}'
assert (_d_full[_common] == _d_cut).all(), 'truncation changed the bar TIMESTAMPS'
_bad = int((_l_full[_common] != _l_cut).sum())
print(f'TRUNCATION PROBE   cut at {_T:%Y-%m-%d %H:%M}   '
      f'{int(_common.sum())} labels compared   mismatches = {_bad}   '
      f'({time.time() - _t0:.1f}s)')
assert _bad == 0, (
    f'*** HARD STOP: {_bad} labels changed when the series was truncated. The '
    'daily->2h projection is NOT causal. Do not work around this. ***')
print('ASSERT OK (proof 2): truncating the series left every label at t <= T '
      'BIT-IDENTICAL.')
print('The daily->2h projection is CAUSAL by measurement, not by claim.')
""")

# ===========================================================================
md(r"""
## 5. Metrics S / L / W, guards G1–G4, and the ZigZag leakage tripwire

Reused **verbatim** from `stability_lag.ipynb` so rows are comparable across the
two notebooks. The ZigZag deliberately uses future prices — it is a
*retrospective* grading device — so it is fenced off from labelling by three
runtime asserts, and `G4` is driven by a degenerate all-`SIDEWAYS` stub and
asserted to void it.
""")

co(r"""
# ===========================================================================
# THE ZIGZAG -- LABEL-BLIND, RETROSPECTIVE. VERBATIM from stability_lag.ipynb.
# ===========================================================================
ZZ_CALLS = 0
ZZ_IDS = set()
ZZ_ARRAYS = []


def zigzag_pivots(px, pct=ZZ_PCT):
    global ZZ_CALLS
    px = np.asarray(px, dtype=float)
    n = len(px)
    if n < 3:
        out = np.array([], dtype=int)
    else:
        hi_i = lo_i = 0
        d, start, piv, ext_i = 0, None, [], 0
        for i in range(1, n):
            if px[i] > px[hi_i]:
                hi_i = i
            if px[i] < px[lo_i]:
                lo_i = i
            if (px[hi_i] - px[i]) / px[hi_i] * 100.0 >= pct:
                d, piv, ext_i, start = -1, [hi_i], i, i
                break
            if (px[i] - px[lo_i]) / px[lo_i] * 100.0 >= pct:
                d, piv, ext_i, start = +1, [lo_i], i, i
                break
        if d == 0:
            out = np.array([], dtype=int)
        else:
            for i in range(start + 1, n):
                if d > 0:
                    if px[i] >= px[ext_i]:
                        ext_i = i
                    elif (px[ext_i] - px[i]) / px[ext_i] * 100.0 >= pct:
                        piv.append(ext_i); d, ext_i = -1, i
                else:
                    if px[i] <= px[ext_i]:
                        ext_i = i
                    elif (px[i] - px[ext_i]) / px[ext_i] * 100.0 >= pct:
                        piv.append(ext_i); d, ext_i = +1, i
            if ext_i != piv[-1]:
                piv.append(ext_i)
            out = np.asarray(piv, dtype=int)
    ZZ_CALLS += 1
    ZZ_IDS.add(id(out))
    ZZ_ARRAYS.append(out)
    return out


def zigzag_swings(px, pct=ZZ_PCT):
    px = np.asarray(px, dtype=float)
    piv = zigzag_pivots(px, pct)
    sw = []
    for a, b in zip(piv[:-1], piv[1:]):
        move = (px[b] - px[a]) / px[a] * 100.0
        if abs(move) >= pct:
            sw.append((int(a), int(b), 1 if move > 0 else -1))
    ZZ_IDS.add(id(sw))
    return sw


# ---- the leakage tripwire -------------------------------------------------
_label_bars_raw = label_bars
LABEL_CALLS = 0
LABEL_PHASE_OPEN = True


def label_bars(*args, **kwargs):
    global LABEL_CALLS
    assert ZZ_CALLS == 0 or LABEL_PHASE_OPEN, (
        'LEAKAGE: a label was produced AFTER a ZigZag had been computed.')
    _bw = kwargs.get('bar_dir_weight', BAR_DIR_WEIGHT)
    assert _bw == 0.0, f'BAR_DIR_WEIGHT drifted to {_bw} inside a labelling call'
    for v in list(args) + list(kwargs.values()):
        assert id(v) not in ZZ_IDS, 'LEAKAGE: a ZigZag output was passed to label_bars'
        if isinstance(v, np.ndarray):
            for z in ZZ_ARRAYS:
                assert not np.shares_memory(v, z), \
                    'LEAKAGE: a label_bars argument aliases ZigZag memory'
    LABEL_CALLS += 1
    return _label_bars_raw(*args, **kwargs)


DIR_OF_LABEL = {'H_BULL': 1, 'L_BULL': 1, 'SIDEWAYS': 0, 'L_BEAR': -1, 'H_BEAR': -1}


def emitted_direction(labels):
    return np.array([DIR_OF_LABEL[x] for x in np.asarray(labels)], dtype=int)


def metric_S(label_sets):
    R = len(label_sets)
    M = np.full((R, R), np.nan)
    vals = []
    for i in range(R):
        M[i, i] = 100.0
        for j in range(i + 1, R):
            a, b = np.asarray(label_sets[i]), np.asarray(label_sets[j])
            assert len(a) == len(b)
            v = 100.0 * float(np.mean(a == b))
            M[i, j] = M[j, i] = v
            vals.append(v)
    return float(np.mean(vals)), M, vals


def metric_L(labels, swings):
    d = emitted_direction(labels)
    lags, unmatched = [], 0
    for i0, i1, sgn in swings:
        seg = d[i0:i1 + 1]
        hit = np.flatnonzero(seg == sgn)
        if hit.size:
            lags.append(int(hit[0]))
        else:
            lags.append(int(i1 - i0)); unmatched += 1
    if not lags:
        return np.nan, np.nan, np.array([]), 0
    a = np.asarray(lags, dtype=float)
    return float(np.median(a)), float(np.percentile(a, 75)), a, unmatched


def metric_W(labels):
    a = np.asarray(labels)
    return 0.0 if len(a) < 2 else 100.0 * float(np.sum(a[1:] != a[:-1])) / (len(a) - 1)


def occupancy_pct(labels):
    a = np.asarray(labels)
    return {lb: 100.0 * float(np.mean(a == lb)) for lb in REGIME_LABELS}


def evaluate_guards(occ, W):
    g = {}
    bad_hi = [l for l in REGIME_LABELS if occ.get(l, 0.0) > OCC_MAX_PCT]
    bad_lo = [l for l in REGIME_LABELS if occ.get(l, 0.0) < OCC_MIN_PCT]
    g['G1'] = (not bad_hi and not bad_lo, 'ok' if not (bad_hi or bad_lo)
               else f'outside [{OCC_MIN_PCT},{OCC_MAX_PCT}]%: '
                    + ','.join(sorted(set(bad_hi + bad_lo))))
    g['G2'] = (not bad_lo, 'ok' if not bad_lo else 'collapsed: ' + ','.join(bad_lo))
    g['G3'] = (W <= W_GUARD_MAX, f'W={W:.2f} vs max {W_GUARD_MAX}')
    g['G4'] = (occ.get('SIDEWAYS', 0.0) <= SIDEWAYS_MAX,
               f"SIDEWAYS={occ.get('SIDEWAYS', 0.0):.1f}% vs max {SIDEWAYS_MAX}%")
    return g


def guards_pass(g):
    return all(v[0] for v in g.values())


# ---- ZigZag unit test on a toy array, then the bookkeeping is RESET so the
# ---- tripwire starts the label phase from a true zero.
_saw = np.array([100, 101, 100.5, 105, 104, 101, 100, 103, 108, 107, 102, 106.])
_sw = zigzag_swings(_saw, pct=2.0)
assert len(_sw) >= 3
for _a, _b, _s in _sw:
    _mv = (_saw[_b] - _saw[_a]) / _saw[_a] * 100
    assert _a < _b and np.sign(_mv) == _s and abs(_mv) >= 2.0
ZZ_CALLS = 0; ZZ_IDS.clear(); ZZ_ARRAYS.clear()
print(f'ZigZag unit test PASS ({len(_sw)} alternating swings); bookkeeping reset, '
      'tripwire starts from ZZ_CALLS = 0.')
print('metrics ready: metric_S (full pairwise matrix), metric_L, metric_W, '
      'evaluate_guards G1-G4. NO composite score is defined anywhere.')
""")

# ===========================================================================
md(r"""
## 6. Label phase — three arms × R = 4 disjoint seed sets

Everything is labelled **before any ZigZag exists**; that ordering is the leakage
proof. Across the three arms only the **direction** axis changes. The intensity
gate, the confirmation delay, the hysteresis, `CONF_L`, the config `N`/`cov` and
the 2h price series are identical in all three — one knob at a time.
""")

co(r"""
# ===========================================================================
# THE LABEL PHASE.
# ===========================================================================
assert ZZ_CALLS == 0, 'a ZigZag was computed before the label phase'

ARMS = [
    dict(name='1 BASELINE (2h fit)', daily=False, features=FEATURES_9,
         note='current 2h fit, existing 9 features'),
    dict(name='2 DAILY-9FEAT', daily=True, features=FEATURES_9,
         note='daily fit, long history, SAME 9 features -- isolates the DATA change'),
    dict(name='3 DAILY-3FEAT', daily=True, features=FEATURES_3,
         note='daily fit + decollinearized features -- adds the FEATURE change'),
]

X2_9 = scaled(FEAT_DF, FEATURES_9, N_FIT)
SEEDSETS = seed_sets(ENSEMBLE_K)
LABELS, FIT_COUNT, _t0 = {}, 0, time.time()

for a in ARMS:
    outs = []
    for sd in SEEDSETS:
        if a['daily']:
            dmodels, md_, pd_ = daily_masses(FEAT_D, a['features'], N_FIT_D, sd)
            m2, p2 = project(md_, pd_, SRC_IDX)
            with projected_direction(m2, p2):
                r = label_bars(dmodels, X2_9, DATES, nifty, TREND_RAW, N_FIT,
                               FEATURES_9, dir_feats=FEAT_DF)
        else:
            Xf = scaled(FEAT_DF, a['features'], N_FIT)
            models, _ = fit_hmm_ensemble(Xf[:N_FIT], BASE_N, BASE_COV,
                                         K=ENSEMBLE_K, base_seed=sd[0])
            r = label_bars(models, Xf, DATES, nifty, TREND_RAW, N_FIT,
                           list(a['features']), dir_feats=FEAT_DF)
        FIT_COUNT += ENSEMBLE_K
        outs.append(r['tactical_regime_state'].values.copy())
    LABELS[a['name']] = outs
    print(f"  labelled {a['name']:<22} feats={len(a['features'])}  "
          f"({time.time() - _t0:5.1f}s cumulative)")

print(f'\nLABEL PHASE COMPLETE in {time.time() - _t0:.1f}s   {FIT_COUNT} HMM fits, '
      f'{LABEL_CALLS} label_bars calls')
assert ZZ_CALLS == 0, 'LEAKAGE: a ZigZag existed during the label phase'
print('ASSERT OK: ZZ_CALLS == 0 -- every label was produced before any ZigZag existed.')
LABEL_PHASE_OPEN = False
print('LABEL PHASE CLOSED. Any label_bars call from here on fails the tripwire.')
""")

co(r"""
# ===========================================================================
# EVAL PHASE. ZigZags may now be computed.
# ===========================================================================
SWINGS = zigzag_swings(CLOSE, ZZ_PCT)
print(f'ZigZag ({ZZ_PCT}%) found {len(SWINGS)} swings over {N_BARS} 2h bars '
      f'(the L denominator).')


def n_regimes_observed(close, pct=MACRO_ZZ_PCT):
    '''Distinct macro price legs inside a window. This is a DESCRIPTIVE count of
    the fit window, computed in the EVAL phase; it never touches a label.'''
    return len(zigzag_swings(np.asarray(close, dtype=float), pct))


NREG_2H = n_regimes_observed(CLOSE[:N_FIT])
NREG_D = n_regimes_observed(NIFTY_D.reindex(FEAT_D.index).values[:N_FIT_D])
_fit_yrs_2h = (DATES[N_FIT - 1] - DATES[0]).days / 365.25
_fit_yrs_d = (FEAT_D.index[N_FIT_D - 1] - FEAT_D.index[0]).days / 365.25

print('\n' + '=' * 96)
print(f'N_REGIMES_OBSERVED  -- distinct {MACRO_ZZ_PCT:.0f}% macro legs inside each '
      f'arm\'s FIT WINDOW')
print('=' * 96)
print(f'  arm 1 BASELINE   2h fit window : {N_FIT:>6} bars  {_fit_yrs_2h:5.2f} yrs  '
      f'{DATES[0]:%Y-%m-%d} -> {DATES[N_FIT-1]:%Y-%m-%d}   '
      f'N_REGIMES_OBSERVED = {NREG_2H}')
print(f'  arms 2 & 3       daily fit win : {N_FIT_D:>6} bars  {_fit_yrs_d:5.2f} yrs  '
      f'{FEAT_D.index[0]:%Y-%m-%d} -> {FEAT_D.index[N_FIT_D-1]:%Y-%m-%d}   '
      f'N_REGIMES_OBSERVED = {NREG_D}')
print('-' * 96)
print(f'  rows per free parameter   2h: {N_FIT / n_params(BASE_N, 9, BASE_COV):.1f}   '
      f'daily 9-feat: {N_FIT_D / n_params(BASE_N, 9, BASE_COV):.1f}   '
      f'daily 3-feat: {N_FIT_D / n_params(BASE_N, 3, BASE_COV):.1f}')
print(f'  REGIME MULTIPLE: the daily fit sees {NREG_D / max(NREG_2H, 1):.1f}x as many '
      f'macro legs as the 2h fit.')
if NREG_D <= NREG_2H * 1.5:
    print('  *** THE PREMISE OF THIS NOTEBOOK IS NOT MET ON THIS DATA: the daily fit')
    print('      window does NOT contain materially more macro regimes. Every result')
    print('      below is therefore testing something weaker than the intended change.')
print('=' * 96)
""")

# ===========================================================================
md(r"""
## 7. Results — S, L and W together, on every row. Baseline is row 1.

The **pairwise S matrix is printed per arm**: the bimodality P1 and P2 are about
is only visible there, never in the mean.
""")

co(r"""
# ===========================================================================
# SUMMARISE EVERY ARM, then DRIVE G4 with a degenerate stub through the SAME
# code path, so the guard is tested rather than trusted.
# ===========================================================================
def summarise_arm(name, label_sets, note='', rows_per_param=np.nan, nreg=np.nan):
    S, M, pair_vals = metric_S(label_sets)
    L_med = [metric_L(lb, SWINGS)[0] for lb in label_sets]
    L_p75 = [metric_L(lb, SWINGS)[1] for lb in label_sets]
    unm = [metric_L(lb, SWINGS)[3] for lb in label_sets]
    Ws = [metric_W(lb) for lb in label_sets]
    occs = [occupancy_pct(lb) for lb in label_sets]
    occ = {lb: float(np.mean([o[lb] for o in occs])) for lb in REGIME_LABELS}
    occ_spread = max(max(o[lb] for o in occs) - min(o[lb] for o in occs)
                     for lb in REGIME_LABELS)
    W = float(np.mean(Ws))
    g = evaluate_guards(occ, W)
    sv = sorted(pair_vals)
    return dict(name=name, note=note, S=S, S_min=min(pair_vals), S_max=max(pair_vals),
                S_matrix=M, S_pairs=pair_vals,
                S_spread=max(pair_vals) - min(pair_vals),
                S_gap=max(np.diff(sv)) if len(sv) > 1 else 0.0,
                n_low_pairs=int(sum(1 for v in pair_vals if v < BASIN_LOW)),
                L=float(np.mean(L_med)), L_lo=float(np.min(L_med)),
                L_hi=float(np.max(L_med)), L75=float(np.mean(L_p75)),
                unmatched=float(np.mean(unm)),
                W=W, W_lo=float(np.min(Ws)), W_hi=float(np.max(Ws)),
                occ=occ, occ_spread=occ_spread, guards=g, void=not guards_pass(g),
                rows_per_param=rows_per_param, nreg=nreg)


def _rpp(a):
    return (round(N_FIT_D / n_params(BASE_N, len(a['features']), BASE_COV), 1)
            if a['daily'] else
            round(N_FIT / n_params(BASE_N, len(a['features']), BASE_COV), 1))


RESULTS = [summarise_arm(a['name'], LABELS[a['name']], a['note'], _rpp(a),
                         NREG_D if a['daily'] else NREG_2H) for a in ARMS]
RES = {r['name']: r for r in RESULTS}
BASE = RESULTS[0]

TAB = pd.DataFrame([{
    'arm': r['name'], 'N_REG': r['nreg'], 'rows/par': r['rows_per_param'],
    'S %': round(r['S'], 2), 'S spread pp': round(r['S_spread'], 2),
    'L med': round(r['L'], 2), 'L p75': round(r['L75'], 2),
    'W /100': round(r['W'], 2), 'unmat': round(r['unmatched'], 1),
    'SIDE %': round(r['occ']['SIDEWAYS'], 1), 'occ spread pp': round(r['occ_spread'], 2),
    'G1': 'P' if r['guards']['G1'][0] else 'F', 'G2': 'P' if r['guards']['G2'][0] else 'F',
    'G3': 'P' if r['guards']['G3'][0] else 'F', 'G4': 'P' if r['guards']['G4'][0] else 'F',
    'verdict': 'VOID' if r['void'] else 'ok',
    'dS': round(r['S'] - BASE['S'], 2), 'dL': round(r['L'] - BASE['L'], 2),
    'dW': round(r['W'] - BASE['W'], 2),
} for r in RESULTS])
pd.set_option('display.width', 220, 'display.max_columns', 40)
print('S = mean pairwise 5-label agreement over R=%d DISJOINT seed sets (higher better)'
      % R_SEED_SETS)
print('L = ZigZag transition lag, median bars, mean over seed sets (lower better)')
print('W = label switches per 100 bars (guard)   N_REG = macro legs in the FIT WINDOW')
print('=' * 190)
print(TAB.to_string(index=False))
print('=' * 190)
print('NO COMPOSITE SCORE IS COMPUTED.')

# ---- G4 must BITE. Same summarise/guard path as the real arms. -------------
_stub = [np.full(N_BARS, 'SIDEWAYS', dtype='<U8') for _ in range(R_SEED_SETS)]
_r4 = summarise_arm('STUB all-SIDEWAYS', _stub, 'degeneracy stub')
assert _r4['S'] == 100.0, 'the stub should score a PERFECT S -- that is the point'
assert not _r4['guards']['G4'][0], 'G4 FAILED TO BITE on an all-SIDEWAYS labelling'
assert _r4['void'], 'the degenerate stub was not VOIDED'
print(f"\nGUARD TEST: an all-SIDEWAYS stub scores S = {_r4['S']:.2f}% -- a PERFECT score, "
      "which is exactly how S is gamed --")
print(f"            and is VOIDED by G4 ({_r4['guards']['G4'][1]}) and G1/G2. "
      'The guard is tested, not trusted.')
""")

co(r"""
# ===========================================================================
# THE FULL PAIRWISE S MATRICES -- one per arm. BIMODALITY IS ONLY VISIBLE HERE.
# ===========================================================================
def is_bimodal(r):
    return (r['S_spread'] >= BIMODAL_SPREAD) and (r['S_gap'] >= BIMODAL_GAP)


for r in RESULTS:
    print(f"\n{r['name']}   S = {r['S']:.2f}%   pairwise range "
          f"[{r['S_min']:.2f}, {r['S_max']:.2f}]   spread {r['S_spread']:.2f} pp   "
          f"largest gap {r['S_gap']:.2f} pp")
    print(pd.DataFrame(r['S_matrix'],
                       index=[f'set{i}' for i in range(R_SEED_SETS)],
                       columns=[f'set{i}' for i in range(R_SEED_SETS)]
                       ).round(2).to_string())
    print(f"   6 pairwise values sorted: "
          f"{[round(v, 2) for v in sorted(r['S_pairs'])]}")
    print(f"   BIMODAL = {is_bimodal(r)}   "
          f"(needs spread >= {BIMODAL_SPREAD} pp AND gap >= {BIMODAL_GAP} pp)   "
          f"across-basin pairs (< {BASIN_LOW}%): {r['n_low_pairs']}")

# ---- HEADROOM CHECK -- REQUIRED BEFORE ANY VERDICT ------------------------
print('\n' + '=' * 96)
print(f'HEADROOM CHECK (baseline, across the R={R_SEED_SETS} disjoint seed sets)')
print('=' * 96)
S_SPREAD = BASE['S_spread']
L_SPREAD = BASE['L_hi'] - BASE['L_lo']
W_SPREAD = BASE['W_hi'] - BASE['W_lo']
print(f"  S : {BASE['S']:.2f}%   pairwise span {BASE['S_min']:.2f} -> {BASE['S_max']:.2f}"
      f"   spread = {S_SPREAD:.2f} pp")
print(f"  L : {BASE['L']:.2f} bars   per-seed-set span {BASE['L_lo']:.2f} -> "
      f"{BASE['L_hi']:.2f}   spread = {L_SPREAD:.2f} bars")
print(f"  W : {BASE['W']:.2f}/100   per-seed-set span {BASE['W_lo']:.2f} -> "
      f"{BASE['W_hi']:.2f}   spread = {W_SPREAD:.2f}")
print('  RULE (pre-declared): a change smaller than the corresponding spread is NOISE.')
for r in RESULTS[1:]:
    _sn = abs(r['S'] - BASE['S']) <= S_SPREAD
    _ln = abs(r['L'] - BASE['L']) <= L_SPREAD
    print(f"  {r['name']:<22} dS {r['S'] - BASE['S']:+7.2f} pp "
          f"({'inside' if _sn else 'OUTSIDE'} the S spread)   "
          f"dL {r['L'] - BASE['L']:+6.2f} bars ({'inside' if _ln else 'OUTSIDE'} "
          'the L spread)')
print('=' * 96)
""")

# ===========================================================================
md(r"""
## 8. Self-grading against the pre-registered predictions

The arithmetic below is exactly the arithmetic printed in section 2, before any
number existed. Nothing is reinterpreted.
""")

co(r"""
# ===========================================================================
# GRADE P1..P4 AND STATE THE DIAGNOSIS VERDICT IN PLAIN WORDS.
# ===========================================================================
DAILY_ARMS = RESULTS[1:]
BEST_DAILY = max(DAILY_ARMS, key=lambda r: r['S'])
print(f"Graded on the best daily arm by S: {BEST_DAILY['name']}  "
      f"(the other daily arm is graded alongside it, not hidden).")
print('=' * 96)

GRADE = {}
for r in DAILY_ARMS:
    p1 = (r['S'] > P1_S_MIN) and (not is_bimodal(r))
    if BASE['n_low_pairs'] == 0 and r['n_low_pairs'] == 0:
        p2, p2txt = None, 'UNTESTABLE'
    else:
        p2 = r['n_low_pairs'] < BASE['n_low_pairs']
        p2txt = 'PASS' if p2 else 'FAIL'
    dL = r['L'] - BASE['L']
    p3 = not (dL < -L_SPREAD)      # a LARGE improvement would be inconsistent with P3
    p4 = r['occ_spread'] < BASE['occ_spread']
    GRADE[r['name']] = dict(P1=p1, P2=p2, P3=p3, P4=p4)
    print(f"\n{r['name']}")
    print(f"  P1  {'PASS' if p1 else 'FAIL'}   S = {r['S']:.2f}% vs required > "
          f"{P1_S_MIN:.1f}%   BIMODAL = {is_bimodal(r)} "
          f"(spread {r['S_spread']:.2f} pp, gap {r['S_gap']:.2f} pp)")
    print(f"  P2  {p2txt}   across-basin pairs (< {BASIN_LOW}%): "
          f"{r['n_low_pairs']} vs baseline {BASE['n_low_pairs']}"
          + ('   [baseline shows NO two-basin structure in this run, so there is '
             'nothing for P2 to weaken]' if p2 is None else ''))
    print(f"  P3  {'consistent' if p3 else 'INCONSISTENT'}   dL = {dL:+.2f} bars "
          f"(baseline L spread {L_SPREAD:.2f}). A worse L is EXPECTED and is NOT "
          'counted against the architecture.')
    print(f"  P4  {'PASS' if p4 else 'FAIL'}   max occupancy spread across seed sets "
          f"{r['occ_spread']:.2f} pp vs baseline {BASE['occ_spread']:.2f} pp")

g = GRADE[BEST_DAILY['name']]
print('\n' + '=' * 96)
print('THE DIAGNOSIS VERDICT')
print('=' * 96)
if g['P1'] and (g['P2'] is True):
    print('P1 and P2 both PASS -> THE SMALL-SAMPLE DIAGNOSIS IS CONFIRMED.')
    print('Fitting direction on daily bars with long history raised label agreement past')
    print(f'{P1_S_MIN:.0f}% and removed the two-basin structure. The instability was an')
    print('effective-sample-size problem, as claimed.')
elif (not g['P1']) and (g['P2'] is False):
    print('P1 and P2 BOTH FAIL -> THE SMALL-SAMPLE DIAGNOSIS IS WRONG.')
    print('Giving the direction model many more macro regimes did NOT stabilise the fit')
    print('and did NOT remove the two-basin structure. Whatever produces the instability')
    print('is not the effective sample size. This is a real result and it is reported as')
    print('a refutation -- no threshold has been moved and no prediction reinterpreted.')
elif g['P2'] is None:
    print('P2 IS UNTESTABLE IN THIS RUN: the baseline itself shows no two-basin structure')
    print('here, so there is nothing for the daily fit to remove. The diagnosis is')
    print(f"NEITHER confirmed nor refuted. P1 alone came out "
          f"{'PASS' if g['P1'] else 'FAIL'}.")
else:
    print('MIXED: P1 =', 'PASS' if g['P1'] else 'FAIL', ' P2 =',
          'PASS' if g['P2'] else 'FAIL')
    print('The diagnosis is PARTIALLY supported only. It is not confirmed, and the half')
    print('that failed is not explained away here.')
print('-' * 96)
print('GUARDS:', ' | '.join(
    f"{r['name']}: {'VOID (' + ' '.join(k for k, v in r['guards'].items() if not v[0]) + ')' if r['void'] else 'all pass'}"
    for r in RESULTS))
if TAC_SYNTH or DAILY_SYNTH:
    print('-' * 96)
    print('*** SYNTHETIC RUN. The verdict above grades the MACHINERY, not the market.  ***')
    print('*** The real P1-P4 verdict requires the Kaggle run with real yfinance data. ***')
print('=' * 96)

# ---- dominance, for completeness. Strict, no composite. -------------------
DOMINATING = [r for r in DAILY_ARMS if (not r['void']) and r['S'] > BASE['S']
              and r['L'] < BASE['L'] and r['W'] <= BASE['W']]
print()
if DOMINATING:
    for r in DOMINATING:
        print(f"DOMINATES THE BASELINE: {r['name']}  S {BASE['S']:.2f}->{r['S']:.2f}  "
              f"L {BASE['L']:.2f}->{r['L']:.2f}  W {BASE['W']:.2f}->{r['W']:.2f}")
else:
    print('NO DOMINATING CONFIGURATION (S up AND L down AND W not worse, all guards ok).')
    print('That is the EXPECTED shape here: P3 pre-declared that L would not improve, so')
    print('strict dominance was never the bar the daily arms were meant to clear. The')
    print('question this notebook asks is P1/P2, and that is answered above.')
""")

# ===========================================================================
md(r"""
## 9. Figures

Three, and only three — each one decides something. The pairwise-S panel is where
P1 and P2 are judged; the (L, S) frontier is where the S-for-L trade is priced;
the reference case is where a human decides whether the trade was worth it.
""")

co(r"""
# ===========================================================================
# FIG 1 - the pairwise S matrices, one per arm. THIS is where P1/P2 are judged.
# ===========================================================================
SAVED_PNGS = []


def save_fig(fig, fname, dpi=130):
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    SAVED_PNGS.append(fname)
    print(f'   saved -> {fname}')


SUFFIX = ('  [SYNTHETIC - ILLUSTRATIVE ONLY]' if (TAC_SYNTH or DAILY_SYNTH)
          else '  [real data]')
_lo = min(min(r['S_pairs']) for r in RESULTS)
VMIN = max(0.0, min(55.0, _lo - 2.0))

fig, axes = plt.subplots(2, len(RESULTS), figsize=(5.4 * len(RESULTS), 8.6),
                         gridspec_kw={'height_ratios': [3, 1.4], 'hspace': 0.42,
                                      'wspace': 0.22})
for j, r in enumerate(RESULTS):
    ax = axes[0, j]
    im = ax.imshow(r['S_matrix'], cmap='RdYlGn', vmin=VMIN, vmax=100.0)
    for i in range(R_SEED_SETS):
        for k in range(R_SEED_SETS):
            ax.text(k, i, f"{r['S_matrix'][i, k]:.1f}", ha='center', va='center',
                    fontsize=9, fontweight='bold' if i != k else 'normal',
                    color='#111111')
    # Ticks on TOP: at this figure height a bottom x-axis on the heatmap collides
    # with the strip panel's title directly beneath it.
    ax.set_xticks(range(R_SEED_SETS)); ax.set_yticks(range(R_SEED_SETS))
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels([f'set{i}' for i in range(R_SEED_SETS)], fontsize=8)
    ax.set_yticklabels([f'set{i}' for i in range(R_SEED_SETS)], fontsize=8)
    # NO COLOURBAR: every cell is annotated with its own number, so a colour key
    # decides nothing -- and at this width it overlapped the third panel.
    ax.set_title(f"{r['name']}\nS = {r['S']:.2f}%   spread {r['S_spread']:.2f} pp"
                 + ('   [VOID]' if r['void'] else ''),
                 fontsize=10, fontweight='bold', loc='left', pad=26)

    # the 6 pairwise values as a strip -- a two-basin structure is a GAP here
    axb = axes[1, j]
    v = sorted(r['S_pairs'])
    axb.scatter(v, np.zeros(len(v)), s=90, c='#1f77b4', edgecolors='black',
                linewidths=0.7, zorder=3)
    axb.axvline(P1_S_MIN, color='#2ca02c', ls='--', lw=1.2)
    axb.axvline(BASIN_LOW, color='#d62728', ls=':', lw=1.2)
    axb.set_yticks([])
    axb.set_xlim(VMIN, 101)
    axb.grid(alpha=0.25, axis='x')
    axb.set_xlabel('the 6 pairwise agreements (%)', fontsize=9)
    axb.set_title(f"BIMODAL = {is_bimodal(r)}   largest gap {r['S_gap']:.2f} pp   "
                  f"pairs < {BASIN_LOW:.0f}%: {r['n_low_pairs']}",
                  fontsize=9, loc='left',
                  color='#d62728' if is_bimodal(r) else '#2ca02c')
fig.suptitle('FIG 1 — PAIRWISE FIT-STABILITY MATRICES  (P1 and P2 are judged here)'
             + SUFFIX + f'\ncell colour: red {VMIN:.0f}% -> green 100%   '
             f'strip: green dashed = P1 threshold {P1_S_MIN:.0f}%, '
             f'red dotted = across-basin line {BASIN_LOW:.0f}%',
             fontsize=13, fontweight='bold')
fig.subplots_adjust(top=0.84)
save_fig(fig, 'fig1_pairwise_S_matrices.png')
plt.show()
""")

co(r"""
# ===========================================================================
# FIG 2 - the (L, S) frontier, all arms.
# ===========================================================================
fig, ax = plt.subplots(figsize=(11, 7))
_st = {'1 BASELINE (2h fit)': ('#d62728', '*', 460),
       '2 DAILY-9FEAT': ('#1f77b4', 'o', 170),
       '3 DAILY-3FEAT': ('#2ca02c', 's', 170)}
for r in RESULTS:
    c, mk, sz = _st.get(r['name'], ('#8c564b', 'D', 140))
    if r['void']:
        ax.scatter([r['L']], [r['S']], facecolors='none', edgecolors=c, marker=mk,
                   s=sz, linewidths=2.0, zorder=4, label=r['name'] + ' [VOID]')
    else:
        ax.scatter([r['L']], [r['S']], c=c, marker=mk, s=sz, zorder=4,
                   edgecolors='black', linewidths=0.7, label=r['name'])
    ax.errorbar([r['L']], [r['S']],
                xerr=[[r['L'] - r['L_lo']], [r['L_hi'] - r['L']]],
                yerr=[[r['S'] - r['S_min']], [r['S_max'] - r['S']]],
                fmt='none', ecolor=c, elinewidth=1.1, capsize=4, alpha=0.65, zorder=3)
    # annotated ABOVE-right, so the label never lands on the P1 / baseline rules
    ax.annotate(f"N_REG={r['nreg']}  W={r['W']:.1f}", (r['L'], r['S']), fontsize=8,
                xytext=(11, 10), textcoords='offset points', color='#333333')
ax.axvline(BASE['L'], color='#d62728', ls=':', lw=1.0)
ax.axhline(BASE['S'], color='#d62728', ls=':', lw=1.0)
ax.axhline(P1_S_MIN, color='#2ca02c', ls='--', lw=1.2)
ax.margins(x=0.14, y=0.10)
_x0, _x1 = ax.get_xlim()          # extra room so the right-most annotation is not clipped
ax.set_xlim(_x0, _x1 + 0.10 * (_x1 - _x0))
ax.annotate(f'P1 threshold  S = {P1_S_MIN:.0f}%', (ax.get_xlim()[1], P1_S_MIN),
            fontsize=9, va='top', ha='right', color='#2ca02c',
            xytext=(-6, -4), textcoords='offset points')
ax.set_xlabel('L — transition lag, median bars  (LOWER IS BETTER  <—)', fontsize=10)
ax.set_ylabel('S — mean pairwise label agreement %  (HIGHER IS BETTER)', fontsize=10)
ax.set_title('error bars are the FULL across-seed-set range of L and the full pairwise '
             'range of S — anything inside them is noise', fontsize=9, loc='left')
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc='best')
fig.suptitle('FIG 2 — the (L, S) frontier: does the daily fit buy stability, and at '
             'what lag cost?' + SUFFIX, fontsize=13, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'fig2_frontier_L_vs_S.png')
plt.show()
""")

co(r"""
# ===========================================================================
# FIG 3 - THE REFERENCE CASE: the May-2025 V-bottom, BASELINE vs the best daily
# arm, in the MASTER NOTEBOOK'S regime-background style: black price line,
# FULL-HEIGHT regime bands, TIGHT y-limits asserted EQUAL and NOT zero-anchored.
# ===========================================================================
REF_START, REF_END = pd.Timestamp('2025-05-05'), pd.Timestamp('2025-05-26')
_m = (DATES >= REF_START) & (DATES <= REF_END)
if _m.sum() < 6:
    _up = max([s for s in SWINGS if s[2] > 0],
              key=lambda s: (CLOSE[s[1]] - CLOSE[s[0]]) / CLOSE[s[0]])
    _a, _b = max(0, _up[0] - 12), min(N_BARS - 1, _up[1] + 12)
    _m = np.zeros(N_BARS, bool); _m[_a:_b + 1] = True
    REF_NOTE = (f'REFERENCE WINDOW {REF_START:%Y-%m-%d} -> {REF_END:%Y-%m-%d} IS NOT IN '
                'THIS DATA SPAN — substituted the largest up-swing in the series')
else:
    REF_NOTE = f'REFERENCE CASE: {REF_START:%Y-%m-%d} -> {REF_END:%Y-%m-%d}'

REF_IDX, REF_CLOSE = DATES[_m], CLOSE[_m]
_ref_move = 100 * (REF_CLOSE[-1] - REF_CLOSE.min()) / REF_CLOSE.min()


def shade_regimes_contiguous(ax, series, alpha=0.35):
    '''Bands that BUTT UP against each other -- the master's regime_blocks ends a
    band on the LAST BAR of its run, which leaves a white stripe across every
    weekend at this zoom and reads as "no regime here", which is false. The
    PAINTER is still the master's shade_bands, unmodified.'''
    vals, idx = np.asarray(series.values), series.index
    spans, start = [], 0
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            spans.append((vals[start], idx[start], idx[i])); start = i
    spans.append((vals[start], idx[start], idx[-1]))
    shade_bands(ax, spans, alpha=alpha)


_panels = [BASE, BEST_DAILY]
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
_ylim = None
for ax, r in zip(axes, _panels):
    lab = pd.Series(LABELS[r['name']][0], index=DATES)[_m]     # seed set 0, both panels
    shade_regimes_contiguous(ax, lab, alpha=0.35)
    ax.plot(REF_IDX, REF_CLOSE, color='black', linewidth=1.8, zorder=4)
    set_price_ylim(ax, REF_CLOSE, pad=0.03, tag=f"FIG3 {r['name']}")
    if _ylim is None:
        _ylim = ax.get_ylim()
    ax.set_ylim(*_ylim)
    ax.set_xlim(REF_IDX[0], REF_IDX[-1])
    ax.set_ylabel('Nifty close', fontsize=9)
    _d = emitted_direction(lab.values)
    _shown = 0
    for i0, i1, sgn in SWINGS:
        if not _m[i0]:
            continue
        c = '#0033cc' if sgn > 0 else '#cc0033'
        t0 = DATES[i0]
        ax.axvline(t0, color=c, ls='--', lw=1.3, zorder=6)
        _off = int(np.flatnonzero(np.asarray(lab.index) == np.datetime64(t0))[0])
        _hit = np.flatnonzero(_d[_off:] == sgn)
        _y = _ylim[0] + (0.94 - 0.085 * _shown) * (_ylim[1] - _ylim[0])
        if _hit.size:
            _lag = int(_hit[0]); t1 = lab.index[_off + _lag]
            ax.annotate('', xy=(t1, _y), xytext=(t0, _y),
                        arrowprops=dict(arrowstyle='<->', color=c, lw=1.5))
            ax.plot([t1], [REF_CLOSE[_off + _lag]], marker='o', ms=9, mfc='none',
                    mec=c, mew=2.0, zorder=7)
            ax.annotate(f"swing {'UP' if sgn > 0 else 'DOWN'} — L = {_lag} bars",
                        ((t0 + (t1 - t0) / 2) if _lag else t0, _y), fontsize=9,
                        color=c, ha='center', va='bottom', fontweight='bold')
        else:
            ax.annotate(f"swing {'UP' if sgn > 0 else 'DOWN'} — NEVER matched",
                        (t0, _y), fontsize=9, color=c, ha='left', va='bottom',
                        fontweight='bold')
        _shown += 1
    _f = np.flatnonzero(_d > 0)
    ax.set_title(f"{r['name']}{'  [VOID]' if r['void'] else ''}   S={r['S']:.2f}%   "
                 f"L={r['L']:.2f} bars (whole series)   W={r['W']:.2f}   |   "
                 + (f'first BULL bar in window: {lab.index[_f[0]]:%Y-%m-%d %H:%M}'
                    if _f.size else 'NEVER labelled BULL in this window'),
                 fontsize=11, fontweight='bold', loc='left')

fig.legend(handles=[mpatches.Patch(color=REGIME_COLORS[l], alpha=0.7, label=l)
                    for l in REGIME_LABELS],
           loc='upper center', bbox_to_anchor=(0.5, 0.925), ncol=5, fontsize=9,
           frameon=False)
axes[-1].set_xlabel('date', fontsize=10)
assert axes[0].get_ylim() == axes[1].get_ylim(), 'panel y-limits differ'
assert axes[0].get_ylim()[0] > 0, 'y-axis is zero-anchored -- tight limits required'
fig.suptitle('FIG 3 — ' + REF_NOTE + f'   ({len(REF_IDX)} bars, low->close '
             f'+{_ref_move:.2f}%)'
             + ('\n[SYNTHETIC - ILLUSTRATIVE ONLY — this is NOT the real May-2025 '
                'V-bottom]' if (TAC_SYNTH or DAILY_SYNTH) else '')
             + '\nsame bars, same intensity axis, same y-limits — only the DIRECTION '
               'fit differs',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.905))
print(f'ASSERT OK: both panels share y-limits {axes[0].get_ylim()} '
      '(tight, NOT zero-anchored)')
save_fig(fig, 'fig3_reference_case_may2025.png')
plt.show()
""")

# ===========================================================================
md(r"""
## 10. Verification and the config hand-off block
""")

co(r"""
# ===========================================================================
# FINAL VERIFICATION.
# ===========================================================================
print('=' * 90)
print('VERIFICATION')
print('=' * 90)
print(f'  label_bars calls (all guarded)   : {LABEL_CALLS}')
print(f'  HMM fits                         : {FIT_COUNT}')
print(f'  ZigZag computations              : {ZZ_CALLS}')
print(f'  arms measured                    : {len(RESULTS)}')
print(f'  seed sets (R, DISJOINT)          : {R_SEED_SETS}  {SEEDSETS}')
print(f'  BAR_DIR_WEIGHT                   : {BAR_DIR_WEIGHT} (asserted every call)')
print(f'  causality proof 1 (per-bar)      : PASS  min source->bar gap '
      f'{_gap_days.min()} day(s)')
print(f'  causality proof 2 (truncation)   : PASS  0 mismatches over '
      f'{int(_common.sum())} labels')
print(f'  G4 driven by a degenerate stub   : PASS  (S=100.00% VOIDED)')
try:
    label_bars(None)
except AssertionError as e:
    print(f'  ZigZag leakage tripwire (live)   : ARMED -> {str(e)[:56]}...')
else:
    raise AssertionError('the leakage tripwire did NOT fire after the eval phase')
print(f'  PNGs written                     : {len(SAVED_PNGS)}  {SAVED_PNGS}')
print(f'  total runtime                    : {time.time() - T_START:.1f}s')
print('=' * 90)
if TAC_SYNTH or DAILY_SYNTH:
    print('*** SYNTHETIC - ILLUSTRATIVE ONLY. No verdict above is decision-grade.  ***')
    print('*** Re-run on Kaggle with real yfinance data.                           ***')
    print('=' * 90)
""")

co(r"""
# ===========================================================================
# CONFIG HAND-OFF BLOCK -- pasteable Python for the winning arm.
# ===========================================================================
_win = BEST_DAILY if (GRADE[BEST_DAILY['name']]['P1']
                      and GRADE[BEST_DAILY['name']]['P2'] is True
                      and not BEST_DAILY['void']) else BASE
_warm = 'the daily-fit arm cleared P1 AND P2' if _win is not BASE else \
        'NO daily arm cleared both P1 and P2 -- the BASELINE is carried forward unchanged'
_wa = [a for a in ARMS if a['name'] == _win['name']][0]

print('# ' + '=' * 78)
print('# REGDET V1.1 -- DAILY-FIT CONFIG HAND-OFF BLOCK')
print(f'# generated {pd.Timestamp.now():%Y-%m-%d %H:%M}   data = '
      f'{"SYNTHETIC (ILLUSTRATIVE ONLY)" if (TAC_SYNTH or DAILY_SYNTH) else "REAL yfinance"}')
print(f'# WINNER: {_win["name"]}   because {_warm}')
print(f"# S={_win['S']:.2f}%  L_med={_win['L']:.2f}  L_p75={_win['L75']:.2f}  "
      f"W={_win['W']:.2f}  N_REGIMES_OBSERVED={_win['nreg']}  "
      f"rows/param={_win['rows_per_param']}")
print('# guards: ' + ' '.join(f"{k}={'PASS' if v[0] else 'FAIL'}"
                              for k, v in _win['guards'].items())
      + f"  -> {'VOID' if _win['void'] else 'ok'}")
print('#')
print('DAILY_FIT_CONFIG = dict(')
print(f'    DIRECTION_CADENCE   = {"daily" if _wa["daily"] else "2h"!r},')
print(f'    DIRECTION_FEATURES  = {list(_wa["features"])!r},')
print(f'    INTENSITY_CADENCE   = {"2h"!r},        # unchanged in every arm')
print(f'    N_STATES            = {BASE_N},')
print(f'    COVARIANCE_TYPE     = {BASE_COV!r},')
print(f'    ENSEMBLE_K          = {ENSEMBLE_K},')
print(f'    BASE_SEED           = {BASE_SEED},')
print(f'    TRAIN_FRACTION      = {TRAIN_FRACTION},')
print(f'    DAILY_FIT_START     = {str(DAILY_START.date())!r},   '
      f'# first date ^NSEI and India VIX BOTH exist')
print(f'    DAILY_FIT_BARS      = {N_FIT_D},')
print(f'    BAR_DIR_WEIGHT      = {W_FIXED},')
print(f'    CONFIRM_BARS        = {CONFIRM_BARS},')
print(f'    INTENSITY_MODE      = {INTENSITY_MODE!r},')
print(f'    DIRECTION_MODE      = {DIRECTION_MODE!r},')
print(f'    DIRECTION_EXCLUDE   = {DIRECTION_EXCLUDE!r},')
print(f'    ESCALATION_DURING_HOLD = {ESCALATION_DURING_HOLD!r},')
print(')')
print('# PROJECTION RULE (non-negotiable): the direction state applied to a 2h bar on')
print('# session d is the daily state of the last FULLY CLOSED session, d-1 or earlier.')
print('# ' + '=' * 78)
""")

md(r"""
## 11. What this notebook does and does not establish

**Does.** It measures S, L and W for a 2h-fit direction model and two daily-fit
ones on the *same* bars with the *same* intensity axis, `R = 4` disjoint seed
sets; it proves the daily→2h projection causal by truncation, not by assertion;
it checks the collinearity claim instead of assuming it; and it grades itself
against predictions frozen before the run.

**Does not.** It computes no forward return, no Sharpe and no economic metric —
deliberately. It does not re-tune `BAR_DIR_WEIGHT` or touch the intensity axis. It
does not combine S, L and W into a score. And on a sandbox run it establishes
nothing about the real market: the synthetic fallback is a plumbing test.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'wrote {OUT} - {len(cells)} cells '
      f'({sum(1 for c in cells if c["cell_type"] == "code")} code)')
