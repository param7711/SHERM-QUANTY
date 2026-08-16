"""Build fx_featureset.ipynb -- does replacing the shipped NESTED momentum
ladder (mom_1d/3d/5d, collinear, 5-day reach) with DISJOINT, non-overlapping
return blocks (0-1/1-5/5-20/20-60 days) fix the "barcode" the detector paints
through sustained grinds, without breaking anything else?

Protocol frozen in protocols/FX_FEATURESET_PROTOCOL.md BEFORE this file was
written: the design (which blocks, which feature dropped) was fixed by
measured collinearity structure alone, not by searching these instruments for
a winner -- so all 7 instruments are graded together, no discovery/held-out
split.

Setup harvested VERBATIM from build_intraday_fx.py, exactly as
build_fx_replication.py does. The engine (build_features, label_bars,
fit_hmm_ensemble, HAC contrast, ZigZag, S/L/W, guards, shade_regimes) is NOT
modified -- a second feature-construction function and a context manager to
swap the feature-set globals are ADDED alongside it.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(
    os.environ.get('REGDET_OUT_DIR',
                   os.path.join(os.path.dirname(HERE), 'notebooks')),
    'fx_featureset.ipynb')
FX_GEN = f'{HERE}/build_intraday_fx.py'

_fsrc = open(FX_GEN).read()
assert "'intraday_fx.ipynb'" in _fsrc
_fsrc = _fsrc.replace("'intraday_fx.ipynb'", "'_fx_featset_harvest.ipynb'", 1)
_ns = {'__name__': '_fx_gen_harvest', '__file__': FX_GEN}
exec(compile(_fsrc, FX_GEN, 'exec'), _ns)
_fcells = _ns['cells']
for _tmp in ('_fx_featset_harvest.ipynb', '_intraday_fx_engine_harvest.ipynb'):
    try:
        os.remove(os.path.join(os.path.dirname(OUT), _tmp))
    except OSError:
        pass


def _s(i):
    return ''.join(_fcells[i]['source'])


PIP, ENGINE, CONST = 1, [3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14]
DATA, HARNESS, EVAL = [18, 19, 20], [22], 28

_eng = '\n'.join(_s(i) for i in ENGINE)
for _y in ('def build_features', 'def label_bars', 'def _w_hac_contrast',
           'def composite_subset', 'def direction_weight', 'def n_params'):
    assert _y in _eng, f'engine harvest lost {_y}'
_con = '\n'.join(_s(i) for i in CONST)
for _y in ('def windows_for', 'def bars_per_day', 'def metric_S',
           'def evaluate_guards', 'FIXED_KNOBS', 'def assert_knobs_unchanged'):
    assert _y in _con, f'constants harvest lost {_y}'
_dat = '\n'.join(_s(i) for i in DATA)
for _y in ('def load_fx_h1', 'def detect_divisor', 'def realised_vol_proxy',
           'FX_PAIRS', 'SCALE_CANDIDATES'):
    assert _y in _dat, f'data harvest lost {_y}'
assert 'def prepare_run' in _s(HARNESS[0]) and 'def label_run' in _s(HARNESS[0])

_ev = _s(EVAL)
_CUT = '\nt0 = time.time()\nfor r in RUNS:'
assert _CUT in _ev
_ev = _ev.split(_CUT)[0]
assert 'def evaluate_run' in _ev and 'for r in RUNS' not in _ev

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": s.strip('\n').splitlines(keepends=True)})


def co(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip('\n').splitlines(keepends=True)})


def raw(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(keepends=True)})


# ===========================================================================
md(r"""
# Disjoint-block feature set -- does it fix the grind-mode barcode?

Two tied observations drove this notebook:

1. **User, reading the regime-on-price charts**: the detector reads violent
   moves well and "barcodes" through sustained grinds -- rapid colour flips
   where a human sees one coherent trend.
2. **User, on the feature set**: "isn't the parameters too many? ... the
   noise could be too much." Measured: the shipped 9 features carry only
   4.20 effective dimensions (participation ratio); `mom_3d`/`mom_5d`/
   `drawdown`/`dist_ma` correlate at 0.71-0.88 -- one dimension measured four
   ways, with a 5-day reach.

Full reasoning, the earlier (wrong) "just add a 60-day rung" idea and why it
was retracted, and the pre-registered predictions: `FX_FEATURESET_PROTOCOL.md`.

**This is not a parameter search.** The block edges (1/5/20/60 days) and the
dropped feature were fixed by measured collinearity structure BEFORE this
notebook ran on any instrument. All 7 instruments are graded together.
""")

raw(_s(PIP))

md("## 1. Engine, constants, metrics, data layer — harvested verbatim\n\n"
   f"From `build_intraday_fx.py` cells {PIP}, {ENGINE}, {CONST}, {DATA}, "
   f"{HARNESS}, {EVAL} (definitions only). `build_features`, `FEATURE_COLS`, "
   "`FEATURE_SIGN`, `FEATURE_MAG`, `DIRECTION_EXCLUDE`, `BAR_DIR_FEATURES` and "
   "`TREND_FEATURE` are the shipped (NESTED) values at this point -- untouched.")

for _i in ENGINE + CONST + DATA + HARNESS:
    raw(_s(_i))
raw(_ev)

# ===========================================================================
md(r"""
## 2. The disjoint feature set — added alongside the shipped one, neither modified in place
""")

co(r"""
# ===========================================================================
# THE DISJOINT-BLOCK FEATURE SET.
#
# d0_1/d1_5/d5_20/d20_60 are NON-OVERLAPPING return blocks built from nested
# cumulative sums by subtraction:
#   d0_1   = cumsum(0->1d)
#   d1_5   = cumsum(0->5d)  - cumsum(0->1d)
#   d5_20  = cumsum(0->20d) - cumsum(0->5d)
#   d20_60 = cumsum(0->60d) - cumsum(0->20d)
# Measured (protocol): max |off-diagonal corr| 0.588 -> 0.036, condition
# number 6.1 -> 1.1, effective dims (participation ratio) 2.74 -> 4.00 of 4,
# at the SAME 4 horizons -- disjointness, not the horizons, buys the rank.
#
# vol_2h, vix_chg, dist_ma are UNCHANGED formulas (same windows). drawdown is
# DROPPED (-0.86 corr with dist_ma -- keep one, not both). ret_2h is dropped
# (subsumed by d0_1, which spans the same first-day window).
#
# TREND_FEATURE ('mom_3d') is still produced, at ITS ORIGINAL 3-day window,
# for the INTENSITY (H vs L) gate ONLY -- deliberately NOT one of the 7 HMM
# inputs. That axis already works (E1 in the diagnostic probes: W falls
# monotonically as eff_fast rises) and is left untouched. One mechanism at a
# time.
# ===========================================================================
DISJOINT_EDGE_DAYS = dict(E1=1, E5=5, E20=20, E60=60)


def build_features_disjoint(close, vix, scale=LOOKBACK_SCALE):
    '''7 causal features: near-orthogonal disjoint return blocks + vol_2h +
    vix_chg + dist_ma. mom_3d is also produced, for TREND_FEATURE only.'''
    bpd, cd = BARS_PER_DAY, CONTEXT_DAYS
    e = {k: max(2, int(round(d * bpd * scale)))
         for k, d in DISJOINT_EDGE_DAYS.items()}
    swing = max(2, int(round(cd * bpd * scale)))
    mom3_w = max(2, int(round(3 * bpd * scale)))
    vol_w = max(2, int(round(10 * scale)))
    r = np.log(close / close.shift(1))
    cum1, cum5 = r.rolling(e['E1']).sum(), r.rolling(e['E5']).sum()
    cum20, cum60 = r.rolling(e['E20']).sum(), r.rolling(e['E60']).sum()
    df = pd.DataFrame(index=close.index)
    df['d0_1'] = cum1
    df['d1_5'] = cum5 - cum1
    df['d5_20'] = cum20 - cum5
    df['d20_60'] = cum60 - cum20
    df['vol_2h'] = r.rolling(vol_w).std()
    df['vix_chg'] = (vix - vix.shift(1)) / vix.shift(1)
    ma = close.rolling(swing).mean()
    df['dist_ma'] = (close - ma) / ma
    df['mom_3d'] = r.rolling(mom3_w).sum()      # TREND_FEATURE only
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


DISJOINT_FEATURE_COLS = ['d0_1', 'd1_5', 'd5_20', 'd20_60', 'vol_2h',
                         'vix_chg', 'dist_ma']
DISJOINT_FEATURE_SIGN = {'d0_1': 1.0, 'd1_5': 1.0, 'd5_20': 1.0,
                         'd20_60': 1.0, 'vol_2h': -1.0, 'vix_chg': -1.0,
                         'dist_ma': 1.0}
DISJOINT_FEATURE_MAG = {'d0_1': 0.4, 'd1_5': 0.4, 'd5_20': 0.4, 'd20_60': 0.4}
DISJOINT_DIRECTION_EXCLUDE = ('vol_2h',)
DISJOINT_BAR_DIR_FEATURES = ('d0_1', 'd1_5', 'd5_20', 'd20_60', 'dist_ma')

# n_params arithmetic -- F4, trivially confirms, printed for the record.
_p_nested = n_params(5, len(FEATURE_COLS), 'diag')
_p_disjoint = n_params(5, len(DISJOINT_FEATURE_COLS), 'diag')
print(f'F4  n_params(N=5, diag): nested {len(FEATURE_COLS)} feat -> '
      f'{_p_nested:.0f} params   disjoint {len(DISJOINT_FEATURE_COLS)} feat '
      f'-> {_p_disjoint:.0f} params')
assert _p_disjoint < _p_nested
print(f'F4  CONFIRMED: {_p_disjoint:.0f} < {_p_nested:.0f}')

DESIGNS = {
    'nested': dict(FEATURE_COLS=list(FEATURE_COLS), FEATURE_SIGN=dict(FEATURE_SIGN),
                   FEATURE_MAG=dict(FEATURE_MAG),
                   DIRECTION_EXCLUDE=tuple(DIRECTION_EXCLUDE),
                   BAR_DIR_FEATURES=tuple(BAR_DIR_FEATURES),
                   build_features=build_features),
    'disjoint': dict(FEATURE_COLS=list(DISJOINT_FEATURE_COLS),
                     FEATURE_SIGN=dict(DISJOINT_FEATURE_SIGN),
                     FEATURE_MAG=dict(DISJOINT_FEATURE_MAG),
                     DIRECTION_EXCLUDE=tuple(DISJOINT_DIRECTION_EXCLUDE),
                     BAR_DIR_FEATURES=tuple(DISJOINT_BAR_DIR_FEATURES),
                     build_features=build_features_disjoint),
}
SWEPT_KEYS = tuple(DESIGNS['nested'].keys())


@contextlib.contextmanager
def feature_design(name):
    '''Install one feature-set arm. Re-arms the fixed-knob invariant around
    exactly the keys the design touches; every other knob stays guarded.'''
    g = globals()
    spec = DESIGNS[name]
    old = {k: g[k] for k in SWEPT_KEYS}
    old_fixed = dict(FIXED_KNOBS)
    g.update(spec)
    FIXED_KNOBS['FEATURE_COLS'] = tuple(spec['FEATURE_COLS'])
    FIXED_KNOBS['N_FEATURES'] = len(spec['FEATURE_COLS'])
    FIXED_KNOBS['DIRECTION_EXCLUDE'] = tuple(spec['DIRECTION_EXCLUDE'])
    try:
        yield name
    finally:
        g.update(old)
        FIXED_KNOBS.clear()
        FIXED_KNOBS.update(old_fixed)


print(f"\nDESIGNS installed: {list(DESIGNS)} -- swept keys: {SWEPT_KEYS}")
print(f"nested   FEATURE_COLS ({len(DESIGNS['nested']['FEATURE_COLS'])}): "
      f"{DESIGNS['nested']['FEATURE_COLS']}")
print(f"disjoint FEATURE_COLS ({len(DESIGNS['disjoint']['FEATURE_COLS'])}): "
      f"{DESIGNS['disjoint']['FEATURE_COLS']}")
""")

# ===========================================================================
md(r"""
## 3. Bit-for-bit equivalence: `feature_design('nested')` must be a true no-op
""")

co(r"""
# ===========================================================================
# Every new switch needs an OFF value asserted bit-for-bit against the
# unmodified engine. Run EUR/USD@1h WITHOUT the context manager (baseline,
# harvested engine exactly as build_intraday_fx.py left it), then again
# INSIDE feature_design('nested'), and diff every label.
# ===========================================================================
assert_knobs_unchanged('before feature_design exists')
FEATURE_COLS_BASELINE = list(FEATURE_COLS)

_probe_close = FX_H1['EURUSD']['close'].astype(float)
_probe_close.name = 'EURUSD'
with bars_per_day(24):
    _probe_vol = realised_vol_proxy(_probe_close, 24)

_base_prep = prepare_run(_probe_close, _probe_vol, 24)
_base_labels, _ = label_run(_base_prep, _probe_close, 24)

with feature_design('nested'):
    assert_knobs_unchanged("inside feature_design('nested')")
    _nest_prep = prepare_run(_probe_close, _probe_vol, 24)
    _nest_labels, _ = label_run(_nest_prep, _probe_close, 24)
assert_knobs_unchanged("after feature_design('nested')")

_diff = sum(int((a != b).sum()) for a, b in zip(_base_labels, _nest_labels))
_total = sum(len(a) for a in _base_labels)
print(f"EUR/USD@1h baseline vs feature_design('nested'): {_diff} of {_total} "
      f"labels differ across {len(_base_labels)} seed sets")
assert _diff == 0, "feature_design('nested') is NOT a bit-for-bit no-op"
print("ASSERT OK: feature_design('nested') reproduces the unmodified engine "
      "exactly.")

with feature_design('disjoint'):
    assert CONTEXT_DAYS == 12
    assert windows_for(24)['SWING_WIN'] == 12 * 24, \
        'CONTEXT_DAYS did not reach the disjoint dist_ma window'
    _dj_test = prepare_run(_probe_close, _probe_vol, 24)
    assert 'd0_1' in _dj_test['feat'].columns and 'mom_3d' in _dj_test['feat'].columns
    assert list(_dj_test['feat'][FEATURE_COLS].columns) == DISJOINT_FEATURE_COLS
print("ASSERT OK: feature_design('disjoint') installs, produces d0_1..d20_60 "
      "+ mom_3d (TREND_FEATURE only), and restores cleanly.")
assert FEATURE_COLS == list(FEATURE_COLS_BASELINE), \
    'feature_design failed to restore FEATURE_COLS after the disjoint arm'
del _base_prep, _base_labels, _nest_prep, _nest_labels, _dj_test
""")

# ===========================================================================
md(r"""
## 4. The instrument set — all 7, graded together (not a search)
""")

co(r"""
# ===========================================================================
# ALL 7 non-redundant instruments in the repo (same set fx_replication.ipynb
# assembled: crosses and 2h excluded for their measured pseudo-replication
# reasons). No discovery/held-out split -- see the protocol for why.
# ===========================================================================
ALL_INST = [
    dict(sym='EURUSD', name='EUR/USD', med_band=(0.9, 1.7),     cls='FX'),
    dict(sym='GBPUSD', name='GBP/USD', med_band=(0.9, 1.7),     cls='FX'),
    dict(sym='USDJPY', name='USD/JPY', med_band=(75.0, 160.0),  cls='FX'),
    dict(sym='XAUUSD', name='XAU/USD', med_band=(1000., 2100.), cls='COMMODITY'),
    dict(sym='AUDUSD', name='AUD/USD', med_band=(0.55, 1.15),   cls='FX'),
    dict(sym='USDCAD', name='USD/CAD', med_band=(0.9, 1.6),     cls='FX'),
    dict(sym='USDCHF', name='USD/CHF', med_band=(0.7, 1.15),    cls='FX'),
]
BPD_1H, TF_NAME = 24, '1h'

FS_H1, FS_DIV = {}, {}
_t0 = time.time()
for d in ALL_INST:
    FS_H1[d['sym']], FS_DIV[d['sym']] = load_fx_h1(d)
print(f'fetched {len(ALL_INST)} hourly series in {time.time() - _t0:.1f}s')
print(f"\n{'instrument':<11}{'class':<12}{'divisor':>9}{'bars':>8}  span"
      f"{'':<18}{'median':>10}")
for d in ALL_INST:
    b = FS_H1[d['sym']]
    print(f"{d['name']:<11}{d['cls']:<12}{FS_DIV[d['sym']]:>9g}{len(b):>8}  "
          f"{b.index[0].date()} -> {b.index[-1].date()}"
          f"{b['close'].median():>10.4f}")
for d in ALL_INST:
    b = FS_H1[d['sym']]
    assert b.index.is_monotonic_increasing and not b.index.duplicated().any()
    assert (b[['open', 'high', 'low', 'close']] > 0).all().all()
print('\nASSERT OK: all 7 series monotone, de-duplicated, positive.')
print('*** ALL SERIES END 2022-03-04. Nothing here speaks to 2022-2026. ***')
""")

co(r"""
# ===========================================================================
# PRE-REGISTERED, frozen in protocols/FX_FEATURESET_PROTOCOL.md.
# ===========================================================================
F_PREDICTIONS = {
 'F1': ('PRIMARY. In the "steady trend, low eff_fast / high eff_slow" cell, '
        'W is LOWER for disjoint than nested on >= 4 of 7 instruments AND on '
        'the pooled mean.',
        'A pooled drop with the majority failing is reported as a split, '
        'not a confirmation.'),
 'F2': ('Disjoint still shows W falling monotonically Q1->Q5 of eff_fast, '
        'pooled across instruments -- the violent-move axis is not broken.',
        'CONFIRMED if the pooled Q1->Q5 mean W is monotone decreasing.'),
 'F3': ('Disjoint has a LOWER feature-correlation condition number and a '
        'HIGHER participation-ratio effective-dimension share than nested, '
        'measured live, on >= 4 of 7 instruments.',
        'Not assumed from the earlier AUD/USD-only probe.'),
 'F4': ('n_params(disjoint) < n_params(nested) at N=5, diag.',
        'Arithmetic; already shown above.'),
 'F5': ('Guards G1-G4 pass on disjoint on AT LEAST as many of the 7 runs as '
        'on nested.',
        'A coherence win that breaks a guard is not reported as a win.'),
 'F6': ('Mean seed-stability S is higher for disjoint than nested, pooled '
        'across the 7 instruments.',
        'The bistability hypothesis PROJECT_STATE has not resolved.'),
 'F7': ('HONESTY CHECK, not the goal. Disjoint OOS (BULL-BEAR) HAC t is not '
        'systematically worse than nested (mean difference >= -0.20).',
        'Not expected to manufacture skill fx_meanrev.ipynb already found '
        'absent. Does not gate F1-F6.'),
}
print('=' * 108)
print('PRE-REGISTERED PREDICTIONS -- printed before a single run exists')
print('=' * 108)
for k, (claim, rule) in F_PREDICTIONS.items():
    print(f'{k}.  {claim}')
    print(f'    RULE: {rule}')
""")

# ===========================================================================
md(r"""
## 5. Runtime measurement — measured first, then extrapolated, then run
""")

co(r"""
# ===========================================================================
# RUNTIME PROBE. One seed set, NESTED design (9 features, more than disjoint's
# 7 -- an upper bound on per-fit cost), on the largest instrument.
# ===========================================================================
TRAIN_CAP_BARS = None
RUNTIME_EST = None

_p0 = ALL_INST[0]
_c0 = FS_H1[_p0['sym']]['close'].astype(float)
_c0.name = _p0['sym']
with bars_per_day(BPD_1H):
    _v0 = realised_vol_proxy(_c0, BPD_1H)

with feature_design('nested'):
    _t = time.time()
    _prep0 = prepare_run(_c0, _v0, BPD_1H)
    _t_prep = time.time() - _t
    _t = time.time()
    _m0, _ = fit_hmm_ensemble(_prep0['Xs'][_prep0['fit_lo']:_prep0['n_fit']],
                              N_STATES_FIXED, COV_FIXED, K=ENSEMBLE_K,
                              base_seed=SEED_SETS[0][0])
    _t_set = time.time() - _t
    FIT_COUNT += ENSEMBLE_K
    _t = time.time()
    with bars_per_day(BPD_1H):
        _l0 = label_bars(_m0, _prep0['Xs'], _prep0['dates'], _c0,
                         _prep0['trend_raw'], _prep0['n_fit'], FEATURE_COLS,
                         dir_feats=_prep0['feat'])
    _t_lab = time.time() - _t

_per_run = R_SEED_SETS * (_t_set + _t_lab) + _t_prep
N_RUNS_TOTAL = len(DESIGNS) * len(ALL_INST)
RUNTIME_EST = N_RUNS_TOTAL * _per_run

print('=' * 100)
print('RUNTIME PROBE -- measured before anything is launched')
print('=' * 100)
print(f"  probe run              : {_p0['name']} @ 1h, nested design")
print(f"  feature bars           : {_prep0['n_bars']}   training rows: "
      f"{_prep0['n_fit'] - _prep0['fit_lo']}")
print(f'  prepare_run            : {_t_prep:.1f}s')
print(f'  ONE seed set (K={ENSEMBLE_K} fits): {_t_set:.1f}s')
print(f'  label_bars full series : {_t_lab:.1f}s')
print(f'  projected per run      : {_per_run:6.0f}s  (R={R_SEED_SETS} seed sets)')
print(f'  {N_RUNS_TOTAL} runs ({len(DESIGNS)} designs x {len(ALL_INST)} '
      f'instruments) -> projected {RUNTIME_EST:.0f}s (~{RUNTIME_EST / 60:.1f} min)')

RUNTIME_BUDGET_S = 25 * 60
if RUNTIME_EST > RUNTIME_BUDGET_S:
    TRAIN_CAP_BARS = 12000     # declared constant -- NEVER wall-clock-derived
    print(f'\nOVER BUDGET ({RUNTIME_BUDGET_S}s) -- capping the training window '
          f'to the most recent {TRAIN_CAP_BARS} bars. R=4 stays intact; no '
          'instrument or design is dropped.')
else:
    print(f'\nUnder the {RUNTIME_BUDGET_S}s budget -- no cap applied. Every run '
          'uses its full leading training window.')
""")

# ===========================================================================
md(r"""
## 6. Label phase — 2 designs × 7 instruments = 14 runs
""")

co(r"""
# ===========================================================================
# LABEL PHASE. Every run: (design, instrument) -> prepare_run + label_run
# inside feature_design(design). CONTEXT_DAYS=12 for BOTH designs -- the only
# thing that changes is feature construction.
# ===========================================================================
FEATSET_RUNS = []
_t0 = time.time()
print(f"{'design':<10}{'instrument':<11}{'bars':>8}{'fit rows':>10}{'secs':>8}")
print('-' * 50)
for design in DESIGNS:
    for d in ALL_INST:
        _tt = time.time()
        close = FS_H1[d['sym']]['close'].astype(float)
        close.name = d['sym']
        with feature_design(design):
            with bars_per_day(BPD_1H):
                vol = realised_vol_proxy(close, BPD_1H)
            prep = prepare_run(close, vol, BPD_1H)
            labels, _m = label_run(prep, close, BPD_1H)
            assert_knobs_unchanged(f"{d['name']} {design}")
        FEATSET_RUNS.append(dict(
            run_id=f"{d['name']}|{design}", inst=d['name'], design=design,
            cls=d['cls'], tf=TF_NAME, error=None, labels=labels,
            close_arr=prep['close'].values, dates=prep['dates'],
            feat=prep['feat'][list(DESIGNS[design]['FEATURE_COLS'])].copy(),
            fwd_h=fwd_horizons_for(BPD_1H),
            n_bars=prep['n_bars'], n_fit=prep['n_fit']))
        print(f"{design:<10}{d['name']:<11}{prep['n_bars']:>8}"
              f"{prep['n_fit']:>10}{time.time() - _tt:>8.1f}")
print('-' * 50)
print(f'LABEL PHASE: {time.time() - _t0:.1f}s   {len(FEATSET_RUNS)} runs   '
      f'{FIT_COUNT} HMM fits total (incl. probes)   TRAIN_CAP_BARS = {TRAIN_CAP_BARS}')
assert ZZ_CALLS == 0, 'LEAKAGE: a ZigZag existed during the label phase'
print('ASSERT OK: ZZ_CALLS == 0 -- every label produced before any ZigZag.')
""")

# ===========================================================================
md(r"""
## 7. Eval phase — S/L/W/guards/IS-OOS (verbatim), plus the grind-coherence metrics
""")

co(r"""
# ===========================================================================
# EVAL PHASE -- strictly after the whole label phase.
# ===========================================================================
LABEL_PHASE_OPEN = False
for r in FEATSET_RUNS:
    evaluate_run(r)
print(f'evaluated {len(FEATSET_RUNS)} runs   ZZ_CALLS={ZZ_CALLS}')
""")

co(r"""
# ===========================================================================
# GRIND-COHERENCE METRICS (F1/F2) -- same eff_fast/eff_slow construction
# already used (and pre-registered) in the diagnostic probes: eff_fast over
# EFF_WIN=9 bars (what the chop filter sees), eff_slow over 20 DAYS (the
# trend a human sees on the chart). Uses labels[0] and the OOS span only,
# same convention as ord_is/ord_oos/means_is/means_oos already use in
# evaluate_run.
# ===========================================================================
EFF_FAST_WIN = EFF_WIN            # 9 bars, the shipped chop-filter horizon
EFF_SLOW_WIN = 20 * BPD_1H        # 20 days


def efficiency(px, w):
    lp = np.log(px)
    d = np.abs(np.diff(lp, prepend=lp[0]))
    net = np.full(len(px), np.nan)
    net[w:] = np.abs(lp[w:] - lp[:-w])
    tot = np.convolve(d, np.ones(w), 'full')[:len(px)]
    with np.errstate(invalid='ignore', divide='ignore'):
        e = net / tot
    return np.clip(e, 0, 1)


def feature_conditioning(X):
    '''(condition number, participation-ratio effective dims) of a raw
    feature matrix -- correlation eigenvalues, no NaNs.'''
    C = np.corrcoef(X, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    cond = float(ev.max() / ev.min())
    eff_dims = float(ev.sum() ** 2 / (ev ** 2).sum())
    return cond, eff_dims


MIN_CELL_BARS = 300
for r in FEATSET_RUNS:
    lab0 = np.asarray(r['labels'][0], object)
    px = r['close_arr']
    lo = r['n_fit']
    ef, es = efficiency(px, EFF_FAST_WIN), efficiency(px, EFF_SLOW_WIN)
    sw = np.zeros(len(lab0))
    sw[1:] = (lab0[1:] != lab0[:-1]).astype(float)
    base = np.zeros(len(px), bool)
    base[lo:] = True
    base &= np.isfinite(ef) & np.isfinite(es)
    mf, ms_ = np.nanmedian(ef[lo:]), np.nanmedian(es[lo:])
    grind = base & (ef < mf) & (es >= ms_)
    r['W_grind'] = (float(sw[grind].mean() * 100) if grind.sum() >= MIN_CELL_BARS
                    else np.nan)
    r['n_grind'] = int(grind.sum())

    q = np.nanquantile(ef[lo:], [0.2, 0.4, 0.6, 0.8])
    r['W_quintile'] = []
    for b in range(5):
        band = (ef >= (q[b - 1] if b else -1)) & (ef < (q[b] if b < 4 else 2))
        m = base & band
        r['W_quintile'].append(float(sw[m].mean() * 100) if m.sum() >= 100
                                else np.nan)

    Xoos = r['feat'].values[lo:]
    Xoos = Xoos[np.isfinite(Xoos).all(axis=1)]
    r['cond'], r['eff_dims'] = feature_conditioning(Xoos)
    r['eff_share'] = r['eff_dims'] / Xoos.shape[1]

print(f'grind-cell coherence + conditioning computed for {len(FEATSET_RUNS)} '
      'runs (OOS span, seed set 0, EFF_FAST=9 bars / EFF_SLOW=20 days).')
""")

# ===========================================================================
md(r"""
## 8. Results — nested vs disjoint, side by side per instrument, never pooled
""")

co(r"""
# ===========================================================================
# MAIN TABLE. Every run appears. Nothing dropped for looking bad.
# ===========================================================================
def guard_str(r):
    return ''.join(('.' if r['guards'][k][0] else 'X') for k in ('G1', 'G2', 'G3', 'G4'))


def get(inst, design):
    return next(r for r in FEATSET_RUNS if r['inst'] == inst and r['design'] == design)


print('=' * 150)
print('FEATURE-SET COMPARISON -- 7 instruments x {nested, disjoint}, 1h, CONTEXT_DAYS=12 for both')
print('=' * 150)
print(f"{'instrument':<10}{'design':<10}{'S':>6}{'L':>7}{'W':>7}{'guards':>8}"
      f"{'W_grind':>9}{'n_grind':>9}{'cond#':>8}{'eff_dims':>9}{'OOS t':>8}")
print('-' * 150)
for d in ALL_INST:
    for design in DESIGNS:
        r = get(d['name'], design)
        print(f"{d['name']:<10}{design:<10}{r['S']:>6.2f}{r['L']:>7.1f}"
              f"{r['W']:>7.2f}{guard_str(r):>8}{r['W_grind']:>9.2f}"
              f"{r['n_grind']:>9}{r['cond']:>8.2f}{r['eff_dims']:>9.2f}"
              f"{r['OOS_T']:>8.2f}")
    print('-' * 150)

_gf = [r for r in FEATSET_RUNS if not r['guards_ok']]
print(f'guards: {len(FEATSET_RUNS) - len(_gf)} of {len(FEATSET_RUNS)} runs pass '
      'all four'
      + ('' if not _gf else '  FAILING: '
         + ', '.join(f"{r['inst']}|{r['design']}" for r in _gf)))
""")

co(r"""
# ===========================================================================
# QUINTILE TABLE (F2) -- W by eff_fast quintile, nested vs disjoint, per
# instrument and pooled.
# ===========================================================================
print('=' * 130)
print('WHIPSAW W BY eff_fast QUINTILE (Q1=choppiest bars by the 9-bar filter, '
      'Q5=most efficient/violent)')
print('=' * 130)
print(f"{'instrument':<10}{'design':<10}" + ''.join(f"{'Q' + str(i):>9}" for i in range(1, 6)))
print('-' * 130)
POOLED_Q = {design: [[] for _ in range(5)] for design in DESIGNS}
for d in ALL_INST:
    for design in DESIGNS:
        r = get(d['name'], design)
        print(f"{d['name']:<10}{design:<10}"
              + ''.join(f"{v:>9.2f}" if np.isfinite(v) else f"{'n/a':>9}"
                        for v in r['W_quintile']))
        for i, v in enumerate(r['W_quintile']):
            if np.isfinite(v):
                POOLED_Q[design][i].append(v)
print('-' * 130)
POOLED_Q_MEAN = {design: [float(np.mean(q)) if q else np.nan for q in POOLED_Q[design]]
                 for design in DESIGNS}
for design in DESIGNS:
    print(f"{'POOLED MEAN':<10}{design:<10}"
          + ''.join(f"{v:>9.2f}" for v in POOLED_Q_MEAN[design]))
""")

# ===========================================================================
md(r"""
## 9. Figures — look at these before trusting any table above
""")

co(r"""
# ===========================================================================
# FIGURE 1 -- regime-on-price, nested vs disjoint, OOS span, 3 instruments
# that showed the barcode most clearly in the earlier diagnostic probes
# (AUD/USD, USD/CAD) plus EUR/USD as a violent-move sanity check.
# ===========================================================================
import matplotlib.pyplot as plt

FIG1_INST = ['AUD/USD', 'USD/CAD', 'EUR/USD']
fig, axes = plt.subplots(len(FIG1_INST), 2, figsize=(15, 11), sharex='row')
for i, inst in enumerate(FIG1_INST):
    for j, design in enumerate(DESIGNS):
        ax = axes[i][j]
        r = get(inst, design)
        lo = r['n_fit']
        dates, px, lab = r['dates'][lo:], r['close_arr'][lo:], np.asarray(r['labels'][0], object)[lo:]
        shade_regimes(ax, pd.Series(lab, index=dates), alpha=0.55)
        ax.plot(dates, px, color='black', lw=0.7)
        ax.set_xlim(dates[0], dates[-1])
        ax.margins(y=0.05)
        occ = {k: float(np.mean(lab == k)) * 100 for k in REGIME_LABELS}
        ax.set_title(f"{inst}  [{design}]   W {r['W']:.1f}/100   "
                     f"W_grind {r['W_grind']:.1f}/100 (n={r['n_grind']})",
                     fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        if j == 0:
            ax.set_ylabel('price', fontsize=8)
handles = [plt.matplotlib.patches.Patch(color=REGIME_COLORS[k], alpha=0.7, label=k)
           for k in REGIME_LABELS]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.005))
fig.suptitle('NESTED (shipped, left) vs DISJOINT (right) -- out-of-sample span '
             'only, CONTEXT_DAYS=12 both, one seed set (for drawing)',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0.035, 1, 0.955])
plt.show()
""")

co(r"""
# ===========================================================================
# FIGURE 2 -- F1/F2 evidence: quintile W curve (line) + grind-cell W (bars).
# ===========================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5))

for design, style in DESIGNS.items():
    axL.plot(range(1, 6), POOLED_Q_MEAN[design], marker='o',
             label=design, lw=2.2)
axL.set_xticks(range(1, 6))
axL.set_xticklabels([f'Q{i}' for i in range(1, 6)])
axL.set_xlabel('eff_fast quintile (Q1=choppiest -> Q5=most violent)')
axL.set_ylabel('mean whipsaw W (switches / 100 bars)')
axL.set_title('F2: whipsaw by move-quality quintile, pooled over 7 instruments')
axL.legend()
axL.grid(alpha=0.3)

_x = np.arange(len(ALL_INST))
_w = 0.36
for k, design in enumerate(DESIGNS):
    vals = [get(d['name'], design)['W_grind'] for d in ALL_INST]
    axR.bar(_x + (k - 0.5) * _w, vals, width=_w, label=design)
axR.set_xticks(_x)
axR.set_xticklabels([d['name'] for d in ALL_INST], rotation=30, ha='right')
axR.set_ylabel('W_grind (switches / 100 bars, "steady trend with wiggles" cell)')
axR.set_title('F1: whipsaw in the exact cell the user flagged by eye')
axR.legend()
axR.grid(alpha=0.3, axis='y')

fig.suptitle('Does the disjoint feature set reduce whipsaw where the user '
             'said it should, without breaking where it already worked?',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
""")

# ===========================================================================
md(r"""
## 10. Grading F1–F7 — against the rules frozen in the protocol, before any number existed
""")

co(r"""
# ===========================================================================
# GRADING -- verbatim against FX_FEATURESET_PROTOCOL.md. No threshold moved.
# ===========================================================================
print('=' * 108)
print('GRADING THE PRE-REGISTERED PREDICTIONS')
print('=' * 108)

# ---- F1: primary, the user's visual complaint --------------------------
_f1_pairs = [(d['name'], get(d['name'], 'nested')['W_grind'],
             get(d['name'], 'disjoint')['W_grind']) for d in ALL_INST]
_f1_def = [(n, wn, wd) for n, wn, wd in _f1_pairs if np.isfinite(wn) and np.isfinite(wd)]
_f1_wins = [n for n, wn, wd in _f1_def if wd < wn]
_f1_pool_nested = float(np.mean([wn for _, wn, _ in _f1_def])) if _f1_def else np.nan
_f1_pool_disjoint = float(np.mean([wd for _, _, wd in _f1_def])) if _f1_def else np.nan
print(f"\nF1  grind-cell W, per instrument (nested -> disjoint):")
for n, wn, wd in _f1_pairs:
    tag = ('disjoint LOWER' if (np.isfinite(wn) and np.isfinite(wd) and wd < wn)
           else ('disjoint HIGHER/EQUAL' if np.isfinite(wn) and np.isfinite(wd)
                 else 'UNDEFINED (< min cell bars)'))
    print(f"    {n:<10} {wn:>7.2f} -> {wd:>7.2f}   {tag}")
_f1_majority = len(_f1_wins) >= 4
_f1_pooled = np.isfinite(_f1_pool_disjoint) and _f1_pool_disjoint < _f1_pool_nested
print(f"    pooled mean: {_f1_pool_nested:.2f} -> {_f1_pool_disjoint:.2f}   "
      f"majority: {len(_f1_wins)} of {len(_f1_def)} defined")
if _f1_majority and _f1_pooled:
    print("F1  CONFIRMED")
elif _f1_pooled or _f1_majority:
    print(f"F1  SPLIT (not a confirmation) -- pooled {'moved' if _f1_pooled else 'did not move'} "
          f"the right way, majority {'held' if _f1_majority else 'did not hold'}")
else:
    print("F1  REFUTED")

# ---- F2: do-no-harm on the violent-move axis ----------------------------
_q_dj = POOLED_Q_MEAN['disjoint']
_f2_monotone = all(_q_dj[i] >= _q_dj[i + 1] for i in range(4)
                    if np.isfinite(_q_dj[i]) and np.isfinite(_q_dj[i + 1]))
print(f"\nF2  disjoint pooled W by quintile: "
      f"{['%.2f' % v if np.isfinite(v) else 'n/a' for v in _q_dj]}")
print(f"F2  {'CONFIRMED' if _f2_monotone else 'REFUTED'}")

# ---- F3: conditioning, measured live ------------------------------------
_f3_hits = 0
print(f"\nF3  conditioning per instrument (nested -> disjoint):")
for d in ALL_INST:
    rn, rd = get(d['name'], 'nested'), get(d['name'], 'disjoint')
    hit = rd['cond'] < rn['cond'] and rd['eff_share'] > rn['eff_share']
    _f3_hits += int(hit)
    print(f"    {d['name']:<10} cond {rn['cond']:>7.2f}->{rd['cond']:>7.2f}   "
          f"eff_share {rn['eff_share']:.2f}->{rd['eff_share']:.2f}   "
          f"{'BOTH IMPROVE' if hit else 'NO'}")
print(f"F3  {_f3_hits} of {len(ALL_INST)} instruments both-improve -- "
      f"{'CONFIRMED' if _f3_hits >= 4 else 'REFUTED'}")

# ---- F4: arithmetic, already shown ---------------------------------------
print(f"\nF4  n_params: nested {_p_nested:.0f} > disjoint {_p_disjoint:.0f}   "
      f"CONFIRMED")

# ---- F5: do-no-harm, structural -------------------------------------------
_nested_pass = sum(1 for d in ALL_INST if get(d['name'], 'nested')['guards_ok'])
_disjoint_pass = sum(1 for d in ALL_INST if get(d['name'], 'disjoint')['guards_ok'])
print(f"\nF5  guards pass: nested {_nested_pass}/{len(ALL_INST)}   "
      f"disjoint {_disjoint_pass}/{len(ALL_INST)}   "
      f"{'CONFIRMED' if _disjoint_pass >= _nested_pass else 'REFUTED'}")

# ---- F6: bistability / seed-stability -------------------------------------
_s_nested = float(np.mean([get(d['name'], 'nested')['S'] for d in ALL_INST]))
_s_disjoint = float(np.mean([get(d['name'], 'disjoint')['S'] for d in ALL_INST]))
print(f"\nF6  mean S: nested {_s_nested:.3f}   disjoint {_s_disjoint:.3f}   "
      f"{'CONFIRMED' if _s_disjoint > _s_nested else 'REFUTED'}")

# ---- F7: honesty check, forward-return skill -------------------------------
_dt = [get(d['name'], 'disjoint')['OOS_T'] - get(d['name'], 'nested')['OOS_T']
       for d in ALL_INST
       if np.isfinite(get(d['name'], 'disjoint')['OOS_T'])
       and np.isfinite(get(d['name'], 'nested')['OOS_T'])]
_mean_dt = float(np.mean(_dt)) if _dt else np.nan
print(f"\nF7  mean OOS HAC-t (disjoint - nested): {_mean_dt:+.3f}   "
      f"(n={len(_dt)} of {len(ALL_INST)} defined)")
print(f"F7  {'CONFIRMED' if np.isfinite(_mean_dt) and _mean_dt >= -0.20 else 'REFUTED'}"
      "  -- NOT the goal; does not gate F1-F6 either way.")
print()
print('Reminder: none of the above is a return, a Sharpe, or a tradeability '
      'claim. Data ends 2022-03-04.')
""")

# ===========================================================================
md(r"""
## 11. Hand-off — what a decision to ship this would actually change
""")

co(r"""
# ===========================================================================
# Nothing here is shipped by running this notebook. This block states what
# WOULD change in the master if the grading above supports adopting it.
# ===========================================================================
print('=' * 100)
print('IF ADOPTED, THIS IS THE DIFF FROM THE SHIPPED MASTER (build_master_notebook_v2.py):')
print('=' * 100)
print(f"  FEATURE_COLS      {FEATURE_COLS_BASELINE}")
print(f"                 -> {DISJOINT_FEATURE_COLS}")
print(f"  FEATURE_SIGN/MAG  updated for the 4 renamed momentum features (see cell 2)")
print(f"  DIRECTION_EXCLUDE ('vol_2h','vol_expansion') -> ('vol_2h',)")
print(f"  BAR_DIR_FEATURES  ('ret_2h','mom_1d','mom_3d','mom_5d','dist_ma')")
print(f"                 -> ('d0_1','d1_5','d5_20','d20_60','dist_ma')")
print(f"  build_features()  replaced by build_features_disjoint() (this notebook, cell 2)")
print(f"  TREND_FEATURE     UNCHANGED ('mom_3d') -- INTENSITY axis untouched")
print(f"  n_params(N=5,diag) {_p_nested:.0f} -> {_p_disjoint:.0f}")
print()
print('NOT changed by this notebook: N_STATES, COVARIANCE, BAR_DIR_WEIGHT, '
      'ENSEMBLE_K, CONF_L, CONFIRM_BARS, Z_HI/EFF_HI bands, CONTEXT_DAYS, '
      'TRAIN_FRACTION, N_FOLDS, INTENSITY_MODE, ESCALATION_DURING_HOLD, '
      'DIRECTION_MODE.')
print()
print('This notebook does not modify build_master_notebook_v2.py. Adoption is '
      'a separate, explicit step.')
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'wrote {OUT} - {len(cells)} cells '
      f'({sum(1 for c in cells if c["cell_type"] == "code")} code, '
      f'{sum(1 for c in cells if c["cell_type"] == "markdown")} markdown)')
