"""Build fx_context.ipynb -- does WIDENING CONTEXT_DAYS past the shipped 12
improve grind-mode coherence on real FX?

User hypothesis, from reading the regime chart in their own real-data Nifty
run: "if we just increase the context window by a little bit, it could be
phenomenal."

Protocol frozen in protocols/FX_CONTEXT_WIDEN_PROTOCOL.md BEFORE this file
was written, including the SATURATION GUARD (C2) -- widening to 20 days on
daily Nifty previously collapsed occupancy to 100% bear, so a whipsaw
improvement bought by losing all discrimination must be caught, not
celebrated.

ONE VARIABLE ONLY: the plain shipped 9-feature NESTED set, unmodified. This
does NOT combine with the disjoint feature set from fx_featureset.ipynb.

Setup harvested VERBATIM from build_intraday_fx.py.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(
    os.environ.get('REGDET_OUT_DIR',
                   os.path.join(os.path.dirname(HERE), 'notebooks')),
    'fx_context.ipynb')
FX_GEN = f'{HERE}/build_intraday_fx.py'

_fsrc = open(FX_GEN).read()
assert "'intraday_fx.ipynb'" in _fsrc
_fsrc = _fsrc.replace("'intraday_fx.ipynb'", "'_fx_ctx_harvest.ipynb'", 1)
_ns = {'__name__': '_fx_gen_harvest', '__file__': FX_GEN}
exec(compile(_fsrc, FX_GEN, 'exec'), _ns)
_fcells = _ns['cells']
for _tmp in ('_fx_ctx_harvest.ipynb', '_intraday_fx_engine_harvest.ipynb'):
    try:
        os.remove(os.path.join(os.path.dirname(OUT), _tmp))
    except OSError:
        pass


def _s(i):
    return ''.join(_fcells[i]['source'])


PIP, ENGINE, CONST = 1, [3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14]
DATA, HARNESS, EVAL = [18, 19, 20], [22], 28

_eng = '\n'.join(_s(i) for i in ENGINE)
for _y in ('def build_features', 'def label_bars', 'def _w_hac_contrast'):
    assert _y in _eng, f'engine harvest lost {_y}'
_con = '\n'.join(_s(i) for i in CONST)
for _y in ('def windows_for', 'def bars_per_day', 'def metric_S',
           'def evaluate_guards', 'FIXED_KNOBS', 'def assert_knobs_unchanged'):
    assert _y in _con, f'constants harvest lost {_y}'
_dat = '\n'.join(_s(i) for i in DATA)
for _y in ('def load_fx_h1', 'def detect_divisor', 'def realised_vol_proxy'):
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
# Does widening the context window fix grind-mode coherence?

**The hypothesis, in the user's words**, looking at the regime chart from
their own real-data Nifty run: *"if we just increase the context window by a
little bit, it could be phenomenal."*

`CONTEXT_DAYS` sets `SWING_WIN` and `VOL_SLOW` — the window behind `drawdown`
and `dist_ma`, the two features that carry the detector's sense of *structure*
rather than *recent move*. Shipped value: **12 days**.

### Why this is genuinely open, in both directions

| already measured | direction | result |
|---|---|---|
| `fx_sensitivity.ipynb`, real FX | NARROWING 12→6→3 | raises whipsaw, lowers lag, **buys no skill** |
| original decline-lag fix, real daily Nifty | WIDENING 7→12 | **fixed a real problem** |
| same, pushed further | WIDENING →20 | **bear 100% / bull 0% / side 0% — total saturation** |

So there is a real fix in the widening direction *and* a real collapse a
little further along it. The zone just past 12 is untested. That is exactly
what this sweeps.

**One variable only.** The plain shipped 9-feature nested set, unmodified —
this does not combine with the disjoint feature set. Only `CONTEXT_DAYS`
moves.

Full pre-registration incl. the saturation guard: `FX_CONTEXT_WIDEN_PROTOCOL.md`.
""")

raw(_s(PIP))

md("## 1. Engine, constants, metrics, data layer — harvested verbatim\n\n"
   f"From `build_intraday_fx.py` cells {PIP}, {ENGINE}, {CONST}, {DATA}, "
   f"{HARNESS}, {EVAL} (definitions only). The 9-feature NESTED set is used "
   "throughout, exactly as shipped.")

for _i in ENGINE + CONST + DATA + HARNESS:
    raw(_s(_i))
raw(_ev)

# ===========================================================================
md(r"""
## 2. The instruments, the arms, and the swept-knob invariant
""")

co(r"""
# ===========================================================================
# All 7 non-redundant instruments (same set as fx_replication /
# fx_featureset: crosses and the 2h axis excluded for measured
# pseudo-replication reasons).
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
CONTEXT_ARMS = (12, 18, 24)        # 12 = shipped baseline
BASELINE_ARM = 12

CTX_H1, CTX_DIV = {}, {}
_t0 = time.time()
for d in ALL_INST:
    CTX_H1[d['sym']], CTX_DIV[d['sym']] = load_fx_h1(d)
print(f'fetched {len(ALL_INST)} hourly series in {time.time() - _t0:.1f}s')
for d in ALL_INST:
    b = CTX_H1[d['sym']]
    assert b.index.is_monotonic_increasing and not b.index.duplicated().any()
    assert (b[['open', 'high', 'low', 'close']] > 0).all().all()
print('ASSERT OK: all 7 series monotone, de-duplicated, positive.')
print('*** ALL SERIES END 2022-03-04. Nothing here speaks to 2022-2026. ***')

SWEPT = ('CONTEXT_DAYS',)
GUARDED = tuple(k for k in FIXED_KNOBS if k not in SWEPT)


@contextlib.contextmanager
def ctx_arm(context_days):
    '''Install one lookback arm; re-arm the invariant around the swept knob.'''
    g = globals()
    old, old_fixed = g['CONTEXT_DAYS'], dict(FIXED_KNOBS)
    g['CONTEXT_DAYS'] = context_days
    FIXED_KNOBS['CONTEXT_DAYS'] = context_days
    try:
        yield context_days
    finally:
        g['CONTEXT_DAYS'] = old
        FIXED_KNOBS.clear()
        FIXED_KNOBS.update(old_fixed)


assert_knobs_unchanged('before any arm')
with ctx_arm(24):
    assert windows_for(BPD_1H)['SWING_WIN'] == 24 * BPD_1H, \
        'CONTEXT_DAYS did not reach windows_for -- the arms would be identical'
    assert windows_for(BPD_1H)['VOL_SLOW'] == 24 * BPD_1H
    assert_knobs_unchanged('inside an arm')
assert CONTEXT_DAYS == 12, 'ctx_arm failed to restore'
print(f'ASSERT OK: ctx_arm installs, reaches BOTH SWING_WIN and VOL_SLOW, and '
      f'restores. {len(GUARDED)} knobs stay guarded.')
print(f'FEATURE_COLS ({len(FEATURE_COLS)}, shipped NESTED, unmodified): {FEATURE_COLS}')
""")

co(r"""
# ===========================================================================
# PRE-REGISTERED, frozen in protocols/FX_CONTEXT_WIDEN_PROTOCOL.md.
# ===========================================================================
C_PREDICTIONS = {
 'C1': ('PRIMARY (the user hypothesis). Grind-cell whipsaw is LOWER at '
        'ctx18 and/or ctx24 than at ctx12, on >= 4 of 7 instruments AND on '
        'the pooled mean, for at least one widened arm.',
        'Both arms reported separately -- "a little bit" and "more" are '
        'different claims.'),
 'C2': ('SATURATION GUARD. No single regime exceeds 90% of OOS bars, and all '
        '5 labels stay reachable, at BOTH widened arms.',
        'A REFUTED C2 VOIDS that arm\'s C1 result. Widening to 20 days on '
        'daily Nifty previously gave bear 100% / bull 0% / side 0% -- a '
        'whipsaw win bought by losing all discrimination is not a win.'),
 'C3': ('Guards G1-G4 pass at the widened arms on >= as many instruments as '
        'at ctx12.', 'Do-no-harm, structural.'),
 'C4': ('The eff_fast quintile W curve does not INVERT: pooled Q5 mean W '
        'stays below pooled Q1 mean W at the widened arms.',
        'Do-no-harm on the violent moves the detector already reads well.'),
 'C5': ('HONESTY CHECK, not the goal. OOS (BULL-BEAR) HAC t at the widened '
        'arms is not systematically worse than ctx12 (mean diff >= -0.20).',
        'Not expected to manufacture skill fx_meanrev.ipynb found absent.'),
}
print('=' * 108)
print('PRE-REGISTERED PREDICTIONS -- printed before a single run exists')
print('=' * 108)
for k, (claim, rule) in C_PREDICTIONS.items():
    print(f'{k}.  {claim}')
    print(f'    RULE: {rule}')
""")

# ===========================================================================
md(r"""
## 3. Runtime probe, then the label phase — 3 arms × 7 instruments = 21 runs
""")

co(r"""
# ===========================================================================
# RUNTIME PROBE on the WIDEST arm (longest warmup -> fewest bars, but the
# same fit cost per seed set; measured rather than assumed).
# ===========================================================================
TRAIN_CAP_BARS = None
_p0 = ALL_INST[0]
_c0 = CTX_H1[_p0['sym']]['close'].astype(float)
_c0.name = _p0['sym']
with ctx_arm(BASELINE_ARM):
    with bars_per_day(BPD_1H):
        _v0 = realised_vol_proxy(_c0, BPD_1H)
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
N_RUNS_TOTAL = len(CONTEXT_ARMS) * len(ALL_INST)
RUNTIME_EST = N_RUNS_TOTAL * _per_run
print('=' * 100)
print('RUNTIME PROBE -- measured before anything is launched')
print('=' * 100)
print(f"  probe run          : {_p0['name']} @ 1h, ctx{BASELINE_ARM}")
print(f"  feature bars       : {_prep0['n_bars']}   training rows: "
      f"{_prep0['n_fit'] - _prep0['fit_lo']}")
print(f'  ONE seed set (K={ENSEMBLE_K}): {_t_set:.1f}s   label_bars: {_t_lab:.1f}s')
print(f'  projected per run  : {_per_run:6.0f}s  (R={R_SEED_SETS} seed sets)')
print(f'  {N_RUNS_TOTAL} runs ({len(CONTEXT_ARMS)} arms x {len(ALL_INST)} '
      f'instruments) -> projected {RUNTIME_EST:.0f}s (~{RUNTIME_EST / 60:.1f} min)')

RUNTIME_BUDGET_S = 25 * 60
if RUNTIME_EST > RUNTIME_BUDGET_S:
    TRAIN_CAP_BARS = 12000     # DECLARED CONSTANT -- never wall-clock-derived
    print(f'\nOVER BUDGET ({RUNTIME_BUDGET_S}s) -- capping training to the most '
          f'recent {TRAIN_CAP_BARS} bars (same declared constant as the sibling '
          'sweeps). R=4 intact; no instrument or arm dropped.')
else:
    print(f'\nUnder the {RUNTIME_BUDGET_S}s budget -- no cap applied.')
""")

co(r"""
# ===========================================================================
# LABEL PHASE.
# ===========================================================================
CTX_RUNS = []
_t0 = time.time()
print(f"{'arm':<8}{'instrument':<11}{'bars':>8}{'fit rows':>10}{'secs':>8}")
print('-' * 46)
for cd in CONTEXT_ARMS:
    for d in ALL_INST:
        _tt = time.time()
        close = CTX_H1[d['sym']]['close'].astype(float)
        close.name = d['sym']
        with ctx_arm(cd):
            with bars_per_day(BPD_1H):
                vol = realised_vol_proxy(close, BPD_1H)
            prep = prepare_run(close, vol, BPD_1H)
            labels, _m = label_run(prep, close, BPD_1H)
            assert_knobs_unchanged(f"{d['name']} ctx{cd}")
        CTX_RUNS.append(dict(
            run_id=f"{d['name']}|ctx{cd}", inst=d['name'], ctx=cd,
            cls=d['cls'], tf=TF_NAME, error=None, labels=labels,
            close_arr=prep['close'].values, dates=prep['dates'],
            fwd_h=fwd_horizons_for(BPD_1H),
            n_bars=prep['n_bars'], n_fit=prep['n_fit']))
        print(f"ctx{cd:<5}{d['name']:<11}{prep['n_bars']:>8}"
              f"{prep['n_fit']:>10}{time.time() - _tt:>8.1f}")
print('-' * 46)
print(f'LABEL PHASE: {time.time() - _t0:.1f}s   {len(CTX_RUNS)} runs   '
      f'{FIT_COUNT} HMM fits   TRAIN_CAP_BARS = {TRAIN_CAP_BARS}')
assert ZZ_CALLS == 0, 'LEAKAGE: a ZigZag existed during the label phase'
print('ASSERT OK: ZZ_CALLS == 0 -- every label produced before any ZigZag.')
""")

# ===========================================================================
md(r"""
## 4. Eval phase — including the saturation guard
""")

co(r"""
# ===========================================================================
# EVAL PHASE -- strictly after the whole label phase.
# ===========================================================================
LABEL_PHASE_OPEN = False
for r in CTX_RUNS:
    evaluate_run(r)
print(f'evaluated {len(CTX_RUNS)} runs   ZZ_CALLS={ZZ_CALLS}')
""")

co(r"""
# ===========================================================================
# GRIND-COHERENCE + SATURATION metrics. Same eff_fast/eff_slow construction
# as fx_featureset.ipynb F1, so the numbers are directly comparable.
# ===========================================================================
EFF_FAST_WIN = EFF_WIN            # 9 bars
EFF_SLOW_WIN = 20 * BPD_1H        # 20 days
MIN_CELL_BARS = 300


def efficiency(px, w):
    lp = np.log(px)
    d = np.abs(np.diff(lp, prepend=lp[0]))
    net = np.full(len(px), np.nan)
    net[w:] = np.abs(lp[w:] - lp[:-w])
    tot = np.convolve(d, np.ones(w), 'full')[:len(px)]
    with np.errstate(invalid='ignore', divide='ignore'):
        e = net / tot
    return np.clip(e, 0, 1)


for r in CTX_RUNS:
    lab0 = np.asarray(r['labels'][0], object)
    px, lo = r['close_arr'], r['n_fit']
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

    # --- C2 SATURATION on the OOS span, the span the charts show -----------
    oos = lab0[lo:]
    occ_oos = {k: float(np.mean(oos == k)) * 100 for k in REGIME_LABELS}
    r['occ_oos'] = occ_oos
    r['max_occ'] = max(occ_oos.values())
    r['dominant'] = max(occ_oos, key=occ_oos.get)
    r['n_reachable'] = sum(1 for v in occ_oos.values() if v > 0)
    r['saturated'] = bool(r['max_occ'] > 90.0 or r['n_reachable'] < 5)

print(f'grind coherence + saturation computed for {len(CTX_RUNS)} runs.')
""")

# ===========================================================================
md(r"""
## 5. Results — every arm, every instrument, side by side
""")

co(r"""
# ===========================================================================
# MAIN TABLE.
# ===========================================================================
def guard_str(r):
    return ''.join(('.' if r['guards'][k][0] else 'X') for k in ('G1', 'G2', 'G3', 'G4'))


def get(inst, cd):
    return next(r for r in CTX_RUNS if r['inst'] == inst and r['ctx'] == cd)


print('=' * 150)
print('CONTEXT-WIDENING SWEEP -- 7 instruments x ctx{12,18,24}, 1h, shipped '
      'NESTED 9-feature set')
print('=' * 150)
print(f"{'instrument':<10}{'arm':<7}{'S':>7}{'L':>7}{'W':>7}{'guards':>8}"
      f"{'W_grind':>9}{'maxocc%':>9}{'dominant':>10}{'lbls':>6}{'OOS t':>8}")
print('-' * 150)
for d in ALL_INST:
    for cd in CONTEXT_ARMS:
        r = get(d['name'], cd)
        flag = '  <-- SATURATED' if r['saturated'] else ''
        print(f"{d['name']:<10}ctx{cd:<4}{r['S']:>7.2f}{r['L']:>7.1f}"
              f"{r['W']:>7.2f}{guard_str(r):>8}{r['W_grind']:>9.2f}"
              f"{r['max_occ']:>9.1f}{r['dominant']:>10}{r['n_reachable']:>6}"
              f"{r['OOS_T']:>8.2f}{flag}")
    print('-' * 150)

_gf = [r for r in CTX_RUNS if not r['guards_ok']]
print(f'guards: {len(CTX_RUNS) - len(_gf)} of {len(CTX_RUNS)} runs pass all four'
      + ('' if not _gf else '  FAILING: '
         + ', '.join(r['run_id'] for r in _gf)))
_sat = [r for r in CTX_RUNS if r['saturated']]
print(f'saturation: {len(_sat)} of {len(CTX_RUNS)} runs saturated'
      + ('' if not _sat else '  ' + ', '.join(r['run_id'] for r in _sat)))
""")

co(r"""
# ===========================================================================
# FULL 5-LABEL OCCUPANCY per arm -- the C2 evidence, shown not summarised.
# ===========================================================================
print('=' * 120)
print('OUT-OF-SAMPLE 5-LABEL OCCUPANCY (%) -- the saturation evidence')
print('=' * 120)
print(f"{'instrument':<10}{'arm':<7}" + ''.join(f'{k:>10}' for k in REGIME_LABELS))
print('-' * 120)
for d in ALL_INST:
    for cd in CONTEXT_ARMS:
        r = get(d['name'], cd)
        print(f"{d['name']:<10}ctx{cd:<4}"
              + ''.join(f"{r['occ_oos'][k]:>10.1f}" for k in REGIME_LABELS))
    print('-' * 120)
print('For reference, the daily-Nifty saturation case that motivated C2:')
print('  CONTEXT_DAYS=20 gave bear 100.0% / bull 0.0% / SIDEWAYS 0.0%.')
""")

# ===========================================================================
md(r"""
## 6. Figures — the chart is the point; look here before the grades
""")

co(r"""
# ===========================================================================
# FIGURE 1 -- regime-on-price across the three arms, OOS span. This is the
# chart the hypothesis is ABOUT, so it is drawn before anything is graded.
# ===========================================================================
import matplotlib.pyplot as plt

FIG1_INST = ['AUD/USD', 'USD/CAD', 'EUR/USD']
fig, axes = plt.subplots(len(FIG1_INST), len(CONTEXT_ARMS), figsize=(19, 11),
                         sharex='row')
for i, inst in enumerate(FIG1_INST):
    for j, cd in enumerate(CONTEXT_ARMS):
        ax = axes[i][j]
        r = get(inst, cd)
        lo = r['n_fit']
        dates = r['dates'][lo:]
        px = r['close_arr'][lo:]
        lab = np.asarray(r['labels'][0], object)[lo:]
        shade_regimes(ax, pd.Series(lab, index=dates), alpha=0.55)
        ax.plot(dates, px, color='black', lw=0.7)
        ax.set_xlim(dates[0], dates[-1])
        ax.margins(y=0.05)
        tag = '  [shipped]' if cd == BASELINE_ARM else ''
        sat = '  SATURATED' if r['saturated'] else ''
        ax.set_title(f"{inst}   CONTEXT_DAYS = {cd}{tag}{sat}\n"
                     f"W {r['W']:.1f}   W_grind {r['W_grind']:.1f}   "
                     f"max occ {r['max_occ']:.0f}% ({r['dominant']})",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel('price', fontsize=8)
handles = [plt.matplotlib.patches.Patch(color=REGIME_COLORS[k], alpha=0.7, label=k)
           for k in REGIME_LABELS]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.005))
fig.suptitle('CONTEXT_DAYS 12 (shipped) -> 18 -> 24, out-of-sample span, '
             'shipped 9-feature set, one seed set (for drawing)',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0.035, 1, 0.955])
plt.show()
""")

co(r"""
# ===========================================================================
# FIGURE 2 -- C1 evidence (grind whipsaw per arm) + C2 evidence (occupancy
# concentration per arm, with the 90% saturation line drawn).
# ===========================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5))

_x = np.arange(len(ALL_INST))
_w = 0.26
for k, cd in enumerate(CONTEXT_ARMS):
    vals = [get(d['name'], cd)['W_grind'] for d in ALL_INST]
    axL.bar(_x + (k - 1) * _w, vals, width=_w, label=f'ctx{cd}')
axL.set_xticks(_x)
axL.set_xticklabels([d['name'] for d in ALL_INST], rotation=30, ha='right')
axL.set_ylabel('W_grind (switches / 100 bars)')
axL.set_title('C1: whipsaw in the "steady trend with wiggles" cell\n'
              'lower = more coherent where the user says it looks worst',
              fontsize=10)
axL.legend()
axL.grid(alpha=0.3, axis='y')

for k, cd in enumerate(CONTEXT_ARMS):
    vals = [get(d['name'], cd)['max_occ'] for d in ALL_INST]
    axR.bar(_x + (k - 1) * _w, vals, width=_w, label=f'ctx{cd}')
axR.axhline(90, color='red', ls='--', lw=1.8, label='C2 saturation limit (90%)')
axR.set_xticks(_x)
axR.set_xticklabels([d['name'] for d in ALL_INST], rotation=30, ha='right')
axR.set_ylabel('largest single-regime share of OOS bars (%)')
axR.set_title('C2: saturation guard\n'
              'bars approaching the red line = losing all discrimination',
              fontsize=10)
axR.legend()
axR.grid(alpha=0.3, axis='y')

fig.suptitle('Does widening the context window buy coherence -- '
             'and does it stay a real detector while doing it?',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
""")

# ===========================================================================
md(r"""
## 7. Grading C1–C5
""")

co(r"""
# ===========================================================================
# GRADING -- verbatim against FX_CONTEXT_WIDEN_PROTOCOL.md.
# ===========================================================================
print('=' * 108)
print('GRADING THE PRE-REGISTERED PREDICTIONS')
print('=' * 108)

# ---- C2 FIRST: it can VOID C1, so it is evaluated before C1 is read ------
print('\nC2  SATURATION GUARD (evaluated FIRST -- it can void C1)')
C2_ARM_OK = {}
for cd in CONTEXT_ARMS:
    sat = [d['name'] for d in ALL_INST if get(d['name'], cd)['saturated']]
    worst = max(get(d['name'], cd)['max_occ'] for d in ALL_INST)
    C2_ARM_OK[cd] = not sat
    print(f'    ctx{cd:<3} worst single-regime share {worst:5.1f}%   '
          + ('no saturation' if not sat else f'SATURATED: {sat}'))
_c2 = all(C2_ARM_OK[cd] for cd in CONTEXT_ARMS if cd != BASELINE_ARM)
print(f"C2  {'CONFIRMED' if _c2 else 'REFUTED'}")

# ---- C1 -------------------------------------------------------------------
print('\nC1  grind-cell W per instrument, by arm:')
print(f"    {'instrument':<11}" + ''.join(f'{"ctx" + str(c):>9}' for c in CONTEXT_ARMS))
for d in ALL_INST:
    print(f"    {d['name']:<11}"
          + ''.join(f"{get(d['name'], c)['W_grind']:>9.2f}" for c in CONTEXT_ARMS))
POOL = {c: float(np.nanmean([get(d['name'], c)['W_grind'] for d in ALL_INST]))
        for c in CONTEXT_ARMS}
print(f"    {'POOLED MEAN':<11}" + ''.join(f'{POOL[c]:>9.2f}' for c in CONTEXT_ARMS))

C1_ARM = {}
for cd in CONTEXT_ARMS:
    if cd == BASELINE_ARM:
        continue
    wins = [d['name'] for d in ALL_INST
            if np.isfinite(get(d['name'], cd)['W_grind'])
            and np.isfinite(get(d['name'], BASELINE_ARM)['W_grind'])
            and get(d['name'], cd)['W_grind'] < get(d['name'], BASELINE_ARM)['W_grind']]
    pooled_ok = POOL[cd] < POOL[BASELINE_ARM]
    passed = len(wins) >= 4 and pooled_ok
    C1_ARM[cd] = dict(wins=len(wins), pooled_ok=pooled_ok, passed=passed,
                      void=not C2_ARM_OK[cd])
    void = '  [VOID -- arm saturated]' if not C2_ARM_OK[cd] else ''
    print(f"    ctx{cd}: {len(wins)}/7 instruments improve, pooled "
          f"{POOL[BASELINE_ARM]:.2f} -> {POOL[cd]:.2f} "
          f"({'lower' if pooled_ok else 'HIGHER'}){void}")
_c1_arms_ok = [cd for cd in C1_ARM if C1_ARM[cd]['passed'] and not C1_ARM[cd]['void']]
print(f"C1  {'CONFIRMED' if _c1_arms_ok else 'REFUTED'}"
      + (f"  (on arm(s): {['ctx' + str(c) for c in _c1_arms_ok]})" if _c1_arms_ok else ''))

# ---- C3 -------------------------------------------------------------------
print('\nC3  guards passing per arm:')
_gp = {c: sum(1 for d in ALL_INST if get(d['name'], c)['guards_ok'])
       for c in CONTEXT_ARMS}
for c in CONTEXT_ARMS:
    print(f'    ctx{c:<3} {_gp[c]}/7')
_c3 = all(_gp[c] >= _gp[BASELINE_ARM] for c in CONTEXT_ARMS if c != BASELINE_ARM)
print(f"C3  {'CONFIRMED' if _c3 else 'REFUTED'}")

# ---- C4 -------------------------------------------------------------------
print('\nC4  pooled W by eff_fast quintile, per arm (Q1 choppiest -> Q5 most violent):')
_c4 = True
for c in CONTEXT_ARMS:
    qs = [float(np.nanmean([get(d['name'], c)['W_quintile'][i] for d in ALL_INST]))
          for i in range(5)]
    ok = qs[4] < qs[0]
    if c != BASELINE_ARM and not ok:
        _c4 = False
    print(f"    ctx{c:<3} " + ''.join(f'{v:>8.2f}' for v in qs)
          + f"   Q5<Q1: {'yes' if ok else 'NO -- INVERTED'}")
print(f"C4  {'CONFIRMED' if _c4 else 'REFUTED'}")

# ---- C5 -------------------------------------------------------------------
print('\nC5  mean OOS HAC-t change vs ctx12 (honesty check, not the goal):')
_c5 = True
for cd in CONTEXT_ARMS:
    if cd == BASELINE_ARM:
        continue
    dts = [get(d['name'], cd)['OOS_T'] - get(d['name'], BASELINE_ARM)['OOS_T']
           for d in ALL_INST
           if np.isfinite(get(d['name'], cd)['OOS_T'])
           and np.isfinite(get(d['name'], BASELINE_ARM)['OOS_T'])]
    m = float(np.mean(dts)) if dts else np.nan
    if not (np.isfinite(m) and m >= -0.20):
        _c5 = False
    print(f'    ctx{cd}: mean delta {m:+.3f}  (n={len(dts)})')
print(f"C5  {'CONFIRMED' if _c5 else 'REFUTED'}")

print()
print('Reminder: not a return, not a Sharpe, not a tradeability claim. Data '
      'ends 2022-03-04. This tests the LEVER on FX; it does not test it on '
      'Nifty (yfinance is firewalled in this sandbox).')
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
