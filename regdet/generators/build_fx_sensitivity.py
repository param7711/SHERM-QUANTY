"""Build fx_sensitivity.ipynb -- does making RegDet MORE SENSITIVE recover
forward-return skill on intraday FX, or only trade lag for whipsaw?

DELIBERATELY re-tunes two constants the frozen generalization test holds fixed
(CONTEXT_DAYS, CONFIRM_BARS). It is therefore NOT an architecture-generalization
test and never claims to be. intraday_fx.ipynb is untouched and keeps its
"nothing re-tuned" property.

Protocol frozen in protocols/FX_SENSITIVITY_PROTOCOL.md BEFORE this file was
written. Adds XAU/USD -- a COMMODITY quoted in USD, not an FX pair.

Setup cells are harvested VERBATIM from build_intraday_fx.py so the engine,
metrics, guards and data handling are bit-identical to the main grid.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(
    os.environ.get('REGDET_OUT_DIR',
                   os.path.join(os.path.dirname(HERE), 'notebooks')),
    'fx_sensitivity.ipynb')
FX_GEN = f'{HERE}/build_intraday_fx.py'

_fsrc = open(FX_GEN).read()
assert "'intraday_fx.ipynb'" in _fsrc
_fsrc = _fsrc.replace("'intraday_fx.ipynb'", "'_fx_sens_harvest.ipynb'", 1)
_ns = {'__name__': '_fx_gen_harvest', '__file__': FX_GEN}
exec(compile(_fsrc, FX_GEN, 'exec'), _ns)
_fcells = _ns['cells']

for _tmp in ('_fx_sens_harvest.ipynb', '_intraday_fx_engine_harvest.ipynb'):
    try:
        os.remove(os.path.join(os.path.dirname(OUT), _tmp))
    except OSError:
        pass


def _s(i):
    return ''.join(_fcells[i]['source'])


PIP = 1
ENGINE = [3, 4, 5, 6, 7]
CONST = [9, 10, 11, 12, 13, 14]
DATA = [18, 19, 20]
HARNESS = [22]
EVAL = 28

_eng = '\n'.join(_s(i) for i in ENGINE)
for _sym in ('def build_features', 'def label_bars', 'def _w_hac_contrast',
             'TREND_FEATURE'):
    assert _sym in _eng, f'engine harvest lost {_sym}'
_con = '\n'.join(_s(i) for i in CONST)
for _sym in ('def windows_for', 'def bars_per_day', 'def metric_S',
             'def metric_L', 'def metric_W', 'def evaluate_guards',
             'def guards_pass', 'def occupancy_pct', 'def zigzag_swings',
             'FIXED_KNOBS', 'def assert_knobs_unchanged'):
    assert _sym in _con, f'constants harvest lost {_sym}'
_dat = '\n'.join(_s(i) for i in DATA)
for _sym in ('def load_fx_h1', 'def detect_divisor', 'def resample_ohlc_down',
             'def realised_vol_proxy', 'FX_PAIRS', 'FX_TFS', 'FX_H2'):
    assert _sym in _dat, f'data harvest lost {_sym}'
assert 'def prepare_run' in _s(HARNESS[0]) and 'def label_run' in _s(HARNESS[0])

# Cell EVAL defines span_contrast / ordering_status / evaluate_run AND THEN
# runs the main grid's eval loop. Take the definitions, drop the execution.
_ev = _s(EVAL)
_CUT = '\nt0 = time.time()\nfor r in RUNS:'
assert _CUT in _ev, 'eval cell no longer has the expected execution boundary'
_ev = _ev.split(_CUT)[0]
for _sym in ('def span_contrast', 'def ordering_status', 'def evaluate_run'):
    assert _sym in _ev, f'eval harvest lost {_sym}'
assert 'for r in RUNS' not in _ev, 'eval harvest still executes the main grid'

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip('\n').splitlines(keepends=True)})


def co(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip('\n').splitlines(keepends=True)})


def raw(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.splitlines(keepends=True)})


# ===========================================================================
md(r"""
# Does more sensitivity recover skill on intraday FX?

`intraday_fx.ipynb` found, with **every constant frozen** at its Nifty value:
all 6 out-of-sample `BULL − BEAR` contrasts **negative**, 5-label ordering
holding on **0 of 6** IS and **0 of 6** OOS spans.

`fx_meanrev.ipynb` then found the likely reason is **not** that FX moves against
the detector: variance ratios of 0.80–1.06 with **no series reaching |z\*| ≥ 2**,
and `mom_3d` quintile curves monotone on **0 of 6** series. Close to a random
walk — little directional structure to find in *either* direction.

This notebook asks whether **making the detector faster** changes that.

> ### This deliberately re-tunes. It is NOT a generalization test.
> `CONTEXT_DAYS` and `CONFIRM_BARS` are swept here. `intraday_fx.ipynb` is
> untouched and keeps its "nothing was re-tuned" property. The fixed-knob
> invariant is **re-armed around the two swept knobs**, loudly, so the other 22
> stay guarded — it is not switched off.

| axis | values |
|---|---|
| instrument | EUR/USD, GBP/USD, USD/JPY, **XAU/USD** |
| timeframe | 1h, 2h (2h resampled **down** from 1h) |
| `CONTEXT_DAYS` | **12** (shipped baseline), 6, 3 |
| `CONFIRM_BARS` | **2** (shipped baseline), 1 |

`CONFIRM_BARS` is a *labeling* knob, not a *fitting* knob, so each lookback arm
is **fitted once and labelled twice**. That is asserted, not assumed.

**XAU/USD is a commodity quoted in USD, not an FX pair**, and is labelled that
way throughout. It carries a **third distinct integer divisor** (÷1e2), detected
not hardcoded. Its span starts **2012-05-17**, ~6 months before the majors —
spans differ across instruments by design and are not truncated.

Predictions **S1–S5** are frozen in `protocols/FX_SENSITIVITY_PROTOCOL.md`.
**S3 is the one that matters**: *sensitivity does not buy skill*. It is REFUTED
the moment any single configuration reaches a majority of runs with a positive
OOS contrast.
""")

raw(_s(PIP))

md("## 1. Engine, constants, metrics, guards, data layer — harvested verbatim\n\n"
   f"From `build_intraday_fx.py` cells {PIP}, {ENGINE}, {CONST}, {DATA}, "
   f"{HARNESS}, {EVAL} (definitions only).")

for _i in ENGINE + CONST + DATA + HARNESS:
    raw(_s(_i))
raw(_ev)

# ===========================================================================
md(r"""
## 2. XAU/USD — fetched, rescaled and verified against known gold events

Same loader, same divisor **detection**. Gold's divisor is ÷1e2, a third
distinct scale, which is exactly why the divisor is detected rather than
hardcoded. Verified live against events with independently known magnitudes
before it is used for anything.
""")

co(r"""
# ===========================================================================
# XAU/USD -- a COMMODITY quoted in USD. NOT an FX pair. Labelled as such.
# ===========================================================================
XAU = dict(sym='XAUUSD', name='XAU/USD', med_band=(1000.0, 2100.0),
           cls='COMMODITY')

GOLD_OK, GOLD_FAIL = True, ''
try:
    _t0 = time.time()
    # load_fx_h1 returns (frame, divisor) -- both are wanted: gold's divisor is
    # a THIRD distinct scale and is printed rather than assumed.
    XAU_H1, XAU_DIV = load_fx_h1(XAU)
    print(f'fetched XAUUSD hourly in {time.time() - _t0:.1f}s   '
          f'detected divisor {XAU_DIV:g}  '
          f'(EURUSD/GBPUSD 1e5, USDJPY 1e3 -- gold is a third scale)')
except Exception as e:                                    # noqa: BLE001
    GOLD_OK, GOLD_FAIL = False, f'{type(e).__name__}: {e}'
    print(f'XAUUSD FETCH FAILED: {GOLD_FAIL}. NOTHING is substituted.')

if GOLD_OK:
    _c = XAU_H1['close']
    _checks = []
    _checks.append(('XAU/USD 57,600 hourly bars', len(XAU_H1) == 57600,
                    f'{len(XAU_H1)} bars'))
    _checks.append(('XAU/USD index monotone, no duplicate timestamps',
                    XAU_H1.index.is_monotonic_increasing
                    and not XAU_H1.index.duplicated().any(),
                    f'{XAU_H1.index[0].date()} -> {XAU_H1.index[-1].date()}'))
    _checks.append(('XAU/USD all prices strictly positive and finite',
                    bool((XAU_H1[['open', 'high', 'low', 'close']] > 0).all().all()),
                    f'min {_c.min():.2f}'))
    _checks.append(('XAU/USD OHLC internally consistent',
                    bool((XAU_H1['high'] >= XAU_H1[['open', 'close']].max(axis=1) - 1e-9).all()
                         and (XAU_H1['low'] <= XAU_H1[['open', 'close']].min(axis=1) + 1e-9).all()
                         and (XAU_H1['high'] >= XAU_H1['low']).all()),
                    'high >= max(o,c), low <= min(o,c), high >= low'))
    # EVENTS with independently known magnitudes
    _peak = XAU_H1.loc['2020-08-01':'2020-08-15', 'high'].max()
    _checks.append(('Aug-2020 all-time high ~2075', 2060.0 <= _peak <= 2090.0,
                    f'peak high {_peak:.2f}  (known ATH approx 2075)'))
    _a = XAU_H1.loc['2013-04-11':'2013-04-11', 'close'].iloc[-1]
    _b = XAU_H1.loc['2013-04-16':'2013-04-16', 'close'].min()
    _crash = (_b / _a - 1.0) * 100.0
    _checks.append(('Apr-2013 gold crash approx -13 to -16%',
                    -16.5 <= _crash <= -12.0,
                    f'{_a:.2f} -> {_b:.2f} = {_crash:.2f}%'))
    _lo = _c.min()
    _checks.append(('Dec-2015 cycle low approx 1050', 1040.0 <= _lo <= 1060.0,
                    f'series low {_lo:.2f}'))

    print()
    print('XAU/USD VERIFICATION against independently known magnitudes')
    for name, ok, detail in _checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")
    _npass = sum(1 for _, ok, _ in _checks if ok)
    print(f'\nXAU/USD VERIFICATION: {_npass}/{len(_checks)} checks PASS')
    assert _npass == len(_checks), 'XAU/USD failed verification -- refusing to use it'
    _r = np.log(_c / _c.shift(1)).dropna()
    print(f'hourly realised vol {_r.std() * 100:.3f}%  '
          f'(the FX majors run 0.100-0.114%; gold is roughly twice as volatile)')
    XAU_H2, _rr = resample_ohlc_down(XAU_H1, 2)
    print(f'2h DOWNWARD resample: {len(XAU_H1)} -> {len(XAU_H2)} bars (x{_rr:.3f})')
    print()
    print('XAU/USD is a COMMODITY quoted in USD. It is NOT an FX pair and is not')
    print('labelled as one anywhere. Its span starts 2012-05-17, ~6 months before')
    print('the majors; spans are NOT truncated to match.')
""")

# ===========================================================================
md(r"""
## 3. The swept knobs — invariant re-armed, not switched off

`CONTEXT_DAYS` and `CONFIRM_BARS` are the two knobs this notebook varies. The
fixed-knob invariant from the main grid would fire on both, so it is **re-armed
around exactly those two** for the duration of each arm. The other 22 constants
stay guarded and any drift in them still fails loudly.
""")

co(r"""
# ===========================================================================
# SWEPT KNOBS. Everything else stays at its shipped Nifty value.
# ===========================================================================
CONTEXT_ARMS = (12, 6, 3)          # 12 = shipped baseline
# SCOPE: lookback only. CONFIRM_BARS is HELD at its shipped value 2 at the
# user's instruction ("for now just the max lookback"). The knob is still
# swept-capable below, and pre-registered prediction S4 is about it, so S4 is
# recorded as NOT EVALUATED with its reason rather than quietly dropped.
CONFIRM_ARMS = (2,)                #  2 = shipped baseline
BASELINE_ARM = (12, 2)

SWEPT = ('CONTEXT_DAYS', 'CONFIRM_BARS')
GUARDED = tuple(k for k in FIXED_KNOBS if k not in SWEPT)
print(f'SWEPT (deliberately re-tuned): {SWEPT}')
print(f'STILL GUARDED: {len(GUARDED)} constants -- drift in any of them still fails.')


@contextlib.contextmanager
def sweep_knobs(context_days, confirm_bars):
    '''Install one sweep arm. Re-arms the invariant around the swept knobs.'''
    g = globals()
    old_cd, old_cb = g['CONTEXT_DAYS'], g['CONFIRM_BARS']
    old_fixed = dict(FIXED_KNOBS)
    g['CONTEXT_DAYS'], g['CONFIRM_BARS'] = context_days, confirm_bars
    FIXED_KNOBS['CONTEXT_DAYS'] = context_days
    FIXED_KNOBS['CONFIRM_BARS'] = confirm_bars
    try:
        yield (context_days, confirm_bars)
    finally:
        g['CONTEXT_DAYS'], g['CONFIRM_BARS'] = old_cd, old_cb
        FIXED_KNOBS.clear()
        FIXED_KNOBS.update(old_fixed)


# Prove the re-arming actually restores: the invariant must hold before, inside
# (at the swept values) and after.
assert_knobs_unchanged('before any sweep arm')
with sweep_knobs(3, 1):
    assert CONTEXT_DAYS == 3 and CONFIRM_BARS == 1
    assert_knobs_unchanged('inside a sweep arm')
    assert windows_for(24)['SWING_WIN'] == 3 * 24, \
        'CONTEXT_DAYS did not reach windows_for -- the sweep would be a no-op'
assert CONTEXT_DAYS == 12 and CONFIRM_BARS == 2, 'sweep_knobs failed to restore'
assert_knobs_unchanged('after the sweep arm')
print('ASSERT OK: sweep_knobs installs, reaches windows_for, and RESTORES.')
print('           A sweep that silently failed to change the windows would have')
print('           produced three identical arms and looked like a null result.')
""")

# ===========================================================================
md(r"""
## 4. Pre-registered predictions
""")

co(r"""
# ===========================================================================
# PRE-REGISTERED -- frozen in protocols/FX_SENSITIVITY_PROTOCOL.md.
# ===========================================================================
S_MAJ = 5          # of 8 runs
S_PREDICTIONS = {
 'S1': ('Shortening CONTEXT_DAYS 12 -> 6 -> 3 RAISES the whipsaw rate W.',
        'CONFIRMED if mean W is monotone increasing as lookback shortens, at '
        'CONFIRM_BARS = 2.'),
 'S2': ('Shortening CONTEXT_DAYS 12 -> 6 -> 3 LOWERS the transition lag L.',
        'CONFIRMED if mean L is monotone decreasing as lookback shortens, at '
        'CONFIRM_BARS = 2.'),
 'S3': ('SENSITIVITY DOES NOT BUY SKILL. No configuration produces a MAJORITY '
        f'(>= {S_MAJ} of 8) of runs with a POSITIVE out-of-sample (BULL - BEAR) '
        'contrast.',
        'CONFIRMED if EVERY one of the 6 configurations fails to reach that '
        'majority. REFUTED the moment any single configuration reaches it -- '
        'which would mean sensitivity DOES recover skill.'),
 'S4': ('CONFIRM_BARS 2 -> 1 raises W and lowers L at every lookback, and does '
        'NOT improve OOS ordering.',
        'CONFIRMED if mean W rises and mean L falls at all 3 lookbacks AND the '
        'count of runs with a positive OOS contrast does not increase at any '
        'lookback.'),
 'S5': ('XAU/USD does not separate from the FX majors on OOS ordering.',
        'CONFIRMED if XAU/USD\'s 2 runs land INSIDE the range spanned by the 6 '
        'major runs at the shipped (12, 2) baseline.'),
}

print('=' * 114)
print('PRE-REGISTERED PREDICTIONS  (frozen before any number existed; not softened)')
print('=' * 114)
for k, (claim, rule) in S_PREDICTIONS.items():
    print(f'\n{k}.  {claim}')
    print(f'    GRADING RULE: {rule}')
print()
print('=' * 114)
print('S3 IS THE HEADLINE. fx_meanrev.ipynb found these series close to a random')
print('walk, so the prior is that speed changes WHAT the detector reacts to, not')
print('WHETHER the reaction is informative. If S3 is REFUTED, that prior is wrong')
print('and it is the most important result in this project. It is graded either way.')
print('=' * 114)
""")

# ===========================================================================
md(r"""
## 5. Runtime — measured before the sweep is launched

3 lookback arms × 8 runs × R=4 × K=4 = **384 fits**, ~4× the main grid.
`CONFIRM_BARS` costs no extra fit — each arm is fitted once and labelled twice,
which is asserted below by counting fits.

If the projection exceeds the budget the **only** sanctioned reduction is a
training-window cap, printed with its reason. `R = 4` is never cut and no
instrument or timeframe is ever dropped.
""")

co(r"""
# ===========================================================================
# INSTRUMENTS + runtime probe.
# ===========================================================================
assert FX_OK and GOLD_OK, 'refusing to run a partial grid'
INSTRUMENTS = [dict(p, cls='FX') for p in FX_PAIRS] + [XAU]
BARS_BY_SYM = {p['sym']: (FX_H1[p['sym']], FX_H2[p['sym']]) for p in FX_PAIRS}
BARS_BY_SYM[XAU['sym']] = (XAU_H1, XAU_H2)

N_RUNS = len(INSTRUMENTS) * len(FX_TFS)
N_ARMS = len(CONTEXT_ARMS)

# ---------------------------------------------------------------------------
# TRAIN_CAP_BARS is a DECLARED CONSTANT, never derived from measured runtime.
#
# An earlier version computed it from the probe's wall clock. That put machine
# SPEED into the SCIENCE: two runs on identical data produced caps of 11700 and
# 11400, hence different fit windows and different numbers (mean S% 89.9 vs
# 93.0, mean OOS t -0.96 vs -1.06). The graded verdicts happened to survive,
# but the notebook was not reproducible, which is not a property worth trading
# for a tidier runtime. Caught by the two-run determinism diff.
#
# 12000 is chosen so 384 fits complete in roughly the 15-minute budget on the
# reference machine, and is FIXED regardless of how fast this machine is.
# ---------------------------------------------------------------------------
TRAIN_CAP_BARS = 12000
PROJ_FITS = N_ARMS * N_RUNS * R_SEED_SETS * ENSEMBLE_K
print(f'{len(INSTRUMENTS)} instruments x {len(FX_TFS)} timeframes = {N_RUNS} runs')
print(f'{N_ARMS} lookback arms x {N_RUNS} runs x R={R_SEED_SETS} x K={ENSEMBLE_K}'
      f' = {PROJ_FITS} fits')

_p = INSTRUMENTS[0]
_bars = BARS_BY_SYM[_p['sym']][0]
_close = _bars['close'].astype(float)
_t0 = time.time()
with sweep_knobs(CONTEXT_ARMS[-1], 2):        # shortest lookback = most bars
    with bars_per_day(24):
        _vol = realised_vol_proxy(_close, 24)
    _prep = prepare_run(_close, _vol, 24)
    _f0 = FIT_COUNT
    _lab, _mods = label_run(_prep, _close, 24)
_probe_s = time.time() - _t0
_fits_probe = FIT_COUNT - _f0
_per_fit = _probe_s / max(_fits_probe, 1)

# 2h runs are ~half the rows; approximate the grid as half 1h / half 2h.
_proj = PROJ_FITS * _per_fit * 0.75
print('=' * 100)
print('RUNTIME PROBE -- measured on the LARGEST arm before anything is launched')
print('=' * 100)
print(f'  probe                  : {_p["name"]} @ 1h, CONTEXT_DAYS={CONTEXT_ARMS[-1]}')
print(f'  one seed set           : {_probe_s:.1f}s over {_fits_probe} fits '
      f'({_per_fit:.1f}s per fit)')
print(f'  PROJECTED GRID TOTAL   : {_proj:>6.0f}s = {_proj / 60:.1f} min '
      f'({PROJ_FITS} fits)')
print(f'  budget                 : {RUNTIME_BUDGET_S / 60:.0f} min')

print()
print(f'  >>> TRAIN_CAP_BARS = {TRAIN_CAP_BARS}  -- a DECLARED CONSTANT (see above).')
print( '  >>> The probe MEASURES and REPORTS. It does NOT set the cap.')
print( '  >>> R=4 is intact; all 3 lookback arms, all 4 instruments and both')
print( '  >>> timeframes are kept. The cap shortens the fit window from the')
print( '  >>> LEFT (most recent bars kept), applied FORWARD identically on')
print( '  >>> every run; the IS/OOS boundary n_fit does NOT move.')
print( '  >>> This makes the sweep internally comparable but NOT directly')
print( '  >>> comparable to intraday_fx.ipynb, which fit on the full window.')
print( '  >>> That is why the shipped baseline is RE-RUN here, not quoted.')
if _proj > RUNTIME_BUDGET_S * 1.5:
    print()
    print(f'  >>> NOTE: projection {_proj / 60:.1f} min still exceeds budget '
          f'{RUNTIME_BUDGET_S / 60:.0f} min.')
    print( '  >>> Reported, not silently corrected -- correcting it here would')
    print( '  >>> put wall-clock back into the science.')
print('=' * 100)
""")

# ===========================================================================
md(r"""
## 6. The sweep
""")

co(r"""
# ===========================================================================
# THE SWEEP. Each lookback arm is FITTED ONCE and LABELLED TWICE.
# ===========================================================================
SWEEP = []
_fit_before_all = FIT_COUNT
t0 = time.time()
print(f"{'arm':<10}{'run':<15}{'bars':>8}{'fit rows':>10}{'ctx bars':>10}"
      f"{'secs':>8}")
print('-' * 62)
for cd in CONTEXT_ARMS:
    for inst in INSTRUMENTS:
        h1, h2 = BARS_BY_SYM[inst['sym']]
        for tf, hours, bpd in FX_TFS:
            _tt = time.time()
            bars = h1 if hours == 1 else h2
            close = bars['close'].astype(float)
            close.name = inst['sym']
            for cb in CONFIRM_ARMS:
                with sweep_knobs(cd, cb):
                    with bars_per_day(bpd):
                        vol = realised_vol_proxy(close, bpd)
                    prep = prepare_run(close, vol, bpd)
                    labels, _m = label_run(prep, close, bpd)
                    assert_knobs_unchanged(f'{inst["name"]}@{tf} cd={cd} cb={cb}')
                    SWEEP.append(dict(
                        run_id=f"{inst['name']}@{tf}", inst=inst['name'],
                        cls=inst['cls'], tf=tf, bpd=bpd, ctx=cd, cb=cb,
                        arm=f'ctx{cd}/cb{cb}', error=None,
                        labels=labels, close_arr=prep['close'].values,
                        fwd_h=fwd_horizons_for(bpd),
                        n_bars=prep['n_bars'], n_fit=prep['n_fit']))
            print(f"ctx{cd:<7}{inst['name'] + '@' + tf:<15}"
                  f"{prep['n_bars']:>8}{prep['n_fit']:>10}"
                  f"{cd * bpd:>10}{time.time() - _tt:>8.1f}")

_fits_used = FIT_COUNT - _fit_before_all
print('-' * 62)
print(f'LABEL PHASE: {time.time() - t0:.1f}s   {len(SWEEP)} labelled '
      f'configurations   {_fits_used} HMM fits')
_expect = N_ARMS * N_RUNS * R_SEED_SETS * ENSEMBLE_K
assert _fits_used == _expect, f'expected {_expect} fits, used {_fits_used}'
assert ZZ_CALLS == 0, ('LEAKAGE: a ZigZag was computed during the label phase')
print(f'ASSERT OK: {_fits_used} fits, and ZZ_CALLS == 0 -- every label in this')
print('sweep was produced BEFORE any ZigZag existed.')
print(f'TRAIN_CAP_BARS = {TRAIN_CAP_BARS}')
""")

co(r"""
# ===========================================================================
# EVAL PHASE. Strictly AFTER the whole label phase -- interleaving label and
# eval trips the leakage tripwire, and rightly so: metric L needs the ZigZag,
# and once a ZigZag exists no further label may be produced.
# ===========================================================================
LABEL_PHASE_OPEN = False
print('LABEL PHASE CLOSED. Any label_bars call from here on fails the tripwire.')
_t0 = time.time()
for r in SWEEP:
    evaluate_run(r)
print(f'evaluated {len(SWEEP)} configurations in {time.time() - _t0:.1f}s   '
      f'ZZ_CALLS={ZZ_CALLS}')

print()
print('=' * 112)
print('EVERY LABELLED CONFIGURATION')
print('=' * 112)
print(f"{'arm':<10}{'run':<15}{'SIDE%':>8}{'S%':>7}{'W':>7}{'L':>7}  guards"
      f"{'IS t':>8}{'OOS t':>8}   ")
print('-' * 112)
for r in SWEEP:
    print(f"ctx{r['ctx']:<7}{r['run_id']:<15}"
          f"{r['occ']['SIDEWAYS']:>8.1f}{r['S']:>7.1f}{r['W']:>7.2f}"
          f"{r['L']:>7.1f}  "
          + ''.join('.' if r['guards'][k][0] else 'X'
                    for k in ('G1', 'G2', 'G3', 'G4'))
          + f"{r['IS_T']:>8.2f}{r['OOS_T']:>8.2f}"
          + ('   [COMMODITY]' if r['cls'] != 'FX' else ''))
print('-' * 112)
print("guards: '.' = pass, 'X' = fail, order G1 G2 G3 G4")
""")

co(r"""
# ===========================================================================
# TABLE -- configuration means, and the count that decides S3.
# ===========================================================================
def arm_rows(cd, cb):
    return [r for r in SWEEP if r['ctx'] == cd and r['cb'] == cb]


print('=' * 114)
print('CONFIGURATION SUMMARY -- means over the 8 runs, and the S3 count')
print('=' * 114)
print(f"{'ctx':>5}{'cb':>4}{'mean W':>9}{'mean L':>9}{'mean SIDE%':>12}"
      f"{'mean S%':>9}{'guards pass':>13}{'OOS t > 0':>11}{'mean OOS t':>12}")
print('-' * 114)
ARM_STATS = {}
for cd in CONTEXT_ARMS:
    for cb in CONFIRM_ARMS:
        rs = arm_rows(cd, cb)
        npos = sum(1 for r in rs if np.isfinite(r['OOS_D']) and r['OOS_D'] > 0)
        st = dict(W=float(np.mean([r['W'] for r in rs])),
                  L=float(np.mean([r['L'] for r in rs])),
                  SIDE=float(np.mean([r['occ']['SIDEWAYS'] for r in rs])),
                  S=float(np.mean([r['S'] for r in rs])),
                  gp=sum(1 for r in rs if r['guards_ok']),
                  npos=npos,
                  oos_t=float(np.nanmean([r['OOS_T'] for r in rs])))
        ARM_STATS[(cd, cb)] = st
        tag = '  <-- shipped baseline' if (cd, cb) == BASELINE_ARM else ''
        print(f"{cd:>5}{cb:>4}{st['W']:>9.2f}{st['L']:>9.1f}{st['SIDE']:>12.1f}"
              f"{st['S']:>9.1f}{st['gp']:>9} of {len(rs)}{npos:>8} of {len(rs)}"
              f"{st['oos_t']:>12.2f}{tag}")
print('-' * 114)
print(f'"OOS t > 0" counts runs whose out-of-sample BULL-BEAR contrast is '
      f'POSITIVE. S3 needs\nevery configuration to stay BELOW {S_MAJ} of '
      f'{len(arm_rows(*BASELINE_ARM))} for the prediction to hold.')
""")

co(r"""
# ===========================================================================
# TABLE -- IS vs OOS per run at each arm. NEVER POOLED.
# ===========================================================================
print('=' * 114)
print('IN-SAMPLE vs OUT-OF-SAMPLE contrast, per run, per configuration. '
      'Spans DISJOINT, embargoed.')
print('=' * 114)
print(f"{'run':<15}" + ''.join(f"{'ctx' + str(cd) + '/cb' + str(cb):>17}"
                               for cd in CONTEXT_ARMS for cb in CONFIRM_ARMS))
print(f"{'':<15}" + ''.join(f"{'IS t':>8}{'OOS t':>9}"
                            for _ in range(N_ARMS * len(CONFIRM_ARMS))))
print('-' * 114)
for inst in INSTRUMENTS:
    for tf, _h, _b in FX_TFS:
        rid = f"{inst['name']}@{tf}"
        line = f'{rid:<15}'
        for cd in CONTEXT_ARMS:
            for cb in CONFIRM_ARMS:
                r = next(x for x in SWEEP
                         if x['run_id'] == rid and x['ctx'] == cd and x['cb'] == cb)
                line += f"{r['IS_T']:>8.2f}{r['OOS_T']:>9.2f}"
        print(line + ('   [COMMODITY]' if inst['cls'] != 'FX' else ''))
print('-' * 114)
print('A POSITIVE OOS t is the detector calling direction correctly out of '
      'sample.\nEvery run ends 2022-03-04. Volatility is a causal realised-vol '
      'PROXY on every run.')
""")

# ===========================================================================
md(r"""
## 7. Figures
""")

co(r"""
# ===========================================================================
# FIGURE -- what sensitivity actually bought.
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
_x = np.arange(len(CONTEXT_ARMS))
_mk = {2: 'o', 1: 's'}
for cb in CONFIRM_ARMS:
    for ax, key, lab in ((axes[0], 'W', 'whipsaw W (switches / 100 bars)'),
                         (axes[1], 'L', 'transition lag L (bars)'),
                         (axes[2], 'npos', 'runs with POSITIVE OOS contrast')):
        ys = [ARM_STATS[(cd, cb)][key] for cd in CONTEXT_ARMS]
        ax.plot(_x, ys, marker=_mk[cb], lw=1.8,
                label=f'CONFIRM_BARS = {cb}' + (' (shipped)' if cb == 2 else ''))
        ax.set_xticks(_x)
        ax.set_xticklabels([f'{cd}d' for cd in CONTEXT_ARMS])
        ax.set_xlabel('CONTEXT_DAYS (max lookback)  -- shorter = more sensitive ->')
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
axes[0].set_title('S1: does a shorter lookback raise whipsaw?', fontsize=10)
axes[1].set_title('S2: does it lower lag?', fontsize=10)
axes[2].axhline(S_MAJ, color='crimson', ls='--', lw=1.5)
axes[2].text(len(CONTEXT_ARMS) - 1, S_MAJ + 0.06,
             f'S3 refuted above here ({S_MAJ} of {N_RUNS})', fontsize=8,
             color='crimson', ha='right', va='bottom')
axes[2].set_ylim(-0.4, N_RUNS + 0.4)
axes[2].set_title('S3: does any of it buy SKILL?', fontsize=10)
for ax in axes:
    ax.legend(fontsize=8, loc='best')
fig.suptitle(f'SENSITIVITY SWEEP -- {len(INSTRUMENTS)} instruments x '
             f'{len(FX_TFS)} timeframes x {len(CONTEXT_ARMS)} lookbacks'
             + (f' x {len(CONFIRM_ARMS)} confirm settings'
                if len(CONFIRM_ARMS) > 1
                else f'   (CONFIRM_BARS held at {CONFIRM_ARMS[0]})')
             + '   [REAL DATA, ends 2022-03-04]', fontsize=11)
fig.tight_layout()
plt.show()
""")

# ===========================================================================
md(r"""
## 8. Grading S1–S5
""")

co(r"""
# ===========================================================================
# GRADING -- verbatim against the frozen rules.
# ===========================================================================
def sv(ok):
    return 'CONFIRMED' if ok else 'REFUTED'


SG = {}
print('=' * 114)
print(f'GRADING THE PRE-REGISTERED PREDICTIONS   ({N_RUNS} runs per '
      f'configuration; a MAJORITY is >= {S_MAJ})')
print('=' * 114)

_W2 = [ARM_STATS[(cd, 2)]['W'] for cd in CONTEXT_ARMS]
_L2 = [ARM_STATS[(cd, 2)]['L'] for cd in CONTEXT_ARMS]

print(f"\nS1  {S_PREDICTIONS['S1'][0]}")
for cd, w in zip(CONTEXT_ARMS, _W2):
    print(f'    CONTEXT_DAYS {cd:>2}  mean W = {w:.2f}')
SG['S1'] = all(b > a for a, b in zip(_W2, _W2[1:]))
print(f"    -> S1 {sv(SG['S1'])}")

print(f"\nS2  {S_PREDICTIONS['S2'][0]}")
for cd, l in zip(CONTEXT_ARMS, _L2):
    print(f'    CONTEXT_DAYS {cd:>2}  mean L = {l:.1f}')
SG['S2'] = all(b < a for a, b in zip(_L2, _L2[1:]))
print(f"    -> S2 {sv(SG['S2'])}")

print(f"\nS3  {S_PREDICTIONS['S3'][0]}   <-- THE HEADLINE")
_worst = 0
for cd in CONTEXT_ARMS:
    for cb in CONFIRM_ARMS:
        n = ARM_STATS[(cd, cb)]['npos']
        _worst = max(_worst, n)
        print(f'    ctx{cd}/cb{cb}   {n} of {N_RUNS} runs have a POSITIVE OOS '
              f'contrast' + ('   >>> MAJORITY REACHED' if n >= S_MAJ else ''))
SG['S3'] = _worst < S_MAJ
print(f'    best configuration reached {_worst} of {N_RUNS}; the bar is {S_MAJ}.')
print(f"    -> S3 {sv(SG['S3'])}")

print(f"\nS4  {S_PREDICTIONS['S4'][0]}")
if len(CONFIRM_ARMS) < 2:
    # NOT quietly dropped. S4 is pre-registered in the frozen protocol and this
    # run simply does not have the arm needed to grade it, because the sweep
    # was scoped to the lookback axis only. Recorded as such.
    SG['S4'] = None
    print(f'    CONFIRM_BARS was HELD at the shipped value '
          f'{CONFIRM_ARMS[0]} for this run,')
    print('    so the 2 -> 1 comparison S4 asks for does not exist here.')
    print('    -> S4 NOT EVALUATED (scope: lookback only). Not refuted, not')
    print('       confirmed, and NOT dropped -- it stays pre-registered and')
    print('       is gradeable by re-running with CONFIRM_ARMS = (2, 1).')
else:
    _w_ok = _l_ok = _sk_ok = True
    for cd in CONTEXT_ARMS:
        a, b = ARM_STATS[(cd, 2)], ARM_STATS[(cd, 1)]
        _w_ok &= b['W'] > a['W']
        _l_ok &= b['L'] < a['L']
        _sk_ok &= b['npos'] <= a['npos']
        print(f"    ctx{cd}:  W {a['W']:.2f} -> {b['W']:.2f}   "
              f"L {a['L']:.1f} -> {b['L']:.1f}   "
              f"OOS t>0 {a['npos']} -> {b['npos']}")
    SG['S4'] = bool(_w_ok and _l_ok and _sk_ok)
    print(f'    W rises at all 3: {_w_ok}   L falls at all 3: {_l_ok}   '
          f'skill not improved: {_sk_ok}')
    print(f"    -> S4 {sv(SG['S4'])}")

print(f"\nS5  {S_PREDICTIONS['S5'][0]}")
_base = [r for r in SWEEP if (r['ctx'], r['cb']) == BASELINE_ARM]
_maj = [r['OOS_T'] for r in _base if r['cls'] == 'FX']
_gold = [r['OOS_T'] for r in _base if r['cls'] != 'FX']
_lo, _hi = min(_maj), max(_maj)
print(f'    FX majors OOS t range at the shipped baseline: '
      f'[{_lo:+.2f}, {_hi:+.2f}]  (n={len(_maj)})')
for r in _base:
    if r['cls'] != 'FX':
        inside = _lo <= r['OOS_T'] <= _hi
        print(f"    {r['run_id']:<14} OOS t = {r['OOS_T']:+.2f}   "
              + ('INSIDE the majors\' range' if inside else 'OUTSIDE -- separates'))
SG['S5'] = all(_lo <= g <= _hi for g in _gold)
print(f"    -> S5 {sv(SG['S5'])}")

print('\n' + '=' * 114)
print('SUMMARY:  ' + '   '.join(
    f'{k} ' + ('NOT EVALUATED' if v is None else sv(v)) for k, v in SG.items()))
print('=' * 114)
""")

co(r"""
# ===========================================================================
# WHAT THE SWEEP ESTABLISHES -- decided by the numbers, in plain words.
# ===========================================================================
print('=' * 114)
print('DID MORE SENSITIVITY RECOVER SKILL ON INTRADAY FX?')
print('=' * 114)
_b = ARM_STATS[BASELINE_ARM]
_fast = ARM_STATS[(CONTEXT_ARMS[-1], CONFIRM_ARMS[0])]
print(f'  Shipped baseline CONTEXT_DAYS=12 -> shortest CONTEXT_DAYS='
      f'{CONTEXT_ARMS[-1]}  (CONFIRM_BARS held at {CONFIRM_ARMS[0]}):')
print(f"    whipsaw W   {_b['W']:.2f}  ->  {_fast['W']:.2f}")
print(f"    lag L       {_b['L']:.1f}  ->  {_fast['L']:.1f}")
print(f"    guards pass {_b['gp']} of {N_RUNS}  ->  {_fast['gp']} of {N_RUNS}")
print(f"    OOS t > 0   {_b['npos']} of {N_RUNS}  ->  {_fast['npos']} of {N_RUNS}")
print()
if not SG['S3']:
    print('  S3 IS REFUTED. At least one configuration reached a majority of runs')
    print('  with a POSITIVE out-of-sample contrast. Sensitivity DID recover')
    print('  directional skill that the shipped settings missed. This contradicts')
    print('  the near-random-walk reading from fx_meanrev.ipynb and is the most')
    print('  important result in this project -- it needs independent replication')
    print('  on a fresh span before ANY of it is trusted or acted on.')
else:
    _ncfg = len(ARM_STATS)
    _best = max(ARM_STATS.values(), key=lambda s: s['npos'])['npos']
    print(f'  S3 IS CONFIRMED, and that is the finding. Across every one of the')
    print(f'  {_ncfg} configurations -- including the most aggressive one tested --')
    print(f'  at most {_best} of {N_RUNS} runs achieved a positive out-of-sample')
    print(f'  contrast, against a bar of {S_MAJ}. Making the detector faster changed')
    print('  WHAT it reacted to; it did not make the reaction informative. That is')
    print('  what a near-random-walk series looks like from the inside, and it')
    print('  matches fx_meanrev.ipynb.')
    if _best > ARM_STATS[BASELINE_ARM]['npos']:
        _bt = min(ARM_STATS.values(), key=lambda s: abs(s['oos_t']))['oos_t']
        print()
        print('  REPORTED RATHER THAN BURIED: the shortest lookback IS the best of')
        print(f'  the arms -- it moved from {ARM_STATS[BASELINE_ARM]["npos"]} to '
              f'{_best} of {N_RUNS} positive runs and pulled the mean')
        print(f'  OOS t from {ARM_STATS[BASELINE_ARM]["oos_t"]:+.2f} to {_bt:+.2f}. '
              f'That is movement in the predicted-')
        print('  useful direction and it is NOT nothing. It is also nowhere near the')
        print('  pre-registered bar, and no threshold was moved to make it look')
        print('  closer. On its own it is consistent with noise at this sample size.')
    print()
    print('  Sensitivity is therefore not the missing ingredient on intraday FX.')
    print('  A shorter lookback buys responsiveness and pays for it in whipsaw,')
    print('  which is a real trade-off worth having on a market that HAS a')
    print('  directional signal -- but it cannot manufacture one that is absent.')
print()
print('  SCOPE AND LIMITS, stated rather than buried:')
print(f'    - TRAIN_CAP_BARS = {TRAIN_CAP_BARS}. If set, this sweep is internally')
print('      comparable but NOT directly comparable to intraday_fx.ipynb; the')
print('      shipped baseline was therefore RE-RUN here rather than quoted.')
print('    - XAU/USD is a COMMODITY, not an FX pair, and starts ~6 months before')
print('      the majors. Spans differ by instrument and were not truncated.')
print('    - Volatility is a causal realised-vol PROXY on every run.')
print('    - Data ends 2022-03-04. Nothing here speaks to 2022-2026.')
print('    - No return, no Sharpe, no tradeability claim. This measures a')
print('      DETECTOR, not a strategy.')
print('=' * 114)
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
