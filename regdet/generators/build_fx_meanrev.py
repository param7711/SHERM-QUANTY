"""Build fx_meanrev.ipynb -- does intraday FX mean-revert at the detector's own
horizons, and does the detector's own trend feature anti-predict?

Tests the hypothesis offered for intraday_fx.ipynb's result (all 6 OOS
BULL-BEAR contrasts negative). Protocol frozen in
protocols/FX_MEANREV_PROTOCOL.md BEFORE this file was written.

Fits NO HMM and emits NO regime label. The setup cells -- engine, frozen
constants, FX data layer, prepare_run -- are harvested VERBATIM from
build_intraday_fx.py so the data handling, the divisor detection, the
verification checks and `mom_3d` are bit-identical to the main grid.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(
    os.environ.get('REGDET_OUT_DIR',
                   os.path.join(os.path.dirname(HERE), 'notebooks')),
    'fx_meanrev.ipynb')
FX_GEN = f'{HERE}/build_intraday_fx.py'

# ---------------------------------------------------------------------------
# HARVEST -- the setup cells from build_intraday_fx.py, VERBATIM. Same
# technique that generator uses on the master: execute it in a private
# namespace with its output path redirected, then read its assembled `cells`.
# ---------------------------------------------------------------------------
_fsrc = open(FX_GEN).read()
assert "'intraday_fx.ipynb'" in _fsrc
_fsrc = _fsrc.replace("'intraday_fx.ipynb'", "'_fx_meanrev_harvest.ipynb'", 1)
_ns = {'__name__': '_fx_gen_harvest', '__file__': FX_GEN}
exec(compile(_fsrc, FX_GEN, 'exec'), _ns)
_fcells = _ns['cells']

for _tmp in ('_fx_meanrev_harvest.ipynb', '_intraday_fx_engine_harvest.ipynb'):
    try:
        os.remove(os.path.join(os.path.dirname(OUT), _tmp))
    except OSError:
        pass


def _s(i):
    return ''.join(_fcells[i]['source'])


# Cell identity asserted by CONTENT, never trusted by number.
PIP = 1
ENGINE = [3, 4, 5, 6, 7]          # engine + HAC contrast + S/L/W harvests
CONST = [9, 10, 11, 12, 13, 14]   # frozen knobs, windows_for, tripwire
DATA = [18, 19, 20]               # fetch + verify + resample
HARNESS = [22]                    # SEED_SETS + prepare_run / label_run

assert '%pip install' in _s(PIP)
_eng = '\n'.join(_s(i) for i in ENGINE)
for _sym in ('def build_features', 'def label_bars', 'def _w_hac_contrast',
             'TREND_FEATURE', 'import numpy'):
    assert _sym in _eng, f'engine harvest lost {_sym}'
_con = '\n'.join(_s(i) for i in CONST)
for _sym in ('def windows_for', 'def bars_per_day', 'CONTEXT_DAYS', 'MOM_DAYS',
             'FIXED_KNOBS'):
    assert _sym in _con, f'constants harvest lost {_sym}'
_dat = '\n'.join(_s(i) for i in DATA)
for _sym in ('def load_fx_h1', 'def detect_divisor', 'def resample_ohlc_down',
             'def realised_vol_proxy', 'FX_PAIRS', 'FX_TFS', 'FX_H2', 'FX_OK'):
    assert _sym in _dat, f'data harvest lost {_sym}'
assert 'def prepare_run' in _s(HARNESS[0]), 'harness harvest lost prepare_run'

PROVENANCE = ('build_intraday_fx.py cells '
              f'{PIP}, {ENGINE}, {CONST}, {DATA}, {HARNESS} -- verbatim')

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
# Does intraday FX mean-revert at RegDet's own horizons?

`intraday_fx.ipynb` measured, with every constant frozen at its Nifty value:

* **all 6** out-of-sample `(mean fwd | BULL) − (mean fwd | BEAR)` contrasts
  **negative** (OOS HAC *t* from −0.18 to −1.62),
* full 5-label ordering holding on **0 of 6** IS spans and **0 of 6** OOS spans.

A hypothesis was offered, and labelled as a hypothesis:

> RegDet is backward-looking momentum — it labels BULL *after* an up-move. If
> FX mean-reverts at the detector's horizons, forward returns after a BULL
> label are negative.

This notebook tests it. **No HMM is fitted and no regime label is emitted.**

The hypothesis needs **two** links, and either one failing refutes it:

1. the series mean-reverts at the horizons the detector uses, and
2. the detector's own trend feature `mom_3d` **anti-predicts** forward return.

Link 2 is the decisive one. If it fails, mean reversion does not explain the
negative contrast and the cause is still unknown — and this notebook says so.

Predictions **M1–M3** are frozen in `protocols/FX_MEANREV_PROTOCOL.md`, written
before this notebook existed. They are graded in the last section.

**Data**: the same community GitHub source as the main grid, same divisor
detection, same 14 verification checks. **It ends 2022-03-04** and says nothing
about 2022–2026. There is **no Nifty comparison** — every verdict is stated
against the random-walk null `VR = 1`, an absolute benchmark.
""")

raw(_s(PIP))

md("## 1. Engine, frozen constants, FX data layer — harvested verbatim\n\n"
   f"Source: `{PROVENANCE}`. `mom_3d` is therefore computed by exactly the code "
   "that produced the labels in the main grid.")

for _i in ENGINE + CONST + DATA + HARNESS:
    raw(_s(_i))

# ===========================================================================
md(r"""
## 2. The pre-registered predictions

Printed before a single number exists. Graded verbatim in section 6.
""")

co(r"""
# ===========================================================================
# PRE-REGISTERED -- frozen in protocols/FX_MEANREV_PROTOCOL.md before this
# notebook was built. Not restated from memory: the grading rules below ARE
# the protocol text.
# ===========================================================================
MR_MAJ = 4          # of 6 series
VR_Z_BAR = 2.0      # |z*| bar for M1
HAC_T_BAR = 2.0     # |t| bar for M3

MR_PREDICTIONS = {
 'M1': ('VR(q) < 1 at the detector\'s own horizons on a MAJORITY of the series.',
        f'CONFIRMED if >= {MR_MAJ} of 6 series have VR(q) < 1 at a tested q AND '
        f'reach |z*| >= {VR_Z_BAR:.0f} at some tested q. A VR below 1 that is '
        'not distinguishable from 1 is NOT evidence.'),
 'M2': ('The mom_3d quintile -> mean-forward-return relation is DECREASING '
        '(top quintile below bottom quintile), h-averaged.',
        f'CONFIRMED if >= {MR_MAJ} of 6 series show top < bottom. This is the '
        'link that would actually explain the BULL-BEAR sign.'),
 'M3': ('The top-minus-bottom quintile spread is NEGATIVE and significant.',
        f'CONFIRMED if >= {MR_MAJ} of 6 series have spread < 0 with HAC '
        f'|t| >= {HAC_T_BAR:.0f}. M2 can pass on sign while M3 fails; that '
        'combination means real in direction, too weak to carry the claim.'),
}

print('=' * 110)
print('PRE-REGISTERED PREDICTIONS  (frozen before any number existed; not softened)')
print('=' * 110)
for k, (claim, rule) in MR_PREDICTIONS.items():
    print(f'\n{k}.  {claim}')
    print(f'    GRADING RULE: {rule}')
print()
print('=' * 110)
print('VERDICT RULE FOR THE HYPOTHESIS AS A WHOLE:')
print('  M2 and M3 CONFIRMED -> mean reversion at the detector\'s own input is a')
print('                         SUFFICIENT explanation for the negative contrast.')
print('  M2 CONFIRMED, M3 REFUTED -> direction consistent, magnitude too weak;')
print('                         report as PARTIAL, mechanism NOT settled.')
print('  M2 REFUTED -> hypothesis REFUTED regardless of M1. Cause still unknown.')
print('  M1 alone settles NOTHING: a series can mean-revert without that being')
print('  what drives this particular detector\'s labels.')
print('=' * 110)
""")

# ===========================================================================
md(r"""
## 3. Variance ratio and autocorrelation

`VR(q) = Var(q-bar log return) / (q · Var(1-bar log return))`, overlapping
windows, Lo–MacKinlay (1988) unbiased correction, with the
heteroskedasticity-robust `z*(q)`. `VR < 1` is mean reversion, `= 1` a random
walk, `> 1` trending. Heteroskedasticity-robust matters here: FX volatility
clusters hard, and the homoskedastic statistic would over-reject.
""")

co(r"""
# ===========================================================================
# Lo-MacKinlay variance ratio, overlapping, heteroskedasticity-robust z*.
# ===========================================================================
def variance_ratio(logp, q):
    '''VR(q) and robust z*(q) on a log-price series. Causal: no forward info.'''
    p = np.asarray(logp, dtype=float)
    p = p[np.isfinite(p)]
    n = len(p) - 1                      # number of 1-bar returns
    if n < 10 * q:
        return np.nan, np.nan
    mu = (p[-1] - p[0]) / n
    r = np.diff(p)
    dev = r - mu
    sig_a = np.sum(dev ** 2) / (n - 1)
    # overlapping q-bar deviations
    qd = p[q:] - p[:-q] - q * mu
    m = q * (n - q + 1) * (1.0 - q / n)
    sig_c = np.sum(qd ** 2) / m
    vr = sig_c / sig_a
    # heteroskedasticity-robust variance of VR (Lo-MacKinlay theorem 4)
    d2 = dev ** 2
    denom = np.sum(d2) ** 2
    theta = 0.0
    for j in range(1, q):
        num = np.sum(d2[j:] * d2[:-j])
        # Lo-MacKinlay delta_j = num / (sum of squared devs)^2. The SQUARED
        # denominator already carries the 1/n normalisation -- multiplying by n
        # here inflates theta ~n-fold and drives every z* to zero. Proven to
        # bite by the synthetic controls below.
        delta = num / denom
        theta += (2.0 * (q - j) / q) ** 2 * delta
    z = (vr - 1.0) / np.sqrt(theta) if theta > 0 else np.nan
    return float(vr), float(z)


def autocorr(r, lags):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    r = r - r.mean()
    d = np.sum(r ** 2)
    return {L: (float(np.sum(r[L:] * r[:-L]) / d) if d > 0 and L < len(r)
                else np.nan) for L in lags}


print('variance_ratio() + autocorr() ready. VR uses OVERLAPPING windows with the '
      'Lo-MacKinlay unbiased correction and the heteroskedasticity-ROBUST z*.')
""")

co(r"""
# ===========================================================================
# MACHINERY PROOF -- the VR statistic must BITE on series whose answer is
# known by construction, before it is trusted on real data. Same discipline as
# the G4 degenerate stub in intraday_fx.ipynb: the test is TESTED, not trusted.
# ===========================================================================
_rng = np.random.default_rng(42)
_n, _q = 40000, 24
print(f'{"control series":<34}{"VR(q)":>9}{"z*":>9}   expected')
print('-' * 78)

# 1. pure random walk -> VR ~ 1, |z*| small
_rw = np.cumsum(_rng.standard_normal(_n))
_v, _z = variance_ratio(_rw, _q)
print(f'{"random walk (iid increments)":<34}{_v:>9.3f}{_z:>9.2f}   VR ~ 1, |z*| small')
assert abs(_v - 1.0) < 0.15, f'VR on a random walk should be ~1, got {_v}'
assert abs(_z) < 3.0, f'|z*| on a random walk should be small, got {_z}'

# 2. strongly MEAN-REVERTING increments -> VR < 1, z* very negative
_e = _rng.standard_normal(_n)
_mr_r = _e[1:] - 0.5 * _e[:-1]           # MA(1) with negative coefficient
_mr = np.concatenate([[0.0], np.cumsum(_mr_r)])
_v, _z = variance_ratio(_mr, _q)
print(f'{"mean-reverting MA(1), theta=-0.5":<34}{_v:>9.3f}{_z:>9.2f}   VR < 1, z* << -2')
assert _v < 0.9, f'VR should be well below 1 on a mean-reverting series, got {_v}'
assert _z < -2.0, f'z* MUST detect strong mean reversion, got {_z}'

# 3. TRENDING increments -> VR > 1, z* very positive
_tr_r = _e[1:] + 0.5 * _e[:-1]           # MA(1) with positive coefficient
_tr = np.concatenate([[0.0], np.cumsum(_tr_r)])
_v, _z = variance_ratio(_tr, _q)
print(f'{"trending MA(1), theta=+0.5":<34}{_v:>9.3f}{_z:>9.2f}   VR > 1, z* >> +2')
assert _v > 1.1, f'VR should be well above 1 on a trending series, got {_v}'
assert _z > 2.0, f'z* MUST detect strong trending, got {_z}'

print('-' * 78)
print('PROOF PASS: the VR statistic separates mean-reverting, random-walk and')
print('trending series by construction, and z* reaches significance on both')
print('non-null controls. Only now is it applied to the real FX series.')
""")

# ===========================================================================
md(r"""
## 4. The decisive test — does `mom_3d` anti-predict?

Every bar is bucketed into quintiles of `mom_3d`, the shipped `TREND_FEATURE`,
computed by the harvested engine at this run's `BARS_PER_DAY`. Mean forward
return is taken per quintile per horizon; the reported statistic is the HAC
(Newey–West, Bartlett, lag = *h*−1) *t* on **top quintile minus bottom
quintile**, averaged over the horizons.

Causal: `mom_3d` at bar *t* reads bars ≤ *t*. The forward return is
retrospective grading only — it never enters a bucket assignment.
""")

co(r"""
# ===========================================================================
# The 6 series, then: VR curve, autocorrelation, and the mom_3d quintile test.
# ===========================================================================
def fwd_ret_pct(close_arr, h):
    f = np.full(len(close_arr), np.nan)
    if h < len(close_arr):
        f[:len(close_arr) - h] = (close_arr[h:] / close_arr[:-h] - 1.0) * 100.0
    return f


N_Q = 5


def quintile_test(trend, close_arr, horizons):
    '''Mean fwd return by mom_3d quintile + HAC t on (top - bottom).'''
    tr = np.asarray(trend, dtype=float)
    ok = np.isfinite(tr)
    edges = np.nanquantile(tr[ok], np.linspace(0, 1, N_Q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bucket = np.full(len(tr), -1, dtype=int)
    bucket[ok] = np.clip(np.digitize(tr[ok], edges[1:-1]), 0, N_Q - 1)

    means = np.full((N_Q, len(horizons)), np.nan)
    ts, ds = [], []
    grp = np.where(bucket == N_Q - 1, 1.0, np.where(bucket == 0, -1.0, 0.0))
    for hi, h in enumerate(horizons):
        f = fwd_ret_pct(close_arr, h)
        for b in range(N_Q):
            v = f[bucket == b]
            v = v[np.isfinite(v)]
            if len(v):
                means[b, hi] = float(np.mean(v))
        d, se, t, npos, nneg = _w_hac_contrast(f, grp, lag=h - 1)
        if np.isfinite(t):
            ts.append(t)
        if np.isfinite(d):
            ds.append(d)
    return dict(means=means, edges=edges,
                t_mean=float(np.mean(ts)) if ts else np.nan,
                d_mean=float(np.mean(ds)) if ds else np.nan,
                per_h_t=ts)


MR_ROWS = []
VR_Q_MULT = (1, 3, 5, 10, 20)      # in DAYS, mapped to bars per timeframe
AC_LAGS = (1, 2, 3, 6, 12, 24, 48, 120)

assert FX_OK, f'FX data unavailable ({FX_FAIL}); nothing is substituted.'

# The series are built EXACTLY as the main grid builds them: 1h straight from
# the file, 2h from FX_H2 (the downward resample), close as float, and the vol
# proxy taken under this run's bars_per_day.
for p in FX_PAIRS:
    for tf, hours, bpd in FX_TFS:
        rid = f"{p['name']}@{tf}"
        bars = FX_H1[p['sym']] if hours == 1 else FX_H2[p['sym']]
        close = bars['close'].astype(float)
        close.name = p['sym']
        with bars_per_day(bpd):
            vol = realised_vol_proxy(close, bpd)
        prep = prepare_run(close, vol, bpd)
        pair_id = p['name']
        trend = prep['trend_raw']
        close_arr = prep['close'].values
        logp = np.log(close_arr)
        horizons = [d * bpd for d in MOM_DAYS]

        vr = {}
        for dmult in VR_Q_MULT:
            vr[dmult] = variance_ratio(logp, dmult * bpd)
        ac = autocorr(np.diff(logp), AC_LAGS)
        qt = quintile_test(trend, close_arr, horizons)

        MR_ROWS.append(dict(run_id=rid, pair=pair_id, tf=tf, bpd=bpd,
                            n=len(close_arr), vr=vr, ac=ac, qt=qt,
                            horizons=horizons))
        print(f'{rid:<14} {len(close_arr):>6} bars   '
              f'VR(1d)={vr[1][0]:.3f}  VR(5d)={vr[5][0]:.3f}  '
              f'quintile Q5-Q1 d={qt["d_mean"]:+.4f}%  t={qt["t_mean"]:+.2f}')

print(f'\n{len(MR_ROWS)} series measured. NO HMM was fitted; NO label was emitted.')
""")

# ===========================================================================
md(r"""
## 5. Results
""")

co(r"""
# ===========================================================================
# TABLE 1 -- variance ratios. VR < 1 = mean reverting, = 1 random walk.
# ===========================================================================
print('=' * 118)
print('VARIANCE RATIO  VR(q) = Var(q-bar ret) / (q * Var(1-bar ret))   '
      'overlapping, Lo-MacKinlay, robust z*')
print('VR < 1 MEAN REVERTING   VR = 1 RANDOM WALK   VR > 1 TRENDING   '
      f'|z*| >= {VR_Z_BAR:.0f} = distinguishable from 1')
print('=' * 118)
hdr = f"{'series':<14}" + ''.join(f'{f"VR({d}d)":>10}{"z*":>8}' for d in VR_Q_MULT)
print(hdr)
print('-' * len(hdr))
for r in MR_ROWS:
    line = f"{r['run_id']:<14}"
    for d in VR_Q_MULT:
        v, z = r['vr'][d]
        line += f'{v:>10.3f}{z:>8.1f}'
    print(line)
print('-' * len(hdr))
print('q is expressed in DAYS and converted to bars per timeframe, so the same '
      'column\nmeans the same wall-clock horizon at 1h and at 2h.')
""")

co(r"""
# ===========================================================================
# TABLE 2 -- THE DECISIVE ONE. mom_3d quintile -> mean forward return.
# ===========================================================================
print('=' * 118)
print('mom_3d QUINTILE -> MEAN FORWARD RETURN (%), averaged over the 1/3/5-day '
      'horizons')
print('Q1 = LOWEST past momentum ... Q5 = HIGHEST. If the detector\'s trend input '
      'PREDICTS, this rises.')
print('If it ANTI-predicts (the mean-reversion hypothesis), it FALLS.')
print('=' * 118)
print(f"{'series':<14}{'Q1':>10}{'Q2':>10}{'Q3':>10}{'Q4':>10}{'Q5':>10}"
      f"{'Q5-Q1':>11}{'HAC t':>9}   direction")
print('-' * 118)
for r in MR_ROWS:
    m = np.nanmean(r['qt']['means'], axis=1)
    d, t = r['qt']['d_mean'], r['qt']['t_mean']
    arrow = 'FALLING (anti-predicts)' if d < 0 else 'RISING (predicts)'
    sig = '' if abs(t) >= HAC_T_BAR else '   [|t| < 2]'
    print(f"{r['run_id']:<14}" + ''.join(f'{v:>10.4f}' for v in m)
          + f'{d:>11.4f}{t:>9.2f}   {arrow}{sig}')
print('-' * 118)
_neg = [r for r in MR_ROWS if r['qt']['d_mean'] < 0]
_negsig = [r for r in _neg if abs(r['qt']['t_mean']) >= HAC_T_BAR]
print(f'{len(_neg)} of {len(MR_ROWS)} series FALL (top quintile below bottom); '
      f'{len(_negsig)} of those reach |HAC t| >= {HAC_T_BAR:.0f}.')

# ---------------------------------------------------------------------------
# NOT PRE-REGISTERED -- reported as an observation, graded against nothing.
# M2 compares only the TWO END quintiles, so a noisy non-monotone curve can
# satisfy it. Whether the relation is actually monotone is a different and
# weaker question, and it is answered here rather than left implied.
# ---------------------------------------------------------------------------
print()
print('OBSERVATION (not pre-registered, grades nothing): is the relation '
      'actually MONOTONE?')
_mono = 0
for r in MR_ROWS:
    m = np.nanmean(r['qt']['means'], axis=1)
    dec = all(a >= b - 1e-12 for a, b in zip(m, m[1:]))
    _mono += dec
    print(f"    {r['run_id']:<14} " + ' '.join(f'{v:+.4f}' for v in m)
          + ('   monotone decreasing' if dec else '   NOT monotone'))
print(f'    {_mono} of {len(MR_ROWS)} series are monotone decreasing across all '
      f'{N_Q} quintiles.')
print('    M2 compares only Q5 vs Q1, so it can pass on a curve that is not')
print('    monotone. Read the two together, not M2 alone.')
""")

co(r"""
# ===========================================================================
# TABLE 3 -- 1-bar return autocorrelation, as a cross-check on the VR sign.
# ===========================================================================
print('=' * 118)
print('AUTOCORRELATION of 1-bar log returns (negative at short lags = '
      'bar-to-bar mean reversion)')
print('=' * 118)
hdr = f"{'series':<14}" + ''.join(f'{f"lag{L}":>10}' for L in AC_LAGS)
print(hdr)
print('-' * len(hdr))
for r in MR_ROWS:
    print(f"{r['run_id']:<14}"
          + ''.join(f"{r['ac'][L]:>10.4f}" for L in AC_LAGS))
print('-' * len(hdr))
""")

# ===========================================================================
md(r"""
## 6. Figures
""")

co(r"""
# ===========================================================================
# FIGURE 1 -- VR curves. The random-walk null is the line at 1.0.
# ===========================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.0))
_cols = plt.cm.tab10(np.linspace(0, 1, 10))
for i, r in enumerate(MR_ROWS):
    xs = list(VR_Q_MULT)
    ys = [r['vr'][d][0] for d in xs]
    zs = [r['vr'][d][1] for d in xs]
    axA.plot(xs, ys, marker='o' if r['tf'] == '1h' else 's', lw=1.4,
             color=_cols[i], label=r['run_id'])
    for x, y, z in zip(xs, ys, zs):
        if np.isfinite(z) and abs(z) >= VR_Z_BAR:
            axA.plot([x], [y], marker='*', ms=13, color=_cols[i], zorder=4)
axA.axhline(1.0, color='black', lw=1.5, ls='-')
axA.text(VR_Q_MULT[-1], 1.005, ' random walk (VR = 1)', fontsize=8,
         va='bottom', ha='right')
axA.set_xscale('log'); axA.set_xticks(list(VR_Q_MULT))
axA.set_xticklabels([f'{d}d' for d in VR_Q_MULT])
axA.set_xlabel('q (horizon, trading days)')
axA.set_ylabel('variance ratio VR(q)')
axA.set_title('Variance ratio vs the random-walk null\n'
              f'star = |z*| >= {VR_Z_BAR:.0f} (distinguishable from 1);  '
              'below the line = mean reverting', fontsize=10)
axA.legend(fontsize=7, ncol=2, loc='best')
axA.grid(alpha=0.25)

for i, r in enumerate(MR_ROWS):
    m = np.nanmean(r['qt']['means'], axis=1)
    axB.plot(range(1, N_Q + 1), m, marker='o' if r['tf'] == '1h' else 's',
             lw=1.4, color=_cols[i], label=r['run_id'])
axB.axhline(0.0, color='black', lw=1)
axB.set_xticks(range(1, N_Q + 1))
axB.set_xticklabels([f'Q{i}' for i in range(1, N_Q + 1)])
axB.set_xlabel('mom_3d quintile  (Q1 = lowest past momentum, Q5 = highest)')
axB.set_ylabel('mean forward return %')
axB.set_title("THE DECISIVE PANEL -- does the detector's own trend input predict?\n"
              'rising = predicts;  falling = ANTI-predicts (mean-reversion '
              'hypothesis)', fontsize=10)
axB.legend(fontsize=7, ncol=2, loc='best')
axB.grid(alpha=0.25)
fig.suptitle('INTRADAY FX MEAN REVERSION -- REAL DATA 2012-11-16 -> 2022-03-04, '
             'community GitHub source (ends 2022-03-04)', fontsize=11)
fig.tight_layout()
plt.show()
""")

# ===========================================================================
md(r"""
## 7. Grading M1–M3, and what it means for the hypothesis
""")

co(r"""
# ===========================================================================
# GRADING -- verbatim against the frozen rules. Not reinterpreted.
# ===========================================================================
def mverdict(ok):
    return 'CONFIRMED' if ok else 'REFUTED'


MR_GRADES = {}
_N = len(MR_ROWS)
print('=' * 118)
print(f'GRADING THE PRE-REGISTERED PREDICTIONS   ({_N} series; a MAJORITY is '
      f'>= {MR_MAJ})')
print('=' * 118)

# ---- M1 -------------------------------------------------------------------
print(f"\nM1  {MR_PREDICTIONS['M1'][0]}")
_m1_ok = []
for r in MR_ROWS:
    below = [d for d in VR_Q_MULT if np.isfinite(r['vr'][d][0]) and r['vr'][d][0] < 1]
    sig = [d for d in VR_Q_MULT
           if np.isfinite(r['vr'][d][1]) and abs(r['vr'][d][1]) >= VR_Z_BAR]
    ok = bool(below) and bool(sig)
    _m1_ok.append(ok)
    print(f"    {r['run_id']:<14} VR<1 at {len(below)}/{len(VR_Q_MULT)} horizons, "
          f"|z*|>={VR_Z_BAR:.0f} at {len(sig)}/{len(VR_Q_MULT)}   -> "
          + ('counts' if ok else 'does NOT count'))
MR_GRADES['M1'] = sum(_m1_ok) >= MR_MAJ
print(f'    {sum(_m1_ok)} of {_N} series qualify.')
print(f"    -> M1 {mverdict(MR_GRADES['M1'])}")

# ---- M2 -- THE DECISIVE ONE ----------------------------------------------
print(f"\nM2  {MR_PREDICTIONS['M2'][0]}")
_m2_ok = [r['qt']['d_mean'] < 0 for r in MR_ROWS]
for r, ok in zip(MR_ROWS, _m2_ok):
    print(f"    {r['run_id']:<14} Q5-Q1 = {r['qt']['d_mean']:+.4f}%   -> "
          + ('DECREASING' if ok else 'INCREASING'))
MR_GRADES['M2'] = sum(_m2_ok) >= MR_MAJ
print(f'    {sum(_m2_ok)} of {_N} series decrease.')
print(f"    -> M2 {mverdict(MR_GRADES['M2'])}")

# ---- M3 -------------------------------------------------------------------
print(f"\nM3  {MR_PREDICTIONS['M3'][0]}")
_m3_ok = [(r['qt']['d_mean'] < 0 and abs(r['qt']['t_mean']) >= HAC_T_BAR)
          for r in MR_ROWS]
for r, ok in zip(MR_ROWS, _m3_ok):
    print(f"    {r['run_id']:<14} Q5-Q1 = {r['qt']['d_mean']:+.4f}%  "
          f"HAC t = {r['qt']['t_mean']:+.2f}   -> "
          + ('negative AND significant' if ok else 'does NOT qualify'))
MR_GRADES['M3'] = sum(_m3_ok) >= MR_MAJ
print(f'    {sum(_m3_ok)} of {_N} series qualify.')
print(f"    -> M3 {mverdict(MR_GRADES['M3'])}")

print('\n' + '=' * 118)
print('SUMMARY:  ' + '   '.join(f'{k} {mverdict(v)}' for k, v in MR_GRADES.items()))
print('=' * 118)
""")

co(r"""
# ===========================================================================
# THE VERDICT ON THE HYPOTHESIS -- decided by the frozen rule, in plain words.
# ===========================================================================
print('=' * 118)
print('DOES MEAN REVERSION EXPLAIN THE NEGATIVE BULL-BEAR CONTRAST?')
print('=' * 118)
print('  Recall what intraday_fx.ipynb measured: all 6 OOS BULL-BEAR contrasts')
print('  NEGATIVE, 5-label ordering holding on 0 of 6 IS and 0 of 6 OOS spans.')
print()
if MR_GRADES['M2'] and MR_GRADES['M3']:
    print('  HYPOTHESIS CONFIRMED. The detector\'s OWN trend input anti-predicts')
    print('  forward return on a majority of the series, significantly. RegDet')
    print('  labels BULL after an up-move; on this data an up-move is followed by')
    print('  a DOWN move often enough to flip the sign of the BULL-BEAR contrast.')
    print('  Mean reversion at the detector\'s input is a SUFFICIENT explanation.')
    print('  IMPLICATION: the failure is not a bug and not a tuning miss. A')
    print('  trend-following labeller is structurally mismatched to a')
    print('  mean-reverting series -- making it FASTER chases the reversion')
    print('  harder, it does not fix the sign.')
elif MR_GRADES['M2'] and not MR_GRADES['M3']:
    print('  HYPOTHESIS PARTIALLY SUPPORTED, and only in the weakest sense. The')
    print('  direction is consistent -- the trend input does anti-predict on a')
    print('  majority -- but the effect does not clear the significance bar, so')
    print('  it cannot carry the explanation. The mechanism is NOT settled.')
    if not MR_GRADES['M1'] and _mono == 0:
        print()
        print('  AND THE COMBINED EVIDENCE POINTS SOMEWHERE ELSE. M1 is REFUTED --')
        print('  no series reaches |z*| >= 2, so none is distinguishably different')
        print(f'  from a random walk -- and {_mono} of {_N} quintile curves are monotone.')
        print('  M2 passes on the two END quintiles of curves that are otherwise')
        print('  shapeless. Read together, that is not a mean-reverting market: it')
        print('  is a NEAR-RANDOM-WALK one.')
        print()
        print('  THE LIKELIER READING: RegDet does not fail on intraday FX because')
        print('  the market moves AGAINST it. It fails because at these horizons')
        print('  there is almost no directional forward-return structure to find,')
        print('  in EITHER direction. That is consistent with the main grid, where')
        print('  every OOS contrast was negative but NOT ONE reached |t| >= 2 --')
        print('  noise scattered around zero, not a systematic anti-signal.')
        print()
        print('  CONSEQUENCE FOR ANY "MAKE IT MORE SENSITIVE" FIX: sensitivity')
        print('  changes what the detector reacts TO, not whether the reaction is')
        print('  informative. Against a near-random-walk series a faster detector')
        print('  should raise the switch rate W and lower the lag L while leaving')
        print('  the forward-return ordering where it is. That is a PREDICTION,')
        print('  measurable by re-running the main grid at a shorter lookback --')
        print('  it is NOT established by this notebook.')
else:
    print('  HYPOTHESIS REFUTED. The detector\'s own trend input does NOT')
    print('  anti-predict forward return on a majority of the series, so mean')
    print('  reversion at the input does NOT explain the negative BULL-BEAR')
    print('  contrast. The cause is still UNKNOWN. Candidates this notebook did')
    print('  not test: the HMM state assignment, the realised-vol proxy standing')
    print('  in for VIX, the frozen bar-count windows (EFF_WIN / VOL_WIN /')
    print('  VOL_FAST do not scale with BARS_PER_DAY), or something unidentified.')
    print('  Saying "unknown" is the honest answer here, not a placeholder.')
print()
print(f"  M1 (does the SERIES mean-revert) graded {mverdict(MR_GRADES['M1'])}, and by the")
print('  frozen rule M1 alone settles nothing: a series can mean-revert without')
print('  that being what drives this particular detector\'s labels.')
print()
print('  SCOPE: 3 pairs, 2 timeframes, 2012-11-16 -> 2022-03-04, community')
print('  GitHub data. No Nifty comparison was run. Nothing here is a return,')
print('  a Sharpe, or a tradeability claim.')
print('=' * 118)
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
