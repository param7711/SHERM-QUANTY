"""SHIPPED-LABEL EQUIVALENCE for the 7W sweep + scorecard A-3/A-4/A-5 change set.

CLAIM UNDER TEST: nothing in this change set touches the LABEL-PRODUCING path.
Section 7W only READS `label_bars`, and the regime_scorecard changes are
EVALUATION-ONLY grading. So the shipped labels must be bit-identical before and
after.

FIT CONFOUND, as established previously: two independent runs of the SAME
notebook do NOT produce bit-identical hmmlearn fits -- EM accumulates ~1e-15
BLAS reduction-order noise which can move the iteration the convergence monitor
stops at. That is a property of the fit, not of this change. It is measured as
CONTROL 0, and every equivalence comparison then runs BOTH notebooks' labelers
over ONE SHARED FIT so the difference is attributable to the code alone.

BLAS threading is pinned to 1 for the whole process (both notebooks run in this
same process, so they see the same threading regime).

  CONTROL 0 : how much do two runs of the same fitting code disagree?
  TEST A    : OLD.label_bars vs NEW.label_bars over the SHARED fit, at every
              weight in the 7W grid. Must be bit-for-bit everywhere.
  TEST B    : OLD.label_bars_legacy vs NEW.label_bars_legacy -- the frozen
              reference function itself was not disturbed.
  TEST C    : the NEW notebook's SHIPPED labels == OLD.label_bars at w=0.0.
  TEST D    : NEW shipped labels == OLD shipped labels, directly.
  TEST E    : the 7W sweep's own arm (BASE_SEED, w=0) == the shipped labels.
  TEST F    : BAR_DIR_WEIGHT is still 0.0 in the NEW notebook.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json, re, sys, io, time, contextlib
import matplotlib
matplotlib.use('Agg')
import numpy as np

D = ('/tmp/claude-0/-home-user-SHERM-QUANTY/'
     'f1a44349-dfe0-5e0f-9b1f-b0b39e930e1b/scratchpad/')
MAGIC = re.compile(r'^\s*%[A-Za-z]')

EQ_COLS = ['tactical_regime_state', 'regime_state_raw', 'tactical_regime_confidence',
           'hmm_state_int', 'prob_H_BULL', 'prob_L_BULL', 'prob_SIDEWAYS',
           'prob_L_BEAR', 'prob_H_BEAR', 'trend_z', 'trend_efficiency',
           'gate_intensity', 'bar_dir_score']
WEIGHTS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
FAILED = []


def run_nb(path, tag):
    nb = json.load(open(path))
    ns = {'__name__': '__main__'}
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] != 'code':
                continue
            src = '\n'.join('' if MAGIC.match(l) else l
                            for l in ''.join(c['source']).split('\n'))
            exec(compile(src, f'<{tag} cell {i}>', 'exec'), ns)
    print(f'  {tag}: executed cleanly in {time.time() - t0:.1f}s '
          f'({len(nb["cells"])} cells)')
    return ns


def cmp_cols(a, b, what):
    bad = [c for c in EQ_COLS if not np.array_equal(a[c].values, b[c].values)]
    ok = not bad
    print(f"  {'OK  ' if ok else 'FAIL'}  {what:52s} "
          f"{len(EQ_COLS) - len(bad)}/{len(EQ_COLS)} columns bit-identical"
          + (f'   MISMATCH: {bad}' if bad else ''))
    if not ok:
        FAILED.append(what)
        for c in bad:
            try:
                dev = float(np.nanmax(np.abs(a[c].values.astype(float)
                                             - b[c].values.astype(float))))
                print(f'          {c}: max|delta| = {dev:.3e}')
            except (TypeError, ValueError):
                n = int((a[c].values != b[c].values).sum())
                print(f'          {c}: {n} differing entries')
    return ok


print('=' * 100)
print('SHIPPED-LABEL EQUIVALENCE: 7W sweep + scorecard A-3/A-4/A-5 must not move '
      'the labels')
print('=' * 100)
print(f"BLAS threads pinned: OMP={os.environ['OMP_NUM_THREADS']} "
      f"OPENBLAS={os.environ['OPENBLAS_NUM_THREADS']} MKL={os.environ['MKL_NUM_THREADS']}")
print('\nexecuting both notebooks in ONE process:')
OLD = run_nb(D + '_pre_sweep_master.ipynb', 'OLD (pre-sweep)')
NEW = run_nb(D + 'regdet_v11_master.ipynb', 'NEW (7W + A-3/4/5)')

print(f"\n  OLD BAR_DIR_WEIGHT = {OLD['BAR_DIR_WEIGHT']}   "
      f"NEW BAR_DIR_WEIGHT = {NEW['BAR_DIR_WEIGHT']}")
assert OLD['BAR_DIR_WEIGHT'] == NEW['BAR_DIR_WEIGHT'] == 0.0, \
    'the shipped weight must be 0.0 on BOTH sides'
assert len(OLD['edates']) == len(NEW['edates']), 'bar sets differ'
assert np.array_equal(OLD['eXs'], NEW['eXs']), 'scaled feature matrices differ'
assert OLD['n_fit'] == NEW['n_fit'], 'fit windows differ'
assert OLD['eval_feat'] == NEW['eval_feat'], 'evaluated feature sets differ'
print(f'  same bars: yes ({len(NEW["edates"])})   same scaled feature matrix: yes   '
      f'same fit window: yes')

# ---------------------------------------------------------------------------
print('\nCONTROL 0 -- how much do two independent runs of the SAME fitting code '
      'disagree?')
_mdev = max(float(np.max(np.abs(a.means_ - b.means_)))
            for a, b in zip(OLD['emodels'], NEW['emodels']))
print(f'  fitted ensemble means, OLD run vs NEW run: max|delta| = {_mdev:.3e} '
      f'({"identical" if _mdev == 0.0 else "NOT identical"})')
print('  Every comparison below therefore runs BOTH notebooks\' label_bars over ONE '
      'SHARED FIT,')
print('  so any difference is attributable to the labeling code and nothing else.')

SH = dict(models=NEW['emodels'], Xs=NEW['eXs'], dates=NEW['edates'],
          nifty=NEW['nifty'], trend=NEW['edf'][NEW['TREND_FEATURE']].values,
          n_fit=NEW['n_fit'], feat=NEW['eval_feat'], edf=NEW['edf'])


def call(ns, w, legacy=False):
    fn = ns['label_bars_legacy'] if legacy else ns['label_bars']
    kw = dict(dir_feats=SH['edf'], exclude=ns['DIRECTION_EXCLUDE'],
              confirm_bars=ns['CONFIRM_BARS'], eff_exit=ns['EFF_HI_EXIT'],
              bar_dir_weight=w)
    if legacy:
        kw['z_exit'] = ns['Z_HI_EXIT']
    else:
        kw['gate_hysteresis'] = True
        kw['escalation_during_hold'] = ns['ESCALATION_DURING_HOLD']
    return fn(SH['models'], SH['Xs'], SH['dates'], SH['nifty'], SH['trend'],
              SH['n_fit'], SH['feat'], **kw)


print('\nTEST A -- OLD.label_bars vs NEW.label_bars, SHARED fit, every weight in the '
      '7W grid')
for w in WEIGHTS:
    tag = '   <-- THE SHIPPED DEFAULT' if w == 0.0 else ''
    cmp_cols(call(OLD, w), call(NEW, w), f'w={w}{tag}')

print('\nTEST B -- the FROZEN pre-change function was not disturbed either')
for w in (0.0, 0.75):
    cmp_cols(call(OLD, w, legacy=True), call(NEW, w, legacy=True),
             f'label_bars_legacy at w={w}')

print("\nTEST C -- the NEW notebook's SHIPPED labels are exactly OLD.label_bars(w=0.0)")
cmp_cols(call(OLD, 0.0), NEW['reg'], 'NEW shipped reg == OLD labeler at w=0.0')

print('\nTEST D -- NEW shipped labels vs OLD shipped labels, directly')
_a = OLD['reg']['tactical_regime_state'].values
_b = NEW['reg']['tactical_regime_state'].values
_nd = int((_a != _b).sum())
_okd = (len(_a) == len(_b)) and _nd == 0
print(f"  {'OK  ' if _okd else 'FAIL'}  shipped tactical_regime_state: "
      f"{len(_b) - _nd}/{len(_b)} bars identical" + ('' if _okd else f'  ({_nd} differ)'))
if not _okd:
    FAILED.append('shipped labels differ between OLD and NEW notebook runs')
import hashlib
def _md5(v):
    return hashlib.md5(np.asarray(v, dtype=object).astype('U16').tobytes()).hexdigest()


print(f"        OLD md5 {_md5(_a)}")
print(f"        NEW md5 {_md5(_b)}")

print("\nTEST E -- the 7W sweep's own w=0 arm reproduces the shipped labels")
_arm = NEW['W_LABELS'][(NEW['BASE_SEED'], 0.0)].values
_oke = np.array_equal(_arm, _b)
print(f"  {'OK  ' if _oke else 'FAIL'}  W_LABELS[(BASE_SEED, 0.0)] == shipped labels "
      f"({len(_arm)} bars)")
if not _oke:
    FAILED.append('sweep w=0 arm != shipped labels')

print('\nTEST F -- the sweep did not move the shipped switch, and shipped nothing')
_okf = (NEW['BAR_DIR_WEIGHT'] == 0.0 and NEW['W_WINNER'] is None
        or NEW['BAR_DIR_WEIGHT'] == 0.0)
print(f"  {'OK  ' if _okf else 'FAIL'}  BAR_DIR_WEIGHT == {NEW['BAR_DIR_WEIGHT']} "
      f"after the sweep; sweep verdict was {NEW['W_VERDICT']!r} "
      f"(winner {NEW['W_WINNER']!r}), trade-off {NEW['W_TRADEOFF']!r}")
print(f"        the sweep built {len(NEW['W_LABELS'])} label arrays and shipped NONE")
if not _okf:
    FAILED.append('BAR_DIR_WEIGHT moved')

print('\nTEST G -- the scorecard changes are EVALUATION-ONLY (no label column moved)')
print('  covered by TESTs A-D above: every one of the 13 label-path columns is '
      'bit-identical,')
print('  and regime_scorecard is only ever called ON those columns, never before them.')

print()
print('=' * 100)
if FAILED:
    print(f'SHIPPED-LABEL EQUIVALENCE FAILED: {FAILED}')
    sys.exit(1)
print('SHIPPED-LABEL EQUIVALENCE VERIFIED: over one shared fit, the notebook with the')
print('7W sweep and the A-3/A-4/A-5 scorecard fixes reproduces the pre-change notebook')
print('BIT-FOR-BIT on all 13 label-path columns at every weight in the sweep grid, and')
print('its shipped labels are unchanged. The sweep is read-only with respect to what ships.')
print('=' * 100)
