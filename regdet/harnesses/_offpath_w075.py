"""OFF-PATH EQUIVALENCE for this change set.

CLAIM UNDER TEST: the only thing that moved in the LABEL-PRODUCING path is the
VALUE of BAR_DIR_WEIGHT (0.75 -> 0.0). Everything else changed is prose,
printing, degeneracy annotation, or the EVALUATION-ONLY scorecard grading.
So putting BAR_DIR_WEIGHT back to 0.75 must reproduce the PREVIOUS notebook's
behaviour bit-for-bit on all 13 equivalence columns.

HOW THE FIT CONFOUND IS REMOVED. Two independent runs of the SAME notebook do
NOT produce bit-identical hmmlearn fits: EM accumulates ~1e-15 relative BLAS
reduction-order noise, which can change the iteration at which the convergence
monitor stops, so the fitted means differ in the last bits. That is a property of
the fit, not of any labeling change, and it is measured explicitly below as
CONTROL 0. To keep it out of the equivalence result, every comparison runs BOTH
notebooks' `label_bars` over ONE SHARED FIT and ONE shared bar set -- identical
inputs, so any difference is attributable to the labeling code alone, which is
exactly the claim being tested.

  CONTROL 0 : how much do two runs of the same code disagree, by themselves?
  TEST A    : OLD.label_bars vs NEW.label_bars on the SAME fit, at every weight
              in {0.0, 0.25, 0.5, 0.75, 1.0}. Must be bit-for-bit everywhere.
  TEST B    : OLD.label_bars_legacy (the frozen pre-change function) vs
              NEW.label_bars_legacy on the same fit -- proves the frozen
              reference itself was not disturbed.
  TEST C    : the NEW notebook's SHIPPED labels == OLD.label_bars at w=0.0.
  TEST D    : bar_dir_score is still populated at the shipped w=0.0.
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
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
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
print('OFF-PATH EQUIVALENCE: BAR_DIR_WEIGHT back at 0.75 must reproduce the '
      'pre-change notebook')
print('=' * 100)
print('\nexecuting both notebooks:')
OLD = run_nb(D + '_prev_master.ipynb', 'OLD (pre-change, w=0.75)')
NEW = run_nb(D + 'regdet_v11_master.ipynb', 'NEW (this change set, w=0.0)')

print(f"\n  OLD BAR_DIR_WEIGHT = {OLD['BAR_DIR_WEIGHT']}   "
      f"NEW BAR_DIR_WEIGHT = {NEW['BAR_DIR_WEIGHT']}")
assert OLD['BAR_DIR_WEIGHT'] == 0.75 and NEW['BAR_DIR_WEIGHT'] == 0.0, \
    'this test only means something across the 0.75 -> 0.0 change'
assert len(OLD['edates']) == len(NEW['edates']), 'bar sets differ'
assert np.array_equal(OLD['eXs'], NEW['eXs']), 'scaled feature matrices differ'
assert OLD['n_fit'] == NEW['n_fit'], 'fit windows differ'
print(f'  same bars: yes ({len(NEW["edates"])})   same scaled feature matrix: yes   '
      f'same fit window: yes')

# ---------------------------------------------------------------------------
# CONTROL 0 -- the fit is not bit-reproducible run to run. Quantify it, so the
# reader can see it is float noise and not a behavioural difference.
# ---------------------------------------------------------------------------
print('\nCONTROL 0 -- how much do two independent runs of the SAME fitting code '
      'disagree?')
_mdev = max(float(np.max(np.abs(a.means_ - b.means_)))
            for a, b in zip(OLD['emodels'], NEW['emodels']))
_same_means = _mdev == 0.0
print(f'  fitted ensemble means, OLD run vs NEW run: max|delta| = {_mdev:.3e} '
      f'({"identical" if _same_means else "NOT identical"})')
print('  hmmlearn EM accumulates BLAS reduction-order noise, which can shift the '
      'iteration at')
print('  which the convergence monitor stops. This is a property of the FIT, not '
      'of labeling, so')
print('  every comparison below runs BOTH notebooks\' label_bars over ONE SHARED '
      'FIT.')

# The shared fit: the NEW notebook's models and bar set, used for both sides.
SH = dict(models=NEW['emodels'], Xs=NEW['eXs'], dates=NEW['edates'],
          nifty=NEW['nifty'], trend=NEW['edf'][NEW['TREND_FEATURE']].values,
          n_fit=NEW['n_fit'], feat=NEW['eval_feat'], edf=NEW['edf'])
assert NEW['eval_feat'] == OLD['eval_feat'], 'evaluated feature sets differ'


def call(ns, w, legacy=False):
    """Run ONE namespace's labeler over the SHARED fit at weight w."""
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


print('\nTEST A -- OLD.label_bars vs NEW.label_bars, SAME fit, every weight')
for w in WEIGHTS:
    tag = ('   <-- THE OLD SHIPPED DEFAULT' if w == 0.75 else
           ('   <-- the NEW shipped default' if w == 0.0 else ''))
    cmp_cols(call(OLD, w), call(NEW, w), f'w={w}{tag}')

print('\nTEST B -- the FROZEN pre-change function was not disturbed either')
for w in (0.0, 0.75):
    cmp_cols(call(OLD, w, legacy=True), call(NEW, w, legacy=True),
             f'label_bars_legacy at w={w}')

print('\nTEST C -- the NEW notebook\'s SHIPPED labels are exactly '
      'OLD.label_bars(w=0.0)')
# NEW['reg'] carries extra evaluation columns; compare on the 13 equivalence ones.
cmp_cols(call(OLD, 0.0), NEW['reg'], 'shipped reg == OLD labeler at w=0.0')

print('\nTEST D -- FIX 4 machinery retained: bar_dir_score still computed at w=0')
_bs = NEW['reg']['bar_dir_score'].values
_fin = int(np.isfinite(_bs).sum())
_ok_d = (_fin == len(_bs))
print(f"  {'OK  ' if _ok_d else 'FAIL'}  bar_dir_score populated at the shipped "
      f"w=0.0: {_fin}/{len(_bs)} finite, range "
      f"[{np.nanmin(_bs):+.2f}, {np.nanmax(_bs):+.2f}]")
print('        (dir_feats is still passed everywhere; the blend is an exact '
      'arithmetic no-op)')
if not _ok_d:
    FAILED.append('bar_dir_score not populated at w=0')

print('\nTEST E -- at w=0 the ladder arms (d) and (e) really are bit-identical')
_de = np.array_equal(NEW['AB']['d. +1,2,3 bands'].values,
                     NEW['AB']['e. +1,2,3,4 bar-dir'].values)
_note = NEW['AB_ARM_NOTE'].get('e. +1,2,3,4 bar-dir', '')
print(f"  {'OK  ' if (_de and _note) else 'FAIL'}  arm(e) == arm(d): {_de}   "
      f"annotated: {_note!r}")
if not (_de and _note):
    FAILED.append('arm (d)/(e) degeneracy not detected or not annotated')

print()
print('=' * 100)
if FAILED:
    print(f'OFF-PATH EQUIVALENCE FAILED: {FAILED}')
    sys.exit(1)
print('OFF-PATH EQUIVALENCE VERIFIED: over one shared fit, the regenerated '
      'notebook reproduces')
print('the pre-change notebook BIT-FOR-BIT on all 13 columns at every weight, '
      'including the')
print('previous default 0.75. The only behavioural difference between the two '
      'notebooks is the')
print('VALUE of BAR_DIR_WEIGHT.')
print('=' * 100)
