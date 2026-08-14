"""FIX B direct test: signed grading, with KNOWN signed effects.

Three independent layers, because a fix to a grading bug that is not itself
tested is not done:

  1. UNIT     -- `_grade_directional` on hand-written (stat, expected_sign)
                 pairs whose correct grade is obvious by inspection.
  2. REPLAY   -- the two ACTUAL mis-graded rows from the user's real run
                 (`4f08e17b-v7_output_hmm_100.ipynb`) pushed back through the
                 corrected grader. These are the numbers the bug report is
                 about, so they are asserted directly rather than paraphrased.
  3. END-TO-END -- a synthesised label/return/confidence series in which
                 high-confidence bars DELIBERATELY underperform and the labels
                 are DELIBERATELY inverted, scored through the real
                 `score_regimes`. Nothing may come back PASS or WARN.

Run: python3 _test_fixb_signed_grading.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/tmp/claude-0/-home-user-SHERM-QUANTY/'
                   'f1a44349-dfe0-5e0f-9b1f-b0b39e930e1b/scratchpad')
import regime_scorecard as RS

FAILURES = []


def check(name, got, want):
    ok = (got == want)
    print(f"  {'OK  ' if ok else 'FAIL'}  {name:62s} got={got!r:14s} want={want!r}")
    if not ok:
        FAILURES.append(name)


print('=' * 100)
print('1. UNIT -- _grade_directional on hand-written signed cases')
print('=' * 100)
P, W = RS.HAC_T_PASS, RS.HAC_T_WARN          # 2.0, 1.0
dP, dW = RS.CONF_INFO_EFFECT_PASS, RS.CONF_INFO_EFFECT_WARN   # 0.20, 0.10

CASES = [
    # (stat, expected_sign, pass, warn, label, want)
    (+2.5, +1.0, P, W, 'H_BULL', 'PASS'),        # bull, strongly up   -> PASS
    (-2.5, -1.0, P, W, 'H_BEAR', 'PASS'),        # bear, strongly down -> PASS
    (+2.5, -1.0, P, W, 'H_BEAR', RS.GRADE_ANTI),  # bear, strongly UP  -> ANTI
    (-2.5, +1.0, P, W, 'H_BULL', RS.GRADE_ANTI),  # bull, strongly DOWN-> ANTI
    (+1.4, +1.0, P, W, 'L_BULL', 'WARN'),
    (+1.4, -1.0, P, W, 'L_BEAR', RS.GRADE_ANTI),  # the live L_BEAR case, scaled
    (+0.5, +1.0, P, W, 'L_BULL', 'FAIL'),        # right sign, too small -> FAIL
    (-0.5, +1.0, P, W, 'L_BULL', 'FAIL'),        # wrong sign but small  -> FAIL
    (+9.9, 0.0, P, W, 'SIDEWAYS', 'NA'),         # no expectation        -> NA
    (np.nan, +1.0, P, W, 'H_BULL', 'NA'),
    (+0.30, +1.0, dP, dW, 'H_BULL', 'PASS'),
    (-0.30, +1.0, dP, dW, 'H_BULL', RS.GRADE_ANTI),
    (-0.30, -1.0, dP, dW, 'H_BEAR', 'PASS'),     # bear: NEGATIVE d is correct
    (+0.30, -1.0, dP, dW, 'H_BEAR', RS.GRADE_ANTI),
]
for stat, es, p, w, lab, want in CASES:
    g, al, note = RS._grade_directional(stat, es, p, w, lab, 'stat')
    check(f'stat={stat:+.2f} exp_sign={es:+.0f} ({lab})', g, want)
    # the raw statistic must never be clipped or abs()'d by the grader
    if not np.isnan(al):
        assert abs(al - stat * es) < 1e-12, 'aligned stat is not stat*sign'

print()
print('=' * 100)
print("2. REPLAY -- the two rows the bug report is about, from the user's real run")
print('=' * 100)
# (a) C_CALIBRATION L_BULL @ fwd_9, d = -0.245.
#     HIGH-confidence bull bars did WORSE than low-confidence ones. Old rule
#     graded abs(-0.245) = 0.245 >= 0.20 -> PASS.
g, al, note = RS._grade_directional(-0.245, RS._expected_sign('L_BULL'),
                                    dP, dW, 'L_BULL', 'cohens_d_hi_vs_lo_conf')
old = RS._grade(abs(-0.245), dP, dW)
print(f'  C_CALIBRATION L_BULL fwd_9   d = -0.245   old={old}  new={g}')
print(f'    note: {note}')
check('L_BULL fwd_9 calibration d=-0.245 is no longer PASS', g, RS.GRADE_ANTI)
check('  ... and the OLD rule really did say PASS (bug reproduced)', old, 'PASS')

# (b) A_DISCRIMINATION L_BEAR @ fwd_3, hac_t = +1.37 with mean fwd ret +0.073%.
#     A BEAR label followed by GAINS. Old rule graded abs(1.37) -> WARN.
g2, al2, note2 = RS._grade_directional(+1.37, RS._expected_sign('L_BEAR'),
                                       P, W, 'L_BEAR', 'hac_t')
old2 = RS._grade(abs(1.37), P, W)
print(f'  A_DISCRIMINATION L_BEAR fwd_3  hac_t = +1.37   old={old2}  new={g2}')
print(f'    note: {note2}')
check('L_BEAR fwd_3 hac_t=+1.37 is no longer WARN', g2, RS.GRADE_ANTI)
check('  ... and the OLD rule really did say WARN (bug reproduced)', old2, 'WARN')

# (c) SIDEWAYS @ fwd_15, hac_t = +1.42 -- meaningless, must be NA with a reason.
g3, al3, note3 = RS._grade_directional(+1.42, RS._expected_sign('SIDEWAYS'),
                                       P, W, 'SIDEWAYS', 'hac_t')
old3 = RS._grade(abs(1.42), P, W)
print(f'  A_DISCRIMINATION SIDEWAYS fwd_15  hac_t = +1.42   old={old3}  new={g3}')
print(f'    note: {note3}')
check('SIDEWAYS fwd_15 hac_t=+1.42 is NA, not WARN', g3, 'NA')
check('  ... old rule said WARN', old3, 'WARN')
if not isinstance(note3, str) or 'no directional claim' not in note3:
    FAILURES.append('SIDEWAYS NA carries no printed reason')

print()
print('=' * 100)
print('3. END-TO-END -- score_regimes on a series built with KNOWN signed effects')
print('=' * 100)
rng = np.random.default_rng(11)
n = 2400
order = ['H_BULL', 'L_BULL', 'SIDEWAYS', 'L_BEAR', 'H_BEAR']
drift = {'H_BULL': 0.12, 'L_BULL': 0.03, 'SIDEWAYS': 0.0,
         'L_BEAR': -0.03, 'H_BEAR': -0.12}

labels, prev = [], None
while len(labels) < n:
    lab = order[rng.integers(0, len(order))]
    while lab == prev:
        lab = order[rng.integers(0, len(order))]
    labels += [lab] * max(1, int(rng.poisson(20)))
    prev = lab
labels = np.array(labels[:n], dtype=object)
rets = (np.array([drift[l] for l in labels]) + rng.normal(0, 0.5, n)) / 100.0
price = np.empty(n)
price[0] = 100.0
for t in range(1, n):
    price[t] = price[t - 1] * (1 + rets[t - 1])
te = np.clip(np.array([0.75 if l in ('H_BULL', 'H_BEAR') else 0.3 for l in labels])
             + rng.normal(0, 0.05, n), 0, 1)

# --- 3a. INVERTED LABELS: every call is exactly backwards ------------------
swap = {'H_BULL': 'H_BEAR', 'L_BULL': 'L_BEAR', 'SIDEWAYS': 'SIDEWAYS',
        'L_BEAR': 'L_BULL', 'H_BEAR': 'H_BULL'}
inv = np.array([swap[l] for l in labels], dtype=object)
conf_flat = np.clip(rng.uniform(0.6, 0.95, n), 0, 1)
sc_i, _ = RS.score_regimes(inv, price, confidence=conf_flat, trend_eff=te,
                           horizons=(3, 9, 15), bars_per_day=3)
A = sc_i[(sc_i.family == 'A_DISCRIMINATION')
         & (sc_i.label.isin(['H_BULL', 'L_BULL', 'H_BEAR', 'L_BEAR',
                             '__BULL_vs_BEAR_agg__', '__H_BULL_vs_H_BEAR__']))]
print(A[['horizon', 'label', 'mean_fwd_ret_pct', 'hac_t', 'cohens_d',
         'expected_sign', 'grade', 'grade_legacy_abs']].to_string(index=False))
check('inverted labels: zero PASS/WARN on discrimination',
      int(A['grade'].isin(['PASS', 'WARN']).sum()), 0)
check('inverted labels: the OLD rule DID score them (bug reproduced)',
      bool(A['grade_legacy_abs'].isin(['PASS', 'WARN']).any()), True)
check('inverted labels: at least one explicit ANTI grade',
      bool((A['grade'] == RS.GRADE_ANTI).any()), True)

# --- 3b. ANTI-INFORMATIVE CONFIDENCE on CORRECT labels ---------------------
# High confidence is assigned to the bars that do WORST for their own label.
fwd9 = RS._forward_returns(price, 9)
conf_anti = np.full(n, 0.5)
for lab in ('H_BULL', 'L_BULL', 'H_BEAR', 'L_BEAR'):
    m = np.flatnonzero(labels == lab)
    s = RS.LABEL_EXPECTED_SIGN[lab]
    r = np.argsort(np.argsort(-s * np.nan_to_num(fwd9[m], nan=0.0)))
    conf_anti[m] = 0.55 + 0.40 * (r / max(1, len(r) - 1))
sc_c, _ = RS.score_regimes(labels, price, confidence=conf_anti, trend_eff=te,
                           horizons=(9,), bars_per_day=3)
C = sc_c[(sc_c.family == 'C_CALIBRATION')
         & (sc_c.label.isin(['H_BULL', 'L_BULL', 'H_BEAR', 'L_BEAR']))]
print()
print(C[['horizon', 'label', 'n', 'cohens_d_hi_vs_lo_conf', 'expected_sign',
         'sign_aligned_stat', 'grade', 'grade_legacy_abs']].to_string(index=False))
check('anti-calibration: zero PASS/WARN', int(C['grade'].isin(['PASS', 'WARN']).sum()), 0)
check('anti-calibration: every directional label graded ANTI',
      bool((C['grade'] == RS.GRADE_ANTI).all()), True)
check('anti-calibration: the OLD rule graded them all PASS (bug reproduced)',
      int(C['grade_legacy_abs'].isin(['PASS', 'WARN']).sum()), len(C))
check('anti-calibration: raw signed d kept negative for BULL labels',
      bool((C.loc[C.label.isin(['H_BULL', 'L_BULL']),
                  'cohens_d_hi_vs_lo_conf'] < 0).all()), True)
check('anti-calibration: raw signed d kept positive for BEAR labels',
      bool((C.loc[C.label.isin(['H_BEAR', 'L_BEAR']),
                  'cohens_d_hi_vs_lo_conf'] > 0).all()), True)

# --- 3c. CORRECT labels + genuinely informative confidence still PASS ------
# The fix must not make everything fail: a correctly-signed effect must score.
conf_ok = np.full(n, 0.5)
for lab in ('H_BULL', 'L_BULL', 'H_BEAR', 'L_BEAR'):
    m = np.flatnonzero(labels == lab)
    s = RS.LABEL_EXPECTED_SIGN[lab]
    r = np.argsort(np.argsort(+s * np.nan_to_num(fwd9[m], nan=0.0)))
    conf_ok[m] = 0.55 + 0.40 * (r / max(1, len(r) - 1))
sc_g, _ = RS.score_regimes(labels, price, confidence=conf_ok, trend_eff=te,
                           horizons=(9,), bars_per_day=3)
G = sc_g[(sc_g.family == 'C_CALIBRATION')
         & (sc_g.label.isin(['H_BULL', 'L_BULL', 'H_BEAR', 'L_BEAR']))]
print()
print(G[['horizon', 'label', 'cohens_d_hi_vs_lo_conf', 'expected_sign',
         'sign_aligned_stat', 'grade']].to_string(index=False))
check('correctly-signed calibration still PASSes (no over-correction)',
      bool((G['grade'] == 'PASS').all()), True)

print()
print('=' * 100)
if FAILURES:
    print(f'FIX B TEST FAILED -- {len(FAILURES)} check(s): {FAILURES}')
    sys.exit(1)
print('FIX B TEST: ALL CHECKS PASSED.')
print('=' * 100)
