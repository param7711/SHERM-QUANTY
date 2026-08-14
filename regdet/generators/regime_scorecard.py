"""
regime_scorecard.py
====================

Multi-metric scorecard for evaluating a discrete market-regime label sequence
(e.g. H_BULL / L_BULL / SIDEWAYS / L_BEAR / H_BEAR) against a price series.

WHY THIS EXISTS
----------------
Previously every model/config decision in this project was ranked on ONE
number: worst-fold Sharpe of a naive long-when-bull/flat-otherwise rule. That
metric (a) collapses High/Low intensity into one bucket, (b) ignores the bear
side entirely (a flat/long rule never shorts), (c) is an economically-derived
quantity rather than a direct measure of regime quality, and (d) is noisy
(driven by a handful of trades). This module replaces it with six named
families of diagnostics (Discrimination, Persistence, Calibration, Intensity,
Coverage, Economic) that must be read TOGETHER. There is deliberately NO
single composite score -- collapsing to one number is exactly the failure
mode this module exists to fix.

SCOPE / SAFETY
--------------
- This module is EVALUATION ONLY. Every quantity here that touches "forward"
  price movement is hindsight computed strictly from bars > t and is used to
  *grade* a label produced at bar t, never to *produce* it. Do not feed any
  output of this module back into a labeling pipeline as a feature.
- Pure function: `score_regimes` has no side effects, no file I/O, no global
  mutable state, and no plotting. Only numpy / pandas / scipy(optional) are
  used.
- Missing/thin labels: any label with fewer than THIN_N observations in the
  window is reported with NaN metrics and flagged `thin=True` rather than
  raising or silently dropping.

OVERLAPPING-WINDOW SIGNIFICANCE CORRECTION
-------------------------------------------
A forward return over `horizon` bars computed at every bar t is an
overlapping / moving-window statistic: the return at t and the return at t+1
share (horizon-1) of their (horizon) price relatives. Treating the n
per-bar observations as independent (naive standard error = std/sqrt(n))
therefore understates the standard error and inflates the t-stat by a factor
that grows with the horizon. We report BOTH:
  1. naive_t   -- mean / (std/sqrt(n)), the (wrong but commonly reported) figure
  2. hac_t     -- mean / HAC(Newey-West) standard error of the mean, using a
                  Bartlett kernel with lag = horizon - 1 (the exact number of
                  bars over which two forward windows can overlap; beyond
                  that lag the theoretical autocovariance of non-overlapping
                  windows is zero, so lag = horizon-1 is the natural,
                  minimal choice rather than an arbitrary bandwidth).
  3. n_eff     -- n / horizon, a crude effective-sample-size cross-check
                  (each non-overlapping block contributes ~1 independent
                  observation). Reported alongside hac_t so the reader can
                  sanity check the HAC correction against a second, cruder
                  method.
We use the HAC t-stat (method 2) for PASS/WARN/FAIL grading because it uses
the full autocovariance structure rather than a hard block cutoff; n_eff is
reported for transparency only.

SIGNED STATISTICS ARE GRADED AGAINST THE LABEL'S OWN EXPECTED DIRECTION
------------------------------------------------------------------------
Families A (discrimination) and C (calibration) both grade SIGNED statistics --
a HAC t-stat on a label's mean forward return, and a Cohen's d between the
forward returns of high- and low-confidence bars. Both used to be graded on the
ABSOLUTE value, which cannot distinguish "the label was right" from "the label
was exactly backwards", so a detector that was reliably INVERTED scored the same
as one that was reliably correct, and could grade PASS.

They are now graded on the statistic multiplied by the label's expected sign:

    H_BULL / L_BULL   expect  +1   (the label claims the market goes UP)
    H_BEAR / L_BEAR   expect  -1   (the label claims the market goes DOWN)
    SIDEWAYS          expect   0   -> NOT GRADED, reported NA with a reason,
                                     because a signed statistic has no meaning
                                     for a label that makes no directional claim
    BULL-vs-BEAR contrasts expect +1 (bull must out-return bear)

The RAW SIGNED statistic is always kept in its own column, unclipped and never
abs()'d; only the grade uses the aligned value. A materially wrong-signed result
(aligned value <= -WARN threshold) is graded `FAIL(ANTI)` rather than plain FAIL,
because anti-signal is a different and worse finding than no signal. Every row
also carries `grade_legacy_abs`, the grade the old abs() rule would have given,
so the change is auditable row by row instead of taken on trust.

THRESHOLDS
----------
Every PASS/WARN/FAIL boundary is a named module-level constant with a
comment justifying it, so the thresholds can be argued with independently of
the code that uses them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Sequence


# ============================================================================
# NAMED THRESHOLDS (argue with these, not with the code)
# ============================================================================

# --- general ---------------------------------------------------------------
THIN_N = 20
# Below this many observations, a per-label statistic is too noisy to grade
# at all (a t-stat or effect size on <20 points is essentially a coin flip
# in either direction). Report NaN + thin=True instead of a misleading grade.

# --- A. DISCRIMINATION -------------------------------------------------------
EFFECT_SIZE_PASS = 0.20
EFFECT_SIZE_WARN = 0.10
# Cohen's d thresholds. 0.2 is the conventional "small" effect size boundary
# (Cohen 1988); below 0.1 the H_BULL-vs-H_BEAR forward-return separation is
# smaller than what's routinely called noise in the social-science
# literature this convention comes from, so FAIL below WARN.

HAC_T_PASS = 2.0
HAC_T_WARN = 1.0
# t >= 2 is the conventional ~95% two-sided significance cutoff for a roughly
# normal statistic; t in [1, 2) is directionally suggestive but not significant
# (WARN); below 1 there is no basis to reject "no separation" (FAIL). Applied to
# the HAC-corrected t-stat, not the naive (inflated) one.
#
# APPLIED TO THE SIGN-ALIGNED t, NOT |t|. See LABEL_EXPECTED_SIGN below: the t is
# multiplied by the label's expected sign first, so a bear label followed by
# significant GAINS scores as far below -2 rather than as +2. t <= -HAC_T_WARN
# after alignment is graded FAIL(ANTI): materially backwards.

MONOTONICITY_ALL_HORIZONS = "PASS"
MONOTONICITY_MAJORITY = "WARN"
# Strict 5-level monotonicity (H_BEAR < L_BEAR < SIDEWAYS < L_BULL < H_BULL
# in mean forward return) graded per horizon; PASS if monotonic at every
# horizon tested, WARN if monotonic at a majority (>50%) of horizons, FAIL
# otherwise. Direction-level monotonicity (BEAR < SIDEWAYS < BULL) graded the
# same way, since it is the weaker, more likely-to-hold claim.

# --- B. PERSISTENCE ----------------------------------------------------------
SWITCHES_PER_100_PASS = 50
SWITCHES_PER_100_WARN = 70
# Baseline: with k=5 labels drawn iid uniformly at random, expected switch
# rate is (k-1)/k = 80 per 100 bars (mean run length 1.25 bars). PASS
# requires clearly sub-random switching (mean run length >= 2 bars, <=50
# switches/100); WARN allows some drift toward random (50-70); >=70 is
# indistinguishable from the 80/100 random baseline (FAIL).

DIAG_DOMINANCE_PASS = 0.50
DIAG_DOMINANCE_WARN = 0.30
# Mean diagonal transition probability (P(stay in same label)). Random
# 5-label baseline is ~0.20 (1/k). PASS requires the diagonal to clearly
# dominate off-diagonal mass (>=0.5, i.e. more likely to persist than not on
# any given bar); WARN is above-random but not persistence-dominant (0.3-0.5);
# below 0.3 is close to the random baseline (FAIL).

DWELL_GAP_RATIO_WARN = 2.0
DWELL_GAP_RATIO_FAIL = 4.0
# Compares expected dwell time implied by the transition matrix, 1/(1-p_ii),
# against the OBSERVED run lengths. Ratio = expected / observed. >2x WARN, >4x
# FAIL.
#
# ---------------------------------------------------------------------------
# DEFECT A-4 (FIXED): this ratio used to divide a MEAN by a MEDIAN.
# ---------------------------------------------------------------------------
# 1/(1-p_ii) is the MEAN of a geometric dwell distribution. The denominator was
# `np.median(run_lengths)`. Mean and median of a geometric are NOT equal -- the
# distribution is right-skewed -- so for a PERFECTLY memoryless Markov chain the
# old ratio sat at ~1.25-2.0 depending on p, never at 1.0. Simulated over
# 400k geometric draws:
#     p=0.50 -> 2.00     p=0.70 -> 1.67     p=0.80 -> 1.25
#     p=0.90 -> 1.43     p=0.95 -> 1.43
# The metric was therefore biased ABOVE 1 before the data said anything, and the
# WARN threshold at 2.0 was being approached by nothing but the mean/median gap.
#
# THE FIX: compare MEAN TO MEAN. The denominator is now the observed MEAN run
# length, which is the direct sample estimator of the same quantity 1/(1-p_ii)
# is the population value of. Verified on simulated perfect Markov chains: the
# ratio is 1.000 to within one part in 10^4 at n=60k, and within 1-7% at n=300
# (the deviation is the single right-censored final run).
#
# The alternative offered -- comparing 1/(1-p) against the THEORETICAL geometric
# median ln(0.5)/ln(p) -- was measured and REJECTED: the observed median of a
# DISCRETE geometric is an integer (or a tie-averaged half-integer) while
# ln(0.5)/ln(p) is continuous, so that variant scored 0.70-1.02 on perfect
# Markov chains. It trades a 25-100% bias for a 0-30% discreteness bias. Not an
# improvement.
#
# ---------------------------------------------------------------------------
# WHAT THIS METRIC CAN AND CANNOT DETECT -- read before quoting its PASS.
# ---------------------------------------------------------------------------
# Once corrected to mean-to-mean the ratio is an ALGEBRAIC IDENTITY, not a
# measurement. `p_ii` here is estimated from the SAME label sequence whose runs
# are being measured. Writing S = number of stay-transitions out of label i and
# E = number of exits:
#       p_hat_ii   = S / (S + E)
#       mean_run_i = (S + E) / E          (each run ends in exactly one exit)
#   =>  1/(1 - p_hat_ii) = (S + E)/E = mean_run_i           EXACTLY.
# So the ratio is 1.0 by construction and can deviate ONLY through end-of-series
# right-censoring (the final run has no exit). Its PASS is therefore NOT evidence
# that the labels are Markov-persistent, and must never be read as such.
#
# It is retained as an INTERNAL CONSISTENCY CHECK: it still fires if the label
# array and the transition matrix ever stop describing the same object (post-hoc
# smoothing applied after the matrix was built, misaligned slicing, a
# sub-window's matrix reused on a different window). That is worth having, and
# it is honest about being all it is.
#
# OPEN ITEM (not silently invented here): the metric as ORIGINALLY INTENDED --
# "an external hysteresis/override rule is cutting runs short relative to what
# the MODEL predicts" -- requires the HMM's OWN transition matrix over hidden
# states, which is an independent object. `score_regimes` is never given it, so
# that comparison cannot be made in this module and is flagged rather than faked.

DWELL_IDENTITY_TOL = 0.15
# Slack allowed on the mean-to-mean identity before it is treated as a real
# inconsistency rather than censoring. One censored run out of R runs perturbs
# the ratio by ~mean_run/(total_bars); 15% covers that comfortably down to a few
# dozen runs while still catching a genuinely mismatched pair.

# --- C. CALIBRATION -----------------------------------------------------------
CONF_DEGENERACY_MEDIAN = 0.99
# If median confidence across ALL bars exceeds this, the confidence channel
# is degenerate (saturated near 1) and cannot be used to distinguish
# reliable from unreliable bars regardless of what the informativeness test
# below says. This mirrors the real-data run description (median confidence
# 0.999, 0.3% of bars below 0.5).

CONF_INFO_EFFECT_PASS = EFFECT_SIZE_PASS
CONF_INFO_EFFECT_WARN = EFFECT_SIZE_WARN
# Effect size (Cohen's d) between forward returns of high-confidence and
# low-confidence bars *within the same label*. Same small-effect convention
# as discrimination: if confidence doesn't separate outcomes within a label
# by even a small effect size, it is not carrying information, regardless of
# its marginal distribution.
#
# APPLIED TO THE SIGN-ALIGNED d, NOT |d|. `d` is (high-conf mean) - (low-conf
# mean), so for a BULL label confidence is informative only when d > 0 and for a
# BEAR label only when d < 0. Aligned d <= -CONF_INFO_EFFECT_WARN is
# ANTI-CALIBRATION -- confidence reliably points the wrong way, which is worse
# than confidence carrying nothing -- and is graded FAIL(ANTI), never PASS.
# SIDEWAYS has no directional expectation, so a signed d is not interpretable for
# it and those rows are NA with a reason instead of being graded on a number that
# does not mean anything. (If a SIDEWAYS calibration grade is ever wanted, the
# right quantity is a different one -- does confidence predict a SMALLER |fwd
# return| -- not this column.)

# --- D. INTENSITY GRADING ------------------------------------------------------
#
# ---------------------------------------------------------------------------
# DEFECT A-3 (FIXED): ONE threshold was applied to TWO different units.
# ---------------------------------------------------------------------------
# There used to be a single `INTENSITY_MARGIN_PASS = 0.05` used for BOTH legs of
# family D:
#   * the trend_efficiency gap  -- a DIMENSIONLESS ratio on a 0-1 scale, where
#     0.05 is 5% of the entire available range: a real bar;
#   * the |forward return| gap  -- PERCENTAGE POINTS of price move, where 0.05
#     is five HUNDREDTHS of one percent. On 2h Nifty bars the typical |fwd_9|
#     is on the order of 0.5-1.0%, so 0.05pp is ~5-10% of a typical move --
#     an all-but-automatic PASS that says nothing.
# Same number, incomparable units, and the return leg -- the ONLY informative
# leg of family D, since the efficiency leg is circular (the H gate is BUILT
# from trend efficiency) -- had effectively no bar at all.
#
# The constant is SPLIT. `INTENSITY_MARGIN_PASS` is deliberately DELETED rather
# than kept as an alias, so any surviving use of it is an ImportError/NameError
# rather than a silent revert to the conflated behaviour.

INTENSITY_MARGIN_PASS_EFFICIENCY = 0.05
# Trend-efficiency leg ONLY. Unchanged in value and in meaning: an absolute gap
# on the dimensionless 0-1 efficiency scale. 0.05 = 5 percent of the full range.
# (This leg remains CIRCULAR by construction -- see project defect 6 -- and is
# reported for completeness, not as evidence.)

INTENSITY_RETURN_GAP_PASS_FRAC = 0.20
INTENSITY_RETURN_GAP_WARN_FRAC = 0.05
# |forward return| leg ONLY, and expressed as a FRACTION OF THE DATA'S OWN
# SCALE rather than as an absolute number of percentage points, because the
# right absolute number depends entirely on the instrument, the bar size and
# the horizon -- all three of which vary within a single scorecard run (h=3 vs
# h=15 differ by ~sqrt(5) in typical magnitude alone).
#
# THE SCALE. For each direction/horizon leg, `scale` = the median |forward
# return| at that horizon POOLED OVER ALL BARS, regardless of label. This is a
# property of the price series and the horizon, and is computed WITHOUT
# reference to the H/L split being graded, so it cannot be tuned by the split.
#
# HOW THE TWO FRACTIONS WERE DERIVED (before any gap was measured):
#   * WARN floor, 0.05 of scale -- the resolution floor. The standard error of
#     a sample median is ~1.253 * sigma / sqrt(n); for a right-skewed |return|
#     distribution sigma is of the same order as the median itself, so with the
#     few-hundred bars a single label carries here the median is only pinned to
#     roughly 1.253/sqrt(300) ~ 7% of its own value, and the DIFFERENCE of two
#     such medians to ~10%. A gap below ~5% of scale is therefore inside the
#     estimation error of the two medians being differenced: it is not
#     distinguishable from zero and must not be reported as a positive finding.
#   * PASS floor, 0.20 of scale -- the materiality bar. "H" claims the bars are
#     MORE EXTREME, not marginally larger. A median move 20% bigger than the
#     series-typical move is the smallest gap that survives being described in
#     plain words as "H bars really are bigger", and it is comfortably outside
#     the ~10% resolution of the difference-of-medians above, so a PASS is a
#     measurement and not a rounding artifact.
# Neither fraction was chosen by looking at what the current labels score. The
# note printed with family D states explicitly whether the change flipped any
# grade, so the effect of the new bar is auditable rather than asserted.
#
# H < L (any NEGATIVE gap) remains FAIL regardless of size on BOTH legs: High is
# supposed to mean "more extreme" by construction, and an inversion is a
# structural failure, not a small effect.

# --- E. COVERAGE ---------------------------------------------------------------
COVERAGE_MIN_PASS = 0.05
COVERAGE_MIN_WARN = 0.03
COVERAGE_MAX_WARN = 0.40
COVERAGE_MAX_FAIL = 0.50
# A label below 5% occupancy contributes too few bars for any of the other
# families to be estimated reliably (ties to THIN_N at typical sample
# sizes); 3-5% WARN, below 3% FAIL. A label above 50% occupancy has
# effectively absorbed most of the bar-space, undermining the claim that 5
# distinct regimes are being detected (FAIL); 40-50% WARN.

# --- F. ECONOMIC (rank LAST -- most derived, noisiest) --------------------------
ECON_SHARPE_WARN_RATIO = 0.90
# PASS requires the regime long/flat strategy to beat buy-and-hold on BOTH
# Sharpe and max drawdown (strictly better risk-adjusted return AND
# strictly smaller drawdown magnitude). WARN if Sharpe is within 90% of
# buy-and-hold even if not strictly better (i.e. "not obviously worse").
# FAIL if it underperforms buy-and-hold on both axes. This is the last
# family precisely because it is the most derived (depends on an arbitrary
# long/flat rule, position sizing = 100%/0%, and trading-cost-free
# execution) and the noisiest (a handful of regime switches dominate it).

BARS_PER_YEAR_TRADING_DAYS = 252
# Standard trading-day annualization constant.


LABEL_ORDER_BEAR_TO_BULL = ["H_BEAR", "L_BEAR", "SIDEWAYS", "L_BULL", "H_BULL"]
BULL_LABELS = {"H_BULL", "L_BULL"}
BEAR_LABELS = {"H_BEAR", "L_BEAR"}
SIDEWAYS_LABELS = {"SIDEWAYS"}
HIGH_LOW_PAIRS = {"BULL": ("H_BULL", "L_BULL"), "BEAR": ("H_BEAR", "L_BEAR")}

# --- DIRECTIONAL EXPECTATION (the sign every signed statistic is graded against) --
#
# THE BUG THIS FIXES. Families A and C both used to grade on the ABSOLUTE value of
# a signed statistic -- `_grade(abs(hac_t), ...)` and `_grade(abs(d), ...)`. An
# absolute value cannot tell "the label was right" from "the label was exactly
# backwards", so a label that predicted the WRONG DIRECTION scored the same as one
# that predicted the right one, and could be graded PASS or WARN. Observed live:
#   * `L_BEAR` @ fwd_3 with hac_t = +1.37 and mean fwd ret = +0.073% -- a BEAR
#     label followed by GAINS -- graded WARN. It must FAIL.
#   * `L_BULL` @ fwd_9 calibration d = -0.245 -- HIGH-confidence bars did WORSE
#     than low-confidence ones, i.e. ANTI-calibration -- graded PASS. It must FAIL.
# Being reliably backwards is worse than carrying no information, not better.
#
# The expected sign is LABEL-DEPENDENT, and for SIDEWAYS there is NO directional
# expectation at all: a signed statistic is simply not interpretable for a label
# that makes no directional claim, so those rows are graded NA WITH A PRINTED
# REASON rather than being scored on a number that does not mean anything.
LABEL_EXPECTED_SIGN = {
    "H_BULL": +1.0, "L_BULL": +1.0,     # bull: higher forward return expected
    "H_BEAR": -1.0, "L_BEAR": -1.0,     # bear: LOWER (more negative) expected
    "SIDEWAYS": 0.0,                    # no directional claim -> not gradeable
}


def _expected_sign(label):
    """+1 if the label claims UP, -1 if it claims DOWN, 0 if it claims neither.

    Unknown labels get 0 (ungraded) rather than a guess -- inventing a direction
    for a label this module has never heard of is exactly the failure mode the
    abs() grading had.
    """
    return LABEL_EXPECTED_SIGN.get(label, 0.0)


# ============================================================================
# small stats helpers
# ============================================================================

def _grade(value, pass_thr, warn_thr, higher_is_better=True):
    """Return 'PASS'/'WARN'/'FAIL'/'NA' given a scalar and two thresholds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if higher_is_better:
        if value >= pass_thr:
            return "PASS"
        elif value >= warn_thr:
            return "WARN"
        else:
            return "FAIL"
    else:
        if value <= pass_thr:
            return "PASS"
        elif value <= warn_thr:
            return "WARN"
        else:
            return "FAIL"


GRADE_ANTI = "FAIL(ANTI)"
# A materially WRONG-SIGNED result. It is a FAIL -- the string starts with FAIL so
# no reader or downstream filter can mistake it for anything else -- but it is
# marked distinctly because it is a different failure from "no effect": the label
# is not uninformative, it is informative in reverse. Anti-signal is worse than
# no signal, and it is invisible under abs() grading.


def _grade_directional(stat, expected_sign, pass_thr, warn_thr, label, stat_name):
    """Grade a SIGNED statistic against `label`'s own directional expectation.

    Returns `(grade, aligned_stat, note)`:
      * `expected_sign == 0` (e.g. SIDEWAYS, or an unknown label) -> ("NA", nan,
        reason). The statistic is reported but NOT graded, because a signed
        quantity has no interpretation for a label that makes no directional
        claim. If the magnitude is nonetheless large, the note says so -- that is
        a real finding (the label is not behaving neutrally) and must not be lost.
      * otherwise `aligned = stat * expected_sign` and the grade is `_grade(...,
        higher_is_better=True)`, so a correctly-signed effect scores and a
        wrong-signed one cannot. `aligned <= -warn_thr` (materially backwards)
        returns GRADE_ANTI.

    The RAW SIGNED statistic is never clipped, abs()'d or overwritten by this
    function; only the grade derives from the aligned value.
    """
    _isnan = stat is None or (isinstance(stat, float) and np.isnan(stat))
    if expected_sign == 0.0:
        note = (f"{label} makes no directional claim -> signed {stat_name} is not "
                f"interpretable, NOT graded")
        if not _isnan and abs(float(stat)) >= warn_thr:
            note += (f"; but |{stat_name}|={abs(float(stat)):.2f} is large, i.e. this "
                     f"label is NOT behaving neutrally")
        return "NA", np.nan, note
    if _isnan:
        return "NA", np.nan, np.nan
    aligned = float(stat) * float(expected_sign)
    if aligned <= -warn_thr:
        _exp = "UP" if expected_sign > 0 else "DOWN"
        _got = "UP" if float(stat) > 0 else "DOWN"
        return GRADE_ANTI, aligned, (
            f"ANTI-SIGNAL: {label} predicts {_exp} but {stat_name}={float(stat):+.3f} "
            f"says {_got} -- materially backwards, worse than no signal")
    return (_grade(aligned, pass_thr, warn_thr, higher_is_better=True),
            aligned, np.nan)


def _cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled_sd


def _hac_mean_se(x, lag):
    """
    Newey-West HAC standard error of the sample mean, Bartlett kernel,
    maximum lag `lag`. x may contain NaN (dropped). Returns (se, n_used).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan, n
    lag = max(0, min(int(lag), n - 1))
    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)  # Bartlett weight
        gamma_k = np.dot(xc[: n - k], xc[k:]) / n
        var += 2.0 * w * gamma_k
    var = max(var, 0.0)  # HAC estimator can be (rarely) slightly negative
    se = np.sqrt(var / n)
    return se, n


def _naive_t(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, n
    se = np.std(x, ddof=1) / np.sqrt(n)
    if se == 0:
        return np.nan, se, n
    return np.mean(x) / se, se, n


def _forward_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """
    Percent forward return from bar t to bar t+horizon: (C[t+h]/C[t] - 1)*100.
    Last `horizon` bars are NaN (no future data available) -- this is the
    ONLY look-ahead in the module, and it exists solely to grade past
    decisions in hindsight; it must never be exposed as a live signal.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    fwd = np.full(n, np.nan)
    if horizon < n:
        fwd[: n - horizon] = (close[horizon:] / close[:n - horizon] - 1.0) * 100.0
    return fwd


def _runs(labels: np.ndarray):
    """Return list of (label, start_idx, length) for consecutive runs."""
    labels = np.asarray(labels)
    n = len(labels)
    runs = []
    if n == 0:
        return runs
    start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            runs.append((labels[start], start, i - start))
            start = i
    return runs


# ============================================================================
# family computations
# ============================================================================

def _discrimination(labels, close, horizons):
    labels = np.asarray(labels, dtype=object)
    present = [l for l in LABEL_ORDER_BEAR_TO_BULL if l in set(labels)]
    rows = []
    horizon_monotonic_5 = []
    horizon_monotonic_dir = []

    per_horizon_means = {}  # horizon -> {label: mean fwd return}
    for h in horizons:
        fwd = _forward_returns(close, h)
        means = {}
        for lab in present:
            mask = labels == lab
            x = fwd[mask]
            x_valid = x[~np.isnan(x)]
            n = len(x_valid)
            mean = np.mean(x_valid) if n else np.nan
            median = np.median(x_valid) if n else np.nan
            win = np.mean(x_valid > 0) * 100 if n else np.nan
            naive_t, naive_se, _ = _naive_t(x_valid)
            hac_se, n_used = _hac_mean_se(x_valid, lag=h - 1)
            hac_t = mean / hac_se if (hac_se and hac_se > 0 and n_used) else np.nan
            n_eff = n / h if n else np.nan
            thin = n < THIN_N
            # SIGN-AWARE GRADE. `hac_t` is signed: a BEAR label with hac_t = +1.37
            # is a bear call followed by GAINS. Under the old `abs(hac_t)` grading
            # that scored WARN, identically to a bear call followed by an equally
            # significant LOSS. It is now aligned to the label's expected sign, so
            # only a correctly-signed t can score, and a materially backwards one
            # is FAIL(ANTI). SIDEWAYS has no directional expectation and is NA
            # with a printed reason. `hac_t` itself stays raw and signed.
            _es = _expected_sign(lab)
            _gr, _al, _note = _grade_directional(
                np.nan if thin else hac_t, _es, HAC_T_PASS, HAC_T_WARN, lab, "hac_t")
            rows.append({
                "family": "A_DISCRIMINATION", "horizon": h, "label": lab,
                "n": n, "n_eff": round(n_eff, 1) if not np.isnan(n_eff) else np.nan,
                "mean_fwd_ret_pct": mean, "median_fwd_ret_pct": median,
                "win_rate_pct": win,
                "naive_t": naive_t, "hac_t": hac_t,
                "thin": thin,
                "expected_sign": _es,
                "sign_aligned_stat": _al,
                "grade": _gr,
                # The grade the pre-fix abs() rule would have produced, carried so
                # the change is auditable row by row rather than asserted.
                "grade_legacy_abs": _grade(abs(hac_t) if not thin else np.nan,
                                           HAC_T_PASS, HAC_T_WARN),
                "grade_note": _note,
            })
            means[lab] = mean

        per_horizon_means[h] = means

        # strict 5-level monotonicity (bear -> bull ascending)
        seq = [means.get(l, np.nan) for l in LABEL_ORDER_BEAR_TO_BULL if l in means]
        mono5 = _is_monotonic_increasing(seq)
        horizon_monotonic_5.append(mono5)

        bear_vals = [means[l] for l in BEAR_LABELS if l in means]
        side_vals = [means[l] for l in SIDEWAYS_LABELS if l in means]
        bull_vals = [means[l] for l in BULL_LABELS if l in means]
        bear_agg = np.nanmean(bear_vals) if bear_vals else np.nan
        side_agg = np.nanmean(side_vals) if side_vals else np.nan
        bull_agg = np.nanmean(bull_vals) if bull_vals else np.nan
        dir_seq = [v for v in [bear_agg, side_agg, bull_agg] if not np.isnan(v)]
        mono_dir = _is_monotonic_increasing(dir_seq) if len(dir_seq) == 3 else False
        horizon_monotonic_dir.append(mono_dir)

        # effect size + hac-t for BULL vs BEAR aggregate at this horizon
        fwd_full = per_horizon_means  # not used directly; recompute aggregate arrays below
        bull_mask = np.isin(labels, list(BULL_LABELS))
        bear_mask = np.isin(labels, list(BEAR_LABELS))
        fwd_h = _forward_returns(close, h)
        d_bull_bear = _cohens_d(fwd_h[bull_mask], fwd_h[bear_mask])
        hbull_mask = labels == "H_BULL"
        hbear_mask = labels == "H_BEAR"
        d_hbull_hbear = _cohens_d(fwd_h[hbull_mask], fwd_h[hbear_mask])
        # BULL-vs-BEAR contrasts have an unambiguous expected sign: BULL must
        # out-return BEAR, so d > 0. Under abs() grading a d of -0.25 -- BEAR
        # out-returning BULL, i.e. the detector inverted -- scored PASS.
        _gr_bb, _al_bb, _nt_bb = _grade_directional(
            d_bull_bear, +1.0, EFFECT_SIZE_PASS, EFFECT_SIZE_WARN,
            "__BULL_vs_BEAR_agg__", "cohens_d")
        rows.append({
            "family": "A_DISCRIMINATION", "horizon": h, "label": "__BULL_vs_BEAR_agg__",
            "n": int(bull_mask.sum() + bear_mask.sum()), "n_eff": np.nan,
            "mean_fwd_ret_pct": np.nan, "median_fwd_ret_pct": np.nan, "win_rate_pct": np.nan,
            "naive_t": np.nan, "hac_t": np.nan, "thin": (bull_mask.sum() < THIN_N or bear_mask.sum() < THIN_N),
            "cohens_d": d_bull_bear,
            "expected_sign": +1.0, "sign_aligned_stat": _al_bb,
            "grade": _gr_bb,
            "grade_legacy_abs": _grade(abs(d_bull_bear), EFFECT_SIZE_PASS, EFFECT_SIZE_WARN),
            "grade_note": _nt_bb,
        })
        _gr_hh, _al_hh, _nt_hh = _grade_directional(
            d_hbull_hbear, +1.0, EFFECT_SIZE_PASS, EFFECT_SIZE_WARN,
            "__H_BULL_vs_H_BEAR__", "cohens_d")
        rows.append({
            "family": "A_DISCRIMINATION", "horizon": h, "label": "__H_BULL_vs_H_BEAR__",
            "n": int(hbull_mask.sum() + hbear_mask.sum()), "n_eff": np.nan,
            "mean_fwd_ret_pct": np.nan, "median_fwd_ret_pct": np.nan, "win_rate_pct": np.nan,
            "naive_t": np.nan, "hac_t": np.nan, "thin": (hbull_mask.sum() < THIN_N or hbear_mask.sum() < THIN_N),
            "cohens_d": d_hbull_hbear,
            "expected_sign": +1.0, "sign_aligned_stat": _al_hh,
            "grade": _gr_hh,
            "grade_legacy_abs": _grade(abs(d_hbull_hbear), EFFECT_SIZE_PASS, EFFECT_SIZE_WARN),
            "grade_note": _nt_hh,
        })

    frac5 = np.mean(horizon_monotonic_5) if horizon_monotonic_5 else np.nan
    frac_dir = np.mean(horizon_monotonic_dir) if horizon_monotonic_dir else np.nan
    mono_grade_5 = "PASS" if frac5 == 1.0 else ("WARN" if frac5 > 0.5 else "FAIL")
    mono_grade_dir = "PASS" if frac_dir == 1.0 else ("WARN" if frac_dir > 0.5 else "FAIL")

    summary = {
        "monotonic_5level_fraction_of_horizons": frac5,
        "monotonic_5level_grade": mono_grade_5,
        "monotonic_direction_fraction_of_horizons": frac_dir,
        "monotonic_direction_grade": mono_grade_dir,
        "labels_present": present,
    }
    return pd.DataFrame(rows), summary


def _is_monotonic_increasing(seq):
    seq = [v for v in seq if not (v is None or (isinstance(v, float) and np.isnan(v)))]
    if len(seq) < 2:
        return False
    return all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))


MIN_RUNS_TO_GRADE = 5
# Fewer than this many runs of a label and its dwell statistics are not
# estimable: the median of <5 numbers, and a p_stay built on a handful of
# transitions, are noise. Rows below this are marked thin and graded "NA" --
# see the `thin` consistency fix below.


def _label_universe(labels, known_labels=None):
    """The labels a family must emit a row for: EXPECTED ones plus any observed.

    ---------------------------------------------------------------------------
    DEFECT A-5 (FIXED): families B and E used to iterate `sorted(set(labels))`.
    ---------------------------------------------------------------------------
    That is the set of labels that ACTUALLY FIRED. A regime that never fires at
    all therefore produced NO ROW -- so E_COVERAGE, the family whose entire job
    is to flag a label that is too rare, was structurally incapable of flagging
    the limiting case of a label that is infinitely rare. A detector that
    collapsed H_BEAR to zero bars scored a CLEAN coverage table, because the
    missing row simply was not there to fail. Absence of evidence was rendered
    as absence of a problem.

    The universe is now the KNOWN label list, so an absent label appears
    explicitly with n=0 and grades FAIL, plus any unexpected label the caller
    actually passed (never silently dropped either).
    """
    known = list(LABEL_ORDER_BEAR_TO_BULL if known_labels is None else known_labels)
    observed = set(np.asarray(labels, dtype=object).tolist())
    return known + sorted(observed - set(known))


def _persistence(labels, known_labels=None):
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    runs = _runs(labels)
    universe = _label_universe(labels, known_labels)
    present = sorted(set(labels.tolist()))

    rows = []
    run_lengths_by_label = {lab: [] for lab in universe}
    for lab, start, length in runs:
        run_lengths_by_label[lab].append(length)

    switches = sum(1 for i in range(1, n) if labels[i] != labels[i - 1])
    switches_per_100 = switches / n * 100 if n else np.nan

    # Transition matrix over the OBSERVED labels only -- a label with zero bars
    # has no transitions to estimate from, and padding the matrix with an
    # all-zero row would put a spurious NaN/0 into it. Its absence is reported
    # on the label's own row (p_stay = NaN) instead.
    trans_counts = pd.DataFrame(0, index=present, columns=present, dtype=float)
    for i in range(1, n):
        trans_counts.loc[labels[i - 1], labels[i]] += 1
    row_sums = trans_counts.sum(axis=1)
    trans_probs = trans_counts.div(row_sums.replace(0, np.nan), axis=0)

    diag = np.array([trans_probs.loc[l, l] if l in trans_probs.columns else np.nan for l in present])
    mean_diag = np.nanmean(diag) if len(diag) else np.nan

    for lab in universe:
        lens = run_lengths_by_label.get(lab, [])
        n_runs = len(lens)
        med = np.median(lens) if lens else np.nan
        mean_len = np.mean(lens) if lens else np.nan
        max_len = np.max(lens) if lens else np.nan
        p_ii = (trans_probs.loc[lab, lab]
                if (lab in trans_probs.index and lab in trans_probs.columns)
                else np.nan)
        expected_dwell = 1.0 / (1.0 - p_ii) if (not np.isnan(p_ii) and p_ii < 1.0) else np.nan

        # A-4 FIX: MEAN to MEAN. `expected_dwell` = 1/(1-p_ii) is the MEAN of the
        # implied geometric dwell distribution, so it is compared against the
        # observed MEAN run length, not the observed MEDIAN. See the long note at
        # DWELL_GAP_RATIO_WARN: the old mean/median ratio sat at 1.25-2.0 for a
        # PERFECTLY Markov process, i.e. it was biased before the data spoke.
        gap_ratio = (expected_dwell / mean_len
                     if (mean_len and mean_len > 0 and not np.isnan(expected_dwell))
                     else np.nan)
        # The pre-fix quantity, kept alongside so the correction is auditable row
        # by row rather than asserted. NOT graded.
        gap_ratio_legacy = (expected_dwell / med
                            if (med and med > 0 and not np.isnan(expected_dwell))
                            else np.nan)

        # `thin` CONSISTENCY FIX. Families A and C already refuse to grade a
        # statistic computed on too little data (A feeds NaN into the grader, C
        # emits "NA" outright). Family B computed `thin` and then graded anyway,
        # so a dwell verdict could be issued off two or three runs. It now
        # forces "NA", matching A and C.
        thin = n_runs < MIN_RUNS_TO_GRADE
        if thin or np.isnan(gap_ratio):
            dwell_flag = "NA"
        elif gap_ratio > DWELL_GAP_RATIO_FAIL:
            dwell_flag = "FAIL"
        elif gap_ratio > DWELL_GAP_RATIO_WARN:
            dwell_flag = "WARN"
        else:
            dwell_flag = "PASS"
        rows.append({
            "family": "B_PERSISTENCE", "label": lab, "n_runs": n_runs,
            "n_bars_label": int(np.sum(labels == lab)),
            "median_run_len": med, "mean_run_len": mean_len, "max_run_len": max_len,
            "p_stay": p_ii, "expected_dwell_from_transmat": expected_dwell,
            "dwell_gap_ratio_expected_over_observed_MEAN": gap_ratio,
            "dwell_gap_ratio_LEGACY_mean_over_median": gap_ratio_legacy,
            "dwell_gap_grade": dwell_flag,
            "thin": thin,
            "never_fires": n_runs == 0,
        })

    summary = {
        "total_bars": n, "total_runs": len(runs), "switches": switches,
        "labels_never_observed": [l for l in universe if l not in present],
        "dwell_ratio_is_an_identity": (
            "1/(1-p_hat_ii) IS the observed mean run length by construction when "
            "p_hat is estimated from these same labels; this ratio is a "
            "CONSISTENCY CHECK, not evidence of Markov persistence. See the note "
            "at DWELL_GAP_RATIO_WARN."),
        "switches_per_100_bars": switches_per_100,
        "switches_grade": _grade(switches_per_100, SWITCHES_PER_100_PASS, SWITCHES_PER_100_WARN, higher_is_better=False),
        "mean_diag_transition_prob": mean_diag,
        "diag_dominance_grade": _grade(mean_diag, DIAG_DOMINANCE_PASS, DIAG_DOMINANCE_WARN),
        "transition_matrix": trans_probs,
    }
    return pd.DataFrame(rows), summary


def _calibration(labels, close, confidence, horizons):
    labels = np.asarray(labels, dtype=object)
    rows = []
    if confidence is None:
        return pd.DataFrame(rows), {
            "available": False, "median_confidence": np.nan,
            "degeneracy_flag": "NA",
        }
    conf = np.asarray(confidence, dtype=float)
    valid_conf = conf[~np.isnan(conf)]
    median_conf = np.median(valid_conf) if len(valid_conf) else np.nan
    iqr = (np.percentile(valid_conf, 75) - np.percentile(valid_conf, 25)) if len(valid_conf) else np.nan
    frac_above_99 = np.mean(valid_conf > 0.99) if len(valid_conf) else np.nan
    degeneracy = median_conf > CONF_DEGENERACY_MEDIAN if not np.isnan(median_conf) else False

    present = sorted(set(labels.tolist()))
    for h in horizons:
        fwd = _forward_returns(close, h)
        for lab in present:
            mask = (labels == lab) & ~np.isnan(conf)
            if mask.sum() < THIN_N:
                rows.append({
                    "family": "C_CALIBRATION", "horizon": h, "label": lab,
                    "n": int(mask.sum()), "thin": True,
                    "cohens_d_hi_vs_lo_conf": np.nan,
                    "expected_sign": _expected_sign(lab),
                    "sign_aligned_stat": np.nan, "grade": "NA",
                    "grade_legacy_abs": "NA", "grade_note": np.nan,
                })
                continue
            lab_conf = conf[mask]
            lab_fwd = fwd[mask]
            med = np.median(lab_conf)
            hi = lab_fwd[lab_conf >= med]
            lo = lab_fwd[lab_conf < med]
            d = _cohens_d(hi, lo)
            # SIGN-AWARE GRADE. `d` is (mean fwd of HIGH-confidence bars) minus
            # (mean fwd of LOW-confidence bars) -- a SIGNED quantity. For a BULL
            # label, confidence carries information only if d > 0; for a BEAR
            # label only if d < 0 (more confident bear -> more negative return).
            # Under the old `abs(d)` grading, `L_BULL` @ fwd_9 with d = -0.245 --
            # high-confidence bull bars doing WORSE than low-confidence ones, i.e.
            # ANTI-calibration -- was graded PASS. SIDEWAYS has no directional
            # expectation, so a signed d is meaningless there and is NOT graded.
            # The raw SIGNED d stays in the column, unclipped.
            _es = _expected_sign(lab)
            _gr, _al, _note = _grade_directional(
                d, _es, CONF_INFO_EFFECT_PASS, CONF_INFO_EFFECT_WARN, lab,
                "cohens_d_hi_vs_lo_conf")
            rows.append({
                "family": "C_CALIBRATION", "horizon": h, "label": lab,
                "n": int(mask.sum()), "thin": False,
                "cohens_d_hi_vs_lo_conf": d,
                "expected_sign": _es,
                "sign_aligned_stat": _al,
                "grade": _gr,
                "grade_legacy_abs": _grade(abs(d), CONF_INFO_EFFECT_PASS,
                                           CONF_INFO_EFFECT_WARN),
                "grade_note": _note,
            })

    summary = {
        "available": True,
        "median_confidence": median_conf,
        "confidence_iqr": iqr,
        "frac_above_0.99": frac_above_99,
        "degeneracy_flag": "WARN(degenerate)" if degeneracy else "PASS(not degenerate)",
    }
    return pd.DataFrame(rows), summary


def _intensity(labels, close, trend_eff, horizons):
    labels = np.asarray(labels, dtype=object)
    rows = []
    for direction, (h_lab, l_lab) in HIGH_LOW_PAIRS.items():
        h_mask = labels == h_lab
        l_mask = labels == l_lab
        thin = h_mask.sum() < THIN_N or l_mask.sum() < THIN_N

        # trend efficiency H vs L
        if trend_eff is not None:
            te = np.asarray(trend_eff, dtype=float)
            h_te = te[h_mask]
            l_te = te[l_mask]
            h_te = h_te[~np.isnan(h_te)]
            l_te = l_te[~np.isnan(l_te)]
            med_h_te = np.median(h_te) if len(h_te) else np.nan
            med_l_te = np.median(l_te) if len(l_te) else np.nan
            gap_te = med_h_te - med_l_te if not (np.isnan(med_h_te) or np.isnan(med_l_te)) else np.nan
        else:
            med_h_te = med_l_te = gap_te = np.nan

        # `thin` CONSISTENCY FIX (matches families A and C): if either side of the
        # H/L pair has too few bars, the two medians being differenced are not
        # estimable and NO grade is issued. Family D used to compute `thin` and
        # then grade regardless.
        if thin:
            grade_te = "NA"
        elif gap_te is None or (isinstance(gap_te, float) and np.isnan(gap_te)):
            grade_te = "NA"
        elif gap_te < 0:
            grade_te = "FAIL"
        elif gap_te >= INTENSITY_MARGIN_PASS_EFFICIENCY:
            grade_te = "PASS"
        else:
            grade_te = "WARN"

        rows.append({
            "family": "D_INTENSITY", "direction": direction, "metric": "trend_efficiency",
            "median_H": med_h_te, "median_L": med_l_te, "gap_H_minus_L": gap_te,
            "n_H": int(h_mask.sum()), "n_L": int(l_mask.sum()),
            # This leg is graded on an ABSOLUTE gap because trend efficiency is a
            # dimensionless 0-1 quantity with a fixed, instrument-independent
            # scale. The |return| leg below is NOT, and is graded relative to the
            # data's own scale -- that is defect A-3.
            "gap_threshold_used": INTENSITY_MARGIN_PASS_EFFICIENCY,
            "gap_units": "dimensionless (0-1 efficiency scale)",
            "thin": thin, "grade": grade_te,
            "grade_legacy_conflated_units": (
                "NA" if (gap_te is None or (isinstance(gap_te, float) and np.isnan(gap_te)))
                else ("FAIL" if gap_te < 0
                      else ("PASS" if gap_te >= 0.05 else "WARN"))),
        })

        # |forward return| magnitude H vs L, per horizon
        for h in horizons:
            fwd = _forward_returns(close, h)
            h_abs = np.abs(fwd[h_mask])
            l_abs = np.abs(fwd[l_mask])
            h_abs = h_abs[~np.isnan(h_abs)]
            l_abs = l_abs[~np.isnan(l_abs)]
            med_h = np.median(h_abs) if len(h_abs) else np.nan
            med_l = np.median(l_abs) if len(l_abs) else np.nan
            gap = med_h - med_l if not (np.isnan(med_h) or np.isnan(med_l)) else np.nan

            # A-3 FIX. The threshold for THIS leg is derived from the DATA'S OWN
            # SCALE at this horizon, because the gap is in percentage points of
            # price and 0.05pp means something completely different at h=3 than
            # at h=15, and different again on a different instrument. `scale` is
            # the median |fwd return| over ALL bars at this horizon -- pooled,
            # label-blind, so the H/L split being graded cannot influence its own
            # bar.
            all_abs = np.abs(fwd)
            all_abs = all_abs[~np.isnan(all_abs)]
            scale = np.median(all_abs) if len(all_abs) else np.nan
            rel_gap = (gap / scale
                       if (not np.isnan(scale) and scale > 0 and not np.isnan(gap))
                       else np.nan)
            pass_abs = (scale * INTENSITY_RETURN_GAP_PASS_FRAC
                        if not np.isnan(scale) else np.nan)
            warn_abs = (scale * INTENSITY_RETURN_GAP_WARN_FRAC
                        if not np.isnan(scale) else np.nan)

            if thin:
                grade = "NA"           # `thin` consistency fix, as above
            elif rel_gap is None or (isinstance(rel_gap, float) and np.isnan(rel_gap)):
                grade = "NA"
            elif rel_gap < 0:
                grade = "FAIL"         # inversion: H is SMALLER than L
            elif rel_gap >= INTENSITY_RETURN_GAP_PASS_FRAC:
                grade = "PASS"
            elif rel_gap >= INTENSITY_RETURN_GAP_WARN_FRAC:
                grade = "WARN"
            else:
                grade = "FAIL"         # positive but inside the medians' own noise
            rows.append({
                "family": "D_INTENSITY", "direction": direction, "metric": f"abs_fwd_ret_h{h}",
                "median_H": med_h, "median_L": med_l, "gap_H_minus_L": gap,
                "n_H": int(h_mask.sum()), "n_L": int(l_mask.sum()),
                "scale_median_abs_fwd_all_bars": scale,
                "gap_rel_to_scale": rel_gap,
                "gap_threshold_used": pass_abs,
                "gap_units": "percentage points of price (scaled by series median)",
                "thin": thin, "grade": grade,
                # What the OLD conflated 0.05-in-both-units rule would have said.
                # Carried so the A-3 correction is auditable per row.
                "grade_legacy_conflated_units": (
                    "NA" if (gap is None or (isinstance(gap, float) and np.isnan(gap)))
                    else ("FAIL" if gap < 0
                          else ("PASS" if gap >= 0.05 else "WARN"))),
            })
    return pd.DataFrame(rows)


def _coverage(labels, known_labels=None):
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    # A-5 FIX: iterate the EXPECTED label universe, not `sorted(set(labels))`.
    # A regime that never fires now gets an explicit n=0 / 0.0% row and FAILs,
    # instead of vanishing from the table and leaving it looking clean.
    universe = _label_universe(labels, known_labels)
    present = set(labels.tolist())
    rows = []
    for lab in universe:
        cnt = int(np.sum(labels == lab))
        pct = cnt / n if n else np.nan
        if pct < COVERAGE_MIN_WARN:
            grade = "FAIL"
        elif pct < COVERAGE_MIN_PASS:
            grade = "WARN"
        elif pct > COVERAGE_MAX_FAIL:
            grade = "FAIL"
        elif pct > COVERAGE_MAX_WARN:
            grade = "WARN"
        else:
            grade = "PASS"
        rows.append({
            "family": "E_COVERAGE", "label": lab, "n": cnt, "pct": pct * 100,
            "thin": cnt < THIN_N, "grade": grade,
            "never_fires": lab not in present,
            "grade_note": (f"{lab} NEVER FIRES on this run (0 bars) -- this row "
                           f"exists only because coverage now iterates the "
                           f"expected label list; before defect A-5 was fixed a "
                           f"missing regime produced no row at all and could not "
                           f"be flagged" if lab not in present else np.nan),
        })
    return pd.DataFrame(rows)


def _economic(labels, close, bars_per_day):
    labels = np.asarray(labels, dtype=object)
    close = np.asarray(close, dtype=float)
    n = len(close)
    if n < 2:
        return pd.DataFrame(), {"available": False}

    ret_1 = close[1:] / close[:-1] - 1.0  # bar t -> t+1 return, len n-1
    # position decided using label KNOWN at bar t (no look-ahead: label[t]
    # determines exposure to the t->t+1 move, which is realized strictly
    # after the decision bar)
    position = np.isin(labels[:-1], list(BULL_LABELS)).astype(float)

    strat_ret = position * ret_1
    bh_ret = ret_1

    ann_factor = np.sqrt(bars_per_day * BARS_PER_YEAR_TRADING_DAYS)

    def sharpe(r):
        r = r[~np.isnan(r)]
        if len(r) < 2 or np.std(r, ddof=1) == 0:
            return np.nan
        return np.mean(r) / np.std(r, ddof=1) * ann_factor

    def max_dd(r):
        r = np.nan_to_num(r, nan=0.0)
        eq = np.cumprod(1 + r)
        peak = np.maximum.accumulate(eq)
        dd = eq / peak - 1.0
        return dd.min() * 100 if len(dd) else np.nan

    strat_sharpe = sharpe(strat_ret)
    bh_sharpe = sharpe(bh_ret)
    strat_dd = max_dd(strat_ret)
    bh_dd = max_dd(bh_ret)
    exposure = np.mean(position) * 100 if len(position) else np.nan

    sharpe_better = (not np.isnan(strat_sharpe) and not np.isnan(bh_sharpe) and strat_sharpe > bh_sharpe)
    dd_better = (not np.isnan(strat_dd) and not np.isnan(bh_dd) and strat_dd > bh_dd)  # less negative = better
    if sharpe_better and dd_better:
        grade = "PASS"
    elif (not np.isnan(bh_sharpe) and bh_sharpe != 0 and
          not np.isnan(strat_sharpe) and strat_sharpe / bh_sharpe >= ECON_SHARPE_WARN_RATIO):
        grade = "WARN"
    else:
        grade = "FAIL"

    summary = {
        "available": True,
        "strategy_sharpe": strat_sharpe, "buy_hold_sharpe": bh_sharpe,
        "strategy_max_dd_pct": strat_dd, "buy_hold_max_dd_pct": bh_dd,
        "exposure_pct": exposure,
        "grade": grade,
    }
    return pd.DataFrame([summary]), summary


# ============================================================================
# main entry point
# ============================================================================

def score_regimes(labels, close, confidence=None, trend_eff=None,
                   horizons=(3, 9, 15), bars_per_day=3, known_labels=None):
    """
    Score a discrete regime-label sequence against a price series across six
    independent families of diagnostics. See module docstring for the full
    rationale, the overlapping-window significance correction, and every
    threshold's justification.

    Parameters
    ----------
    labels : array-like of str, length n
        Regime label at each bar. Expected label set (for monotonicity /
        intensity checks): {H_BULL, L_BULL, SIDEWAYS, L_BEAR, H_BEAR}. Other
        label sets are still scored on discrimination/persistence/coverage;
        intensity grading is skipped (NA) if H/L pairs are absent.
    close : array-like of float, length n
        Close price at each bar, index-aligned with `labels`.
    confidence : array-like of float, length n, optional
        Model confidence at each bar, in [0, 1]. If omitted, family C is
        skipped (reported unavailable, not silently zero).
    trend_eff : array-like of float, length n, optional
        Trend-efficiency (or similar path-quality) metric at each bar, used
        only for the H-vs-L intensity check in family D.
    horizons : tuple of int
        Forward-return horizons (in bars) evaluated for discrimination,
        calibration, and intensity magnitude checks.
    bars_per_day : int
        Bars per trading day, used to annualize the economic Sharpe ratio
        (2h bars on NSE cash-session hours -> ~3/day).
    known_labels : sequence of str, optional
        The labels that are EXPECTED to exist. Families B and E emit a row for
        every one of them even if it never fires (n=0 -> FAIL), plus a row for
        any unexpected label actually present. Defaults to
        LABEL_ORDER_BEAR_TO_BULL. Before defect A-5 was fixed these families
        iterated only the observed labels, so a regime that never fired produced
        no row and E_COVERAGE could not flag it.

    Returns
    -------
    scorecard_df : pd.DataFrame
        Long-format table, one row per metric, with a `family` column and a
        `grade` column (PASS/WARN/FAIL/NA where applicable).
    summary : dict
        Family-level roll-ups (fractions, degeneracy flags, overall grade
        distribution per family) plus the raw transition matrix (family B)
        for direct inspection.

    Notes
    -----
    - No look-ahead: all "forward" computations use bars strictly after t to
      grade the label already assigned at t; nothing computed here is fed
      back as a labeling input.
    - This function returns NaN/'thin'/'NA' rather than raising when a label
      has fewer than THIN_N observations in the window, so callers can run
      it safely over short walk-forward folds.
    """
    labels = np.asarray(labels, dtype=object)
    close = np.asarray(close, dtype=float)
    n = len(labels)
    assert len(close) == n, "labels and close must be the same length"
    if confidence is not None:
        assert len(confidence) == n, "confidence must match labels length"
    if trend_eff is not None:
        assert len(trend_eff) == n, "trend_eff must match labels length"

    disc_df, disc_summary = _discrimination(labels, close, horizons)
    pers_df, pers_summary = _persistence(labels, known_labels)
    calib_df, calib_summary = _calibration(labels, close, confidence, horizons)
    intens_df = _intensity(labels, close, trend_eff, horizons)
    cov_df = _coverage(labels, known_labels)
    econ_df, econ_summary = _economic(labels, close, bars_per_day)

    scorecard_df = pd.concat(
        [disc_df, pers_df, calib_df, intens_df, cov_df, econ_df],
        ignore_index=True, sort=False,
    )

    def _family_grade_counts(df):
        if "grade" not in df.columns or df.empty:
            return {}
        return df["grade"].value_counts().to_dict()

    summary = {
        "n_bars": n,
        "A_discrimination": {**disc_summary, "grade_counts": _family_grade_counts(disc_df)},
        "B_persistence": {**pers_summary, "grade_counts": _family_grade_counts(pers_df)},
        "C_calibration": {**calib_summary, "grade_counts": _family_grade_counts(calib_df)},
        "D_intensity": {"grade_counts": _family_grade_counts(intens_df)},
        "E_coverage": {"grade_counts": _family_grade_counts(cov_df)},
        "F_economic": econ_summary,
        "note": ("No single composite score is computed by design. Read families "
                 "in order A-F; F (economic) is the most derived and noisiest and "
                 "should never be used alone to rank configs."),
    }
    return scorecard_df, summary


# ============================================================================
# self-test
# ============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def _simulate(n, drift_map, order, run_len_lambda, conf_informative, te_informative):
        """
        Build a synthetic (labels, close, confidence, trend_eff) tuple.
        Labels are generated as runs (geometric-ish dwell times) cycling
        through `order`; each bar's return from t->t+1 is drawn with a
        label-dependent drift (drift_map) plus noise, so a label at bar t
        genuinely precedes (causes, in the simulation) the return realized
        after t -- there is no look-ahead in the construction.
        """
        # NOTE: regimes are picked i.i.d. at random for each run (NOT a fixed
        # deterministic cycle). A fixed cycle (e.g. H_BULL -> L_BULL ->
        # SIDEWAYS -> L_BEAR -> H_BEAR -> repeat) would make H_BEAR always
        # immediately followed by H_BULL, which contaminates longer-horizon
        # forward returns with the *next* regime's drift rather than the
        # current one (run lengths ~run_len_lambda are shorter than the
        # h=15 horizon). Random ordering avoids that artificial anti-
        # correlation so the planted per-label drift is what the forward-
        # return test actually measures.
        labels = []
        i = 0
        prev_lab = None
        while i < n:
            lab = order[rng.integers(0, len(order))]
            while lab == prev_lab:  # avoid trivial zero-length "runs" merging
                lab = order[rng.integers(0, len(order))]
            run_len = max(1, rng.poisson(run_len_lambda))
            labels.extend([lab] * run_len)
            i += run_len
            prev_lab = lab
        labels = np.array(labels[:n], dtype=object)

        drift = np.array([drift_map[l] for l in labels])
        noise = rng.normal(0, 0.5, size=n)  # pct noise per bar
        rets = (drift + noise) / 100.0
        close = 100.0 * np.cumprod(1 + rets)
        close = np.concatenate([[100.0], close])[:n]  # length n, no shift issue below
        # rebuild close properly length n with a starting price
        price = np.empty(n)
        price[0] = 100.0
        for t in range(1, n):
            price[t] = price[t - 1] * (1 + rets[t - 1])

        if conf_informative:
            # confidence higher when |drift| is larger (genuinely stronger regime)
            base = 0.5 + np.clip(np.abs(drift) / (np.abs(drift).max() + 1e-9), 0, 1) * 0.45
            confidence = np.clip(base + rng.normal(0, 0.03, n), 0.05, 0.999)
        else:
            confidence = np.clip(rng.uniform(0.9, 1.0, n), 0, 1)

        if te_informative:
            te = np.array([
                0.75 if l in ("H_BULL", "H_BEAR") else (0.35 if l in ("L_BULL", "L_BEAR") else 0.25)
                for l in labels
            ]) + rng.normal(0, 0.05, n)
            te = np.clip(te, 0, 1)
        else:
            te = np.clip(rng.uniform(0.2, 0.8, n), 0, 1)

        return labels, price, confidence, te

    n_bars = 3000
    drift_map_planted = {
        "H_BULL": 0.12, "L_BULL": 0.03, "SIDEWAYS": 0.0, "L_BEAR": -0.03, "H_BEAR": -0.12,
    }
    order = ["H_BULL", "L_BULL", "SIDEWAYS", "L_BEAR", "H_BEAR"]

    labels_p, close_p, conf_p, te_p = _simulate(
        n_bars, drift_map_planted, order, run_len_lambda=25,
        conf_informative=True, te_informative=True,
    )

    print("=" * 80)
    print("TEST 1: PLANTED SIGNAL (H_BULL genuinely precedes positive returns, etc.)")
    print("=" * 80)
    sc_p, summ_p = score_regimes(labels_p, close_p, confidence=conf_p, trend_eff=te_p,
                                  horizons=(3, 9, 15), bars_per_day=3)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 200)
    print("\n--- Discrimination rows (horizon=9) ---")
    print(sc_p[(sc_p.family == "A_DISCRIMINATION") & (sc_p.horizon == 9)]
          .to_string(index=False))
    print("\n--- Discrimination summary ---")
    for k, v in summ_p["A_discrimination"].items():
        print(f"  {k}: {v}")
    print("\n--- Persistence rows ---")
    print(sc_p[sc_p.family == "B_PERSISTENCE"].to_string(index=False))
    print("\n--- Persistence summary (excluding transition matrix) ---")
    for k, v in summ_p["B_persistence"].items():
        if k == "transition_matrix":
            continue
        print(f"  {k}: {v}")
    print("\n--- Calibration summary ---")
    for k, v in summ_p["C_calibration"].items():
        print(f"  {k}: {v}")
    print("\n--- Intensity rows ---")
    print(sc_p[sc_p.family == "D_INTENSITY"].to_string(index=False))
    print("\n--- Coverage rows ---")
    print(sc_p[sc_p.family == "E_COVERAGE"].to_string(index=False))
    print("\n--- Economic summary ---")
    for k, v in summ_p["F_economic"].items():
        print(f"  {k}: {v}")

    # assertions: planted signal must be detected
    disc_rows_p = sc_p[(sc_p.family == "A_DISCRIMINATION") & (sc_p.label == "__H_BULL_vs_H_BEAR__")]
    assert (disc_rows_p["grade"] == "PASS").all(), "planted H_BULL vs H_BEAR should PASS at every horizon"
    assert summ_p["A_discrimination"]["monotonic_5level_grade"] == "PASS", "planted 5-level monotonicity should PASS"
    assert summ_p["B_persistence"]["switches_grade"] in ("PASS", "WARN"), "planted persistence should not be random-like"
    assert summ_p["C_calibration"]["degeneracy_flag"].startswith("PASS"), "planted confidence should not be degenerate"
    print("\n[TEST 1] All planted-signal assertions PASSED.")

    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 2: RANDOM LABELS on the SAME price series (null case)")
    print("=" * 80)
    close_random = close_p.copy()  # reuse a real price path
    labels_random = rng.choice(order, size=n_bars, replace=True)  # iid, no relation to price
    conf_random = np.clip(rng.uniform(0.9, 1.0, n_bars), 0, 1)
    te_random = np.clip(rng.uniform(0.2, 0.8, n_bars), 0, 1)

    sc_r, summ_r = score_regimes(labels_random, close_random, confidence=conf_random,
                                  trend_eff=te_random, horizons=(3, 9, 15), bars_per_day=3)

    print("\n--- Discrimination rows (horizon=9) ---")
    print(sc_r[(sc_r.family == "A_DISCRIMINATION") & (sc_r.horizon == 9)]
          .to_string(index=False))
    print("\n--- Discrimination summary ---")
    for k, v in summ_r["A_discrimination"].items():
        print(f"  {k}: {v}")
    print("\n--- Persistence summary (excluding transition matrix) ---")
    for k, v in summ_r["B_persistence"].items():
        if k == "transition_matrix":
            continue
        print(f"  {k}: {v}")

    disc_rows_r = sc_r[(sc_r.family == "A_DISCRIMINATION") & (sc_r.label == "__H_BULL_vs_H_BEAR__")]
    n_fail_or_warn = (disc_rows_r["grade"] != "PASS").sum()
    assert n_fail_or_warn == len(disc_rows_r), (
        "random labels should NOT show a PASS-grade H_BULL vs H_BEAR separation at any horizon"
    )
    assert summ_r["A_discrimination"]["monotonic_5level_grade"] in ("WARN", "FAIL"), (
        "random labels should not show clean monotonicity"
    )
    assert summ_r["B_persistence"]["switches_grade"] == "FAIL", (
        "iid random labels should show random-baseline (~80/100) switch rate -> FAIL"
    )
    print("\n[TEST 2] All random-label (null case) assertions PASSED.")

    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 3: INVERTED LABELS (anti-signal) -- the abs()-grading bug")
    print("=" * 80)
    # Take the planted series and SWAP bull <-> bear. Every label is now exactly
    # backwards: H_BULL sits on the bars that fall, H_BEAR on the bars that rise.
    # The old abs()-based grading scored this IDENTICALLY to the correct labels,
    # because |t| and |d| cannot tell "right" from "exactly wrong". Nothing here
    # may grade PASS or WARN.
    _swap = {"H_BULL": "H_BEAR", "L_BULL": "L_BEAR",
             "SIDEWAYS": "SIDEWAYS", "L_BEAR": "L_BULL", "H_BEAR": "H_BULL"}
    labels_inv = np.array([_swap[l] for l in labels_p], dtype=object)
    sc_i, summ_i = score_regimes(labels_inv, close_p, confidence=conf_p, trend_eff=te_p,
                                 horizons=(3, 9, 15), bars_per_day=3)

    _dir_lab = ["H_BULL", "L_BULL", "H_BEAR", "L_BEAR",
                "__BULL_vs_BEAR_agg__", "__H_BULL_vs_H_BEAR__"]
    _inv_A = sc_i[(sc_i.family == "A_DISCRIMINATION") & (sc_i.label.isin(_dir_lab))]
    print("\n--- A_DISCRIMINATION on INVERTED labels (new grade vs the old abs() grade) ---")
    print(_inv_A[["horizon", "label", "mean_fwd_ret_pct", "hac_t", "cohens_d",
                  "expected_sign", "sign_aligned_stat", "grade",
                  "grade_legacy_abs"]].to_string(index=False))
    assert not _inv_A["grade"].isin(["PASS", "WARN"]).any(), (
        "inverted labels must not score PASS/WARN on discrimination -- abs() bug is back"
    )
    assert (_inv_A["grade"] == GRADE_ANTI).any(), (
        "inverted labels must produce at least one explicit ANTI grade"
    )
    _regressed = _inv_A["grade_legacy_abs"].isin(["PASS", "WARN"]).sum()
    print(f"\n  the OLD abs() rule graded {_regressed}/{len(_inv_A)} of these "
          f"exactly-backwards rows PASS or WARN; the new signed rule grades "
          f"{int(_inv_A['grade'].isin(['PASS', 'WARN']).sum())}.")

    # SIDEWAYS must be NA with a reason, in BOTH families, never silently graded.
    for _fam in ("A_DISCRIMINATION", "C_CALIBRATION"):
        _sw = sc_i[(sc_i.family == _fam) & (sc_i.label == "SIDEWAYS")]
        assert (_sw["grade"] == "NA").all(), f"{_fam} SIDEWAYS must be NA, not graded"
        assert _sw["grade_note"].notna().all(), f"{_fam} SIDEWAYS NA must carry a reason"
    print("  SIDEWAYS rows are NA with a printed reason in both A and C.")

    # Direct calibration anti-test: confidence deliberately ANTI-informative.
    # High-confidence bars are given the WORSE outcome within each label, so d
    # has the wrong sign for every directional label.
    _fwd15 = _forward_returns(close_p, 9)
    conf_anti = np.full(len(labels_p), 0.5)
    for _lab in ("H_BULL", "L_BULL", "H_BEAR", "L_BEAR"):
        _m = np.flatnonzero(labels_p == _lab)
        _v = _fwd15[_m]
        _sgn = LABEL_EXPECTED_SIGN[_lab]
        # rank so that the WORST-for-this-label outcomes get the HIGHEST confidence
        _r = np.argsort(np.argsort(-_sgn * np.nan_to_num(_v, nan=0.0)))
        conf_anti[_m] = 0.55 + 0.40 * (_r / max(1, len(_r) - 1))
    sc_c, _ = score_regimes(labels_p, close_p, confidence=conf_anti, trend_eff=te_p,
                            horizons=(9,), bars_per_day=3)
    _cal = sc_c[(sc_c.family == "C_CALIBRATION") & (sc_c.label.isin(
        ["H_BULL", "L_BULL", "H_BEAR", "L_BEAR"]))]
    print("\n--- C_CALIBRATION with deliberately ANTI-informative confidence ---")
    print(_cal[["horizon", "label", "n", "cohens_d_hi_vs_lo_conf", "expected_sign",
                "sign_aligned_stat", "grade", "grade_legacy_abs"]].to_string(index=False))
    assert not _cal["grade"].isin(["PASS", "WARN"]).any(), (
        "anti-calibration must never grade PASS/WARN"
    )
    assert (_cal["grade"] == GRADE_ANTI).all(), (
        "every directional label with deliberately reversed confidence must be ANTI"
    )
    print(f"  the OLD abs() rule graded "
          f"{int(_cal['grade_legacy_abs'].isin(['PASS', 'WARN']).sum())}/{len(_cal)} "
          f"of these anti-calibrated rows PASS or WARN; the new rule grades 0.")
    print("\n[TEST 3] All anti-signal assertions PASSED.")

    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 4 (defect A-4): a PERFECT Markov chain must score dwell ratio ~1.0")
    print("=" * 80)
    # Generate labels whose run lengths are EXACTLY geometric with a known stay
    # probability. There is no hysteresis, no override, no smoothing: the process
    # IS memoryless. Any dwell ratio away from 1.0 is therefore a property of the
    # METRIC, not of the data -- which is exactly what defect A-4 was.
    def _perfect_markov(n, p, order_, rg):
        out, prev = [], None
        while len(out) < n:
            lab = order_[rg.integers(0, len(order_))]
            while lab == prev:
                lab = order_[rg.integers(0, len(order_))]
            out.extend([lab] * int(rg.geometric(1.0 - p)))
            prev = lab
        return np.array(out[:n], dtype=object)

    print(f"{'p_stay':>8} | {'FIXED mean/mean':>16} | {'LEGACY mean/median':>19} | "
          f"{'legacy bias':>11}")
    print("-" * 66)
    _a4_fixed_max_dev, _a4_legacy_max_dev = 0.0, 0.0
    for _p in (0.50, 0.70, 0.80, 0.90, 0.95):
        _rg = np.random.default_rng(1234)
        _lab_mk = _perfect_markov(40000, _p, order, _rg)
        _pers_df, _pers_sm = _persistence(_lab_mk)
        _fx = _pers_df["dwell_gap_ratio_expected_over_observed_MEAN"].astype(float)
        _lg = _pers_df["dwell_gap_ratio_LEGACY_mean_over_median"].astype(float)
        _fx_m, _lg_m = float(np.nanmean(_fx)), float(np.nanmean(_lg))
        _a4_fixed_max_dev = max(_a4_fixed_max_dev, abs(_fx_m - 1.0))
        _a4_legacy_max_dev = max(_a4_legacy_max_dev, abs(_lg_m - 1.0))
        print(f"{_p:>8.2f} | {_fx_m:>16.4f} | {_lg_m:>19.4f} | "
              f"{100 * (_lg_m - 1):>+10.1f}%")
        # The whole point of the fix.
        assert abs(_fx_m - 1.0) < 0.02, (
            f"A-4: perfect Markov chain at p={_p} must score ~1.0 on the FIXED "
            f"mean-to-mean ratio, got {_fx_m:.4f}")
        assert (_pers_df["dwell_gap_grade"].isin(["PASS", "NA"])).all(), (
            f"A-4: perfect Markov chain at p={_p} must not WARN/FAIL on dwell")
    print(f"\n  FIXED  ratio deviates from 1.0 by at most {100*_a4_fixed_max_dev:.2f}% "
          f"across p in [0.5, 0.95]")
    print(f"  LEGACY ratio deviates from 1.0 by up to    {100*_a4_legacy_max_dev:.1f}% "
          f"on the SAME perfectly-Markov data -- that was the bias")
    assert _a4_legacy_max_dev > 0.20, (
        "the legacy mean/median ratio should be visibly biased; if it is not, "
        "this test is not exercising the defect it claims to")

    # And the reason the FIXED ratio is so exact: it is an algebraic identity.
    # p_hat_ii = S/(S+E) and mean_run = (S+E)/E  =>  1/(1-p_hat_ii) == mean_run.
    # Assert that identity directly, so the docstring claim is tested and not
    # merely asserted in prose. Deviation comes only from the censored final run.
    _rg = np.random.default_rng(99)
    _lab_id = _perfect_markov(20000, 0.85, order, _rg)
    _pid, _ = _persistence(_lab_id)
    _dev_id = float(np.nanmax(np.abs(
        _pid["expected_dwell_from_transmat"].astype(float)
        - _pid["mean_run_len"].astype(float))))
    print(f"  IDENTITY CHECK: max |1/(1-p_hat) - observed mean run| = {_dev_id:.2e} "
          f"bars over 5 labels / 20k bars")
    print("    -> the corrected ratio is an ALGEBRAIC IDENTITY, so its PASS is a "
          "CONSISTENCY CHECK,")
    print("       NOT evidence of Markov persistence. Reading it as evidence is "
          "the remaining trap.")
    assert _dev_id < 0.05, "mean-to-mean dwell identity should hold to censoring only"
    print("\n[TEST 4] defect A-4 assertions PASSED.")

    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 5 (defect A-5): a regime that NEVER FIRES must produce a FAIL row")
    print("=" * 80)
    # Take the planted series and delete H_BEAR entirely -- fold it into L_BEAR.
    # A detector that has collapsed one of its five regimes to zero bars is
    # broken; before the fix, E_COVERAGE showed a clean table because the missing
    # row simply was not emitted.
    labels_miss = np.array(["L_BEAR" if l == "H_BEAR" else l for l in labels_p],
                           dtype=object)
    assert "H_BEAR" not in set(labels_miss.tolist())
    sc_m, summ_m = score_regimes(labels_miss, close_p, confidence=conf_p,
                                 trend_eff=te_p, horizons=(3, 9, 15), bars_per_day=3)
    _covm = sc_m[sc_m.family == "E_COVERAGE"]
    print("\n--- E_COVERAGE with H_BEAR entirely absent ---")
    print(_covm[["label", "n", "pct", "never_fires", "grade"]].to_string(index=False))
    _hb = _covm[_covm.label == "H_BEAR"]
    assert len(_hb) == 1, "A-5: the absent label must still get exactly one row"
    assert int(_hb["n"].iloc[0]) == 0, "A-5: absent label row must report n=0"
    assert _hb["grade"].iloc[0] == "FAIL", "A-5: an absent regime must grade FAIL"
    assert bool(_hb["never_fires"].iloc[0]), "A-5: absent label must be flagged"
    assert isinstance(_hb["grade_note"].iloc[0], str), "A-5: FAIL must carry a reason"
    # Same in family B.
    _perm = sc_m[sc_m.family == "B_PERSISTENCE"]
    _hbp = _perm[_perm.label == "H_BEAR"]
    assert len(_hbp) == 1 and int(_hbp["n_runs"].iloc[0]) == 0, (
        "A-5: B_PERSISTENCE must also emit a row for a label that never fires")
    assert _hbp["dwell_gap_grade"].iloc[0] == "NA", (
        "A-5/thin: a label with 0 runs must be NA, not graded")
    assert summ_m["B_persistence"]["labels_never_observed"] == ["H_BEAR"]
    # And the control: with all five present, no spurious never_fires row.
    _cov_ok = sc_p[sc_p.family == "E_COVERAGE"]
    assert len(_cov_ok) == 5 and not _cov_ok["never_fires"].any(), (
        "A-5: with every label present there must be exactly 5 rows and no "
        "never_fires flag")
    print("\n  control: with all 5 labels present, 5 rows and 0 never_fires flags.")
    print("[TEST 5] defect A-5 assertions PASSED.")

    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 6 (defect A-3): the two D_INTENSITY legs use SEPARATE units")
    print("=" * 80)
    # A construction where H's |return| gap is POSITIVE but negligible relative
    # to the series' own scale. The old single 0.05pp threshold, being an
    # absolute number of percentage points, graded such a gap on a coin flip
    # decided by the instrument's volatility rather than by the labels.
    _d = sc_p[sc_p.family == "D_INTENSITY"]
    print(_d[["direction", "metric", "median_H", "median_L", "gap_H_minus_L",
              "scale_median_abs_fwd_all_bars", "gap_rel_to_scale",
              "gap_threshold_used", "gap_units", "grade",
              "grade_legacy_conflated_units"]].to_string(index=False))
    _te_rows = _d[_d.metric == "trend_efficiency"]
    _rt_rows = _d[_d.metric.str.startswith("abs_fwd_ret")]
    assert (_te_rows["gap_units"] == "dimensionless (0-1 efficiency scale)").all()
    assert _rt_rows["gap_units"].str.startswith("percentage points").all()
    # The return leg's threshold must now VARY with horizon, because the scale
    # does. A single constant could not.
    _thr = _rt_rows["gap_threshold_used"].astype(float).dropna().unique()
    print(f"\n  return-leg PASS thresholds actually used: "
          f"{sorted(round(float(t), 4) for t in _thr)}")
    assert len(_thr) > 1, (
        "A-3: the return-leg threshold must scale with the data; a single value "
        "across horizons means the split did not take effect")
    assert (_rt_rows["gap_threshold_used"].astype(float) > 0.05).any(), (
        "A-3: on a series with typical |moves| well above 0.05pp, the new "
        "data-scaled bar must be HIGHER than the old flat 0.05pp somewhere")
    _flip = _rt_rows[_rt_rows["grade"] != _rt_rows["grade_legacy_conflated_units"]]
    print(f"  return-leg rows whose grade CHANGED under the split: "
          f"{len(_flip)}/{len(_rt_rows)}")
    if len(_flip):
        print(_flip[["direction", "metric", "gap_H_minus_L", "gap_rel_to_scale",
                     "grade_legacy_conflated_units", "grade"]].to_string(index=False))
    print("  (reported either way -- the thresholds were fixed before this was run)")
    print("[TEST 6] defect A-3 assertions PASSED.")

    print("\n" + "=" * 80)
    print("SELF-TEST COMPLETE: scorecard detects planted signal, correctly fails")
    print("random labels, correctly FAILS exactly-backwards labels and")
    print("anti-calibrated confidence instead of scoring them on abs(), scores a")
    print("perfect Markov chain at dwell ratio 1.0 (A-4), flags a regime that")
    print("never fires (A-5), and grades the two intensity legs in their own")
    print("units (A-3).")
    print("=" * 80)
