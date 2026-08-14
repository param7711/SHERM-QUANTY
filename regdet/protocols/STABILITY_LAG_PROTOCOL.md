# Fit stability + transition lag — pre-declared protocol

Written BEFORE any numbers exist. Criteria frozen at commit time.

## The two defects, as measured on real data (2894 bars, 2023-08 -> 2026-07)

**D1. Fit instability.** Refitting the same arm with a disjoint seed set flips
~6% of all bars (label agreement 93.81% at w=0.00). This EXCEEDS the entire
difference between the 25% and 100% HMM arms, so it contaminates every
comparison in the project — including the config lottery.

**D2. Transition lag.** Labels describe the regime that just ended. Reference
case: the May-2025 V-bottom, where a +2.7% advance (24,050 -> 24,700) is
labelled L_BEAR for its entire duration, turning BULL only near the top. Present
in ALL FIVE HMM-share arms, including w=0.75 which was added to fix exactly this.

## THE CENTRAL TENSION — state it up front, do not hide it
Lower lag = react sooner = react to smaller moves = MORE switches and MORE
sensitivity to the underlying fit. These objectives oppose each other. The
honest deliverable may be a FRONTIER, not a fix. If no configuration improves
one without degrading the other, say so in those words and present the frontier.

## Metrics (all frozen here)

### S — fit stability
    S = mean pairwise label agreement (% of bars) across R INDEPENDENT seed sets.
    R = 4 disjoint seed sets. Report the full pairwise matrix, not just the mean.
    Higher is better. Current baseline to beat: 93.81%.

### L — transition lag
    Identify major price swings with a LABEL-BLIND, retrospective ZigZag
    (threshold = 2.0% reversal). For each swing, L = number of bars between the
    swing's start and the first bar whose emitted direction matches the swing
    direction. Unmatched swings (never labelled correctly) count as the full
    swing length — they are the worst case, not a skipped row.
    Report median and 75th percentile. Lower is better.

    NOTE ON LOOK-AHEAD: the ZigZag uses future prices. That is legitimate here
    because it is a RETROSPECTIVE EVALUATION metric, exactly like forward
    returns — it grades a past decision. It must NEVER feed a label. Assert that
    no ZigZag output reaches `label_bars`.

### W — whipsaw (the guard that stops L being gamed)
    W = switches per 100 bars. Current baseline 18.93.
    A lag fix that reduces L while inflating W has bought nothing.

## Interventions to test (one at a time, then the best combination)

### For S (stability)
    S1. ENSEMBLE_K: 6 -> 12 -> 24. More seeds, more averaging. Cost is runtime.
    S2. Drop bad local optima: discard ensemble members whose converged
        log-likelihood is materially below the ensemble best (threshold declared
        as a fraction of the LL spread, NOT tuned to a result).
        LEGITIMATE: LL is the fit's own objective; this uses no forward returns.
    S3. Majority-vote the per-seed EMITTED direction instead of averaging
        direction MASS. More robust to a single bad member.
    S4. Deterministic init (fixed k-means++ / fixed quantile partition) to cut
        across-seed variance by construction.

### For L (lag)
    L1. CONFIRM_BARS 2 -> 1.
    L2. Adaptive confirmation: require 2 bars normally, 1 bar when the direction
        signal is strong (|direction mass| above a declared threshold).
        STRICTLY CAUSAL — uses only bar t.
    L3. Asymmetric confirmation: faster to EXIT an extreme regime than to enter.
    L4. ESCALATION_DURING_HOLD 'block' -> 'allow' (already exists as a switch).

## Guards — a candidate is VOID if any fails
    G1. Occupancy: every label in [3%, 50%].
    G2. No label may collapse (a label falling below 3% voids the arm).
    G3. W must not exceed 25 switches/100 bars (baseline 18.93 + ~30%).
    G4. Degeneracy: a config that pushes SIDEWAYS above 50% is void.
        WHY: stability is TRIVIALLY maximised by labelling everything SIDEWAYS.
        This guard exists specifically to stop S being gamed that way.

## Reporting rules
- Report S, L and W TOGETHER for every candidate. Never one alone.
- Baseline (current shipped config) is row 1 of every table.
- Plot the (L, S) frontier and the (L, W) frontier. Mark the baseline.
- NO composite score combining S, L and W. Same rule as the six families.
- Declare a winner ONLY if a candidate strictly dominates the baseline:
  S higher AND L lower AND W not worse, all guards passing.
  If nothing dominates, print "NO DOMINATING CONFIGURATION" and show the frontier.
- HEADROOM CHECK before any verdict: report the seed-to-seed spread of S and L
  themselves. If a candidate's improvement is inside that spread, it is noise.

## Scope limits
- Real verdict requires the Kaggle run. Synthetic is machinery-proof only.
- Do NOT re-tune BAR_DIR_WEIGHT. It is settled: 25%-100% HMM are equivalent
  within noise, and it is fixed at 0.0 for this work.
- Do NOT touch the intensity (H vs L) axis in this round.
