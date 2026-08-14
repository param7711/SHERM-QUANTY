# BAR_DIR_WEIGHT sweep — pre-declared protocol

Written BEFORE any sweep numbers exist. Criterion is frozen at commit time.
`w = BAR_DIR_WEIGHT`. Direction mass = `(1-w)*HMM + w*per-bar-momentum`.
**w = 0.0 -> 100% HMM. w = 1.0 -> 0% HMM (the HMM-free control).**

## Grid
    w in {0.00, 0.10, 0.25, 0.50, 0.75, 1.00}
Six arms. w=0.0 is the incumbent; w=1.0 is the control that must be run and
reported even if it wins.

## Selection must be out-of-sample
One fit, one label pass per w, then:

- **SELECT** on the first 70% of bars (the existing production-style train span).
- **REPORT** on the final 30%, untouched during selection.
- The reported number for the chosen w is the ONLY number quoted as the result.
- All six arms' holdout numbers are printed anyway, so the selection-vs-holdout
  gap is visible. If the chosen w is not also the holdout best, SAY SO.

No re-selection after seeing holdout. If the holdout contradicts the choice,
that is the finding, not a reason to re-pick.

## Primary criterion (frozen, in order)
A single scalar, decided in advance, on the SELECTION span only:

    PRIMARY = HAC t-stat of (mean fwd return | BULL bars) - (mean fwd return | BEAR bars),
              pooled across horizons 3/9/15 via the mean of the three HAC t's.

Rationale: this is the one quantity the detector exists to produce — a
directional split that survives overlap correction. It is not a composite of
unrelated families, and it cannot be gamed by occupancy.

### Tie-break, only if PRIMARY differs by < 0.25 between two arms
1. Higher `__BULL_vs_BEAR_agg__` Cohen's d at fwd_9.
2. Fewer switches per 100 bars (cheaper to trade).
3. Lower w is NOT preferred a priori. No thumb on the scale for the HMM.

## Guards that can veto the winner
Applied AFTER the primary ranking, each printed pass/fail:
- **G1 occupancy**: every label in [3%, 50%]. An arm that collapses a label is void.
- **G2 direction ordering**: BULL > SIDEWAYS > BEAR mean fwd return must HOLD at
  >= 2 of 3 horizons on the selection span.
- **G3 no anti-signal**: no label may score FAIL(ANTI) on A_DISCRIMINATION at
  fwd_3 or fwd_9 (the new signed grading).
- **G4 stability**: refit with 3 different seeds; PRIMARY spread across seeds must
  be < 0.5, else the arm is flagged UNSTABLE and cannot be selected.

G4 exists because defect 4 showed rankings flip on refit. An arm that only wins
on one seed has not won.

## Mandatory honest outcomes
- If **w=1.0 wins**, the conclusion is "the HMM contributes nothing measurable
  to direction" and it must be reported in exactly those words. The HMM is not
  retained for having been the plan.
- If **the six arms are within noise of each other** (PRIMARY spread < the
  seed spread from G4), the conclusion is "w is not identifiable on this
  sample" and the incumbent w=0.0 is kept on parsimony grounds, labelled as an
  arbitrary tie-break rather than a result.
- Sweep runs on REAL data on Kaggle. Synthetic output is illustrative only.

## Headroom check, printed before the verdict
Print the seed-to-seed PRIMARY spread (G4) first. If the between-w spread does
not exceed it, print UNMEASURABLE and stop — do not name a winner.

## What this protocol does NOT do
- Does not tune Z_HI / EFF_HI / N / cov jointly with w. One knob at a time.
- Does not touch the intensity (H vs L) axis, which contains no HMM at all.
  This sweep can only ever settle the DIRECTION axis.
