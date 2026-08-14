# Daily-fit architecture — pre-declared protocol

Written BEFORE any numbers exist. "Learn slow, apply fast."

## The diagnosis this is built on
1. Effective sample for regime inference is ~6-10 macro regimes in 2.9 years,
   against 114 free parameters. Everything unstable follows from this.
2. Causal lag is near its information floor already (7.1 bars for the reference
   swing; measured 7.00-9.75). Lag is NOT the remaining problem.
3. The fit is BISTABLE: two basins, ~98% agreement within, ~63% across.
4. Binding constraint: yfinance serves only ~730d of intraday data.

## The change
Fit the DIRECTION model on DAILY bars with the longest available history
(^NSEI daily reaches ~2007, so ~18 years), then APPLY it to 2h bars.
Keep the INTENSITY (H vs L) axis on 2h data, unchanged — it is volatility-driven,
it already works, and it needs the fast resolution.

Rationale: direction is slow, weak and hard to learn -> needs many regimes to
estimate. Intensity is fast, strong and easy -> needs fine resolution. Fit each
where its information actually lives.

## STRICT CAUSALITY — the one thing that can invalidate everything
At 2h bar t on calendar day d, the direction state MUST come from the last
FULLY CLOSED daily bar, i.e. day d-1 or earlier. Using day d's daily bar would
leak the rest of day d into the morning of day d.
- Implement as: daily state -> shift by one full day -> forward-fill onto 2h bars.
- ASSERT: for every 2h bar, the source daily bar's timestamp < that bar's session date.
- Truncation probe REQUIRED: cutting the series at T must not change any label at t <= T.
- The daily HMM fit itself must be anchored (fit on a leading window, applied forward).

## PRE-REGISTERED PREDICTIONS (falsifiable; record before running)
If the small-sample diagnosis is right, then versus the 2h-fit baseline:
  P1. S (seed-set label agreement) RISES materially. Baseline 88.29%, seed spread
      14.94pp. Prediction: S > 95% AND the pairwise matrix stops being bimodal.
  P2. The two-basin structure disappears or weakens: pairwise agreement within a
      config becomes unimodal (no ~60% off-diagonal cluster).
  P3. L does NOT improve much and may worsen slightly. Daily direction is coarser.
      This is EXPECTED and is not a failure — lag is already at its floor.
  P4. Occupancy becomes more stable across seed sets.
IF P1 AND P2 FAIL, the small-sample diagnosis is WRONG. Say so plainly in the
output. Do not reinterpret the prediction after seeing the numbers.

## ONE KNOB AT A TIME
Arm 1: BASELINE            — current 2h fit, 2y, existing features.
Arm 2: DAILY-FIT, SAME features (daily equivalents), long history.
Arm 3: DAILY-FIT + decollinearized features.
Arm 2 isolates the DATA change. Arm 3 adds the FEATURE change. Do not conflate.

Decollinearized set (arm 3): ONE momentum, ONE volatility, ONE dispersion/context.
Drop the nested trio mom_1d/mom_3d/mom_5d down to a single horizon. State the
correlation matrix of both feature sets so the collinearity claim is checked,
not assumed.

## METRICS — identical to STABILITY_LAG_PROTOCOL.md so results are comparable
  S = mean pairwise label agreement over R=4 disjoint seed sets.
  L = ZigZag transition lag (bars). ZigZag is retrospective grading ONLY;
      assert it never reaches a label.
  W = switches per 100 bars.
  Plus: rows-per-parameter, and N_REGIMES_OBSERVED (count of distinct macro runs
  in the FIT window) — this is the number the whole diagnosis turns on, so
  print it explicitly for every arm.
Guards G1-G4 unchanged (occupancy 3-50%, no collapse, W <= 25, SIDEWAYS <= 50%).

## REPORTING
- Baseline is row 1. Report S, L, W together. No composite score.
- Print the pairwise S matrix per arm (bimodality is only visible there).
- Verdict on the PREDICTIONS first, then on dominance.
- Reference-case figure: May-2025 V-bottom, baseline vs best daily-fit arm,
  regime-background style, tight equal y-limits.
- HEADROOM: report seed spread before any verdict.

## SCOPE LIMITS
- Do NOT re-tune BAR_DIR_WEIGHT (settled, 0.0) or the intensity axis.
- Do NOT compute forward returns, Sharpe or any economic metric here.
- Real verdict needs the Kaggle run; synthetic is machinery-proof only.
- If daily history is shorter than expected, report the ACTUAL span and bar
  count — the whole argument rests on getting materially more regimes.
