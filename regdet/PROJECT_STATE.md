# RegDet V1.1 — project state (durable; survives context compaction)

Last updated: after the cross-market/cross-timeframe generalization test (`generalization.ipynb`).

## What this is
2-hour Nifty regime detector, 5 labels (H_BULL/L_BULL/SIDEWAYS/L_BEAR/H_BEAR),
delivered as ONE master notebook the user runs on Kaggle with real yfinance data.
Generator `build_master_notebook_v2.py` is the source of truth; the .ipynb is always
regenerated, never hand-edited.

## IMPORTANT: file persistence risk
All work lives in this session's scratchpad:
`/tmp/claude-0/-home-user-SHERM-QUANTY/f1a44349-dfe0-5e0f-9b1f-b0b39e930e1b/scratchpad/`
This is tied to the current container and is NOT guaranteed to survive into a new
session. `/home/user/SHERM-QUANTY` (the git repo) is READ-ONLY and nothing has been
committed there — that is deliberate, per standing constraints, not an oversight.
The user has local copies of every notebook actually delivered via SendUserFile
(see "Deliverables sent to the user" below) — those are the durable copies if this
container is lost.

## Standing constraints (do not violate)
- Nothing graduates to the git repo without explicit user approval. `/home/user/SHERM-QUANTY` is READ-ONLY.
- No look-ahead anywhere: a signal at bar t uses only bars <= t. Truncation probes required.
- yfinance is FIREWALLED in the sandbox (403 via proxy). stooq, FRED, histdata.com, Google Drive,
  GitHub's code-search UI are ALSO blocked. `raw.githubusercontent.com` and `api.github.com`
  (except its /search endpoint) ARE reachable — this is the only outbound data route from here.
  Synthetic/sandbox results are ILLUSTRATIVE ONLY and must be labelled as such.
- No figure or section may be deleted when optimising. Speed changes must not alter results.
- Every new switch must have an OFF value asserted bit-for-bit against the frozen `label_bars_legacy`.
- State pass/fail criteria BEFORE the numbers (frozen protocol files, written before any run),
  and print a HEADROOM CHECK before any verdict. If a measured quantity cannot move far enough
  to clear the bar, print UNMEASURABLE — not a fail, not a manufactured pass.
- User has repeatedly asked for LEAN deliverables: include only what determines a parameter
  or a verdict. No prose sections, no diagnostics nothing reads. Split new work into its own
  notebook rather than growing the master by default.
- Nothing gets built at once without the user's go-ahead on scope — ask before big new builds.

## Shipped configuration (current — v6 base + 6 audited bugs fixed + 12-day context window)
```
BAR_DIR_WEIGHT=0.5          # v6 value. Direction = 50% HMM + 50% per-bar momentum.
ENSEMBLE_K=4                 # seeds [42,43,44,45]
INTENSITY_MODE='frozen_z'    # v6 value (NOT 'vol_norm' — that was the v7 change, reverted)
ESCALATION_DURING_HOLD='allow'  # v6 value (NOT 'block')
DIRECTION_MODE='rank'        # confirmed BIT-FOR-BIT NO-OP vs the shipped scheme (0/2744 bars differ)
CONF_L=0.50                  # low-confidence -> SIDEWAYS override
CONFIRM_BARS=2
Z_HI=0.5  EFF_HI=0.35  Z_HI_EXIT=0.35  EFF_HI_EXIT=0.25  EFF_WIN=9
Momentum ladder: 1/3/5 days (UNCHANGED, kept fast deliberately)
Context window (drawdown, dist_ma / SWING_WIN, VOL_SLOW): 12 days = 36 bars
  (WIDENED from 6.7 days = 20 bars — this is the fix for the "bull bars during a
  sustained decline" complaint. See "The decline-lag fix" below.)
ADOPTED_CONFIG_NAME='A: lean-cov'   # N=5, diag, 9 features, 114 params. PINNED on
  identifiability/capacity grounds, NOT on a Sharpe contest — that contest is proven
  undecidable at this sample size (see "Config ranking" below).
N_FOLDS=4  TRAIN_FRACTION=0.70
```
Why v6 and not v7: the user identified v6 as the best version by memory. v7 (which
shipped BAR_DIR_WEIGHT=0.0, vol_norm, block, K=6) tripled SIDEWAYS occupancy vs v6
(10.6% -> 36.3%) on real data. Root-caused (see "SIDEWAYS investigation" below) and
the shipped config reverted to v6's mechanics, layered with genuine bug fixes only.

## Architecture facts (verified, not assumed)
- The H-vs-L INTENSITY axis contains ZERO HMM. It is `intensity_state(trend_z,
  trend_efficiency)` — two rolling price statistics. Verified via the H-vs-L decision
  plane figure (trend_z x trend_efficiency scatter, only top-left/top-right escalate to H).
- The DIRECTION axis blends `bull_mass = (1-w)*hmm_mass + w*bar_momentum_mass`.
  A controlled 5-arm sweep (ONE shared fit, only w varied) on real 2h data found w barely
  moves anything between 0.0 and 0.75 (occupancy within ~2pp everywhere). A frozen,
  pre-declared protocol (W_SWEEP_PROTOCOL.md) then showed the difference in PREDICTIVE
  skill across w is SMALLER than the noise from merely reseeding the same w
  (between-w spread 0.297 vs seed spread 0.615 -> UNMEASURABLE, no winner named).
  Predictive skill vs "looks right" (descriptive fidelity) are IN TENSION and were shown
  NOT combinable into one score — presented as a trade-off curve, no winner declared.
  CONCLUSION: BAR_DIR_WEIGHT is a settled, low-stakes knob. Stop tuning it.
- EM is bistable: refitting with different seed sets produces two distinct "basins"
  (~98% label agreement within a basin, ~63% across). Nested, highly-correlated momentum
  features (mom_1d/3d/5d overlap heavily) are the suspected cause. NOT fully resolved.
- Config ranking (V1.0 prod / A: lean-cov / B: lean-feat) is PROVABLY UNDECIDABLE at this
  sample size: with N_FOLDS=4, even the best selector statistic is close to a coin flip at
  distinguishing genuinely-different configs (simulated: ~50-55% accuracy, ranking reorders
  on ~60% of reruns). Measured live: gap to runner-up = +0.032 Sharpe vs noise scale 1.360
  (>40x smaller). The notebook now prints NOT DECIDABLE instead of a false winner. This is
  the mechanical explanation for the earlier "config lottery" (3 runs, 3 different winners).

## The decline-lag fix (12-day context window)
User complaint: chart showed bull-coloured bars partway down a sustained multi-week decline.
Diagnosis: the ORIGINAL longest feature window was 20 bars = 6.7 trading days — inside any
6.7-day slice of a 2-month decline there genuinely ARE bullish-looking bounces, so the
detector was answering a narrower question than the chart was being judged on. NOT a coding
bug — a timescale mismatch.
Measured on real daily data, context-window reach only:
  ~7 days (old)  -> bear 90.7% / bull 7.4% / side 1.9%   (the problem)
  12 days (fix)  -> bear 94.4% / bull 0.0% / side 5.6%
  20 days        -> bear 100.0% / bull 0.0% / side 0.0%  (saturation point)
Fix: SWING_WIN and VOL_SLOW widened 20 -> 36 bars (6.7d -> 12d). Momentum ladder (1/3/5 days)
deliberately LEFT FAST — uniform scaling would trade responsiveness for context; splitting
them keeps the detector reactive while giving it a wider frame to react WITHIN.
VERIFIED on the user's real Kaggle run (final_regdet_v9.ipynb): May-2025 V-bottom and the
2026 decline both show mostly correct bear/grey shading now, versus solid green before.
One residual: a thin green patch sometimes appears right at the very bottom of a decline,
just before the sharpest recovery leg — not fully eliminated, flagged not re-investigated.

## SIDEWAYS investigation (why v7 had 36% SIDEWAYS)
Two hypotheses tested and REFUTED before the real cause was found:
  - CONF_L (0.50 override): REFUTED. Fires on only 0.07% of real daily bars (median
    winning mass 0.99 at the shipped config w=0.5 baseline).
  - DIRECTION_MODE='rank' vs the shipped scheme: REFUTED. Confirmed BIT-FOR-BIT identical
    (0/2744 bars differ) — v6 already used rank buckets.
ACTUAL CAUSE (found via v6-anchored ablation, A0..A6, each isolating ONE v6->v7 change):
  BAR_DIR_WEIGHT 0.5->0.75 alone: SIDEWAYS +16.4pp (real 2h data)
  ENSEMBLE_K 4->6 alone: SIDEWAYS +4.0pp
  All other v7 changes ([5] vol_norm, [6] rank, [7] block): 0.00pp each
Mechanism: blending toward per-bar momentum does not make SIDE *win* the argmax — it makes
EVERY winner's mass smaller (median winning mass 0.693 -> 0.564 at w=0.75). That then trips
the CONF_L=0.50 floor far more often (3.8% of bars -> 37.7%). CONF_L and BAR_DIR_WEIGHT
interact; neither alone explains it.
A candidate fix (SIDE_TARGET_RATE, a fit-window-calibrated margin, same device as
H_TARGET_RATE) was built and measured in `sideways_fix.ipynb`: SIDEWAYS 29.5% -> 20.6% on
real daily data, S (stability) unchanged, guards pass, verified NOT just relabelling noise
(AUC flat, ordering test passes, beats a random-promotion control). **NOT folded into the
shipped master** — it changes emitted labels and needs a deliberate user decision. This is
now moot at the shipped w=0.5 (CONF_L only fires 0.07% of the time there), but would matter
again if BAR_DIR_WEIGHT is ever raised.

## The six audited bugs (all fixed; verified against `PROJECT_STATE.md`'s own v6 anchor)
Full part-by-part audit of the generator, labels re-verified bit-identical to v6 after each fix.
1. `direction_buckets()` gave every leftover state to BULL via floor division (N=6 -> 3 bull /
   2 bear structural bias). FIXED: bear/bull counts equal by construction. N=5 shipped
   config bit-identical before/after.
2. No guard against inverted hysteresis bands (exit stricter than entry -> silent no-op).
   FIXED: assert exit <= entry.
3. VIX alignment used `.ffill().bfill()` — bfill pulled LEADING bars (before VIX existed)
   from the FIRST FUTURE VIX value. A genuine, if narrow, look-ahead. FIXED: ffill only;
   leading bars fall out via the existing feature warmup (correct, since those bars
   genuinely carry no VIX information).
4. Config ranking (see above) printed a confident "winner" with no decidability check.
   FIXED: gap-vs-noise-scale check added, prints NOT DECIDABLE when warranted.
5. `regime_blocks()` (chart shading) ended each coloured band at ITS OWN last bar while the
   next began at the FOLLOWING bar — the gap between them (up to ~64h across a weekend) was
   left unpainted, scattering white stripes through every multi-year chart. FIXED: each
   block now runs to the start of the next; invariant assert added. Plotting-only, no
   statistic depends on these edges.
6. (Checked, not a bug) BAR_DIR_WEIGHT equivalence-matrix coverage confirmed still pinned
   as literals {0.0, 0.25, 0.75, 1.0}, all bit-identical — a prior fix holding.

## Rejected / investigated and NOT adopted
- Vortex Indicator (intrabar High/Low momentum) — considered, NOT implemented. User chose
  the 12-day context window instead. Needs High/Low data the pipeline doesn't currently fetch.
- `DIRECTION_MODE='soft'` — fixes the underlying mechanism 3.2x but regresses emitted labels
  via a 5-point sd standardisation bug. NOT adopted.
- Trailing-quantile thresholds — worse than status quo.
- Multi-restart EM — collapsed LL spread but not segmentation (ARI 0.673) — doesn't fix bistability.
- N_STATES=6 (won an earlier stability study) — that study ran WITH the bug-1 bullish bias
  active (3 bull/2 bear at N=6). Needs re-measuring post-fix before trusting that result.

## Cross-market / cross-timeframe generalization test (LATEST WORK — `generalization.ipynb`)
Protocol: `GENERALIZATION_PROTOCOL.md` (frozen before any numbers existed). Tests whether the
ARCHITECTURE (not re-tuned) generalizes beyond Nifty 2h. Every constant held FIXED across all
runs (N=5, CONF_L, CONFIRM_BARS, the 12-day window, etc.) — only the StandardScaler and
fit-window baselines re-derive per market, by design.
- Grid A (yfinance: S&P500/NASDAQ/EURUSD/GBPUSD/JPYUSD x 1h/2h/4h, 15 runs): SYNTHETIC ONLY
  in this sandbox (yfinance blocked). Machinery-proof, not decision-grade. Real Kaggle run
  still needed for this grid.
- Grid B (GitHub-sourced REAL Fed H.10 daily FX — JPY/GBP/EUR vs USD, 1971/1971/1999 -> 2026,
  ~55 years, + weekly resample): REAL DATA, decision-grade for FX.
KEY FINDING (real data): **4 of 6 Grid-B runs show the forward-return signal INVERTING
sign between in-sample and out-of-sample** (e.g. JPY/USD daily: IS HAC t=+2.06, OOS t=-0.38).
This is the classic overfitting signature and is REPORTED AS SUCH — the pre-registered
prediction "P3: OOS degrades but does not invert" is REFUTED on real data. Also: 4 of 6
Grid-B runs fail the G3 whipsaw guard (switch rate 27-30/100 bars, roughly every 3.5 bars,
over 55 years of daily FX).
IMPLICATION: the architecture's forward-return edge (what little of it survives HAC/Cohen's-d
scrutiny on Nifty) does not obviously transfer to other markets/timescales as-is. This is a
genuine, not-yet-resolved finding — no fix attempted yet, pending user direction.
STATUS: notebook built, verified (0 cell failures, 4 figures, 336 HMM fits, determinism
checked via double-run diff), but NOT YET SENT to the user or independently re-verified by
the supervisor session — the user said "stop" mid-handoff and this was paused. NEXT STEP:
verify + send `generalization.ipynb`, then decide whether/how to respond to the P3 finding.

## Intraday FX test — BUILT AND VERIFIED (`intraday_fx.ipynb`)   ← LATEST WORK
Real long-history intraday FX, reachable from the sandbox (this is the first intraday grid
in the project that is NOT synthetic).
SOURCE: https://github.com/ejtraderLabs/historical-data (community repo, NOT an official feed)
  https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/{SYM}/{SYM}{tf}.csv
  SYM in {EURUSD,GBPUSD,USDJPY}; tf in {m15,m30,h1,h4,d1}; cols Date,open,high,low,close,tick_volume
  PRICES ARE INTEGER-SCALED, SCALE DIFFERS BY PAIR: EURUSD/GBPUSD /1e5, USDJPY /1e3.
  57,600 bars/pair, 2012-11-16 -> 2022-03-04. DATA ENDS 2022-03 — says nothing about 2022-2026.
  Verified 14/14: monotone, 0 dups, positive, OHLC-consistent; hourly vol 0.100/0.114/0.109%;
  BREXIT GBPUSD 2016-06-23/24 1.5006->1.3379 = -10.84%; COVID EURUSD Mar-2020 7.7% swing.
GRID: 3 pairs x {1h direct, 2h resampled DOWN from h1} = 6 runs. All constants frozen at the
shipped values; only scaler/fit-window baselines re-derive. BARS_PER_DAY 24 (1h) / 12 (2h).
STATUS: VERIFIED. Full cell-by-cell run, all 6 runs, 0 cell failures, 3 figures, ~430s.
Runtime probe projected 409s and the grid took 399s — nothing capped, R=4 intact, no pair
or timeframe dropped. Data re-verified live: 14/14 structural + event checks pass, all the
numbers above reproduced exactly. BARS_PER_DAY realised 24.00 / 12.00 vs assumed 24 / 12
(ratio 1.00). Determinism: two full runs, OMP_NUM_THREADS=1, 564 comparable lines IDENTICAL.
Truncation probe: 0 labels differ at t<=T on one run per timeframe. G4 degenerate stub and
G3 whipsaw stub both BITE. ZZ_CALLS==0 through the whole label phase.

RESULTS (all 6 runs, nothing re-tuned, nothing dropped):
  P1 CONFIRMED  guards G1-G4 pass on 6 of 6 runs (note: the DAILY sibling failed G3 on 4 of
                6; intraday W is 10-15 switches/100 bars, comfortably under the 25 limit)
  P2 CONFIRMED  SIDEWAYS in [20,45]% on 5 of 6 (EUR/USD@1h is the outlier at 10.2%)
  P3 REFUTED    3 of 6 runs INVERT sign IS->OOS (GBP/USD@1h, USD/JPY@1h, USD/JPY@2h)
  P4 CONFIRMED  same-pair mean distance 3.030 < same-timeframe 3.535

THE HEADLINE, stated more sharply than the inversion count alone implies:
  ALL SIX runs have a NEGATIVE out-of-sample (BULL - BEAR) contrast. Not one run shows
  BULL-labelled bars outperforming BEAR-labelled bars out of sample. The three runs that
  "kept their sign" kept a NEGATIVE one in BOTH spans — they pass the non-inversion test by
  being consistently ANTI-PREDICTIVE, which is not evidence the detector works. Full 5-label
  ordering HOLDS on 0 of 6 IS spans and 0 of 6 OOS spans. OOS HAC t range -0.18 to -1.62.
INTRADAY vs DAILY: they AGREE, on the failing side. P3 was REFUTED on both grids. The raw
  inversion counts differ by ONE run (4/6 daily vs 3/6 intraday), which at n=6 per grid is
  not separable from sampling noise — so this is NOT read as the daily inversion "failing to
  generalise" down the timeframe axis. ~3.1x better sample per run did not overturn the
  daily result; it reproduced the same refutation. CAVEAT: the grids do not cover the same
  span (daily 1971/1999->2026 vs intraday 2012-11->2022-03), so cadence and sample period
  are confounded and this design cannot separate them.
NOTE: an earlier draft of the notebook's own verdict cell read this comparison as
  "DISAGREES ... the daily inversion therefore does not generalise down the timeframe axis",
  resting entirely on the 4/6-vs-3/6 majority flip. That was corrected — it promoted a
  one-run gap into a qualitative claim, in the flattering direction. No GRADE was changed.

## Generator portability + a pre-existing breakage found while verifying
Every `build_*.py` had `HERE` pinned to a dead session scratchpad, so the declared source of
truth could not rebuild anything from a fresh clone. All 8 now resolve paths relative to the
generator file; `REGDET_OUT_DIR` overrides the output dir. Same fix in `harnesses/_fx_cellrun.py`
(`NB_DIR` override). `harnesses/_fx_diff.py` was reporting FALSE differences — it stripped only
`12.3s`-style timings and missed integer seconds and the bare `secs` column; now normalises all.
PRE-EXISTING, NOT introduced by this work, verified against the committed baseline: 5 of 8
generators cannot rebuild their notebooks against the current master —
`build_stability_lag`, `build_hmm_share_charts`, `build_daily_fit`, `build_v6_ablation`,
`build_sideways_fix` all assert the master's constants cell contains `BAR_DIR_WEIGHT   = 0.0`,
but the shipped master is 0.5. Their .ipynb files therefore cannot currently be regenerated
from their generators. NOT fixed here — fixing it risks changing shipped notebooks and needs
a user decision. `build_master_notebook_v2`, `build_generalization`, `build_intraday_fx` build.

## Deliverables sent to the user (via SendUserFile — these are the durable copies)
- `regdet_v11_master.ipynb` — the master detector (v6 config + 6 bugs + 12-day window).
  Confirmed by the user as matching their `final_regdet_v9.ipynb` Kaggle run — "v9 is final".
- `hmm_share_charts.ipynb` — 5-arm HMM-share comparison charts (100/75/50/25/0% HMM).
- `stability_lag.ipynb` — fit-stability (S) / transition-lag (L) / whipsaw (W) study.
- `daily_fit.ipynb` — daily-fit-then-project-to-2h architecture test.
- `v6_ablation.ipynb` — isolates the exact v6->v7 SIDEWAYS regression, arm by arm.
- `sideways_fix.ipynb` — the SIDE_TARGET_RATE candidate fix (not shipped, see above).
- `regdet_notes.pdf` — 14-page study notes: architecture, HMM/EM math, all 15 master-notebook
  charts explained, the 6 bugs, shipped config, open items. Built for someone with zero
  context on this conversation to read standalone.
- `generalization.ipynb` — BUILT, VERIFIED, NOT YET SENT (see above).

## Verification discipline that has caught real bugs (keep using this)
- Run notebooks cell-by-cell the way Jupyter does — compile each cell SEPARATELY (shared
  namespace). Concatenating cells into one file breaks `from __future__ import annotations`.
  Cell 0 is always `%pip install` — a Jupyter magic, a SyntaxError in plain Python, NOT a failure.
- Pin `OMP_NUM_THREADS=1` for any equivalence check comparing two full runs — multi-threaded
  BLAS makes two runs differ by reduction-order noise even at identical config.
- A TRUNCATION PROBE (cut the series at T, assert no label at t<=T changes) is the real
  causality test — stronger than checking only the final bar.
- LOOK AT RENDERED PNGs before trusting a notebook's own text output — real rendering defects
  (weekend-gap shading, semantically-inverted heatmap colours, overlapping legends) have been
  caught this way at least 6 times across sibling notebooks, never by reading code alone.
- Every "obvious" bug hypothesis in this project has been WRONG at least once when actually
  measured (CONF_L, DIRECTION_MODE='rank', N=6 capacity->stability). Measure, do not assume.
- Pre-register predictions/criteria in a frozen .md file BEFORE running anything. Grade
  against them afterward in plain words (CONFIRMED/REFUTED); do not reinterpret after seeing
  the number. This has directly produced the project's most trustworthy findings (the w-sweep
  UNMEASURABLE verdict, the P3 overfitting refutation).
