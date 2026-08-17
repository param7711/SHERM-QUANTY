# RegDet — FINAL

The Sherm Quanty market regime detector. **5 labels**
(`H_BULL` / `L_BULL` / `SIDEWAYS` / `L_BEAR` / `H_BEAR`), built from two
independent axes:

- **Direction** — a Gaussian HMM over 9 causal features, blended 50/50 with a
  per-bar momentum score.
- **Intensity** (High vs Low conviction) — pure rolling price statistics
  (`trend_z` × `trend_efficiency`). **Contains no HMM at all.** This is
  load-bearing: it is why changing the HMM's weight barely moves occupancy.

**Status: FINAL. No further tuning.** All experimental and superseded versions
have been removed from the tree; git history retains them if ever needed.

## What is here

```
notebooks/regdet_v11_master.ipynb        the detector on NIFTY 2h        <- final
notebooks/fx_v9.ipynb                    the SAME detector on FX/gold    <- final
generators/build_master_notebook_v2.py   source of truth for the master
generators/build_fx_v9.py                source of truth for the FX build
generators/regime_scorecard.py           scorecard, inlined by the master
harnesses/_execute_nb.py                 runs a notebook, captures outputs + figures
protocols/W_SWEEP_PROTOCOL.md            implemented end-to-end in master section 7W
notes/regdet_notes.pdf                   14-page standalone explainer
notes/fx_v9_all7_runlog.txt              the final 7-instrument run log
FINAL.md                                 close-out: results, limits, rejected ideas
PROJECT_STATE.md                         full experiment-by-experiment history
```

## The generator -> notebook rule

`build_*.py` files are authoritative. The `.ipynb` files are build products.
**Never hand-edit a notebook** — change the generator and regenerate. Several
bugs in this project's history came from notebooks drifting from generators.

```bash
python3 regdet/generators/build_master_notebook_v2.py   # -> notebooks/regdet_v11_master.ipynb
python3 regdet/generators/build_fx_v9.py                # -> notebooks/fx_v9.ipynb
```

Both rebuild **byte-identical** to what is committed.

## Running the FX version

Open `notebooks/fx_v9.ipynb` and change **one line** at the top of the data cell:

```python
FX_INSTRUMENT = 'EURUSD'   # 'GBPUSD' 'USDJPY' 'XAUUSD' 'AUDUSD' 'USDCAD' 'USDCHF'
```

Re-run. Every chart, axis label, table and scorecard repoints automatically.
It differs from the Nifty master by **one functional cell** (the data loader)
plus five display strings — **not one constant is re-tuned**. FX at 8h is
3 bars/day, matching Nifty 2h's 3 bars/session, so `BARS_PER_DAY = 3` stays
literally correct and `assert MOM_3D_BARS == 9` passes untouched.

## Shipped configuration (identical across every market)

```
BAR_DIR_WEIGHT=0.5      ENSEMBLE_K=4        CONF_L=0.50       CONFIRM_BARS=2
INTENSITY_MODE='frozen_z'                   ESCALATION_DURING_HOLD='allow'
DIRECTION_MODE='rank'   (verified bit-for-bit no-op)
Z_HI=0.5  EFF_HI=0.35   Z_HI_EXIT=0.35      EFF_HI_EXIT=0.25   EFF_WIN=9
Momentum ladder: 1/3/5 days (kept fast)     Context window: 12 days
ADOPTED_CONFIG_NAME='A: lean-cov'  (N=5, diag, 9 features, 114 params)
```

## Read this before building on it

RegDet is a **descriptive regime segmenter**, not a forward-return predictor.
Measured, not assumed:

- forward-return ordering is **BROKEN at all 3 horizons on all 8 runs**, Nifty
  included — a property of the architecture, not an FX-specific problem;
- on FX, 7–13 scorecard rows per instrument are materially **backwards**
  (anti-signal) versus 2 on Nifty, on **7 of 7** instruments;
- intraday FX variance ratios are 0.80–1.06 with no series reaching `|z*| >= 2`
  — near-random-walk.

**Consume the labels as state/context, never as a directional signal.** Full
detail, including everything tried and rejected, is in `FINAL.md`.

## Hard-won facts worth not rediscovering

- **`BAR_DIR_WEIGHT` is settled.** Its effect between 0.0–0.75 is smaller than
  the noise from merely reseeding the same value — UNMEASURABLE under a frozen
  protocol.
- **Config ranking is provably undecidable** at 4 folds; the notebook prints
  `NOT DECIDABLE` rather than a false winner.
- **EM is bistable** — refits land in one of two basins (~98% agreement within,
  ~63% across). Suspected cause: collinear nested momentum features. Unresolved.
- **Match bar density before comparing regime charts across markets.** Drawing
  57,311 bars into a panel sized for 2,858 merges shading into stripes
  regardless of label quality — this artefact drove several wrong conclusions.
- **Pre-registration is necessary but not sufficient.** One test passed all six
  pre-registered predictions and was still a failure, because a prediction was
  framed against the wrong control. Pre-register against *what ships*.
- **yfinance is firewalled in the sandbox.** Only `raw.githubusercontent.com` is
  reachable. FX data was validated against Brexit (−10.84%) and COVID before use.

## Not wired into production

`regime_engine_tactical.py` at the repo root is the production tactical engine
and remains **Nifty-only**; nothing in `regdet/` is imported by it. Integration
is a separate, deliberate step that has not been taken.
