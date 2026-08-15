# Held-out replication protocol — frozen BEFORE the run

Grade in plain words afterward. Do not reinterpret after seeing the number.
Do not soften. **A null result here is INCONCLUSIVE, not "no effect"** — that is
pre-registered below, with the arithmetic that forces it.

## What is being replicated, and why it needs an out-of-sample instrument set

`fx_sensitivity.ipynb` found that shortening `CONTEXT_DAYS` 12 → 3 moved the
mean out-of-sample `BULL − BEAR` contrast from −0.88 to −0.40, a paired
improvement of **+0.484** across 4 instruments.

That number is **hypothesis-generating, not evidence**. The four instruments
(EUR/USD, GBP/USD, USD/JPY, XAU/USD) are where the effect was *discovered*.
Re-testing on them is circular. This notebook tests it on instruments held out
from that discovery.

## The instrument universe, measured not assumed

The source repository contains 12 instruments at `h1`. They are **not 12
independent observations**:

| group | instruments | status |
|---|---|---|
| discovery | EURUSD, GBPUSD, USDJPY, XAUUSD | hypothesis generated here — **not evidence** |
| **held out** | **AUDUSD, USDCAD, USDCHF** | **the actual test, n = 3** |
| derived crosses | EURGBP, EURJPY, GBPJPY, AUDJPY, EURCHF | **excluded**, see below |

**The crosses are arithmetic, not markets.** Measured correlation of each
cross's hourly log return against its triangular synthetic:

```
EURGBP vs +EURUSD -GBPUSD   corr = +0.9893
EURJPY vs +EURUSD +USDJPY   corr = +0.9985
GBPJPY vs +GBPUSD +USDJPY   corr = +0.9985
AUDJPY vs +AUDUSD +USDJPY   corr = +0.9983
EURCHF vs +EURUSD +USDCHF   corr = +0.9931
```

Including them would inflate *n* from 7 to 12 while adding essentially zero
information. They are excluded **for that measured reason**, recorded here so
the exclusion cannot later look like cherry-picking.

The 7 non-derived instruments are genuinely distinct — mean off-diagonal return
correlation **+0.326**, PC1 explaining only **43.2%**, participation ratio
**≈ 4.0 effective dimensions**. Not one USD trade in seven costumes.

## POWER — computed before the run, and it is the headline

Assuming the discovery estimate is exactly true (effect +0.484, sd 0.626):

```
n =  3   power = 13.0%   <-- the held-out set that exists
n =  7   power = 40.6%   <-- EVERY independent instrument in the repo
n = 16   power = 80%     <-- what the question actually requires
```

**This design cannot confirm the effect.** At n = 3, if the effect is real and
exactly as large as discovered, this test misses it ~87% of the time. Adding
every remaining instrument in the universe still leaves it under-powered.

That is stated first, before any number exists, so that:

* a **failure to reach significance is reported as INCONCLUSIVE**, never as
  evidence the effect is absent; and
* no one is tempted to keep adding instruments until something clears 0.05.
  There is no *n* available here that would make that legitimate.

## What this design CAN do

It cannot confirm. It **can** falsify, and it can estimate:

* If the held-out effect comes back **negative or near zero**, that is real
  evidence the discovery result was noise — a sign flip does not need power to
  be informative.
* The held-out **point estimate and its CI** are reportable regardless, and
  whether that CI contains the discovery estimate (+0.484) or zero is the
  substantive output.

## Fixed design

* **1h only.** The 2h axis is dropped: measured `corr(1h delta, 2h delta) =
  +0.986` across instruments — it is the same series sampled twice and it
  inflated the earlier paired *t* from 1.55 to 2.32 by pure pseudo-replication.
* **Two arms only**: `CONTEXT_DAYS` ∈ {12, 3}. `CONFIRM_BARS` stays 2. No other
  knob moves. Nothing is re-tuned per instrument.
* **NO TRAINING CAP.** `TRAIN_CAP_BARS = None`, full leading window. The capped
  sweep shifted individual runs by up to **0.60** in OOS *t* — the same order as
  the effect under study. Runtime is spent instead of data.
* The discovery instruments are **re-run here at the same uncapped settings**,
  because their +0.484 came from a capped run and is not otherwise comparable.
* R = 4 seed sets, K = 4. Nothing dropped.

## Pre-registered

```
R1: SIGN. The held-out paired effect (ctx3 - ctx12, mean OOS t) is POSITIVE.
    CONFIRMED if the held-out mean > 0. REFUTED if <= 0 -- and a REFUTED R1 is
    strong evidence the discovery effect was noise, power notwithstanding.

R2: MAGNITUDE. The held-out 95% CI contains the discovery estimate +0.484.
    CONFIRMED if so. This is a consistency check, not a significance test.

R3: ZERO. The held-out 95% CI excludes zero.
    Pre-declared as UNLIKELY at 13% power. Reported for completeness. A
    failure here is INCONCLUSIVE by the power arithmetic above and must be
    written that way.

R4: The discovery-set effect REPRODUCES at uncapped settings -- i.e. re-running
    the 4 discovery instruments without the training cap still yields a
    positive paired effect. CONFIRMED if the discovery mean > 0.
    A REFUTED R4 would mean the original +0.484 was an artefact of the
    training cap rather than of the lookback, which would retire the
    hypothesis outright.
```

## Reporting rules

* Discovery and held-out sets reported **side by side, NEVER pooled**. Pooling
  them would launder the discovery set into the evidence.
* All 7 instruments reported. None dropped for any reason.
* No threshold moved. No arm added after seeing results.
* If R1 confirms and R3 does not, the honest summary is *"consistent, still
  unproven"* — and the notebook must say the universe is too small to do
  better, rather than implying more work would help.

## Out of scope

No return, no Sharpe, no tradeability claim. Data ends 2022-03-04. Volatility
is a causal realised-vol **proxy** on every run.
