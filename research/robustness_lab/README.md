# Time-Series Overfitting & Robustness Laboratory

A Jupyter notebook that tests whether a trading strategy's apparent edge is a
genuine statistical relationship or a fit to one particular historical path.

The question it answers:

> *If I had observed a different but statistically similar history of the
> market, would my strategy probably still have appeared profitable?*

## The method

```
REAL MARKET DATA
      -> MEASURE ITS STATISTICAL PROPERTIES
      -> BUILD GENERATORS CALIBRATED TO THOSE PROPERTIES
      -> VALIDATE THE GENERATORS AGAINST THE REAL DATA
      -> FREEZE THEM
      -> RUN THE STRATEGY
      -> COMPARE
```

Generators are calibrated against *market statistics* and never against
*strategy performance*. The notebook enforces this by ordering: generators are
built, scored and frozen in Sections 08-09, before a strategy is defined in
Section 10. A fingerprint of every generator's calibration is recorded at the
freeze and re-verified in the Section 22 leakage audit.

## Running it

The notebook needs a scientific Python stack. From the repository root:

```bash
uv venv --python 3.14 .venv314
uv pip install --python .venv314/bin/python \
    numpy pandas scipy matplotlib statsmodels arch scikit-learn \
    joblib tqdm nbformat nbclient ipykernel pyarrow
.venv314/bin/python -m ipykernel install --user \
    --name robustlab-py314 --display-name "Python 3.14 (robustness lab)"
```

Then open `robustness_lab.ipynb` and run all cells, or run headless:

```bash
cd research/robustness_lab
ROBUSTLAB_MODE=QUICK python build_nb.py --exec      # ~7 min
ROBUSTLAB_MODE=FULL  python build_nb.py --exec      # production counts
```

`QUICK` uses 120 synthetic paths per generator; `FULL` uses 1000. Both are
configurable in Section 00. `Restart Kernel -> Run All` works from a clean
kernel with no hidden state.

Only `numpy`, `pandas`, `scipy` and `matplotlib` are required. `arch`,
`statsmodels` and `scikit-learn` each unlock specific generators or tests; when
one is missing the dependent section reports itself as SKIPPED rather than
silently substituting another method.

## Pointing it at your own data

Section 00:

```python
DATA_PATH   = "path/to/data"     # directory of <ASSET>.parquet, or one CSV/Parquet
DATE_COLUMN = "Date"
PRICE_COLUMN = "close"
ASSETS       = ["GOLD", "BTCUSD"]
PRIMARY_ASSET = "GOLD"
```

or set `ROBUSTLAB_DATA` in the environment. If the path does not exist the
notebook falls back to a calibrated demonstration series and says so loudly, so
it always runs end to end.

## Plugging in your own strategy

Write a function with this signature and register it:

```python
def my_strategy(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """data is a canonical OHLCV frame. Return a frame indexed like `data`
    carrying at least a `position` column: the target exposure decided at each
    bar's CLOSE (+1 long, 0 flat, -1 short, fractional for sized positions)."""
    ...
    return pd.DataFrame({"signal": sig, "position": pos}, index=data.index)

STRATEGY_REGISTRY["my_strategy"] = my_strategy
STRATEGY_NAME = "my_strategy"          # in Section 00
```

Nothing else changes. Every generator and every test operates through that one
interface.

Note what the strategy does **not** do: it never computes its own P&L. The
position lag, transaction costs, slippage and execution delay are applied in
exactly one place (`apply_execution`), for every strategy including the
placebos. That is deliberate — applying a position to the same bar that
generated the signal is the most common backtesting bug, it inflates results
enormously, and it is invisible in the output. Centralising it means a
plugged-in strategy cannot accidentally exempt itself, and there is a single
function for the leakage audit to verify.

## What it runs

| section | contents |
|---|---|
| 00-04 | configuration, dependency check, loading, validation, exploratory analysis |
| 05 | statistical characterisation: distribution, tails, dependence, volatility, trend, drawdowns, shocks |
| 06 | chronological splits: holdout, expanding and rolling walk-forward, embargo |
| 07 | regime detection: quantile, Gaussian mixture, and a hand-rolled HMM with a parameter-recovery test |
| 08 | 20 synthetic generators: IID / moving-block / circular / stationary bootstraps, regime-conditioned, Markov regime, shock-preserving, GARCH / EGARCH / GJR, filtered historical simulation, Fourier and IAAFT surrogates, trend-preserving, parametric jump-diffusion |
| 09 | fidelity scoring against the real data, with a self-coverage calibration that establishes the realistic ceiling; generators frozen |
| 10-12 | strategy interface, backtest engine, no-look-ahead verification, historical backtest |
| 13 | synthetic path test, split by measured temporal-structure preservation, with a buy-and-hold benchmark on identical paths |
| 14 | parameter surfaces, perturbation, and the data-mining tax measured on synthetic markets |
| 15 | walk-forward with strict train -> validate -> test isolation |
| 16 | cost, slippage, delay, signal-noise and data-perturbation sensitivity |
| 17 | placebo controls matched on exposure and trade count |
| 18 | trade-order Monte Carlo |
| 19 | Bonferroni, Benjamini-Hochberg, Deflated Sharpe Ratio, PBO via CSCV |
| 20 | stress scenarios anchored to historical extremes |
| 21-23 | dashboard, leakage audit, evidence-based report, full export |

## Outputs

Written to `results/`:

```
historical_results.csv        synthetic_results.csv       robustness_results.csv
parameter_sensitivity.csv     parameter_perturbation.csv  walk_forward_results.csv
monte_carlo_results.csv       placebo_results.csv         stress_results.csv
cost_sensitivity.csv          data_perturbation.csv       data_mining_tax.csv
synthetic_statistics.csv      synthetic_distance_tests.csv
leakage_audit.csv             experiment_config.json
final_report.md               final_report.html
figures/dashboard.png         synthetic_samples/<generator>.csv
```

`experiment_config.json` records the seed, data range, every parameter, the
selected volatility model and its selection rule, and the generator
fingerprints — enough to reproduce or audit the run.

## What it deliberately does not produce

There is no overall robustness score. A number like "87/100" implies a
calibrated scale that does not exist, and collapsing independent diagnostics
into one figure destroys exactly the information that makes them useful. The
report instead separates four things that are routinely conflated: evidence
(what was measured), statistical significance (what survives a null),
assumptions (what was taken on faith), and limitations (what this cannot show).

## Source layout

The notebook is authored as ordinary Python files in `nbsrc/` using the `# %%`
cell convention, because a 4,000-line `.ipynb` is impossible to diff or grep.
`build_nb.py` assembles them into `robustness_lab.ipynb`, and can execute it
from a clean kernel and report any failing cell:

```bash
python build_nb.py              # build only
python build_nb.py --exec       # build, execute, report failures
```

`robustness_lab_flat.py` is the concatenated source as a runnable script — the
fast path for iterating without notebook overhead.
