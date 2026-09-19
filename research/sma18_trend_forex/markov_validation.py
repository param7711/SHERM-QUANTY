"""
Markov-chain synthetic-series validation: is the edge real, or luck?

The logic of the test
---------------------
A backtest Sharpe means nothing on its own — you need to know what Sharpe
this rule would produce on data with no exploitable edge but otherwise the
same statistical character. So for each instrument we build synthetic price
histories, re-run the *identical* strategy on each, and ask where the real
result falls in that distribution.

Two different null models are generated, because they answer different
questions:

  IID SHUFFLE — the asset's own daily returns, randomly reordered. This
  destroys every trace of serial structure (trends, momentum, mean
  reversion) while keeping the exact return distribution: same mean, same
  volatility, same fat tails, same skew. If the strategy still makes money
  here, its "edge" is coming from the shape of the return distribution
  (drift and skew), not from trend behaviour at all.

  MARKOV CHAIN — returns are discretized into states (by magnitude and
  sign), a first-order transition matrix is estimated from the real series,
  and synthetic paths are drawn from that chain, with each state emitting a
  return sampled from the real returns that fell in that state. This keeps
  the asset's first-order serial dependence — including whatever short-run
  trend persistence a one-step-memory process can express — plus the real
  return distribution. Beating this null means the rule is exploiting
  structure deeper than one-step persistence.

Reading the result: p_value is the fraction of synthetic runs whose Sharpe
matched or beat the real one. Low p under BOTH nulls is the strong case for
a real edge. A high p under the Markov null while low under the shuffle null
means the edge is real but is just first-order persistence — still tradable,
but not mysterious.

Bars are synthesized as full OHLC, not just closes, because the exit rule
needs intrabar highs and lows. Bar shapes (open/high/low relative to close)
are resampled from the instrument's own real bars, so synthetic candles have
realistic ranges and always satisfy low <= min(open, close) <= max(open,
close) <= high.

Run: python research/sma18_trend_forex/markov_validation.py [n_sims] [direction]
Outputs: outputs/markov_validation[_SUFFIX].{csv,json}
"""

import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import (SMA_PERIOD, add_sma, bars_per_year, performance_stats,
                       run_strategy, run_strategy_arrays, sharpe_from_equity)
from universe import UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "1d")
OUT_DIR = os.path.join(HERE, "outputs")

N_SIMS = 400
N_STATES = 5           # return quintiles: strong down .. strong up
SEED = 20260919
MIN_BARS = 500


def _sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Null-model generators
# ---------------------------------------------------------------------------

def fit_markov(log_ret: np.ndarray, n_states: int = N_STATES):
    """Discretize returns into quantile states, estimate the first-order
    transition matrix, and keep the pool of real returns per state."""
    edges = np.quantile(log_ret, np.linspace(0, 1, n_states + 1)[1:-1])
    states = np.searchsorted(edges, log_ret)          # 0 .. n_states-1

    counts = np.zeros((n_states, n_states))
    np.add.at(counts, (states[:-1], states[1:]), 1)
    # Laplace smoothing so no transition is impossible in the synthetic world.
    counts += 1e-6
    trans = counts / counts.sum(axis=1, keepdims=True)

    pools = [log_ret[states == s] for s in range(n_states)]
    pools = [p if len(p) else log_ret for p in pools]
    return trans, pools, states, edges


def markov_paths(trans, pools, states, n_steps, n_sims, rng):
    """Draw n_sims synthetic log-return paths from the fitted chain."""
    n_states = trans.shape[0]
    cum = trans.cumsum(axis=1)
    out = np.empty((n_sims, n_steps))
    cur = rng.choice(states, size=n_sims)
    # Pre-draw uniforms for state transitions and within-state return picks.
    u_state = rng.random((n_sims, n_steps))
    for t in range(n_steps):
        cur = (u_state[:, t][:, None] > cum[cur]).sum(axis=1)
        cur = np.clip(cur, 0, n_states - 1)
        for s in range(n_states):
            m = cur == s
            k = int(m.sum())
            if k:
                out[m, t] = rng.choice(pools[s], size=k, replace=True)
    return out


def shuffle_paths(log_ret, n_steps, n_sims, rng):
    """IID null: the same returns in random order."""
    return rng.choice(log_ret, size=(n_sims, n_steps), replace=True)


def synth_ohlc_arrays(log_ret_path, first_close, shape_pools, edges, rng):
    """Turn a log-return path into OHLC arrays using resampled real bar shapes.

    Shapes are drawn CONDITIONAL on the bar's own return bucket, not at
    random. In real data a day's return and the position of its high/low
    are strongly linked (on gold, corr(return, low/close) = -0.47): big up
    days close near their high, down days close near their low. Pairing
    shapes at random would hand up-days the deep lows of down-days and
    manufacture SMA touches that could never happen, making the null model
    easier to beat and the p-values look better than they are.
    """
    close = first_close * np.exp(np.cumsum(log_ret_path))
    states = np.searchsorted(edges, log_ret_path)
    shapes = np.empty((len(close), 3))
    for s, pool in enumerate(shape_pools):
        m = states == s
        k = int(m.sum())
        if k:
            shapes[m] = pool[rng.integers(0, len(pool), size=k)]
    open_ = close * shapes[:, 0]
    high = close * shapes[:, 1]
    low = close * shapes[:, 2]
    # Guarantee a valid bar even after mixing a shape onto a different close.
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    return open_, high, low, close


def _rolling_mean(x: np.ndarray, period: int) -> np.ndarray:
    """SMA with a NaN warm-up, matching pandas rolling(period).mean()."""
    out = np.full(len(x), np.nan)
    if len(x) >= period:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[period - 1:] = (c[period:] - c[:-period]) / period
    return out


# ---------------------------------------------------------------------------
# Per-asset validation
# ---------------------------------------------------------------------------

def validate_asset(name: str, meta: dict, n_sims: int, direction: str, rng) -> dict:
    path = os.path.join(DATA_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    if len(df) < MIN_BARS:
        return None

    real = add_sma(df)
    real_trades, real_equity = run_strategy(real, direction=direction)
    real_perf = performance_stats(real_trades, real_equity)

    close = df["close"].values
    log_ret = np.diff(np.log(close))
    log_ret = log_ret[np.isfinite(log_ret)]
    if len(log_ret) < MIN_BARS - 1:
        return None

    trans, pools, states, edges = fit_markov(log_ret)

    # Real bar shapes, bucketed by the return state of the bar they came
    # from, so synthetic bars get shapes that match their own direction.
    shape_all = np.column_stack([
        (df["open"] / df["close"]).values[1:],
        (df["high"] / df["close"]).values[1:],
        (df["low"] / df["close"]).values[1:],
    ])[:len(states)]
    finite = np.isfinite(shape_all).all(axis=1)
    shape_pools = []
    for s in range(N_STATES):
        pool = shape_all[finite & (states[:len(shape_all)] == s)]
        shape_pools.append(pool if len(pool) else shape_all[finite])
    n_steps = len(log_ret)
    bpy = bars_per_year(df.index)
    n_years = n_steps / bpy if bpy > 0 else np.nan

    results = {}
    for null_name, paths in (
        ("markov", markov_paths(trans, pools, states, n_steps, n_sims, rng)),
        ("shuffle", shuffle_paths(log_ret, n_steps, n_sims, rng)),
    ):
        sharpes = np.empty(n_sims)
        cagrs = np.empty(n_sims)
        for s in range(n_sims):
            o, h, l, c = synth_ohlc_arrays(paths[s], close[0], shape_pools, edges, rng)
            sma = _rolling_mean(c, SMA_PERIOD)
            _, _, _, eq = run_strategy_arrays(o, h, l, c, sma, direction=direction)
            sharpes[s] = sharpe_from_equity(eq, bpy)
            cagrs[s] = (eq[-1] ** (1 / n_years) - 1) * 100 if n_years and n_years > 0 and eq[-1] > 0 else np.nan
        real_sharpe = real_perf["sharpe"]
        # One-sided p: how often does a no-edge world match or beat reality?
        p_value = float((sharpes >= real_sharpe).mean())
        results[null_name] = {
            "p_value": p_value,
            "synth_sharpe_mean": float(np.nanmean(sharpes)),
            "synth_sharpe_std": float(np.nanstd(sharpes)),
            "synth_sharpe_p05": float(np.nanpercentile(sharpes, 5)),
            "synth_sharpe_p50": float(np.nanpercentile(sharpes, 50)),
            "synth_sharpe_p95": float(np.nanpercentile(sharpes, 95)),
            "synth_cagr_p50": float(np.nanpercentile(cagrs, 50)),
            "z_score": float((real_sharpe - np.nanmean(sharpes)) / np.nanstd(sharpes))
                        if np.nanstd(sharpes) > 0 else None,
            "sharpe_samples": [round(float(x), 4) for x in sharpes[:400]],
        }

    return {
        "asset": name, "asset_class": meta["asset_class"],
        "start": df.index[0].date().isoformat(), "end": df.index[-1].date().isoformat(),
        "n_bars": len(df), "n_sims": n_sims, "direction": direction,
        "real_sharpe": real_perf["sharpe"], "real_cagr_pct": real_perf["cagr_pct"],
        "real_n_trades": real_perf["n_trades"], "real_win_rate": real_perf["win_rate"],
        "markov": results["markov"], "shuffle": results["shuffle"],
    }


def main(n_sims: int = N_SIMS, direction: str = "long"):
    suffix = "" if direction == "long" else f"_{direction}"
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    records = []
    t0 = time.time()
    for name, meta in UNIVERSE.items():
        r = validate_asset(name, meta, n_sims, direction, rng)
        if r is None:
            print(f"  [SKIP] {name}")
            continue
        records.append(r)
        print(f"  {name:10s} real_sharpe={r['real_sharpe']:+.2f}  "
              f"markov_p={r['markov']['p_value']:.3f}  shuffle_p={r['shuffle']['p_value']:.3f}  "
              f"({time.time()-t0:.0f}s)")

    flat = pd.DataFrame([{
        "asset": r["asset"], "asset_class": r["asset_class"], "n_bars": r["n_bars"],
        "start": r["start"], "end": r["end"],
        "real_sharpe": r["real_sharpe"], "real_cagr_pct": r["real_cagr_pct"],
        "real_n_trades": r["real_n_trades"],
        "markov_p": r["markov"]["p_value"], "markov_synth_mean": r["markov"]["synth_sharpe_mean"],
        "markov_z": r["markov"]["z_score"],
        "shuffle_p": r["shuffle"]["p_value"], "shuffle_synth_mean": r["shuffle"]["synth_sharpe_mean"],
        "shuffle_z": r["shuffle"]["z_score"],
    } for r in records])
    flat.to_csv(os.path.join(OUT_DIR, f"markov_validation{suffix}.csv"), index=False)

    n = len(flat)
    payload = {
        "direction": direction, "n_sims": n_sims, "n_states": N_STATES,
        "assets": records,
        "flat": flat.round(4).to_dict(orient="records"),
        "summary": {
            "n_assets": n,
            "n_markov_sig_05": int((flat["markov_p"] < 0.05).sum()),
            "n_shuffle_sig_05": int((flat["shuffle_p"] < 0.05).sum()),
            "n_both_sig_05": int(((flat["markov_p"] < 0.05) & (flat["shuffle_p"] < 0.05)).sum()),
            "median_markov_p": round(float(flat["markov_p"].median()), 4) if n else None,
            "median_shuffle_p": round(float(flat["shuffle_p"].median()), 4) if n else None,
            # Benjamini-Hochberg at 5% across assets, to handle testing 30 at once.
            "n_markov_sig_bh": int(_bh_count(flat["markov_p"].values, 0.05)),
            "n_shuffle_sig_bh": int(_bh_count(flat["shuffle_p"].values, 0.05)),
        },
    }
    with open(os.path.join(OUT_DIR, f"markov_validation{suffix}.json"), "w") as f:
        json.dump(_sanitize(payload), f, default=str)

    print(f"\n=== Markov validation (direction={direction}, {n_sims} sims/asset) ===")
    print(flat[["asset", "asset_class", "real_sharpe", "markov_p", "shuffle_p"]]
          .sort_values("markov_p").round(3).to_string(index=False))
    s = payload["summary"]
    print(f"\nSignificant at p<0.05 — markov: {s['n_markov_sig_05']}/{n}, "
          f"shuffle: {s['n_shuffle_sig_05']}/{n}, both: {s['n_both_sig_05']}/{n}")
    print(f"After Benjamini-Hochberg (5% FDR) — markov: {s['n_markov_sig_bh']}/{n}, "
          f"shuffle: {s['n_shuffle_sig_bh']}/{n}")
    return flat, payload


def _bh_count(pvals, alpha=0.05):
    """How many hypotheses survive Benjamini-Hochberg FDR control."""
    p = np.sort(np.asarray(pvals, dtype=float))
    m = len(p)
    if m == 0:
        return 0
    thresh = alpha * np.arange(1, m + 1) / m
    passing = np.where(p <= thresh)[0]
    return int(passing[-1] + 1) if len(passing) else 0


if __name__ == "__main__":
    _n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SIMS
    _d = sys.argv[2] if len(sys.argv) > 2 else "long"
    main(_n, _d)
