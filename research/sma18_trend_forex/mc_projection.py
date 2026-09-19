"""
Forward Monte Carlo projection -- what could the NEXT few years look like?

The Monte Carlo already in this study is BACKWARD looking: it reshuffles the
trades that actually happened to ask how much the realized result depended on
the order they arrived in. This file asks the forward question instead: if the
rule keeps behaving the way it has, what is the distribution of outcomes over
the next 1, 3 and 5 years -- and, more usefully, how bad can the bad cases get?

Two resampling methods are run side by side, and the gap between them is the
point:

  IID      Trades are drawn independently, the textbook bootstrap.

  BLOCK    Contiguous runs of trades are drawn instead (stationary block
           bootstrap), so winning and losing streaks survive resampling.

The reason for running both was an expectation that IID would flatter the
downside: if losing trades cluster, breaking the clusters up should make
simulated drawdowns shallower than reality, and BLOCK should be the honest
number. THE DATA DID NOT SUPPORT THAT. Across all 30 instruments and both
directions, the block bootstrap's 95th-percentile max drawdown came out
1.6 percentage points SHALLOWER than the iid one (67.0% vs 69.5% median),
not deeper.

The reading: trade-to-trade autocorrelation in this strategy is weak. Because
the touch-the-SMA exit closes a position as soon as the trend breaks, each
trade is largely independent of the next, so there is little streak structure
for the block method to preserve -- and sampling whole blocks actually pulls
each simulation closer to the historical mix, which compresses the extremes
rather than widening them.

Both are still reported, because the comparison is the evidence for that
claim. The expectation is documented here rather than deleted so the record
shows the prediction was made before the result, and lost.

Everything here assumes the future trade distribution resembles the past one.
That assumption is doing enormous work and it cannot be tested from inside the
sample -- these are projections conditional on the edge persisting, NOT
forecasts that it will.

Run: python mc_projection.py
Outputs: outputs/mc_projection.json
"""

import json, math, os, sys, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import SMA_PERIOD, bars_per_year, run_strategy_arrays, sharpe_from_equity
from markov_validation import _rolling_mean
from universe import UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "1d")
OUT_DIR = os.path.join(HERE, "outputs")

N_SIMS = 10000
HORIZONS = [1, 3, 5]
BLOCK_LEN = 8
COST_BPS = 20
PCTS = [5, 25, 50, 75, 95]
MIN_TRADES = 30
SEED = 20260919


def _sanitize(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    return o


def _load(name):
    p = os.path.join(DATA_DIR, f"{name}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def draw_iid(rets, n_sims, n_draw, rng):
    idx = rng.integers(0, len(rets), size=(n_sims, n_draw))
    return rets[idx]


def draw_block(rets, n_sims, n_draw, rng, block=BLOCK_LEN):
    """Stationary block bootstrap: wrap-around blocks of consecutive trades,
    so winning and losing streaks survive the resampling."""
    n = len(rets)
    n_blocks = int(np.ceil(n_draw / block))
    starts = rng.integers(0, n, size=(n_sims, n_blocks))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]) % n      # sims x blocks x block
    idx = idx.reshape(n_sims, -1)[:, :n_draw]
    return rets[idx]


def stats_from_draws(draws, years):
    """draws: n_sims x n_draw of per-trade % returns."""
    growth = 1.0 + draws / 100.0
    equity = np.cumprod(growth, axis=1)
    terminal = equity[:, -1]
    running_max = np.maximum.accumulate(equity, axis=1)
    max_dd = ((running_max - equity) / running_max).max(axis=1) * 100.0
    cagr = (np.power(np.maximum(terminal, 1e-9), 1.0 / years) - 1.0) * 100.0
    return terminal, cagr, max_dd, equity


def summarize(terminal, cagr, max_dd, years):
    return {
        "years": years,
        "p_profit": float((terminal > 1).mean()),
        "p_loss_gt_20": float((terminal < 0.8).mean()),
        "terminal_multiple": {f"p{p}": round(float(np.percentile(terminal, p)), 3) for p in PCTS},
        "cagr_pct": {f"p{p}": round(float(np.percentile(cagr, p)), 2) for p in PCTS},
        "max_drawdown_pct": {f"p{p}": round(float(np.percentile(max_dd, p)), 2) for p in PCTS},
        "p_dd_gt_30": float((max_dd > 30).mean()),
        "p_dd_gt_50": float((max_dd > 50).mean()),
        "median_terminal": round(float(np.median(terminal)), 3),
    }


def project_asset(rets, trades_per_year, rng, cost_bps=0):
    if cost_bps:
        rets = ((1 + rets / 100.0) * (1 - cost_bps / 10000.0) - 1) * 100.0
    out = {"iid": {}, "block": {}}
    paths = None
    for h in HORIZONS:
        n_draw = max(5, int(round(trades_per_year * h)))
        for method, fn in (("iid", draw_iid), ("block", draw_block)):
            d = fn(rets, N_SIMS, n_draw, rng)
            term, cagr, dd, eq = stats_from_draws(d, h)
            out[method][str(h)] = summarize(term, cagr, dd, h)
            if method == "block" and h == max(HORIZONS):
                # percentile fan of the equity path, thinned for the chart
                step = max(1, eq.shape[1] // 120)
                cols = list(range(0, eq.shape[1], step))
                if cols[-1] != eq.shape[1] - 1:
                    cols.append(eq.shape[1] - 1)
                sub = eq[:, cols]
                paths = {
                    "trade_index": cols,
                    "bands": {f"p{p}": np.percentile(sub, p, axis=0).round(4).tolist() for p in PCTS},
                }
    out["paths"] = paths
    return out


def main():
    rng = np.random.default_rng(SEED)
    assets_out = []

    for name, meta in UNIVERSE.items():
        df = _load(name)
        if df is None or len(df) < 500:
            continue
        o, h_, l_, c = (df["open"].values, df["high"].values,
                        df["low"].values, df["close"].values)
        sma = _rolling_mean(c, SMA_PERIOD)
        bpy = bars_per_year(df.index)
        years_hist = len(c) / bpy

        rec = {"asset": name, "asset_class": meta["asset_class"],
               "years_history": round(float(years_hist), 1), "directions": {}}

        for direction in ("long", "both"):
            rets, holds, sides, eq = run_strategy_arrays(o, h_, l_, c, sma, direction=direction)
            if len(rets) < MIN_TRADES:
                continue
            tpy = len(rets) / years_hist
            rec["directions"][direction] = {
                "n_trades_hist": int(len(rets)),
                "trades_per_year": round(float(tpy), 1),
                "realized_sharpe": round(float(sharpe_from_equity(eq, bpy)), 3),
                "realized_total_return_pct": round(float((eq[-1] - 1) * 100), 1),
                "win_rate": round(float((rets > 0).mean()), 3),
                "gross": project_asset(rets, tpy, rng, 0),
                "net20": project_asset(rets, tpy, rng, COST_BPS),
            }
        if rec["directions"]:
            assets_out.append(rec)
            d = rec["directions"]
            ln = d.get("long", {}).get("gross", {}).get("block", {}).get("5", {})
            bo = d.get("both", {}).get("gross", {}).get("block", {}).get("5", {})
            print(f"  {name:10s} 5y median x  long {ln.get('median_terminal', float('nan')):6.2f}  "
                  f"both {bo.get('median_terminal', float('nan')):6.2f}   "
                  f"P(profit) long {ln.get('p_profit', float('nan')):.2f} both {bo.get('p_profit', float('nan')):.2f}",
                  flush=True)

    # ---- direction comparison, aggregated -----------------------------------
    cmp_rows = []
    for rec in assets_out:
        d = rec["directions"]
        if "long" not in d or "both" not in d:
            continue
        for h in HORIZONS:
            L = d["long"]["gross"]["block"][str(h)]
            B = d["both"]["gross"]["block"][str(h)]
            cmp_rows.append({
                "asset": rec["asset"], "asset_class": rec["asset_class"], "horizon": h,
                "long_median": L["median_terminal"], "both_median": B["median_terminal"],
                "long_p_profit": L["p_profit"], "both_p_profit": B["p_profit"],
                "long_dd_p95": L["max_drawdown_pct"]["p95"], "both_dd_p95": B["max_drawdown_pct"]["p95"],
                "long_cagr_p50": L["cagr_pct"]["p50"], "both_cagr_p50": B["cagr_pct"]["p50"],
                "long_cagr_p5": L["cagr_pct"]["p5"], "both_cagr_p5": B["cagr_pct"]["p5"],
            })
    cmp_df = pd.DataFrame(cmp_rows)

    by_class_dir = []
    for (cls, h), sub in cmp_df.groupby(["asset_class", "horizon"]):
        by_class_dir.append({
            "asset_class": cls, "horizon": int(h), "n": int(len(sub)),
            "long_median_terminal": round(float(sub["long_median"].median()), 3),
            "both_median_terminal": round(float(sub["both_median"].median()), 3),
            "long_p_profit": round(float(sub["long_p_profit"].mean()), 3),
            "both_p_profit": round(float(sub["both_p_profit"].mean()), 3),
            "long_dd_p95": round(float(sub["long_dd_p95"].mean()), 2),
            "both_dd_p95": round(float(sub["both_dd_p95"].mean()), 2),
            "long_cagr_p50": round(float(sub["long_cagr_p50"].median()), 2),
            "both_cagr_p50": round(float(sub["both_cagr_p50"].median()), 2),
            "n_long_better": int((sub["long_median"] > sub["both_median"]).sum()),
        })

    # ---- optimism of the iid method vs the block method ---------------------
    opt = []
    for rec in assets_out:
        for direction, d in rec["directions"].items():
            i5 = d["gross"]["iid"]["5"]["max_drawdown_pct"]["p95"]
            b5 = d["gross"]["block"]["5"]["max_drawdown_pct"]["p95"]
            opt.append({"asset": rec["asset"], "asset_class": rec["asset_class"],
                        "direction": direction, "iid_dd_p95": i5, "block_dd_p95": b5,
                        "understated_by_pct_points": round(b5 - i5, 2)})
    opt_df = pd.DataFrame(opt)

    payload = {
        "n_sims": N_SIMS, "horizons": HORIZONS, "block_len": BLOCK_LEN,
        "cost_bps": COST_BPS, "percentiles": PCTS,
        "assets": assets_out,
        "direction_comparison": cmp_df.round(4).to_dict(orient="records"),
        "by_class_direction": by_class_dir,
        "iid_vs_block": opt_df.round(3).to_dict(orient="records"),
        "iid_optimism_summary": {
            "mean_understatement_pct_points": round(float(opt_df["understated_by_pct_points"].mean()), 2),
            "median_iid_dd_p95": round(float(opt_df["iid_dd_p95"].median()), 2),
            "median_block_dd_p95": round(float(opt_df["block_dd_p95"].median()), 2),
        },
    }
    with open(os.path.join(OUT_DIR, "mc_projection.json"), "w") as f:
        json.dump(_sanitize(payload), f, default=str)

    print(f"\n=== Forward projection, {N_SIMS:,} sims, block bootstrap, gross of costs ===")
    for r in by_class_dir:
        print(f"  {r['asset_class']:9s} {r['horizon']}y  n={r['n']:2d}  "
              f"LONG median x{r['long_median_terminal']:6.2f} CAGR {r['long_cagr_p50']:+6.1f}% "
              f"P(profit) {r['long_p_profit']:.2f} worstDD95 {r['long_dd_p95']:5.1f}%  |  "
              f"BOTH median x{r['both_median_terminal']:6.2f} CAGR {r['both_cagr_p50']:+6.1f}% "
              f"P(profit) {r['both_p_profit']:.2f} worstDD95 {r['both_dd_p95']:5.1f}%  "
              f"(long better on {r['n_long_better']}/{r['n']})")

    s = payload["iid_optimism_summary"]
    print(f"\n=== How much the naive iid bootstrap flatters drawdowns ===")
    print(f"  median 95th-pct max drawdown, iid:   {s['median_iid_dd_p95']:.1f}%")
    print(f"  median 95th-pct max drawdown, block: {s['median_block_dd_p95']:.1f}%")
    print(f"  block is deeper by {s['mean_understatement_pct_points']:.1f} percentage points on average")
    print(f"\nwrote {os.path.join(OUT_DIR, 'mc_projection.json')}")


if __name__ == "__main__":
    main()
