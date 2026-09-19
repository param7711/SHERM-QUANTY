"""
Two more ways the result could be luck, tested directly.

1. PARAMETER SENSITIVITY. If 18 is the only SMA length that works, the
   result is a curve-fit artifact — someone tried lengths until one paid.
   If performance sits on a broad plateau across neighbouring lengths, the
   effect is a property of the market, not of the number 18. This sweeps
   SMA 5..60 on every instrument and reports where 18 ranks.

2. TRANSACTION COSTS. A backtest edge that dies at 5 bps is not an edge,
   it is a spread-collection illusion. This charges a round-trip cost on
   every trade and finds the breakeven cost at which each instrument's
   Sharpe hits zero.

Run: python research/sma18_trend_forex/robustness_extras.py [direction]
Outputs: outputs/robustness_extras[_SUFFIX].json
"""

import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markov_validation import _rolling_mean
from strategy import SMA_PERIOD, bars_per_year, run_strategy_arrays, sharpe_from_equity
from universe import UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "1d")
OUT_DIR = os.path.join(HERE, "outputs")

PERIODS = list(range(5, 61))
COST_BPS = [0, 2, 5, 10, 15, 20, 30, 50]
MIN_BARS = 500


def _sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _load(name):
    path = os.path.join(DATA_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df if len(df) >= MIN_BARS else None


def sharpe_with_cost(rets_pct, equity, bpy, n_bars, cost_bps):
    """Apply a round-trip cost per trade to the equity curve.

    Costs are applied as a flat haircut per trade spread over the curve's
    own returns, which is the standard approximation when you only have
    bar-level equity: total cost = n_trades * cost, subtracted geometrically.
    """
    if cost_bps == 0:
        return sharpe_from_equity(equity, bpy), float((equity[-1] - 1) * 100)
    n_trades = len(rets_pct)
    if n_trades == 0:
        return sharpe_from_equity(equity, bpy), float((equity[-1] - 1) * 100)
    net = (1 + rets_pct / 100.0) * (1 - cost_bps / 10000.0)
    net_equity_final = float(np.prod(net))
    # Rebuild a cost-adjusted curve by scaling each bar's return down by the
    # average per-bar drag, so Sharpe reflects the cost too.
    drag_per_bar = (n_trades * cost_bps / 10000.0) / max(n_bars - 1, 1)
    ret = np.diff(equity) / equity[:-1] - drag_per_bar
    sd = ret.std()
    sharpe = float(ret.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0
    return sharpe, (net_equity_final - 1) * 100


def main(direction: str = "long"):
    suffix = "" if direction == "long" else f"_{direction}"
    period_rows, cost_rows = [], []

    for name, meta in UNIVERSE.items():
        df = _load(name)
        if df is None:
            continue
        o, h, l, c = (df["open"].values, df["high"].values,
                      df["low"].values, df["close"].values)
        bpy = bars_per_year(df.index)

        # --- 1. SMA period sweep -------------------------------------------
        sweep = {}
        for p in PERIODS:
            sma = _rolling_mean(c, p)
            rets, _, _, eq = run_strategy_arrays(o, h, l, c, sma, direction=direction)
            sweep[p] = sharpe_from_equity(eq, bpy)
        vals = np.array([sweep[p] for p in PERIODS])
        s18 = sweep[SMA_PERIOD]
        rank18 = int((vals > s18).sum()) + 1
        period_rows.append({
            "asset": name, "asset_class": meta["asset_class"],
            "sharpe_18": s18,
            "sweep": {str(p): round(float(sweep[p]), 4) for p in PERIODS},
            "pct_periods_positive": float((vals > 0).mean()),
            "mean_sharpe_all_periods": float(vals.mean()),
            "best_period": int(PERIODS[int(np.argmax(vals))]),
            "best_sharpe": float(vals.max()),
            "rank_of_18": rank18, "n_periods": len(PERIODS),
        })

        # --- 2. Cost sensitivity -------------------------------------------
        sma18 = _rolling_mean(c, SMA_PERIOD)
        rets, _, _, eq = run_strategy_arrays(o, h, l, c, sma18, direction=direction)
        per_cost = {}
        breakeven = None
        for bps in COST_BPS:
            sh, tot = sharpe_with_cost(rets, eq, bpy, len(c), bps)
            per_cost[str(bps)] = {"sharpe": round(sh, 4), "total_return_pct": round(tot, 2)}
            if breakeven is None and sh <= 0:
                breakeven = bps
        cost_rows.append({
            "asset": name, "asset_class": meta["asset_class"],
            "n_trades": int(len(rets)),
            "by_cost": per_cost,
            "breakeven_bps": breakeven if breakeven is not None else f">{COST_BPS[-1]}",
        })

    pdf = pd.DataFrame(period_rows)
    cdf = pd.DataFrame(cost_rows)

    # Aggregate the sweep by asset class for the report's plateau chart.
    sweep_by_class = {}
    for cls in pdf["asset_class"].unique():
        sub = pdf[pdf["asset_class"] == cls]
        sweep_by_class[cls] = {
            str(p): round(float(np.mean([r["sweep"][str(p)] for _, r in sub.iterrows()])), 4)
            for p in PERIODS
        }

    cost_by_class = {}
    for cls in cdf["asset_class"].unique():
        sub = cdf[cdf["asset_class"] == cls]
        cost_by_class[cls] = {
            str(b): round(float(np.mean([r["by_cost"][str(b)]["sharpe"] for _, r in sub.iterrows()])), 4)
            for b in COST_BPS
        }

    payload = {
        "direction": direction, "periods": PERIODS, "cost_bps": COST_BPS,
        "sma_period_tested": SMA_PERIOD,
        "per_asset_periods": period_rows,
        "per_asset_costs": cost_rows,
        "sweep_by_class": sweep_by_class,
        "cost_by_class": cost_by_class,
        "summary": {
            "n_assets": int(len(pdf)),
            "mean_pct_periods_positive": round(float(pdf["pct_periods_positive"].mean()), 4),
            "median_rank_of_18": int(pdf["rank_of_18"].median()),
            "n_assets_where_18_is_best": int((pdf["rank_of_18"] == 1).sum()),
            "n_assets_positive_at_10bps": int(sum(
                1 for r in cost_rows if r["by_cost"]["10"]["sharpe"] > 0)),
            "n_assets_positive_at_20bps": int(sum(
                1 for r in cost_rows if r["by_cost"]["20"]["sharpe"] > 0)),
        },
    }
    with open(os.path.join(OUT_DIR, f"robustness_extras{suffix}.json"), "w") as f:
        json.dump(_sanitize(payload), f, default=str)

    print(f"=== Parameter sensitivity (SMA {PERIODS[0]}..{PERIODS[-1]}, direction={direction}) ===")
    print(f"mean % of SMA lengths that are profitable: {payload['summary']['mean_pct_periods_positive']:.0%}")
    print(f"median rank of SMA-18 among {len(PERIODS)} lengths: "
          f"{payload['summary']['median_rank_of_18']} "
          f"(1 = best; a high rank means 18 is nothing special, which is good)")
    print(f"assets where 18 happens to be the single best length: "
          f"{payload['summary']['n_assets_where_18_is_best']}/{len(pdf)}")
    print("\nmean Sharpe by SMA length, by class (every 5th):")
    for cls, sw in sweep_by_class.items():
        pts = " ".join(f"{p}:{sw[str(p)]:+.2f}" for p in PERIODS[::5])
        print(f"  {cls:10s} {pts}")

    print(f"\n=== Cost sensitivity ===")
    print("mean Sharpe by round-trip cost (bps), by class:")
    for cls, cc in cost_by_class.items():
        print(f"  {cls:10s} " + " ".join(f"{b}bps:{cc[str(b)]:+.2f}" for b in COST_BPS))
    print(f"\nassets still positive at 10bps: {payload['summary']['n_assets_positive_at_10bps']}/{len(cdf)}")
    print(f"assets still positive at 20bps: {payload['summary']['n_assets_positive_at_20bps']}/{len(cdf)}")
    return payload


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "long")
