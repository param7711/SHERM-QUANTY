"""
Era-split robustness: does the edge survive in each independent era, or
does one regime carry the whole result?

The full daily history is chopped into non-overlapping calendar eras. The
strategy is re-run inside each era in isolation (SMA warm-up included from
the era's own bars, so no information crosses an era boundary). An edge
that only shows up in one era — say, crypto 2020-21 — is a regime artifact,
not a rule that works.

Run: python research/sma18_trend_forex/era_splits.py [long|both]
Outputs: outputs/era_splits[_SUFFIX].{csv,json}
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
from strategy import SMA_PERIOD, add_sma, performance_stats, run_strategy
from universe import UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "1d")
OUT_DIR = os.path.join(HERE, "outputs")

ERAS = [
    ("1998-2003", "1998-01-01", "2002-12-31"),
    ("2003-2008", "2003-01-01", "2007-12-31"),
    ("2008-2013", "2008-01-01", "2012-12-31"),
    ("2013-2018", "2013-01-01", "2017-12-31"),
    ("2018-2023", "2018-01-01", "2022-12-31"),
    ("2023-2026", "2023-01-01", "2026-12-31"),
]

# An era needs enough bars for the SMA to warm up plus a usable sample.
MIN_BARS_PER_ERA = 250
MIN_TRADES_PER_ERA = 10


def _sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def main(direction: str = "long"):
    suffix = "" if direction == "long" else f"_{direction}"
    rows = []

    for name, meta in UNIVERSE.items():
        path = os.path.join(DATA_DIR, f"{name}.parquet")
        if not os.path.exists(path):
            print(f"  [SKIP] {name}: no cached daily data")
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        for era_name, start, end in ERAS:
            era = df.loc[(df.index >= start) & (df.index <= end)]
            if len(era) < MIN_BARS_PER_ERA:
                continue
            era = add_sma(era)
            trades, equity = run_strategy(era, direction=direction)
            perf = performance_stats(trades, equity)
            if perf["n_trades"] < MIN_TRADES_PER_ERA:
                continue
            rows.append({
                "asset": name, "asset_class": meta["asset_class"], "era": era_name,
                "start": era.index[0].date().isoformat(), "end": era.index[-1].date().isoformat(),
                "n_bars": len(era), "n_trades": perf["n_trades"],
                "win_rate": perf["win_rate"], "sharpe": perf["sharpe"],
                "cagr_pct": perf["cagr_pct"], "total_return_pct": perf["total_return_pct"],
                "max_drawdown_pct": perf["max_drawdown_pct"],
                "profit_factor": perf["profit_factor"],
            })

    era_df = pd.DataFrame(rows)
    era_df.to_csv(os.path.join(OUT_DIR, f"era_splits{suffix}.csv"), index=False)

    # Per asset: in how many of its testable eras was the edge positive?
    per_asset = era_df.groupby(["asset", "asset_class"]).agg(
        n_eras=("era", "count"),
        n_eras_positive=("sharpe", lambda s: int((s > 0).sum())),
        mean_sharpe=("sharpe", "mean"),
        worst_era_sharpe=("sharpe", "min"),
        best_era_sharpe=("sharpe", "max"),
    ).reset_index()
    per_asset["pct_eras_positive"] = per_asset["n_eras_positive"] / per_asset["n_eras"]
    per_asset = per_asset.sort_values("pct_eras_positive", ascending=False)

    by_era = era_df.groupby("era").agg(
        n_assets=("asset", "count"),
        mean_sharpe=("sharpe", "mean"),
        pct_positive=("sharpe", lambda s: float((s > 0).mean())),
    ).reindex([e[0] for e in ERAS]).dropna(how="all").reset_index()

    by_class_era = era_df.groupby(["asset_class", "era"]).agg(
        n_assets=("asset", "count"),
        mean_sharpe=("sharpe", "mean"),
        pct_positive=("sharpe", lambda s: float((s > 0).mean())),
    ).reset_index()

    n_pos = int((era_df["sharpe"] > 0).sum())
    payload = {
        "direction": direction,
        "eras": [e[0] for e in ERAS],
        "rows": era_df.round(4).to_dict(orient="records"),
        "per_asset": per_asset.round(4).to_dict(orient="records"),
        "by_era": by_era.round(4).to_dict(orient="records"),
        "by_class_era": by_class_era.round(4).to_dict(orient="records"),
        "summary": {
            "n_asset_eras": int(len(era_df)),
            "n_positive": n_pos,
            "pct_positive": round(n_pos / len(era_df), 4) if len(era_df) else None,
            "mean_sharpe": round(float(era_df["sharpe"].mean()), 4) if len(era_df) else None,
            "n_assets_all_eras_positive": int((per_asset["pct_eras_positive"] == 1.0).sum()),
            "n_assets": int(len(per_asset)),
        },
    }
    with open(os.path.join(OUT_DIR, f"era_splits{suffix}.json"), "w") as f:
        json.dump(_sanitize(payload), f, default=str)

    pd.set_option("display.width", 180)
    print(f"\n=== Era splits (direction={direction}) ===")
    print(f"{len(era_df)} asset-era cells, {n_pos} positive ({n_pos/max(len(era_df),1):.0%})\n")
    print("By era:")
    print(by_era.round(3).to_string(index=False))
    print("\nBy asset class x era (mean Sharpe):")
    pivot = era_df.pivot_table(index="asset_class", columns="era", values="sharpe", aggfunc="mean")
    print(pivot.reindex(columns=[e[0] for e in ERAS]).round(3).to_string())
    print("\nAssets positive in every testable era:",
          ", ".join(per_asset[per_asset["pct_eras_positive"] == 1.0]["asset"].tolist()) or "(none)")
    return era_df, payload


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "long")
