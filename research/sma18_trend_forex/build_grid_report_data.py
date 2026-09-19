"""Builds the compact JSON payload for the timeframe x asset-class grid section.

Usage: python build_grid_report_data.py [long|both]
"""

import json
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

TIMEFRAME_ORDER = ["15m", "30m", "4h", "1d"]

# One illustrative fan chart per asset class (its strongest showing) plus one
# cautionary case, so the Monte Carlo section doesn't try to show all 80.
CURATED_MC = {
    "long": [
        ("BTCUSD", "1d", "Best crypto result on the longest sample"),
        ("GOLD", "1d", "Deepest commodity history (2000-)"),
        ("ETHUSD", "1d", "Second crypto, independent history"),
        ("CRUDE", "1d", "Energy, 26 years of daily bars"),
        ("COFFEE", "30m", "Cautionary case — worst cell in the grid"),
    ],
    "both": [
        ("BTCUSD", "1d", "Same pick as long-only, for comparison"),
        ("GOLD", "1d", "Same pick as long-only, for comparison"),
        ("ETHUSD", "1d", "Crypto with shorts added"),
        ("CRUDE", "1d", "Commodity with shorts added"),
        ("COFFEE", "30m", "Cautionary case, shorts added"),
    ],
}


def _sanitize(obj):
    """Recursively replace NaN/Inf with None — Python's json module writes
    the literal (invalid-JSON) tokens NaN/Infinity by default, which then
    fail JSON.parse in the browser."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def main(direction: str = "long"):
    suffix = "" if direction == "long" else f"_{direction}"
    g = pd.read_csv(os.path.join(OUT_DIR, f"grid_summary{suffix}.csv"))
    ok = g[g["error"].isna()].copy().drop(columns=["error"])

    corr_by_tf = {
        tf: round(ok.loc[ok.timeframe == tf, ["ann_vol", "sharpe"]].corr().iloc[0, 1], 3)
        for tf in TIMEFRAME_ORDER
    }
    corr_pooled = round(ok[["ann_vol", "sharpe"]].corr().iloc[0, 1], 3)

    by_tf = ok.groupby("timeframe")[
        ["sharpe", "cagr_pct", "total_return_pct", "win_rate", "max_drawdown_pct", "n_trades"]
    ].mean().reindex(TIMEFRAME_ORDER)

    by_class = ok.groupby("asset_class")[
        ["sharpe", "cagr_pct", "win_rate", "max_drawdown_pct", "n_trades"]
    ].mean().sort_values("sharpe", ascending=False)

    by_voltier = ok.groupby(["timeframe", "vol_tier_empirical"])[
        ["sharpe", "cagr_pct", "win_rate", "n_trades", "ann_vol"]
    ].mean().reset_index()

    asset_order = (ok[ok.timeframe == "1d"]
                   .sort_values("ann_vol")["asset"].tolist())
    remaining = [a for a in ok["asset"].unique() if a not in asset_order]
    asset_order += remaining

    heatmap_rows = []
    for asset in asset_order:
        sub = ok[ok.asset == asset]
        if sub.empty:
            continue
        cls = sub["asset_class"].iloc[0]
        cells = {}
        for tf in TIMEFRAME_ORDER:
            r = sub[sub.timeframe == tf]
            cells[tf] = None if r.empty else {
                "sharpe": round(float(r["sharpe"].iloc[0]), 3),
                "n_trades": int(r["n_trades"].iloc[0]),
                "ann_vol": round(float(r["ann_vol"].iloc[0]), 4),
                "win_rate": round(float(r["win_rate"].iloc[0]), 3),
                "cagr_pct": round(float(r["cagr_pct"].iloc[0]), 2),
            }
        heatmap_rows.append({"asset": asset, "asset_class": cls, "cells": cells})

    # Grouped-by-timeframe view of the full grid, each group sorted by
    # Sharpe descending — the organization the report actually shows.
    grid_by_timeframe = []
    for tf in TIMEFRAME_ORDER:
        sub = ok[ok.timeframe == tf].sort_values("sharpe", ascending=False)
        grid_by_timeframe.append({
            "timeframe": tf,
            "rows": sub.round(4).to_dict(orient="records"),
        })

    mc_paths_path = os.path.join(OUT_DIR, f"mc_paths{suffix}.json")
    mc_paths = {}
    if os.path.exists(mc_paths_path):
        with open(mc_paths_path) as f:
            mc_paths = json.load(f)

    curated_mc = []
    for asset, tf, label in CURATED_MC[direction]:
        key = f"{asset}|{tf}"
        detail = mc_paths.get(key)
        row_match = ok[(ok.asset == asset) & (ok.timeframe == tf)]
        if detail is None or row_match.empty:
            continue
        curated_mc.append({
            "asset": asset, "timeframe": tf, "label": label,
            "asset_class": row_match["asset_class"].iloc[0],
            "actual_sharpe": round(float(row_match["sharpe"].iloc[0]), 3),
            "detail": detail,
        })

    payload = {
        "direction": direction,
        "timeframe_order": TIMEFRAME_ORDER,
        "grid": ok.round(4).to_dict(orient="records"),
        "grid_by_timeframe": grid_by_timeframe,
        "skipped": g[g["error"].notna()][["asset", "timeframe", "error"]].to_dict(orient="records"),
        "correlations": {"pooled": corr_pooled, "by_timeframe": corr_by_tf},
        "by_timeframe": by_tf.reset_index().round(4).to_dict(orient="records"),
        "by_asset_class": by_class.reset_index().round(4).to_dict(orient="records"),
        "by_vol_tier": by_voltier.round(4).to_dict(orient="records"),
        "heatmap": heatmap_rows,
        "curated_mc": curated_mc,
        "totals": {"n_combinations": int(len(g)), "n_ok": int(len(ok)),
                    "n_assets": int(ok["asset"].nunique())},
    }

    out_path = os.path.join(OUT_DIR, f"grid_report_data{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(_sanitize(payload), f, default=str)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    _direction = sys.argv[1] if len(sys.argv) > 1 else "long"
    main(_direction)
