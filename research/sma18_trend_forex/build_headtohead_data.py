"""Merges the long-only and long+short grids into a head-to-head comparison
by (asset, timeframe), plus aggregate deltas by timeframe and asset class."""

import json
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

TIMEFRAME_ORDER = ["15m", "30m", "4h", "1d"]


def _sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def main():
    long_df = pd.read_csv(os.path.join(OUT_DIR, "grid_summary.csv"))
    both_df = pd.read_csv(os.path.join(OUT_DIR, "grid_summary_both.csv"))
    long_df = long_df[long_df["error"].isna()].copy()
    both_df = both_df[both_df["error"].isna()].copy()

    keep_cols = ["asset", "timeframe", "asset_class", "sharpe", "cagr_pct",
                 "win_rate", "max_drawdown_pct", "n_trades", "total_return_pct"]
    merged = long_df[keep_cols].merge(
        both_df[keep_cols], on=["asset", "timeframe", "asset_class"],
        suffixes=("_long", "_both"),
    )
    merged["sharpe_delta"] = merged["sharpe_both"] - merged["sharpe_long"]
    merged["cagr_delta"] = merged["cagr_pct_both"] - merged["cagr_pct_long"]
    merged["short_helps"] = merged["sharpe_delta"] > 0

    n_helps = int(merged["short_helps"].sum())
    n_total = len(merged)

    by_tf = merged.groupby("timeframe")[
        ["sharpe_long", "sharpe_both", "sharpe_delta", "cagr_pct_long", "cagr_pct_both"]
    ].mean().reindex(TIMEFRAME_ORDER)

    by_class = merged.groupby("asset_class")[
        ["sharpe_long", "sharpe_both", "sharpe_delta", "cagr_pct_long", "cagr_pct_both"]
    ].mean().sort_values("sharpe_delta", ascending=False)

    payload = {
        "timeframe_order": TIMEFRAME_ORDER,
        "rows": merged.round(4).to_dict(orient="records"),
        "by_timeframe": by_tf.reset_index().round(4).to_dict(orient="records"),
        "by_asset_class": by_class.reset_index().round(4).to_dict(orient="records"),
        "summary": {
            "n_total": n_total, "n_short_helps": n_helps,
            "pct_short_helps": round(n_helps / n_total, 4) if n_total else None,
            "mean_sharpe_delta": round(float(merged["sharpe_delta"].mean()), 4),
        },
    }

    out_path = os.path.join(OUT_DIR, "headtohead_data.json")
    with open(out_path, "w") as f:
        json.dump(_sanitize(payload), f, default=str)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    print(f"short side helped in {n_helps}/{n_total} combinations "
          f"(mean Sharpe delta {merged['sharpe_delta'].mean():+.3f})")


if __name__ == "__main__":
    main()
