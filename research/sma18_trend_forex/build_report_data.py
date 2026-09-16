"""Builds the compact JSON payload consumed by the report artifact."""

import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")


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


def main():
    with open(os.path.join(OUT_DIR, "results.json")) as f:
        results = json.load(f)

    summary = pd.DataFrame(results["summary"])
    summary["is_jpy"] = summary["pair"].str.contains("JPY")

    corr_cols = ["ann_vol", "avg_up_run_days", "pct_time_above_sma18"]
    corr_sharpe = summary[corr_cols].corrwith(summary["sharpe"]).round(3).to_dict()
    corr_cagr = summary[corr_cols].corrwith(summary["cagr_pct"]).round(3).to_dict()

    def group_stats(mask_col):
        g = summary.groupby(mask_col)[
            ["sharpe", "cagr_pct", "win_rate", "max_drawdown_pct",
             "ann_vol", "avg_up_run_days", "avg_return_pct", "n_trades"]
        ].mean()
        return {str(k): v.round(4).to_dict() for k, v in g.iterrows()}

    payload = {
        "params": results["params"],
        "pairs": summary.sort_values("ann_vol").to_dict(orient="records"),
        "correlations": {"vs_sharpe": corr_sharpe, "vs_cagr": corr_cagr},
        "group_by_fit": group_stats("hypothesis_fit"),
        "group_by_jpy": group_stats("is_jpy"),
        "equity_curves": results["equity_curves"],
        "trade_totals": {
            "total_trades": int(summary["n_trades"].sum()),
            "n_pairs": int(len(summary)),
        },
    }

    out_path = os.path.join(OUT_DIR, "report_data.json")
    with open(out_path, "w") as f:
        json.dump(_sanitize(payload), f, default=str)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
