"""Consolidates every luck-test into one payload for the report artifact."""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")


def _sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _load(fname):
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    markov = _load("markov_validation.json")
    era_long = _load("era_splits.json")
    era_both = _load("era_splits_both.json")
    extras = _load("robustness_extras.json")

    grid = pd.read_csv(os.path.join(OUT_DIR, "grid_summary.csv"))
    grid = grid[grid["error"].isna()]

    # Engine test output, captured live so the report can't claim a green
    # suite that wasn't actually run.
    try:
        proc = subprocess.run([sys.executable, os.path.join(HERE, "test_engine.py")],
                               capture_output=True, text=True, timeout=600)
        test_lines = [l for l in proc.stdout.splitlines() if l.startswith(("PASS", "SKIP", "FAIL"))]
        tests_passed = proc.returncode == 0
    except Exception as e:
        test_lines, tests_passed = [f"FAIL  could not run tests: {e}"], False

    mk = pd.DataFrame(markov["flat"]) if markov else pd.DataFrame()
    if not mk.empty:
        mk["verdict"] = np.where(
            (mk["markov_p"] < 0.05) & (mk["shuffle_p"] < 0.05), "beats both nulls",
            np.where(mk["shuffle_p"] < 0.05, "beats random order only",
                     np.where(mk["markov_p"] < 0.05, "beats markov only", "indistinguishable from luck")))
        by_class = mk.groupby("asset_class").agg(
            n=("asset", "count"),
            mean_real_sharpe=("real_sharpe", "mean"),
            mean_markov_synth=("markov_synth_mean", "mean"),
            mean_shuffle_synth=("shuffle_synth_mean", "mean"),
            n_markov_sig=("markov_p", lambda s: int((s < 0.05).sum())),
            n_shuffle_sig=("shuffle_p", lambda s: int((s < 0.05).sum())),
        ).reset_index()
    else:
        by_class = pd.DataFrame()

    payload = {
        "engine_tests": {"passed": tests_passed, "lines": test_lines},
        "markov": {
            "summary": markov["summary"] if markov else None,
            "n_sims": markov["n_sims"] if markov else None,
            "n_states": markov["n_states"] if markov else None,
            "flat": mk.round(4).to_dict(orient="records") if not mk.empty else [],
            "by_class": by_class.round(4).to_dict(orient="records") if not by_class.empty else [],
            "distributions": [
                {
                    "asset": a["asset"], "asset_class": a["asset_class"],
                    "real_sharpe": a["real_sharpe"],
                    "markov_samples": a["markov"]["sharpe_samples"],
                    "shuffle_samples": a["shuffle"]["sharpe_samples"],
                    "markov_p": a["markov"]["p_value"], "shuffle_p": a["shuffle"]["p_value"],
                }
                for a in (markov["assets"] if markov else [])
                if a["asset"] in ("BTCUSD", "GOLD", "ETHUSD", "CRUDE", "SILVER", "NATGAS")
            ],
        },
        "eras": {
            "long": {k: era_long[k] for k in ("eras", "rows", "by_era", "by_class_era", "per_asset", "summary")} if era_long else None,
            "both": {k: era_both[k] for k in ("by_era", "summary")} if era_both else None,
        },
        "extras": {
            "periods": extras["periods"] if extras else None,
            "cost_bps": extras["cost_bps"] if extras else None,
            "sweep_by_class": extras["sweep_by_class"] if extras else None,
            "cost_by_class": extras["cost_by_class"] if extras else None,
            "per_asset_costs": [
                {"asset": r["asset"], "asset_class": r["asset_class"],
                 "n_trades": r["n_trades"], "breakeven_bps": r["breakeven_bps"],
                 "sharpe_0": r["by_cost"]["0"]["sharpe"], "sharpe_10": r["by_cost"]["10"]["sharpe"],
                 "sharpe_20": r["by_cost"]["20"]["sharpe"]}
                for r in (extras["per_asset_costs"] if extras else [])
            ],
            "per_asset_periods": [
                {"asset": r["asset"], "asset_class": r["asset_class"],
                 "sharpe_18": r["sharpe_18"], "rank_of_18": r["rank_of_18"],
                 "n_periods": r["n_periods"], "best_period": r["best_period"],
                 "pct_periods_positive": r["pct_periods_positive"]}
                for r in (extras["per_asset_periods"] if extras else [])
            ],
            "summary": extras["summary"] if extras else None,
        },
        "universe": {
            "n_assets": int(grid["asset"].nunique()),
            "n_commodity": int(grid[grid.asset_class == "COMMODITY"]["asset"].nunique()),
            "n_crypto": int(grid[grid.asset_class == "CRYPTO"]["asset"].nunique()),
            "daily_span": {
                r["asset"]: {"start": r["start"][:10], "end": r["end"][:10], "n_bars": int(r["n_bars"])}
                for _, r in grid[grid.timeframe == "1d"].iterrows()
            },
        },
    }

    out = os.path.join(OUT_DIR, "validation_report_data.json")
    with open(out, "w") as f:
        json.dump(_sanitize(payload), f, default=str)
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    if markov:
        s = markov["summary"]
        print(f"markov: {s['n_markov_sig_05']}/{s['n_assets']} beat the Markov null at p<0.05 "
              f"({s['n_markov_sig_bh']} after BH); shuffle: {s['n_shuffle_sig_05']}/{s['n_assets']}")
    print(f"engine tests passed: {tests_passed}")


if __name__ == "__main__":
    main()
