"""
Asset-class x timeframe grid for the SMA18 long-trend rule.

Same rule as backtest.py (see strategy.py), run across every
(asset, timeframe) combination in universe.py, so the question
"does this work better on volatile or non-volatile assets, and on
which candle size" has one consistent, comparable answer instead of
four separate one-off scripts.

Timeframe data-availability constraints (Yahoo/yfinance, not a choice
made here): 15m and 30m bars are only available for the trailing 60
days; 4h bars are built by resampling 1h bars, capped at the trailing
730 days; daily bars go back to START_DATE. The 15m/30m results are
therefore a much smaller sample than 4h/1d — flagged in the output,
not hidden.

Usage: python research/sma18_trend_forex/run_grid.py [long|short|both]
  (default: long — the original long-only rule; 'both' is the long+short
  reversal variant, entering short on two closes below the SMA18 the
  same way it enters long on two closes above it)
Outputs: research/sma18_trend_forex/outputs/{grid_summary[_SUFFIX].csv,
         grid_results[_SUFFIX].json, mc_paths[_SUFFIX].json}
  where SUFFIX is empty for 'long' (keeps the existing Part 2 pipeline's
  filenames stable) and the direction name otherwise.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monte_carlo import bootstrap_trade_mc
from strategy import add_sma, characterize, performance_stats, run_strategy
from universe import TIMEFRAMES, UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

DAILY_START = "1990-01-01"  # yfinance clips to each ticker's actual start
DAILY_END = pd.Timestamp.today().strftime("%Y-%m-%d")

# Minimum bars required for the SMA18 rule + a usable sample; below this
# a combination is skipped rather than reported on noise.
MIN_BARS = 60


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df.index = pd.to_datetime(df.index)
    return df[["open", "high", "low", "close"]].dropna()


def load_bars(name: str, ticker: str, timeframe: str) -> pd.DataFrame:
    tf_dir = os.path.join(DATA_DIR, timeframe)
    os.makedirs(tf_dir, exist_ok=True)
    path = os.path.join(tf_dir, f"{name}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    if timeframe == "1d":
        raw = yf.download(ticker, start=DAILY_START, end=DAILY_END,
                           interval="1d", progress=False, auto_adjust=True)
        df = _flatten_yf(raw) if raw is not None and not raw.empty else pd.DataFrame()

    elif timeframe == "4h":
        raw = yf.download(ticker, period="730d", interval="1h",
                           progress=False, auto_adjust=True)
        h1 = _flatten_yf(raw) if raw is not None and not raw.empty else pd.DataFrame()
        if h1.empty:
            df = h1
        else:
            df = h1.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()

    elif timeframe in ("15m", "30m"):
        raw = yf.download(ticker, period="60d", interval=timeframe,
                           progress=False, auto_adjust=True)
        df = _flatten_yf(raw) if raw is not None and not raw.empty else pd.DataFrame()

    else:
        raise ValueError(f"unsupported timeframe {timeframe}")

    if not df.empty:
        df.to_parquet(path)
    return df


def run_one(name: str, meta: dict, timeframe: str, direction: str = "long") -> tuple:
    """Returns (row_dict, mc_detail_or_None)."""
    try:
        df = load_bars(name, meta["ticker"], timeframe)
    except Exception as e:
        return {"error": f"download failed: {e}"}, None

    if df.empty or len(df) < MIN_BARS:
        return {"error": f"insufficient bars ({len(df)} < {MIN_BARS})"}, None

    df = add_sma(df)
    char = characterize(df)
    trades, equity = run_strategy(df, direction=direction)
    perf = performance_stats(trades, equity)

    row = {
        "asset": name, "asset_class": meta["asset_class"],
        "expected_vol_tier": meta["vol_tier"], "timeframe": timeframe,
        "direction": direction, "n_bars": len(df),
        "start": df.index[0].isoformat(), "end": df.index[-1].isoformat(),
    }
    row.update(char)
    row.update(perf)
    if direction == "both" and not trades.empty:
        row["n_long_trades"] = int((trades["side"] == "long").sum())
        row["n_short_trades"] = int((trades["side"] == "short").sum())
    row["error"] = None

    mc_detail = None
    closed = trades[~trades["open"]] if not trades.empty else trades
    if not closed.empty:
        mc = bootstrap_trade_mc(closed["return_pct"].values)
        if mc is not None:
            row["mc_p_profit"] = mc["p_profit"]
            row["mc_return_p05"] = mc["final_return_p05"]
            row["mc_return_p50"] = mc["final_return_p50"]
            row["mc_return_p95"] = mc["final_return_p95"]
            row["mc_max_dd_p50"] = mc["max_dd_p50"]
            row["mc_max_dd_p95"] = mc["max_dd_p95"]
            mc_detail = mc
    return row, mc_detail


def main(direction: str = "long"):
    suffix = "" if direction == "long" else f"_{direction}"
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    mc_paths = {}
    for name, meta in UNIVERSE.items():
        for tf in TIMEFRAMES:
            print(f"  {name:10s} {tf:4s} ...", end=" ", flush=True)
            r, mc = run_one(name, meta, tf, direction=direction)
            if r.get("error"):
                print("SKIP:", r["error"])
                rows.append({"asset": name, "asset_class": meta["asset_class"],
                              "expected_vol_tier": meta["vol_tier"], "timeframe": tf,
                              "error": r["error"]})
            else:
                mc_note = f"  MC p_profit={r['mc_p_profit']:.2f}" if "mc_p_profit" in r else ""
                print(f"n_trades={r['n_trades']:>4d}  sharpe={r['sharpe']:+.2f}{mc_note}")
                rows.append(r)
                if mc is not None:
                    mc_paths[f"{name}|{tf}"] = mc

    grid = pd.DataFrame(rows)
    ok = grid[grid["error"].isna()].copy()

    # Empirical volatility split WITHIN each timeframe (vol scale differs a
    # lot by candle size, so a single global median would just separate
    # timeframes rather than assets).
    ok["vol_tier_empirical"] = "low"
    for tf in ok["timeframe"].unique():
        mask = ok["timeframe"] == tf
        med = ok.loc[mask, "ann_vol"].median()
        ok.loc[mask, "vol_tier_empirical"] = np.where(
            ok.loc[mask, "ann_vol"] >= med, "high", "low")

    grid = grid.merge(
        ok[["asset", "timeframe", "vol_tier_empirical"]],
        on=["asset", "timeframe"], how="left"
    )

    grid.to_csv(os.path.join(OUT_DIR, f"grid_summary{suffix}.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"mc_paths{suffix}.json"), "w") as f:
        json.dump(mc_paths, f, default=str)

    by_tf = ok.groupby("timeframe")[["sharpe", "cagr_pct", "win_rate",
                                      "max_drawdown_pct", "n_trades"]].mean()
    by_class = ok.groupby("asset_class")[["sharpe", "cagr_pct", "win_rate",
                                           "max_drawdown_pct", "n_trades"]].mean()
    by_voltier = ok.groupby(["timeframe", "vol_tier_empirical"])[
        ["sharpe", "cagr_pct", "win_rate", "n_trades", "ann_vol"]].mean()

    results = {
        "grid": grid.to_dict(orient="records"),
        "by_timeframe": by_tf.reset_index().to_dict(orient="records"),
        "by_asset_class": by_class.reset_index().to_dict(orient="records"),
        "by_vol_tier": by_voltier.reset_index().to_dict(orient="records"),
        "n_combinations": len(grid), "n_successful": len(ok), "n_skipped": len(grid) - len(ok),
    }
    with open(os.path.join(OUT_DIR, f"grid_results{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== direction={direction}: {len(ok)}/{len(grid)} combinations produced a result ===\n")

    print("=== Mean performance by timeframe ===")
    print(by_tf.round(3))

    print("\n=== Mean performance by asset class ===")
    print(by_class.round(3))

    print("\n=== Mean performance by (timeframe, empirical vol tier) ===")
    print(by_voltier.round(3))

    return grid, results


if __name__ == "__main__":
    _direction = sys.argv[1] if len(sys.argv) > 1 else "long"
    main(_direction)
