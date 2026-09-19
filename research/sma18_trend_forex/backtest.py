"""
Hypothesis test: long-only 18-SMA trend-continuation in FX.

Rule under test:
    Universe:  FX pairs where trend tends to sustain and relative
               volatility is slightly lower than peer pairs.
    Indicator: 18-period SMA of daily close.
    Entry:     close > SMA18 on two consecutive candles -> buy at the
               close of the second candle.
    Exit:      price touches the SMA18 again (intrabar low <= SMA18,
               or the open itself gapped through it) -> flat.
    Direction: long only, at most one open position per pair.

This script does not assume which pairs satisfy the "sustains / lower
relative vol" description — it measures volatility and SMA18 trend
persistence per pair from data and reports the split, then runs the
identical rule on every pair so the hypothesis-fit subset can be
compared against the rest of the universe.

Usage: python research/sma18_trend_forex/backtest.py
Outputs: research/sma18_trend_forex/outputs/{summary.csv, trades.csv, results.json}
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
from strategy import (SMA_PERIOD, add_sma, characterize, max_drawdown,
                       performance_stats, run_strategy)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

START_DATE = "1990-01-01"  # yfinance clips to each ticker's actual start
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

# Majors + liquid crosses — broad enough to let the ranking step
# discover which pairs actually fit the hypothesis' pair-selection
# criterion, rather than cherry-picking them up front.
PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X",
    "AUDJPY": "AUDJPY=X",
    "EURCHF": "EURCHF=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pair(name: str, ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                      progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"no data for {name} ({ticker})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    df = df[["open", "high", "low", "close"]].dropna()
    df.to_parquet(path)
    return df


# characterize(), run_strategy(), performance_stats(), max_drawdown() now
# live in strategy.py, shared with the multi-asset/multi-timeframe grid
# in run_grid.py. Daily bars == daily "bars", so the generic bar-based
# field names from strategy.py are aliased back to the day-based names
# this script's report_data.json / the published artifact already use.

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_trades = []
    equity_curves = {}
    rows = []

    for name, ticker in PAIRS.items():
        df = add_sma(load_pair(name, ticker))

        char = characterize(df)
        char["avg_up_run_days"] = char.pop("avg_up_run_bars")
        trades, equity = run_strategy(df)
        perf = performance_stats(trades, equity)
        perf["avg_holding_days"] = perf.pop("avg_holding_bars")

        trades = trades.copy()
        trades["pair"] = name
        all_trades.append(trades)
        equity_curves[name] = equity

        row = {"pair": name, "start": df.index[0].date().isoformat(),
               "end": df.index[-1].date().isoformat(), "n_days": len(df)}
        row.update(char)
        row.update(perf)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("pair")

    # Empirical hypothesis-fit split: below-median annualized vol AND
    # above-median SMA18 up-run persistence, both computed across this
    # universe (not assumed a priori).
    vol_med = summary["ann_vol"].median()
    run_med = summary["avg_up_run_days"].median()
    summary["hypothesis_fit"] = (
        (summary["ann_vol"] <= vol_med) & (summary["avg_up_run_days"] >= run_med)
    )

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"))
    trades_df.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)

    fit_cols = ["sharpe", "cagr_pct", "win_rate", "max_drawdown_pct",
                "avg_return_pct", "n_trades", "profit_factor"]
    group_compare = summary.groupby("hypothesis_fit")[fit_cols].mean()

    equity_json = {
        name: {"dates": [d.date().isoformat() for d in s.index[::5]],
               "equity": [round(v, 5) for v in s.values[::5]]}
        for name, s in equity_curves.items()
    }

    results = {
        "params": {"sma_period": SMA_PERIOD, "start": START_DATE, "end": END_DATE,
                    "vol_median_threshold": vol_med, "run_median_threshold": run_med},
        "summary": summary.reset_index().to_dict(orient="records"),
        "group_compare": group_compare.reset_index().to_dict(orient="records"),
        "equity_curves": equity_json,
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print("\n=== Pair characterization (ranked by ann_vol asc) ===")
    print(summary[["ann_vol", "avg_up_run_days", "pct_time_above_sma18",
                    "hypothesis_fit"]].sort_values("ann_vol").round(4))

    print("\n=== Backtest performance per pair ===")
    print(summary[["n_trades", "win_rate", "avg_return_pct", "cagr_pct",
                    "sharpe", "max_drawdown_pct", "profit_factor",
                    "avg_holding_days"]].round(3))

    print("\n=== Hypothesis-fit subset vs rest (group means) ===")
    print(group_compare.round(3))

    return summary, trades_df, results


if __name__ == "__main__":
    main()
