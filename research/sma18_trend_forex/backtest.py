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
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")

SMA_PERIOD = 18
START_DATE = "2015-01-01"
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


# ---------------------------------------------------------------------------
# Pair characterization — volatility & trend persistence
# ---------------------------------------------------------------------------

def characterize(df: pd.DataFrame) -> dict:
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    ann_vol = float(log_ret.std() * np.sqrt(252))

    valid = df["sma18"].notna()
    above = (df["close"] > df["sma18"])[valid]
    # Run-length of consecutive days spent above the SMA (the state that
    # defines "trend sustains" for a long-only SMA-touch system).
    run_id = (above != above.shift(1)).cumsum()
    run_lengths = above.groupby(run_id).agg(["sum", "size"])
    up_runs = run_lengths[above.groupby(run_id).first()]["size"]
    avg_up_run = float(up_runs.mean()) if len(up_runs) else 0.0
    pct_time_above = float(above.mean())

    return {
        "ann_vol": ann_vol,
        "avg_up_run_days": avg_up_run,
        "pct_time_above_sma18": pct_time_above,
        "n_up_runs": int(len(up_runs)),
    }


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def run_strategy(df: pd.DataFrame) -> tuple:
    """Returns (trades_df, equity_curve[pd.Series indexed by date, base=1.0])."""
    close, open_, low, high, sma = (df["close"], df["open"], df["low"],
                                     df["high"], df["sma18"])

    trades = []
    equity = np.empty(len(df))
    equity[0] = 1.0

    in_position = False
    entry_price = entry_date = None

    for i in range(1, len(df)):
        prev_close = close.iloc[i - 1]

        if in_position:
            sma_now = sma.iloc[i]
            exit_price = None
            if pd.notna(sma_now):
                if open_.iloc[i] <= sma_now:
                    exit_price = open_.iloc[i]          # gapped through at the open
                elif low.iloc[i] <= sma_now <= high.iloc[i]:
                    exit_price = sma_now                  # intrabar touch

            if exit_price is not None:
                day_ret = exit_price / prev_close - 1
                equity[i] = equity[i - 1] * (1 + day_ret)
                trades.append({
                    "entry_date": entry_date, "exit_date": df.index[i],
                    "entry_price": entry_price, "exit_price": exit_price,
                    "return_pct": (exit_price / entry_price - 1) * 100,
                    "holding_days": (df.index[i] - entry_date).days,
                    "open": False,
                })
                in_position = False
            else:
                day_ret = close.iloc[i] / prev_close - 1
                equity[i] = equity[i - 1] * (1 + day_ret)
        else:
            equity[i] = equity[i - 1]
            sma_now, sma_prev = sma.iloc[i], sma.iloc[i - 1]
            if pd.notna(sma_now) and pd.notna(sma_prev):
                if close.iloc[i] > sma_now and close.iloc[i - 1] > sma_prev:
                    in_position = True
                    entry_price = close.iloc[i]
                    entry_date = df.index[i]

    if in_position:
        trades.append({
            "entry_date": entry_date, "exit_date": df.index[-1],
            "entry_price": entry_price, "exit_price": close.iloc[-1],
            "return_pct": (close.iloc[-1] / entry_price - 1) * 100,
            "holding_days": (df.index[-1] - entry_date).days,
            "open": True,
        })

    equity_curve = pd.Series(equity, index=df.index)
    return pd.DataFrame(trades), equity_curve


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(abs(dd.min()))


def performance_stats(trades: pd.DataFrame, equity: pd.Series) -> dict:
    daily_ret = equity.pct_change().dropna()
    closed = trades[~trades["open"]] if not trades.empty else trades
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25

    if closed.empty:
        win_rate = avg_ret = median_ret = profit_factor = np.nan
    else:
        wins = closed[closed["return_pct"] > 0]["return_pct"]
        losses = closed[closed["return_pct"] <= 0]["return_pct"]
        win_rate = len(wins) / len(closed)
        avg_ret = float(closed["return_pct"].mean())
        median_ret = float(closed["return_pct"].median())
        profit_factor = (wins.sum() / abs(losses.sum())
                          if losses.sum() != 0 else np.inf)

    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)
    cagr = (equity.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan

    return {
        "n_trades": int(len(closed)),
        "n_open_at_end": int(len(trades) - len(closed)),
        "win_rate": win_rate,
        "avg_return_pct": avg_ret,
        "median_return_pct": median_ret,
        "profit_factor": profit_factor,
        "worst_trade_pct": float(closed["return_pct"].min()) if not closed.empty else np.nan,
        "best_trade_pct": float(closed["return_pct"].max()) if not closed.empty else np.nan,
        "avg_holding_days": float(closed["holding_days"].mean()) if not closed.empty else np.nan,
        "total_return_pct": float((equity.iloc[-1] - 1) * 100),
        "cagr_pct": float(cagr * 100) if pd.notna(cagr) else np.nan,
        "ann_vol_of_equity_pct": float(daily_ret.std() * np.sqrt(252) * 100),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_drawdown(equity) * 100),
    }


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
        df = load_pair(name, ticker)
        df["sma18"] = df["close"].rolling(SMA_PERIOD).mean()

        char = characterize(df)
        trades, equity = run_strategy(df)
        perf = performance_stats(trades, equity)

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
