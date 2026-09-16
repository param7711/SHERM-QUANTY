"""
Hypothesis test, extended: same long-only 18-SMA trend rule as
research/sma18_trend_forex/backtest.py, run across four candle
timeframes (15m, 13m, 4h, 1d) and across asset classes of differing
volatility (FX, crypto, commodities, indices, mega-cap equities) —
to see on which timeframe x asset-class combinations it actually holds up.

Same rule under test:
    Entry:  close > SMA18 on two consecutive candles -> buy at the
            close of the second candle.
    Exit:   price touches the SMA18 again (intrabar low <= SMA18,
            or the open gapped through it) -> flat.
    Direction: long only, at most one open position per instrument.

Timeframe sourcing (yfinance caps how far back intraday data goes):
    1d   native daily bars, ~10y history where available.
    4h   resampled from native 1h bars (yfinance's 730-day cap on 1h).
    15m  native 15m bars (yfinance's 60-day cap on 15m).
    13m  NOT a native interval anywhere — resampled from native 5m bars
         (60-day cap on 5m). This is the best available proxy for a
         13-minute candle; it is not literal 13m bars built from 1m
         data (1m is capped at 7 days by yfinance, too short to be
         useful here).

Sharpe/CAGR are annualized from the *actual observed bar frequency* in
each series (bars/year measured from the data), not a hardcoded 252 —
necessary because FX/crypto trade ~24/7, equities/indices trade only
market hours, and bar density differs by timeframe.

Usage: python research/sma18_multiframe/backtest.py
Outputs: research/sma18_multiframe/outputs/{summary.csv, report_data.json}
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
DAILY_START = "2016-01-01"

ASSET_UNIVERSE = {
    "FX": {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X", "GBPJPY": "GBPJPY=X", "EURJPY": "EURJPY=X",
    },
    "CRYPTO": {
        "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
        "SOLUSD": "SOL-USD", "DOGEUSD": "DOGE-USD",
    },
    "COMMODITY": {
        "GOLD": "GC=F", "SILVER": "SI=F", "CRUDE": "CL=F",
    },
    "INDEX": {
        "SP500": "^GSPC", "NASDAQ100": "^NDX", "DOWJONES": "^DJI",
    },
    "EQUITY": {
        "AAPL": "AAPL", "MSFT": "MSFT", "JPM": "JPM", "TSLA": "TSLA",
    },
}

# timeframe -> (fetch_interval, fetch_period_or_None, resample_rule_or_None)
TIMEFRAMES = {
    "15m": ("15m", "60d", None),
    "13m": ("5m", "60d", "13min"),
    "4h":  ("1h", "730d", "4h"),
    "1d":  ("1d", None, None),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    })
    return out.dropna(subset=["close"])


def load_series(name: str, ticker: str, tf: str) -> pd.DataFrame:
    interval, period, resample_rule = TIMEFRAMES[tf]
    cache_key = f"{name}_{tf}"
    path = os.path.join(DATA_DIR, f"{cache_key}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    kwargs = {"interval": interval, "progress": False, "auto_adjust": True}
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = DAILY_START

    df = yf.download(ticker, **kwargs)
    if df is None or df.empty:
        raise ValueError(f"no data for {name} ({ticker}, {tf})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close"]].dropna()

    if resample_rule:
        df = _resample_ohlc(df, resample_rule)

    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Pair characterization
# ---------------------------------------------------------------------------

def characterize(df: pd.DataFrame, bars_per_year: float) -> dict:
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    ann_vol = float(log_ret.std() * np.sqrt(bars_per_year)) if bars_per_year > 0 else np.nan

    valid = df["sma18"].notna()
    above = (df["close"] > df["sma18"])[valid]
    run_id = (above != above.shift(1)).cumsum()
    run_lengths = above.groupby(run_id).agg(["sum", "size"])
    up_runs = run_lengths[above.groupby(run_id).first()]["size"]
    avg_up_run = float(up_runs.mean()) if len(up_runs) else 0.0
    pct_time_above = float(above.mean())

    return {
        "ann_vol": ann_vol,
        "avg_up_run_bars": avg_up_run,
        "pct_time_above_sma18": pct_time_above,
        "n_up_runs": int(len(up_runs)),
    }


# ---------------------------------------------------------------------------
# Strategy (identical rule to the FX daily version; timeframe-agnostic)
# ---------------------------------------------------------------------------

def run_strategy(df: pd.DataFrame) -> tuple:
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
                    exit_price = open_.iloc[i]
                elif low.iloc[i] <= sma_now <= high.iloc[i]:
                    exit_price = sma_now

            if exit_price is not None:
                day_ret = exit_price / prev_close - 1
                equity[i] = equity[i - 1] * (1 + day_ret)
                trades.append({
                    "entry_date": entry_date, "exit_date": df.index[i],
                    "entry_price": entry_price, "exit_price": exit_price,
                    "return_pct": (exit_price / entry_price - 1) * 100,
                    "holding_bars": i - entry_index,
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
                    entry_index = i

    if in_position:
        trades.append({
            "entry_date": entry_date, "exit_date": df.index[-1],
            "entry_price": entry_price, "exit_price": close.iloc[-1],
            "return_pct": (close.iloc[-1] / entry_price - 1) * 100,
            "holding_bars": len(df) - 1 - entry_index,
            "open": True,
        })

    equity_curve = pd.Series(equity, index=df.index)
    return pd.DataFrame(trades), equity_curve


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(abs(dd.min()))


def performance_stats(trades: pd.DataFrame, equity: pd.Series, bars_per_year: float) -> dict:
    bar_ret = equity.pct_change().dropna()
    closed = trades[~trades["open"]] if not trades.empty else trades
    span_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    n_years = span_days / 365.25

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

    sharpe = (bar_ret.mean() / bar_ret.std() * np.sqrt(bars_per_year)
              if bar_ret.std() > 0 and bars_per_year > 0 else 0.0)
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
        "avg_holding_bars": float(closed["holding_bars"].mean()) if not closed.empty else np.nan,
        "total_return_pct": float((equity.iloc[-1] - 1) * 100),
        "cagr_pct": float(cagr * 100) if pd.notna(cagr) else np.nan,
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_drawdown(equity) * 100),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    equity_samples = {}  # (tf, name) -> downsampled equity for the artifact

    for category, assets in ASSET_UNIVERSE.items():
        for name, ticker in assets.items():
            for tf in TIMEFRAMES:
                try:
                    df = load_series(name, ticker, tf)
                except Exception as e:
                    print(f"  [SKIP] {name} {tf}: {e}")
                    continue
                if len(df) < SMA_PERIOD * 5:
                    print(f"  [SKIP] {name} {tf}: only {len(df)} bars, too short")
                    continue

                df = df.copy()
                df["sma18"] = df["close"].rolling(SMA_PERIOD).mean()
                span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
                bars_per_year = (len(df) - 1) / span_days * 365.25 if span_days > 0 else np.nan

                char = characterize(df, bars_per_year)
                trades, equity = run_strategy(df)
                perf = performance_stats(trades, equity, bars_per_year)

                row = {"category": category, "asset": name, "ticker": ticker,
                       "timeframe": tf, "n_bars": len(df),
                       "start": df.index[0].isoformat(), "end": df.index[-1].isoformat(),
                       "bars_per_year_est": bars_per_year}
                row.update(char)
                row.update(perf)
                rows.append(row)

                step = max(1, len(equity) // 400)
                equity_samples[f"{tf}|{name}"] = {
                    "dates": [d.isoformat() for d in equity.index[::step]],
                    "equity": [round(v, 5) for v in equity.values[::step]],
                }

                print(f"  {category:10s} {name:10s} {tf:4s}  "
                      f"n={len(df):6d}  trades={perf['n_trades']:4d}  "
                      f"sharpe={perf['sharpe']:+.2f}  cagr={perf['cagr_pct']:+7.1f}%  "
                      f"maxdd={perf['max_drawdown_pct']:5.1f}%")

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    # Empirical vol split, computed WITHIN each timeframe (vol scale differs
    # hugely between 15m and 1d bars, so cross-timeframe comparison would be
    # meaningless — the split has to be per-timeframe).
    summary["vol_bucket"] = "low"
    for tf in summary["timeframe"].unique():
        mask = summary["timeframe"] == tf
        med = summary.loc[mask, "ann_vol"].median()
        summary.loc[mask & (summary["ann_vol"] > med), "vol_bucket"] = "high"

    by_tf_bucket = (summary.groupby(["timeframe", "vol_bucket"])
                     [["sharpe", "cagr_pct", "win_rate", "max_drawdown_pct",
                       "profit_factor", "n_trades"]].mean().round(4))
    by_tf_category = (summary.groupby(["timeframe", "category"])
                       [["sharpe", "cagr_pct", "win_rate", "max_drawdown_pct",
                         "profit_factor", "n_trades"]].mean().round(4))

    print("\n=== Mean performance by timeframe x volatility bucket ===")
    print(by_tf_bucket)
    print("\n=== Mean performance by timeframe x asset class ===")
    print(by_tf_category)

    payload = {
        "params": {"sma_period": SMA_PERIOD, "timeframes": list(TIMEFRAMES.keys())},
        "rows": json.loads(summary.to_json(orient="records")),
        "by_tf_bucket": json.loads(
            by_tf_bucket.reset_index().to_json(orient="records")),
        "by_tf_category": json.loads(
            by_tf_category.reset_index().to_json(orient="records")),
        "equity_samples": equity_samples,
    }
    with open(os.path.join(OUT_DIR, "report_data.json"), "w") as f:
        json.dump(payload, f, default=str)

    print(f"\nWrote {os.path.join(OUT_DIR, 'report_data.json')}")
    return summary


if __name__ == "__main__":
    main()
