"""
Shared SMA18 long-trend engine — timeframe- and asset-agnostic.

Entry: close > SMA18 on two consecutive candles -> buy at the close of
       the second candle.
Exit:  flat as soon as price touches the SMA18 again (intrabar low <=
       SMA18, or the open itself gapped through it).
Long only, at most one open position at a time.

Used by both the original daily-FX backtest (backtest.py) and the
multi-asset / multi-timeframe grid (run_grid.py) so the rule is defined
in exactly one place.
"""

import numpy as np
import pandas as pd

SMA_PERIOD = 18


def add_sma(df: pd.DataFrame, period: int = SMA_PERIOD) -> pd.DataFrame:
    df = df.copy()
    df["sma18"] = df["close"].rolling(period).mean()
    return df


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Empirical bar frequency, inferred from the data itself rather than
    assumed — avoids hardcoding trading-hours conventions that differ
    across FX (24/5), crypto (24/7), and exchange-hours equities/commods."""
    span_days = (index[-1] - index[0]).days
    if span_days <= 0:
        return float(len(index))
    return len(index) / (span_days / 365.25)


def characterize(df: pd.DataFrame) -> dict:
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    bpy = bars_per_year(df.index)
    ann_vol = float(log_ret.std() * np.sqrt(bpy))

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
        "bars_per_year": bpy,
    }


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
                    "holding_bars": i - entry_idx,
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
                    entry_idx = i

    if in_position:
        trades.append({
            "entry_date": entry_date, "exit_date": df.index[-1],
            "entry_price": entry_price, "exit_price": close.iloc[-1],
            "return_pct": (close.iloc[-1] / entry_price - 1) * 100,
            "holding_bars": len(df) - 1 - entry_idx,
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
    bpy = bars_per_year(equity.index)
    n_years = len(equity) / bpy if bpy > 0 else np.nan

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

    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(bpy)
              if daily_ret.std() > 0 else 0.0)
    cagr = (equity.iloc[-1] ** (1 / n_years) - 1) if n_years and n_years > 0 else np.nan

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
        "ann_vol_of_equity_pct": float(daily_ret.std() * np.sqrt(bpy) * 100),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_drawdown(equity) * 100),
    }
