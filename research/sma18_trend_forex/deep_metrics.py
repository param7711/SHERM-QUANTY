"""
The standard quant risk-metric toolkit, computed for the 6 crypto (deep,
long-only, 15m/30m/1h/4h) and 6 commodity (daily, long-only) instruments.

Per user request: long-only only (long+short dropped from this page), and
"every metric a quant uses" rather than just Sharpe. All computed from the
same trade-return and bar-equity arrays the rest of this study uses, no
new backtests beyond what deep_full_battery.py and the daily study already
ran -- this is a pure post-processing pass over cached results/data.

Metrics, and why each one is here:
  SHARPE       return per unit of total volatility. The default, and the
               weakest, because it penalizes upside swings same as downside.
  SORTINO      return per unit of DOWNSIDE volatility only -- the fix for
               Sharpe's blind spot on trend-following, which by design has
               occasional huge up-moves that Sharpe punishes as "risk."
  CALMAR       CAGR / max drawdown. What return you got per unit of the
               worst peak-to-trough pain you had to sit through.
  MAX DRAWDOWN worst peak-to-trough decline in the equity curve. The number
               that determines whether a real person could actually hold
               this through the bad stretch.
  ULCER INDEX  root-mean-square of the drawdown series, not just its worst
               point -- captures how long and how often the curve is
               underwater, not only how deep it got once.
  RECOVERY     total return / max drawdown. Similar to Calmar but on total
    FACTOR     return rather than annualized, so it doesn't reward a short
               history with a lucky recent run the way CAGR can.
  PROFIT       gross profit / gross loss across all trades. Below 1 means
    FACTOR     the strategy loses money even before you consider its Sharpe.
  WIN RATE     % of trades that were profitable. Expected to be LOW for
               this rule by construction (see the note below the table).
  EXPECTANCY   mean return per trade. The number that, multiplied by trade
               frequency, is where all the profit actually comes from.
  SKEW         asymmetry of the trade-return distribution. Positive skew
               (rare big winners, frequent small losers) is the classic
               trend-following signature and is a GOOD sign here, unlike
               in most other contexts where positive skew is unremarkable.
  KURTOSIS     (excess) how fat the tails are versus a normal distribution.
               High kurtosis means the average/std understate how extreme
               the rare trades can be -- a caveat on every other metric here.
  CVAR 95      mean return of the worst 5% of trades (expected shortfall).
               The answer to "how bad is bad," which max/worst trade alone
               doesn't average out.

Run: python deep_metrics.py
Outputs: outputs/deep_metrics.json
"""

import json, math, os, sys, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import SMA_PERIOD, bars_per_year, run_strategy_arrays, sharpe_from_equity
from markov_validation import _rolling_mean
from deep_intraday_validation import _load, SELECTED_CRYPTO

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
DATA_1D = os.path.join(HERE, "data", "1d")
SELECTED_COMMODITY = ["CRUDE", "GOLD", "SUGAR", "HEATOIL", "SOYBEAN", "GASOLINE"]

TF_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def _sanitize(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [_sanitize(v) for v in o]
    return o


def compute_metrics(rets, holds, eq, bpy, tf):
    n = len(eq)
    n_years = n / bpy if bpy > 0 else np.nan
    total_return_pct = float((eq[-1] - 1) * 100)
    cagr_pct = float((eq[-1] ** (1 / n_years) - 1) * 100) if n_years and n_years > 0 and eq[-1] > 0 else float("nan")

    bar_ret = np.diff(eq) / eq[:-1]
    sharpe = sharpe_from_equity(eq, bpy)
    downside = bar_ret[bar_ret < 0]
    sortino = float(bar_ret.mean() / downside.std() * np.sqrt(bpy)) if len(downside) > 1 and downside.std() > 0 else float("nan")

    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    max_dd_pct = float(-dd.min() * 100)
    ulcer_index = float(np.sqrt(np.mean(dd ** 2)) * 100)
    calmar = float(cagr_pct / max_dd_pct) if max_dd_pct > 0 else float("nan")
    recovery_factor = float(total_return_pct / max_dd_pct) if max_dd_pct > 0 else float("nan")

    if len(rets) == 0:
        return None
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    win_rate = float((rets > 0).mean())
    expectancy = float(rets.mean())
    mu, sd = rets.mean(), rets.std()
    skew = float(np.mean((rets - mu) ** 3) / sd ** 3) if sd > 0 else float("nan")
    kurt = float(np.mean((rets - mu) ** 4) / sd ** 4 - 3) if sd > 0 else float("nan")
    cvar95 = float(np.mean(np.sort(rets)[:max(1, int(np.ceil(len(rets) * 0.05)))]))
    avg_holding_bars = float(np.mean(holds)) if len(holds) else float("nan")
    avg_holding_hours = avg_holding_bars * TF_MINUTES[tf] / 60.0

    return {
        "n_trades": int(len(rets)), "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr_pct, 2), "sharpe": round(float(sharpe), 4),
        "sortino": round(sortino, 4) if not math.isnan(sortino) else None,
        "calmar": round(calmar, 4) if not math.isnan(calmar) else None,
        "max_drawdown_pct": round(max_dd_pct, 2), "ulcer_index": round(ulcer_index, 3),
        "recovery_factor": round(recovery_factor, 4) if not math.isnan(recovery_factor) else None,
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else None,
        "win_rate": round(win_rate, 4), "expectancy_pct": round(expectancy, 4),
        "skew": round(skew, 3) if not math.isnan(skew) else None,
        "kurtosis": round(kurt, 3) if not math.isnan(kurt) else None,
        "cvar95_pct": round(cvar95, 4), "best_trade_pct": round(float(rets.max()), 3),
        "worst_trade_pct": round(float(rets.min()), 3),
        "avg_holding_bars": round(avg_holding_bars, 2), "avg_holding_hours": round(avg_holding_hours, 2),
    }


def main():
    crypto_rows = []
    for tf in ("15m", "30m", "1h", "4h"):
        for name in SELECTED_CRYPTO:
            df = _load(tf, name)
            if df is None or len(df) < 500:
                continue
            o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
            sma = _rolling_mean(c, SMA_PERIOD)
            bpy = bars_per_year(df.index)
            rets, holds, sides, eq = run_strategy_arrays(o, h, l, c, sma, direction="long")
            m = compute_metrics(rets, holds, eq, bpy, tf)
            if m:
                crypto_rows.append({"asset": name, "timeframe": tf, **m})
                print(f"  {tf} {name:10s} Sharpe {m['sharpe']:+.2f}  Sortino {m['sortino']}  "
                      f"Calmar {m['calmar']}  MaxDD {m['max_drawdown_pct']:.1f}%  "
                      f"PF {m['profit_factor']}  skew {m['skew']}", flush=True)

    commodity_rows = []
    for name in SELECTED_COMMODITY:
        path = os.path.join(DATA_1D, f"{name}.parquet")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
        sma = _rolling_mean(c, SMA_PERIOD)
        bpy = bars_per_year(df.index)
        rets, holds, sides, eq = run_strategy_arrays(o, h, l, c, sma, direction="long")
        m = compute_metrics(rets, holds, eq, bpy, "1d")
        if m:
            commodity_rows.append({"asset": name, "timeframe": "1d", **m})
            print(f"  1d  {name:10s} Sharpe {m['sharpe']:+.2f}  Sortino {m['sortino']}  "
                  f"Calmar {m['calmar']}  MaxDD {m['max_drawdown_pct']:.1f}%  "
                  f"PF {m['profit_factor']}  skew {m['skew']}", flush=True)

    payload = {"crypto": crypto_rows, "commodity": commodity_rows}
    with open(os.path.join(OUT_DIR, "deep_metrics.json"), "w") as f:
        json.dump(_sanitize(payload), f, default=str)
    print(f"\nwrote {os.path.join(OUT_DIR, 'deep_metrics.json')}")


if __name__ == "__main__":
    main()
