"""
Re-runs the full validation stack on crypto 15m/30m/1h/4h using the
multi-year Coinbase Exchange history (fetch_coinbase_intraday.py) instead
of yfinance's ~60-day (15m/30m) or ~730-day (1h/4h) samples -- the direct
answer to "the grid shows short timeframes as the best, but that's from a
small sample; what happens with a real one?"

This mirrors markov_validation.py + era_splits.py, but scoped to crypto
only (commodities have no free multi-year intraday source) and to the four
intraday timeframes. 15m/30m/1h are fetched directly; 4h is DERIVED by
resampling the deep 1h data (open=first, high=max, low=min, close=last),
the same construction run_grid.py uses to build 4h from yfinance's 1h feed
-- there is no native 4h candle on the exchange, and there does not need
to be one.

Outputs: outputs/deep_intraday_validation.json
Run: python deep_intraday_validation.py [n_sims]
"""

import json, math, os, sys, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import SMA_PERIOD, bars_per_year, run_strategy_arrays, sharpe_from_equity
from markov_validation import (_rolling_mean, fit_markov, synth_ohlc_arrays,
                                markov_paths, N_STATES)
from universe import UNIVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
N_ERAS = 3          # split each instrument's history into 3 non-overlapping chunks
MIN_TRADES_ERA = 15

# Scoped to 6 crypto instruments per user request (was all 14) -- matches
# fetch_coinbase_intraday.py's SELECTED_CRYPTO exactly, and is the only
# universe with a real multi-year intraday feed at all.
SELECTED_CRYPTO = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD"]


def _sanitize(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    return o


def _resample(df, rule):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])


def _load(tf, name):
    """Deep sample. 15m and 1h are the only granularities Coinbase Exchange
    actually serves (confirmed directly: granularity=1800 for 30m returns
    HTTP 400 "Unsupported granularity" -- its supported set is exactly
    1m/5m/15m/1h/6h/1d). 30m is derived by resampling the deep 15m data;
    4h is derived by resampling the deep 1h data. Both inherit their
    source's full multi-year span rather than needing a separately-capped
    fetch -- there's no native candle to fetch even if there were time."""
    if tf == "30m":
        base = _load("15m", name)
        return _resample(base, "30min") if base is not None and len(base) >= 4 else None
    if tf == "4h":
        base = _load("1h", name)
        return _resample(base, "4h") if base is not None and len(base) >= 8 else None
    p = os.path.join(HERE, "data", f"{tf}_cb", f"{name}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def _load_shallow(tf, name):
    """The original yfinance sample for that timeframe (60-day cap on
    15m/30m, ~730-day cap on 1h/4h), for the side-by-side comparison."""
    p = os.path.join(HERE, "data", tf, f"{name}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def backtest(df):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    sma = _rolling_mean(c, SMA_PERIOD)
    bpy = bars_per_year(df.index)
    rets, holds, sides, eq = run_strategy_arrays(o, h, l, c, sma, direction="long")
    return {
        "n_bars": len(df), "n_trades": int(len(rets)),
        "sharpe": round(float(sharpe_from_equity(eq, bpy)), 4),
        "cagr_pct": round(float((eq[-1] ** (bpy / len(eq)) - 1) * 100), 2) if len(eq) else None,
        "total_return_pct": round(float((eq[-1] - 1) * 100), 2) if len(eq) else None,
        "win_rate": round(float((rets > 0).mean()), 4) if len(rets) else None,
        "bars_per_year": round(float(bpy), 1),
    }, (o, h, l, c, sma, bpy, rets, eq)


def era_split(df, n_eras=N_ERAS):
    n = len(df)
    edges = np.linspace(0, n, n_eras + 1).astype(int)
    out = []
    for i in range(n_eras):
        chunk = df.iloc[edges[i]:edges[i + 1]]
        if len(chunk) < 200:
            continue
        stats, _ = backtest(chunk)
        if stats["n_trades"] < MIN_TRADES_ERA:
            continue
        out.append({
            "era": i + 1, "start": str(chunk.index[0].date()), "end": str(chunk.index[-1].date()),
            **stats,
        })
    return out


def markov_test(df, n_sims, rng):
    """Same construction as markov_validation.py's validate_asset(): fit the
    chain on log returns, resample bar shapes conditional on the bar's own
    return state (see synth_ohlc_arrays' docstring for why that matters),
    backtest the rule on each synthetic market, and compare to the real
    Sharpe. Reuses that module's exact functions rather than reimplementing
    them, so this p-value is measured the same way as every other one in
    the study."""
    close = df["close"].values
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, close
    sma = _rolling_mean(c, SMA_PERIOD)
    bpy = bars_per_year(df.index)
    rets, _, _, eq = run_strategy_arrays(o, h, l, c, sma, direction="long")
    if len(rets) < 15:
        return None
    real_sharpe = sharpe_from_equity(eq, bpy)

    log_ret = np.diff(np.log(close))
    log_ret = log_ret[np.isfinite(log_ret)]
    trans, pools, states, edges = fit_markov(log_ret)

    shape_all = np.column_stack([
        (df["open"] / df["close"]).values[1:],
        (df["high"] / df["close"]).values[1:],
        (df["low"] / df["close"]).values[1:],
    ])[:len(states)]
    finite = np.isfinite(shape_all).all(axis=1)
    shape_pools = []
    for s in range(N_STATES):
        pool = shape_all[finite & (states[:len(shape_all)] == s)]
        shape_pools.append(pool if len(pool) else shape_all[finite])

    n_steps = len(log_ret)
    paths = markov_paths(trans, pools, states, n_steps, n_sims, rng)
    sharpes = np.empty(n_sims)
    for i in range(n_sims):
        so, sh, sl, sc = synth_ohlc_arrays(paths[i], close[0], shape_pools, edges, rng)
        s_sma = _rolling_mean(sc, SMA_PERIOD)
        s_rets, _, _, s_eq = run_strategy_arrays(so, sh, sl, sc, s_sma, direction="long")
        sharpes[i] = sharpe_from_equity(s_eq, bpy)

    p_value = float((sharpes >= real_sharpe).mean())
    return {
        "real_sharpe": round(float(real_sharpe), 4), "n_trades": int(len(rets)),
        "n_bars": int(len(df)), "markov_p": round(p_value, 4),
        "synth_mean": round(float(sharpes.mean()), 4),
        "synth_samples": [round(float(x), 4) for x in sharpes[:400]],
    }


def main(n_sims=300):
    rng = np.random.default_rng(20260920)
    rows, eras, markov = [], [], []
    for tf in ("15m", "30m", "1h", "4h"):
        for name in SELECTED_CRYPTO:
            deep = _load(tf, name)
            if deep is None or len(deep) < 500:
                continue
            shallow = _load_shallow(tf, name)
            deep_stats, _ = backtest(deep)
            shallow_stats, _ = backtest(shallow) if shallow is not None and len(shallow) >= 200 else (None, None)

            rows.append({
                "asset": name, "timeframe": tf,
                "deep_n_bars": deep_stats["n_bars"], "deep_days": int((deep.index[-1] - deep.index[0]).days),
                "deep_sharpe": deep_stats["sharpe"], "deep_n_trades": deep_stats["n_trades"],
                "deep_cagr_pct": deep_stats["cagr_pct"], "deep_win_rate": deep_stats["win_rate"],
                "shallow_sharpe": shallow_stats["sharpe"] if shallow_stats else None,
                "shallow_n_trades": shallow_stats["n_trades"] if shallow_stats else None,
                "shallow_days": int((shallow.index[-1] - shallow.index[0]).days) if shallow is not None else None,
            })
            print(f"  {tf} {name:10s} deep: {deep_stats['n_bars']:>7,} bars / "
                  f"{(deep.index[-1]-deep.index[0]).days:>4} days  Sharpe {deep_stats['sharpe']:+6.2f}   "
                  f"(shallow was {shallow_stats['sharpe']:+6.2f})" if shallow_stats else
                  f"  {tf} {name:10s} deep only", flush=True)

            for e in era_split(deep):
                eras.append({"asset": name, "timeframe": tf, **e})

            mk = markov_test(deep, n_sims, rng)
            if mk:
                markov.append({"asset": name, "timeframe": tf, **mk})
                print(f"      markov p={mk['markov_p']:.3f}  synth_mean={mk['synth_mean']:+.2f}", flush=True)

    df_rows = pd.DataFrame(rows)
    df_eras = pd.DataFrame(eras)
    df_markov = pd.DataFrame([{k: v for k, v in m.items() if k != "synth_samples"} for m in markov])

    payload = {
        "n_sims": n_sims, "n_eras_target": N_ERAS,
        "per_asset": df_rows.round(4).to_dict(orient="records"),
        "eras": df_eras.round(4).to_dict(orient="records") if not df_eras.empty else [],
        "markov": markov,
        "summary": {
            "n_instruments_tested": int(df_rows[["asset", "timeframe"]].drop_duplicates().shape[0]),
            "n_markov_sig_05": int((df_markov["markov_p"] < 0.05).sum()) if not df_markov.empty else 0,
            "n_markov_total": int(len(df_markov)),
            "by_timeframe": [
                {
                    "timeframe": tf,
                    "n": int(len(sub)),
                    "mean_deep_sharpe": round(float(sub["deep_sharpe"].mean()), 3),
                    "mean_shallow_sharpe": round(float(sub["shallow_sharpe"].mean()), 3) if sub["shallow_sharpe"].notna().any() else None,
                    "mean_deep_days": round(float(sub["deep_days"].mean()), 0),
                    "mean_shallow_days": round(float(sub["shallow_days"].mean()), 0) if sub["shallow_days"].notna().any() else None,
                }
                for tf, sub in df_rows.groupby("timeframe")
            ] if not df_rows.empty else [],
        },
    }
    with open(os.path.join(OUT_DIR, "deep_intraday_validation.json"), "w") as f:
        json.dump(_sanitize(payload), f, default=str)

    s = payload["summary"]
    print(f"\n=== Deep-sample crypto intraday: summary ===")
    for row in s["by_timeframe"]:
        print(f"  {row['timeframe']:4s} mean Sharpe -- deep ({row['mean_deep_days']:.0f}d): {row['mean_deep_sharpe']:+.2f}   "
              f"shallow ({row['mean_shallow_days'] or 0:.0f}d): {row['mean_shallow_sharpe']}")
    print(f"  Markov test: {s['n_markov_sig_05']}/{s['n_markov_total']} significant at p<0.05")
    print(f"\nwrote {os.path.join(OUT_DIR, 'deep_intraday_validation.json')}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
