"""
The full test battery, applied to the 6 crypto + 6 commodity focused set --
everything the main study runs, run again here so this dedicated artifact
isn't a thinner version of it.

For the 6 crypto instruments (BTCUSD, ETHUSD, SOLUSD, XRPUSD, ADAUSD,
DOGEUSD) across all 4 deep intraday timeframes (15m/30m/1h/4h, 24 combos):
  1. Long-only AND long+short backtest, compared directly.
  2. Markov-chain synthetic-market null (already computed by
     deep_intraday_validation.py, reused here).
  3. I.I.D. SHUFFLE null -- the same returns in random order, computed for
     the first time at these timeframes. Weaker test than Markov, run
     alongside it because that's what the main study does.
  4. Benjamini-Hochberg FDR correction across the 24 Markov p-values and,
     separately, the 24 shuffle p-values -- 24 simultaneous tests inflate
     false positives, same reasoning as the main study's 30-asset version.
  5. Parameter sweep, SMA 5..60, on the deep sample -- is 18 special, or
     does this timeframe show the same broad plateau the daily data does?
  6. Cost sensitivity, 0..50bps round-trip, on the deep sample -- at 15m
     this is the sharpest test in the whole study, because trade counts
     are far higher than daily.
  7. Trade-order bootstrap Monte Carlo (2000 resamples), the same
     backward-looking test the main report's grid section uses.

For the 6 commodities, no new computation runs -- every one of these
tests already exists for them from the main study's decades of real daily
data (markov_validation.py, era_splits.py, robustness_extras.py). This
script just gathers those results into the same shape as the crypto side.

Run: python deep_full_battery.py [n_sims]
Outputs: outputs/deep_full_battery.json
"""

import json, math, os, sys, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import SMA_PERIOD, bars_per_year, run_strategy_arrays, sharpe_from_equity
from markov_validation import (_rolling_mean, fit_markov, synth_ohlc_arrays,
                                markov_paths, shuffle_paths, N_STATES)
from monte_carlo import bootstrap_trade_mc
from deep_intraday_validation import _load, SELECTED_CRYPTO

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
TIMEFRAMES = ["15m", "30m", "1h", "4h"]
PERIODS = list(range(5, 61))
COST_BPS = [0, 2, 5, 10, 15, 20, 30, 50]
SELECTED_COMMODITY = ["CRUDE", "GOLD", "SUGAR", "HEATOIL", "SOYBEAN", "GASOLINE"]


def _sanitize(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [_sanitize(v) for v in o]
    return o


def bh_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg: returns the count significant after correction."""
    p = np.sort(np.asarray(pvals))
    n = len(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passing = p <= thresh
    if not passing.any():
        return 0
    return int(np.max(np.where(passing)[0]) + 1)


def fit_both_nulls(df):
    close = df["close"].values
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
    return log_ret, trans, pools, states, edges, shape_pools


def run_null(paths, close0, shape_pools, edges, bpy, rng, n_sims):
    sharpes = np.empty(n_sims)
    for i in range(n_sims):
        so, sh, sl, sc = synth_ohlc_arrays(paths[i], close0, shape_pools, edges, rng)
        s_sma = _rolling_mean(sc, SMA_PERIOD)
        _, _, _, s_eq = run_strategy_arrays(so, sh, sl, sc, s_sma, direction="long")
        sharpes[i] = sharpe_from_equity(s_eq, bpy)
    return sharpes


def crypto_battery(name, tf, n_sims, rng):
    df = _load(tf, name)
    if df is None or len(df) < 500:
        return None
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    sma = _rolling_mean(c, SMA_PERIOD)
    bpy = bars_per_year(df.index)

    rets_l, holds_l, sides_l, eq_l = run_strategy_arrays(o, h, l, c, sma, direction="long")
    rets_b, holds_b, sides_b, eq_b = run_strategy_arrays(o, h, l, c, sma, direction="both")
    if len(rets_l) < 15:
        return None
    sharpe_l = sharpe_from_equity(eq_l, bpy)
    sharpe_b = sharpe_from_equity(eq_b, bpy)

    log_ret, trans, pools, states, edges, shape_pools = fit_both_nulls(df)
    n_steps = len(log_ret)
    m_paths = markov_paths(trans, pools, states, n_steps, n_sims, rng)
    s_paths = shuffle_paths(log_ret, n_steps, n_sims, rng)
    markov_sharpes = run_null(m_paths, c[0], shape_pools, edges, bpy, rng, n_sims)
    shuffle_sharpes = run_null(s_paths, c[0], shape_pools, edges, bpy, rng, n_sims)
    markov_p = float((markov_sharpes >= sharpe_l).mean())
    shuffle_p = float((shuffle_sharpes >= sharpe_l).mean())

    # Parameter sweep
    sweep = {}
    for p in PERIODS:
        s = _rolling_mean(c, p)
        r, _, _, e = run_strategy_arrays(o, h, l, c, s, direction="long")
        sweep[p] = sharpe_from_equity(e, bpy)
    vals = np.array([sweep[p] for p in PERIODS])
    rank18 = int((vals > sweep[SMA_PERIOD]).sum()) + 1

    # Cost sensitivity
    cost_curve = {}
    breakeven = None
    n_bars = len(c)
    for bps in COST_BPS:
        if bps == 0:
            sh = sharpe_l
        else:
            n_trades = len(rets_l)
            drag_per_bar = (n_trades * bps / 10000.0) / max(n_bars - 1, 1)
            ret = np.diff(eq_l) / eq_l[:-1] - drag_per_bar
            sd = ret.std()
            sh = float(ret.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0
        cost_curve[str(bps)] = round(float(sh), 4)
        if breakeven is None and sh <= 0:
            breakeven = bps

    # Trade-order bootstrap Monte Carlo
    mc = bootstrap_trade_mc(rets_l, n_sims=2000)

    return {
        "asset": name, "timeframe": tf, "n_bars": int(len(df)), "n_trades_long": int(len(rets_l)),
        "n_trades_both": int(len(rets_b)), "sharpe_long": round(float(sharpe_l), 4),
        "sharpe_both": round(float(sharpe_b), 4), "sharpe_delta": round(float(sharpe_b - sharpe_l), 4),
        "markov_p": round(markov_p, 4), "markov_synth_mean": round(float(markov_sharpes.mean()), 4),
        "shuffle_p": round(shuffle_p, 4), "shuffle_synth_mean": round(float(shuffle_sharpes.mean()), 4),
        "sweep": {str(p): round(float(sweep[p]), 4) for p in PERIODS},
        "rank_of_18": rank18, "best_period": int(PERIODS[int(np.argmax(vals))]),
        "pct_periods_positive": round(float((vals > 0).mean()), 4),
        "cost_curve": cost_curve, "breakeven_bps": breakeven if breakeven is not None else f">{COST_BPS[-1]}",
        "mc": None if mc is None else {
            "p_profit": mc["p_profit"], "band_p05": mc["band_p05"][::max(1, len(mc["band_p05"]) // 150)],
            "band_p95": mc["band_p95"][::max(1, len(mc["band_p95"]) // 150)],
            "band_p50": mc["band_p50"][::max(1, len(mc["band_p50"]) // 150)],
            "actual_path": mc["actual_path"][::max(1, len(mc["actual_path"]) // 150)],
        },
    }


def commodity_summary():
    mk = {r["asset"]: r for r in json.load(open(os.path.join(OUT_DIR, "markov_validation.json")))["flat"]}
    eras = {r["asset"]: r for r in json.load(open(os.path.join(OUT_DIR, "era_splits.json")))["per_asset"]}
    extras = json.load(open(os.path.join(OUT_DIR, "robustness_extras.json")))
    periods_by_asset = {r["asset"]: r for r in extras["per_asset_periods"]}
    costs_by_asset = {r["asset"]: r for r in extras["per_asset_costs"]}
    grid = pd.read_csv(os.path.join(OUT_DIR, "grid_summary.csv"))
    grid = grid[(grid.error.isna()) & (grid.timeframe == "1d")]

    out = []
    for name in SELECTED_COMMODITY:
        m, e, p, c = mk[name], eras[name], periods_by_asset[name], costs_by_asset[name]
        row = grid[grid.asset == name].iloc[0]
        out.append({
            "asset": name, "n_bars": m["n_bars"], "sharpe": m["real_sharpe"],
            "n_trades": m["real_n_trades"], "markov_p": m["markov_p"], "shuffle_p": m["shuffle_p"],
            "n_eras": e["n_eras"], "n_eras_positive": e["n_eras_positive"],
            "worst_era_sharpe": e["worst_era_sharpe"], "rank_of_18": p["rank_of_18"],
            "n_periods": p["n_periods"], "pct_periods_positive": p["pct_periods_positive"],
            "breakeven_bps": c["breakeven_bps"], "sharpe_at_20bps": c["by_cost"]["20"]["sharpe"],
            "win_rate": float(row["win_rate"]), "max_drawdown_pct": float(row["max_drawdown_pct"]),
        })
    return out


def main(n_sims=300):
    rng = np.random.default_rng(20260920)
    crypto_rows = []
    t_start = time.time()
    for tf in TIMEFRAMES:
        for name in SELECTED_CRYPTO:
            t0 = time.time()
            r = crypto_battery(name, tf, n_sims, rng)
            if r is None:
                print(f"  {tf} {name:10s} skipped (insufficient data)", flush=True)
                continue
            crypto_rows.append(r)
            print(f"  {tf} {name:10s} long={r['sharpe_long']:+.2f} both={r['sharpe_both']:+.2f}  "
                  f"markov_p={r['markov_p']:.3f} shuffle_p={r['shuffle_p']:.3f}  "
                  f"rank18={r['rank_of_18']}/{len(PERIODS)}  breakeven={r['breakeven_bps']}bps  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    markov_ps = [r["markov_p"] for r in crypto_rows]
    shuffle_ps = [r["shuffle_p"] for r in crypto_rows]
    n_markov_bh = bh_correct(markov_ps)
    n_shuffle_bh = bh_correct(shuffle_ps)

    commodity_rows = commodity_summary()

    payload = {
        "n_sims": n_sims, "periods": PERIODS, "cost_bps": COST_BPS,
        "crypto": crypto_rows, "commodity": commodity_rows,
        "summary": {
            "n_crypto_tests": len(crypto_rows),
            "n_markov_sig_05": int(sum(1 for p in markov_ps if p < 0.05)),
            "n_markov_sig_bh": n_markov_bh,
            "n_shuffle_sig_05": int(sum(1 for p in shuffle_ps if p < 0.05)),
            "n_shuffle_sig_bh": n_shuffle_bh,
            "n_long_beats_both": int(sum(1 for r in crypto_rows if r["sharpe_long"] > r["sharpe_both"])),
            "mean_pct_periods_positive": round(float(np.mean([r["pct_periods_positive"] for r in crypto_rows])), 4),
            "median_rank_of_18": int(np.median([r["rank_of_18"] for r in crypto_rows])),
            "n_positive_at_20bps": int(sum(1 for r in crypto_rows if r["cost_curve"]["20"] > 0)),
            "total_runtime_s": round(time.time() - t_start, 1),
        },
    }
    with open(os.path.join(OUT_DIR, "deep_full_battery.json"), "w") as f:
        json.dump(_sanitize(payload), f, default=str)

    s = payload["summary"]
    print(f"\n=== Full battery summary ({s['total_runtime_s']:.0f}s) ===")
    print(f"Markov:  {s['n_markov_sig_05']}/{s['n_crypto_tests']} sig at p<0.05, {s['n_markov_sig_bh']} after BH")
    print(f"Shuffle: {s['n_shuffle_sig_05']}/{s['n_crypto_tests']} sig at p<0.05, {s['n_shuffle_sig_bh']} after BH")
    print(f"Long beats long+short on {s['n_long_beats_both']}/{s['n_crypto_tests']} combos")
    print(f"Median rank of SMA-18 among {len(PERIODS)} lengths: {s['median_rank_of_18']}")
    print(f"Positive at 20bps: {s['n_positive_at_20bps']}/{s['n_crypto_tests']}")
    print(f"\nwrote {os.path.join(OUT_DIR, 'deep_full_battery.json')}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
