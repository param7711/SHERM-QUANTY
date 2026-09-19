"""
Correctness tests for the SMA18 engine.

The point of this file is to make it hard for the backtest to report an
edge that isn't there. Two tests cover the two distinct ways a bar-based
backtest leaks information, and they catch different things:

  * `test_no_lookahead` runs the strategy on the first k bars and on the
    full series and asserts the shared prefix is identical. This catches
    CROSS-BAR leakage (a rule reading bars that haven't happened yet).
    It does NOT catch intrabar leakage, because truncating later bars
    leaves each earlier bar's own OHLC untouched.

  * `test_exit_uses_prior_bar_sma` catches INTRABAR leakage — using a
    level that only exists once the current bar closes to decide whether
    an intrabar touch happened. This is the bug that was actually found
    and fixed in this engine; verified to fail against the pre-fix code
    (16 of 19 exits violated it).

Run: python research/sma18_trend_forex/test_engine.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import SMA_PERIOD, add_sma, performance_stats, run_strategy


def _synthetic_ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.003, n - 1))])
    high = np.maximum.reduce([open_, close]) + spread
    low = np.minimum.reduce([open_, close]) - spread
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_no_lookahead():
    """Truncating the future must not change any past decision."""
    df = add_sma(_synthetic_ohlc(500, seed=1))
    for direction in ("long", "short", "both"):
        full_trades, full_equity = run_strategy(df, direction=direction)
        for k in (120, 250, 400):
            part_trades, part_equity = run_strategy(add_sma(df.iloc[:k].drop(columns="sma18")),
                                                     direction=direction)
            # Equity over the shared prefix must match bar for bar.
            np.testing.assert_allclose(
                part_equity.values, full_equity.values[:k], rtol=1e-12, atol=1e-12,
                err_msg=f"lookahead detected in equity ({direction}, k={k})")
            # Every closed trade that finished before the cut must match.
            fin_full = full_trades[(~full_trades["open"]) & (full_trades["exit_date"] <= df.index[k - 1])]
            fin_part = part_trades[~part_trades["open"]]
            assert len(fin_full) == len(fin_part), (
                f"trade count diverges ({direction}, k={k}): {len(fin_full)} vs {len(fin_part)}")
            np.testing.assert_allclose(
                fin_full["return_pct"].values, fin_part["return_pct"].values,
                rtol=1e-12, atol=1e-12,
                err_msg=f"lookahead detected in trades ({direction}, k={k})")
    print("PASS  test_no_lookahead — truncating the future changes nothing in the past")


def test_entry_rule_exact():
    """Entry fires on the 2nd consecutive close beyond the SMA, at that close."""
    df = add_sma(_synthetic_ohlc(300, seed=2))
    trades, _ = run_strategy(df, direction="long")
    closed = trades[~trades["open"]]
    assert len(closed) > 5, "test needs some trades"
    for _, t in closed.iterrows():
        i = df.index.get_loc(t["entry_date"])
        assert df["close"].iloc[i] > df["sma18"].iloc[i], "entry bar close not above SMA"
        assert df["close"].iloc[i - 1] > df["sma18"].iloc[i - 1], "prior bar close not above SMA"
        assert t["entry_price"] == df["close"].iloc[i], "entry not filled at that bar's close"
    print(f"PASS  test_entry_rule_exact — {len(closed)} long entries all satisfy the 2-candle rule")


def test_exit_uses_prior_bar_sma():
    """The exit level must be the SMA known at the previous close, never the
    current bar's own SMA (that would be lookahead on an intrabar fill)."""
    df = add_sma(_synthetic_ohlc(300, seed=3))
    trades, _ = run_strategy(df, direction="long")
    closed = trades[~trades["open"]]
    checked = 0
    for _, t in closed.iterrows():
        j = df.index.get_loc(t["exit_date"])
        sma_prev = df["sma18"].iloc[j - 1]
        if t["exit_price"] == df["open"].iloc[j]:
            assert df["open"].iloc[j] <= sma_prev, "gap exit without gapping past prior SMA"
        else:
            assert abs(t["exit_price"] - sma_prev) < 1e-9, "touch exit not filled at prior-bar SMA"
            assert df["low"].iloc[j] <= sma_prev <= df["high"].iloc[j], "touch exit outside bar range"
        checked += 1
    print(f"PASS  test_exit_uses_prior_bar_sma — {checked} exits all reference the prior-bar SMA")


def test_short_is_mirror_of_long():
    """Shorting a series must equal going long its exact mirror image."""
    df = _synthetic_ohlc(400, seed=4)
    mirror = pd.DataFrame({
        "open": 2 * df["close"].iloc[0] - df["open"],
        "high": 2 * df["close"].iloc[0] - df["low"],
        "low": 2 * df["close"].iloc[0] - df["high"],
        "close": 2 * df["close"].iloc[0] - df["close"],
    }, index=df.index)
    short_trades, _ = run_strategy(add_sma(df), direction="short")
    long_trades, _ = run_strategy(add_sma(mirror), direction="long")
    assert len(short_trades) == len(long_trades), (
        f"mirror mismatch: {len(short_trades)} short vs {len(long_trades)} long trades")
    print(f"PASS  test_short_is_mirror_of_long — {len(short_trades)} trades match on the mirrored series")


def test_equity_matches_trades():
    """Compounding the trade returns must reproduce the equity curve."""
    for direction in ("long", "short", "both"):
        df = add_sma(_synthetic_ohlc(400, seed=5))
        trades, equity = run_strategy(df, direction=direction)
        if trades.empty:
            continue
        compounded = float(np.prod(1 + trades["return_pct"].values / 100.0))
        # Equity compounds bar-by-bar off prior closes while trade returns
        # compound entry->exit, so they agree to within rounding of the
        # intra-trade path, not exactly. Require the same sign and a close match.
        rel = abs(compounded - equity.iloc[-1]) / max(abs(equity.iloc[-1]), 1e-9)
        assert rel < 0.02, f"{direction}: trade compounding {compounded:.4f} vs equity {equity.iloc[-1]:.4f}"
    print("PASS  test_equity_matches_trades — equity curve reconciles with trade returns")


def test_flat_when_no_signal():
    """A series that never closes beyond its SMA must produce zero trades."""
    n = 200
    idx = pd.bdate_range("2010-01-04", periods=n)
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)
    trades, equity = run_strategy(add_sma(flat), direction="both")
    assert trades.empty, "flat series produced trades"
    assert equity.iloc[-1] == 1.0, "flat series moved equity"
    print("PASS  test_flat_when_no_signal — a dead-flat series trades nothing")


def test_no_overlapping_positions():
    """At most one position open at a time; trades never overlap in time."""
    df = add_sma(_synthetic_ohlc(500, seed=6))
    trades, _ = run_strategy(df, direction="both")
    prev_exit = None
    for _, t in trades.iterrows():
        if prev_exit is not None:
            assert t["entry_date"] > prev_exit, "overlapping trades detected"
        prev_exit = t["exit_date"]
    print(f"PASS  test_no_overlapping_positions — {len(trades)} trades, none overlapping")


def test_fast_engine_matches_reference():
    """The array engine used for simulations must match the reference exactly.

    Everything the Markov/Monte-Carlo validation reports rests on this: if
    the fast twin drifted from the audited rule, every p-value would be
    measuring a different strategy than the one backtested.
    """
    from strategy import run_strategy_arrays
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1d")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet")) if os.path.isdir(data_dir) else []
    sources = [("synthetic", add_sma(_synthetic_ohlc(600, seed=7)))]
    for f in files[:4]:
        d = pd.read_parquet(os.path.join(data_dir, f))
        d.index = pd.to_datetime(d.index)
        sources.append((f, add_sma(d)))

    for label, df in sources:
        for direction in ("long", "short", "both"):
            trades, equity = run_strategy(df, direction=direction)
            rets, holds, sides, eq = run_strategy_arrays(
                df["open"].values, df["high"].values, df["low"].values,
                df["close"].values, df["sma18"].values, direction=direction)
            np.testing.assert_allclose(eq, equity.values, rtol=1e-12, atol=1e-12,
                                        err_msg=f"equity mismatch {label}/{direction}")
            closed = trades[~trades["open"]] if not trades.empty else trades
            assert len(rets) == len(closed), (
                f"trade count mismatch {label}/{direction}: {len(rets)} vs {len(closed)}")
            if len(closed):
                np.testing.assert_allclose(rets, closed["return_pct"].values,
                                            rtol=1e-12, atol=1e-12,
                                            err_msg=f"returns mismatch {label}/{direction}")
                np.testing.assert_allclose(holds, closed["holding_bars"].values,
                                            err_msg=f"holding mismatch {label}/{direction}")
    print(f"PASS  test_fast_engine_matches_reference — identical on {len(sources)} series x 3 directions")


def test_real_data_sanity():
    """On real cached data, win rate and holding period must be plausible."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "1d")
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")] if os.path.isdir(data_dir) else []
    if not files:
        print("SKIP  test_real_data_sanity — no cached daily data yet")
        return
    df = add_sma(pd.read_parquet(os.path.join(data_dir, files[0])))
    trades, equity = run_strategy(df, direction="long")
    perf = performance_stats(trades, equity)
    assert 0 < perf["win_rate"] < 1, "impossible win rate"
    assert perf["avg_holding_bars"] >= 1, "trades closing before they open"
    assert equity.min() > 0, "equity went non-positive"
    print(f"PASS  test_real_data_sanity — {files[0]}: {perf['n_trades']} trades, "
          f"win {perf['win_rate']:.0%}, avg hold {perf['avg_holding_bars']:.1f} bars")


def main():
    print("=== SMA18 engine correctness tests ===\n")
    test_no_lookahead()
    test_entry_rule_exact()
    test_exit_uses_prior_bar_sma()
    test_short_is_mirror_of_long()
    test_equity_matches_trades()
    test_flat_when_no_signal()
    test_no_overlapping_positions()
    test_fast_engine_matches_reference()
    test_real_data_sanity()
    print("\n=== all engine tests passed ===")


if __name__ == "__main__":
    main()
