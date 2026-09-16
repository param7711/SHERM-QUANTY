"""
Trade-sequence bootstrap Monte Carlo.

Given the actual multiset of trade outcomes a backtest produced, resample
that sequence with replacement many times to see how much of the reported
Sharpe/CAGR/drawdown is a property of the edge versus the luck of which
order those wins and losses happened to land in. This does not simulate
new price paths or new signals — it only asks "given these exact trade
outcomes, how variable is the result across possible orderings/samples
of them," which is the standard, cheap way to put a confidence band on a
backtest without re-running the strategy.

Caveat inherited by every caller: trades from the same instrument are not
strictly independent (they share the same trending/ranging regime), so
this understates real-world uncertainty somewhat — it is a lower bound on
how much the result could vary, not an upper bound.
"""

import numpy as np

MIN_TRADES_FOR_MC = 20
N_SIMS = 2000
SEED = 7


def bootstrap_trade_mc(trade_returns_pct, n_sims: int = N_SIMS, seed: int = SEED) -> dict:
    trade_returns_pct = np.asarray(trade_returns_pct, dtype=float)
    n = len(trade_returns_pct)
    if n < MIN_TRADES_FOR_MC:
        return None

    rng = np.random.default_rng(seed)
    sims = rng.choice(trade_returns_pct, size=(n_sims, n), replace=True)
    equity_paths = np.cumprod(1 + sims / 100.0, axis=1)          # (n_sims, n)
    final_ret_pct = (equity_paths[:, -1] - 1) * 100

    running_max = np.maximum.accumulate(equity_paths, axis=1)
    dd_paths = (equity_paths - running_max) / running_max
    max_dd_pct = -dd_paths.min(axis=1) * 100

    band = np.percentile(equity_paths, [5, 25, 50, 75, 95], axis=0)  # (5, n)
    actual_path = np.cumprod(1 + trade_returns_pct / 100.0)

    return {
        "n_sims": n_sims, "n_trades": n,
        "p_profit": float((final_ret_pct > 0).mean()),
        "final_return_p05": float(np.percentile(final_ret_pct, 5)),
        "final_return_p25": float(np.percentile(final_ret_pct, 25)),
        "final_return_p50": float(np.percentile(final_ret_pct, 50)),
        "final_return_p75": float(np.percentile(final_ret_pct, 75)),
        "final_return_p95": float(np.percentile(final_ret_pct, 95)),
        "max_dd_p50": float(np.percentile(max_dd_pct, 50)),
        "max_dd_p95": float(np.percentile(max_dd_pct, 95)),
        "band_p05": band[0].round(5).tolist(), "band_p25": band[1].round(5).tolist(),
        "band_p50": band[2].round(5).tolist(), "band_p75": band[3].round(5).tolist(),
        "band_p95": band[4].round(5).tolist(),
        "actual_path": actual_path.round(5).tolist(),
    }
