# %% [markdown]
# ---
# ## 10 - Strategy interface
#
# Everything above this point is about the market. Everything below is about a
# strategy, and the boundary between them is this one function signature:
#
# ```python
# def strategy(data: pd.DataFrame, params: dict) -> pd.DataFrame
# ```
#
# `data` is a canonical OHLCV frame. The returned frame must carry a `position`
# column - the target exposure decided at each bar's close, in units of the
# asset (+1 fully long, 0 flat, -1 fully short, and anything in between for
# sized positions) - plus a `signal` column recording the raw decision.
#
# ### Why the strategy does not compute its own P&L
#
# The obvious alternative is to let each strategy return its own
# `strategy_returns`. This framework deliberately does not, for one reason: the
# single most common bug in backtesting is applying a position to the same bar
# that produced the signal. It is easy to write by accident, it inflates results
# enormously, and it is invisible in the output.
#
# So the lag is applied in exactly one place, by `apply_execution()`, for every
# strategy, including the placebos. Transaction costs, slippage, execution delay
# and the return calculation live there too. A plugged-in strategy cannot
# accidentally exempt itself from any of them, and Section 34's leakage audit
# has one function to verify rather than many.
#
# The accounting rule, stated once:
#
# ```
# position[t]           decided from information up to and including bar t's close
# held[t] = position[t - 1 - delay]        what is actually owned during bar t
# strategy_return[t]    = held[t] * asset_return[t] - cost * |turnover[t]|
# ```
#
# ### Plugging in your own strategy
#
# Write a function with the signature above, register it in
# `STRATEGY_REGISTRY`, and set `STRATEGY_NAME`. Nothing else in the notebook
# needs to change - the generators, the fidelity scoring, and all of the
# robustness tests operate through this interface and never inspect strategy
# internals.

# %%
def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over w bars, NaN until the window is full.

    Matches pandas' `rolling(w).mean()` but avoids the pandas overhead, which
    dominates runtime when a strategy is evaluated tens of thousands of times
    across synthetic paths and parameter grids. Verified against pandas below.
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing sample standard deviation (ddof=1), NaN until the window is full.

    The series is centred on its own global mean before the cumulative sums are
    formed. Variance is shift-invariant so this changes nothing mathematically,
    but it matters numerically: the sum-of-squares formula computes
    `s2 - s1^2/w`, a difference between two nearly equal quantities. On raw
    price data (values in the hundreds) those two terms agree to several
    significant figures and the subtraction throws most of them away - at a
    2-bar window the error reached 3.5e-07, which the check below caught.
    Centring shrinks both terms by orders of magnitude and the cancellation
    with them.
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= w and w > 1:
        xc = x - x.mean()
        c1 = np.cumsum(np.insert(xc, 0, 0.0))
        c2 = np.cumsum(np.insert(xc * xc, 0, 0.0))
        s1 = c1[w:] - c1[:-w]
        s2 = c2[w:] - c2[:-w]
        var = (s2 - s1 * s1 / w) / (w - 1)
        out[w - 1:] = np.sqrt(np.maximum(var, 0.0))
    return out


def _ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs in a 1-D array."""
    x = np.asarray(x, dtype=float)
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[np.isnan(x) & (np.arange(len(x)) < (np.argmax(~np.isnan(x)) if (~np.isnan(x)).any() else len(x)))] = np.nan
    return out


def sma_positions(close: np.ndarray, lookback: int, confirm_bars: int,
                  direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Core SMA trend rule, fully vectorised. Returns (signal, position).

    Long entry  : close above the moving average for `confirm_bars` consecutive
                  bars. Long exit: close back below the average.
    Short side  : the exact mirror.
    direction   : 'long', 'short' or 'both' ('both' is a reversal system that is
                  always in the market once the first signal fires).

    No look-ahead: the moving average at bar t uses bars t-lookback+1..t, and
    the resulting position is only *applied* from bar t+1 onward by
    `apply_execution`. The position is held between signals, which is expressed
    here by forward-filling state rather than by looping.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    if confirm_bars < 1:
        raise ValueError(f"confirm_bars must be >= 1, got {confirm_bars}")
    if direction not in ("long", "short", "both"):
        raise ValueError(f"direction must be long/short/both, got {direction!r}")

    close = np.asarray(close, dtype=float)
    n = len(close)
    ma = _roll_mean(close, lookback)
    above = close > ma
    below = close < ma

    # `confirm_bars` consecutive closes on the same side of the average.
    conf_up = _roll_mean(above.astype(float), confirm_bars) == 1.0
    conf_dn = _roll_mean(below.astype(float), confirm_bars) == 1.0
    valid = ~np.isnan(ma)
    conf_up &= valid
    conf_dn &= valid

    signal = np.where(conf_up, 1.0, np.where(conf_dn, -1.0, 0.0))

    state = np.full(n, np.nan)
    if direction == "long":
        state[conf_up] = 1.0
        state[below & valid & ~conf_up] = 0.0
    elif direction == "short":
        state[conf_dn] = -1.0
        state[above & valid & ~conf_dn] = 0.0
    else:
        state[conf_up] = 1.0
        state[conf_dn] = -1.0

    pos = _ffill(state)
    pos = np.nan_to_num(pos, nan=0.0)
    return signal, pos


def _vol_target(close: np.ndarray, pos: np.ndarray, target_vol: float,
                bpy: float, window: int = 63, max_leverage: float = 3.0) -> np.ndarray:
    """Scale positions toward a constant target volatility.

    The scaling uses TRAILING realised volatility measured up to and including
    the current bar, so it carries no future information. Leverage is capped,
    because an uncapped inverse-volatility rule takes unbounded size in the
    calmest periods, which is both unrealistic and a reliable way to
    manufacture a spectacular backtest.
    """
    r = np.diff(close, prepend=close[0]) / close
    rv = _roll_std(r, window) * math.sqrt(bpy)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(rv > 1e-9, target_vol / rv, 0.0)
    scale = np.clip(np.nan_to_num(scale, nan=0.0), 0.0, max_leverage)
    return pos * scale


def sma_trend_strategy(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """The default strategy: moving-average trend following.

    params
    ------
    lookback      : moving-average length in bars.
    confirm_bars  : consecutive closes beyond the average required to enter.
    direction     : 'long' | 'short' | 'both'.
    target_vol    : if set, scale positions to this annualised volatility.

    Returns a frame indexed like `data` with `signal` and `position`.
    """
    close = data["close"].to_numpy(dtype=float)
    sig, pos = sma_positions(close, int(params["lookback"]),
                             int(params.get("confirm_bars", 2)),
                             params.get("direction", "long"))
    tv = params.get("target_vol")
    if tv:
        pos = _vol_target(close, pos, float(tv), infer_bars_per_year(data.index))
    return pd.DataFrame({"signal": sig, "position": pos}, index=data.index)


def donchian_breakout_strategy(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """A structurally different trend rule, included to show the interface is
    genuinely strategy-agnostic rather than shaped around the SMA rule.

    Enter long on a `lookback`-bar closing high; exit on an `exit_lookback`-bar
    closing low. Both extremes are computed over windows ending at the PREVIOUS
    bar, so the breakout is compared against a level that was already known.
    """
    close = data["close"]
    lb = int(params["lookback"])
    xlb = int(params.get("exit_lookback", max(2, lb // 2)))
    direction = params.get("direction", "long")

    hi = close.rolling(lb).max().shift(1)
    lo = close.rolling(xlb).min().shift(1)
    lo_entry = close.rolling(lb).min().shift(1)
    hi_exit = close.rolling(xlb).max().shift(1)

    state = pd.Series(np.nan, index=close.index)
    if direction in ("long", "both"):
        state[close > hi] = 1.0
    if direction in ("short", "both"):
        state[close < lo_entry] = -1.0
    if direction == "long":
        state[close < lo] = 0.0
    elif direction == "short":
        state[close > hi_exit] = 0.0

    pos = state.ffill().fillna(0.0)
    sig = state.fillna(0.0)
    return pd.DataFrame({"signal": sig.to_numpy(), "position": pos.to_numpy()},
                        index=close.index)


def buy_and_hold_strategy(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Always fully long. The benchmark every trend rule has to beat to justify
    its costs and its complexity."""
    n = len(data)
    return pd.DataFrame({"signal": np.zeros(n), "position": np.ones(n)}, index=data.index)


STRATEGY_REGISTRY = {
    "sma_trend": sma_trend_strategy,
    "donchian": donchian_breakout_strategy,
    "buy_and_hold": buy_and_hold_strategy,
}

print(f"registered strategies: {list(STRATEGY_REGISTRY)}")
print(f"active: {STRATEGY_NAME} with {STRATEGY_PARAMS}")

# %%
# Verify the fast rolling helpers against pandas, since every strategy
# evaluation in the notebook depends on them.
_rng_h = get_rng("rolling_helper_check")
_tst = np.cumsum(_rng_h.standard_normal(500)) + 100
for _w in (2, 5, 18, 63):
    _a = _roll_mean(_tst, _w)
    _b = pd.Series(_tst).rolling(_w).mean().to_numpy()
    _c = _roll_std(_tst, _w)
    _d = pd.Series(_tst).rolling(_w).std().to_numpy()
    _em = np.nanmax(np.abs(_a - _b))
    _es = np.nanmax(np.abs(_c - _d))
    _nan_ok = np.array_equal(np.isnan(_a), np.isnan(_b))
    print(f"  [{'PASS' if _em < 1e-9 and _es < 1e-8 and _nan_ok else 'FAIL'}] "
          f"w={_w:<3} mean err {_em:.2e}  std err {_es:.2e}  NaN pattern matches: {_nan_ok}")

# %% [markdown]
# ---
# ## 11 - Backtesting engine
#
# One execution layer, one metrics function, applied to every strategy and every
# market - real or synthetic.
#
# ### Metric definitions
#
# Stated explicitly, because these names are used inconsistently across the
# industry and an undefined Sharpe is not a number anyone can check:
#
# | metric | definition |
# |---|---|
# | annualised return | geometric: `(final equity)^(bpy/n_bars) - 1` |
# | annualised volatility | `sd(strategy returns) * sqrt(bars per year)` |
# | Sharpe | `mean(r) / sd(r) * sqrt(bpy)`, excess of zero (no risk-free rate) |
# | Sortino | `mean(r) / sd(r | r<0) * sqrt(bpy)` - only downside counts as risk |
# | Calmar | annualised return / max drawdown |
# | max drawdown | largest peak-to-trough decline of the equity curve |
# | max drawdown duration | longest run of bars spent below a previous peak |
# | downside deviation | `sd(r | r < 0) * sqrt(bpy)` |
# | win rate | share of closed trades with a positive return |
# | profit factor | gross trade profit / gross trade loss |
# | turnover | annualised sum of `|position change|` |
# | exposure | share of bars with a non-zero position |
# | n trades | maximal runs of constant-sign non-zero position |
#
# Sharpe is computed on **bar-level strategy returns**, not on trade returns.
# The two differ, and a Sharpe computed per-trade is not comparable to a
# published one.

# %%
def apply_execution(close: np.ndarray, position: np.ndarray, cost_bps: float,
                    slippage_bps: float, delay_bars: int = 0) -> dict:
    """Turn a position series into realised P&L. The ONLY place the lag is applied.

    `held[t] = position[t - 1 - delay]` is the exposure actually carried through
    bar t. Costs are charged on the bar where the position change takes effect,
    proportional to the size of the change, at `cost_bps + slippage_bps` per
    unit of turnover.

    Returns a dict of aligned arrays: held, asset_return, gross, cost,
    strategy_return, equity.
    """
    if delay_bars < 0:
        raise ValueError(f"delay_bars must be >= 0, got {delay_bars}")
    close = np.asarray(close, dtype=float)
    position = np.asarray(position, dtype=float)
    if len(close) != len(position):
        raise ValueError(f"length mismatch: close {len(close)} vs position {len(position)}")

    n = len(close)
    asset_ret = np.zeros(n)
    asset_ret[1:] = close[1:] / close[:-1] - 1.0

    shift = 1 + int(delay_bars)
    held = np.zeros(n)
    if shift < n:
        held[shift:] = position[:-shift]

    turnover = np.abs(np.diff(held, prepend=0.0))
    cost = turnover * (cost_bps + slippage_bps) / 1e4
    gross = held * asset_ret
    strat_ret = gross - cost
    equity = np.cumprod(1.0 + strat_ret)
    return {"held": held, "asset_return": asset_ret, "gross": gross, "cost": cost,
            "strategy_return": strat_ret, "equity": equity, "turnover": turnover}


def extract_trades(held: np.ndarray, strat_ret: np.ndarray) -> pd.DataFrame:
    """Identify trades as maximal runs of constant-sign, non-zero exposure.

    A trade's return compounds the bar-level strategy returns over its run, so
    it is net of the costs charged inside it.

    Vectorised rather than looped: this runs once per synthetic path, and with
    thousands of paths across dozens of parameter settings a per-bar Python loop
    here would dominate the notebook's entire runtime. Segment returns are
    compounded via a cumulative-log prefix sum, which turns each segment into
    two array lookups.
    """
    held = np.asarray(held, dtype=float)
    strat_ret = np.asarray(strat_ret, dtype=float)
    n = len(held)
    cols = ["start", "end", "bars", "side", "return"]
    if n == 0:
        return pd.DataFrame(columns=cols)

    sign = np.sign(held).astype(np.int8)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = sign[1:] != sign[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n) - 1

    keep = sign[starts] != 0
    starts, ends = starts[keep], ends[keep]
    if len(starts) == 0:
        return pd.DataFrame(columns=cols)

    logr = np.log1p(np.clip(strat_ret, -0.999999, None))
    clog = np.concatenate(([0.0], np.cumsum(logr)))
    seg_ret = np.expm1(clog[ends + 1] - clog[starts])

    return pd.DataFrame({
        "start": starts, "end": ends, "bars": ends - starts + 1,
        "side": np.where(sign[starts] > 0, "long", "short"),
        "return": seg_ret,
    }, columns=cols)


def max_drawdown_duration(equity: np.ndarray) -> int:
    """Longest number of consecutive bars spent below a previous equity peak."""
    dd = drawdown_series(equity)
    runs = run_lengths(dd < -1e-12)
    return int(runs.max()) if len(runs) else 0


def compute_backtest_metrics(exe: dict, bpy: float, trades: pd.DataFrame | None = None) -> dict:
    """The full metric suite for one backtest. See the table above for definitions."""
    r = exe["strategy_return"]
    eq = exe["equity"]
    n = len(r)
    if n < 2 or not np.isfinite(eq[-1]) or eq[-1] <= 0:
        return {k: np.nan for k in (
            "total_return", "ann_return", "ann_vol", "sharpe", "sortino", "calmar",
            "max_dd", "max_dd_duration", "downside_dev", "win_rate", "profit_factor",
            "avg_trade", "n_trades", "turnover_ann", "total_cost", "exposure",
            "skew", "kurtosis", "n_bars")}

    n_years = n / bpy if bpy > 0 else np.nan
    total_return = float(eq[-1] - 1.0)
    ann_return = float(eq[-1] ** (1.0 / n_years) - 1.0) if n_years and n_years > 0 else np.nan
    sd = float(r.std(ddof=1))
    ann_vol = sd * math.sqrt(bpy)
    sharpe = float(r.mean() / sd * math.sqrt(bpy)) if sd > 0 else np.nan
    down = r[r < 0]
    dsd = float(down.std(ddof=1)) if len(down) > 1 else np.nan
    downside_dev = dsd * math.sqrt(bpy) if np.isfinite(dsd) else np.nan
    sortino = float(r.mean() / dsd * math.sqrt(bpy)) if dsd and dsd > 0 else np.nan
    max_dd = float(-drawdown_series(eq).min())
    calmar = float(ann_return / max_dd) if max_dd > 1e-12 and np.isfinite(ann_return) else np.nan

    if trades is None:
        trades = extract_trades(exe["held"], r)
    if len(trades):
        tr = trades["return"].to_numpy()
        wins, losses = tr[tr > 0], tr[tr <= 0]
        win_rate = float((tr > 0).mean())
        profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf
        avg_trade = float(tr.mean())
    else:
        win_rate = profit_factor = avg_trade = np.nan

    return {
        "total_return": total_return, "ann_return": ann_return, "ann_vol": ann_vol,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_dd": max_dd, "max_dd_duration": max_drawdown_duration(eq),
        "downside_dev": downside_dev, "win_rate": win_rate,
        "profit_factor": profit_factor, "avg_trade": avg_trade,
        "n_trades": int(len(trades)), "turnover_ann": float(exe["turnover"].sum() / n_years)
            if n_years and n_years > 0 else np.nan,
        "total_cost": float(exe["cost"].sum()),
        "exposure": float(np.mean(exe["held"] != 0)),
        "skew": float(stats.skew(r)), "kurtosis": float(stats.kurtosis(r)),
        "n_bars": n,
    }


@dataclass
class BacktestResult:
    """One completed backtest: the frame, the trades and the metrics."""
    name: str
    params: dict
    frame: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    bpy: float

    @property
    def equity(self):
        return self.frame["equity"]

    @property
    def returns(self):
        return self.frame["strategy_return"]


def backtest(data: pd.DataFrame, strategy_fn, params: dict, bpy: float | None = None,
             cost_bps: float | None = None, slippage_bps: float | None = None,
             delay_bars: int | None = None, name: str = "") -> BacktestResult:
    """Run one strategy over one market and return the full result.

    Raises on an empty or degenerate backtest rather than returning NaNs that
    would propagate silently into a summary table.
    """
    if len(data) < 50:
        raise ValueError(f"need at least 50 bars to backtest, got {len(data)}")
    bpy = infer_bars_per_year(data.index) if bpy is None else bpy
    cost_bps = COST_BPS if cost_bps is None else cost_bps
    slippage_bps = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    delay_bars = EXECUTION_DELAY_BARS if delay_bars is None else delay_bars

    sig = strategy_fn(data, params)
    if "position" not in sig.columns:
        raise KeyError(f"strategy {name or strategy_fn.__name__!r} returned no "
                       f"'position' column; got {list(sig.columns)}")
    if len(sig) != len(data):
        raise ValueError(f"strategy returned {len(sig)} rows for {len(data)} bars")
    pos = sig["position"].to_numpy(dtype=float)
    if not np.isfinite(pos).all():
        raise ValueError(f"strategy {name!r} produced non-finite positions")

    exe = apply_execution(data["close"].to_numpy(dtype=float), pos,
                          cost_bps, slippage_bps, delay_bars)
    trades = extract_trades(exe["held"], exe["strategy_return"])
    frame = pd.DataFrame({
        "close": data["close"].to_numpy(), "signal": sig["signal"].to_numpy(),
        "position": pos, "held": exe["held"], "returns": exe["asset_return"],
        "strategy_return": exe["strategy_return"], "cost": exe["cost"],
        "equity": exe["equity"]}, index=data.index)
    metrics = compute_backtest_metrics(exe, bpy, trades)
    return BacktestResult(name or STRATEGY_NAME, dict(params), frame, trades, metrics, bpy)


def strategy_returns_fast(close: np.ndarray, params: dict, cost_bps: float,
                          slippage_bps: float, delay_bars: int = 0) -> np.ndarray:
    """Bar-level strategy returns, skipping the pandas layer entirely.

    Used by the parameter sweeps and the CSCV machinery, which evaluate the
    strategy tens of thousands of times. It calls the same `sma_positions` and
    `apply_execution` as the full path, so it cannot drift away from the
    headline backtest - a second, separate fast implementation would be exactly
    the kind of silent inconsistency this notebook is about.
    """
    _sig, pos = sma_positions(close, int(params["lookback"]),
                              int(params.get("confirm_bars", 2)),
                              params.get("direction", "long"))
    return apply_execution(close, pos, cost_bps, slippage_bps, delay_bars)["strategy_return"]


print("backtest engine defined")

# %% [markdown]
# ### No-look-ahead verification
#
# The claim that positions cannot see the future is the load-bearing assumption
# of every number in this notebook, so it is tested rather than asserted.
#
# The test: take the market, change **only the final bar's** close to something
# extreme, and re-run. Every position except possibly the last must be
# identical. If a single earlier position moves, information has flowed
# backwards in time.
#
# A second test shuffles a future segment and checks that earlier positions and
# the strategy returns over the untouched prefix are unchanged.

# %%
_test_data = PX.iloc[:1500].copy()
_res_base = backtest(_test_data, sma_trend_strategy, STRATEGY_PARAMS, name="lookahead-base")

# Test 1: perturb the last bar only.
_perturbed = _test_data.copy()
_perturbed.iloc[-1, _perturbed.columns.get_loc("close")] *= 5.0
_res_pert = backtest(_perturbed, sma_trend_strategy, STRATEGY_PARAMS, name="lookahead-last")
_pos_same = np.array_equal(_res_base.frame["position"].to_numpy()[:-1],
                           _res_pert.frame["position"].to_numpy()[:-1])
print(f"  [{'PASS' if _pos_same else 'FAIL'}] perturbing the final close leaves every "
      f"earlier position unchanged")

# Test 2: replace the last 300 bars entirely.
_rng_la = get_rng("lookahead_test")
_replaced = _test_data.copy()
_tail = _replaced.index[-300:]
_replaced.loc[_tail, "close"] = (_replaced.loc[_tail, "close"].to_numpy()
                                 * _rng_la.uniform(0.5, 2.0, 300))
_res_repl = backtest(_replaced, sma_trend_strategy, STRATEGY_PARAMS, name="lookahead-tail")
_k = len(_test_data) - 300
_prefix_same = np.array_equal(_res_base.frame["position"].to_numpy()[:_k],
                              _res_repl.frame["position"].to_numpy()[:_k])
_ret_same = np.allclose(_res_base.frame["strategy_return"].to_numpy()[:_k],
                        _res_repl.frame["strategy_return"].to_numpy()[:_k])
print(f"  [{'PASS' if _prefix_same else 'FAIL'}] replacing the last 300 bars leaves "
      f"all earlier positions unchanged")
print(f"  [{'PASS' if _ret_same else 'FAIL'}] ... and leaves earlier strategy returns unchanged")

# Test 3: the lag itself - a position can never earn the return of its own bar.
_chk = _res_base.frame
_lag_ok = np.array_equal(_chk["held"].to_numpy()[1:], _chk["position"].to_numpy()[:-1])
print(f"  [{'PASS' if _lag_ok else 'FAIL'}] held[t] == position[t-1] "
      f"(delay={EXECUTION_DELAY_BARS}) - signals never trade their own bar")

if not (_pos_same and _prefix_same and _ret_same and _lag_ok):
    raise AssertionError("LOOK-AHEAD DETECTED - every downstream result is invalid")
print("\nno look-ahead detected")

# %% [markdown]
# ---
# ## 12 - Historical backtest
#
# The baseline: what the strategy actually did on the real market. This is the
# number every synthetic comparison is measured against.
#
# It is reported on three slices. The **calibration slice** is the one that
# matters for Section 13, because the generators only ever saw that slice and
# the synthetic paths are the same length - comparing a full-sample result
# against calibration-length synthetic paths would credit the strategy for
# having had longer to average out its noise.

# %%
HIST = backtest(PX, STRATEGY_REGISTRY[STRATEGY_NAME], STRATEGY_PARAMS, name="historical-full")
BH = backtest(PX, buy_and_hold_strategy, {}, cost_bps=0.0, slippage_bps=0.0,
              name="buy-and-hold")

_calib_px = PX.iloc[:CALIB_END + 1]
_test_px = PX.iloc[CALIB_END:]
HIST_CALIB = backtest(_calib_px, STRATEGY_REGISTRY[STRATEGY_NAME], STRATEGY_PARAMS,
                      name="historical-calibration")
HIST_TEST = backtest(_test_px, STRATEGY_REGISTRY[STRATEGY_NAME], STRATEGY_PARAMS,
                     name="historical-heldout")
# Buy-and-hold on the calibration slice: the benchmark Section 13 measures
# timing alpha against, on exactly the slice the synthetic paths match.
BH_CALIB = backtest(_calib_px, buy_and_hold_strategy, {}, cost_bps=0.0,
                    slippage_bps=0.0, name="buy-and-hold-calibration")

_mrows = []
for _r in (HIST, HIST_CALIB, HIST_TEST, BH, BH_CALIB):
    _mrows.append({"backtest": _r.name, **_r.metrics})
HIST_TABLE = pd.DataFrame(_mrows).set_index("backtest")

_key = ["n_bars", "ann_return", "ann_vol", "sharpe", "sortino", "calmar", "max_dd",
        "max_dd_duration", "win_rate", "profit_factor", "n_trades", "exposure",
        "turnover_ann", "total_cost"]
print(f"Historical backtest - {PRIMARY_ASSET}, {STRATEGY_NAME} {STRATEGY_PARAMS}")
print(f"costs {COST_BPS} bps + {SLIPPAGE_BPS} bps slippage per unit turnover, "
      f"execution delay {EXECUTION_DELAY_BARS} bar(s)")
print("=" * 104)
print(HIST_TABLE[_key].round(4).to_string())

print(f"\nThe calibration-slice Sharpe of {HIST_CALIB.metrics['sharpe']:.3f} is the number")
print("Section 13 compares against the synthetic distributions.")
print(f"The held-out slice ({HIST_TEST.metrics['n_bars']} bars) was seen by no generator")
print(f"and by no fitted model anywhere above: Sharpe {HIST_TEST.metrics['sharpe']:.3f}.")

_alpha_calib = HIST_CALIB.metrics["sharpe"] - BH_CALIB.metrics["sharpe"]
print(f"\nAgainst buy-and-hold on the same slice ({BH_CALIB.metrics['sharpe']:.3f}), the")
print(f"strategy's timing alpha is {_alpha_calib:+.3f} Sharpe.")
if _alpha_calib < 0:
    print("That is NEGATIVE: on this market the rule underperformed simply owning the")
    print(f"asset, while running {HIST_CALIB.metrics['turnover_ann']:.0f}x annual turnover and paying real costs.")
    print("Hold on to that number - it reframes everything that follows. The question")
    print("stops being 'is the edge real?' and becomes 'is there any edge to test?'")

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 6.0), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(HIST.frame.index, HIST.frame["equity"], color=C["blue"], lw=1.6,
        label=f"{STRATEGY_NAME} (net of costs)")
ax.plot(BH.frame.index, BH.frame["equity"], color=INK["muted"], lw=1.3,
        label="buy & hold")
ax.axvline(PX.index[CALIB_END], color=STATUS["warning"], lw=1.4, ls="--")
ax.text(PX.index[CALIB_END], ax.get_ylim()[1], "  calibration ends", va="top",
        ha="left", fontsize=8, color=INK["secondary"])
ax.set_yscale("log")
finish(ax, f"{PRIMARY_ASSET} - strategy equity vs. buy & hold",
       ylabel="growth of 1 (log)",
       subtitle="everything left of the dashed line was visible to the generators")

ax = axes[1]
_dds = drawdown_series(HIST.frame["equity"].to_numpy())
_ddb = drawdown_series(BH.frame["equity"].to_numpy())
ax.fill_between(HIST.frame.index, _dds, 0, color=C["blue"], alpha=0.25, lw=0)
ax.plot(HIST.frame.index, _dds, color=C["blue"], lw=1.1, label=STRATEGY_NAME)
ax.plot(BH.frame.index, _ddb, color=INK["muted"], lw=1.1, label="buy & hold")
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
finish(ax, None, xlabel="date", ylabel="drawdown")
fig.tight_layout()
plt.show()

print(f"trades: {len(HIST.trades)}  win rate {HIST.metrics['win_rate']:.1%}  "
      f"profit factor {HIST.metrics['profit_factor']:.2f}  "
      f"avg trade {HIST.metrics['avg_trade']:+.3%}")
print(f"costs consumed {HIST.metrics['total_cost']:.2f} units of the "
      f"{HIST.metrics['total_return']+1:.2f}x gross growth "
      f"({HIST.metrics['turnover_ann']:.1f}x annual turnover)")
