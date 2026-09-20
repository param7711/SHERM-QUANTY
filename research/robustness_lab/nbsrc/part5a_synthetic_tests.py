# %% [markdown]
# ---
# ## 13 - Synthetic backtests (overfitting test 1: the synthetic path test)
#
# The frozen generators now produce alternative histories, and the unmodified
# strategy runs on every one. The question is where the real result falls inside
# the resulting distribution.
#
# ### How to read this - and the trap to avoid
#
# The intuitive reading is "the historical result should beat the synthetic
# distribution, and the further it beats it the better." That reading is wrong,
# and applying it uniformly here would produce a badly mistaken conclusion.
#
# Two comparisons carry the weight, and neither is "did the real result win."
#
# **Comparison A - real versus the dependence-destroying nulls.** Some
# generators keep the return distribution intact while destroying temporal
# ordering entirely: the IID bootstrap most completely, the Fourier and IAAFT
# surrogates in more targeted ways. A strategy that times the market should do
# *worse* on those than on the real thing, because the thing it times is gone.
# If it performs the same, its returns are coming from the distribution - from
# drift and exposure - and not from timing at all.
#
# Generators are ordered below by a **measured** dependence score: the Section
# 09 coverage of the dependence and volatility statistic groups. A low score
# means that generator genuinely failed to reproduce the market's temporal
# structure, which is exactly what makes it a useful null. This replaces a
# cruder classification based on the variance ratio alone, which turned out to
# be uninformative on this data for a reason the next cell makes clear.
#
# **Comparison B - the strategy versus buy-and-hold on the same paths.** A
# long-only strategy in a rising market earns money simply by being exposed.
# That is not an edge; it is beta, available for free and without turnover. So
# every synthetic path is also traded by a buy-and-hold rule, and the difference
# between the two is reported as **timing alpha**. If timing alpha is around
# zero or negative, the strategy is an expensive way to hold the asset.
#
# Together the two comparisons separate three possibilities that a raw Sharpe
# ratio cannot tell apart: a genuine timing edge, drift capture dressed up as
# one, and a fit to one particular historical path.

# %%
def make_shape_pools(df: pd.DataFrame, logret: np.ndarray, n_buckets: int = 5):
    """Bar shapes (open/close, high/close, low/close), bucketed by return size.

    Shapes are sampled CONDITIONAL on the bar's own return bucket rather than at
    random, because in real data the size of a move and the position of the high
    and low within the bar are strongly linked: big up bars close near their
    high, big down bars near their low. Pairing shapes at random would hand an
    up bar the deep low of a down bar, manufacturing intrabar excursions that
    could never have happened and making any intrabar stop or touch rule look
    different than it should.
    """
    edges = np.quantile(logret, np.linspace(0, 1, n_buckets + 1)[1:-1])
    states = np.searchsorted(edges, logret)
    c = df["close"].to_numpy()[1:len(logret) + 1]
    shapes = np.column_stack([df["open"].to_numpy()[1:len(logret) + 1] / c,
                              df["high"].to_numpy()[1:len(logret) + 1] / c,
                              df["low"].to_numpy()[1:len(logret) + 1] / c])
    ok = np.isfinite(shapes).all(axis=1)
    pools = []
    for s in range(n_buckets):
        pool = shapes[ok & (states[:len(shapes)] == s)]
        pools.append(pool if len(pool) >= 5 else shapes[ok])
    return edges, pools


SHAPE_EDGES, SHAPE_POOLS = make_shape_pools(PX.iloc[:CALIB_END + 1], CALIB_LOGRET)


def synthetic_frame(logret_path: np.ndarray, index: pd.DatetimeIndex, rng,
                    base_price: float = 100.0) -> pd.DataFrame:
    """Turn a synthetic log-return path into a full OHLCV frame.

    Full OHLC (rather than close mirrored four ways) is synthesised so that a
    plugged-in strategy using intrabar information still works against every
    generator. The reconstruction guarantees high >= max(open, close) and
    low <= min(open, close), so no bar is internally impossible.
    """
    close = base_price * np.exp(np.cumsum(logret_path))
    states = np.searchsorted(SHAPE_EDGES, logret_path)
    shp = np.empty((len(close), 3))
    for s, pool in enumerate(SHAPE_POOLS):
        m = states == s
        k = int(m.sum())
        if k:
            shp[m] = pool[rng.integers(0, len(pool), size=k)]
    open_ = close * shp[:, 0]
    high = np.maximum.reduce([close * shp[:, 1], open_, close])
    low = np.minimum.reduce([close * shp[:, 2], open_, close])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.nan}, index=index[:len(close)])


CALIB_INDEX = RET.index[:N_STEPS]


def backtest_paths(logret_paths: np.ndarray, strategy_fn, params: dict, rng,
                   cost_bps=None, slippage_bps=None, delay_bars=None) -> pd.DataFrame:
    """Run one strategy across many synthetic paths -> one metrics row per path."""
    rows = []
    for p in logret_paths:
        frame = synthetic_frame(p, CALIB_INDEX, rng)
        try:
            res = backtest(frame, strategy_fn, params, bpy=BPY, cost_bps=cost_bps,
                           slippage_bps=slippage_bps, delay_bars=delay_bars)
            rows.append(res.metrics)
        except Exception:
            rows.append({k: np.nan for k in HIST_CALIB.metrics})
    return pd.DataFrame(rows)


# First: does this market contain any trend structure for a trend rule to find?
# That question has to be settled before the strategy's results mean anything,
# and it is answered by the price data alone.
_real_vr = REAL_CALIB_STATS["vr10"]
_vr_full, _vr_z, _vr_p = variance_ratio(LOGRET.to_numpy(), 10)
print("=" * 74)
print("PRIOR QUESTION: is there trend structure in this market at all?")
print("=" * 74)
print(f"  10-bar variance ratio, full sample   {_vr_full:.4f}  (1.0 = random walk)")
print(f"  heteroskedasticity-robust p-value    {_vr_p:.4f}")
print(f"  10-bar variance ratio, calibration   {_real_vr:.4f}")
print(f"  Hurst exponent                       {REAL_CALIB_STATS['hurst']:.4f}  (0.5 = random walk)")
if _vr_p > 0.05:
    print(f"\n  The random-walk null is NOT rejected (p = {_vr_p:.3f}), and the point")
    print(f"  estimate sits {'below' if _vr_full < 1 else 'above'} 1. On this market, at this horizon, there is no")
    print("  statistically detectable trend structure to exploit.")
    print("\n  That does not make the strategy's historical profit fake - it makes its")
    print("  SOURCE the open question. A long-only rule can earn money from simple")
    print("  exposure to a rising asset without timing anything. The two comparisons")
    print("  below are built to tell those apart.")
else:
    print(f"\n  The random-walk null IS rejected (p = {_vr_p:.4f}): this market carries")
    print("  measurable trend structure, so a trend rule has something real to find.")

# Classify generators by MEASURED preservation of temporal structure, using the
# Section 09 dependence and volatility coverage. The variance ratio alone is not
# usable for this here: with the real VR already indistinguishable from 1, a
# generator that destroys all dependence also lands near 1 and would be scored
# as "matching" - the IID bootstrap would be classed as trend-preserving, which
# is plainly wrong. Dependence coverage does not have that degeneracy.
DEP_CLASS = {}
for _n in SYNTH_STATS:
    _f = FIDELITY.get(_n, {})
    _dep = float(np.nanmean([_f.get("dependence", np.nan), _f.get("volatility", np.nan)]))
    DEP_CLASS[_n] = {
        "dep_score": _dep,
        "vr10_median": float(SYNTH_STATS[_n]["vr10"].median()),
        "acf_abs1_median": float(SYNTH_STATS[_n]["acf_abs1"].median()),
        "preserves_dependence": bool(_dep >= 0.5),
    }
_dc = pd.DataFrame(DEP_CLASS).T.sort_values("dep_score")
print(f"\nMeasured temporal-structure preservation "
      f"(real ACF(1) of |returns| = {REAL_CALIB_STATS['acf_abs1']:.4f})")
print(_dc.to_string())
print("\nThe generators at the TOP of this table are the informative nulls: they")
print("reproduce the return distribution but measurably fail to reproduce the")
print("market's temporal structure.")

# %%
STRAT_FN = STRATEGY_REGISTRY[STRATEGY_NAME]
SYNTH_BT: dict[str, pd.DataFrame] = {}

print(f"running {STRATEGY_NAME} across {N_SIMULATIONS} paths per generator\n")
for _name in [n for n, g in GENERATORS.items() if g.available]:
    _t0 = _time.time()
    try:
        _rng_b = get_rng(f"backtest::{_name}")
        _paths = GENERATORS[_name](_rng_b, N_SIMULATIONS, N_STEPS)
        _bt = backtest_paths(_paths, STRAT_FN, STRATEGY_PARAMS, _rng_b)
        # Buy-and-hold on the identical paths. With a constant unit position and
        # no costs, the strategy return each bar IS the asset return, so this is
        # just the Sharpe of the path itself - no separate backtest needed.
        _simple = np.expm1(_paths)
        _sd_p = _simple.std(axis=1, ddof=1)
        _bt["bh_sharpe"] = np.where(_sd_p > 0,
                                    _simple.mean(axis=1) / _sd_p * math.sqrt(BPY), np.nan)
        _bt["timing_alpha"] = _bt["sharpe"] - _bt["bh_sharpe"]
        SYNTH_BT[_name] = _bt
        _s = _bt["sharpe"]
        print(f"  {_name:<28} median Sharpe {_s.median():+.3f}  "
              f"[{_s.quantile(0.05):+.2f}, {_s.quantile(0.95):+.2f}]   "
              f"alpha vs B&H {_bt['timing_alpha'].median():+.3f}  "
              f"{_time.time()-_t0:5.1f}s")
    except Exception as exc:
        print(f"  {_name:<28} FAILED: {type(exc).__name__}: {exc}")

print(f"\nhistorical (calibration slice) Sharpe = {HIST_CALIB.metrics['sharpe']:+.4f}")

# %%
def percentile_of(value: float, sample: pd.Series) -> float:
    """Fraction of the sample at or below `value`."""
    s = sample.replace([np.inf, -np.inf], np.nan).dropna()
    return float((s <= value).mean()) if len(s) else np.nan


_H = HIST_CALIB.metrics
_real_alpha = _H["sharpe"] - BH_CALIB.metrics["sharpe"]
_rows = []
for _name, _bt in SYNTH_BT.items():
    _s = _bt["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(_s) < 10:
        continue
    _pct = percentile_of(_H["sharpe"], _s)
    _a = _bt["timing_alpha"].replace([np.inf, -np.inf], np.nan).dropna()
    _rows.append({
        "generator": _name,
        "dep_score": DEP_CLASS.get(_name, {}).get("dep_score", np.nan),
        "fidelity": FIDELITY.get(_name, {}).get("overall", np.nan),
        "median_sharpe": float(_s.median()),
        "p5_sharpe": float(_s.quantile(0.05)),
        "p95_sharpe": float(_s.quantile(0.95)),
        "median_bh": float(_bt["bh_sharpe"].median()),
        "timing_alpha": float(_a.median()) if len(_a) else np.nan,
        "median_cagr": float(_bt["ann_return"].median()),
        "median_maxdd": float(_bt["max_dd"].median()),
        "hist_pctile": _pct,
        "p_value": float(1.0 - _pct),          # P(synthetic Sharpe >= historical)
        "z_score": float((_H["sharpe"] - _s.mean()) / _s.std(ddof=1)) if _s.std(ddof=1) > 0 else np.nan,
    })

SYNTH_SUMMARY = pd.DataFrame(_rows).set_index("generator").sort_values("dep_score")
print("Central result table - historical vs. synthetic market performance")
print(f"historical Sharpe (calibration slice) = {_H['sharpe']:+.4f}, "
      f"CAGR {_H['ann_return']:+.2%}, max DD {_H['max_dd']:.1%}")
print(f"buy-and-hold on the same slice = {BH_CALIB.metrics['sharpe']:+.4f}  "
      f"-> real timing alpha {_real_alpha:+.4f}")
print("sorted by dep_score: the LOWEST rows destroyed the most temporal structure")
print("=" * 132)
print(SYNTH_SUMMARY.round(4).to_string())

# --- Comparison A: real versus the dependence-destroying nulls ---------------
_nulls = SYNTH_SUMMARY.nsmallest(4, "dep_score")
print("\n" + "=" * 74)
print("COMPARISON A - does the edge need temporal structure?")
print("=" * 74)
print("The four generators that destroyed the most temporal structure:")
for _n, _r in _nulls.iterrows():
    print(f"  {_n:<26} dep {_r['dep_score']:.2f}  median Sharpe {_r['median_sharpe']:+.3f}  "
          f"p = {_r['p_value']:.3f}")
_null_med = _nulls["median_sharpe"].median()
_gap = _H["sharpe"] - _null_med
print(f"\n  real Sharpe                     {_H['sharpe']:+.4f}")
print(f"  median across these nulls       {_null_med:+.4f}")
print(f"  difference                      {_gap:+.4f}")
if abs(_gap) < 0.15:
    print("\n  These are effectively the same number. On markets where the ordering of")
    print("  returns has been destroyed - where there is nothing whatsoever to time -")
    print("  the strategy earns what it earned on the real market. Its returns are")
    print("  therefore coming from the RETURN DISTRIBUTION, not from timing.")
else:
    print(f"\n  The real result sits {_gap:+.3f} Sharpe away from the structure-free nulls,")
    print("  which is the signature of an edge that needs real temporal structure.")

# --- Comparison B: timing alpha versus buy-and-hold -------------------------
print("\n" + "=" * 74)
print("COMPARISON B - does timing beat simply holding the asset?")
print("=" * 74)
_alpha_neg = int((SYNTH_SUMMARY["timing_alpha"] < 0).sum())
print(f"  real timing alpha                       {_real_alpha:+.4f}")
print(f"  median synthetic timing alpha           "
      f"{SYNTH_SUMMARY['timing_alpha'].median():+.4f}")
print(f"  generators where timing alpha < 0       {_alpha_neg}/{len(SYNTH_SUMMARY)}")
if _real_alpha < 0 and _alpha_neg > len(SYNTH_SUMMARY) / 2:
    print("\n  The strategy underperforms buy-and-hold on the real market AND on most")
    print("  synthetic markets. It is not converting its activity into anything a")
    print("  flat long position would not have delivered - at a fraction of the")
    print("  turnover and none of the cost.")

# %%
fig, ax = plt.subplots(figsize=(10.5, 7.4))
_ord = SYNTH_SUMMARY.sort_values("dep_score")
_y = np.arange(len(_ord))
_col = [C["orange"] if d < 0.5 else C["blue"] for d in _ord["dep_score"]]

for _i, (_n, _r) in enumerate(_ord.iterrows()):
    ax.plot([_r["p5_sharpe"], _r["p95_sharpe"]], [_i, _i], color=_col[_i], lw=2.4,
            solid_capstyle="round", alpha=0.85)
ax.scatter(_ord["median_sharpe"], _y, s=36, color=_col, zorder=4,
           edgecolor=INK["surface"], linewidth=1.2)
ax.scatter(_ord["median_bh"], _y, s=30, marker="|", color=INK["muted"], zorder=3,
           linewidth=1.8, label="buy & hold on same paths")
ax.axvline(_H["sharpe"], color=INK["primary"], lw=1.8, ls="--", zorder=5)
ax.text(_H["sharpe"], -0.9, f" real {_H['sharpe']:+.2f}", fontsize=9,
        color=INK["primary"], va="bottom")
ax.axvline(BH_CALIB.metrics["sharpe"], color=STATUS["warning"], lw=1.6, ls=":", zorder=5)
ax.text(BH_CALIB.metrics["sharpe"], -0.9, f" real B&H {BH_CALIB.metrics['sharpe']:+.2f}",
        fontsize=9, color=INK["secondary"], va="bottom")
ax.set_yticks(_y)
ax.set_yticklabels([f"{n}  ({d:.2f})" for n, d in zip(_ord.index, _ord["dep_score"])],
                   fontsize=8)
for _lab, _c in [("destroyed temporal structure (dep < 0.5)", C["orange"]),
                 ("preserved temporal structure", C["blue"])]:
    ax.plot([], [], color=_c, lw=2.4, label=_lab)
finish(ax, "Strategy Sharpe across synthetic markets",
       xlabel="Sharpe ratio", subtitle="dot = median, bar = 5th-95th percentile of "
       f"{N_SIMULATIONS} paths; y-labels carry the dependence score")
ax.grid(axis="y", visible=False)
fig.tight_layout()
plt.show()

# %%
_pick = [n for n in ["iid_bootstrap", "iaaft_surrogate", "stationary_L20",
                     "filtered_historical_sim"] if n in SYNTH_BT][:4]
if _pick:
    fig, axes = plt.subplots(1, len(_pick), figsize=(3.0 * len(_pick), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, _n in zip(axes, _pick):
        _s = SYNTH_BT[_n]["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
        _is_null = SYNTH_SUMMARY.loc[_n, "dep_score"] < 0.5
        ax.hist(_s, bins=26, color=C["orange"] if _is_null else C["blue"], alpha=0.8)
        ax.axvline(_H["sharpe"], color=INK["primary"], lw=1.8, ls="--")
        finish(ax, f"{_n}\np = {SYNTH_SUMMARY.loc[_n,'p_value']:.3f}",
               xlabel="Sharpe", ylabel="paths" if _n == _pick[0] else None, legend=False)
    fig.suptitle("Synthetic Sharpe distributions (dashed line = the real result)",
                 x=0.005, ha="left", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()
    print("If the real line falls in the middle of the orange (structure-destroyed)")
    print("histograms, the strategy performs no better on the real market than on")
    print("markets built to contain nothing it could possibly time.")

# %% [markdown]
# ---
# ## 14 - Parameter robustness (overfitting test 2)
#
# A strategy that works at `lookback=18` and nowhere else has not found a market
# property - it has found a coincidence. Real structure is smooth: if an 18-bar
# average captures something, a 17- or 19-bar average should capture nearly the
# same thing.
#
# Three things are measured:
#
# 1. **The parameter surface** on real data - is the peak a plateau or a spike?
# 2. **Local perturbation** - what happens at +/-5% and +/-10% of the chosen value?
# 3. **The data-mining tax** - how much in-sample Sharpe does searching the grid
#    buy you *on synthetic markets*, where by construction there is nothing extra
#    to find? That distribution is the benchmark for judging how impressive the
#    real optimum is.
#
# The third is the one that makes this a genuine overfitting test rather than a
# sensitivity plot. Picking the best of 56 parameter values always produces a
# flattering number; the only way to know how flattering is to run the identical
# search on markets where the answer is known to be noise.
#
# Thresholds are configurable (`ROBUST_PLATEAU_TOLERANCE`) and nothing here
# declares a strategy robust or not - it reports the surface and lets the
# evidence speak.

# %%
_grid = PARAM_GRID["lookback"]
_calib_close = PX["close"].to_numpy()[:CALIB_END + 1]
_full_close = PX["close"].to_numpy()


def sweep_lookback(close: np.ndarray, grid, params: dict, bpy: float,
                   cost_bps=None, slippage_bps=None) -> np.ndarray:
    """Sharpe for every lookback in the grid, on one price series."""
    cost_bps = COST_BPS if cost_bps is None else cost_bps
    slippage_bps = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    out = np.full(len(grid), np.nan)
    for i, lb in enumerate(grid):
        if lb >= len(close) // 4:
            continue
        r = strategy_returns_fast(close, {**params, "lookback": int(lb)},
                                  cost_bps, slippage_bps, EXECUTION_DELAY_BARS)
        sd = r.std(ddof=1)
        out[i] = r.mean() / sd * math.sqrt(bpy) if sd > 0 else np.nan
    return out


SWEEP_CALIB = sweep_lookback(_calib_close, _grid, STRATEGY_PARAMS, BPY)
SWEEP_FULL = sweep_lookback(_full_close, _grid, STRATEGY_PARAMS, BPY)

_chosen = int(STRATEGY_PARAMS["lookback"])
_ci = _grid.index(_chosen) if _chosen in _grid else int(np.argmin(np.abs(np.array(_grid) - _chosen)))
_best_i = int(np.nanargmax(SWEEP_CALIB))
_peak = SWEEP_CALIB[_best_i]
_tol_level = _peak * (1 - ROBUST_PLATEAU_TOLERANCE) if _peak > 0 else -np.inf
_within = np.array(SWEEP_CALIB) >= _tol_level
_plateau_frac = float(np.nanmean(_within))

# Width of the contiguous plateau containing the chosen parameter.
_lo = _ci
while _lo > 0 and _within[_lo - 1]:
    _lo -= 1
_hi = _ci
while _hi < len(_grid) - 1 and _within[_hi + 1]:
    _hi += 1
_contig = (_grid[_hi] - _grid[_lo] + 1) if _within[_ci] else 0

# Bound to stable names so Section 22's report cannot pick up a recycled
# temporary from a later cell.
PARAM_SWEEP = {
    "grid": list(_grid), "calib": np.asarray(SWEEP_CALIB), "full": np.asarray(SWEEP_FULL),
    "chosen": _chosen, "chosen_idx": _ci, "chosen_sharpe": float(SWEEP_CALIB[_ci]),
    "best_lookback": _grid[_best_i], "peak_sharpe": float(_peak),
    "premium": float(_peak - SWEEP_CALIB[_ci]), "plateau_frac": _plateau_frac,
    "contig_width": _contig, "contig_lo": _grid[_lo], "contig_hi": _grid[_hi],
}

print(f"Parameter sweep over lookback {_grid[0]}..{_grid[-1]} ({len(_grid)} values)")
print(f"  chosen lookback        {_chosen}  -> calibration Sharpe {SWEEP_CALIB[_ci]:+.4f}")
print(f"  best on calibration    {_grid[_best_i]}  -> Sharpe {_peak:+.4f}")
print(f"  optimisation premium   {_peak - SWEEP_CALIB[_ci]:+.4f} Sharpe "
      f"(what searching the grid would have bought)")
print(f"  plateau (within {ROBUST_PLATEAU_TOLERANCE:.0%} of peak): "
      f"{_plateau_frac:.0%} of the grid")
print(f"  contiguous plateau containing the chosen value: "
      f"lookback {_grid[_lo]}..{_grid[_hi]} ({_contig} values)")
if _contig >= 5:
    print("  -> the chosen value sits inside a broad plateau, not on a spike")
else:
    print("  -> WARNING: the chosen value sits in a narrow region; small parameter")
    print("     changes move performance a lot, which is a curve-fitting signature")

# %%
print("Local perturbation of the chosen parameter")
_rows = []
for _p in PARAM_PERTURB_PCT:
    _lb = max(2, int(round(_chosen * (1 + _p))))
    _r = strategy_returns_fast(_calib_close, {**STRATEGY_PARAMS, "lookback": _lb},
                               COST_BPS, SLIPPAGE_BPS, EXECUTION_DELAY_BARS)
    _sd = _r.std(ddof=1)
    _rows.append({"perturbation": f"{_p:+.0%}", "lookback": _lb,
                  "sharpe": _r.mean() / _sd * math.sqrt(BPY) if _sd > 0 else np.nan})
PARAM_PERTURB = pd.DataFrame(_rows)
PARAM_PERTURB["vs_base"] = PARAM_PERTURB["sharpe"] - PARAM_PERTURB.loc[
    PARAM_PERTURB["perturbation"] == "+0%", "sharpe"].iloc[0]
print(PARAM_PERTURB.round(4).to_string(index=False))
_spread = PARAM_PERTURB["sharpe"].max() - PARAM_PERTURB["sharpe"].min()
print(f"\nSharpe range across +/-10% of the parameter: {_spread:.4f}")
print(f"relative to the base Sharpe of {SWEEP_CALIB[_ci]:.4f}, that is a "
      f"{_spread/abs(SWEEP_CALIB[_ci]):.0%} swing" if SWEEP_CALIB[_ci] else "")

# %%
# The data-mining tax: sweep the identical grid on synthetic markets.
_sweep_gens = [n for n in ["iid_bootstrap", "iaaft_surrogate", "stationary_L20",
                           "filtered_historical_sim", "trend_preserving"]
               if n in SYNTH_BT][:5]
_n_sweep = int(min(N_SIMULATIONS, 80))
MINING_TAX = {}

print(f"sweeping the full {len(_grid)}-value grid on {_n_sweep} synthetic paths "
      f"for {len(_sweep_gens)} generators")
for _name in _sweep_gens:
    _t0 = _time.time()
    _rng_s = get_rng(f"sweep::{_name}")
    _paths = GENERATORS[_name](_rng_s, _n_sweep, N_STEPS)
    _best, _fixed = [], []
    for _p in _paths:
        _close = 100.0 * np.exp(np.cumsum(_p))
        _sh = sweep_lookback(_close, _grid, STRATEGY_PARAMS, BPY)
        if np.isfinite(_sh).any():
            _best.append(np.nanmax(_sh))
            _fixed.append(_sh[_ci])
    MINING_TAX[_name] = pd.DataFrame({"best": _best, "fixed": _fixed})
    MINING_TAX[_name]["premium"] = MINING_TAX[_name]["best"] - MINING_TAX[_name]["fixed"]
    print(f"  {_name:<28} median premium {MINING_TAX[_name]['premium'].median():+.4f}  "
          f"{_time.time()-_t0:5.1f}s")

_real_premium = _peak - SWEEP_CALIB[_ci]
_rows = []
for _name, _df in MINING_TAX.items():
    _rows.append({
        "generator": _name,
        "median_best_sharpe": float(_df["best"].median()),
        "p95_best_sharpe": float(_df["best"].quantile(0.95)),
        "median_premium": float(_df["premium"].median()),
        "p95_premium": float(_df["premium"].quantile(0.95)),
        "real_best_pctile": percentile_of(_peak, _df["best"]),
        "real_premium_pctile": percentile_of(_real_premium, _df["premium"]),
    })
_mine_cols = ["generator", "median_best_sharpe", "p95_best_sharpe", "median_premium",
              "p95_premium", "real_best_pctile", "real_premium_pctile"]
MINING_SUMMARY = pd.DataFrame(_rows, columns=_mine_cols).set_index("generator")
print(f"\nThe data-mining tax (real optimisation premium = {_real_premium:+.4f})")
print("=" * 108)
print(MINING_SUMMARY.round(4).to_string() if len(MINING_SUMMARY)
      else "  SKIPPED: no generator was available for the sweep")
print("\nmedian_premium is the Sharpe you gain purely by picking the best of")
print(f"{len(_grid)} parameters on a market where nothing extra is there to find.")
print("real_premium_pctile places the real optimisation gain inside that")
print("distribution: a value near 0.5 means the real grid search bought no more")
print("than noise-fitting normally buys, which is the reassuring outcome for a")
print("pre-committed parameter. A value near 1.0 would mean the real peak stands")
print("out even against pure data mining - evidence the parameter matters.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))

ax = axes[0]
ax.plot(_grid, SWEEP_CALIB, color=C["blue"], lw=2.0, label="calibration slice")
ax.plot(_grid, SWEEP_FULL, color=C["orange"], lw=1.5, label="full sample")
if _sweep_gens:
    _mat = []
    _rng_band = get_rng("sweep_band")
    _paths = GENERATORS[_sweep_gens[0]](_rng_band, min(40, _n_sweep), N_STEPS)
    for _p in _paths:
        _mat.append(sweep_lookback(100.0 * np.exp(np.cumsum(_p)), _grid, STRATEGY_PARAMS, BPY))
    _mat = np.array(_mat)
    ax.fill_between(_grid, np.nanpercentile(_mat, 5, axis=0),
                    np.nanpercentile(_mat, 95, axis=0), color=INK["muted"], alpha=0.22,
                    lw=0, label=f"{_sweep_gens[0]} 5-95%")
ax.axvline(_chosen, color=INK["primary"], lw=1.3, ls="--")
ax.text(_chosen, ax.get_ylim()[1], f" chosen={_chosen}", fontsize=8,
        color=INK["primary"], va="top")
ax.axhline(0, color=INK["axis"], lw=0.9)
finish(ax, "Sharpe vs. lookback", xlabel="lookback (bars)", ylabel="Sharpe",
       subtitle="a broad hump is structure; a lone spike is a fitted coincidence")

ax = axes[1]
if MINING_TAX:
    for _i, (_name, _df) in enumerate(MINING_TAX.items()):
        ax.hist(_df["premium"], bins=22, histtype="step", lw=1.7,
                color=SERIES[_i], label=_name, density=True)
    ax.axvline(_real_premium, color=INK["primary"], lw=1.8, ls="--")
    ax.text(_real_premium, ax.get_ylim()[1], f" real {_real_premium:+.2f}",
            fontsize=8, color=INK["primary"], va="top")
finish(ax, "Optimisation premium on synthetic markets",
       xlabel="best-in-grid Sharpe minus fixed-parameter Sharpe", ylabel="density",
       subtitle="how much grid search buys where there is nothing extra to find")
fig.tight_layout()
plt.show()

# %%
# Two-dimensional surface: are there performance cliffs?
_cb_grid = [1, 2, 3, 4, 5]
_lb_grid = [x for x in _grid if x % 2 == 0][:28]
_surface = np.full((len(_cb_grid), len(_lb_grid)), np.nan)
for _i, _cb in enumerate(_cb_grid):
    for _j, _lb in enumerate(_lb_grid):
        _r = strategy_returns_fast(_calib_close,
                                   {**STRATEGY_PARAMS, "lookback": int(_lb), "confirm_bars": int(_cb)},
                                   COST_BPS, SLIPPAGE_BPS, EXECUTION_DELAY_BARS)
        _sd = _r.std(ddof=1)
        _surface[_i, _j] = _r.mean() / _sd * math.sqrt(BPY) if _sd > 0 else np.nan

fig, ax = plt.subplots(figsize=(11, 3.2))
_vmax = np.nanmax(np.abs(_surface))
_div = mpl.colors.LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#6da7ec", "#f0efec", "#e88a89", "#c22d2d"])
_im = ax.imshow(_surface, aspect="auto", cmap=_div, vmin=-_vmax, vmax=_vmax, origin="lower")
ax.set_xticks(range(0, len(_lb_grid), 2))
ax.set_xticklabels([_lb_grid[i] for i in range(0, len(_lb_grid), 2)], fontsize=8)
ax.set_yticks(range(len(_cb_grid)))
ax.set_yticklabels(_cb_grid, fontsize=8)
ax.scatter([min(range(len(_lb_grid)), key=lambda i: abs(_lb_grid[i] - _chosen))],
           [_cb_grid.index(int(STRATEGY_PARAMS.get("confirm_bars", 2)))],
           marker="o", s=70, facecolor="none", edgecolor=INK["primary"], linewidth=2, zorder=5)
ax.set_title("Sharpe surface: lookback x confirm_bars (circle = chosen setting)", loc="left")
ax.set_xlabel("lookback (bars)")
ax.set_ylabel("confirm bars")
ax.grid(False)
fig.colorbar(_im, ax=ax, shrink=0.9, label="Sharpe")
fig.tight_layout()
plt.show()
print("A diverging scale is used because the sign of the Sharpe is the meaningful")
print("boundary here: blue is losing, red is winning, and the neutral midpoint is")
print("zero. Sharp colour changes between neighbouring cells are performance")
print("cliffs - settings where a one-bar change flips the result.")

# %% [markdown]
# ---
# ## 15 - Walk-forward testing (overfitting test 3)
#
# The strictest protocol in the notebook, and the one closest to how a strategy
# would actually be deployed. For each fold:
#
# 1. **Train** - grid-search the parameter, keeping the top candidates.
# 2. **Validate** - among those candidates only, pick the one that does best on
#    a later, disjoint slice. This second step matters: the single best
#    in-sample parameter is usually the luckiest rather than the best, and
#    validating a shortlist discards most of that luck.
# 3. **Test** - evaluate once on a third slice that neither step saw.
#
# Test data is never used for selection. The fold's test result is recorded and
# the window moves on.
#
# Two schemes are compared. **Expanding** keeps all history, so it has more data
# but can be anchored to a regime that no longer exists. **Rolling** uses a fixed
# recent window, so it adapts but throws information away. Which wins is an
# empirical question about this market, not a matter of principle.
#
# The headline number is the **efficiency ratio**: out-of-sample Sharpe divided
# by in-sample Sharpe. A value near 1 means the in-sample result transferred. A
# value near 0 means the in-sample result was selection noise.

# %%
def walk_forward(close: np.ndarray, splits: list[Split], grid, params: dict, bpy: float,
                 top_k: int = 5, val_frac: float = 0.3) -> pd.DataFrame:
    """Train -> validate -> test walk-forward with strict slice isolation.

    The training range of each split is itself divided in time: the earlier part
    grid-searches, the later part chooses among the shortlist. The test range is
    touched exactly once, for evaluation.
    """
    rows = []
    for sp in splits:
        tr_lo, tr_hi = sp.train
        n_train = tr_hi - tr_lo
        if n_train < 150:
            continue
        cut = tr_lo + int(n_train * (1 - val_frac))
        train_close = close[tr_lo:cut]
        val_close = close[cut:tr_hi]
        test_close = close[sp.test[0]:sp.test[1]]
        if min(len(train_close), len(val_close), len(test_close)) < 60:
            continue

        sh_train = sweep_lookback(train_close, grid, params, bpy)
        if not np.isfinite(sh_train).any():
            continue
        order = np.argsort(np.where(np.isfinite(sh_train), sh_train, -np.inf))[::-1]
        shortlist = [grid[i] for i in order[:top_k]]

        sh_val = [sweep_lookback(val_close, [lb], params, bpy)[0] for lb in shortlist]
        pick = shortlist[int(np.nanargmax(np.where(np.isfinite(sh_val), sh_val, -np.inf)))]

        is_sharpe = float(sh_train[grid.index(pick)]) if pick in grid else np.nan
        oos = sweep_lookback(test_close, [pick], params, bpy)[0]
        fixed_oos = sweep_lookback(test_close, [int(params["lookback"])], params, bpy)[0]

        rows.append({"fold": sp.name, "picked_lookback": pick,
                     "is_sharpe": is_sharpe,
                     "val_sharpe": float(np.nanmax(sh_val)),
                     "oos_sharpe": float(oos),
                     "oos_sharpe_fixed": float(fixed_oos),
                     "n_train": n_train, "n_test": sp.test[1] - sp.test[0]})
    return pd.DataFrame(rows)


WF_RESULTS = {}
for _scheme, _splits in [("expanding", WF_EXPANDING), ("rolling", WF_ROLLING)]:
    WF_RESULTS[_scheme] = walk_forward(_full_close, _splits, _grid, STRATEGY_PARAMS, BPY)
    _df = WF_RESULTS[_scheme]
    if _df.empty:
        print(f"{_scheme}: no usable folds")
        continue
    _eff = (_df["oos_sharpe"].mean() / _df["is_sharpe"].mean()
            if _df["is_sharpe"].mean() != 0 else np.nan)
    print(f"\n--- {_scheme} walk-forward ({len(_df)} folds)")
    print(_df.round(4).to_string(index=False))
    print(f"  mean in-sample Sharpe      {_df['is_sharpe'].mean():+.4f}")
    print(f"  mean out-of-sample Sharpe  {_df['oos_sharpe'].mean():+.4f}")
    print(f"  efficiency ratio (OOS/IS)  {_eff:.3f}")
    print(f"  fixed lookback={STRATEGY_PARAMS['lookback']} OOS Sharpe "
          f"{_df['oos_sharpe_fixed'].mean():+.4f}")
    print(f"  re-optimising beat the fixed parameter in "
          f"{int((_df['oos_sharpe'] > _df['oos_sharpe_fixed']).sum())}/{len(_df)} folds")
    print(f"  parameters chosen: {sorted(_df['picked_lookback'].tolist())}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

ax = axes[0]
for _i, (_scheme, _df) in enumerate(WF_RESULTS.items()):
    if _df.empty:
        continue
    _x = np.arange(len(_df))
    ax.plot(_x, _df["is_sharpe"], marker="o", ms=5, color=SERIES[_i], lw=1.6,
            label=f"{_scheme} in-sample")
    ax.plot(_x, _df["oos_sharpe"], marker="s", ms=5, color=SERIES[_i], lw=1.6,
            ls="--", label=f"{_scheme} out-of-sample")
ax.axhline(0, color=INK["axis"], lw=0.9)
finish(ax, "In-sample vs out-of-sample by fold", xlabel="fold", ylabel="Sharpe",
       subtitle="the gap between solid and dashed is the selection premium")

ax = axes[1]
_all = pd.concat([d.assign(scheme=s) for s, d in WF_RESULTS.items() if not d.empty])
if len(_all):
    ax.scatter(_all["is_sharpe"], _all["oos_sharpe"], s=42, color=C["blue"],
               alpha=0.8, label="fold")
    _lim = [min(_all[["is_sharpe", "oos_sharpe"]].min()) - 0.2,
            max(_all[["is_sharpe", "oos_sharpe"]].max()) + 0.2]
    ax.plot(_lim, _lim, color=INK["muted"], ls="--", lw=1.2, label="perfect transfer")
    ax.axhline(0, color=INK["axis"], lw=0.9)
    ax.axvline(0, color=INK["axis"], lw=0.9)
finish(ax, "Does in-sample performance transfer?", xlabel="in-sample Sharpe",
       ylabel="out-of-sample Sharpe",
       subtitle="points below the diagonal lost performance out of sample")
fig.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 16 - Execution robustness
#
# A strategy that only works at zero cost does not work. This section applies
# realistic frictions and reports where the edge disappears.
#
# The break-even cost is the most decision-relevant number in the notebook: it
# converts an abstract Sharpe into "how good does my execution have to be for
# this to be worth trading." Comparing it against the actual cost of trading the
# instrument answers the question directly.

# %%
_rows = []
for _c in COST_GRID_BPS:
    _r = strategy_returns_fast(_calib_close, STRATEGY_PARAMS, _c, 0.0, EXECUTION_DELAY_BARS)
    _sd = _r.std(ddof=1)
    _rows.append({"cost_bps": _c,
                  "sharpe": _r.mean() / _sd * math.sqrt(BPY) if _sd > 0 else np.nan,
                  "ann_return": float(np.expm1(np.log1p(_r).sum() / (len(_r) / BPY)))})
COST_SENSITIVITY = pd.DataFrame(_rows)

# Break-even cost: bisect on the cost that drives the Sharpe to zero.
def _sharpe_at_cost(c):
    r = strategy_returns_fast(_calib_close, STRATEGY_PARAMS, c, 0.0, EXECUTION_DELAY_BARS)
    sd = r.std(ddof=1)
    return r.mean() / sd * math.sqrt(BPY) if sd > 0 else np.nan

_lo_c, _hi_c = 0.0, 500.0
if _sharpe_at_cost(_lo_c) <= 0:
    BREAKEVEN_BPS = 0.0
elif _sharpe_at_cost(_hi_c) > 0:
    BREAKEVEN_BPS = float("inf")
else:
    for _ in range(40):
        _mid = 0.5 * (_lo_c + _hi_c)
        if _sharpe_at_cost(_mid) > 0:
            _lo_c = _mid
        else:
            _hi_c = _mid
    BREAKEVEN_BPS = 0.5 * (_lo_c + _hi_c)

print("Cost sensitivity (calibration slice, no slippage on top)")
print(COST_SENSITIVITY.round(4).to_string(index=False))
print(f"\nbreak-even round-trip cost: {BREAKEVEN_BPS:.1f} bps per unit turnover")
print(f"configured cost + slippage:  {COST_BPS + SLIPPAGE_BPS:.1f} bps")
print(f"annual turnover:             {HIST_CALIB.metrics['turnover_ann']:.1f}x")
if np.isfinite(BREAKEVEN_BPS):
    _margin = BREAKEVEN_BPS / max(1e-9, COST_BPS + SLIPPAGE_BPS)
    print(f"margin of safety:            {_margin:.1f}x the assumed cost")
    if _margin < 2:
        print("  -> thin. Small errors in the cost assumption change the conclusion.")

# %%
_rows = []
for _d in EXECUTION_DELAY_GRID:
    _r = strategy_returns_fast(_calib_close, STRATEGY_PARAMS, COST_BPS, SLIPPAGE_BPS, _d)
    _sd = _r.std(ddof=1)
    _rows.append({"delay_bars": _d,
                  "sharpe": _r.mean() / _sd * math.sqrt(BPY) if _sd > 0 else np.nan})
DELAY_SENSITIVITY = pd.DataFrame(_rows)

_rng_noise = get_rng("signal_noise")
_rows = []
for _q in SIGNAL_NOISE_GRID:
    _sh = []
    for _rep in range(12 if _q > 0 else 1):
        _sig, _pos = sma_positions(_calib_close, int(STRATEGY_PARAMS["lookback"]),
                                   int(STRATEGY_PARAMS.get("confirm_bars", 2)),
                                   STRATEGY_PARAMS.get("direction", "long"))
        if _q > 0:
            # With probability q the update is missed and the previous position
            # is carried - a direct model of a signal that is not acted on.
            _seen = _rng_noise.random(len(_pos)) >= _q
            _p = np.where(_seen, _pos, np.nan)
            _pos = np.nan_to_num(_ffill(_p), nan=0.0)
        _e = apply_execution(_calib_close, _pos, COST_BPS, SLIPPAGE_BPS, EXECUTION_DELAY_BARS)
        _r = _e["strategy_return"]
        _sd = _r.std(ddof=1)
        _sh.append(_r.mean() / _sd * math.sqrt(BPY) if _sd > 0 else np.nan)
    _rows.append({"missed_signal_rate": _q, "mean_sharpe": float(np.nanmean(_sh)),
                  "min_sharpe": float(np.nanmin(_sh))})
SIGNAL_NOISE = pd.DataFrame(_rows)

print("Execution delay")
print(DELAY_SENSITIVITY.round(4).to_string(index=False))
print("\nSignal noise (probability a position update is missed)")
print(SIGNAL_NOISE.round(4).to_string(index=False))

# %%
# Data perturbation: does the result depend on exact historical values?
_rng_pert = get_rng("data_perturbation")
_base_sharpe = SWEEP_CALIB[_ci]
_rows = []

for _eps in (0.0001, 0.0005, 0.001, 0.005):
    _sh = []
    for _ in range(10):
        _c = _calib_close * (1 + _rng_pert.normal(0, _eps, len(_calib_close)))
        _sh.append(sweep_lookback(_c, [_chosen], STRATEGY_PARAMS, BPY)[0])
    _rows.append({"perturbation": f"price noise {_eps:.2%}",
                  "mean_sharpe": float(np.nanmean(_sh)), "sd": float(np.nanstd(_sh))})

for _drop in (0.01, 0.05, 0.10):
    _sh = []
    for _ in range(10):
        _keep = _rng_pert.random(len(_calib_close)) >= _drop
        _keep[0] = _keep[-1] = True
        _sh.append(sweep_lookback(_calib_close[_keep], [_chosen], STRATEGY_PARAMS, BPY)[0])
    _rows.append({"perturbation": f"drop {_drop:.0%} of bars",
                  "mean_sharpe": float(np.nanmean(_sh)), "sd": float(np.nanstd(_sh))})

DATA_PERTURB = pd.DataFrame(_rows)
DATA_PERTURB["vs_base"] = DATA_PERTURB["mean_sharpe"] - _base_sharpe
print(f"Data perturbation (base Sharpe {_base_sharpe:+.4f})")
print(DATA_PERTURB.round(4).to_string(index=False))
print("\nA strategy whose result survives small price noise and randomly missing")
print("bars is reading a broad feature of the data. One that collapses is keyed to")
print("exact historical values, which will not repeat.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

ax = axes[0]
ax.plot(COST_SENSITIVITY["cost_bps"], COST_SENSITIVITY["sharpe"], marker="o", ms=5,
        color=C["blue"], lw=1.9)
ax.axhline(0, color=INK["axis"], lw=1.0)
if np.isfinite(BREAKEVEN_BPS) and BREAKEVEN_BPS < max(COST_GRID_BPS) * 1.5:
    ax.axvline(BREAKEVEN_BPS, color=STATUS["critical"], lw=1.5, ls="--")
    ax.text(BREAKEVEN_BPS, ax.get_ylim()[1], f" break-even {BREAKEVEN_BPS:.0f}bps",
            fontsize=8, color=STATUS["critical"], va="top")
ax.axvline(COST_BPS + SLIPPAGE_BPS, color=INK["primary"], lw=1.3, ls=":")
finish(ax, "Sharpe vs. transaction cost", xlabel="cost (bps per unit turnover)",
       ylabel="Sharpe", legend=False, subtitle="dotted line = the assumed cost")

ax = axes[1]
ax.plot(DELAY_SENSITIVITY["delay_bars"], DELAY_SENSITIVITY["sharpe"], marker="o",
        ms=6, color=C["orange"], lw=1.9)
ax.axhline(0, color=INK["axis"], lw=1.0)
ax.set_xticks(list(EXECUTION_DELAY_GRID))
finish(ax, "Sharpe vs. execution delay", xlabel="extra bars between signal and fill",
       ylabel="Sharpe", legend=False)

ax = axes[2]
ax.plot(SIGNAL_NOISE["missed_signal_rate"], SIGNAL_NOISE["mean_sharpe"], marker="o",
        ms=6, color=C["aqua"], lw=1.9)
ax.axhline(0, color=INK["axis"], lw=1.0)
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
finish(ax, "Sharpe vs. missed signals", xlabel="probability a position update is missed",
       ylabel="Sharpe", legend=False)
fig.tight_layout()
plt.show()
