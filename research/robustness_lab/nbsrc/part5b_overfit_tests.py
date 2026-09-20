# %% [markdown]
# ---
# ## 17 - Placebo tests
#
# A Sharpe of 0.6 sounds like skill until you learn that randomly-timed trades
# with the same exposure would have earned 0.5 on the same market. Placebos
# supply that missing comparison.
#
# The design point that makes a placebo informative is **matching**. A control
# that trades less, or holds less, or trades at different times of day, loses
# for reasons that have nothing to do with signal quality. Each placebo below
# therefore preserves something specific about the real strategy and destroys
# only the thing being tested:
#
# | placebo | preserves | destroys | isolates |
# |---|---|---|---|
# | **block shuffle** | exposure, trade count, every trade duration | alignment between trades and price | whether the *timing* carried information |
# | **random relocation** | trade count and durations | entry times entirely | same, with freely chosen entries |
# | **delayed signal** | the rule itself, exactly | timeliness (acts k bars late) | how perishable the signal is |
# | **random parameter** | the rule's form | the specific parameter choice | whether the chosen parameter mattered |
#
# The block shuffle is the sharpest of the four. It takes the real position
# series, cuts it at every position change, and reorders those runs. The
# strategy's exposure, its number of trades and the length of every single trade
# are preserved exactly; only their placement against the price series changes.
# If the real strategy cannot beat that, its entry timing carried no information.

# %%
def position_runs(pos: np.ndarray) -> list[np.ndarray]:
    """Cut a position series at every change into constant-value runs."""
    pos = np.asarray(pos, dtype=float)
    if len(pos) == 0:
        return []
    change = np.empty(len(pos), dtype=bool)
    change[0] = True
    change[1:] = pos[1:] != pos[:-1]
    idx = np.flatnonzero(change)
    return [pos[a:b] for a, b in zip(idx, np.append(idx[1:], len(pos)))]


def placebo_block_shuffle(pos: np.ndarray, rng) -> np.ndarray:
    """Reorder the position runs. Exposure and trade count are preserved exactly."""
    runs = position_runs(pos)
    order = rng.permutation(len(runs))
    return np.concatenate([runs[i] for i in order])[:len(pos)]


def placebo_random_relocate(pos: np.ndarray, rng) -> np.ndarray:
    """Keep every trade's duration and side, but place them at random times."""
    pos = np.asarray(pos, dtype=float)
    out = np.zeros_like(pos)
    runs = [r for r in position_runs(pos) if r[0] != 0]
    n = len(pos)
    for r in runs:
        d = len(r)
        if d >= n:
            continue
        for _ in range(20):                      # retry on collision, then give up
            s = int(rng.integers(0, n - d))
            if np.all(out[s:s + d] == 0):
                out[s:s + d] = r[0]
                break
    return out


def placebo_delayed(pos: np.ndarray, k: int) -> np.ndarray:
    """The real signal, acted on k bars late."""
    out = np.zeros_like(pos)
    if k < len(pos):
        out[k:] = pos[:-k] if k > 0 else pos
    return out


def _sharpe_of_positions(close, pos, bpy):
    e = apply_execution(close, pos, COST_BPS, SLIPPAGE_BPS, EXECUTION_DELAY_BARS)
    r = e["strategy_return"]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(bpy)) if sd > 0 else np.nan


_rng_pl = get_rng("placebo")
_sig_real, _pos_real = sma_positions(_calib_close, int(STRATEGY_PARAMS["lookback"]),
                                     int(STRATEGY_PARAMS.get("confirm_bars", 2)),
                                     STRATEGY_PARAMS.get("direction", "long"))
_real_sharpe = _sharpe_of_positions(_calib_close, _pos_real, BPY)
_n_placebo = int(min(N_SIMULATIONS * 4, 600))

PLACEBOS = {}
PLACEBOS["block_shuffle"] = np.array([
    _sharpe_of_positions(_calib_close, placebo_block_shuffle(_pos_real, _rng_pl), BPY)
    for _ in range(_n_placebo)])
PLACEBOS["random_relocate"] = np.array([
    _sharpe_of_positions(_calib_close, placebo_random_relocate(_pos_real, _rng_pl), BPY)
    for _ in range(min(_n_placebo, 200))])
PLACEBOS["random_parameter"] = np.array([
    sweep_lookback(_calib_close, [int(_rng_pl.choice(_grid))], STRATEGY_PARAMS, BPY)[0]
    for _ in range(min(_n_placebo, 300))])

_delay_rows = []
for _k in (1, 3, 5, 10, 20):
    _delay_rows.append({"delay_bars": _k,
                        "sharpe": _sharpe_of_positions(_calib_close,
                                                       placebo_delayed(_pos_real, _k), BPY)})
PLACEBO_DELAY = pd.DataFrame(_delay_rows)

_rows = []
for _name, _arr in PLACEBOS.items():
    _a = pd.Series(_arr).replace([np.inf, -np.inf], np.nan).dropna()
    _rows.append({
        "placebo": _name, "n": len(_a), "median": float(_a.median()),
        "p5": float(_a.quantile(0.05)), "p95": float(_a.quantile(0.95)),
        "real_pctile": percentile_of(_real_sharpe, _a),
        "p_value": float(1 - percentile_of(_real_sharpe, _a)),
    })
PLACEBO_SUMMARY = pd.DataFrame(_rows).set_index("placebo")
PLACEBO_REAL_SHARPE = float(_real_sharpe)

print(f"Real strategy Sharpe (calibration slice): {_real_sharpe:+.4f}")
print(f"exposure {np.mean(_pos_real != 0):.1%}, {len(extract_trades(_pos_real, np.zeros_like(_pos_real)))} trades\n")
print("Placebo controls")
print(PLACEBO_SUMMARY.round(4).to_string())
print("\nDelayed-signal placebo (the real rule, acted on late)")
print(PLACEBO_DELAY.round(4).to_string(index=False))
print(f"\nThe block shuffle holds exposure and every trade duration fixed and changes")
print(f"only WHEN the trades happen. Its p-value of "
      f"{PLACEBO_SUMMARY.loc['block_shuffle','p_value']:.4f} is therefore a direct")
print("test of whether the entry timing carried information.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7))

ax = axes[0]
for _i, (_name, _arr) in enumerate(PLACEBOS.items()):
    _a = pd.Series(_arr).replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(_a, bins=30, histtype="step", lw=1.8, color=SERIES[_i], density=True,
            label=f"{_name} (p={PLACEBO_SUMMARY.loc[_name,'p_value']:.3f})")
ax.axvline(_real_sharpe, color=INK["primary"], lw=2.0, ls="--")
ax.text(_real_sharpe, ax.get_ylim()[1], f" real {_real_sharpe:+.2f}", fontsize=8.5,
        color=INK["primary"], va="top")
finish(ax, "Placebo Sharpe distributions", xlabel="Sharpe", ylabel="density",
       subtitle="each control preserves a different property of the real strategy")

ax = axes[1]
ax.plot(PLACEBO_DELAY["delay_bars"], PLACEBO_DELAY["sharpe"], marker="o", ms=6,
        color=C["orange"], lw=1.9, label="delayed signal")
ax.axhline(_real_sharpe, color=INK["primary"], lw=1.6, ls="--", label="real (no delay)")
ax.axhline(0, color=INK["axis"], lw=1.0)
finish(ax, "Signal perishability", xlabel="bars of delay before acting",
       ylabel="Sharpe",
       subtitle="a fast decay means the edge is timing-sensitive and execution matters")
fig.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 18 - Monte Carlo trade resampling
#
# This resamples the **sequence of trades that actually happened** and rebuilds
# the equity curve many times over.
#
# It is important to be precise about what this does and does not answer,
# because it is routinely over-interpreted:
#
# > **It answers:** given that these trades occurred, how much did their
# > particular ORDER matter? How different could the drawdown have looked?
# >
# > **It does not answer:** would these trades have occurred at all in a
# > different market history?
#
# The second question is the one that matters for overfitting, and only the
# synthetic-market tests in Section 13 address it. Trade resampling holds the
# trade population fixed - it takes the strategy's own results as given and
# reshuffles them - so it cannot detect a strategy whose trades were a product
# of one particular path. A strategy can look wonderfully stable here and still
# be completely overfit.
#
# What it is genuinely good for is risk: the distribution of maximum drawdown
# across orderings is a far better guide to the pain to expect than the single
# historical drawdown, which is one draw from exactly this distribution.
#
# Both an IID and a block resample are run. The IID version assumes trades are
# independent; the block version keeps short runs of consecutive trades
# together, which respects the fact that consecutive trades share a market
# regime. Where the two disagree, the block version is the more honest.

# %%
_trades = HIST_CALIB.trades
_tr = _trades["return"].to_numpy() if len(_trades) else np.array([])
print(f"{len(_tr)} closed trades on the calibration slice, "
      f"mean {_tr.mean():+.3%}, sd {_tr.std(ddof=1):.3%}" if len(_tr) else "no trades")


def resample_trades(tr: np.ndarray, n_sims: int, rng, block: int = 1) -> dict:
    """Rebuild equity curves from resampled trade sequences.

    block=1 resamples trades independently. block>1 draws contiguous runs,
    preserving the local clustering that comes from consecutive trades sharing
    a market regime.
    """
    n = len(tr)
    if n < 10:
        raise ValueError(f"need at least 10 trades to resample, got {n}")
    if block <= 1:
        draws = rng.choice(tr, size=(n_sims, n), replace=True)
    else:
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, max(1, n - block + 1), size=(n_sims, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :])
        idx = np.clip(idx.reshape(n_sims, -1)[:, :n], 0, n - 1)
        draws = tr[idx]

    eq = np.cumprod(1.0 + draws, axis=1)
    final = eq[:, -1]
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = (eq / peak - 1.0).min(axis=1)
    logr = np.log1p(np.clip(draws, -0.999999, None))
    sd = logr.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_per_trade = np.where(sd > 0, logr.mean(axis=1) / sd, np.nan)
    return {"equity": eq, "final": final, "max_dd": mdd,
            "sharpe_per_trade": sharpe_per_trade, "draws": draws}


if len(_tr) >= 10:
    _rng_mc = get_rng("trade_monte_carlo")
    MC_IID = resample_trades(_tr, N_MC_TRADE_RESAMPLES, _rng_mc, block=1)
    MC_BLOCK = resample_trades(_tr, N_MC_TRADE_RESAMPLES, _rng_mc, block=5)

    _n_years = HIST_CALIB.metrics["n_bars"] / BPY
    _rows = []
    for _nm, _mc in [("iid", MC_IID), ("block-5", MC_BLOCK)]:
        _cagr = np.expm1(np.log(np.maximum(_mc["final"], 1e-9)) / _n_years)
        _rows.append({
            "resample": _nm,
            "median_final": float(np.median(_mc["final"])),
            "p5_final": float(np.percentile(_mc["final"], 5)),
            "p95_final": float(np.percentile(_mc["final"], 95)),
            "median_cagr": float(np.median(_cagr)),
            "p5_cagr": float(np.percentile(_cagr, 5)),
            "median_maxdd": float(np.median(_mc["max_dd"])),
            "p95_maxdd": float(np.percentile(_mc["max_dd"], 5)),
            "p_loss": float(np.mean(_mc["final"] < 1.0)),
            "p_dd_worse_than_hist": float(np.mean(_mc["max_dd"] < -HIST_CALIB.metrics["max_dd"])),
            "p_ruin_50pct": float(np.mean(_mc["max_dd"] <= -0.50)),
        })
    MC_SUMMARY = pd.DataFrame(_rows).set_index("resample")
    print("\nTrade-order Monte Carlo")
    print(MC_SUMMARY.round(4).to_string())
    print(f"\nhistorical: final {HIST_CALIB.metrics['total_return']+1:.3f}x, "
          f"max drawdown {-HIST_CALIB.metrics['max_dd']:.1%}")
    print(f"\n{MC_SUMMARY.loc['block-5','p_dd_worse_than_hist']:.1%} of reorderings of the")
    print("SAME trades produced a worse drawdown than the one that actually happened.")
    print("That is the honest read on the historical drawdown: it is one draw, not a")
    print("worst case, and planning around it alone would understate the risk.")
else:
    MC_SUMMARY = pd.DataFrame()
    MC_IID = MC_BLOCK = None
    print("\nSKIPPED: fewer than 10 trades, so trade resampling is not meaningful.")

# %%
if MC_BLOCK is not None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    ax = axes[0]
    _rng_show = get_rng("mc_display")
    _pick = _rng_show.choice(len(MC_BLOCK["equity"]), size=60, replace=False)
    for _i in _pick:
        ax.plot(MC_BLOCK["equity"][_i], color=C["blue"], lw=0.5, alpha=0.18)
    _hist_eq = np.cumprod(1 + _tr)
    ax.plot(_hist_eq, color=INK["primary"], lw=2.0, label="actual order", zorder=5)
    ax.plot(np.median(MC_BLOCK["equity"], axis=0), color=C["orange"], lw=1.8,
            label="median reordering", zorder=4)
    ax.set_yscale("log")
    finish(ax, "Equity under reordered trades", xlabel="trade number",
           ylabel="growth of 1 (log)", subtitle="60 of "
           f"{N_MC_TRADE_RESAMPLES} shown")

    ax = axes[1]
    ax.hist(MC_BLOCK["max_dd"] * 100, bins=45, color=C["blue"], alpha=0.8)
    ax.axvline(-HIST_CALIB.metrics["max_dd"] * 100, color=INK["primary"], lw=1.9, ls="--")
    ax.text(-HIST_CALIB.metrics["max_dd"] * 100, ax.get_ylim()[1], " actual",
            fontsize=8.5, color=INK["primary"], va="top")
    finish(ax, "Max drawdown distribution", xlabel="max drawdown (%)", ylabel="paths",
           legend=False, subtitle="the historical value is one draw from this")

    ax = axes[2]
    ax.hist(MC_BLOCK["final"], bins=45, color=C["aqua"], alpha=0.85)
    ax.axvline(1.0, color=STATUS["critical"], lw=1.5, ls=":")
    ax.axvline(HIST_CALIB.metrics["total_return"] + 1, color=INK["primary"], lw=1.9, ls="--")
    ax.set_xscale("log")
    finish(ax, "Final equity distribution", xlabel="growth multiple (log)", ylabel="paths",
           legend=False, subtitle="dotted line = breakeven")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ## 19 - Multiple testing, PBO and the Deflated Sharpe Ratio
#
# Testing 56 parameter values and reporting the best one is not the same as
# testing one parameter and finding it works. The best of 56 draws from a
# distribution centred on zero is comfortably positive, every time. This section
# quantifies that.
#
# Four corrections, each answering a different question:
#
# **Bonferroni** controls the chance of *any* false positive across the family.
# It is severe and assumes independence, which is badly violated here - a
# 17-bar and an 18-bar moving average are nearly the same strategy, so the 56
# tests are nothing like 56 independent ones. Bonferroni is therefore reported
# as a conservative bound, not as the answer.
#
# **Benjamini-Hochberg** controls the expected *proportion* of false discoveries
# instead, which is the more sensible target when the tests are correlated and
# you expect several genuine effects.
#
# **The Deflated Sharpe Ratio** asks directly: given that N variants were tried,
# given the sample length, and given that these returns are skewed and
# fat-tailed, what is the probability the true Sharpe exceeds zero? It corrects
# for all three at once. The skew and kurtosis adjustment matters a great deal
# here - the standard Sharpe significance test assumes normal returns, and these
# are anything but, which makes the naive test far too permissive.
#
# **PBO** (Probability of Backtest Overfitting, via Combinatorially Symmetric
# Cross-Validation) asks the sharpest question of all: when you pick the best
# variant in-sample, how often does it land below median out-of-sample? If that
# happens more than half the time, your selection procedure is worse than
# choosing at random, and the in-sample ranking is pure noise.

# %%
def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, sr_variance: float,
                          bpy: float) -> dict:
    """Bailey & Lopez de Prado's Deflated Sharpe Ratio.

    All Sharpe quantities are handled in PER-BAR units and only annualised for
    display. Mixing annualised and per-bar Sharpes inside these formulas is the
    standard way to get an answer that is wrong by a factor of sqrt(bpy).

    n_trials     : how many strategy variants were tried (here, the grid size).
    sr_variance  : variance of the per-bar Sharpe ACROSS those trials, which
                   sets how high the best of N would be expected to reach by
                   chance alone.

    Returns the expected maximum Sharpe under the null, the deflated
    probability that the true Sharpe is positive, and the minimum track record
    length needed for significance.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    sd = r.std(ddof=1)
    if T < 30 or sd <= 0:
        return {k: np.nan for k in ("sr_per_bar", "sr_ann", "sr0_per_bar", "sr0_ann",
                                    "dsr", "min_trl_bars", "min_trl_years",
                                    "skew", "kurtosis")}
    sr = r.mean() / sd
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))      # NON-excess kurtosis

    # Expected maximum of N independent draws from N(0, sr_variance).
    gamma = 0.5772156649015329                        # Euler-Mascheroni
    N = max(2, int(n_trials))
    z1 = stats.norm.ppf(1.0 - 1.0 / N)
    z2 = stats.norm.ppf(1.0 - 1.0 / (N * math.e))
    sr0 = math.sqrt(max(sr_variance, 0.0)) * ((1 - gamma) * z1 + gamma * z2)

    denom_sq = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denom_sq <= 0:
        dsr = np.nan
    else:
        dsr = float(stats.norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom_sq)))

    if sr > sr0 and denom_sq > 0:
        min_trl = 1 + denom_sq * (stats.norm.ppf(1 - ALPHA) / (sr - sr0)) ** 2
    else:
        min_trl = np.inf

    return {"sr_per_bar": float(sr), "sr_ann": float(sr * math.sqrt(bpy)),
            "sr0_per_bar": float(sr0), "sr0_ann": float(sr0 * math.sqrt(bpy)),
            "dsr": dsr, "min_trl_bars": float(min_trl),
            "min_trl_years": float(min_trl / bpy) if np.isfinite(min_trl) else np.inf,
            "skew": g3, "kurtosis": g4}


# Per-bar Sharpes across the whole parameter grid set the trial variance.
_sweep_per_bar = np.array(SWEEP_CALIB) / math.sqrt(BPY)
_sr_var = float(np.nanvar(_sweep_per_bar, ddof=1))
_r_chosen = strategy_returns_fast(_calib_close, STRATEGY_PARAMS, COST_BPS,
                                  SLIPPAGE_BPS, EXECUTION_DELAY_BARS)
DSR = deflated_sharpe_ratio(_r_chosen, len(_grid), _sr_var, BPY)

print("Deflated Sharpe Ratio")
print(f"  trials (grid size)          {len(_grid)}")
print(f"  observed Sharpe (ann.)      {DSR['sr_ann']:+.4f}")
print(f"  expected max under null     {DSR['sr0_ann']:+.4f}  (ann.) - what the BEST of")
print(f"                              {len(_grid)} variants reaches by chance alone")
print(f"  return skew / kurtosis      {DSR['skew']:+.3f} / {DSR['kurtosis']:.2f}")
print(f"  DEFLATED probability        {DSR['dsr']:.4f}  that the true Sharpe > 0")
print(f"  minimum track record        {DSR['min_trl_years']:.1f} years needed for "
      f"significance at alpha={ALPHA}")
print(f"  actual track record         {len(_r_chosen)/BPY:.1f} years")
if np.isfinite(DSR["min_trl_years"]):
    print("  -> " + ("the record is long enough" if DSR["min_trl_years"] <= len(_r_chosen)/BPY
                     else "the record is TOO SHORT to establish significance"))
else:
    print("  -> the observed Sharpe does not exceed the expected maximum under the")
    print("     null, so no track record length would make it significant")

# %%
# Bonferroni and Benjamini-Hochberg across the parameter grid.
_T = len(_r_chosen)
_pvals = []
for _i, _lb in enumerate(_grid):
    _s = _sweep_per_bar[_i]
    if not np.isfinite(_s):
        _pvals.append(np.nan)
        continue
    _t = _s * math.sqrt(_T)                       # t-statistic for mean return > 0
    _pvals.append(float(1.0 - stats.norm.cdf(_t)))
_pvals = np.array(_pvals)


def benjamini_hochberg(p: np.ndarray, alpha: float) -> np.ndarray:
    """Return a boolean mask of hypotheses surviving BH FDR control."""
    ok = np.isfinite(p)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    out = np.zeros(len(p), dtype=bool)
    if m == 0:
        return out
    thresh = alpha * np.arange(1, m + 1) / m
    passing = np.flatnonzero(p[order] <= thresh)
    if len(passing):
        out[order[:passing[-1] + 1]] = True
    return out


_raw_sig = np.nansum(_pvals < ALPHA)
_bonf_sig = np.nansum(_pvals < ALPHA / len(_grid))
_bh_mask = benjamini_hochberg(_pvals, FDR_ALPHA)

MULTIPLE_TESTING = {
    "n_trials": len(_grid), "pvals": _pvals, "bh_mask": _bh_mask,
    "n_raw_sig": int(_raw_sig), "n_bonferroni_sig": int(_bonf_sig),
    "n_bh_sig": int(_bh_mask.sum()), "chosen_p": float(_pvals[_ci]),
    "chosen_survives_bonferroni": bool(_pvals[_ci] < ALPHA / len(_grid)),
    "chosen_survives_bh": bool(_bh_mask[_ci]),
}

print(f"\nMultiple-testing corrections across {len(_grid)} parameter variants")
print(f"  uncorrected p < {ALPHA}:            {int(_raw_sig)}/{len(_grid)}")
print(f"  Bonferroni p < {ALPHA}/{len(_grid)}:        {int(_bonf_sig)}/{len(_grid)}")
print(f"  Benjamini-Hochberg FDR {FDR_ALPHA}:   {int(_bh_mask.sum())}/{len(_grid)}")
print(f"  chosen lookback={_chosen}: p = {_pvals[_ci]:.5f}, "
      f"survives Bonferroni: {bool(_pvals[_ci] < ALPHA/len(_grid))}, "
      f"survives BH: {bool(_bh_mask[_ci])}")
print("\nCAVEAT: these 56 tests are heavily correlated - a 17-bar and an 18-bar")
print("average are nearly the same strategy - so Bonferroni's independence")
print("assumption is violated and it is far too strict here. It is a lower bound")
print("on what survives, not an estimate. BH is the more appropriate correction,")
print("and the Deflated Sharpe above handles the same problem more directly.")

# %%
def cscv_pbo(ret_matrix: np.ndarray, n_splits: int) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    ret_matrix : (T, N) bar-level returns, one column per strategy variant.

    The series is cut into `n_splits` contiguous blocks. For every way of
    choosing half the blocks as in-sample (the rest out-of-sample), the best
    variant in-sample is identified and its out-of-sample RANK among all
    variants is recorded. PBO is the share of splits where that rank lands in
    the bottom half.

    PBO > 0.5 means selecting on in-sample performance does worse than picking
    at random - the in-sample ranking carries no out-of-sample information.

    Implementation note: Sharpe over any union of blocks is reconstructed from
    per-block counts, sums and sums of squares, so each of the thousands of
    combinations costs a handful of array operations rather than a full
    recomputation.
    """
    M = np.asarray(ret_matrix, dtype=float)
    T, N = M.shape
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}")
    if T < n_splits * 20:
        raise ValueError(f"need >= {n_splits*20} bars for {n_splits} splits, got {T}")

    bounds = np.linspace(0, T, n_splits + 1).astype(int)
    cnt = np.empty(n_splits)
    s1 = np.empty((n_splits, N))
    s2 = np.empty((n_splits, N))
    for b in range(n_splits):
        seg = M[bounds[b]:bounds[b + 1]]
        cnt[b] = len(seg)
        s1[b] = seg.sum(axis=0)
        s2[b] = (seg**2).sum(axis=0)

    def sharpe_of(blocks):
        c = cnt[list(blocks)].sum()
        m = s1[list(blocks)].sum(axis=0) / c
        v = s2[list(blocks)].sum(axis=0) / c - m**2
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(v > 1e-18, m / np.sqrt(np.maximum(v, 1e-18)), np.nan)

    all_b = set(range(n_splits))
    lambdas, is_sr, oos_sr, picks = [], [], [], []
    for train in combinations(range(n_splits), n_splits // 2):
        test = tuple(sorted(all_b - set(train)))
        sr_is = sharpe_of(train)
        sr_oos = sharpe_of(test)
        if not np.isfinite(sr_is).any() or not np.isfinite(sr_oos).any():
            continue
        n_star = int(np.nanargmax(sr_is))
        finite = np.isfinite(sr_oos)
        rank = float((sr_oos[finite] <= sr_oos[n_star]).sum())
        omega = rank / (finite.sum() + 1.0)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lambdas.append(math.log(omega / (1 - omega)))
        is_sr.append(float(sr_is[n_star]))
        oos_sr.append(float(sr_oos[n_star]))
        picks.append(n_star)

    lam = np.array(lambdas)
    return {"pbo": float(np.mean(lam < 0)) if len(lam) else np.nan,
            "n_combinations": len(lam), "lambdas": lam,
            "is_sharpe": np.array(is_sr), "oos_sharpe": np.array(oos_sr),
            "picks": np.array(picks),
            "prob_oos_loss": float(np.mean(np.array(oos_sr) < 0)) if len(oos_sr) else np.nan}


# Build the variant matrix: bar-level returns for every parameter on the grid.
_variant_cols, _variant_lb = [], []
for _lb in _grid:
    if _lb >= len(_full_close) // 4:
        continue
    _rr = strategy_returns_fast(_full_close, {**STRATEGY_PARAMS, "lookback": int(_lb)},
                                COST_BPS, SLIPPAGE_BPS, EXECUTION_DELAY_BARS)
    if np.isfinite(_rr).all() and _rr.std() > 0:
        _variant_cols.append(_rr)
        _variant_lb.append(_lb)
VARIANT_MATRIX = np.column_stack(_variant_cols)
print(f"variant matrix: {VARIANT_MATRIX.shape[0]} bars x {VARIANT_MATRIX.shape[1]} parameters")

PBO = cscv_pbo(VARIANT_MATRIX, CSCV_N_SPLITS)
print(f"\nCSCV with S={CSCV_N_SPLITS} blocks -> {PBO['n_combinations']} train/test splits")
print(f"  PBO = {PBO['pbo']:.4f}")
print(f"  probability the selected variant loses money out of sample = "
      f"{PBO['prob_oos_loss']:.4f}")
print(f"  mean in-sample Sharpe of the selected variant  "
      f"{PBO['is_sharpe'].mean()*math.sqrt(BPY):+.4f} (ann.)")
print(f"  mean out-of-sample Sharpe of the same variant  "
      f"{PBO['oos_sharpe'].mean()*math.sqrt(BPY):+.4f} (ann.)")
print(f"  most frequently selected lookback: "
      f"{_variant_lb[int(stats.mode(PBO['picks'], keepdims=False).mode)]}")
if PBO["pbo"] < 0.2:
    print("\n  PBO below 0.2: in-sample ranking carries real out-of-sample information.")
elif PBO["pbo"] < 0.5:
    print("\n  PBO between 0.2 and 0.5: selection helps, but a substantial share of the")
    print("  in-sample advantage does not survive.")
else:
    print("\n  PBO at or above 0.5: selecting the in-sample best is no better than")
    print("  choosing at random. The parameter ranking is noise.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

ax = axes[0]
ax.hist(PBO["lambdas"], bins=40, color=C["blue"], alpha=0.85)
ax.axvline(0, color=STATUS["critical"], lw=1.8, ls="--")
ax.text(0, ax.get_ylim()[1], "  below 0 = overfit", fontsize=8.5,
        color=STATUS["critical"], va="top")
finish(ax, f"CSCV logit ranks (PBO = {PBO['pbo']:.3f})",
       xlabel="logit of out-of-sample rank", ylabel="splits", legend=False,
       subtitle="mass left of 0 is where in-sample selection failed")

ax = axes[1]
_x = PBO["is_sharpe"] * math.sqrt(BPY)
_y = PBO["oos_sharpe"] * math.sqrt(BPY)
ax.scatter(_x, _y, s=9, color=C["blue"], alpha=0.35, label="split")
if len(_x) > 2:
    _b, _a = np.polyfit(_x, _y, 1)
    _xs = np.linspace(_x.min(), _x.max(), 50)
    ax.plot(_xs, _a + _b * _xs, color=C["orange"], lw=2.0,
            label=f"fit: slope {_b:.2f}")
_lim = [min(_x.min(), _y.min()), max(_x.max(), _y.max())]
ax.plot(_lim, _lim, color=INK["muted"], ls="--", lw=1.2, label="perfect transfer")
ax.axhline(0, color=INK["axis"], lw=0.9)
finish(ax, "Performance degradation", xlabel="in-sample Sharpe (ann.)",
       ylabel="out-of-sample Sharpe (ann.)",
       subtitle="a flat or negative slope means in-sample rank predicts nothing")

ax = axes[2]
ax.plot(_grid, -np.log10(np.where(np.isfinite(_pvals), np.maximum(_pvals, 1e-300), np.nan)),
        color=C["blue"], lw=1.8, label="-log10 p")
ax.axhline(-math.log10(ALPHA), color=INK["muted"], ls="--", lw=1.2,
           label=f"uncorrected {ALPHA}")
ax.axhline(-math.log10(ALPHA / len(_grid)), color=STATUS["critical"], ls=":", lw=1.4,
           label="Bonferroni")
ax.axvline(_chosen, color=INK["primary"], lw=1.2, ls="--")
finish(ax, "Significance across the grid", xlabel="lookback", ylabel="-log10(p)",
       subtitle="higher is more significant")
fig.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 20 - Stress testing
#
# Everything above asks whether the strategy's edge is real. This section asks a
# different question: **what happens to it in conditions the historical sample
# did not contain?**
#
# These scenarios are **not forecasts and not calibrated nulls**. Each one is an
# explicit assumption, built by taking a real bootstrapped path and imposing a
# named stress on it. They are labelled as assumptions throughout and excluded
# from the fidelity scorecard, because matching historical statistics is
# precisely what they are designed not to do.
#
# The useful output is not a Sharpe, it is a *shape*: which scenarios does this
# strategy survive, and which break it? A trend follower should handle a long
# bear market reasonably (it can go flat or short) and should struggle badly in
# a choppy sideways regime (repeated false breakouts). If the results contradict
# that expectation, something about the strategy is not what it appears.

# %%
def _base_paths(n_paths, n_steps, rng):
    """Bootstrapped real returns: the substrate every scenario modifies."""
    src = CALIB_LOGRET
    L = 20
    n_blocks = int(np.ceil(n_steps / L))
    starts = rng.integers(0, len(src) - L + 1, size=(n_paths, n_blocks))
    idx = starts[:, :, None] + np.arange(L)[None, None, :]
    return src[idx.reshape(n_paths, -1)[:, :n_steps]]


# Scenario magnitudes are anchored to real history where one exists, so they are
# severe-but-observed rather than arbitrary.
_worst_win = 250
_roll_sum = pd.Series(CALIB_LOGRET).rolling(_worst_win).sum()
WORST_BEAR_DRIFT = float(_roll_sum.min() / _worst_win) if _roll_sum.notna().any() else -0.0005
BEST_BULL_DRIFT = float(_roll_sum.max() / _worst_win) if _roll_sum.notna().any() else 0.0005
WORST_DAY = float(CALIB_LOGRET.min())


def scenario_crash(p, rng):
    """A single severe gap down, followed by a stretch of doubled volatility."""
    out = p.copy()
    n = out.shape[1]
    for i in range(len(out)):
        t = int(rng.integers(n // 5, n - 120))
        out[i, t] += WORST_DAY * 2.5
        out[i, t + 1:t + 60] *= 2.0
    return out


def scenario_vol_explosion(p, rng):
    """A permanent regime shift to triple volatility part-way through."""
    out = p.copy()
    n = out.shape[1]
    for i in range(len(out)):
        t = int(rng.integers(n // 4, 3 * n // 4))
        out[i, t:] *= 3.0
    return out


def scenario_bear_market(p, rng):
    """A long stretch at the worst sustained drift the real sample contained."""
    out = p.copy()
    n = out.shape[1]
    for i in range(len(out)):
        t = int(rng.integers(0, max(1, n - _worst_win * 2)))
        out[i, t:t + _worst_win * 2] += WORST_BEAR_DRIFT
    return out


def scenario_melt_up(p, rng):
    """A strong sustained rally that reverses sharply - the trap for a trend rule."""
    out = p.copy()
    n = out.shape[1]
    for i in range(len(out)):
        t = int(rng.integers(0, max(1, n - _worst_win - 60)))
        out[i, t:t + _worst_win] += BEST_BULL_DRIFT
        out[i, t + _worst_win:t + _worst_win + 40] += WORST_BEAR_DRIFT * 4
    return out


def scenario_sideways(p, rng):
    """Directionless chop: drift removed and returns made mean-reverting.

    Built by applying a negative moving-average filter, which flips the sign of
    short-horizon autocorrelation while leaving the volatility scale intact.
    This is the environment a trend rule should find hardest.
    """
    out = p.copy()
    out = out - out.mean(axis=1, keepdims=True)
    filt = np.empty_like(out)
    filt[:, 0] = out[:, 0]
    filt[:, 1:] = out[:, 1:] - 0.55 * out[:, :-1]
    return filt


def scenario_vol_cycle(p, rng):
    """Low -> high -> low volatility, a full regime round trip."""
    out = p.copy()
    n = out.shape[1]
    mult = np.ones(n)
    a, b = n // 3, 2 * n // 3
    mult[:a] = 0.6
    mult[a:b] = 2.5
    mult[b:] = 0.6
    return out * mult[None, :]


SCENARIOS = {
    "crash": scenario_crash,
    "vol_explosion": scenario_vol_explosion,
    "bear_market": scenario_bear_market,
    "melt_up": scenario_melt_up,
    "sideways_chop": scenario_sideways,
    "vol_cycle": scenario_vol_cycle,
}

_n_stress = int(min(N_SIMULATIONS, 150))
_rng_st = get_rng("stress")
STRESS_BT = {}

print(f"scenario anchors from the real sample: worst sustained {_worst_win}-bar drift "
      f"{WORST_BEAR_DRIFT*100:.3f}%/bar,")
print(f"best {BEST_BULL_DRIFT*100:.3f}%/bar, worst single bar {WORST_DAY*100:.2f}%\n")

_base = _base_paths(_n_stress, N_STEPS, _rng_st)
STRESS_BT["baseline_bootstrap"] = backtest_paths(_base, STRAT_FN, STRATEGY_PARAMS, _rng_st)
for _nm, _fn in SCENARIOS.items():
    _p = _fn(_base.copy(), _rng_st)
    STRESS_BT[_nm] = backtest_paths(_p, STRAT_FN, STRATEGY_PARAMS, _rng_st)
    print(f"  {_nm:<22} median Sharpe "
          f"{STRESS_BT[_nm]['sharpe'].median():+.3f}  "
          f"median maxDD {STRESS_BT[_nm]['max_dd'].median():.1%}")

_rows = []
_base_sh = STRESS_BT["baseline_bootstrap"]["sharpe"].median()
for _nm, _bt in STRESS_BT.items():
    _s = _bt["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    _rows.append({
        "scenario": _nm, "median_sharpe": float(_s.median()),
        "p5_sharpe": float(_s.quantile(0.05)), "p95_sharpe": float(_s.quantile(0.95)),
        "vs_baseline": float(_s.median() - _base_sh),
        "median_maxdd": float(_bt["max_dd"].median()),
        "frac_losing": float((_bt["total_return"] < 0).mean()),
    })
STRESS_SUMMARY = pd.DataFrame(_rows).set_index("scenario")
print("\nStress scenarios (ASSUMPTIONS, not forecasts)")
print(STRESS_SUMMARY.round(4).to_string())

_worst = STRESS_SUMMARY.drop("baseline_bootstrap")["median_sharpe"].idxmin()
print(f"\nharshest scenario: {_worst} "
      f"({STRESS_SUMMARY.loc[_worst,'median_sharpe']:+.3f} median Sharpe, "
      f"{STRESS_SUMMARY.loc[_worst,'frac_losing']:.0%} of paths lose money)")

# %%
fig, ax = plt.subplots(figsize=(10, 4.0))
_ord = STRESS_SUMMARY.sort_values("median_sharpe")
_y = np.arange(len(_ord))
_cols = [INK["muted"] if n == "baseline_bootstrap" else
         (STATUS["critical"] if v < 0 else C["blue"])
         for n, v in zip(_ord.index, _ord["median_sharpe"])]
for _i, (_n, _r) in enumerate(_ord.iterrows()):
    ax.plot([_r["p5_sharpe"], _r["p95_sharpe"]], [_i, _i], color=_cols[_i], lw=2.4,
            alpha=0.85, solid_capstyle="round")
ax.scatter(_ord["median_sharpe"], _y, s=40, color=_cols, zorder=4,
           edgecolor=INK["surface"], linewidth=1.2)
ax.axvline(0, color=INK["axis"], lw=1.1)
ax.axvline(_base_sh, color=INK["primary"], lw=1.4, ls="--")
ax.text(_base_sh, len(_ord) - 0.3, " baseline", fontsize=8.5, color=INK["primary"], va="top")
ax.set_yticks(_y)
ax.set_yticklabels(_ord.index, fontsize=9)
finish(ax, "Strategy performance under stress scenarios", xlabel="Sharpe ratio",
       legend=False, subtitle=f"dot = median, bar = 5th-95th percentile of "
       f"{_n_stress} paths; these are assumptions, not predictions")
ax.grid(axis="y", visible=False)
fig.tight_layout()
plt.show()
