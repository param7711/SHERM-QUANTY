# %% [markdown]
# ---
# ## 08 - Synthetic data engine
#
# The central component. Every generator here is calibrated against the market
# statistics measured in Sections 05 and 07, on the calibration slice only, and
# then **frozen** at the end of Section 09 before any strategy exists.
#
# No single generator is trusted. Each one preserves some properties of the real
# market and destroys others, and that is deliberate: comparing across a family
# whose members fail in *different* ways is what localises where a strategy's
# edge actually comes from. A strategy that survives the block bootstrap but
# dies on the IID bootstrap is exploiting temporal structure. One that survives
# both equally is probably exploiting the return distribution. One that dies on
# everything including generators that preserve trend was fitted to the specific
# historical path.
#
# ### What each method preserves and destroys
#
# | generator | preserves | destroys |
# |---|---|---|
# | **IID bootstrap** | exact marginal distribution (tails, skew) | *all* temporal dependence: no vol clustering, no trend |
# | **Moving block** | dependence inside a block of length L | dependence beyond L; under-samples the series' endpoints |
# | **Circular block** | same, with every observation equally likely | dependence beyond L; splices the end onto the start |
# | **Stationary bootstrap** | dependence over a *random* block length (mean L) | long-range dependence; output is strictly stationary by construction |
# | **Regime bootstrap** | regime persistence + the real return distribution inside each regime | dependence within a regime beyond the bar level |
# | **Markov regime (Gaussian)** | regime persistence, per-regime mean and variance | within-regime fat tails (emissions are Gaussian) |
# | **Shock-preserving block** | shock frequency, magnitude, and clustering explicitly | dependence beyond L |
# | **GARCH / EGARCH / GJR** | volatility clustering and persistence; leverage (E/GJR) | trend; exact marginal shape |
# | **Filtered historical simulation** | GARCH vol dynamics *and* the empirical innovation distribution | ordering of the innovations |
# | **Fourier surrogate** | power spectrum, hence the full linear autocorrelation | non-linear structure; marginal is driven toward Gaussian |
# | **IAAFT surrogate** | power spectrum *and* the exact marginal distribution | non-linear dependence (vol clustering is largely lost) |
# | **Trend-preserving** | measured trend durations, drifts and reversal rates | higher-order structure within a trend segment |
# | **Parametric jump-diffusion** | calibrated drift, AR, GARCH vol and jump process | anything the parametric form cannot express |
#
# **Note the deliberate contrast at the two extremes.** The IID bootstrap is the
# strongest possible null for a trend strategy: same returns, no order. The
# IAAFT surrogate is subtler - it keeps the exact return histogram *and* the
# linear autocorrelation structure, so it isolates whether the strategy needs
# non-linear dependence (volatility clustering) specifically.

# %%
CALIB_LOGRET = np.log1p(np.clip(CALIB_RET, -0.999999, None))
# Asserted rather than filtered: dropping non-finite entries here would silently
# misalign the return array against the regime labels fitted on CALIB_RET.
assert np.isfinite(CALIB_LOGRET).all(), "non-finite calibration returns"
assert len(CALIB_LOGRET) == len(CALIB_RET)
N_CALIB = len(CALIB_LOGRET)

# Synthetic paths are the SAME LENGTH as the slice the generators were calibrated
# on, and Section 13 compares them against the strategy's performance on that
# same slice. Both halves of that matter. A Sharpe ratio's sampling error scales
# with 1/sqrt(T), so comparing a 6,500-bar real result against 3,300-bar
# synthetic paths would make the real one look special purely through having
# had longer to average out - an artefact, not a finding.
N_STEPS = N_CALIB

# Shock parameters for the generators are measured on the CALIBRATION slice, not
# on REAL_STATS (which covers the full sample including the untouched test set).
# Reading them from REAL_STATS would leak test-period information into the
# generators and quietly invalidate the out-of-sample claim.
CALIB_SHOCK_THR = float(np.quantile(np.abs(CALIB_LOGRET), SHOCK_PRIMARY_QUANTILE))
CALIB_SHOCK_FREQ = float(np.mean(np.abs(CALIB_LOGRET) >= CALIB_SHOCK_THR))

print(f"calibration returns: {N_CALIB} bars "
      f"({RET.index[0].date()} .. {RET.index[CALIB_END-1].date()})")
print(f"synthetic paths will be {N_STEPS} bars long (matching the calibration slice)")
print(f"shock threshold (calibration only): |log-ret| >= {CALIB_SHOCK_THR:.4f}, "
      f"rate {CALIB_SHOCK_FREQ:.3%}")


@dataclass
class Generator:
    """One frozen, calibrated synthetic-market generator.

    `sample(rng, n_paths, n_steps)` returns an (n_paths, n_steps) array of LOG
    returns. Log space is used internally throughout so that paths compound
    exactly and cannot produce a non-positive price; callers convert with
    `np.expm1` at the boundary.
    """
    name: str
    family: str                 # bootstrap | model | surrogate | scenario
    preserves: str
    destroys: str
    sample: object              # Callable[[Generator, rng, int, int], np.ndarray]
    calibration: dict = field(default_factory=dict)
    available: bool = True
    error: str | None = None
    note: str = ""

    def __call__(self, rng, n_paths, n_steps):
        if not self.available:
            raise RuntimeError(f"generator {self.name!r} unavailable: {self.error}")
        out = self.sample(self, rng, n_paths, n_steps)
        out = np.asarray(out, dtype=float)
        if out.shape != (n_paths, n_steps):
            raise ValueError(f"{self.name}: expected {(n_paths, n_steps)}, got {out.shape}")
        if not np.isfinite(out).all():
            n_bad = int((~np.isfinite(out)).sum())
            raise ValueError(f"{self.name}: produced {n_bad} non-finite log returns")
        return out


# ---------------------------------------------------------------------------
# METHOD A - IID bootstrap
# ---------------------------------------------------------------------------
def _sample_iid(gen, rng, n_paths, n_steps):
    """Resample returns with replacement, independently at every bar.

    The purest null for anything that trades sequence: the return distribution
    is preserved exactly (every drawn value is a real historical return) while
    every trace of ordering is destroyed. A strategy that still profits here is
    profiting from the shape of the distribution - drift and skew - and not from
    any pattern in time.
    """
    src = gen.calibration["returns"]
    return rng.choice(src, size=(n_paths, n_steps), replace=True)


# ---------------------------------------------------------------------------
# METHOD B/C - block bootstraps
# ---------------------------------------------------------------------------
def _sample_moving_block(gen, rng, n_paths, n_steps):
    """Non-overlapping concatenation of randomly chosen contiguous blocks.

    Block starts are drawn from [0, N-L], so observations near the two ends of
    the series appear in fewer possible blocks and are therefore under-sampled.
    That bias is the reason the circular variant exists.
    """
    src, L = gen.calibration["returns"], gen.calibration["block_length"]
    n_blocks = int(np.ceil(n_steps / L))
    starts = rng.integers(0, len(src) - L + 1, size=(n_paths, n_blocks))
    idx = starts[:, :, None] + np.arange(L)[None, None, :]
    return src[idx.reshape(n_paths, -1)[:, :n_steps]]


def _sample_circular_block(gen, rng, n_paths, n_steps):
    """Block bootstrap on the series wrapped into a circle.

    Wrapping means every observation belongs to exactly L blocks, so all of them
    are sampled with equal probability and the endpoint bias of the moving block
    bootstrap disappears. The price is one artificial splice per wrap, where the
    end of the sample is glued to its beginning.
    """
    src, L = gen.calibration["returns"], gen.calibration["block_length"]
    N = len(src)
    n_blocks = int(np.ceil(n_steps / L))
    starts = rng.integers(0, N, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % N
    return src[idx.reshape(n_paths, -1)[:, :n_steps]]


def _sample_stationary(gen, rng, n_paths, n_steps):
    """Politis-Romano stationary bootstrap: geometric block lengths, mean L.

    At each step the walker either advances one position along the real series
    (probability 1 - 1/L) or jumps to a fresh uniformly random position
    (probability 1/L). Randomising the block length is what makes the resampled
    series strictly stationary, which fixed-length block bootstraps are not.
    """
    src, L = gen.calibration["returns"], gen.calibration["block_length"]
    N = len(src)
    p = 1.0 / L
    idx = np.empty((n_paths, n_steps), dtype=np.int64)
    cur = rng.integers(0, N, size=n_paths)
    idx[:, 0] = cur
    jump = rng.random((n_paths, n_steps)) < p
    fresh = rng.integers(0, N, size=(n_paths, n_steps))
    for t in range(1, n_steps):
        cur = np.where(jump[:, t], fresh[:, t], (cur + 1) % N)
        idx[:, t] = cur
    return src[idx]


# ---------------------------------------------------------------------------
# Regime-conditioned generators
# ---------------------------------------------------------------------------
def _simulate_regime_sequence(A, pi, n_paths, n_steps, rng):
    """Draw new regime paths from an estimated transition matrix."""
    K = len(pi)
    cum = np.cumsum(A, axis=1)
    states = np.empty((n_paths, n_steps), dtype=np.int64)
    cur = rng.choice(K, size=n_paths, p=pi / pi.sum())
    states[:, 0] = cur
    u = rng.random((n_paths, n_steps))
    for t in range(1, n_steps):
        cur = (u[:, t][:, None] > cum[cur]).sum(axis=1)
        cur = np.clip(cur, 0, K - 1)
        states[:, t] = cur
    return states


def _sample_regime_bootstrap(gen, rng, n_paths, n_steps):
    """New regime sequence from the transition matrix; real returns within each.

    Semi-parametric: the *ordering* of market states is simulated, but the
    returns emitted inside a state are genuine historical returns drawn from the
    bars that state actually covered. So per-regime fat tails and skew survive
    intact, while the sequence of calm and crisis stretches is genuinely new.
    """
    cal = gen.calibration
    states = _simulate_regime_sequence(cal["A"], cal["pi"], n_paths, n_steps, rng)
    out = np.empty((n_paths, n_steps))
    for k, pool in enumerate(cal["pools"]):
        m = states == k
        cnt = int(m.sum())
        if cnt:
            out[m] = rng.choice(pool, size=cnt, replace=True)
    return out


def _sample_markov_gaussian(gen, rng, n_paths, n_steps):
    """New regime sequence; Gaussian emissions with the fitted per-regime moments.

    Fully parametric counterpart to the regime bootstrap. Comparing the two
    isolates how much of the market's tail behaviour is *within*-regime rather
    than a by-product of switching between regimes of different variance.
    """
    cal = gen.calibration
    states = _simulate_regime_sequence(cal["A"], cal["pi"], n_paths, n_steps, rng)
    mu, sd = cal["mu"], cal["sigma"]
    return mu[states] + sd[states] * rng.standard_normal((n_paths, n_steps))


# ---------------------------------------------------------------------------
# Shock-preserving block bootstrap
# ---------------------------------------------------------------------------
def _sample_shock_block(gen, rng, n_paths, n_steps):
    """Stratified block bootstrap that reproduces the shock arrival rate.

    A plain block bootstrap reproduces shock frequency only on average, and in
    any single path it can drift well off. Here blocks are split into those
    containing at least one shock and those containing none, and are then drawn
    so that the proportion of shock-bearing blocks matches the historical
    proportion in *every* path.

    Because whole real blocks are used, a shock arrives with its genuine
    magnitude, its genuine neighbours, and the genuine elevated volatility that
    followed it - the aftermath is preserved, not just the spike.
    """
    cal = gen.calibration
    src, L = cal["returns"], cal["block_length"]
    shock_starts, calm_starts, frac = cal["shock_starts"], cal["calm_starts"], cal["shock_block_frac"]
    n_blocks = int(np.ceil(n_steps / L))
    n_shock = int(round(frac * n_blocks))

    pick = np.empty((n_paths, n_blocks), dtype=np.int64)
    if n_shock > 0 and len(shock_starts):
        pick[:, :n_shock] = rng.choice(shock_starts, size=(n_paths, n_shock), replace=True)
    if n_blocks - n_shock > 0:
        pool = calm_starts if len(calm_starts) else shock_starts
        pick[:, n_shock:] = rng.choice(pool, size=(n_paths, n_blocks - n_shock), replace=True)
    # Shuffle each path's block order so shocks are not all stacked at the front.
    order = rng.permuted(np.tile(np.arange(n_blocks), (n_paths, 1)), axis=1)
    pick = np.take_along_axis(pick, order, axis=1)

    idx = pick[:, :, None] + np.arange(L)[None, None, :]
    return src[idx.reshape(n_paths, -1)[:, :n_steps]]


# ---------------------------------------------------------------------------
# METHOD - conditional volatility models (GARCH family)
# ---------------------------------------------------------------------------
def _innovation_callable(dist: str, nu: float, rng):
    """Return f(n) -> n unit-variance innovations, drawn from our seeded stream.

    The variance recursion itself is delegated to `arch` (see
    `_simulate_vol_model`), so the only thing defined here is the innovation
    draw - which is simple enough to state and verify outright. A Student-t with
    nu degrees of freedom has variance nu/(nu-2), so it is divided by the square
    root of that to reach unit variance; omitting that scaling is the classic
    way to end up with a simulation whose volatility is silently wrong.
    """
    if dist == "normal":
        return lambda n: rng.standard_normal(n)
    if dist == "t":
        if nu is None or nu <= 2:
            raise ValueError(f"Student-t needs nu > 2 for finite variance, got {nu}")
        scale = math.sqrt(nu / (nu - 2.0))
        return lambda n: rng.standard_t(nu, size=n) / scale
    raise ValueError(f"unknown innovation distribution {dist!r}")


def _simulate_vol_model(p, n_paths, n_steps, rng, burn=750, innov_factory=None):
    """Simulate a GARCH / GJR / EGARCH path set, then apply the AR(1) mean.

    The variance recursion is run by `arch`'s own `volatility.simulate`, using
    the exact parameter vector `arch` estimated. That is deliberate. Each of
    these models has parameterisation details that differ between textbooks -
    whether EGARCH's asymmetry term subtracts E|z| under the normal or under the
    fitted distribution, how the recursion is initialised, how burn-in is
    handled - and a mis-transcription would not crash. It would quietly produce
    a market calibrated to something other than the data, which is precisely the
    failure this notebook exists to detect in other people's work.

    Reproducibility is kept by supplying our own seeded innovation callable
    rather than using `arch`'s internal random state.

    `innov_factory(rng) -> f(n)` overrides the innovation draw, which is how
    filtered historical simulation injects bootstrapped empirical residuals.

    Returns log returns (converted out of the percent units the models were
    fitted in), shape (n_paths, n_steps).
    """
    vol_obj, vol_params = p["vol_obj"], p["vol_params"]
    out = np.empty((n_paths, n_steps))
    for i in range(n_paths):
        cb = (innov_factory(rng) if innov_factory is not None
              else _innovation_callable(p["dist"], p.get("nu"), rng))
        eps, _sigma2 = vol_obj.simulate(vol_params, n_steps, cb, burn=burn)
        out[i] = eps

    const, phi = p["const"], p["phi"]
    r = np.empty_like(out)
    prev = np.full(n_paths, const / (1.0 - phi) if abs(phi) < 0.999 else const)
    for t in range(n_steps):
        prev = const + phi * prev + out[:, t]
        r[:, t] = prev
    return r / 100.0


def _sample_vol_model(gen, rng, n_paths, n_steps):
    return _simulate_vol_model(gen.calibration, n_paths, n_steps, rng)


def _sample_fhs(gen, rng, n_paths, n_steps):
    """Filtered historical simulation.

    Three steps: filter the real returns through the fitted volatility model to
    get standardised residuals; block-bootstrap those residuals (blocks, not IID
    draws, so runs of unusually large innovations survive); push them back
    through the same variance recursion.

    The result keeps the conditional-volatility dynamics of the parametric model
    *and* the empirical innovation distribution, so the tails are real observed
    tails rather than a fitted Student-t's idea of them. This is the standard
    method in market-risk practice, and it is often the most faithful generator
    in the whole family - Section 09 tests whether that holds here rather than
    assuming it.
    """
    cal = gen.calibration
    z_emp, L = cal["std_resid"], cal["resid_block"]

    def factory(_rng):
        def draw(n):
            n_blocks = int(np.ceil(n / L))
            starts = _rng.integers(0, len(z_emp) - L + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
            return z_emp[idx]
        return draw

    return _simulate_vol_model(cal, n_paths, n_steps, rng, innov_factory=factory)


# ---------------------------------------------------------------------------
# Surrogate data
# ---------------------------------------------------------------------------
def _sample_fourier(gen, rng, n_paths, n_steps):
    """Fourier phase randomisation.

    The amplitude spectrum is kept and the phases are replaced with uniform
    random ones. Since the power spectrum and the autocovariance are a Fourier
    pair, this preserves the entire *linear* correlation structure exactly while
    destroying everything non-linear.

    A consequence worth stating: summing many independent random-phase
    components drives the marginal distribution toward Gaussian, so fat tails
    are largely lost. IAAFT below is the fix for precisely that.
    """
    src = gen.calibration["returns"]
    n = len(src)
    mu = src.mean()
    X = np.fft.rfft(src - mu)
    amp = np.abs(X)
    out = np.empty((n_paths, n))
    for i in range(n_paths):
        ph = rng.uniform(0, 2 * np.pi, len(X))
        ph[0] = 0.0                       # DC term stays real
        if n % 2 == 0:
            ph[-1] = 0.0                  # Nyquist term stays real
        out[i] = np.fft.irfft(amp * np.exp(1j * ph), n=n) + mu
    return _fit_length(out, n_steps, rng)


def _sample_iaaft(gen, rng, n_paths, n_steps):
    """Iterative amplitude-adjusted Fourier transform.

    Alternates between two projections: impose the target amplitude spectrum in
    the frequency domain, then impose the exact target value set by rank-ordering
    in the time domain. It converges to a series that has (very nearly) the real
    power spectrum AND (exactly) the real return histogram.

    That makes it the sharpest surrogate in the family. Any strategy edge that
    survives the IID bootstrap but dies on IAAFT is not coming from the marginal
    distribution and not from linear autocorrelation, which leaves non-linear
    dependence - volatility clustering - as the source.
    """
    src = gen.calibration["returns"]
    n = len(src)
    target_amp = np.abs(np.fft.rfft(src))
    sorted_vals = np.sort(src)
    n_iter = gen.calibration["iaaft_iters"]
    out = np.empty((n_paths, n))
    for i in range(n_paths):
        y = rng.permutation(src)
        for _ in range(n_iter):
            Y = np.fft.rfft(y)
            phase = np.angle(Y)
            y = np.fft.irfft(target_amp * np.exp(1j * phase), n=n)
            ranks = np.argsort(np.argsort(y))
            y = sorted_vals[ranks]
        out[i] = y
    return _fit_length(out, n_steps, rng)


def _fit_length(paths, n_steps, rng):
    """Trim or tile surrogate paths to the requested length.

    Surrogates are defined on the calibration sample's own length. When more
    bars are requested, independent surrogate segments are concatenated rather
    than a single one being repeated, so the output never contains a duplicated
    stretch of itself.
    """
    n_paths, n = paths.shape
    if n == n_steps:
        return paths
    if n > n_steps:
        offs = rng.integers(0, n - n_steps + 1, size=n_paths)
        return np.stack([paths[i, o:o + n_steps] for i, o in enumerate(offs)])
    reps = int(np.ceil(n_steps / n))
    # Shuffle ROW order between repetitions (not elements within columns, which
    # would scramble time), so a tiled path splices together different surrogates
    # rather than repeating one of them.
    tiled = np.concatenate([paths[rng.permutation(n_paths)] for _ in range(reps)], axis=1)
    return tiled[:, :n_steps]


# ---------------------------------------------------------------------------
# Trend-preserving generator
# ---------------------------------------------------------------------------
def _segment_trends(logret, window):
    """Split a series into trend segments at moving-average crossings.

    A segment is a maximal run of bars on one side of the trailing moving
    average. Each segment keeps its ACTUAL return sequence, not a summary of it
    - see `_sample_trend_preserving` for why that distinction turned out to
    matter.
    """
    price = np.exp(np.cumsum(logret))
    ma = pd.Series(price).rolling(window).mean().to_numpy()
    above = price > ma
    valid = np.isfinite(ma)
    segs = []
    i = window
    while i < len(logret):
        if not valid[i]:
            i += 1
            continue
        state = above[i]
        j = i
        while j < len(logret) and valid[j] and above[j] == state:
            j += 1
        if j - i >= 3:
            segs.append({"up": bool(state), "returns": logret[i:j].copy(),
                         "duration": int(j - i), "drift": float(logret[i:j].mean())})
        i = j
    return segs


def _sample_trend_preserving(gen, rng, n_paths, n_steps):
    """Renewal process that chains whole real trend segments in a new order.

    Each segment is a genuine contiguous stretch of history - one complete run
    above or below the moving average, with its own real returns intact.
    Segments are drawn from the empirical pool and chained using the observed
    up-after-down and down-after-up rates, so the synthetic path has bull runs,
    bear runs and reversals of realistic length and magnitude in an order
    history never produced.

    An earlier version of this generator summarised each segment as a
    (duration, drift, residual volatility) triple and rebuilt it as a constant
    drift plus IID noise. That version FAILED its own fidelity check - it
    scored 0.14 on the trend group, the very property it is named for. The
    reason is instructive: a constant drift with IID noise is a *purer* trend
    than any real market produces, so the variance ratios came out far above the
    real ones. Replacing summary statistics with the real return sequences fixes
    it, because within-segment structure is then preserved rather than
    idealised. The fidelity scorecard is what caught this; without it the
    generator would have shipped looking plausible.
    """
    cal = gen.calibration
    up_segs, dn_segs = cal["up_segments"], cal["down_segments"]
    p_up_after_dn, p_dn_after_up = cal["p_up_after_dn"], cal["p_dn_after_up"]

    out = np.empty((n_paths, n_steps))
    for i in range(n_paths):
        pos = 0
        up = bool(rng.random() < 0.5)
        buf = []
        while pos < n_steps:
            pool = up_segs if up else dn_segs
            seg = pool[int(rng.integers(0, len(pool)))]
            take = min(len(seg), n_steps - pos)
            buf.append(seg[:take])
            pos += take
            up = (rng.random() < p_up_after_dn) if not up else (rng.random() >= p_dn_after_up)
        out[i] = np.concatenate(buf)[:n_steps]
    return out


# ---------------------------------------------------------------------------
# Parametric jump-diffusion
# ---------------------------------------------------------------------------
def _sample_parametric(gen, rng, n_paths, n_steps):
    """AR(1) drift + GARCH volatility + compound-Poisson jumps.

    Every parameter is calibrated, and the diffusion and jump components are
    **identified separately** so that they do not double-count the same moves.

    The naive version of this generator - fit GARCH to the returns, then add a
    jump process on top - is wrong, and wrong in a direction that flatters the
    generator's realism while inflating its volatility. The returns used for the
    GARCH fit already contain the historical crashes, so the fitted diffusion
    has absorbed them; adding an independent jump process then counts them
    twice, and the synthetic market comes out materially more volatile than the
    real one.

    The decomposition used instead is exact:
        r_t = winsorise(r_t, +/- threshold)  +  excess_t
    where `excess_t` is non-zero only on shock bars. The diffusion is fitted to
    the winsorised series and the jump sizes are the excesses, so recombining
    them reconstructs the original series exactly and the variance budget adds
    up.
    """
    cal = gen.calibration
    base = _simulate_vol_model(cal, n_paths, n_steps, rng)
    hit = rng.random((n_paths, n_steps)) < cal["jump_intensity"]
    jumps = rng.choice(cal["jump_pool"], size=(n_paths, n_steps), replace=True)
    return base + hit * jumps


print("generator sampling functions defined")

# %% [markdown]
# ### Fitting the conditional-volatility models
#
# Three specifications are fitted with two innovation distributions each, and
# the winner is chosen **by BIC and residual diagnostics** - never by which one
# flatters the strategy. The strategy does not exist yet, which is the point of
# the section ordering.
#
# The diagnostics that matter:
#
# - **Ljung-Box on standardised residuals** - if the model has captured the
#   volatility dynamics, the squared standardised residuals should no longer be
#   autocorrelated. A small p-value means leftover structure the model missed.
# - **Persistence** (`alpha + beta`, or `alpha + gamma/2 + beta` for GJR) - how
#   long a volatility shock echoes. Values at or above 1 mean the unconditional
#   variance does not exist, and a simulation from such a fit will wander
#   without a level to revert to. Where that happens it is reported and the
#   persistence is capped for simulation, because an uncapped IGARCH path is not
#   a market, it is a divergence.

# %%
def fit_vol_model(y, kind, spec_kw, dist, tag):
    """Fit one conditional-volatility model and package it for simulation.

    Returns (params_dict, diagnostics_dict). The params dict carries `arch`'s
    own volatility object and its exact estimated parameter vector, so
    simulation replays the estimated recursion rather than a re-derivation of
    it.

    Raises on a failed fit; the caller reports the failure rather than
    substituting a different model.
    """
    am = arch_model(y, mean="AR", lags=1, dist=dist, **spec_kw)
    res = am.fit(disp="off", show_warning=False)
    pr = res.params

    # res.params is ordered [mean..., volatility..., distribution...]. The mean
    # is AR(1), so it contributes exactly two entries.
    n_mean = 2
    vol_names = am.volatility.parameter_names()
    n_vol = len(vol_names)
    vals = res.params.values.astype(float)
    vol_params = vals[n_mean:n_mean + n_vol].copy()
    dist_params = vals[n_mean + n_vol:].copy()
    ix = {nm: i for i, nm in enumerate(vol_names)}

    def persistence_of(vp):
        if kind == "EGARCH":
            return float(vp[ix["beta[1]"]])
        a = float(vp[ix["alpha[1]"]])
        g = float(vp[ix["gamma[1]"]]) if "gamma[1]" in ix else 0.0
        return a + g / 2.0 + float(vp[ix["beta[1]"]])

    persist = persistence_of(vol_params)
    capped = False
    if persist >= 0.9999:
        # At or above 1 the unconditional variance does not exist and simulated
        # variance random-walks instead of reverting, which produces paths whose
        # volatility level is arbitrary. Scale the ARCH/GARCH terms just below 1
        # and say so, rather than shipping a divergent "market".
        scale = 0.999 / persist
        for nm in ("alpha[1]", "gamma[1]", "beta[1]"):
            if nm in ix:
                vol_params[ix[nm]] *= scale
        persist = persistence_of(vol_params)
        capped = True

    z = np.asarray(res.resid, dtype=float) / np.asarray(res.conditional_volatility, dtype=float)
    z = z[np.isfinite(z)]

    params = {
        "kind": kind, "dist": dist, "tag": tag,
        "const": float(pr["Const"]), "phi": float(pr["y[1]"]),
        "vol_obj": am.volatility, "vol_params": vol_params,
        "dist_params": dist_params, "persistence": persist,
        "std_resid": z, "capped": capped,
    }
    if dist == "t":
        params["nu"] = float(pr["nu"])

    lb = (float(acorr_ljungbox(z**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
          if HAVE_STATSMODELS else np.nan)
    diag = {"model": tag, "loglik": float(res.loglikelihood), "aic": float(res.aic),
            "bic": float(res.bic), "persistence": persist,
            "nu": params.get("nu", np.nan), "lb_p_z2": lb, "capped": capped}
    return params, diag


VOL_MODELS = {}
VOL_FIT_TABLE = []

if HAVE_ARCH:
    _specs = [
        ("GARCH", dict(vol="GARCH", p=1, o=0, q=1), "normal"),
        ("GARCH", dict(vol="GARCH", p=1, o=0, q=1), "t"),
        ("GJR", dict(vol="GARCH", p=1, o=1, q=1), "t"),
        ("EGARCH", dict(vol="EGARCH", p=1, o=1, q=1), "t"),
    ]
    _y = CALIB_LOGRET * 100.0
    for _kind, _kw, _dist in _specs:
        _tag = f"{_kind}-{_dist}"
        try:
            _p, _diag = fit_vol_model(_y, _kind, _kw, _dist, _tag)
            VOL_MODELS[_tag] = _p
            VOL_FIT_TABLE.append(_diag)
        except Exception as exc:
            print(f"  FAILED to fit {_tag}: {type(exc).__name__}: {exc}")
            VOL_FIT_TABLE.append({"model": _tag, "loglik": np.nan, "aic": np.nan,
                                  "bic": np.nan, "persistence": np.nan, "nu": np.nan,
                                  "lb_p_z2": np.nan, "capped": False})
else:
    print("SKIPPED: `arch` is not installed, so the GARCH/EGARCH/GJR generators and")
    print("filtered historical simulation are unavailable. Every other generator")
    print("still runs; the final report will list these as missing rather than")
    print("substituting something else for them.")

VOL_SELECTION_RULE = None
if VOL_FIT_TABLE:
    _vf = pd.DataFrame(VOL_FIT_TABLE).sort_values("bic")
    print("Conditional-volatility model comparison (fitted on the calibration slice)")
    print(_vf.round(4).to_string(index=False))

    # Selection rule, fixed in advance and stated: among the models whose
    # standardised residuals show no leftover volatility structure (Ljung-Box
    # p > 0.05), take the lowest BIC. A model that fits the likelihood best but
    # leaves autocorrelation in its squared residuals has not finished
    # explaining the volatility, and simulating from it propagates whatever it
    # missed. Note what this rule is NOT: it makes no reference whatsoever to
    # trading performance, and it is applied here, before a strategy exists.
    _valid = _vf[_vf["bic"].notna()]
    _clean = _valid[_valid["lb_p_z2"] > 0.05]
    if len(_clean):
        BEST_VOL_MODEL = _clean.iloc[0]["model"]
        VOL_SELECTION_RULE = "lowest BIC among models passing Ljung-Box on squared residuals"
    elif len(_valid):
        BEST_VOL_MODEL = _valid.iloc[0]["model"]
        VOL_SELECTION_RULE = "lowest BIC (NO model passed the residual diagnostic)"
    else:
        BEST_VOL_MODEL = None
        VOL_SELECTION_RULE = "none - every fit failed"

    print(f"\nselection rule: {VOL_SELECTION_RULE}")
    print(f"selected: {BEST_VOL_MODEL}")

    if BEST_VOL_MODEL:
        _b = VOL_MODELS[BEST_VOL_MODEL]
        _row = _valid[_valid["model"] == BEST_VOL_MODEL].iloc[0]
        print(f"  persistence = {_b['persistence']:.4f}"
              + ("  [CAPPED below 1 for simulation - the raw fit was at or above 1, "
                 "where the unconditional variance does not exist]" if _b["capped"] else ""))
        if np.isfinite(_row["lb_p_z2"]):
            print(f"  Ljung-Box p on squared standardised residuals = {_row['lb_p_z2']:.4f}")

        _bic_best = _valid.iloc[0]["model"]
        if _bic_best != BEST_VOL_MODEL:
            print(f"\n  NOTE: BIC alone would have chosen {_bic_best} "
                  f"(BIC {_valid.iloc[0]['bic']:.1f} vs {_row['bic']:.1f}), but its")
            print(f"  squared standardised residuals still carry autocorrelation "
                  f"(Ljung-Box p = {_valid.iloc[0]['lb_p_z2']:.4f}), meaning volatility")
            print("  structure it did not capture. The two criteria genuinely disagree")
            print("  here; the diagnostic is given precedence because the model's job in")
            print("  this notebook is to GENERATE realistic volatility dynamics, not to")
            print("  maximise in-sample likelihood. Both models are kept as separate")
            print("  generators regardless, so the choice only affects which one seeds")
            print("  filtered historical simulation and the jump-diffusion base.")
        elif len(_clean) == 0:
            print("\n  WARNING: no model passed the residual diagnostic. Every GARCH-family")
            print("  generator here leaves some volatility structure unexplained, and their")
            print("  synthetic paths will understate volatility clustering to that extent.")
else:
    BEST_VOL_MODEL = None

# %% [markdown]
# ### Verifying the simulation pipeline against `arch`
#
# The variance recursion is `arch`'s own, but three pieces around it are ours:
# the seeded innovation draw, the AR(1) mean recursion, and the conversion out
# of percent units. Any of those could be wrong in a way that still looks
# plausible - a Student-t that was never rescaled to unit variance, for
# instance, produces perfectly reasonable-looking volatility clustering at
# entirely the wrong level.
#
# So the assembled pipeline is compared against `arch.simulate`, which does the
# whole job internally, using the same estimated parameters. Both are stochastic,
# and with a near-unit-persistence model the volatility level of any single path
# is genuinely variable, so the comparison is made on **medians across many
# paths** rather than on one draw.

# %%
if BEST_VOL_MODEL and HAVE_ARCH:
    _p = VOL_MODELS[BEST_VOL_MODEL]
    _rng_v = get_rng("vol_simulator_check")
    _n_chk, _len_chk = 24, 3000

    _ours = _simulate_vol_model(_p, _n_chk, _len_chk, _rng_v) * 100.0

    _spec = {"GARCH": dict(vol="GARCH", p=1, o=0, q=1),
             "GJR": dict(vol="GARCH", p=1, o=1, q=1),
             "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1)}[_p["kind"]]
    _am = arch_model(CALIB_LOGRET * 100.0, mean="AR", lags=1, dist=_p["dist"], **_spec)
    _res = _am.fit(disp="off", show_warning=False)
    _theirs = np.stack([np.asarray(_am.simulate(_res.params.values, _len_chk)["data"])
                        for _ in range(_n_chk)])

    def _path_moments(M):
        return {"sd": float(np.median([np.std(x) for x in M])),
                "kurtosis": float(np.median([stats.kurtosis(x) for x in M])),
                "acf_abs1": float(np.median([acf(np.abs(x), 1)[1] for x in M])),
                "acf_abs10": float(np.median([acf(np.abs(x), 10)[10] for x in M]))}

    _rows = [{"source": "our pipeline", **_path_moments(_ours)},
             {"source": "arch.simulate", **_path_moments(_theirs)},
             {"source": "real (calibration)", **_path_moments(CALIB_LOGRET[None, :] * 100.0)}]
    _cmp = pd.DataFrame(_rows)
    print(f"Simulation cross-check - {BEST_VOL_MODEL}, median over {_n_chk} paths "
          f"of {_len_chk} bars")
    print(_cmp.round(4).to_string(index=False))

    _sd_ratio = _cmp.loc[0, "sd"] / _cmp.loc[1, "sd"]
    _ac_diff = abs(_cmp.loc[0, "acf_abs1"] - _cmp.loc[1, "acf_abs1"])
    print(f"\n  [{'PASS' if 0.85 < _sd_ratio < 1.18 else 'FAIL'}] "
          f"median path sd ratio (ours / arch) = {_sd_ratio:.3f} (tol 0.85-1.18)")
    print(f"  [{'PASS' if _ac_diff < 0.06 else 'FAIL'}] "
          f"|difference| in median ACF(1) of |returns| = {_ac_diff:.4f} (tol 0.06)")
    if not (0.85 < _sd_ratio < 1.18 and _ac_diff < 0.06):
        print("\n  WARNING: the pipeline disagrees with arch's own simulator. Treat every")
        print("  GARCH-family generator below as suspect until this is resolved.")
    print(f"\nPersistence of this model is {_p['persistence']:.4f}"
          + (" (CAPPED below 1 for simulation)" if _p["capped"] else "")
          + ". The closer that sits to 1, the more the")
    print("volatility level of an individual path wanders, and the wider the spread")
    print("of per-path sd around the median - which is why medians are compared.")
else:
    print("SKIPPED simulation cross-check (no fitted volatility model available)")

# %% [markdown]
# ### Building and freezing the generator registry
#
# Each generator is now instantiated with its calibration attached. A generator
# whose prerequisites are missing is registered as **unavailable with a stated
# reason** rather than dropped, so the final report can distinguish "this method
# found no problem" from "this method never ran".

# %%
def build_generators() -> dict[str, Generator]:
    """Instantiate every generator against the calibration slice.

    All calibration happens here, once, from market data only. After this
    function returns, generators are frozen: nothing downstream re-fits them,
    and in particular nothing re-fits them in response to strategy results.
    """
    gens: dict[str, Generator] = {}
    r = CALIB_LOGRET

    def add(g):
        gens[g.name] = g

    # --- A: IID -----------------------------------------------------------
    add(Generator("iid_bootstrap", "bootstrap",
                  "exact marginal distribution", "all temporal dependence",
                  _sample_iid, {"returns": r}))

    # --- B/C: block family ------------------------------------------------
    for L in BLOCK_LENGTHS:
        if L >= len(r) // 4:
            continue
        add(Generator(f"moving_block_L{L}", "bootstrap",
                      f"dependence within {L} bars", f"dependence beyond {L}; endpoint bias",
                      _sample_moving_block, {"returns": r, "block_length": int(L)}))
        add(Generator(f"stationary_L{L}", "bootstrap",
                      f"dependence over geometric blocks (mean {L})",
                      "long-range dependence", _sample_stationary,
                      {"returns": r, "block_length": int(L)}))
    _Lc = int(np.median(BLOCK_LENGTHS))
    add(Generator(f"circular_block_L{_Lc}", "bootstrap",
                  f"dependence within {_Lc} bars, no endpoint bias",
                  f"dependence beyond {_Lc}; one artificial splice per wrap",
                  _sample_circular_block, {"returns": r, "block_length": _Lc}))

    # --- shock-preserving -------------------------------------------------
    L_s = int(np.median(BLOCK_LENGTHS))
    thr = CALIB_SHOCK_THR              # calibration slice only - see the note above
    is_shock = np.abs(r) >= thr
    starts = np.arange(len(r) - L_s + 1)
    has_shock = np.array([is_shock[s:s + L_s].any() for s in starts])
    add(Generator(f"shock_block_L{L_s}", "bootstrap",
                  "shock frequency, magnitude, clustering and aftermath",
                  f"dependence beyond {L_s} bars",
                  _sample_shock_block,
                  {"returns": r, "block_length": L_s,
                   "shock_starts": starts[has_shock], "calm_starts": starts[~has_shock],
                   "shock_block_frac": float(has_shock.mean()), "threshold": float(thr)}))

    # --- regime family ----------------------------------------------------
    if REGIME_PRIMARY_METHOD == "hmm" and "model" in PRIMARY_REGIMES:
        m = PRIMARY_REGIMES["model"]
        labels = PRIMARY_REGIMES["labels"]
        pools = []
        ok = True
        for k in range(N_REGIMES):
            pool = r[:len(labels)][labels == k]
            if len(pool) < 20:
                ok = False
            pools.append(pool if len(pool) >= 5 else r)
        add(Generator("regime_bootstrap", "bootstrap",
                      "regime persistence + real per-regime return distribution",
                      "within-regime serial dependence",
                      _sample_regime_bootstrap,
                      {"A": m["A"], "pi": m["pi"], "pools": pools},
                      note="" if ok else "some regimes had few observations; pools are thin"))
        add(Generator("markov_regime_gaussian", "model",
                      "regime persistence, per-regime mean and variance",
                      "within-regime fat tails (Gaussian emissions)",
                      _sample_markov_gaussian,
                      {"A": m["A"], "pi": m["pi"],
                       "mu": m["mu"] / 100.0, "sigma": m["sigma"] / 100.0}))
    else:
        for nm in ("regime_bootstrap", "markov_regime_gaussian"):
            add(Generator(nm, "bootstrap", "-", "-", None, {}, available=False,
                          error=f"requires the HMM regime fit; primary method is "
                                f"{REGIME_PRIMARY_METHOD!r}"))

    # --- volatility models ------------------------------------------------
    if VOL_MODELS and BEST_VOL_MODEL:
        for tag, p in VOL_MODELS.items():
            add(Generator(f"{tag.lower().replace('-', '_')}", "model",
                          "volatility clustering and persistence"
                          + (", leverage effect" if p["kind"] in ("GJR", "EGARCH") else ""),
                          "trend structure; exact marginal shape",
                          _sample_vol_model, p))
        pb = VOL_MODELS[BEST_VOL_MODEL]
        add(Generator("filtered_historical_sim", "model",
                      "GARCH volatility dynamics + empirical innovation distribution",
                      "the original ordering of the innovations",
                      _sample_fhs, {**pb, "resid_block": 10}))

        # Separate the diffusion from the jumps before fitting either, so the
        # two components do not both claim the same historical crashes.
        r_winsor = np.clip(r, -thr, thr)
        excess = r - r_winsor                       # non-zero only on shock bars
        jump_pool = excess[excess != 0.0]
        try:
            spec = {"GARCH": dict(vol="GARCH", p=1, o=0, q=1),
                    "GJR": dict(vol="GARCH", p=1, o=1, q=1),
                    "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1)}[pb["kind"]]
            pj, _ = fit_vol_model(r_winsor * 100.0, pb["kind"], spec, pb["dist"],
                                  "jump-diffusion-base")
            add(Generator("parametric_jump_diffusion", "model",
                          "calibrated AR drift, GARCH volatility, Poisson jumps",
                          "anything outside the parametric form",
                          _sample_parametric,
                          {**pj, "jump_intensity": CALIB_SHOCK_FREQ,
                           "jump_pool": jump_pool if len(jump_pool) > 10 else r},
                          note=f"diffusion fitted to returns winsorised at "
                               f"+/-{thr:.4f}; {len(jump_pool)} jump excesses"))
        except Exception as exc:
            add(Generator("parametric_jump_diffusion", "model", "-", "-", None, {},
                          available=False,
                          error=f"winsorised base fit failed: {type(exc).__name__}: {exc}"))
    else:
        for nm in ("garch_normal", "garch_t", "gjr_t", "egarch_t",
                   "filtered_historical_sim", "parametric_jump_diffusion"):
            add(Generator(nm, "model", "-", "-", None, {}, available=False,
                          error="`arch` not installed"))

    # --- surrogates -------------------------------------------------------
    add(Generator("fourier_surrogate", "surrogate",
                  "power spectrum, hence all linear autocorrelation",
                  "non-linear structure; marginal driven toward Gaussian",
                  _sample_fourier, {"returns": r}))
    add(Generator("iaaft_surrogate", "surrogate",
                  "power spectrum AND the exact marginal distribution",
                  "non-linear dependence (volatility clustering)",
                  _sample_iaaft, {"returns": r, "iaaft_iters": IAAFT_ITERS}))

    # --- trend-preserving -------------------------------------------------
    segs = _segment_trends(r, REGIME_FEATURE_WINDOW)
    up_list = [s["returns"] for s in segs if s["up"]]
    dn_list = [s["returns"] for s in segs if not s["up"]]
    if len(segs) >= 20 and up_list and dn_list:
        frac_up = len(up_list) / len(segs)
        add(Generator("trend_preserving", "model",
                      "real trend segments: durations, magnitudes and reversal rates",
                      "the historical ORDER of trends (segments are re-chained)",
                      _sample_trend_preserving,
                      {"up_segments": up_list, "down_segments": dn_list,
                       "p_up_after_dn": frac_up, "p_dn_after_up": 1.0 - frac_up,
                       "n_segments": len(segs)},
                      note=f"{len(up_list)} up / {len(dn_list)} down segments, "
                           f"mean length {np.mean([len(s['returns']) for s in segs]):.1f} bars"))
    else:
        add(Generator("trend_preserving", "model", "-", "-", None, {}, available=False,
                      error=f"only {len(segs)} trend segments found; need >= 20"))

    return gens


GENERATORS = build_generators()

_avail = {k: g for k, g in GENERATORS.items() if g.available}
_unavail = {k: g for k, g in GENERATORS.items() if not g.available}
print(f"{len(_avail)} generator(s) available, {len(_unavail)} unavailable\n")
print(pd.DataFrame([
    {"generator": g.name, "family": g.family, "preserves": g.preserves[:52]}
    for g in _avail.values()]).to_string(index=False))
if _unavail:
    print("\nUNAVAILABLE (reported, not silently replaced):")
    for g in _unavail.values():
        print(f"  {g.name:<30} {g.error}")

# %%
# Smoke-test every available generator on a short path before the real run, so a
# broken generator fails here with its own name attached rather than deep inside
# a 1,000-path loop.
_rng_smoke = get_rng("generator_smoke_test")
_smoke_fail = []
for _nm, _g in list(_avail.items()):
    try:
        _p = _g(_rng_smoke, 3, 600)
        assert _p.shape == (3, 600)
        assert np.isfinite(_p).all()
        if np.allclose(_p[0], _p[1]):
            raise AssertionError("paths 0 and 1 are identical - generator is not random")
        _ann = np.expm1(_p).std() * math.sqrt(BPY)
        print(f"  [PASS] {_nm:<30} ann.vol {_ann:6.1%}  "
              f"kurt {stats.kurtosis(np.expm1(_p).ravel()):7.2f}")
    except Exception as exc:
        print(f"  [FAIL] {_nm:<30} {type(exc).__name__}: {exc}")
        _smoke_fail.append(_nm)
        _g.available, _g.error = False, f"smoke test failed: {exc}"

if _smoke_fail:
    print(f"\n{len(_smoke_fail)} generator(s) failed the smoke test and have been marked")
    print("unavailable. They will appear as gaps in the report, not as silent omissions.")
else:
    print(f"\nall {len(_avail)} generators passed the smoke test")
