"""
Rolling linear-regression channel — the regime filter under test.

At every bar t the channel is fitted on the trailing window
close[t-N+1 .. t] only, and every quantity is read at the *endpoint* of
that fit. Nothing from bar t+1 onward touches bar t's numbers, so the
signal series is causal by construction.

    center_t : value of the fitted line at t
    sigma_t  : standard deviation of the fit residuals in the window
    z_t      : (close_t - center_t) / sigma_t      -> where price sits
    smaz_t   : (sma_t   - center_t) / sigma_t      -> where the SMA sits
    slope_t  : per-bar drift of the fitted line

The band at +/- k sigma is therefore |z| = k, and "the 20-SMA broke above
the channel" is smaz > k.
"""

import numpy as np
import pandas as pd


def regression_channel(close: pd.Series, n: int, ddof: int = 0) -> pd.DataFrame:
    """Rolling OLS of close on bar index over a trailing window of n bars.

    Computed on explicitly centred windows rather than from raw rolling
    sums: on price levels near 1e3-1e4 the sum-of-squares shortcut loses
    most of its significant digits to cancellation, and sigma is exactly
    the quantity that cancellation destroys.

    `ddof` sets the residual normalisation. DevLucem's "Linear Regression++"
    divides the summed squared residuals by the window length (ddof=0);
    ddof=2 is the textbook correction for the two fitted parameters. The
    two differ by sqrt(n/(n-2)) -- 1.0% at n=100 -- which shifts every band
    by the same factor and is worth matching when the goal is to reproduce
    what the TradingView chart drew.
    """
    y = close.to_numpy(dtype=float)
    t = y.shape[0]
    if t < n:
        raise ValueError(f'need at least {n} bars, got {t}')

    win = np.lib.stride_tricks.sliding_window_view(y, n)     # (t-n+1, n)
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    sxx = (xc * xc).sum()

    y_bar = win.mean(axis=1)
    yc = win - y_bar[:, None]
    slope = (yc @ xc) / sxx
    resid = yc - slope[:, None] * xc[None, :]
    sigma_v = np.sqrt((resid * resid).sum(axis=1) / (n - ddof))
    center_v = y_bar + slope * xc[-1]

    center = np.full(t, np.nan)
    sigma = np.full(t, np.nan)
    slope_full = np.full(t, np.nan)
    center[n - 1:] = center_v
    sigma[n - 1:] = np.where(sigma_v > 0, sigma_v, np.nan)
    slope_full[n - 1:] = slope

    return pd.DataFrame({
        'center': center,
        'sigma':  sigma,
        'slope':  slope_full,
        'z':      (y - center) / sigma,
    }, index=close.index)


def channel_features(close: pd.Series, n: int, sma_len: int,
                     ddof: int = 0, inner_sd: float = 0.3) -> pd.DataFrame:
    """Channel, the SMA's position inside it, and the SMA's own bands."""
    ch = regression_channel(close, n, ddof=ddof)
    sma = close.rolling(sma_len).mean()
    ch['sma'] = sma
    ch['smaz'] = (sma - ch['center']) / ch['sigma']

    # Bollinger-style bands on the SMA itself. Pine's ta.stdev is the
    # population standard deviation, so ddof=0 here.
    sd = close.rolling(sma_len).std(ddof=0)
    ch['bb_sd'] = sd
    ch['bbz'] = (close - sma) / sd.replace(0, np.nan)
    ch['inner_up'] = sma + inner_sd * sd
    ch['inner_dn'] = sma - inner_sd * sd
    return ch


def devlucem_reference(close: pd.Series, length: int = 100,
                       dev: float = 2.0) -> pd.DataFrame:
    """Literal transcription of Linear Regression ++ [Dev Lucem], Pine v5.

    Kept deliberately unvectorised and in Pine's own variable names so the
    two implementations can be diffed line by line:

        linreg   = ta.linreg(source, length, offset)
        linreg_p = ta.linreg(source, length, offset + 1)
        slope    = linreg - linreg_p
        intercept = linreg - x * slope
        deviationSum += pow(source[i] - (slope * (x - i) + intercept), 2)
        deviation = sqrt(deviationSum / length)

    Note the divisor: `length`, not `length - 2`. The bands are the
    population standard deviation of the residuals, so `ddof=0` is what
    matches the indicator on a chart.
    """
    y = close.to_numpy(dtype=float)
    t = y.shape[0]
    out = np.full((t, 4), np.nan)          # linreg, slope, deviation, z
    xs = np.arange(length, dtype=float)
    for b in range(length - 1, t):
        win = y[b - length + 1:b + 1]
        m, c = np.polyfit(xs, win, 1)      # ta.linreg is OLS on bar index
        linreg = m * (length - 1) + c      # offset 0 -> the endpoint
        linreg_p = m * (length - 2) + c    # offset 1 -> one bar back
        slope = linreg - linreg_p
        intercept = linreg - b * slope
        deviation_sum = 0.0
        for i in range(length):
            fitted = slope * (b - i) + intercept
            deviation_sum += (y[b - i] - fitted) ** 2
        deviation = np.sqrt(deviation_sum / length)
        out[b] = (linreg, slope, deviation,
                  (y[b] - linreg) / deviation if deviation > 0 else np.nan)
    return pd.DataFrame(out, columns=['center', 'slope', 'sigma', 'z'],
                        index=close.index)


def _pine_parity_check():
    """The vectorised channel must equal the Pine transcription exactly."""
    rng = np.random.default_rng(11)
    px = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, .01, 600))),
                   index=pd.date_range('2020-01-01', periods=600, freq='B'))
    length = 100
    ref = devlucem_reference(px, length).iloc[length - 1:]
    ours = regression_channel(px, length, ddof=0).iloc[length - 1:]
    for col in ('center', 'slope', 'sigma', 'z'):
        err = np.abs(ref[col].to_numpy() - ours[col].to_numpy()).max()
        assert err < 1e-8, (col, err)
        print(f'  {col:7s} max abs diff vs Pine transcription: {err:.2e}')

    # And the n-2 convention is a real, quantifiable difference, not noise.
    wide = regression_channel(px, length, ddof=2).iloc[length - 1:]
    ratio = (wide['sigma'] / ours['sigma']).dropna()
    print(f'  ddof=2 bands are {ratio.mean():.4f}x the DevLucem width '
          f'(= sqrt({length}/{length - 2}) = {np.sqrt(length / (length - 2)):.4f})')
    print('channel.py Pine parity check passed')


def _self_check():
    """A pure straight line must give ~zero residual and exact slope."""
    idx = pd.date_range('2020-01-01', periods=300, freq='B')
    line = pd.Series(100 + 0.5 * np.arange(300), index=idx)
    ch = regression_channel(line, 100)
    assert np.allclose(ch['slope'].dropna(), 0.5), ch['slope'].dropna().head()
    # Zero residual is degenerate and is deliberately mapped to NaN so that
    # z never divides by zero.
    assert ch['sigma'].iloc[99:].isna().all()

    # A known noisy case: compare against numpy.polyfit on the last window.
    rng = np.random.default_rng(0)
    noisy = pd.Series(100 + 0.2 * np.arange(300) + rng.normal(0, 2, 300), index=idx)
    n = 120
    ch = regression_channel(noisy, n)
    w = noisy.to_numpy()[-n:]
    b, a = np.polyfit(np.arange(n), w, 1)
    fit = a + b * np.arange(n)
    ref_sigma = np.sqrt(((w - fit) ** 2).sum() / (n - 2))
    assert abs(ch['slope'].iloc[-1] - b) < 1e-9
    assert abs(ch['center'].iloc[-1] - fit[-1]) < 1e-8
    assert abs(ch['sigma'].iloc[-1] - ref_sigma) < 1e-8

    # Causality: truncating the future must not change past values.
    full = regression_channel(noisy, n)
    trunc = regression_channel(noisy.iloc[:250], n)
    assert np.allclose(full['z'].iloc[:250].dropna(),
                       trunc['z'].dropna(), equal_nan=True)
    print('channel.py self-check passed (slope, center, sigma, causality)')
    _pine_parity_check()


if __name__ == '__main__':
    _self_check()
