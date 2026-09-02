"""
Hourly NSE bars, built from 1-minute prints.

SOURCE.  github.com/aeron7/nifty-banknifty-intraday-data -- one text file
per symbol per session, 2012-2021, laid out as

    SYMBOL,YYYYMMDD,HH:MM,open,high,low,close,volume,open_interest

RESAMPLING.  NSE's cash session runs 09:15-15:30, so TradingView anchors
hourly bars to the open, not to the clock: 09:15, 10:15, 11:15, 12:15,
13:15, 14:15 and a short 15:15 stub.  Seven bars a day.  Anything printed
outside 09:15-15:29 (pre-open crosses, block windows) is dropped rather
than folded into the first or last bar.

WHAT THIS DATA IS NOT.  These are raw exchange prints: unadjusted for
splits, bonuses and demergers.  An unadjusted 1:1 bonus is a clean -50%
gap that no band model can tell from a crash, and it would manufacture a
spectacular fake long.  `split_dates` finds them by looking for overnight
ratios sitting on a known corporate-action fraction, and `load` trims each
series to start after the last one -- the same screen the daily pass used.
"""

import os
import glob
import subprocess
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = '/home/user/aeron7/nifty-banknifty-intraday-data'
OUT_DIR = os.path.join(ROOT, 'data', 'research_1h')
LIST = os.path.join(OUT_DIR, '_files.lst')

INDICES = ['NIFTY', 'BANKNIFTY']
STOCKS = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'ITC', 'LT',
          'AXISBANK', 'KOTAKBANK', 'BHARTIARTL', 'SBIN', 'MARUTI',
          'HINDUNILVR', 'ASIANPAINT', 'BAJFINANCE']
SYMBOLS = INDICES + STOCKS

COLS = ['sym', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'oi']

SPLIT_RATIOS = (0.5, 0.2, 0.1, 0.25, 1 / 3, 2 / 3, 0.05)
SPLIT_TOL = 0.03


def build_file_list() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(LIST):
        with open(LIST, 'w') as fh:
            subprocess.run(['find', '.', '-type', 'f', '-name', '*.txt'],
                           cwd=SRC, stdout=fh, check=True)
    return LIST


def _raw(sym: str) -> pd.DataFrame:
    """Every 1-minute print for one symbol, concatenated and de-duplicated."""
    build_file_list()
    with open(LIST) as fh:
        paths = [ln.strip() for ln in fh if ln.strip().endswith(f'/{sym}.txt')]
    if not paths:
        raise FileNotFoundError(sym)
    frames = []
    for p in paths:
        try:
            d = pd.read_csv(os.path.join(SRC, p), header=None, names=COLS,
                            dtype={'sym': str, 'date': str, 'time': str})
        except Exception:
            continue
        if len(d):
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df['sym'].str.upper() == sym]
    ts = pd.to_datetime(df['date'] + ' ' + df['time'],
                        format='%Y%m%d %H:%M', errors='coerce')
    df = df.assign(ts=ts).dropna(subset=['ts'])
    df = df.set_index('ts').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    for c in ('open', 'high', 'low', 'close', 'volume'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df[df[['open', 'high', 'low', 'close']].gt(0).all(axis=1)]


OPEN_MIN = 555      # 09:15, the NSE session open
LATE_MIN = 570      # 09:30, skipping the opening auction and the gap burst


def to_hourly(m: pd.DataFrame, anchor: int = OPEN_MIN) -> pd.DataFrame:
    """1-minute prints -> hourly bars anchored on `anchor` minutes past midnight.

    Anchor 555 (09:15) gives the seven bars TradingView draws.  Anchor 570
    (09:30) drops the first fifteen minutes of every session outright --
    the opening auction and the burst of gap-driven prints that follow it
    -- and leaves six clean bars, 09:30 through 15:29.  That is the only
    way to express "no trades before 09:30" on hourly bars: the 09:15
    anchor's first bar runs to 10:14, so blocking it costs a whole hour of
    tradeable session.
    """
    mins = m.index.hour * 60 + m.index.minute
    m = m[(mins >= anchor) & (mins <= 929)]              # anchor .. 15:29
    idx = (m.index.hour * 60 + m.index.minute - anchor) // 60
    bucket = (m.index.normalize()
              + pd.to_timedelta(anchor + 60 * idx, unit='m'))
    g = m.groupby(bucket)
    h = pd.DataFrame({
        'open': g['open'].first(),
        'high': g['high'].max(),
        'low': g['low'].min(),
        'close': g['close'].last(),
        'volume': g['volume'].sum(),
        'ticks': g['close'].size(),
    })
    h.index.name = 'ts'
    # A bar built from one or two stray prints is not an hour of trading.
    return h[h['ticks'] >= 5].drop(columns='ticks')


DISLOCATION = 0.20     # a bar this far from its own session's level is a bad print


def clean_sessions(h: pd.DataFrame, tol: float = DISLOCATION) -> tuple:
    """Drop whole sessions that contain a dislocated print.

    The source files carry occasional foreign prices -- a BANKNIFTY bar
    with a low of 1439 inside a session trading at 18400, an ASIANPAINT
    bar 89% below its neighbours.  These are not moves, they are another
    instrument's prints landing in the wrong file, and a band model reads
    them as the trade of the decade.

    A real intraday move, however violent, leaves the session's price
    LEVEL somewhere near itself; a bad print does not.  So the test is
    per session: take the median of that session's closes and drop the
    whole session if any bar strays more than `tol` from it.  Dropping the
    session rather than the bar means no repaired stub is left behind for
    a signal to form on.

    This uses the full session -- it is data repair, not a trading rule,
    and no signal is computed until after it has run.
    """
    day = h.index.normalize()
    med = h.groupby(day)['close'].transform('median')
    lo = h.groupby(day)['low'].transform('min')
    hi = h.groupby(day)['high'].transform('max')
    off = ((h['close'] / med - 1).abs() > tol) | ((lo / med - 1).abs() > tol) \
        | ((hi / med - 1).abs() > tol)
    bad = sorted(set(day[off]))
    return h[~day.isin(bad)], [str(d.date()) for d in bad]


MAX_GAP_DAYS = 10      # a hole longer than this is a break in the record
SPIKE = 0.06           # a move this big, immediately undone, is contamination


def segments(h: pd.DataFrame, max_gap: int = MAX_GAP_DAYS) -> list:
    """Split the record wherever the calendar jumps.

    The source repo is not contiguous: most stock files run 2013-06 to
    2014-06, then stop dead until 2016-09.  A 400-bar regression fitted
    across that hole is fitted across a two-year price move that never
    appeared on any chart, and every band it draws is fiction.  Worse,
    splits that happened inside the hole (ICICIBANK 1:5, SBIN 1:10) never
    show up as a single-day ratio, so the split screen cannot see them.

    So the record is cut at every gap and only whole segments are used.
    """
    days = pd.Index(sorted(set(h.index.normalize())))
    if len(days) == 0:
        return []
    brk = np.where(np.diff(days.values).astype('timedelta64[D]').astype(int) > max_gap)[0]
    bounds = np.r_[0, brk + 1, len(days)]
    return [(days[bounds[i]], days[bounds[i + 1] - 1])
            for i in range(len(bounds) - 1)]


def longest_segment(h: pd.DataFrame, max_gap: int = MAX_GAP_DAYS) -> pd.DataFrame:
    segs = segments(h, max_gap)
    if not segs:
        return h
    a, b = max(segs, key=lambda s: (s[1] - s[0]).days)
    return h[(h.index >= a) & (h.index <= b + pd.Timedelta(days=1))]


def drop_spike_sessions(h: pd.DataFrame, thr: float = SPIKE) -> tuple:
    """Drop sessions holding a move that is undone by the very next bar.

    Around some bonus dates the file interleaves adjusted and unadjusted
    prints, so the close alternates between two levels ten percent apart.
    A genuine move does not round-trip in one hour; this does.
    """
    r = h['close'].pct_change()
    a, b = r.to_numpy(), np.r_[r.to_numpy()[1:], np.nan]
    osc = (np.abs(a) > thr) & (np.abs(b) > thr) & (np.sign(a) != np.sign(b))
    day = h.index.normalize()
    bad = sorted(set(day[np.nan_to_num(osc).astype(bool)]))
    return h[~day.isin(bad)], [str(d.date()) for d in bad]


def split_dates(h: pd.DataFrame) -> list:
    """Session-boundary gaps that land on a corporate-action ratio."""
    daily = h['close'].groupby(h.index.normalize()).last()
    ratio = daily / daily.shift(1)
    hits = []
    for d, r in ratio.dropna().items():
        for target in SPLIT_RATIOS:
            if abs(r - target) <= SPLIT_TOL * target:
                hits.append((d, float(r), target))
                break
    return hits


def drop_level_breaks(h: pd.DataFrame, tol: float = 0.30, win: int = 11) -> tuple:
    """Drop whole sessions sitting at a price level their neighbours reject.

    `clean_sessions` compares each bar to its own session's median, so it
    is blind to a session in which EVERY bar is wrong -- and those exist:
    BANKNIFTY 2015-06-24 is six consecutive bars around 1440 while the
    index was trading near 18,400.  A whole foreign day is invisible from
    the inside; it is only visible against the days on either side.

    Runs after back-adjustment, so a real split has already been removed
    and any level break left is an error rather than a corporate action.
    """
    day = h.index.normalize()
    daily = h['close'].groupby(day).median()
    ref = daily.rolling(win, center=True, min_periods=3).median()
    bad = daily.index[(daily / ref - 1).abs() > tol]
    return h[~day.isin(bad)], [str(d.date()) for d in bad]


def back_adjust(h: pd.DataFrame) -> tuple:
    """Undo unadjusted corporate actions by scaling the pre-event history.

    The observed overnight ratio carries that session's real return on top
    of the corporate action, so it is snapped to the nearest canonical
    fraction (0.504 -> 0.5) before use: adjusting by the raw ratio would
    smear one day's move back across every earlier bar.
    """
    sp = split_dates(h)
    if not sp:
        return h, []
    h = h.copy()
    applied = []
    for d, obs, target in sorted(sp):
        pre = h.index < d
        if not pre.any():
            continue
        h.loc[pre, ['open', 'high', 'low', 'close']] *= target
        applied.append({'date': str(d.date()), 'observed': round(obs, 4),
                        'applied': round(target, 4)})
    return h, applied


def build(symbols=None, verbose=True, anchor: int = OPEN_MIN,
          out_dir: str = None) -> pd.DataFrame:
    """Raw 1-minute files -> clean, contiguous, split-adjusted hourly bars."""
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sym in (symbols or SYMBOLS):
        m = _raw(sym)
        h0 = to_hourly(m, anchor)
        h0.to_parquet(os.path.join(out_dir, f'{sym}.parquet'))
        h = load(sym, out_dir=out_dir)
        segs = segments(h0)
        r = {'symbol': sym, 'minute_rows': len(m), 'hourly_raw': len(h0),
             'segments': len(segs), 'hourly_clean': len(h),
             'sessions': int(h.index.normalize().nunique()),
             'per_session': round(len(h) / max(h.index.normalize().nunique(), 1), 2),
             'start': str(h.index[0])[:10], 'end': str(h.index[-1])[:10],
             'splits': len(split_dates(longest_segment(h0))),
             'max_abs_ret': float(h['close'].pct_change().abs().max())}
        rows.append(r)
        if verbose:
            print(f'  {sym:<11} {len(h0):>6} raw -> {len(h):>6} clean  '
                  f'segs={len(segs)}  {r["start"]}..{r["end"]}  '
                  f'splits={r["splits"]}  max|ret|={r["max_abs_ret"]:.3f}')
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(out_dir, '_quality.csv'), index=False)
    return rep


def load(sym: str, adjust: str = 'back', out_dir: str = None) -> pd.DataFrame:
    """The cleaning pipeline, in the order the problems have to be solved.

    1. drop sessions containing a dislocated print (a foreign instrument)
    2. drop sessions whose prices oscillate between two adjusted levels
    3. keep only the longest gap-free stretch of the record
    4. back-adjust the splits that remain inside that stretch
    5. drop sessions whose whole level is rejected by their neighbours
    """
    h = pd.read_parquet(os.path.join(out_dir or OUT_DIR, f'{sym}.parquet'))
    h, _ = clean_sessions(h)
    h, _ = drop_spike_sessions(h)
    h = longest_segment(h)
    if adjust == 'back':
        h, _ = back_adjust(h)
    h, _ = drop_level_breaks(h)
    return h


if __name__ == '__main__':
    import sys
    a = LATE_MIN if '--late' in sys.argv else OPEN_MIN
    o = OUT_DIR + '_0930' if '--late' in sys.argv else None
    syms = [x for x in sys.argv[1:] if not x.startswith('--')] or None
    print(build(syms, anchor=a, out_dir=o).to_string(index=False))
