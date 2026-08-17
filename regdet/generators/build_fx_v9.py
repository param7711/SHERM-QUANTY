"""Build fx_v9_<SYM>.ipynb -- the SHIPPED v9 master notebook run on FX PAIRS.

The user's request, restated plainly: they wanted the v9 final master TESTED ON
FOREX -- EUR/USD, XAU/USD and the rest -- not on Nifty.

  FX_INSTRUMENT=EURUSD python3 build_fx_v9.py      # one pair
  (see INSTRUMENTS below for the 7 available)

WHAT DIFFERS FROM THE SHIPPED MASTER, exhaustively:

  1. THE DATA CELL (cell 5). It defines exactly three globals -- `nifty`,
     `vix`, `TAC_SYNTH` -- so substituting an FX close series and a causal
     realised-vol proxy is a clean swap with no downstream edits.
  2. FIVE DISPLAY STRINGS in the chart cells. These are axis labels and legend
     entries only ('Nifty 50 close' -> 'EUR/USD close', 'India VIX' ->
     'realised-vol proxy', ...). They touch NO number. They exist because the
     first version of this notebook was CORRECT but MISLEADING: the charts
     plotted EUR/USD at 1.08-1.22 while the y-axis still said "Nifty 50
     close", and the user reasonably read that as "you only showed me Nifty".
     A chart that lies about its own instrument is worse than a chart with a
     patched label. The substitution list is asserted to be display-only:
     the set of cells it modifies is checked against a hard-coded allowlist.

NOT ONE CONSTANT IS RE-TUNED. Nifty 2h has 3 bars per NSE session; FX at 8h has
3 bars per day (24/8). So BARS_PER_DAY = 3 stays LITERALLY CORRECT for both,
every day-denominated window keeps the same meaning, and the master's own
`assert MOM_3D_BARS == BASE_WIN['MOM_3D'] == 9` passes untouched. The 8h
cadence was chosen for exactly that reason.

BAR COUNT IS MATCHED to the user's Nifty run (2,858). Every earlier FX chart in
this project drew 57,311 bars into one panel -- 20x the density -- and at that
density regime shading merges into stripes REGARDLESS of label quality. That
rendering artefact polluted the "FX barcodes" reading, so it is removed here.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER_GEN = f'{HERE}/build_master_notebook_v2.py'

# The 7 non-redundant instruments in the repo (crosses excluded -- they are
# arithmetic on the majors, corr +0.989..+0.9985 vs triangular synthetics).
INSTRUMENTS = {
    'EURUSD': ('EUR/USD', (0.9, 1.7),     'FX'),
    'GBPUSD': ('GBP/USD', (0.9, 1.7),     'FX'),
    'USDJPY': ('USD/JPY', (75.0, 160.0),  'FX'),
    'XAUUSD': ('XAU/USD', (1000., 2100.), 'COMMODITY (gold, quoted in USD)'),
    'AUDUSD': ('AUD/USD', (0.55, 1.15),   'FX'),
    'USDCAD': ('USD/CAD', (0.9, 1.6),     'FX'),
    'USDCHF': ('USD/CHF', (0.7, 1.15),    'FX'),
}
DEFAULT_SYM = os.environ.get('FX_INSTRUMENT', 'EURUSD').upper()
assert DEFAULT_SYM in INSTRUMENTS, f'unknown {DEFAULT_SYM}; pick from {list(INSTRUMENTS)}'

OUT = os.path.join(
    os.environ.get('REGDET_OUT_DIR',
                   os.path.join(os.path.dirname(HERE), 'notebooks')),
    'fx_v9.ipynb')

_gsrc = open(MASTER_GEN).read()
assert "regdet_v11_master.ipynb'" in _gsrc
_gsrc = _gsrc.replace("regdet_v11_master.ipynb'", "_fx_v9_harvest.ipynb'", 1)
_ns = {'__name__': '_master_harvest', '__file__': MASTER_GEN}
exec(compile(_gsrc, MASTER_GEN, 'exec'), _ns)
cells = [dict(c) for c in _ns['cells']]
try:
    os.remove(os.path.join(os.path.dirname(OUT), '_fx_v9_harvest.ipynb'))
except OSError:
    pass

IDX_LOAD = 5
_orig = ''.join(cells[IDX_LOAD]['source'])
assert 'def load_2h' in _orig and 'nifty, vix, TAC_SYNTH = load_2h()' in _orig, \
    'cell 5 is not the data loader -- refusing to substitute blind'

# The 8h-FX-cadence match is the reason no constant needs changing. If a future
# edit to the master moves BARS_PER_DAY off 3, that premise silently breaks.
_const = ''.join(cells[3]['source'])
assert 'BARS_PER_DAY   = 3' in _const, \
    'master BARS_PER_DAY is no longer 3 -- the 8h FX cadence match is VOID'

# ---------------------------------------------------------------------------
# COSMETIC LABEL SUBSTITUTION -- display strings only, asserted as such.
# Note 'nifty_close' (the COLUMN NAME) is deliberately absent: it is code, not
# a label, and every quoted form below carries its quotes so it cannot match.
# ---------------------------------------------------------------------------
LABEL_SUBS = [
    ("'Nifty 50 close'", "f'{FX_NAME} close'"),
    ("'Nifty 50 (2h)'",  "f'{FX_NAME} (8h)'"),
    ("'Nifty close'",    "f'{FX_NAME} close'"),
    ("'India VIX'",      "'realised-vol proxy (NOT a VIX)'"),
    ("'Nifty'",          "f'{FX_NAME}'"),
]
LABEL_CELLS_ALLOWED = {27, 40, 42, 60, 66}
_touched = set()
for i, c in enumerate(cells):
    if c['cell_type'] != 'code' or i == IDX_LOAD:
        continue
    s = ''.join(c['source'])
    new = s
    for a, b in LABEL_SUBS:
        new = new.replace(a, b)
    if new != s:
        _touched.add(i)
        cells[i] = dict(c, source=new.splitlines(keepends=True))
assert _touched <= LABEL_CELLS_ALLOWED, (
    f'label substitution reached unexpected cells {_touched - LABEL_CELLS_ALLOWED} '
    '-- refusing to ship a patch whose blast radius is not understood')
print(f'cosmetic label substitution touched cells {sorted(_touched)} '
      f'(allowlist {sorted(LABEL_CELLS_ALLOWED)})')

FX_LOADER = f'''
# ===========================================================================
# DATA LAYER -- THE ONLY FUNCTIONAL CELL THAT DIFFERS FROM THE SHIPPED MASTER.
# Defines the same three globals the original defines: nifty, vix, TAC_SYNTH.
#
# WHY 8-HOUR BARS: Nifty 2h = 3 bars per NSE session. FX 8h = 3 bars per day
# (24/8). BARS_PER_DAY = 3 is therefore LITERALLY CORRECT for both, and the
# master's own `assert MOM_3D_BARS == BASE_WIN['MOM_3D'] == 9` passes
# UNTOUCHED. Not one constant is re-tuned anywhere in this notebook.
#
# WHY THE BAR COUNT IS MATCHED: earlier FX charts drew 57,311 bars into one
# panel -- 20x Nifty's density -- which produces stripes REGARDLESS of regime
# quality. Matching n makes the visual comparison honest.
# ===========================================================================
# >>>>>>>>>>>>>>>>>>>>>>  CHANGE THIS ONE LINE  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
FX_INSTRUMENT = {DEFAULT_SYM!r}
# Options: 'EURUSD' 'GBPUSD' 'USDJPY' 'XAUUSD' 'AUDUSD' 'USDCAD' 'USDCHF'
# Re-run the whole notebook after changing it. Every chart, table, label and
# scorecard below repoints automatically -- nothing else needs editing.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
FX_INSTRUMENTS = {INSTRUMENTS!r}
assert FX_INSTRUMENT in FX_INSTRUMENTS, (
    f'unknown instrument {{FX_INSTRUMENT!r}}; pick one of '
    f'{{list(FX_INSTRUMENTS)}}')
FX_SYM = FX_INSTRUMENT
FX_NAME, FX_MED_BAND, FX_CLASS = FX_INSTRUMENTS[FX_SYM]
FX_HOURS = 8            # 24 / 8 = 3 bars per day == Nifty's 3 bars per session
MATCH_BARS = 2858       # the Nifty v9 run this is being compared against
FX_URL = ('https://raw.githubusercontent.com/ejtraderLabs/historical-data/'
          f'main/{{FX_SYM}}/{{FX_SYM}}h1.csv')
FX_SOURCE_LABEL = ('COMMUNITY GITHUB DATA (ejtraderLabs/historical-data) - '
                   'verified against Brexit/COVID, NOT an official feed')


def _fx_load_2h():
    """FX 1h -> 8h (DOWNWARD only), scale-detected, matched to MATCH_BARS.

    Same return contract as the master's load_2h(): (close, vol, is_synth).
    """
    df = pd.read_csv(FX_URL, parse_dates=['Date'])
    assert list(df.columns) == ['Date', 'open', 'high', 'low', 'close',
                                'tick_volume'], f'unexpected columns {{list(df.columns)}}'
    df = df.set_index('Date').sort_index()
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert (df[['open', 'high', 'low', 'close']] > 0).values.all()

    # Integer scale DETECTED, never hardcoded.
    med = float(np.median(df['close'].values))
    hits = [d for d in [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
            if FX_MED_BAND[0] <= med / d <= FX_MED_BAND[1]]
    assert len(hits) == 1, f'scale detection AMBIGUOUS for {{FX_SYM}}: {{hits}}'
    div = hits[0]
    px = df[['open', 'high', 'low', 'close']] / div

    # DOWNWARD aggregation only -- asserted, never assumed.
    step_in = pd.Series(px.index).diff().median()
    bars = px.resample(f'{{FX_HOURS}}h').agg(
        {{'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}}
    ).dropna(how='any')
    step_out = pd.Series(bars.index).diff().median()
    assert step_out > step_in, f'UPWARD resample attempted: {{step_in}} -> {{step_out}}'

    realised_bpd = float(pd.Series(1, index=bars.index)
                         .groupby(bars.index.normalize()).sum().median())
    close_full = bars['close'].astype(float)

    # The vol PROXY is computed on FULL history then sliced. It reads bars <= t
    # only (rolling sd, no shift, no centring, NEVER bfill), so slicing after
    # cannot leak; doing it the other way would burn VOL_SLOW bars of the
    # matched window on warm-up for nothing.
    r = np.log(close_full / close_full.shift(1))
    vol_full = (r.rolling(int(BASE_WIN['VOL_SLOW'])).std()
                * np.sqrt(max(BARS_PER_DAY, 1) * 252.0) * 100.0)
    vol_full = vol_full.replace([np.inf, -np.inf], np.nan)

    close = close_full.iloc[-MATCH_BARS:]
    vol = vol_full.reindex(close.index)
    close.name, vol.name = 'nifty', 'vix'   # downstream reads these NAMES

    print(f'FX {{FX_NAME}} 1h -> {{FX_HOURS}}h: {{len(close_full)}} bars available, '
          f'using the most recent {{len(close)}} to match the Nifty v9 run')
    print(f'  divisor DETECTED = {{div:g}}   realised bars/day = {{realised_bpd:.2f}} '
          f'(BARS_PER_DAY assumed {{BARS_PER_DAY}})')
    print(f'  volatility input = causal realised-vol PROXY. '
          f'{{"Gold" if FX_SYM == "XAUUSD" else "Spot FX"}} has NO VIX. This is NOT a VIX.')
    return close, vol, False


nifty, vix, TAC_SYNTH = _fx_load_2h()

print()
print('=' * 100)
print('*** THIS IS NOT NIFTY. ***')
print(f'*** Instrument : {{FX_NAME}}   ({{FX_CLASS}}, 8h bars, 3 per day)')
print(f'*** Provenance : {{FX_SOURCE_LABEL}}')
print('*** DATA ENDS 2022-03-04. This notebook says NOTHING about 2022-2026.')
print('***')
print('*** NOT ONE CONSTANT is changed from the shipped v9 master. Only this')
print('*** data cell differs functionally; five axis/legend LABELS were also')
print('*** repointed at this instrument so the charts do not claim to be Nifty.')
print('*** Prose in later sections still describes NIFTY episodes -- that text')
print('*** is left alone deliberately, because editing it would forfeit the')
print('*** "unmodified" property. Read the numbers, discount that prose.')
print('=' * 100)
print(f"\\nSpan: {{nifty.index[0]}}  ->  {{nifty.index[-1]}}")
'''

cells[IDX_LOAD] = {"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": FX_LOADER.splitlines(keepends=True)}

HEADER = f"""
# RegDet v9 — the SHIPPED master, run on **FX pairs**

*Same notebook, same 15 charts, same scorecard, **same constants** as the Nifty
run — only the data differs.*

### ▶ To switch instrument: change **one line** at the top of the data cell

```python
FX_INSTRUMENT = 'EURUSD'   # 'GBPUSD' 'USDJPY' 'XAUUSD' 'AUDUSD' 'USDCAD' 'USDCHF'
```

Re-run the notebook. Every chart, axis label, table and scorecard repoints
automatically. Seven instruments, one file, nothing else to edit.

### Why `BARS_PER_DAY` did not have to change

Nifty 2h = 3 bars per NSE session. FX 8h = 3 bars per day (24 ÷ 8). So
`BARS_PER_DAY = 3` is *literally correct* for both, every day-denominated
window keeps its meaning, and the master's own `assert MOM_3D_BARS == 9`
passes untouched. **Nothing is re-tuned for this instrument.**

### Why the bar count is matched to 2,858

| | Nifty v9 run | earlier FX notebooks | **this notebook** |
|---|---|---|---|
| bars per day | 3 (2h × session) | 24 (1h) | **3 (8h) — matched** |
| bars drawn per chart | 2,858 | 57,311 | **~2,850 — matched** |
| constants re-tuned | — | none | **none** |

57,311 bars in one panel is **20× the density** of 2,858, and at that density
regime shading merges into stripes whatever the labels say. Part of the earlier
"FX barcodes" impression was *plotting*, not detection. Matching the bar count
removes that confound.

### The Nifty numbers this is graded against

From the user's own `final_regdet_v9.ipynb` Kaggle run (2,858 bars, 2023-09 →
2026-07, real yfinance data) — printed **before** any FX number exists, so
nothing below can be graded generously after the fact:

| | Nifty v9 |
|---|---|
| forward-return ordering | **BROKEN at all 3 horizons** |
| median run length | H_BULL 5.0, L_BULL 2.0, SIDEWAYS 5.0, L_BEAR 3.0, H_BEAR 5.0 |
| occupancy | 20.0 / 14.8 / 28.8 / 16.3 / 20.1 % |
| scorecard (7A, 70% fit) | FAIL 29, PASS 9, NA 6, WARN 4, **FAIL(ANTI) 2** |
| regime long/flat vs B&H | Sharpe **0.41 vs 0.60** (underperforms) |

**Caveat, stated up front:** the FX window is 2018→2022 and the Nifty window is
2023→2026. Market *era* is therefore confounded with market. Comparisons below
are read with that in mind.
"""
cells.insert(0, {"cell_type": "markdown", "metadata": {},
                 "source": HEADER.strip('\n').splitlines(keepends=True)})

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'wrote {OUT} - {len(cells)} cells; default instrument {DEFAULT_SYM}, switchable in-notebook')
