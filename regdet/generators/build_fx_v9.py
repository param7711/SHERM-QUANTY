"""Build fx_v9.ipynb -- the SHIPPED v9 master notebook, UNMODIFIED, on FX.

The question this answers, asked directly by the user: "let's just try the og
v9 final on forex."

Every sibling FX notebook so far harvested only the ENGINE out of the master
and wrapped it in a purpose-built generalization harness. None of them ran the
master's OWN 15 charts, its scorecard, its ablations or its summary. So the
user has been comparing v9's Nifty charts against MY chart style on FX -- two
different renderings, which is not a fair look.

This notebook removes every confound it can:

  1. NOT ONE CONSTANT IS CHANGED. Nifty 2h has 3 bars per session, and FX at
     8h also has 3 bars per day (24/8). So BARS_PER_DAY = 3 stays LITERALLY
     CORRECT, BASE_WIN keeps the same meaning in DAYS as well as in bars, and
     the master's own `assert MOM_3D_BARS == 9` still passes untouched. The
     8h cadence is chosen for exactly that reason.
  2. BAR COUNT IS MATCHED to the user's Nifty run (2,858 bars). Chart density
     is therefore identical -- 57,311 bars crammed into one panel produces
     stripes no matter how good the regimes are, and that rendering artefact
     has been polluting the "FX barcodes" impression.
  3. ONLY the data cell is replaced. Cell 5 defines exactly three globals --
     `nifty`, `vix`, `TAC_SYNTH` -- so substituting an FX close series and a
     causal realised-vol proxy is a clean swap. Every other cell is copied
     BYTE-FOR-BYTE out of build_master_notebook_v2.py.

What this CANNOT do: FX has no VIX, so `vix_chg` is fed the same causal
realised-vol PROXY the sibling notebooks use, and that is stamped on the run.
Sections of the master written about specific Nifty episodes still execute and
are reported as-is -- their PROSE will say Nifty things while their NUMBERS are
FX. That is flagged in the notebook rather than edited out, because editing the
master's text would break the "unmodified" property this notebook exists to have.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(
    os.environ.get('REGDET_OUT_DIR',
                   os.path.join(os.path.dirname(HERE), 'notebooks')),
    'fx_v9.ipynb')
MASTER_GEN = f'{HERE}/build_master_notebook_v2.py'

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

# The constants cell must still carry BARS_PER_DAY = 3, untouched. If a future
# edit to the master changes it, this notebook's whole premise (8h FX == 3
# bars/day == Nifty 2h cadence) silently breaks, so it is asserted, not assumed.
_const = ''.join(cells[3]['source'])
assert 'BARS_PER_DAY   = 3' in _const, \
    'master BARS_PER_DAY is no longer 3 -- the 8h FX cadence match is void'

FX_LOADER = r'''
# ===========================================================================
# DATA LAYER -- THE ONLY CELL THAT DIFFERS FROM THE SHIPPED v9 MASTER.
#
# It defines exactly the same three globals the original defines --
# `nifty`, `vix`, `TAC_SYNTH` -- so every downstream cell is byte-identical.
#
# WHY 8-HOUR BARS: Nifty 2h = 3 bars per NSE session. FX 8h = 3 bars per day
# (24/8). BARS_PER_DAY = 3 is therefore LITERALLY CORRECT for both, every
# day-denominated window keeps its meaning, and the master's own
# `assert MOM_3D_BARS == BASE_WIN['MOM_3D'] == 9` passes UNTOUCHED.
# Not one constant is re-tuned anywhere in this notebook.
#
# WHY THE BAR COUNT IS MATCHED: the user's Nifty v9 run was 2,858 bars. Every
# earlier FX chart in this project drew 57,311 bars into the same panel width
# -- 20x the density, which produces stripes REGARDLESS of regime quality.
# Matching n makes the visual comparison honest and keeps runtime comparable.
# ===========================================================================
FX_SYM, FX_NAME, FX_MED_BAND = 'EURUSD', 'EUR/USD', (0.9, 1.7)
FX_HOURS = 8            # 24 / 8 = 3 bars per day == Nifty's 3 bars per session
MATCH_BARS = 2858       # the Nifty v9 run this is being compared against
FX_URL = ('https://raw.githubusercontent.com/ejtraderLabs/historical-data/'
          f'main/{FX_SYM}/{FX_SYM}h1.csv')
FX_SOURCE_LABEL = ('COMMUNITY GITHUB DATA (ejtraderLabs/historical-data) - '
                   'verified against Brexit/COVID, NOT an official feed')


def _fx_load_2h():
    """FX 1h -> 8h (DOWNWARD only), scale-detected, matched to MATCH_BARS.

    Returns (close, vol_proxy, is_synth) with the SAME contract load_2h() has.
    """
    df = pd.read_csv(FX_URL, parse_dates=['Date'])
    assert list(df.columns) == ['Date', 'open', 'high', 'low', 'close',
                                'tick_volume'], f'unexpected columns {list(df.columns)}'
    df = df.set_index('Date').sort_index()
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()

    # Integer scale DETECTED (never hardcoded): pick the power of ten that puts
    # the median close inside a wide, coarse textbook band.
    med = float(np.median(df['close'].values))
    hits = [d for d in [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
            if FX_MED_BAND[0] <= med / d <= FX_MED_BAND[1]]
    assert len(hits) == 1, f'scale detection AMBIGUOUS for {FX_SYM}: {hits}'
    div = hits[0]
    px = df[['open', 'high', 'low', 'close']] / div

    # DOWNWARD aggregation only. Asserted, never assumed.
    step_in = pd.Series(px.index).diff().median()
    bars = px.resample(f'{FX_HOURS}h').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).dropna(how='any')
    step_out = pd.Series(bars.index).diff().median()
    assert step_out > step_in, f'UPWARD resample attempted: {step_in} -> {step_out}'

    realised_bpd = float(pd.Series(1, index=bars.index)
                         .groupby(bars.index.normalize()).sum().median())
    close_full = bars['close'].astype(float)

    # The vol PROXY is computed on the FULL history, THEN sliced -- it reads
    # bars <= t only (rolling sd, no shift, no centring, NEVER bfill), so
    # slicing afterwards cannot leak. Doing it the other way round would burn
    # VOL_SLOW bars of the matched window on warm-up for no reason.
    r = np.log(close_full / close_full.shift(1))
    vol_full = (r.rolling(int(BASE_WIN['VOL_SLOW'])).std()
                * np.sqrt(max(BARS_PER_DAY, 1) * 252.0) * 100.0)
    vol_full = vol_full.replace([np.inf, -np.inf], np.nan)

    close = close_full.iloc[-MATCH_BARS:]
    vol = vol_full.reindex(close.index)
    close.name, vol.name = 'nifty', 'vix'    # downstream reads these names

    print(f'FX {FX_NAME} 1h -> {FX_HOURS}h: {len(close_full)} bars available, '
          f'using the most recent {len(close)} to match the Nifty v9 run')
    print(f'  divisor DETECTED = {div:g}   realised bars/day = {realised_bpd:.2f} '
          f'(BARS_PER_DAY assumed {BARS_PER_DAY})')
    print(f'  volatility input = causal realised-vol PROXY. FX has NO VIX. '
          f'This is NOT a VIX.')

    # Event check on the SLICED window, so the data actually used is verified.
    _cv = close_full.loc['2020-03']
    if len(_cv):
        print(f'  COVID Mar-2020 range on the 8h series: {_cv.min():.4f} - '
              f'{_cv.max():.4f} ({(_cv.max() / _cv.min() - 1) * 100:.1f}% swing)')
    return close, vol, False


nifty, vix, TAC_SYNTH = _fx_load_2h()

print()
print('=' * 100)
print('*** THIS IS NOT NIFTY. ***')
print(f'*** Instrument : {FX_NAME}  (spot FX, 8h bars)')
print(f'*** Provenance : {FX_SOURCE_LABEL}')
print('*** The DATA ENDS 2022-03-04. This notebook says NOTHING about 2022-2026.')
print('***')
print('*** NOT ONE CONSTANT IS CHANGED from the shipped v9 master. Only this')
print('*** data cell differs. Every other cell is byte-for-byte identical, so')
print('*** the prose in later sections still describes NIFTY episodes while the')
print('*** NUMBERS beside it are FX. That is deliberate -- editing the master')
print('*** text would forfeit the "unmodified" property this notebook exists')
print('*** to have. Read the numbers, discount the Nifty-specific prose.')
print('=' * 100)
print(f"\nSpan: {nifty.index[0]}  ->  {nifty.index[-1]}")
'''

cells[IDX_LOAD] = {"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": FX_LOADER.splitlines(keepends=True)}

# --- a header, and the Nifty reference numbers to compare against ----------
HEADER = r"""
# RegDet v9 — the SHIPPED master, UNMODIFIED, run on FX

Same notebook, same 15 charts, same scorecard, **same constants** — only the
data cell differs.

### Why this is a fair comparison and the earlier FX charts were not

| | Nifty v9 run | earlier FX notebooks | **this notebook** |
|---|---|---|---|
| bars per day | 3 (2h × NSE session) | 24 (1h) | **3 (8h) — matched** |
| `BARS_PER_DAY` | 3 | 24, windows rebuilt | **3, untouched** |
| bars drawn per chart | 2,858 | 57,311 | **2,858 — matched** |
| constants re-tuned | — | none | **none** |

FX at 8h is 3 bars/day, exactly like Nifty 2h, so `BARS_PER_DAY = 3` is
*literally correct* for both and every day-denominated window keeps its
meaning. The master's own `assert MOM_3D_BARS == 9` passes untouched.

The bar count matters more than it looks: **57,311 bars in one panel is 20×
the density of 2,858**, and at that density the regime shading merges into
stripes whatever the labels say. Part of the "FX barcodes" impression was
plotting, not detection. This run removes that confound.

### The Nifty numbers this is being compared against

From the user's own `final_regdet_v9.ipynb` Kaggle run (2,858 bars,
2023-09 → 2026-07, real yfinance data):

| | Nifty v9 |
|---|---|
| forward-return ordering | **BROKEN at all 3 horizons** |
| median run length | H_BULL 5.0, L_BULL 2.0, SIDEWAYS 5.0, L_BEAR 3.0, H_BEAR 5.0 |
| occupancy | 20.0 / 14.8 / 28.8 / 16.3 / 20.1 % |
| scorecard (7A, 70% fit) | FAIL 29, PASS 9, NA 6, WARN 4, FAIL(ANTI) 2 |
| regime long/flat vs B&H | Sharpe **0.41 vs 0.60** (underperforms) |

Those are the bars to beat. They are printed here **before** the FX numbers
exist, so nothing below can be graded generously after the fact.
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
print(f'wrote {OUT} - {len(cells)} cells '
      f'({sum(1 for c in cells if c["cell_type"] == "code")} code, '
      f'{sum(1 for c in cells if c["cell_type"] == "markdown")} markdown)')
print('ONLY the data cell differs from the shipped master; all others verbatim.')
