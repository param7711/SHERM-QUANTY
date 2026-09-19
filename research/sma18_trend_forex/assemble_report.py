"""Assembles the validation report artifact from the old template's reusable
parts (CSS, SVG chart library, grid/long-short renderers) plus the new
luck-testing body and script."""
import json, os, re

SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

old = open(f"{SP}/report_template.html").read().split("\n")
def lines(a, b): return "\n".join(old[a-1:b])

css        = lines(1, 184)      # <title>, <style>, fonts link
rc_lib     = lines(504, 693)    # chart library (inside <script>)
grid_js    = lines(851, 1169)   # Part 2 + Part 3 + head-to-head renderers
body_rest  = lines(292, 495)    # grid / long+short / head-to-head body
val_body   = open(f"{SP}/val_body.html").read()
val_js     = open(f"{SP}/val_js.js").read()
hist_js    = open(f"{SP}/rc_histogram.js").read()
extra_body = open(f"{SP}/extra_body.html").read()
extra_js   = open(f"{SP}/extra_js.js").read()

# --- 1. extend the chart library -------------------------------------------
rc_lib = rc_lib.replace(
    "  return { lineChart: lineChart, scatterChart: scatterChart, THEME: THEME };",
    hist_js + "  return { lineChart: lineChart, scatterChart: scatterChart, histogram: histogram, THEME: THEME };")

# zero line + vertical marks support in lineChart
anchor = "    container.innerHTML =\n      '<svg viewBox=\"0 0 ' + W + ' ' + H + '\""
extra = """    var extraSvg = '';
    if (cfg.zeroLine && yMin < 0 && yMax > 0) {
      var zy = yScale(0);
      extraSvg += '<line x1="' + mL + '" y1="' + zy + '" x2="' + (W - mR) + '" y2="' + zy +
        '" stroke="' + THEME.muted + '" stroke-width="1" stroke-dasharray="3 3"/>';
    }
    (cfg.marks || []).forEach(function (m) {
      if (m.index == null || m.index < 0) return;
      var mx = xScale(m.index);
      extraSvg += '<line x1="' + mx + '" y1="' + mT + '" x2="' + mx + '" y2="' + (mT + plotH) +
        '" stroke="' + m.color + '" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.75"/>' +
        '<text x="' + (mx + 4) + '" y="' + (mT + 11) + '" font-size="9" fill="' + m.color +
        '" font-family="' + "'IBM Plex Mono',monospace" + '" font-weight="600">' + m.label + '</text>';
    });
"""
assert anchor in rc_lib
rc_lib = rc_lib.replace(anchor, extra + anchor, 1)
rc_lib = rc_lib.replace("gridSvg + xTicksSvg + bandsSvg + linesSvg + dotsSvg +",
                        "gridSvg + xTicksSvg + bandsSvg + extraSvg + linesSvg + dotsSvg +")

# --- 2. update the surviving body copy for the new universe -----------------
reps = [
    ("Part 2 · does timeframe or asset volatility change the answer?",
     "The wide sweep · four candle sizes"),
    ("Same rule, four candle sizes, four asset classes — where does it actually work?",
     "Same rule, four candle sizes, 30 instruments — where does it actually work?"),
    ("Identical entry/exit as Part 1, run on <b>15m, 30m, 4h and 1d</b> bars across\n        <b>20 instruments</b> spanning FX, equity indices, commodities and crypto — so\n        \"volatile vs. non-volatile\" and \"which timeframe\" get one consistent, comparable\n        answer instead of separate one-off tests. Volatility tiers are measured per\n        instrument from its own data, not assumed.",
     "The same entry/exit, run on <b>15m, 30m, 4h and 1d</b> bars across all\n        <b>30 instruments</b>. Everything above tests the daily timeframe, where the history is\n        long enough to support a null model; this section is the breadth check. Read it as a survey,\n        not as evidence — the intraday rows rest on 60 days of data and no p-value backs them."),
    ("the identical 20-instrument &times; 4-timeframe\n        grid as Part 2", "the identical 30-instrument &times; 4-timeframe\n        grid"),
    ("Same 80 instrument&times;timeframe combinations, long-only (Part 2) vs. long+short\n      (Part 3), matched pair by pair.",
     "Every instrument&times;timeframe combination, long-only vs. long+short, matched pair by pair."),
    ('id="part2"', 'id="grid"'),
    ("Part 3 · adding the short side", "Adding the short side"),
    ("Same 18-period SMA, same touch-exit.", "Same 18-period SMA, same touch-exit."),
    ("<h2>Reading Part 2 (and the Monte Carlo) honestly</h2>", "<h2>Reading the grid honestly</h2>"),
    ("<h2>Reading Part 3 &amp; the head-to-head honestly</h2>", "<h2>Reading the long+short result honestly</h2>"),
    ("Same bootstrap method as Part 2's Monte Carlo", "Same bootstrap method as the Monte Carlo above"),
    ("Two picks are the strongest long+short results found; two are the same instruments Part 2\n      highlighted", "Two picks are the strongest long+short results found; two are the same instruments\n      highlighted above"),
    ("Sharpe by instrument &times; timeframe — long + short", "Sharpe by instrument &times; timeframe — long + short"),
]
for a, b in reps:
    if a not in body_rest:
        raise SystemExit(f"replacement not found:\n{a[:90]}")
    body_rest = body_rest.replace(a, b)

# Replace the two stale caveat lists wholesale.
def swap_ul(html, head_marker, new_items):
    i = html.index(head_marker)
    s = html.index("<ul>", i); e = html.index("</ul>", s) + 5
    return html[:s] + "<ul>\n" + new_items + "    </ul>" + html[e:]

GRID_CAVEATS = """      <li><b>15m and 30m are ~60 calendar days of data — a yfinance hard cap, not a choice.</b>
        4h (resampled from 1h) reaches ~2 years; only 1d carries full history. Every statistical test on this
        page runs on daily bars for exactly that reason: 60 days is not enough to fit a null model to, so the
        intraday rows below have <i>no p-value behind them at all</i>. Their headline Sharpes are also
        annualized from a 60-day window, which magnifies small differences enormously. Trust the direction,
        never the magnitude.</li>
      <li><b>The Monte Carlo here resamples trade order, not the market.</b> It answers "how much of this
        depended on the sequence these wins and losses happened to arrive in" — a much weaker question than the
        Markov test above, which generates entirely new markets. It also treats trades as independent draws,
        which understates uncertainty, since consecutive trades on one instrument share a regime. Treat these
        bands as a lower bound on uncertainty. Combinations with fewer than 20 trades are excluded.</li>
      <li><b>Crypto beats commodities at every single timeframe</b>, which lines up with the Markov result
        (14/14 crypto significant vs 11/16 commodities) and with the cost result. Three separate tests pointing
        the same way is the most reassuring pattern in this study.</li>
      <li><b>The 4h row is the odd one out</b> — commodity 4h is the only negative class&times;timeframe cell in
        the grid. I have no mechanism to explain that, and with ~2 years of 4h history I would not bet on it
        being real. Flagging it rather than explaining it away.</li>
      <li><b>Results for one asset across four timeframes are not four independent trials</b> — they are four
        views of the same price series. The grid is a sweep, not 120 experiments.</li>
      <li><b>No costs are modeled anywhere in this grid.</b> The cost section above shows what that is worth on
        daily bars; at 15m and 30m, where trades are far more frequent, a realistic spread would very likely
        erase the edge completely.</li>
"""
H2H_CAVEATS = """      <li><b>Adding the short side made things worse in roughly five of every six combinations tested</b>
        (<span id="h2h-inline-stat"></span>). Over this sample commodities and crypto spent far more time
        trending up than down, so mirroring the long entry into a short mostly means selling into an uptrend and
        buying back the countertrend bounce — the losing side of momentum.</li>
      <li><b>This is the clearest "regime, not law" finding on the page.</b> The long-only result survives a
        Markov null; the short side's failure does not get the same treatment, and it plainly reflects a sample
        in which both asset classes rose. A prolonged commodity bear market or a real crypto winter could flip
        it. The head-to-head table is there so you can see exactly which cells would have to reverse.</li>
      <li><b>Crypto loses the most from adding shorts</b> — the same asset class that trended up hardest and
        that the Markov test likes most. That is consistent rather than contradictory: a system built to ride
        sustained trends is hurt most by fading the strongest ones.</li>
      <li><b>A long+short system trades roughly twice as often</b>, so every cost caveat above applies twice as
        hard — and costs are still not modeled in this grid.</li>
      <li><b>Over the era splits, long+short is worse on both measures</b>: 71% of asset-era cells positive at a
        mean Sharpe of 0.25, against 80% and 0.39 for long-only. The short side's weakness is not confined to
        one period.</li>
"""
# The vol/Sharpe inline callout lived in the old caveat list that section 2
# replaced, so the renderer's id-truthiness guard is no longer sufficient —
# make it check for the element itself.
_guard = "  if (ids.corrPooledInline) {"
assert _guard in grid_js
grid_js = grid_js.replace(
    _guard, "  if (ids.corrPooledInline && document.getElementById(ids.corrPooledInline)) {", 1)

body_rest = swap_ul(body_rest, "<h2>Reading the grid honestly</h2>", GRID_CAVEATS)
body_rest = swap_ul(body_rest, "<h2>Reading the long+short result honestly</h2>", H2H_CAVEATS)

footer = """  <footer>
    research/sma18_trend_forex/ · SHERM-QUANTY · reproduce the luck tests with
    <code>python test_engine.py</code>, <code>python markov_validation.py 500 long</code>,
    <code>python era_splits.py</code>, <code>python robustness_extras.py</code>
  </footer>
</div>"""

# --- 3. load data -----------------------------------------------------------
def rd(n): return open(os.path.join(OUT, n)).read()
payloads = {
    "validation-data": rd("validation_report_data.json"),
    "projection-data": rd("mc_projection.json"),
    "liquidity-data": rd("liquidity_test.json"),
    "grid-report-data": rd("grid_report_data.json"),
    "grid-report-data-both": rd("grid_report_data_both.json"),
    "headtohead-data": rd("headtohead_data.json"),
}
for k, v in payloads.items():
    json.loads(v)  # fail loudly rather than shipping a page that won't parse
    if re.search(r"\b(NaN|Infinity|-Infinity)\b", v):
        raise SystemExit(f"{k} contains non-JSON NaN/Infinity tokens")

_vp = '<section class="panel caveats" id="verdict-panel">'
assert _vp in val_body, "verdict panel anchor missing"
val_body = val_body.replace(_vp, extra_body + "\n" + _vp, 1)

html = (
    css + "\n\n<div class=\"wrap\">\n\n" + val_body + "\n\n" + body_rest + "\n\n" + footer + "\n\n"
    + "\n".join(f'<script id="{k}" type="application/json">{v}</script>' for k, v in payloads.items())
    + "\n\n<script>\n" + rc_lib + "\n\n" + val_js + "\n" + extra_js + "\n" + grid_js + "\n</script>\n"
)

dest = os.path.join(SP, "validation_report.html")
open(dest, "w").write(html)
print(f"wrote {dest} ({len(html):,} bytes)")

# title is the artifact's name in the gallery — keep it a name, not a summary
import pathlib
p = pathlib.Path(dest)
p.write_text(p.read_text().replace("<title>SMA18 Trend Hypothesis</title>",
                                   "<title>Real Edge or Luck</title>", 1))
print("title set")
