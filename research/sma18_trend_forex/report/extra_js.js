// ===========================================================================
// Forward Monte Carlo projections + the liquidity test
// ===========================================================================
(function () {
  "use strict";
  var P = JSON.parse(document.getElementById('projection-data').textContent);
  var L = JSON.parse(document.getElementById('liquidity-data').textContent);
  var root = getComputedStyle(document.documentElement);
  function tok(n) { return root.getPropertyValue(n).trim(); }
  var ACCENT = tok('--accent'), ACCENT_SOFT = tok('--accent-soft'), CONTEXT = tok('--context'),
      GOOD = tok('--good'), CRIT = tok('--critical'), MUTED = tok('--ink-muted');
  var CLS_COLOR = { CRYPTO: ACCENT, COMMODITY: '#128a5e' };
  function sgn(v) { return v > 0 ? 'pos' : (v < 0 ? 'neg' : ''); }

  // ---- long vs long+short by class -----------------------------------------
  var rows = P.by_class_direction.slice().sort(function (a, b) {
    return a.asset_class.localeCompare(b.asset_class) || a.horizon - b.horizon;
  });
  document.querySelector('#proj-class-table tbody').innerHTML = rows.map(function (r) {
    function better(a, b, lowerIsBetter) {
      var win = lowerIsBetter ? a < b : a > b;
      return win ? 'font-weight:600;color:' + GOOD : '';
    }
    return '<tr>' +
      '<td><span class="chip cls-' + r.asset_class + '">' + r.asset_class.toLowerCase() + '</span></td>' +
      '<td>' + r.horizon + 'y</td>' +
      '<td style="' + better(r.long_median_terminal, r.both_median_terminal) + '">&times;' + r.long_median_terminal.toFixed(2) + '</td>' +
      '<td style="' + better(r.both_median_terminal, r.long_median_terminal) + '">&times;' + r.both_median_terminal.toFixed(2) + '</td>' +
      '<td class="' + sgn(r.long_cagr_p50) + '">' + r.long_cagr_p50.toFixed(1) + '%</td>' +
      '<td class="' + sgn(r.both_cagr_p50) + '">' + r.both_cagr_p50.toFixed(1) + '%</td>' +
      '<td>' + (r.long_p_profit * 100).toFixed(0) + '%</td>' +
      '<td>' + (r.both_p_profit * 100).toFixed(0) + '%</td>' +
      '<td style="' + better(r.long_dd_p95, r.both_dd_p95, true) + '">' + r.long_dd_p95.toFixed(0) + '%</td>' +
      '<td style="' + better(r.both_dd_p95, r.long_dd_p95, true) + '">' + r.both_dd_p95.toFixed(0) + '%</td>' +
      '<td style="text-align:left">' + r.n_long_better + '/' + r.n + '</td></tr>';
  }).join('');

  var c5 = rows.filter(function (r) { return r.asset_class === 'COMMODITY' && r.horizon === 5; })[0];
  var k5 = rows.filter(function (r) { return r.asset_class === 'CRYPTO' && r.horizon === 5; })[0];
  document.getElementById('proj-verdict').innerHTML =
    '<b>On commodities the short side is simply destructive:</b> long-only projects a median of ' +
    '&times;' + c5.long_median_terminal.toFixed(2) + ' over five years against &times;' +
    c5.both_median_terminal.toFixed(2) + ' with shorts added &mdash; a losing proposition &mdash; while the ' +
    'worst-case drawdown <i>deepens</i> from ' + c5.long_dd_p95.toFixed(0) + '% to ' +
    c5.both_dd_p95.toFixed(0) + '%. Long-only wins on ' + c5.n_long_better + ' of ' + c5.n +
    ' instruments. Worse return, worse risk, no trade-off to weigh. ' +
    '<b>On crypto the picture is subtler and easy to misread.</b> Adding shorts raises the median ' +
    'slightly (&times;' + k5.both_median_terminal.toFixed(2) + ' vs &times;' + k5.long_median_terminal.toFixed(2) +
    ') and long-only wins on only ' + k5.n_long_better + ' of ' + k5.n + ' instruments &mdash; which looks ' +
    'like a point for shorting until you read the risk column: the 95th-percentile drawdown goes from ' +
    k5.long_dd_p95.toFixed(0) + '% to ' + k5.both_dd_p95.toFixed(0) + '%, and the probability of profit is ' +
    'unchanged at ' + (k5.long_p_profit * 100).toFixed(0) + '%. <b>You are paying ' +
    (k5.both_dd_p95 - k5.long_dd_p95).toFixed(0) + ' extra points of drawdown for essentially the same money.</b> ' +
    'That is the same conclusion the Sharpe-based head-to-head reached, arrived at independently.';

  // ---- five-year fan charts ------------------------------------------------
  var PICKS = ['BTCUSD', 'ETHUSD', 'GOLD', 'CRUDE', 'SUGAR', 'DOTUSD'];
  var byName = {};
  P.assets.forEach(function (a) { byName[a.asset] = a; });
  var picked = PICKS.map(function (n) { return byName[n]; }).filter(function (a) {
    return a && a.directions.long && a.directions.long.gross.paths;
  });
  document.getElementById('proj-grid').innerHTML = picked.map(function (a, i) {
    var s = a.directions.long.gross.block['5'];
    return '<div class="mc-card">' +
      '<div class="mc-card-head"><span class="name">' + a.asset +
        ' <span class="chip cls-' + a.asset_class + '">' + a.asset_class.toLowerCase() + '</span></span>' +
        '<span class="badge-label">' + a.years_history + 'y history</span></div>' +
      '<div class="chart-wrap" id="proj-' + i + '"></div>' +
      '<div class="mc-stats">' +
        '<span>median <b>&times;' + s.median_terminal.toFixed(2) + '</b></span>' +
        '<span>5&ndash;95%: <b>&times;' + s.terminal_multiple.p5.toFixed(2) + ' to &times;' +
          s.terminal_multiple.p95.toFixed(2) + '</b></span>' +
        '<span>P(profit) <b>' + (s.p_profit * 100).toFixed(0) + '%</b></span>' +
        '<span>worst DD <b>' + s.max_drawdown_pct.p95.toFixed(0) + '%</b></span>' +
      '</div></div>';
  }).join('');
  picked.forEach(function (a, i) {
    var p = a.directions.long.gross.paths, b = p.bands;
    var n = p.trade_index.length;
    RC.lineChart(document.getElementById('proj-' + i), {
      n: n, height: 170,
      bands: [{ lower: b.p5, upper: b.p95, color: ACCENT_SOFT },
              { lower: b.p25, upper: b.p75, color: ACCENT_SOFT }],
      lines: [{ label: 'median', color: ACCENT, width: 2, values: b.p50 }],
      xLabels: p.trade_index.map(function (t) { return t + ' tr'; }), xTickCount: 4,
      yFormat: function (v) { return '×' + v.toFixed(1); },
      tooltipTitle: function (j) { return 'after ' + p.trade_index[j] + ' trades'; }
    });
  });

  // ---- liquidity scatter ---------------------------------------------------
  var pts = L.per_asset.filter(function (r) { return !r.volume_unreliable; });
  RC.scatterChart(document.getElementById('liqScatter'), {
    xLabel: 'median notional USD traded per day (log scale)',
    yLabel: 'realized Sharpe',
    xFormat: function (v) {
      var d = Math.pow(10, v);
      return d >= 1e9 ? '$' + (d / 1e9).toFixed(0) + 'B' : '$' + (d / 1e6).toFixed(0) + 'M';
    },
    yFormat: function (v) { return v.toFixed(1); },
    points: pts.map(function (r) {
      return {
        x: r.log10_dollar_vol, y: r.real_sharpe,
        color: CLS_COLOR[r.asset_class] || CONTEXT,
        tooltip: '<div style="font-weight:600">' + r.asset + '</div>' +
          '<div>$' + (r.median_dollar_vol / 1e9).toFixed(2) + 'B/day</div>' +
          '<div>Sharpe ' + r.real_sharpe.toFixed(2) + '</div>' +
          '<div>Markov p = ' + r.markov_p.toFixed(3) + '</div>'
      };
    })
  });
  var cc = L.correlations;
  document.getElementById('liq-corr').innerHTML =
    '<span><span class="legend-dot" style="background:' + ACCENT + '"></span>crypto &mdash; within-class rank correlation <b class="num">' +
      cc.by_class.CRYPTO.log10_dollar_vol.vs_sharpe.spearman.toFixed(2) + '</b></span>' +
    '<span><span class="legend-dot" style="background:#128a5e"></span>commodity &mdash; within-class rank correlation <b class="num">' +
      cc.by_class.COMMODITY.log10_dollar_vol.vs_sharpe.spearman.toFixed(2) + '</b></span>' +
    '<span>pooled across both classes: <b class="num">' +
      cc.pooled.log10_dollar_vol.vs_sharpe.spearman.toFixed(2) + '</b> &mdash; essentially zero</span>';

  // ---- liquidity tiers -----------------------------------------------------
  document.getElementById('liq-tiers').innerHTML = L.tiers.map(function (t) {
    return '<div class="compare-card">' +
      '<span class="metric-label">' + t.asset_class + ' &middot; ' + t.tier + '</span>' +
      '<div style="font-family:var(--mono);font-size:1.3rem;font-weight:600" class="' +
        sgn(t.mean_sharpe) + '">' + t.mean_sharpe.toFixed(2) + '</div>' +
      '<div style="font-size:0.74rem;color:var(--ink-muted);font-family:var(--mono)">mean Sharpe &middot; ' +
        t.n_pass + '/' + t.n + ' beat the Markov null</div>' +
      '<div style="font-size:0.72rem;color:var(--ink-2);line-height:1.5">' + t.assets.join(', ') + '</div>' +
    '</div>';
  }).join('');

  var cs = cc.by_class.CRYPTO.log10_dollar_vol.vs_sharpe.spearman;
  var ms = cc.by_class.COMMODITY.log10_dollar_vol.vs_sharpe.spearman;
  document.getElementById('liq-verdict').innerHTML =
    '<b>The answer is "partly, and only within a class."</b> Rank liquidity against Sharpe inside ' +
    'crypto and the relationship is strong (<span class="num">' + cs.toFixed(2) + '</span>); inside ' +
    'commodities it is positive but weaker (<span class="num">' + ms.toFixed(2) + '</span>, on just ' +
    cc.by_class.COMMODITY.log10_dollar_vol.vs_sharpe.n + ' instruments after the bad-volume exclusions). ' +
    'The more liquid half of each class beats the thinner half on mean Sharpe in both cases. ' +
    '<b>But pool the two classes and the relationship vanishes entirely</b> (<span class="num">' +
    cc.pooled.log10_dollar_vol.vs_sharpe.spearman.toFixed(2) + '</span>), because the effect runs backwards ' +
    'across classes: ATOMUSD trades <span class="num">$0.09B</span> a day against gold\'s ' +
    '<span class="num">$56.7B</span> &mdash; 600&times; thinner &mdash; and posts a comparable Sharpe, while ' +
    'the entire crypto complex is less liquid than crude oil and outperforms it threefold. ' +
    'So liquidity sorts instruments <i>inside</i> a class but does not explain the split <i>between</i> them, ' +
    'and it cannot be the primary driver. One strong caveat: inside crypto, dollar volume is tangled up ' +
    'with market cap, age and how major the coin is &mdash; BTC and ETH are not merely more liquid, they are ' +
    'also the names with the longest and strongest trends in the sample. With 14 instruments those cannot ' +
    'be separated, so read that ' + cs.toFixed(2) + ' as "bigger coins did better," which is a weaker and ' +
    'less actionable claim than "liquidity causes the edge."';
})();
