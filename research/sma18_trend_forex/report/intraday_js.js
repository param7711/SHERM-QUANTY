// ===========================================================================
// Intraday liquidity vs result (15m / 30m / 1h), vs the daily reference
// ===========================================================================
(function () {
  "use strict";
  var I = JSON.parse(document.getElementById('intraday-liquidity-data').textContent);
  var root = getComputedStyle(document.documentElement);
  function tok(n) { return root.getPropertyValue(n).trim(); }
  var ACCENT = tok('--accent'), CONTEXT = tok('--context'), GOOD = tok('--good'),
      CRIT = tok('--critical'), MUTED = tok('--ink-muted');
  var CLS_COLOR = { CRYPTO: ACCENT, COMMODITY: '#128a5e' };
  function sgn(v) { return v > 0 ? 'pos' : (v < 0 ? 'neg' : ''); }

  var order = I.timeframes.concat(['1d']);
  var byTf = {};
  I.correlations.forEach(function (c) { byTf[c.timeframe] = c; });
  byTf['1d'] = I.daily_reference;

  // ---- grouped bar chart, via three lineChart-style series on a categorical x
  var series = {
    pooled: order.map(function (tf) { return byTf[tf].pooled ? byTf[tf].pooled.spearman : null; }),
    COMMODITY: order.map(function (tf) { return (byTf[tf].by_class.COMMODITY || {}).spearman; }),
    CRYPTO: order.map(function (tf) { return (byTf[tf].by_class.CRYPTO || {}).spearman; }),
  };
  RC.lineChart(document.getElementById('intraCorrChart'), {
    n: order.length, height: 240,
    lines: [
      { label: 'pooled (both classes)', color: MUTED, width: 2, values: series.pooled },
      { label: 'commodity', color: CLS_COLOR.COMMODITY, width: 2.5, values: series.COMMODITY },
      { label: 'crypto', color: CLS_COLOR.CRYPTO, width: 2.5, values: series.CRYPTO },
    ],
    xLabels: order, xTickCount: order.length - 1,
    yFormat: function (v) { return v.toFixed(2); },
    tooltipTitle: function (i) { return order[i] + (order[i] === '1d' ? ' (daily, established edge)' : ' (noisy, no p-value)'); },
    zeroLine: true
  });

  function fmtC(c) { return !c ? 'n/a' : (c.spearman >= 0 ? '+' : '') + c.spearman.toFixed(2) + ' (n=' + c.n + ')'; }
  document.getElementById('intra-verdict').innerHTML =
    '<b>The clean daily-timeframe story does not repeat intraday.</b> At 15m and 30m the ' +
    'commodity correlation stays strongly positive (' + fmtC(byTf['15m'].by_class.COMMODITY) +
    ' and ' + fmtC(byTf['30m'].by_class.COMMODITY) + ', rising to ' + fmtC(byTf['1h'].by_class.COMMODITY) +
    ' at 1h) &mdash; more liquid commodities keep winning by a wide margin at every intraday ' +
    'timeframe, which is the most consistent single finding in this section. ' +
    '<b>Crypto does not behave the same way it did on daily bars.</b> The strong daily relationship ' +
    '(' + fmtC(byTf['1d'].by_class.CRYPTO) + ') is not visible intraday: 15m is flat-to-negative (' +
    fmtC(byTf['15m'].by_class.CRYPTO) + '), 30m is weak (' + fmtC(byTf['30m'].by_class.CRYPTO) +
    '), 1h is negative (' + fmtC(byTf['1h'].by_class.CRYPTO) + '). And pooling both classes together ' +
    'flips sign entirely between timeframes (' + fmtC(byTf['15m'].pooled) + ' at 15m to ' +
    fmtC(byTf['1h'].pooled) + ' at 1h) purely from which class happens to be having a good 60-day window. ' +
    '<b>Read that as noise, not as a reversal of the daily finding</b> &mdash; there is no null model behind ' +
    'any of these three numbers, and a correlation that flips sign between adjacent timeframes on the same ' +
    'underlying instruments is the signature of a measurement with too little data behind it, not of a real ' +
    'effect that appears and disappears. The one part that reads as a real pattern rather than noise is the ' +
    'commodity result, because it agrees in direction and magnitude at all three intraday timeframes AND with ' +
    'the daily result &mdash; four independent looks pointing the same way.';

  // ---- tiers ----------------------------------------------------------------
  var tfLabel = { '15m': '15 min', '30m': '30 min', '1h': '1 hour' };
  document.getElementById('intra-tiers').innerHTML = I.tiers.map(function (t) {
    return '<div class="compare-card">' +
      '<span class="metric-label">' + (tfLabel[t.timeframe] || t.timeframe) + ' &middot; ' +
        t.asset_class + ' &middot; ' + t.tier + '</span>' +
      '<div style="font-family:var(--mono);font-size:1.3rem;font-weight:600" class="' +
        sgn(t.mean_sharpe) + '">' + t.mean_sharpe.toFixed(2) + '</div>' +
      '<div style="font-size:0.74rem;color:var(--ink-muted);font-family:var(--mono)">mean Sharpe &middot; ' +
        t.n + ' instruments &middot; avg ' + t.mean_trades + ' trades</div>' +
      '<div style="font-size:0.72rem;color:var(--ink-2);line-height:1.5">' + t.assets.join(', ') + '</div>' +
    '</div>';
  }).join('');

  // ---- full table -------------------------------------------------------
  var rows = I.per_row.slice();
  function renderTable() {
    document.querySelector('#intra-table tbody').innerHTML = rows.map(function (r) {
      return '<tr>' +
        '<td>' + r.asset + '</td>' +
        '<td style="text-align:left"><span class="chip cls-' + r.asset_class + '">' +
          r.asset_class.toLowerCase() + '</span></td>' +
        '<td>' + r.timeframe + '</td>' +
        '<td>$' + (r.median_dollar_vol >= 1e9 ? (r.median_dollar_vol / 1e9).toFixed(2) + 'B'
                                               : (r.median_dollar_vol / 1e6).toFixed(1) + 'M') + '</td>' +
        '<td class="' + sgn(r.sharpe) + '">' + r.sharpe.toFixed(2) + '</td>' +
        '<td>' + r.n_trades + '</td>' +
        '<td>' + (r.win_rate * 100).toFixed(0) + '%</td></tr>';
    }).join('');
  }
  renderTable();
  var dir = {};
  document.querySelectorAll('#intra-table th').forEach(function (th) {
    th.addEventListener('click', function () {
      var k = th.getAttribute('data-key');
      dir[k] = !dir[k];
      rows.sort(function (a, b) {
        var x = a[k], y = b[k];
        if (typeof x === 'string') return dir[k] ? x.localeCompare(y) : y.localeCompare(x);
        return dir[k] ? x - y : y - x;
      });
      renderTable();
    });
  });
})();
