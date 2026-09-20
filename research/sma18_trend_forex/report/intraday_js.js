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
  var tfLabel = { '15m': '15 min', '30m': '30 min', '1h': '1 hour', '1d': 'daily (reference)' };

  // ---- one scatter per timeframe, same visual as the daily liquidity scatter
  document.getElementById('intra-scatter-grid').innerHTML = order.map(function (tf, i) {
    var c = byTf[tf];
    return '<div class="mc-card">' +
      '<div class="mc-card-head"><span class="name">' + tfLabel[tf] + '</span>' +
        (tf === '1d' ? '<span class="badge-label">reference</span>' : '') + '</div>' +
      '<div class="chart-wrap" id="intra-scatter-' + i + '"></div>' +
      '<div class="mc-stats">' +
        '<span>commodity r<sub>s</sub> <b class="' + sgn((c.by_class.COMMODITY || {}).spearman) + '">' +
          ((c.by_class.COMMODITY || {}).spearman || 0).toFixed(2) + '</b></span>' +
        '<span>crypto r<sub>s</sub> <b class="' + sgn((c.by_class.CRYPTO || {}).spearman) + '">' +
          ((c.by_class.CRYPTO || {}).spearman || 0).toFixed(2) + '</b></span>' +
      '</div></div>';
  }).join('');

  document.getElementById('intra-corr-legend').innerHTML =
    '<span><span class="legend-dot" style="background:' + CLS_COLOR.COMMODITY + '"></span>commodity</span>' +
    '<span><span class="legend-dot" style="background:' + CLS_COLOR.CRYPTO + '"></span>crypto</span>' +
    '<span>r<sub>s</sub> = Spearman rank correlation, liquidity vs. Sharpe, within that class</span>';

  var dailyPts = null;
  try { dailyPts = JSON.parse(document.getElementById('liquidity-data').textContent).per_asset; } catch (e) {}

  order.forEach(function (tf, i) {
    var pts;
    if (tf === '1d') {
      pts = (dailyPts || []).filter(function (r) { return !r.volume_unreliable; })
        .map(function (r) {
          return { x: r.log10_dollar_vol, y: r.real_sharpe, asset_class: r.asset_class,
                   asset: r.asset, n_trades: r.real_n_trades };
        });
    } else {
      pts = I.per_row.filter(function (r) { return r.timeframe === tf; })
        .map(function (r) {
          return { x: r.log10_dollar_vol, y: r.sharpe, asset_class: r.asset_class,
                   asset: r.asset, n_trades: r.n_trades };
        });
    }
    RC.scatterChart(document.getElementById('intra-scatter-' + i), {
      xLabel: 'median notional USD traded per day (log scale)',
      yLabel: 'Sharpe',
      xFormat: function (v) {
        var d = Math.pow(10, v);
        return d >= 1e9 ? '$' + (d / 1e9).toFixed(0) + 'B' : '$' + (d / 1e6).toFixed(0) + 'M';
      },
      yFormat: function (v) { return v.toFixed(1); },
      points: pts.map(function (r) {
        return {
          x: r.x, y: r.y, color: CLS_COLOR[r.asset_class] || CONTEXT,
          tooltip: '<div style="font-weight:600">' + r.asset + '</div>' +
            '<div>$' + (Math.pow(10, r.x) / 1e9).toFixed(2) + 'B/day</div>' +
            '<div>Sharpe ' + r.y.toFixed(2) + '</div>' +
            '<div>' + r.n_trades + ' trades</div>'
        };
      })
    });
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
