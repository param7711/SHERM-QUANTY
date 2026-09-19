// ===========================================================================
// Validation — the luck tests
// ===========================================================================
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById('validation-data').textContent);
  var root = getComputedStyle(document.documentElement);
  function tok(n) { return root.getPropertyValue(n).trim(); }
  var ACCENT = tok('--accent'), CONTEXT = tok('--context'), GOOD = tok('--good'),
      CRIT = tok('--critical'), MUTED = tok('--ink-muted'), GRID = tok('--grid'),
      SURF2 = tok('--surface-2');
  var CLS_COLOR = { CRYPTO: ACCENT, COMMODITY: '#128a5e' };

  function tile(label, value, sub, cls) {
    return '<div class="tile"><span class="label">' + label + '</span>' +
      '<span class="value' + (cls ? ' ' + cls : '') + '">' + value + '</span>' +
      '<span class="sub">' + sub + '</span></div>';
  }
  function sgn(v) { return v > 0 ? 'pos' : (v < 0 ? 'neg' : ''); }

  // ---- headline tiles -----------------------------------------------------
  var ms = D.markov.summary, es = D.eras.long.summary, xs = D.extras.summary;
  document.getElementById('val-tiles').innerHTML = [
    tile('Beat the Markov null', ms.n_markov_sig_05 + '/' + ms.n_assets,
         ms.n_markov_sig_bh + ' still significant after Benjamini-Hochberg FDR correction'),
    tile('Beat both nulls', ms.n_both_sig_05 + '/' + ms.n_assets,
         'Markov <i>and</i> i.i.d. shuffle, both at p&lt;0.05'),
    tile('Asset-era cells positive', (es.pct_positive * 100).toFixed(0) + '%',
         es.n_positive + ' of ' + es.n_asset_eras + ' instrument&times;era combinations, each run in isolation'),
    tile('SMA lengths profitable', (xs.mean_pct_periods_positive * 100).toFixed(0) + '%',
         'of lengths 5&ndash;60; 18 is the best length on ' + xs.n_assets_where_18_is_best + ' of 30 assets'),
    tile('Survive 20bps cost', xs.n_assets_positive_at_20bps + '/' + ms.n_assets,
         'round-trip charged on every trade'),
    tile('Engine tests', D.engine_tests.passed ? 'all pass' : 'FAILING',
         D.engine_tests.lines.length + ' checks, re-run live when this page was built',
         D.engine_tests.passed ? 'pos' : 'neg')
  ].join('');

  // ---- engine test output -------------------------------------------------
  document.getElementById('test-output').innerHTML =
    '<div style="background:' + SURF2 + ';border-radius:8px;padding:12px 14px;font-family:var(--mono);' +
    'font-size:0.74rem;line-height:1.85;overflow-x:auto">' +
    D.engine_tests.lines.map(function (l) {
      var ok = l.indexOf('PASS') === 0;
      var color = ok ? GOOD : (l.indexOf('SKIP') === 0 ? MUTED : CRIT);
      return '<div><span style="color:' + color + ';font-weight:600">' + l.slice(0, 4) + '</span>' +
             '<span style="color:var(--ink-2)">' + l.slice(4) + '</span></div>';
    }).join('') + '</div>';

  // ---- distribution histograms -------------------------------------------
  document.getElementById('dist-grid').innerHTML = D.markov.distributions.map(function (d, i) {
    var beats = d.markov_p < 0.05;
    return '<div class="mc-card">' +
      '<div class="mc-card-head"><span class="name">' + d.asset +
        ' <span class="chip cls-' + d.asset_class + '">' + d.asset_class.toLowerCase() + '</span></span>' +
        '<span class="badge-label" style="color:' + (beats ? GOOD : CRIT) + '">p = ' +
        d.markov_p.toFixed(3) + '</span></div>' +
      '<div class="chart-wrap" id="dist-' + i + '"></div>' +
      '<div class="mc-stats"><span>real <b class="' + sgn(d.real_sharpe) + '">' +
        d.real_sharpe.toFixed(2) + '</b></span>' +
        '<span>synthetic mean <b>' + (d.markov_samples.reduce(function (a, b) { return a + b; }, 0) /
          d.markov_samples.length).toFixed(2) + '</b></span>' +
        '<span>' + (beats ? 'not explained by chance' : 'inside the noise') + '</span></div>' +
      '</div>';
  }).join('');
  D.markov.distributions.forEach(function (d, i) {
    RC.histogram(document.getElementById('dist-' + i), {
      samples: d.markov_samples, marker: d.real_sharpe, height: 150,
      barColor: CONTEXT, markerColor: ACCENT, markerLabel: 'real'
    });
  });

  // ---- markov by class ----------------------------------------------------
  document.getElementById('markov-class-grid').innerHTML = D.markov.by_class.map(function (c) {
    var vals = [
      { name: 'real market', v: c.mean_real_sharpe, color: CLS_COLOR[c.asset_class] || ACCENT },
      { name: 'synthetic', v: c.mean_markov_synth, color: CONTEXT }
    ];
    var maxAbs = Math.max.apply(null, vals.map(function (r) { return Math.abs(r.v); })) || 1;
    return '<div class="compare-card">' +
      '<span class="metric-label">' + c.asset_class + ' &middot; ' + c.n + ' instruments</span>' +
      vals.map(function (r) {
        var w = Math.abs(r.v) / maxAbs * 100;
        return '<div class="bar-row"><span class="name">' + r.name + '</span>' +
          '<span class="bar-track"><span class="bar-fill" style="left:' + (r.v < 0 ? (50 - w / 2) : 50) +
          '%;width:' + (w / 2) + '%;background:' + (r.v < 0 ? CRIT : r.color) + '"></span></span>' +
          '<span class="bar-val ' + sgn(r.v) + '">' + r.v.toFixed(2) + '</span></div>';
      }).join('') +
      '<div style="font-size:0.74rem;color:var(--ink-muted);font-family:var(--mono)">' +
      c.n_markov_sig + '/' + c.n + ' clear the Markov null &middot; ' +
      c.n_shuffle_sig + '/' + c.n + ' clear the shuffle</div></div>';
  }).join('');

  // ---- markov table -------------------------------------------------------
  var mkRows = D.markov.flat.slice();
  function pColor(p) { return p < 0.01 ? GOOD : (p < 0.05 ? tok('--ink') : CRIT); }
  function renderMarkov() {
    document.querySelector('#markov-table tbody').innerHTML = mkRows.map(function (r) {
      return '<tr><td>' + r.asset + '</td>' +
        '<td style="text-align:left"><span class="chip cls-' + r.asset_class + '">' +
          r.asset_class.toLowerCase() + '</span></td>' +
        '<td>' + r.n_bars.toLocaleString() + '</td>' +
        '<td>' + r.real_n_trades + '</td>' +
        '<td class="' + sgn(r.real_sharpe) + '">' + r.real_sharpe.toFixed(2) + '</td>' +
        '<td class="' + sgn(r.markov_synth_mean) + '">' + r.markov_synth_mean.toFixed(2) + '</td>' +
        '<td>' + r.markov_z.toFixed(1) + '</td>' +
        '<td style="color:' + pColor(r.markov_p) + ';font-weight:600">' + r.markov_p.toFixed(3) + '</td>' +
        '<td style="color:' + pColor(r.shuffle_p) + '">' + r.shuffle_p.toFixed(3) + '</td>' +
        '<td style="text-align:left;font-family:var(--sans);font-weight:400;font-size:0.78rem;color:' +
          (r.markov_p < 0.05 && r.shuffle_p < 0.05 ? GOOD : (r.markov_p < 0.05 ? 'var(--ink-2)' : CRIT)) +
          '">' + r.verdict + '</td></tr>';
    }).join('');
  }
  renderMarkov();
  var mkDir = {};
  document.querySelectorAll('#markov-table th').forEach(function (th) {
    th.addEventListener('click', function () {
      var k = th.getAttribute('data-key');
      mkDir[k] = !mkDir[k];
      mkRows.sort(function (a, b) {
        var x = a[k], y = b[k];
        if (typeof x === 'string') return mkDir[k] ? x.localeCompare(y) : y.localeCompare(x);
        return mkDir[k] ? x - y : y - x;
      });
      renderMarkov();
    });
  });
  document.getElementById('markov-fdr').innerHTML =
    '<span>' + D.markov.n_sims + ' synthetic universes per asset &middot; ' + D.markov.n_states +
    ' return states &middot; ' + (D.markov.n_sims * ms.n_assets * 2).toLocaleString() +
    ' total simulated backtests</span>' +
    '<span>median Markov p = <b class="num">' + ms.median_markov_p.toFixed(3) + '</b></span>' +
    '<span>after Benjamini-Hochberg (5% FDR): <b class="num">' + ms.n_markov_sig_bh + '/' +
    ms.n_assets + '</b></span>';

  // ---- era heatmap --------------------------------------------------------
  var eras = D.eras.long.eras;
  var eraRows = D.eras.long.rows || [];
  // Build asset -> era -> sharpe from per-asset rows if present, else from rows[]
  var byAsset = {};
  eraRows.forEach(function (r) {
    if (!byAsset[r.asset]) byAsset[r.asset] = { cls: r.asset_class, cells: {} };
    byAsset[r.asset].cells[r.era] = r;
  });
  var assetOrder = D.eras.long.per_asset.slice().sort(function (a, b) {
    return (b.pct_eras_positive - a.pct_eras_positive) || (b.mean_sharpe - a.mean_sharpe);
  }).map(function (r) { return r.asset; });

  var allS = [];
  Object.keys(byAsset).forEach(function (a) {
    Object.keys(byAsset[a].cells).forEach(function (e) { allS.push(byAsset[a].cells[e].sharpe); });
  });
  var maxAbsEra = Math.max.apply(null, allS.map(Math.abs)) || 1;
  function hexToRgb(h) { var n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  function heatColor(s) {
    var t = Math.min(1, Math.abs(s) / maxAbsEra);
    var c = hexToRgb(s >= 0 ? GOOD : CRIT);
    var bg = hexToRgb(tok('--surface-2') || '#f2f1ec');
    return 'rgb(' + c.map(function (ch, i) {
      return Math.round(bg[i] + (ch - bg[i]) * (0.15 + 0.75 * t));
    }).join(',') + ')';
  }
  var eraTbl = '<thead><tr><th style="text-align:left">Instrument</th>' +
    eras.map(function (e) { return '<th>' + e + '</th>'; }).join('') +
    '<th>eras +ve</th></tr></thead><tbody>' +
    assetOrder.map(function (a) {
      var rec = byAsset[a];
      if (!rec) return '';
      var nPos = 0, nTot = 0;
      var cells = eras.map(function (e) {
        var c = rec.cells[e];
        if (!c) return '<td class="heat-cell empty">&ndash;</td>';
        nTot++; if (c.sharpe > 0) nPos++;
        return '<td class="heat-cell" style="background:' + heatColor(c.sharpe) +
          ';color:' + (Math.abs(c.sharpe) / maxAbsEra > 0.55 ? '#fff' : 'var(--ink)') +
          '" title="' + a + ' ' + e + ': Sharpe ' + c.sharpe.toFixed(2) +
          ', ' + c.n_trades + ' trades">' + c.sharpe.toFixed(2) + '</td>';
      }).join('');
      return '<tr><td class="asset-cell">' + a + ' <span class="chip cls-' + rec.cls + '">' +
        (rec.cls === 'CRYPTO' ? 'c' : 'cm') + '</span></td>' + cells +
        '<td class="heat-cell" style="background:transparent;color:var(--ink-2)">' +
        nPos + '/' + nTot + '</td></tr>';
    }).join('') + '</tbody>';
  document.getElementById('era-heatmap').innerHTML = eraTbl;
  document.getElementById('era-legend').innerHTML =
    '<span><span class="legend-dot" style="background:' + GOOD + '"></span>positive Sharpe in that era</span>' +
    '<span><span class="legend-dot" style="background:' + CRIT + '"></span>negative</span>' +
    '<span>' + es.n_assets_all_eras_positive + ' instruments are positive in <i>every</i> era they can be tested in ' +
    '&mdash; but read the caveat below: most crypto has only 2 testable eras.</span>';

  // ---- era compare --------------------------------------------------------
  var maxEraAbs = Math.max.apply(null, D.eras.long.by_era.map(function (e) { return Math.abs(e.mean_sharpe); })) || 1;
  document.getElementById('era-compare').innerHTML = D.eras.long.by_era.map(function (e) {
    var w = Math.abs(e.mean_sharpe) / maxEraAbs * 100;
    return '<div class="compare-card"><span class="metric-label">' + e.era + '</span>' +
      '<div class="bar-row"><span class="name">mean Sharpe</span>' +
      '<span class="bar-track"><span class="bar-fill" style="left:' +
        (e.mean_sharpe < 0 ? (50 - w / 2) : 50) + '%;width:' + (w / 2) + '%;background:' +
        (e.mean_sharpe < 0 ? CRIT : GOOD) + '"></span></span>' +
      '<span class="bar-val ' + sgn(e.mean_sharpe) + '">' + e.mean_sharpe.toFixed(2) + '</span></div>' +
      '<div style="font-size:0.74rem;color:var(--ink-muted);font-family:var(--mono)">' +
        (e.pct_positive * 100).toFixed(0) + '% of ' + e.n_assets + ' instruments positive</div></div>';
  }).join('');
  var worst = D.eras.long.by_era.slice().sort(function (a, b) { return a.mean_sharpe - b.mean_sharpe; })[0];
  document.getElementById('era-prose').innerHTML =
    'No single era carries the result &mdash; which is exactly what you want to see. The weakest window is <b>' +
    worst.era + '</b> at a mean Sharpe of <span class="num">' + worst.mean_sharpe.toFixed(2) +
    '</span>, and the rule was still profitable on ' + (worst.pct_positive * 100).toFixed(0) +
    '% of instruments in it. But note the instrument counts: the early eras contain commodities only, because no ' +
    'crypto existed yet. The 2018&ndash;2023 and 2023&ndash;2026 columns are the only ones where all 30 ' +
    'instruments are present, and they are also the two strongest &mdash; some of that is crypto being added, ' +
    'not the rule improving. Long+short over the same eras: ' +
    (D.eras.both.summary.pct_positive * 100).toFixed(0) + '% of cells positive at a mean Sharpe of ' +
    D.eras.both.summary.mean_sharpe.toFixed(2) + ', worse than long-only on both counts.';

  // ---- parameter sweep ----------------------------------------------------
  var periods = D.extras.periods;
  var sweepLines = Object.keys(D.extras.sweep_by_class).map(function (cls) {
    return {
      label: cls.toLowerCase(), color: CLS_COLOR[cls] || CONTEXT, width: 2,
      values: periods.map(function (p) { return D.extras.sweep_by_class[cls][String(p)]; })
    };
  });
  RC.lineChart(document.getElementById('sweepChart'), {
    n: periods.length, height: 280, lines: sweepLines,
    xLabels: periods.map(String), xTickCount: 11,
    yFormat: function (v) { return v.toFixed(2); },
    tooltipTitle: function (i) { return 'SMA ' + periods[i]; },
    marks: [{ index: periods.indexOf(18), label: 'SMA 18', color: tok('--ink') }]
  });
  document.getElementById('param-tiles').innerHTML = [
    tile('Median rank of SMA-18', xs.median_rank_of_18 + ' / ' + periods.length,
         '1 would mean 18 is the single best length &mdash; a curve-fit signature'),
    tile('Assets where 18 is best', xs.n_assets_where_18_is_best + ' / 30',
         'zero is the ideal answer here'),
    tile('Lengths that are profitable', (xs.mean_pct_periods_positive * 100).toFixed(0) + '%',
         'averaged over all 30 instruments, lengths 5&ndash;60')
  ].join('');

  // ---- cost sweep ---------------------------------------------------------
  var costs = D.extras.cost_bps;
  var costLines = Object.keys(D.extras.cost_by_class).map(function (cls) {
    return {
      label: cls.toLowerCase(), color: CLS_COLOR[cls] || CONTEXT, width: 2,
      values: costs.map(function (b) { return D.extras.cost_by_class[cls][String(b)]; })
    };
  });
  RC.lineChart(document.getElementById('costChart'), {
    n: costs.length, height: 260, lines: costLines,
    xLabels: costs.map(function (b) { return b + 'bps'; }), xTickCount: costs.length - 1,
    yFormat: function (v) { return v.toFixed(2); },
    tooltipTitle: function (i) { return costs[i] + ' bps round trip'; },
    zeroLine: true
  });
  document.getElementById('cost-tiles').innerHTML = [
    tile('Positive at 10bps', xs.n_assets_positive_at_10bps + ' / 30', 'round-trip cost on every trade'),
    tile('Positive at 20bps', xs.n_assets_positive_at_20bps + ' / 30', 'a realistic retail commodity round trip'),
    tile('Commodity @ 50bps',
         D.extras.cost_by_class.COMMODITY['50'].toFixed(2),
         'from ' + D.extras.cost_by_class.COMMODITY['0'].toFixed(2) + ' gross &mdash; the edge is gone',
         sgn(D.extras.cost_by_class.COMMODITY['50'])),
    tile('Crypto @ 50bps',
         D.extras.cost_by_class.CRYPTO['50'].toFixed(2),
         'from ' + D.extras.cost_by_class.CRYPTO['0'].toFixed(2) + ' gross &mdash; barely dented',
         sgn(D.extras.cost_by_class.CRYPTO['50']))
  ].join('');

  // ---- cost table ---------------------------------------------------------
  var costRows = D.extras.per_asset_costs.slice().sort(function (a, b) { return b.sharpe_20 - a.sharpe_20; });
  function renderCosts() {
    document.querySelector('#cost-table tbody').innerHTML = costRows.map(function (r) {
      return '<tr><td>' + r.asset + '</td>' +
        '<td style="text-align:left"><span class="chip cls-' + r.asset_class + '">' +
          r.asset_class.toLowerCase() + '</span></td>' +
        '<td>' + r.n_trades + '</td>' +
        '<td class="' + sgn(r.sharpe_0) + '">' + r.sharpe_0.toFixed(2) + '</td>' +
        '<td class="' + sgn(r.sharpe_10) + '">' + r.sharpe_10.toFixed(2) + '</td>' +
        '<td class="' + sgn(r.sharpe_20) + '">' + r.sharpe_20.toFixed(2) + '</td>' +
        '<td style="font-weight:600;color:' +
          (String(r.breakeven_bps).indexOf('>') === 0 ? GOOD : CRIT) + '">' +
          r.breakeven_bps + '</td></tr>';
    }).join('');
  }
  renderCosts();
  var cDir = {};
  document.querySelectorAll('#cost-table th').forEach(function (th) {
    th.addEventListener('click', function () {
      var k = th.getAttribute('data-key');
      cDir[k] = !cDir[k];
      costRows.sort(function (a, b) {
        var x = a[k], y = b[k];
        if (k === 'breakeven_bps') { x = String(x).indexOf('>') === 0 ? 999 : +x; y = String(y).indexOf('>') === 0 ? 999 : +y; }
        if (typeof x === 'string') return cDir[k] ? x.localeCompare(y) : y.localeCompare(x);
        return cDir[k] ? x - y : y - x;
      });
      renderCosts();
    });
  });
})();
