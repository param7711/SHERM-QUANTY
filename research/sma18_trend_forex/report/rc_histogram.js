  // Histogram with a single highlighted marker line — used to show where the
  // real result sits inside the distribution of results on synthetic markets.
  function histogram(container, cfg) {
    var W = 420, H = cfg.height || 160, mL = 30, mR = 10, mT = 10, mB = 22;
    var plotW = W - mL - mR, plotH = H - mT - mB;
    var s = cfg.samples.filter(function (v) { return v != null && !isNaN(v); });
    var nBins = cfg.bins || 28;
    var lo = Math.min.apply(null, s), hi = Math.max.apply(null, s);
    if (cfg.marker != null) { lo = Math.min(lo, cfg.marker); hi = Math.max(hi, cfg.marker); }
    var pad = (hi - lo) * 0.04 || 0.1;
    lo -= pad; hi += pad;
    var bw = (hi - lo) / nBins;
    var counts = new Array(nBins).fill(0);
    s.forEach(function (v) {
      var b = Math.floor((v - lo) / bw);
      counts[Math.max(0, Math.min(nBins - 1, b))]++;
    });
    var maxC = Math.max.apply(null, counts) || 1;

    function xScale(v) { return mL + (v - lo) / (hi - lo) * plotW; }
    var mono = "'IBM Plex Mono',monospace";

    var bars = counts.map(function (c, i) {
      var x0 = xScale(lo + i * bw), x1 = xScale(lo + (i + 1) * bw);
      var h = c / maxC * plotH;
      return '<rect class="hist-bar" data-i="' + i + '" x="' + (x0 + 0.5) + '" y="' + (mT + plotH - h) +
        '" width="' + Math.max(0.5, x1 - x0 - 1) + '" height="' + h + '" fill="' + cfg.barColor +
        '" opacity="0.65"/>';
    }).join('');

    var axis = '<line x1="' + mL + '" y1="' + (mT + plotH) + '" x2="' + (W - mR) + '" y2="' + (mT + plotH) +
      '" stroke="' + THEME.grid + '" stroke-width="1"/>';
    var zx = xScale(0);
    var zero = (0 > lo && 0 < hi)
      ? '<line x1="' + zx + '" y1="' + mT + '" x2="' + zx + '" y2="' + (mT + plotH) + '" stroke="' +
        THEME.grid + '" stroke-width="1" stroke-dasharray="2 3"/>' +
        '<text x="' + zx + '" y="' + (H - 6) + '" text-anchor="middle" font-size="8" fill="' +
        THEME.muted + '" font-family="' + mono + '">0</text>'
      : '';

    var mk = '';
    if (cfg.marker != null) {
      var mx = xScale(cfg.marker);
      var anchor = mx > mL + plotW * 0.7 ? 'end' : 'start';
      var lx = mx + (anchor === 'end' ? -4 : 4);
      mk = '<line x1="' + mx + '" y1="' + (mT - 2) + '" x2="' + mx + '" y2="' + (mT + plotH) +
        '" stroke="' + cfg.markerColor + '" stroke-width="2"/>' +
        '<text x="' + lx + '" y="' + (mT + 8) + '" text-anchor="' + anchor + '" font-size="9" fill="' +
        cfg.markerColor + '" font-family="' + mono + '" font-weight="600">' +
        (cfg.markerLabel || '') + ' ' + cfg.marker.toFixed(2) + '</text>';
    }

    var xlab = [lo, (lo + hi) / 2, hi].map(function (v) {
      return '<text x="' + xScale(v) + '" y="' + (H - 6) + '" text-anchor="middle" font-size="8" fill="' +
        THEME.muted + '" font-family="' + mono + '">' + v.toFixed(1) + '</text>';
    }).join('');
    var ylab = '<text x="' + (mL - 5) + '" y="' + (mT + 8) + '" text-anchor="end" font-size="8" fill="' +
      THEME.muted + '" font-family="' + mono + '">' + maxC + '</text>' +
      '<text x="' + (mL - 5) + '" y="' + (mT + plotH) + '" text-anchor="end" font-size="8" fill="' +
      THEME.muted + '" font-family="' + mono + '">0</text>';

    container.innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;height:auto;display:block" preserveAspectRatio="xMidYMid meet">' +
        bars + axis + zero + mk + xlab + ylab +
      '</svg><div class="chart-tooltip" style="opacity:0"></div>';

    var tooltip = container.querySelector('.chart-tooltip');
    container.querySelectorAll('.hist-bar').forEach(function (bar) {
      var i = parseInt(bar.getAttribute('data-i'), 10);
      bar.addEventListener('pointerenter', function (evt) {
        bar.setAttribute('opacity', 1);
        tooltip.innerHTML = '<div style="color:' + THEME.ink + '">Sharpe ' + (lo + i * bw).toFixed(2) +
          ' to ' + (lo + (i + 1) * bw).toFixed(2) + '</div><div>' + counts[i] + ' of ' + s.length +
          ' synthetic runs</div>';
        positionTooltip(container, tooltip, evt);
        tooltip.style.opacity = 1;
      });
      bar.addEventListener('pointermove', function (evt) { positionTooltip(container, tooltip, evt); });
      bar.addEventListener('pointerleave', function () {
        bar.setAttribute('opacity', 0.65); tooltip.style.opacity = 0;
      });
    });
  }

