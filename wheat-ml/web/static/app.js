/* 豫麦智测 · 图表主题、进入动效、上传校验与二次确认 */
(function () {
  'use strict';

  var MUTED = '#5A6657';
  var GRID = 'rgba(27,36,26,.08)';
  var GREEN = '#1D4A2C';
  var GOLD = '#D2A24C';
  var RAIN = '#5C7E86';
  var RED = '#B4472F';
  var FIELD = '#5B7F3F';
  var reduceMotion = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var numberFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 });
  var FONT = window.getComputedStyle(document.body).fontFamily;

  function onReady(fn) {
    if (document.readyState !== 'loading') {
      fn();
    } else {
      document.addEventListener('DOMContentLoaded', fn);
    }
  }

  function formatNumber(value) {
    return numberFormat.format(value);
  }

  function formatSize(bytes) {
    if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  }

  function hexToRgba(hex, alpha) {
    var value = String(hex || '').replace('#', '');
    if (value.length === 3) {
      value = value.split('').map(function (ch) { return ch + ch; }).join('');
    }
    var number = parseInt(value, 16);
    if (isNaN(number)) return hex;
    return 'rgba(' + ((number >> 16) & 255) + ',' + ((number >> 8) & 255) + ',' +
      (number & 255) + ',' + alpha + ')';
  }

  function tickLimit(count) {
    if (count <= 12) return count;
    if (count <= 24) return 8;
    return 7;
  }

  function xAxis(count, shortenDates) {
    var ticks = {
      color: MUTED,
      maxRotation: 0,
      autoSkip: true,
      maxTicksLimit: tickLimit(count),
      padding: 8,
      font: { size: 10.5 }
    };
    if (shortenDates) {
      ticks.callback = function (value) {
        var label = this.getLabelForValue(value);
        return label.length === 10 && label.charAt(4) === '-' ? label.slice(5) : label;
      };
    }
    return {
      grid: { display: false },
      border: { display: false },
      ticks: ticks
    };
  }

  function yAxis(title, right, beginAtZero) {
    return {
      position: right ? 'right' : 'left',
      beginAtZero: !!beginAtZero,
      grid: right
        ? { drawOnChartArea: false, color: 'transparent' }
        : { color: GRID, drawTicks: false },
      border: { display: false },
      ticks: {
        color: MUTED,
        maxTicksLimit: 6,
        padding: 8,
        font: { size: 10.5 },
        callback: function (value) { return formatNumber(value); }
      },
      title: {
        display: !!title,
        text: title || '',
        color: MUTED,
        padding: { top: 4, bottom: 4 },
        font: { size: 10.5, weight: '400' }
      }
    };
  }

  function defaultLabel(context) {
    var value = context.parsed ? context.parsed.y : null;
    if (value === null || value === undefined) return context.dataset.label + '：—';
    var unit = context.dataset.unit || '';
    return context.dataset.label + '：' + formatNumber(value) + (unit ? ' ' + unit : '');
  }

  function chartOptions(options) {
    options = options || {};
    var legendLabels = {
      usePointStyle: true,
      pointStyle: 'circle',
      boxWidth: 7,
      boxHeight: 7,
      padding: 16,
      color: MUTED,
      font: { size: 11.5 }
    };
    if (options.legendFilter) legendLabels.filter = options.legendFilter;

    var tooltip = {
      backgroundColor: '#FFFDF7',
      titleColor: '#1B241A',
      bodyColor: '#5A6657',
      borderColor: 'rgba(29,74,44,.18)',
      borderWidth: 1,
      padding: 12,
      cornerRadius: 9,
      usePointStyle: true,
      boxWidth: 7,
      boxHeight: 7,
      titleFont: { size: 11, weight: '600' },
      bodyFont: { size: 11.5 },
      callbacks: { label: options.labelCallback || defaultLabel }
    };
    if (options.tooltipFilter) tooltip.filter = options.tooltipFilter;

    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: reduceMotion ? false : { duration: 700, easing: 'easeOutQuart' },
      interaction: options.interaction || { mode: 'index', intersect: false },
      layout: { padding: { top: 4, right: 4, bottom: 0, left: 0 } },
      plugins: {
        legend: {
          display: !!options.legend,
          position: 'top',
          align: 'end',
          labels: legendLabels
        },
        tooltip: tooltip
      },
      scales: options.scales || {}
    };
  }

  function renderLine(card, spec) {
    var payload = spec.payload || {};
    var canvas = card.querySelector('canvas');
    if (!canvas || !payload.labels) return;

    var hasY2 = false;
    var datasets = (payload.datasets || []).map(function (dataset) {
      if (dataset.axis === 'y2') hasY2 = true;
      return {
        label: dataset.label,
        data: dataset.data,
        unit: dataset.axis === 'y2' ? (payload.y2Label || '') : (payload.yLabel || ''),
        borderColor: dataset.faint ? hexToRgba(dataset.color, 0.28) : dataset.color,
        backgroundColor: 'transparent',
        borderWidth: dataset.faint ? 1 : 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: '#FFFDF7',
        pointHoverBorderColor: dataset.color,
        pointHoverBorderWidth: 2,
        tension: 0.3,
        spanGaps: true,
        yAxisID: dataset.axis === 'y2' ? 'y2' : 'y'
      };
    });

    var scales = {
      x: xAxis(payload.labels.length, payload.labels.length > 40),
      y: yAxis(payload.yLabel || '', false, false)
    };
    if (hasY2) scales.y2 = yAxis(payload.y2Label || '', true, false);

    new Chart(canvas, {
      type: 'line',
      data: { labels: payload.labels, datasets: datasets },
      options: chartOptions({ legend: datasets.length > 1, scales: scales })
    });
  }

  function renderPrecip(card, spec) {
    var payload = spec.payload || {};
    var canvas = card.querySelector('canvas');
    if (!canvas || !payload.labels) return;

    var hasCumulative = !!(payload.cumulative && payload.cumulative.length);
    var datasets = [{
      type: 'bar',
      label: payload.barLabel || '降水量',
      data: payload.bars,
      unit: payload.yLabel || 'mm',
      backgroundColor: hexToRgba(RAIN, 0.5),
      hoverBackgroundColor: hexToRgba(RAIN, 0.82),
      borderRadius: 3,
      borderSkipped: 'start',
      maxBarThickness: 26,
      order: 2,
      yAxisID: 'y'
    }];

    if (hasCumulative) {
      datasets.push({
        type: 'line',
        label: payload.lineLabel || '累计',
        data: payload.cumulative,
        unit: payload.yLabel || 'mm',
        borderColor: GOLD,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25,
        spanGaps: true,
        order: 1,
        yAxisID: 'y2'
      });
    }

    var scales = {
      x: xAxis(payload.labels.length, payload.labels.length > 40),
      y: yAxis(payload.yLabel || 'mm', false, true)
    };
    if (hasCumulative) scales.y2 = yAxis('累计 / mm', true, true);

    new Chart(canvas, {
      type: 'bar',
      data: { labels: payload.labels, datasets: datasets },
      options: chartOptions({ legend: hasCumulative, scales: scales })
    });
  }

  function renderMonthCompare(card, spec) {
    var payload = spec.payload || {};
    var canvases = card.querySelectorAll('canvas');
    var channels = [
      { data: payload.temp, beginAtZero: false },
      { data: payload.precip, beginAtZero: true }
    ];

    channels.forEach(function (channel, index) {
      var canvas = canvases[index];
      if (!canvas || !channel.data || !channel.data.datasets) return;

      var datasets = channel.data.datasets.slice().sort(function (a, b) {
        return (a.highlight ? 1 : 0) - (b.highlight ? 1 : 0);
      }).map(function (dataset) {
        return {
          label: dataset.label,
          data: dataset.data,
          unit: channel.data.unit || '',
          borderColor: dataset.highlight ? GREEN : '#C9D6B4',
          backgroundColor: 'transparent',
          borderWidth: dataset.highlight ? 2.4 : 1,
          pointRadius: dataset.highlight ? 2.5 : 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: '#FFFDF7',
          pointHoverBorderColor: dataset.highlight ? GREEN : FIELD,
          tension: 0.25,
          spanGaps: true
        };
      });

      new Chart(canvas, {
        type: 'line',
        data: { labels: payload.labels, datasets: datasets },
        options: chartOptions({
          legend: true,
          legendFilter: function (item) {
            return item.text === String(payload.highlight);
          },
          interaction: { mode: 'nearest', intersect: false },
          scales: {
            x: xAxis(payload.labels.length),
            y: yAxis(channel.data.unit || '', false, channel.beginAtZero)
          }
        })
      });
    });
  }

  function renderPanelBars(card, spec) {
    var panels = (spec.payload || {}).panels || [];
    var canvases = card.querySelectorAll('canvas');

    panels.forEach(function (panel, index) {
      var canvas = canvases[index];
      if (!canvas) return;

      var meanData = panel.values.map(function () { return panel.mean; });
      var colors = panel.labels.map(function (label) {
        return Number(label) === panel.highlight ? GREEN : '#D8E2CB';
      });

      new Chart(canvas, {
        type: 'bar',
        data: {
          labels: panel.labels,
          datasets: [
            {
              type: 'bar',
              label: panel.title,
              data: panel.values,
              unit: panel.unit,
              backgroundColor: colors,
              hoverBackgroundColor: hexToRgba(GREEN, 0.84),
              borderRadius: 3,
              maxBarThickness: 24,
              order: 2
            },
            {
              type: 'line',
              label: '多年平均',
              data: meanData,
              unit: panel.unit,
              borderColor: GOLD,
              backgroundColor: 'transparent',
              borderWidth: 1.6,
              borderDash: [5, 4],
              pointRadius: 0,
              fill: false,
              order: 1
            }
          ]
        },
        options: chartOptions({
          legend: false,
          labelCallback: function (context) {
            return panel.title + '：' + formatNumber(context.parsed.y) +
              (panel.unit ? ' ' + panel.unit : '');
          },
          tooltipFilter: function (item) { return item.datasetIndex === 0; },
          scales: {
            x: xAxis(panel.labels.length),
            y: yAxis(panel.unit || '', false, true)
          }
        })
      });
    });
  }

  function renderPredictionTrend(card, spec) {
    var payload = spec.payload || {};
    var canvas = card.querySelector('canvas');
    if (!canvas || !payload.labels) return;

    var selectedIndex = payload.labels.indexOf(String(payload.selected));
    var actual = payload.actual || [];
    var predicted = payload.predicted || [];
    var values = actual.concat(predicted).filter(function (value) {
      return typeof value === 'number';
    });
    values.push(payload.threshold);

    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var padding = Math.max((max - min) * 0.14, 5);
    var pointRadius = function (context) {
      return context.dataIndex === selectedIndex ? 5.5 : 3;
    };

    var scales = {
      x: xAxis(payload.labels.length),
      y: yAxis('kg/亩', false, false)
    };
    scales.y.suggestedMin = Math.max(0, Math.floor(min - padding));
    scales.y.suggestedMax = Math.ceil(max + padding);

    new Chart(canvas, {
      type: 'line',
      data: {
        labels: payload.labels,
        datasets: [
          {
            label: '实测单产',
            data: actual,
            unit: 'kg/亩',
            borderColor: GREEN,
            backgroundColor: GREEN,
            borderWidth: 2,
            pointRadius: pointRadius,
            pointHoverRadius: 6,
            pointBorderColor: '#FFFDF7',
            pointBorderWidth: 1.5,
            tension: 0.25,
            spanGaps: true,
            order: 2
          },
          {
            label: '预测单产',
            data: predicted,
            unit: 'kg/亩',
            borderColor: GOLD,
            backgroundColor: GOLD,
            borderWidth: 2,
            borderDash: [6, 3],
            pointRadius: pointRadius,
            pointHoverRadius: 6,
            pointBorderColor: '#FFFDF7',
            pointBorderWidth: 1.5,
            tension: 0.25,
            spanGaps: true,
            order: 1
          },
          {
            label: '预警线',
            data: payload.labels.map(function () { return payload.threshold; }),
            unit: 'kg/亩',
            borderColor: hexToRgba(RED, 0.82),
            backgroundColor: 'transparent',
            borderWidth: 1.4,
            borderDash: [4, 5],
            pointRadius: 0,
            fill: false,
            order: 0
          }
        ]
      },
      options: chartOptions({
        legend: true,
        scales: scales,
        interaction: { mode: 'index', intersect: false }
      })
    });
  }

  function renderHBar(card, spec) {
    var payload = spec.payload || {};
    var canvas = card.querySelector('canvas');
    if (!canvas || !payload.labels) return;

    var values = payload.values || [];
    var max = Math.max.apply(null, values.concat([0]));
    var unit = payload.unit || '';
    var colors = values.map(function (value) {
      if (value >= max * 0.5) return GREEN;
      if (value >= max * 0.2) return FIELD;
      return '#C9D6B4';
    });

    var options = chartOptions({
      legend: false,
      interaction: { mode: 'nearest', intersect: true },
      labelCallback: function (context) {
        return context.dataset.label + '：' + formatNumber(context.parsed.x) + (unit ? ' ' + unit : '');
      },
      scales: {
        x: {
          beginAtZero: true,
          suggestedMax: max * 1.08,
          grid: { color: GRID, drawTicks: false },
          border: { display: false },
          ticks: {
            color: MUTED,
            maxTicksLimit: 6,
            font: { size: 10.5 },
            callback: function (value) { return formatNumber(value) + (unit ? unit : ''); }
          }
        },
        y: {
          grid: { display: false },
          border: { display: false },
          ticks: { color: MUTED, font: { size: 10.5 }, autoSkip: false }
        }
      }
    });
    options.indexAxis = 'y';

    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: payload.labels,
        datasets: [{
          label: spec.title || '特征重要性',
          data: values,
          unit: unit,
          backgroundColor: colors,
          hoverBackgroundColor: hexToRgba(GREEN, 0.84),
          borderRadius: 3,
          maxBarThickness: 18
        }]
      },
      options: options
    });
  }

  var renderers = {
    line: renderLine,
    precip: renderPrecip,
    monthCompare: renderMonthCompare,
    panelBars: renderPanelBars,
    predictionTrend: renderPredictionTrend,
    hBar: renderHBar
  };

  function initCharts() {
    var scripts = document.querySelectorAll('.chart-spec');
    if (!scripts.length) return;

    if (typeof Chart === 'undefined') {
      scripts.forEach(function (script) {
        var card = script.closest('.chart-card');
        if (!card) return;
        card.querySelectorAll('.chart-body').forEach(function (body) {
          body.innerHTML = '<div class="chart-fallback">图表组件未加载，请刷新页面或检查静态资源。</div>';
        });
      });
      return;
    }

    Chart.defaults.font.family = FONT;
    Chart.defaults.font.size = 11;
    Chart.defaults.color = MUTED;

    scripts.forEach(function (script) {
      var card = script.closest('.chart-card');
      if (!card) return;
      var spec;
      try {
        spec = JSON.parse(script.textContent);
      } catch (error) {
        console.error('图表数据解析失败', error);
        return;
      }
      var renderer = renderers[spec.kind];
      if (renderer) renderer(card, spec);
    });
  }

  function initHeaderState() {
    var head = document.getElementById('site-head');
    if (!head) return;
    function update() {
      head.classList.toggle('is-scrolled', window.scrollY > 8);
    }
    update();
    window.addEventListener('scroll', update, { passive: true });
  }

  function initActiveNav() {
    var nav = document.querySelector('.main-nav');
    var active = nav && nav.querySelector('a.is-active');
    if (!nav || !active || nav.scrollWidth <= nav.clientWidth) return;

    var navRect = nav.getBoundingClientRect();
    var activeRect = active.getBoundingClientRect();
    var target = nav.scrollLeft + (activeRect.left - navRect.left) -
      (nav.clientWidth - activeRect.width) / 2;
    nav.scrollTo({
      left: Math.max(0, target),
      behavior: reduceMotion ? 'auto' : 'smooth'
    });
  }

  function initSkipLink() {
    var skip = document.querySelector('.skip-link');
    var main = document.getElementById('main');
    if (!skip || !main) return;
    skip.addEventListener('click', function () {
      window.setTimeout(function () {
        main.focus({ preventScroll: true });
      }, 0);
    });
  }

  function initReveals() {
    var elements = Array.prototype.slice.call(document.querySelectorAll('[data-reveal]'));
    if (!elements.length) return;
    if (reduceMotion || !('IntersectionObserver' in window)) {
      elements.forEach(function (element) { element.classList.add('is-in'); });
      return;
    }
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-in');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
    elements.forEach(function (element) { observer.observe(element); });
  }

  function initCounters() {
    var elements = Array.prototype.slice.call(document.querySelectorAll('[data-count]'));
    if (!elements.length || reduceMotion || !('IntersectionObserver' in window)) return;
    var counterFormat = new Intl.NumberFormat('zh-CN');
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        observer.unobserve(entry.target);
        var element = entry.target;
        var target = parseFloat(element.getAttribute('data-count'));
        if (!isFinite(target)) return;
        var start = null;
        var duration = 900;
        function step(timestamp) {
          if (start === null) start = timestamp;
          var progress = Math.min((timestamp - start) / duration, 1);
          var eased = 1 - Math.pow(1 - progress, 3);
          element.textContent = counterFormat.format(Math.round(target * eased));
          if (progress < 1) {
            window.requestAnimationFrame(step);
          } else {
            element.textContent = counterFormat.format(target);
          }
        }
        window.requestAnimationFrame(step);
      });
    }, { threshold: 0.4 });
    elements.forEach(function (element) { observer.observe(element); });
  }

  function initInsightJump() {
    var link = document.querySelector('.insight-link');
    var detail = document.getElementById('year-detail');
    if (!detail) return;

    function openDetail() {
      detail.open = true;
      var summary = detail.querySelector('summary');
      if (summary) summary.focus({ preventScroll: true });
      detail.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, '', '#year-detail');
      }
    }

    if (link) {
      link.addEventListener('click', function (event) {
        event.preventDefault();
        openDetail();
      });
    }
    if (window.location.hash === '#year-detail') {
      window.setTimeout(openDetail, 0);
    }
  }

  function initNavigationFeedback() {
    var bar = document.getElementById('nav-progress');
    if (!bar) return;
    var active = false;
    var timer = null;

    function reset() {
      active = false;
      if (timer) window.clearTimeout(timer);
      timer = null;
      bar.classList.remove('is-active', 'is-done');
      bar.style.width = '0';
    }

    function show() {
      if (active) return;
      active = true;
      bar.classList.add('is-active');
      bar.style.width = '12%';
      timer = window.setTimeout(function () {
        if (active) bar.style.width = '62%';
      }, 220);
    }

    function isInternalLink(link) {
      var href = link.getAttribute('href');
      if (!href || href.charAt(0) === '#' || link.target === '_blank' || link.hasAttribute('download')) return false;
      if (/^(mailto:|tel:|javascript:)/i.test(href)) return false;
      var url;
      try { url = new URL(link.href, window.location.href); } catch (error) { return false; }
      if (url.origin !== window.location.origin) return false;
      if (url.pathname === window.location.pathname && url.search === window.location.search && url.hash) return false;
      return true;
    }

    document.addEventListener('click', function (event) {
      if (event.defaultPrevented) return;
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      var link = event.target.closest('a');
      if (!link || !isInternalLink(link)) return;
      window.setTimeout(show, 150);
    });

    document.addEventListener('submit', function (event) {
      if (event.defaultPrevented) return;
      var form = event.target;
      if (form.id === 'upload-form') return;
      var button = form.querySelector('button[type="submit"]');
      if (button) {
        button.disabled = true;
        button.textContent = '加载中…';
      }
      show();
    });

    window.addEventListener('pageshow', reset);
  }

  function initUpload() {
    var zone = document.getElementById('dropzone');
    var input = document.getElementById('file-input');
    var form = document.getElementById('upload-form');
    var submit = document.getElementById('upload-submit');
    if (!zone || !input) return;

    var fileLabel = zone.querySelector('[data-file-name]');
    var maxSize = parseFloat(zone.getAttribute('data-max-size')) || 50;
    var allowed = ['.csv', '.xlsx', '.xls'];

    function checkFile() {
      var file = input.files && input.files[0];
      if (!file) {
        fileLabel.textContent = '尚未选择文件';
        fileLabel.classList.remove('has-file');
        return false;
      }
      var dot = file.name.lastIndexOf('.');
      var extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : '';
      var sizeMB = file.size / 1024 / 1024;

      if (allowed.indexOf(extension) < 0) {
        fileLabel.textContent = '不支持的文件类型，请选择 CSV / Excel';
        fileLabel.classList.remove('has-file');
        return false;
      }
      if (sizeMB > maxSize) {
        fileLabel.textContent = '文件 ' + sizeMB.toFixed(1) + ' MB，超过 ' + maxSize + ' MB 限制';
        fileLabel.classList.remove('has-file');
        return false;
      }
      fileLabel.textContent = file.name + ' · ' + formatSize(file.size);
      fileLabel.classList.add('has-file');
      return true;
    }

    ['dragenter', 'dragover'].forEach(function (name) {
      zone.addEventListener(name, function (event) {
        event.preventDefault();
        zone.classList.add('is-drag');
      });
    });
    ['dragleave', 'dragend', 'drop'].forEach(function (name) {
      zone.addEventListener(name, function () {
        zone.classList.remove('is-drag');
      });
    });
    zone.addEventListener('drop', function (event) {
      event.preventDefault();
      if (event.dataTransfer && event.dataTransfer.files.length) {
        try {
          input.files = event.dataTransfer.files;
        } catch (error) {
          return;
        }
        checkFile();
      }
    });

    input.addEventListener('change', checkFile);

    if (form) {
      form.addEventListener('submit', function (event) {
        if (!checkFile()) {
          event.preventDefault();
          return;
        }
        if (submit) {
          submit.disabled = true;
          submit.textContent = '正在校验…';
        }
      });
    }

    window.addEventListener('pageshow', function () {
      if (submit) {
        submit.disabled = false;
        submit.textContent = '上传并校验';
      }
    });
  }

  function initConfirms() {
    document.querySelectorAll('[data-confirm]').forEach(function (element) {
      element.addEventListener('click', function (event) {
        var message = element.getAttribute('data-confirm');
        if (message && !window.confirm(message)) {
          event.preventDefault();
        }
      });
    });
  }

  onReady(function () {
    initHeaderState();
    initActiveNav();
    initSkipLink();
    initReveals();
    initCounters();
    initInsightJump();
    initUpload();
    initConfirms();
    initNavigationFeedback();
    initCharts();
  });

  window.addEventListener('resize', function () {
    initActiveNav();
  }, { passive: true });
})();
