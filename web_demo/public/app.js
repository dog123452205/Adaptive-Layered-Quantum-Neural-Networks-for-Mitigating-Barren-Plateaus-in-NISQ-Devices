const datasetSelect = document.getElementById('datasetSelect');
const modelList = document.getElementById('modelList');
const sampleIndex = document.getElementById('sampleIndex');
const sampleRange = document.getElementById('sampleRange');
const randomBtn = document.getElementById('randomBtn');
const runBtn = document.getElementById('runBtn');
const introToggle = document.getElementById('introToggle');
const introBody = document.getElementById('introBody');

const resultEmpty = document.getElementById('resultEmpty');
const resultBody = document.getElementById('resultBody');
const sampleTitle = document.getElementById('sampleTitle');
const trueLabelBadge = document.getElementById('trueLabelBadge');
const featureTable = document.getElementById('featureTable');
const verdictCards = document.getElementById('verdictCards');

const statsEmpty = document.getElementById('statsEmpty');
const statsTable = document.getElementById('statsTable');
const statsHead = document.getElementById('statsHead');
const statsBody = document.getElementById('statsBody');

let datasets = [];
// stable color per model key, regardless of how many models a dataset has
const MODEL_COLORS = {
  rule: '#34e0e0',      // cyan
  greedy: '#f5a623',    // amber
  ppo: '#a56bff',       // purple
  dqn: '#ff6fae',       // pink
  classical: '#5b8cff', // blue
  rf: '#7ee787',        // green
};
const MUTED = '#8894b8';
const FALLBACK_COLORS = ['#34e0e0', '#f5a623', '#a56bff', '#ff6fae', '#5b8cff', '#7ee787'];
function colorFor(key, i) { return MODEL_COLORS[key] || FALLBACK_COLORS[i % FALLBACK_COLORS.length]; }

let metricsChart, timingChart, confidenceChart, rocChart;

Chart.defaults.color = MUTED;
Chart.defaults.borderColor = '#232b45';
Chart.defaults.font.size = 11;

introToggle.addEventListener('click', () => {
  introBody.classList.toggle('hidden');
  introToggle.textContent = introBody.classList.contains('hidden')
    ? 'Introduction (click to show/hide) ▸' : 'Introduction (click to show/hide) ▾';
});

async function loadDatasets() {
  const res = await fetch('/api/datasets');
  datasets = await res.json();
  datasetSelect.innerHTML = datasets.map(
    (d) => `<option value="${d.key}">${d.friendly} (${d.nTotal ? d.nTotal + ' samples, ' : ''}${d.nSamples} held out for test)</option>`
  ).join('');
  await onDatasetChange();
}

datasetSelect.addEventListener('change', () => {
  sampleIndex.value = '';
  onDatasetChange();
});

function currentDataset() {
  return datasets.find((d) => d.key === datasetSelect.value);
}

async function onDatasetChange() {
  const d = currentDataset();
  if (!d) return;
  sampleRange.textContent = `Valid range: 0 – ${d.nSamples - 1} in the held-out test set (leave "Sample #" blank to pick randomly)`;
  sampleIndex.max = d.nSamples - 1;
  resultEmpty.classList.remove('hidden');
  resultBody.classList.add('hidden');

  modelList.innerHTML = d.models.map((m) => `
    <label class="model-row">
      <input type="checkbox" value="${m.key}" checked> <span>${m.label}</span>
    </label>`).join('');

  await loadStats(d.key);
  await loadConfidence(d.key);
}

async function loadStats(key) {
  const res = await fetch(`/api/stats/${key}`);
  const data = await res.json();
  const s = data.stats;
  const models = data.models || [];
  if (!s || models.length === 0) {
    statsEmpty.classList.remove('hidden');
    statsEmpty.textContent = 'Not enough statistics available for this task yet (missing stats_*.json).';
    statsTable.classList.add('hidden');
    destroyChart('metricsChart'); destroyChart('timingChart');
    return;
  }
  statsEmpty.classList.add('hidden');
  statsTable.classList.remove('hidden');

  statsHead.innerHTML = '<tr><th>Metric</th>' + models.map((m) => `<th>${m.label}</th>`).join('') + '</tr>';

  const metricRows = [
    ['Accuracy', 'accuracy', 'pct'],
    ['Precision', 'precision', 'pct'],
    ['Recall', 'recall', 'pct'],
    ['F1', 'f1', 'pct'],
    ['AUC', 'auc', 'pct'],
    ['Avg. inference time (ms/sample)', 'avg_infer_ms', 'ms_lower_better'],
    ['Total time on test set (ms)', 'total_infer_ms', 'ms_lower_better'],
  ];

  statsBody.innerHTML = metricRows.map(([label, field, kind]) => {
    const vals = models.map((m) => (s[m.statsKey] ? s[m.statsKey][field] : null));
    let cells;
    if (kind === 'pct') {
      const best = Math.max(...vals.filter((v) => v != null));
      cells = vals.map((v) => v == null ? '<td>—</td>'
        : `<td class="${v === best ? 'better' : ''}">${(v * 100).toFixed(1)}%</td>`);
    } else {
      const present = vals.filter((v) => v != null);
      const best = present.length ? Math.min(...present) : null;
      cells = vals.map((v) => v == null ? '<td>—</td>'
        : `<td class="${v === best ? 'better' : ''}">${v.toFixed(3)}</td>`);
    }
    return `<tr><td>${label}</td>${cells.join('')}</tr>`;
  }).join('');

  const qubitVals = models.map((m) => (s[m.statsKey] && (s[m.statsKey].n_qubits ?? s[m.statsKey].n_features)) ?? '—');
  statsBody.innerHTML += `<tr><td>Qubits / features</td>${qubitVals.map((v) => `<td>${v}</td>`).join('')}</tr>`;

  renderMetricsChart(s, models);
  renderTimingChart(s, models);
}

function destroyChart(name) {
  const map = { metricsChart, timingChart, confidenceChart, rocChart };
  if (map[name]) { map[name].destroy(); }
  if (name === 'metricsChart') metricsChart = null;
  if (name === 'timingChart') timingChart = null;
  if (name === 'confidenceChart') confidenceChart = null;
  if (name === 'rocChart') rocChart = null;
}

function renderMetricsChart(s, models) {
  destroyChart('metricsChart');
  const labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'];
  const fields = ['accuracy', 'precision', 'recall', 'f1', 'auc'];
  metricsChart = new Chart(document.getElementById('metricsChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: models.map((m, i) => ({
        label: m.label,
        data: fields.map((f) => (s[m.statsKey] ? s[m.statsKey][f] : null)),
        backgroundColor: colorFor(m.key, i),
      })),
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, max: 1, ticks: { callback: (v) => (v * 100) + '%' } } },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

function renderTimingChart(s, models) {
  destroyChart('timingChart');
  timingChart = new Chart(document.getElementById('timingChart'), {
    type: 'bar',
    data: {
      labels: ['Avg. inference time (ms/sample)'],
      datasets: models.map((m, i) => ({
        label: m.label,
        data: [s[m.statsKey] ? s[m.statsKey].avg_infer_ms : null],
        backgroundColor: colorFor(m.key, i),
      })),
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: { x: { type: 'logarithmic', title: { display: true, text: 'ms (log scale)', color: MUTED } } },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

async function loadConfidence(key) {
  const res = await fetch(`/api/confidence/${key}`);
  const data = await res.json();
  destroyChart('confidenceChart');
  destroyChart('rocChart');
  const models = data.models || [];
  if (models.length === 0) return;
  renderRocChart(data, models);

  const binEdges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001];
  const binLabels = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
                     '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'];
  function histogram(rows) {
    if (!rows) return null;
    const counts = new Array(binLabels.length).fill(0);
    rows.forEach((r) => {
      for (let i = 0; i < binEdges.length - 1; i++) {
        if (r.proba >= binEdges[i] && r.proba < binEdges[i + 1]) { counts[i]++; break; }
      }
    });
    return counts.map((c) => (100 * c) / rows.length);
  }

  const chartDatasets = models.map((m, i) => ({
    label: m.label, data: histogram(data[m.key]), backgroundColor: colorFor(m.key, i),
  })).filter((ds) => ds.data);

  confidenceChart = new Chart(document.getElementById('confidenceChart'), {
    type: 'bar',
    data: { labels: binLabels, datasets: chartDatasets },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, title: { display: true, text: '% of test samples', color: MUTED } },
        x: { title: { display: true, text: 'P(positive)', color: MUTED } },
      },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

function computeRoc(rows) {
  if (!rows || !rows.length) return null;
  const P = rows.filter((r) => r.trueLabel === 1).length;
  const N = rows.length - P;
  if (P === 0 || N === 0) return null;
  const points = [];
  for (let t = 1.0; t >= -0.0001; t -= 0.02) {
    let tp = 0, fp = 0;
    rows.forEach((r) => {
      const predPos = r.proba >= t;
      if (predPos && r.trueLabel === 1) tp++;
      else if (predPos && r.trueLabel === 0) fp++;
    });
    points.push({ x: fp / N, y: tp / P });
  }
  return points;
}

function renderRocChart(data, models) {
  const chartDatasets = models.map((m, i) => {
    const roc = computeRoc(data[m.key]);
    if (!roc) return null;
    return {
      label: m.label, data: roc, borderColor: colorFor(m.key, i), backgroundColor: colorFor(m.key, i),
      pointRadius: 0, borderWidth: 2, tension: 0,
    };
  }).filter(Boolean);
  chartDatasets.push({
    label: 'Random guess', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: MUTED, borderDash: [4, 4], pointRadius: 0, borderWidth: 1,
  });

  rocChart = new Chart(document.getElementById('rocChart'), {
    type: 'line',
    data: { datasets: chartDatasets },
    options: {
      responsive: true,
      parsing: false,
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False positive rate', color: MUTED } },
        y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'True positive rate', color: MUTED } },
      },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

randomBtn.addEventListener('click', () => {
  const d = currentDataset();
  if (!d) return;
  sampleIndex.value = Math.floor(Math.random() * d.nSamples);
});

runBtn.addEventListener('click', runPrediction);

async function runPrediction() {
  const d = currentDataset();
  if (!d) return;
  const models = Array.from(modelList.querySelectorAll('input[type=checkbox]:checked')).map((c) => c.value);
  if (models.length === 0) { alert('Select at least one model.'); return; }

  runBtn.disabled = true; runBtn.textContent = '⏳ Running...';
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: d.key, index: sampleIndex.value, models }),
    });
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    alert('Error while running: ' + e.message);
  } finally {
    runBtn.disabled = false; runBtn.textContent = '▶ Run';
  }
}

function renderResult(data) {
  resultEmpty.classList.add('hidden');
  resultBody.classList.remove('hidden');
  sampleIndex.value = data.index;

  sampleTitle.textContent = `${data.friendly} — sample #${data.index} / ${data.nSamples - 1}`;
  const isTruePos = data.trueLabel === 1;
  trueLabelBadge.textContent = 'Ground truth: ' + (isTruePos ? data.posLabel : data.negLabel);

  const cols = Object.keys(data.features);
  featureTable.innerHTML =
    '<tr>' + cols.map((c) => `<th>${c}</th>`).join('') + '</tr>' +
    '<tr>' + cols.map((c) => `<td>${data.features[c]}</td>`).join('') + '</tr>';

  verdictCards.innerHTML = Object.entries(data.results).map(([modelKey, r]) => {
    const cls = r.pred === 1 ? 'pos' : 'neg';
    const resultTag = r.correct
      ? '<span class="correct">✓ correct</span>'
      : '<span class="wrong">✗ wrong</span>';
    const timing = r.avgInferMs != null
      ? ` · avg inference ${r.avgInferMs.toFixed(3)} ms/sample`
      : '';
    return `
      <div class="verdict-card ${cls}">
        <div class="model-name">${r.modelLabel}</div>
        <div class="verdict-text">${r.verdictLabel}</div>
        <div class="verdict-meta">
          confidence ${(r.confidence * 100).toFixed(1)}% · P(positive)=${r.probaClass1.toFixed(3)}
          &nbsp;·&nbsp; ${resultTag}${timing}
        </div>
      </div>`;
  }).join('');
}

loadDatasets();
