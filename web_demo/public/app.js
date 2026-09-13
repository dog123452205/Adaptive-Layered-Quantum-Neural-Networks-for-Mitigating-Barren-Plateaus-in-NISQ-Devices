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
const statsBody = document.getElementById('statsBody');

let datasets = [];
const CYAN = '#34e0e0', BLUE = '#5b8cff', MUTED = '#8894b8';
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
    (d) => `<option value="${d.key}">${d.friendly} (${d.nSamples} samples)</option>`
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
  sampleRange.textContent = `Valid range: 0 – ${d.nSamples - 1} (leave "Sample #" blank to pick randomly)`;
  sampleIndex.max = d.nSamples - 1;
  resultEmpty.classList.remove('hidden');
  resultBody.classList.add('hidden');
  await loadStats(d.key);
  await loadConfidence(d.key);
}

async function loadStats(key) {
  const res = await fetch(`/api/stats/${key}`);
  const data = await res.json();
  const s = data.stats;
  if (!s || !s.quantum || !s.classical) {
    statsEmpty.classList.remove('hidden');
    statsEmpty.textContent = 'Not enough statistics available for this task yet (missing stats_*.json).';
    statsTable.classList.add('hidden');
    destroyChart('metricsChart'); destroyChart('timingChart');
    return;
  }
  statsEmpty.classList.add('hidden');
  statsTable.classList.remove('hidden');

  const rows = [
    ['Accuracy', s.quantum.accuracy, s.classical.accuracy, 'pct'],
    ['Precision', s.quantum.precision, s.classical.precision, 'pct'],
    ['Recall', s.quantum.recall, s.classical.recall, 'pct'],
    ['F1', s.quantum.f1, s.classical.f1, 'pct'],
    ['AUC', s.quantum.auc, s.classical.auc, 'pct'],
    ['Avg. inference time (ms/sample)', s.quantum.avg_infer_ms, s.classical.avg_infer_ms, 'ms_lower_better'],
    ['Total time on test set (ms)', s.quantum.total_infer_ms, s.classical.total_infer_ms, 'ms_lower_better'],
    ['Qubits / features', s.quantum.n_qubits, s.classical.n_features, 'raw'],
  ];

  statsBody.innerHTML = rows.map(([label, qv, cv, kind]) => {
    let qCell = qv, cCell = cv, qClass = '', cClass = '';
    if (kind === 'pct') {
      qCell = qv == null ? '—' : (qv * 100).toFixed(1) + '%';
      cCell = cv == null ? '—' : (cv * 100).toFixed(1) + '%';
      if (qv != null && cv != null) { if (qv > cv) qClass = 'better'; else if (cv > qv) cClass = 'better'; }
    } else if (kind === 'ms_lower_better') {
      qCell = qv == null ? '—' : qv.toFixed(3);
      cCell = cv == null ? '—' : cv.toFixed(3);
      if (qv != null && cv != null) { if (qv < cv) qClass = 'better'; else if (cv < qv) cClass = 'better'; }
    }
    return `<tr><td>${label}</td><td class="${qClass}">${qCell}</td><td class="${cClass}">${cCell}</td></tr>`;
  }).join('');

  renderMetricsChart(s);
  renderTimingChart(s);
}

function destroyChart(name) {
  const map = { metricsChart, timingChart, confidenceChart, rocChart };
  if (map[name]) { map[name].destroy(); }
  if (name === 'metricsChart') metricsChart = null;
  if (name === 'timingChart') timingChart = null;
  if (name === 'confidenceChart') confidenceChart = null;
  if (name === 'rocChart') rocChart = null;
}

function renderMetricsChart(s) {
  destroyChart('metricsChart');
  const labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'];
  const qVals = [s.quantum.accuracy, s.quantum.precision, s.quantum.recall, s.quantum.f1, s.quantum.auc];
  const cVals = [s.classical.accuracy, s.classical.precision, s.classical.recall, s.classical.f1, s.classical.auc];
  metricsChart = new Chart(document.getElementById('metricsChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'AL-QNN (Quantum)', data: qVals, backgroundColor: CYAN },
        { label: 'Logistic Regression (Classical)', data: cVals, backgroundColor: BLUE },
      ],
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, max: 1, ticks: { callback: (v) => (v * 100) + '%' } } },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

function renderTimingChart(s) {
  destroyChart('timingChart');
  timingChart = new Chart(document.getElementById('timingChart'), {
    type: 'bar',
    data: {
      labels: ['Avg. inference time (ms/sample)'],
      datasets: [
        { label: 'AL-QNN (Quantum)', data: [s.quantum.avg_infer_ms], backgroundColor: CYAN },
        { label: 'Logistic Regression (Classical)', data: [s.classical.avg_infer_ms], backgroundColor: BLUE },
      ],
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
  if (!data.quantum && !data.classical) return;
  renderRocChart(data);

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

  const qHist = histogram(data.quantum);
  const cHist = histogram(data.classical);
  const chartDatasets = [];
  if (qHist) chartDatasets.push({ label: 'AL-QNN (Quantum)', data: qHist, backgroundColor: CYAN });
  if (cHist) chartDatasets.push({ label: 'Logistic Regression (Classical)', data: cHist, backgroundColor: BLUE });

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

function renderRocChart(data) {
  const qRoc = computeRoc(data.quantum);
  const cRoc = computeRoc(data.classical);
  const chartDatasets = [];
  if (qRoc) chartDatasets.push({
    label: 'AL-QNN (Quantum)', data: qRoc, borderColor: CYAN, backgroundColor: CYAN,
    pointRadius: 0, borderWidth: 2, tension: 0,
  });
  if (cRoc) chartDatasets.push({
    label: 'Logistic Regression (Classical)', data: cRoc, borderColor: BLUE, backgroundColor: BLUE,
    pointRadius: 0, borderWidth: 2, tension: 0,
  });
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
