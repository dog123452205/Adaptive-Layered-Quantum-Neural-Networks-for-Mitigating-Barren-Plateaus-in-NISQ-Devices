// server.js — Express backend for the AL-QNN quantum-vs-classical demo UI.
//
// Reads the SAME CSV/JSON files produced by export_predictions.py,
// build_heart_named.py, and export_classical_baseline.py in the project
// root (one level up) — no Python needed at runtime, this just replays the
// already-computed, verified predictions and stats.
//
// RUN:  cd web_demo && npm install && npm start   -> http://localhost:4000

const express = require('express');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const PORT = 4000;

// dataset key -> config. csvQuantum/csvClassical are relative to ROOT.
const DATASETS = {
  heart: {
    friendly: 'Heart Disease (Cleveland)',
    pos: 'disease', neg: 'healthy',
    posLabel: 'DISEASE PRESENT', negLabel: 'NO DISEASE',
    csvQuantum: 'predictions_heart_named.csv',
    csvClassical: 'predictions_heart_classical.csv',
    statsFile: 'stats_heart.json',
  },
  cancer: {
    friendly: 'Breast Cancer Wisconsin',
    pos: 'benign', neg: 'malignant',
    posLabel: 'BENIGN', negLabel: 'MALIGNANT',
    csvQuantum: 'predictions_cancer.csv',
    csvClassical: 'predictions_cancer_classical.csv',
    statsFile: 'stats_cancer.json',
  },
  parkinsons: {
    friendly: "Parkinson's (Oxford)",
    pos: 'parkinsons', neg: 'healthy',
    posLabel: 'PARKINSON\'S DETECTED', negLabel: 'HEALTHY',
    csvQuantum: 'predictions_parkinsons.csv',
    csvClassical: 'predictions_parkinsons_classical.csv',
    statsFile: 'stats_parkinsons.json',
  },
  predictive_maintenance: {
    friendly: 'AI4I Predictive Maintenance',
    pos: 'failure', neg: 'ok',
    posLabel: 'FAILURE PREDICTED', negLabel: 'NORMAL OPERATION',
    csvQuantum: 'predictions_predictive_maintenance.csv',
    csvClassical: 'predictions_predictive_maintenance_classical.csv',
    statsFile: 'stats_predictive_maintenance.json',
  },
};

function parseCsv(filePath) {
  const text = fs.readFileSync(filePath, 'utf-8').trim();
  const lines = text.split(/\r?\n/);
  const header = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const cells = line.split(',');
    const row = {};
    header.forEach((h, i) => {
      const v = cells[i];
      row[h] = v === undefined || v === '' ? null : Number(v);
    });
    return row;
  });
}

function loadDataset(key) {
  const cfg = DATASETS[key];
  if (!cfg) return null;
  const quantumPath = path.join(ROOT, cfg.csvQuantum);
  const classicalPath = path.join(ROOT, cfg.csvClassical);
  const statsPath = path.join(ROOT, cfg.statsFile);
  const out = { key, ...cfg, quantum: null, classical: null, stats: null };
  if (fs.existsSync(quantumPath)) out.quantum = parseCsv(quantumPath);
  if (fs.existsSync(classicalPath)) out.classical = parseCsv(classicalPath);
  if (fs.existsSync(statsPath)) out.stats = JSON.parse(fs.readFileSync(statsPath, 'utf-8'));
  return out;
}

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/datasets', (req, res) => {
  const list = Object.keys(DATASETS).map((key) => {
    const d = loadDataset(key);
    return {
      key,
      friendly: d.friendly,
      nSamples: d.quantum ? d.quantum.length : 0,
      hasQuantum: !!d.quantum,
      hasClassical: !!d.classical,
      hasStats: !!d.stats,
    };
  });
  res.json(list);
});

app.get('/api/stats/:dataset', (req, res) => {
  const d = loadDataset(req.params.dataset);
  if (!d) return res.status(404).json({ error: 'unknown dataset' });
  res.json({ key: d.key, friendly: d.friendly, stats: d.stats });
});

app.get('/api/confidence/:dataset', (req, res) => {
  const d = loadDataset(req.params.dataset);
  if (!d) return res.status(404).json({ error: 'unknown dataset' });
  const pick = (rows) => rows ? rows.map((r) => ({
    proba: r.proba_class1, correct: r.pred === r.true_label,
  })) : null;
  res.json({
    key: d.key, friendly: d.friendly,
    quantum: pick(d.quantum), classical: pick(d.classical),
  });
});

app.post('/api/run', (req, res) => {
  const { dataset, index, models } = req.body || {};
  const d = loadDataset(dataset);
  if (!d) return res.status(404).json({ error: 'unknown dataset' });
  const n = d.quantum ? d.quantum.length : 0;
  if (n === 0) return res.status(404).json({ error: 'no data for this dataset' });

  const idx = index === undefined || index === null || index === ''
    ? Math.floor(Math.random() * n)
    : Math.max(0, Math.min(n - 1, Number(index)));

  const wantModels = Array.isArray(models) && models.length ? models : ['quantum', 'classical'];
  const featureCols = Object.keys(d.quantum[idx]).filter(
    (c) => !['true_label', 'pred', 'confidence', 'proba_class1'].includes(c)
  );

  const results = {};
  for (const m of wantModels) {
    const rows = m === 'quantum' ? d.quantum : d.classical;
    if (!rows) continue;
    const row = rows[idx];
    const isPos = row.pred === 1;
    results[m] = {
      pred: row.pred,
      trueLabel: row.true_label,
      correct: row.pred === row.true_label,
      confidence: row.confidence,
      probaClass1: row.proba_class1,
      verdictLabel: isPos ? d.posLabel : d.negLabel,
      modelLabel: d.stats && d.stats[m] ? d.stats[m].label : m,
      avgInferMs: d.stats && d.stats[m] ? d.stats[m].avg_infer_ms : null,
    };
  }

  const features = {};
  featureCols.forEach((c) => { features[c] = d.quantum[idx][c]; });

  res.json({
    dataset: d.key, friendly: d.friendly, index: idx, nSamples: n,
    features, trueLabel: d.quantum[idx].true_label,
    posLabel: d.posLabel, negLabel: d.negLabel,
    results,
  });
});

app.listen(PORT, '127.0.0.1', () => {
  console.log(`AL-QNN demo UI running at http://localhost:${PORT}`);
});
