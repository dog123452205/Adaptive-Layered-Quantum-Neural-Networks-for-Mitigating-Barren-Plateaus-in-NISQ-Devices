// server.js — Express backend for the AL-QNN quantum-vs-classical demo UI.
//
// Reads the SAME CSV/JSON files produced by export_predictions.py,
// export_scheduler_predictions.py, build_*_named.py, and
// export_classical_baseline.py in ../misc — no Python needed at runtime,
// this just replays the already-computed, verified predictions and stats.
//
// Each dataset declares an ordered list of MODELS (scheduler variants +
// classical baseline). Demo scope is Heart + AI4I only, both offering
// Rule / Greedy / PPO / DQN schedulers (all trained/evaluated on the
// identical held-out TEST split) plus the classical baseline.
//
// RUN:  cd web_demo && npm install && npm start   -> http://localhost:4000

const express = require('express');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', 'misc');
const PORT = 4000;

const MODEL_LABELS = {
  rule: 'AL-QNN — Rule Scheduler',
  greedy: 'AL-QNN — Greedy Scheduler',
  ppo: 'AL-QNN — PPO Scheduler',
  dqn: 'AL-QNN — DQN Scheduler',
  classical: 'Logistic Regression (Classical)',
  rf: 'Random Forest (Classical)',
};

// dataset key -> config. Each entry in `models` is { key, statsKey, csv }
// (csv path relative to ROOT). statsKey is the field name inside
// stats_<dataset>.json holding that model's metrics.
const DATASETS = {
  heart: {
    friendly: 'Heart Disease (Cleveland)',
    pos: 'disease', neg: 'healthy',
    posLabel: 'DISEASE PRESENT', negLabel: 'NO DISEASE',
    // real ceiling after class-balancing UCI Cleveland (274 rows) — not a
    // cap that's actually cutting anything off
    totalSamples: 274,
    statsFile: 'stats_heart.json',
    models: [
      { key: 'rule', statsKey: 'quantum', csv: 'predictions_heart_named.csv' },
      { key: 'greedy', statsKey: 'greedy', csv: 'predictions_heart_greedy.csv' },
      { key: 'ppo', statsKey: 'ppo', csv: 'predictions_heart_ppo.csv' },
      { key: 'dqn', statsKey: 'dqn', csv: 'predictions_heart_dqn.csv' },
      { key: 'classical', statsKey: 'classical', csv: 'predictions_heart_classical.csv' },
      { key: 'rf', statsKey: 'rf', csv: 'predictions_heart_rf.csv' },
    ],
  },
  // Cancer and Parkinsons were dropped from the demo scope — only Heart and
  // AI4I get shown (both have the full Rule/Greedy/PPO/DQN comparison).
  // Their export scripts/data files are left untouched in ../misc in case
  // they're needed again later.
  predictive_maintenance: {
    friendly: 'AI4I Predictive Maintenance',
    pos: 'failure', neg: 'ok',
    posLabel: 'FAILURE PREDICTED', negLabel: 'NORMAL OPERATION',
    // full AI4I 2020 dataset, no cap — Cyclic Partitioned Undersampling
    // keeps every real row (see BalancedBatchSampler in scripts/dataset.py)
    totalSamples: 10000,
    statsFile: 'stats_predictive_maintenance.json',
    models: [
      { key: 'rule', statsKey: 'quantum', csv: 'predictions_predictive_maintenance_named.csv' },
      { key: 'greedy', statsKey: 'greedy', csv: 'predictions_predictive_maintenance_greedy.csv' },
      { key: 'ppo', statsKey: 'ppo', csv: 'predictions_predictive_maintenance_ppo.csv' },
      { key: 'dqn', statsKey: 'dqn', csv: 'predictions_predictive_maintenance_dqn.csv' },
      { key: 'classical', statsKey: 'classical', csv: 'predictions_predictive_maintenance_classical.csv' },
      { key: 'rf', statsKey: 'rf', csv: 'predictions_predictive_maintenance_rf.csv' },
    ],
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

// Loads a dataset config + whichever model CSVs currently exist on disk.
// out.models: only the configured models whose CSV file is present.
function loadDataset(key) {
  const cfg = DATASETS[key];
  if (!cfg) return null;
  const statsPath = path.join(ROOT, cfg.statsFile);
  const stats = fs.existsSync(statsPath) ? JSON.parse(fs.readFileSync(statsPath, 'utf-8')) : null;

  const models = [];
  const rowsByKey = {};
  for (const m of cfg.models) {
    const csvPath = path.join(ROOT, m.csv);
    if (!fs.existsSync(csvPath)) continue;
    const rows = parseCsv(csvPath);
    rowsByKey[m.key] = rows;
    models.push({
      key: m.key,
      statsKey: m.statsKey,
      label: (stats && stats[m.statsKey] && stats[m.statsKey].label) || MODEL_LABELS[m.key] || m.key,
      nSamples: rows.length,
    });
  }
  const nSamples = models.length ? Math.max(...models.map((m) => m.nSamples)) : 0;
  return { key, ...cfg, stats, models, rowsByKey, nSamples };
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
      nSamples: d.nSamples,
      nTotal: d.totalSamples || null,
      models: d.models.map(({ key, label }) => ({ key, label })),
    };
  });
  res.json(list);
});

app.get('/api/stats/:dataset', (req, res) => {
  const d = loadDataset(req.params.dataset);
  if (!d) return res.status(404).json({ error: 'unknown dataset' });
  res.json({
    key: d.key, friendly: d.friendly, stats: d.stats,
    models: d.models.map(({ key, statsKey, label }) => ({ key, statsKey, label })),
  });
});

app.get('/api/confidence/:dataset', (req, res) => {
  const d = loadDataset(req.params.dataset);
  if (!d) return res.status(404).json({ error: 'unknown dataset' });
  const out = { key: d.key, friendly: d.friendly, models: [] };
  for (const m of d.models) {
    const rows = d.rowsByKey[m.key];
    out[m.key] = rows.map((r) => ({
      proba: r.proba_class1, correct: r.pred === r.true_label, trueLabel: r.true_label,
    }));
    out.models.push({ key: m.key, label: m.label });
  }
  res.json(out);
});

app.post('/api/run', (req, res) => {
  const { dataset, index, models } = req.body || {};
  const d = loadDataset(dataset);
  if (!d) return res.status(404).json({ error: 'unknown dataset' });
  const n = d.nSamples;
  if (n === 0) return res.status(404).json({ error: 'no data for this dataset' });

  const idx = index === undefined || index === null || index === ''
    ? Math.floor(Math.random() * n)
    : Math.max(0, Math.min(n - 1, Number(index)));

  const availableKeys = d.models.map((m) => m.key);
  const wantModels = Array.isArray(models) && models.length
    ? models.filter((m) => availableKeys.includes(m))
    : availableKeys;

  // any model's row carries the same X_test sample -> borrow feature values
  // from whichever model is loaded first (prefer 'rule' if present).
  const featureSourceKey = d.rowsByKey.rule ? 'rule' : d.models[0] && d.models[0].key;
  const featureRow = featureSourceKey ? d.rowsByKey[featureSourceKey][idx] : null;
  const featureCols = featureRow
    ? Object.keys(featureRow).filter((c) => !['true_label', 'pred', 'confidence', 'proba_class1'].includes(c))
    : [];

  const results = {};
  for (const m of wantModels) {
    const rows = d.rowsByKey[m];
    if (!rows) continue;
    const row = rows[idx];
    const isPos = row.pred === 1;
    const meta = d.models.find((x) => x.key === m);
    results[m] = {
      pred: row.pred,
      trueLabel: row.true_label,
      correct: row.pred === row.true_label,
      confidence: row.confidence,
      probaClass1: row.proba_class1,
      verdictLabel: isPos ? d.posLabel : d.negLabel,
      modelLabel: meta ? meta.label : m,
      avgInferMs: d.stats && d.stats[meta.statsKey] ? d.stats[meta.statsKey].avg_infer_ms : null,
    };
  }

  const features = {};
  featureCols.forEach((c) => { features[c] = featureRow[c]; });

  res.json({
    dataset: d.key, friendly: d.friendly, index: idx, nSamples: n,
    features, trueLabel: featureRow ? featureRow.true_label : null,
    posLabel: d.posLabel, negLabel: d.negLabel,
    results,
  });
});

app.listen(PORT, '127.0.0.1', () => {
  console.log(`AL-QNN demo UI running at http://localhost:${PORT}`);
});
