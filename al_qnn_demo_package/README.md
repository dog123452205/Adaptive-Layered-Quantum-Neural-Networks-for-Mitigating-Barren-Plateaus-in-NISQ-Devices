# AL-QNN Demo — Quantum vs Classical

## Requirements
- [Node.js](https://nodejs.org) 18+ (comes with `npm`)
- No Python needed — this demo only replays already-computed prediction
  results (CSV/JSON files included in this package).

## How to run

```bash
cd web_demo
npm install
node server.js
```

Then open **http://localhost:4000** in a browser.

## What's included

- `web_demo/` — Express server (`server.js`) + frontend (`public/`)
- `predictions_*.csv` — real per-sample predictions for 4 datasets
  (Heart Disease, Breast Cancer, Parkinson's, AI4I Predictive Maintenance),
  each with both an **AL-QNN (quantum)** and a **Logistic Regression
  (classical)** model, evaluated on the same held-out test split.
- `stats_*.json` — accuracy/precision/recall/F1/AUC and measured inference
  time (ms/sample) per dataset per model.

## Folder structure (do not rename/move files)

```
al_qnn_demo_package/
├── predictions_*.csv       <- must stay next to web_demo/, NOT inside it
├── stats_*.json
└── web_demo/
    ├── server.js
    ├── package.json
    └── public/
        ├── index.html
        ├── style.css
        └── app.js
```

`server.js` reads the CSV/JSON files from the folder **one level above**
`web_demo/` — keep this exact layout when you unzip/copy the package.

## Notes
- Port 4000 is hardcoded in `server.js` (`const PORT = 4000`) — change it
  there if it's already in use on your machine.
- All prediction data is precomputed (a "replay", not a live model) —
  there's no training or quantum simulation happening at runtime, so it
  starts instantly.
