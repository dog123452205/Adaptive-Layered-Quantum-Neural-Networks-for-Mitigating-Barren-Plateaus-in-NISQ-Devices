"""
export_classical_baseline.py — Classical ("chưa lượng tử hóa") baseline for
comparison against the AL-QNN quantum classifier, for the LIT demo.

Trains a plain scikit-learn LogisticRegression on the SAME train split (same
seed/val_frac/test_frac/max_samples as export_predictions.py) and evaluates
it on the SAME held-out TEST split — a genuinely fair, apples-to-apples
comparison against the quantum model, not a re-labelled version of it.

RUN FROM PROJECT ROOT:
    python export_classical_baseline.py heart

Produces predictions_<dataset>_classical.csv with the SAME raw, human-
readable feature columns as predictions_<dataset>_named.csv (or f0..fk for
datasets without named features) plus true_label/pred/confidence/
proba_class1 — same shape as the quantum predictions file, so lit_demo.py
can register both as separate MODELS against the same LIT Dataset.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT)); sys.path.append(str(ROOT / "scripts"))

from dataset import load_binary_dataset

# must match export_predictions.py exactly, or the TEST split won't line up
SEED = 123; VAL_FRAC = 0.3; TEST_FRAC = 0.2
MAXS_BY_DATASET = {"heart": 280, "cancer": 569, "parkinsons": 280, "predictive_maintenance": 3000}
DEFAULT_MAXS = 280

# dataset key -> (load_name, n_qubits, friendly, named-features csv to borrow raw columns from, or None)
DATASETS = [
    ("heart", "heart", 8, "Heart Disease (Cleveland)", "predictions_heart_named.csv"),
    ("predictive_maintenance", "predictive_maintenance", 6, "AI4I Predictive Maintenance", None),
    ("cancer", "cancer", 8, "Breast Cancer Wisconsin", None),
    ("parkinsons", "parkinsons", 8, "Parkinson's (Oxford)", None),
]

only = set(sys.argv[1:])
if only:
    DATASETS = [d for d in DATASETS if d[0] in only]

for key, load_name, nq, friendly, named_csv in DATASETS:
    maxs = MAXS_BY_DATASET.get(key, DEFAULT_MAXS)
    print(f"\n=== {friendly} ({key}) — classical LogisticRegression baseline (max_samples={maxs}) ===")
    try:
        data = load_binary_dataset(name=load_name, n_qubits=nq, val_frac=VAL_FRAC,
                                   test_frac=TEST_FRAC, seed=SEED, max_samples=maxs)
    except Exception as e:
        print(f"  SKIP (load fail): {e}"); continue
    if data["n_test"] == 0:
        print(f"  SKIP (no test split)"); continue

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(data["X_train"], data["y_train"])
    Xt, yt = data["X_test"], data["y_test"]

    clf.predict_proba(Xt[:1])  # warm-up, excluded from timing
    infer_times = []
    proba = np.empty(len(Xt))
    for i in range(len(Xt)):
        t0 = time.perf_counter()
        proba[i] = clf.predict_proba(Xt[i:i + 1])[:, 1][0]
        infer_times.append(time.perf_counter() - t0)

    pred = (proba > 0.5).astype(int)
    acc = float((pred == yt).mean())
    avg_ms = 1000 * float(np.mean(infer_times))
    total_ms = 1000 * float(np.sum(infer_times))
    print(f"  TEST acc = {acc:.3f}  (n={len(yt)}), avg inference {avg_ms:.4f} ms/sample, "
          f"total {total_ms:.2f} ms")

    if named_csv and (ROOT / named_csv).exists():
        # borrow the raw, human-readable feature columns already verified
        # row-aligned to this exact TEST split (see predictions_heart_named.csv)
        named = pd.read_csv(ROOT / named_csv)
        feat_cols = [c for c in named.columns if c not in
                    ("true_label", "pred", "confidence", "proba_class1")]
        out = named[feat_cols].copy()
    else:
        out = pd.DataFrame({f"f{i}": np.round(Xt[:, i], 5) for i in range(Xt.shape[1])})

    out["true_label"] = yt.astype(int)
    out["pred"] = pred
    out["confidence"] = np.round(np.maximum(proba, 1 - proba), 4)
    out["proba_class1"] = np.round(proba, 4)

    out_path = ROOT / f"predictions_{key}_classical.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> {out_path.name}  ({len(out)} rows)")

    stats_path = ROOT / f"stats_{key}.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats["dataset"] = key; stats["friendly"] = friendly
    stats["classical"] = {
        "label": "Logistic Regression (Classical)",
        "accuracy": round(acc, 4),
        "precision": round(precision_score(yt, pred, zero_division=0), 4),
        "recall": round(recall_score(yt, pred, zero_division=0), 4),
        "f1": round(f1_score(yt, pred, zero_division=0), 4),
        "auc": round(roc_auc_score(yt, proba), 4) if len(set(yt)) > 1 else None,
        "n_features": int(Xt.shape[1]),
        "avg_infer_ms": round(avg_ms, 4),
        "total_infer_ms": round(total_ms, 2),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {stats_path.name} updated (classical stats)")
