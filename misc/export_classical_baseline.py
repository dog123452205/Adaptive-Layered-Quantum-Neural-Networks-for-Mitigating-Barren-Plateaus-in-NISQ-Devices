"""
export_classical_baseline.py — Classical ("chưa lượng tử hóa") baselines for
comparison against the AL-QNN quantum classifier: Logistic Regression
(linear) and Random Forest (non-linear ensemble).

Trains each on the SAME train split (same seed/val_frac/test_frac/max_samples
as export_predictions.py) and evaluates on the SAME held-out TEST split — a
genuinely fair, apples-to-apples comparison against the quantum model, not a
re-labelled version of it.

RUN FROM PROJECT ROOT:
    python export_classical_baseline.py heart

Produces predictions_<dataset>_classical.csv / predictions_<dataset>_rf.csv
with the SAME raw, human-readable feature columns as predictions_<dataset>_named.csv
(or f0..fk for datasets without named features) plus true_label/pred/
confidence/proba_class1 — same shape as the quantum predictions file.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT)); sys.path.append(str(ROOT / "scripts"))

from dataset import load_binary_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_heart_data_stable(seed, val_frac, test_frac):
    """
    dataset.py's shared 'heart' -> 'heart_disease' path now pulls from
    fetch_openml('heart') (teammate rewrite for the AI4I cyclic-sampler
    work) instead of the local data/heart.csv used to produce the existing
    predictions_heart*.csv — that swap silently changed the row count/split
    (54 vs 55 test rows), breaking alignment. Bypass it here and replicate
    the ORIGINAL local-CSV split directly (verified byte-for-byte against
    predictions_heart.csv's true_label column) so classical/RF stay aligned
    with the already-published Heart demo data regardless of future changes
    to the shared dataset.py.
    """
    df = pd.read_csv(ROOT.parent / "data" / "heart.csv")
    X = df[[f"f{i}" for i in range(8)]].to_numpy(float)
    y = df["Target"].to_numpy(int)
    X_fit, X_test, y_fit, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y)
    rel_val = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_fit, y_fit, test_size=rel_val, random_state=seed, stratify=y_fit)
    sc = MinMaxScaler(feature_range=(0.0, np.pi)).fit(X_train)
    return {
        "X_train": sc.transform(X_train), "y_train": y_train.astype(float),
        "X_val": sc.transform(X_val), "y_val": y_val.astype(float),
        "X_test": sc.transform(X_test), "y_test": y_test.astype(float),
        "n_test": len(y_test),
    }

# must match export_predictions.py exactly, or the TEST split won't line up
SEED = 123; VAL_FRAC = 0.3; TEST_FRAC = 0.2
MAXS_BY_DATASET = {"heart": 280, "cancer": 569, "parkinsons": 280, "predictive_maintenance": None}
DEFAULT_MAXS = 280

# dataset key -> (load_name, n_qubits, friendly, named-features csv to borrow raw columns from, or None)
DATASETS = [
    ("heart", "heart", 8, "Heart Disease (Cleveland)", "predictions_heart_named.csv"),
    ("predictive_maintenance", "predictive_maintenance", 8, "AI4I Predictive Maintenance", "predictions_predictive_maintenance_named.csv"),
    ("cancer", "cancer", 8, "Breast Cancer Wisconsin", None),
    ("parkinsons", "parkinsons", 8, "Parkinson's (Oxford)", None),
]

# stats-key -> (label, sklearn estimator factory)
MODELS = {
    "classical": ("Logistic Regression (Classical)",
                 lambda: LogisticRegression(max_iter=1000, random_state=SEED)),
    "rf": ("Random Forest (Classical)",
          lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
}

only = set(sys.argv[1:])
if only:
    DATASETS = [d for d in DATASETS if d[0] in only]


def run_model(stats_key, label, make_clf, data, key, friendly, named_csv, Xt, yt):
    if "batch_sampler" in data:
        # AI4I cyclic path: data["X_train"] is just the first 340-row chunk —
        # fit on the FULL natural-ratio train pool instead, with
        # class_weight='balanced' to compensate (sklearn fits instantly
        # regardless of imbalance, no cyclic batching needed here).
        clf = make_clf()
        if hasattr(clf, "class_weight"):
            clf.set_params(class_weight="balanced")
        clf.fit(data["X_train_full"], data["y_train_full"])
    else:
        clf = make_clf()
        clf.fit(data["X_train"], data["y_train"])

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
    print(f"  [{stats_key}] TEST acc = {acc:.3f}  (n={len(yt)}), avg inference {avg_ms:.4f} ms/sample, "
          f"total {total_ms:.2f} ms")

    if named_csv and (ROOT / named_csv).exists():
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

    out_path = ROOT / f"predictions_{key}_{stats_key}.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> {out_path.name}  ({len(out)} rows)")

    stats_path = ROOT / f"stats_{key}.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats["dataset"] = key; stats["friendly"] = friendly
    stats[stats_key] = {
        "label": label,
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
    print(f"  -> {stats_path.name} updated ({stats_key} stats)")


for key, load_name, nq, friendly, named_csv in DATASETS:
    maxs = MAXS_BY_DATASET.get(key, DEFAULT_MAXS)
    print(f"\n=== {friendly} ({key}) — classical baselines (max_samples={maxs}) ===")
    try:
        if key == "heart":
            data = load_heart_data_stable(SEED, VAL_FRAC, TEST_FRAC)
        else:
            data = load_binary_dataset(name=load_name, n_qubits=nq, val_frac=VAL_FRAC,
                                       test_frac=TEST_FRAC, seed=SEED, max_samples=maxs)
    except Exception as e:
        print(f"  SKIP (load fail): {e}"); continue
    if data["n_test"] == 0:
        print(f"  SKIP (no test split)"); continue

    Xt, yt = data["X_test"], data["y_test"]
    for stats_key, (label, make_clf) in MODELS.items():
        run_model(stats_key, label, make_clf, data, key, friendly, named_csv, Xt, yt)
