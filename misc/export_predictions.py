"""
export_predictions.py — Export per-sample VQC predictions for the demo.

RUN FROM PROJECT ROOT (needs torch + pennylane installed):
    python export_predictions.py                 # all datasets
    python export_predictions.py heart pima       # only the named ones

For each dataset it deploys a real 8-qubit (6 for AI4I) VQC with the RULE
scheduler on PennyLane, then dumps the trained circuit's per-sample prediction
on a HELD-OUT TEST split (TEST_FRAC of the data, carved out BEFORE train/val
and never touched during training):
    predictions_<dataset>.csv   with columns: f0..fk, true_label, pred, confidence, proba_class1
    predictions_meta.csv        with per-dataset deployed test-accuracy.

Why test, not val: the trainer's mutation engine (add/prune/reduce accept-or-
rollback) reads validation loss EVERY epoch to decide whether to keep a
mutation -> the val split indirectly shapes the final circuit, so numbers
shown from it are not "unseen data" in the strict sense. The test split never
enters training in any form (no gradient, no scheduler decision) -> what the
live-prediction demo panel shows is genuinely first-contact for the model.

The demo (streamlit) reads these to show REAL VQC predictions for known data.
Rule scheduler is used because it is deterministic and needs no agent checkpoint.
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT)); sys.path.append(str(ROOT / "scripts"))

from dataset import load_binary_dataset
from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend
from src.synthetic_bp.stage3.trainer import AdaptiveTrainer
from src.synthetic_bp.stage2.rule_based_scheduler import RuleBasedScheduler

# (load_name, n_qubits, friendly)  — chỉ giữ dataset đã có data thật
DATASETS = [
    ("predictive_maintenance", 8, "AI4I Predictive Maintenance"),
    ("cancer",                 8, "Breast Cancer Wisconsin"),   # BCW = sklearn load_breast_cancer
    ("german_credit",          8, "German Credit"),
    ("pima",                   8, "Pima Diabetes"),             # cần pima.csv trong data/
    ("heart",                  8, "Heart Disease (Cleveland)"),
    ("parkinsons",             8, "Parkinson's (Oxford)"),
]
SEED = 123; EPOCHS = 40; COST = "local"; TOPO = "linear"
VAL_FRAC = 0.3    # dùng lúc train (scheduler accept/rollback)
TEST_FRAC = 0.2   # tách riêng cho demo, KHÔNG đụng lúc train

# per-dataset sample cap — as large as each REAL source dataset allows, no
# synthetic padding. Hard ceilings: heart 274 rows (UCI Cleveland, balanced),
# cancer 569 (sklearn breast_cancer), parkinsons ~195 raw / 96 balanced.
MAXS_BY_DATASET = {
    "heart": 280,
    "cancer": 569,       # full sklearn breast_cancer (569 rows, no class balancing applied)
    "parkinsons": 280,
    "predictive_maintenance": None,   # None -> dataset.py's Cyclic Partitioned Undersampling path: keeps ALL 10,000 real rows, no cap. Requires batch_sampler passed to PennyLaneBackend below.
}
DEFAULT_MAXS = 280

only = set(sys.argv[1:])
if only:
    DATASETS = [d for d in DATASETS if d[0] in only]

meta = []
for name, nq, friendly in DATASETS:
    maxs = MAXS_BY_DATASET.get(name, DEFAULT_MAXS)
    print(f"\n=== {friendly} ({name}, {nq}q, max_samples={maxs}) ===")
    try:
        data = load_binary_dataset(name=name, n_qubits=nq, val_frac=VAL_FRAC,
                                   test_frac=TEST_FRAC, seed=SEED, max_samples=maxs)
    except Exception as e:
        print(f"  SKIP (load fail): {e}"); continue
    if data["n_test"] == 0:
        print(f"  SKIP (dataset quá nhỏ để tách test_frac={TEST_FRAC}): "
              f"n_train={data['n_train']} n_val={data['n_val']}"); continue
    try:
        b = PennyLaneBackend(data["X_train"], data["y_train"],
                             data["X_val"], data["y_val"],  # val: dùng lúc train để scheduler quyết định
                             n_qubits=nq, initial_depth=2, max_depth=8,
                             topology=TOPO, entangler_type="cnot",
                             cost_type=COST, seed=SEED,
                             batch_sampler=data.get("batch_sampler"))
        sched = RuleBasedScheduler(calibration=None)
        tr = AdaptiveTrainer(b, sched, n_qubits=nq, cost_type=COST,
                             topology=TOPO, epochs=EPOCHS)
        tr.run()                                   # trains b.theta in place
        qnode = b._make_qnode()
        Xt, yt = data["X_test"], data["y_test"]     # test: model CHƯA từng thấy dưới bất kỳ hình thức nào
        qnode(b.theta, Xt[0])                       # warm-up (JIT/tracing) — excluded from timing
        rows, correct, infer_times = [], 0, []
        for x, y in zip(Xt, yt):
            t0 = time.perf_counter()
            p = float((qnode(b.theta, x) + 1) / 2)  # proba of class 1
            infer_times.append(time.perf_counter() - t0)
            pred = int(p > 0.5); correct += int(pred == int(y))
            rec = {f"f{i}": round(float(v), 5) for i, v in enumerate(x)}
            rec.update(true_label=int(y), pred=pred,
                       confidence=round(max(p, 1 - p), 4), proba_class1=round(p, 4))
            rows.append(rec)
        out = f"predictions_{name}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        acc = correct / max(len(yt), 1)
        avg_ms = 1000 * float(np.mean(infer_times))
        total_ms = 1000 * float(np.sum(infer_times))
        print(f"  -> {out}  ({len(rows)} samples, TEST acc {acc:.3f}, final depth {b.depth}, "
              f"avg inference {avg_ms:.2f} ms/sample, total {total_ms:.1f} ms)")
        meta.append({"dataset": name, "friendly": friendly, "n_qubits": nq,
                     "test_accuracy": round(acc, 4), "n_test": len(rows),
                     "final_depth": int(b.depth),
                     "avg_infer_ms": round(avg_ms, 4), "total_infer_ms": round(total_ms, 2)})

        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        yp = [r["pred"] for r in rows]; yt_list = [r["true_label"] for r in rows]
        yprob = [r["proba_class1"] for r in rows]
        stats_path = Path(f"stats_{name}.json")
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        stats["dataset"] = name; stats["friendly"] = friendly; stats["n_qubits"] = nq
        stats["n_train"] = data["n_train"]; stats["n_val"] = data["n_val"]; stats["n_test"] = len(rows)
        stats["quantum"] = {
            "label": "AL-QNN (Quantum)",
            "accuracy": round(acc, 4),
            "precision": round(precision_score(yt_list, yp, zero_division=0), 4),
            "recall": round(recall_score(yt_list, yp, zero_division=0), 4),
            "f1": round(f1_score(yt_list, yp, zero_division=0), 4),
            "auc": round(roc_auc_score(yt_list, yprob), 4) if len(set(yt_list)) > 1 else None,
            "final_depth": int(b.depth),
            "n_qubits": nq,
            "avg_infer_ms": round(avg_ms, 4),
            "total_infer_ms": round(total_ms, 2),
        }
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {stats_path.name} updated (quantum stats)")
    except Exception as e:
        print(f"  SKIP (deploy fail): {type(e).__name__}: {e}")

if meta:
    new_df = pd.DataFrame(meta)
    meta_path = Path("predictions_meta.csv")
    if meta_path.exists():
        old_df = pd.read_csv(meta_path)
        old_df = old_df[~old_df["dataset"].isin(new_df["dataset"])]
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(meta_path, index=False)
    print("\nSaved predictions_meta.csv")
    print("Copy all predictions_*.csv")
else:
    print("\nNo predictions exported. Check torch/pennylane install and dataset availability.")