"""
Baseline "khong luong tu hoa": chay vai mo hinh classical ML chuan (Logistic
Regression, Random Forest, SVM) tren CUNG dataset/feature (qua dataset.py,
tuc la CUNG so chieu n_qubits sau PCA nhu VQC dang dung) de so sanh voi VQC
(quantum, script 05/06) -> tra loi cau hoi "luong tu hoa du lieu vao mach co
that su mang lai loi ich hay khong".

Metric tinh giong het circuit_backend.py.evaluate(): accuracy_score, f1_score
(zero_division=0), roc_auc_score tren xac suat lop 1 -> so sanh cong bang.

Output cung dinh dang cot voi comparison.csv cua script 05 (scheduler,
acc_end, f1_end, roc_auc_end, ...) de nap chung (pd.concat) va ve 1 bang/1
hinh "quantum (VQC) vs classical" duy nhat.

Chay:
    # 3 model classical mac dinh tren dataset heart, n_qubits=8 (khop VQC)
    python scripts/07_classical_baseline.py --dataset heart --n-qubits 8

    # nhieu dataset 1 luot + gop voi ket qua VQC da co san
    python scripts/07_classical_baseline.py --dataset heart,cancer,pima \
        --n-qubits 8 --vqc-comparison-csv outputs/comparison/pennylane/comparison.csv
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

MODELS = {
    "logreg": lambda seed: __import__(
        "sklearn.linear_model", fromlist=["LogisticRegression"]
    ).LogisticRegression(max_iter=2000, random_state=seed),
    "random_forest": lambda seed: __import__(
        "sklearn.ensemble", fromlist=["RandomForestClassifier"]
    ).RandomForestClassifier(n_estimators=200, random_state=seed),
    "svm": lambda seed: __import__(
        "sklearn.svm", fromlist=["SVC"]
    ).SVC(probability=True, random_state=seed),
}


def load_data(dataset, n_qubits, max_samples, seed):
    from dataset import load_binary_dataset
    d = load_binary_dataset(dataset, n_qubits=n_qubits, max_samples=max_samples, seed=seed)
    print(f"[data] {d['name']}: train={d['n_train']} val={d['n_val']} n_features={n_qubits}")
    return d


def evaluate(model, X_val, y_val):
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    try:
        auc = float(roc_auc_score(y_val, probs))
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds, zero_division=0)),
        "roc_auc": auc,
    }


def run_one(model_name, dataset, data, seed):
    model = MODELS[model_name](seed)
    t0 = time.perf_counter()
    model.fit(data["X_train"], data["y_train"])
    fit_time = time.perf_counter() - t0
    m = evaluate(model, data["X_val"], data["y_val"])
    return {
        "scheduler": f"classical_{model_name}",
        "backend": "classical",
        "dataset": dataset,
        "seed": seed,
        "epochs": None,
        "acc_start": float("nan"),
        "acc_end": round(m["accuracy"], 4),
        "acc_delta": float("nan"),
        "f1_end": round(m["f1"], 4),
        "roc_auc_end": round(m["roc_auc"], 4),
        "depth_start": 0,
        "depth_end": 0,
        "n_commits": 0,
        "n_rollbacks": 0,
        "fit_time_s": round(fit_time, 4),
    }


def plot(rows, vqc_rows, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    all_rows = list(vqc_rows) + list(rows)
    names = [r["scheduler"] for r in all_rows]
    acc = [r["acc_end"] for r in all_rows]
    colors = ["#5b8def" if r["backend"] == "classical" else "#e07a3f" for r in all_rows]

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 4.5))
    ax.bar(names, acc, color=colors)
    ax.set(title="VQC (luong tu hoa) vs Classical (khong luong tu hoa)",
          ylabel="accuracy (val)", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=30)
    for i, a in enumerate(acc):
        ax.text(i, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="heart",
                    help="1 hoac nhieu dataset cach nhau boi dau phay (vd 'heart,cancer,pima')")
    ap.add_argument("--n-qubits", type=int, default=8,
                    help="so feature sau PCA -- PHAI khop n_qubits dang dung cho VQC de so sanh cong bang")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--models", default="logreg,random_forest,svm")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--vqc-comparison-csv", default=None,
                    help="duong dan comparison.csv cua VQC (tu script 05/06) de gop chung bang/hinh")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out or ROOT / "outputs" / "comparison" / "classical")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in datasets:
        data = load_data(dataset, args.n_qubits, args.max_samples, args.seed)
        for model_name in model_names:
            print(f"[classical] fit {model_name} on {dataset}")
            rows.append(run_one(model_name, dataset, data, args.seed))

    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "comparison_classical.csv", index=False)

    cols = ["scheduler", "dataset", "acc_end", "f1_end", "roc_auc_end", "fit_time_s"]
    print(comp[cols].to_string(index=False))
    print(f"[classical] saved -> {out_dir / 'comparison_classical.csv'}")

    vqc_rows = []
    if args.vqc_comparison_csv and Path(args.vqc_comparison_csv).exists():
        vqc_df = pd.read_csv(args.vqc_comparison_csv)
        vqc_df["backend"] = vqc_df.get("backend", "pennylane")
        vqc_rows = vqc_df.to_dict("records")
        merged = pd.concat([vqc_df, comp], ignore_index=True, sort=False)
        merged.to_csv(out_dir / "comparison_quantum_vs_classical.csv", index=False)
        print(f"[classical] gop voi VQC -> {out_dir / 'comparison_quantum_vs_classical.csv'}")
        plot(rows, vqc_rows, out_dir / "comparison_quantum_vs_classical.png")
    else:
        print("[classical] chua truyen --vqc-comparison-csv -> chi luu ket qua classical "
              "(chay lai voi --vqc-comparison-csv outputs/comparison/<backend>/comparison.csv "
              "de gop bang + ve hinh quantum vs classical).")


if __name__ == "__main__":
    main()
