"""
Baseline "fix depth k" (khong doi): chay AdaptiveTrainer voi FixedScheduler
(khong bao gio mutate) tai nhieu muc depth k co dinh, de so sanh voi
rule/greedy/ppo/dqn trong outputs/comparison/<backend>/comparison.csv.

Chay:
    # sweep may depth mac dinh (2,4,6,8,10,12,16) tren backend simulate
    python scripts/06_fixed_depth_baseline.py --backend simulate

    # tuy chinh danh sach depth + dataset pennylane
    python scripts/06_fixed_depth_baseline.py --backend pennylane --dataset heart \
        --depths 2,4,6,8,10
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.synthetic_bp.stage2.fixed_scheduler import FixedScheduler
from src.synthetic_bp.stage3.trainer import AdaptiveTrainer


def load_data(args):
    if args.backend != "pennylane":
        return None
    from dataset import load_binary_dataset
    d = load_binary_dataset(args.dataset, n_qubits=args.n_qubits,
                            max_samples=args.max_samples, seed=42)
    print(f"[data] {d['name']}: train={d['n_train']} val={d['n_val']}")
    return d


def build_backend(args, data, seed, depth):
    from src.synthetic_bp.stage3.circuit_backend import SimulateBackend
    if args.backend == "pennylane":
        from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend
        return PennyLaneBackend(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_qubits=args.n_qubits, initial_depth=depth,
            max_depth=depth, topology=args.topology,
            entangler_type=args.entangler, cost_type=args.cost_type, seed=seed)
    return SimulateBackend(n_qubits=args.n_qubits, initial_depth=depth,
                           max_depth=depth, cost_type=args.cost_type,
                           topology=args.topology, seed=seed)


def deploy(args, data, depth):
    b = build_backend(args, data, seed=args.seed, depth=depth)
    tr = AdaptiveTrainer(b, FixedScheduler(), n_qubits=args.n_qubits,
                         cost_type=args.cost_type, topology=args.topology,
                         epochs=args.epochs)
    return tr.run()


def summarize(name, df, args, depth):
    first, last = df.iloc[0], df.iloc[-1]
    return {
        "scheduler": name,
        "backend": args.backend,
        "seed": args.seed,
        "epochs": args.epochs,
        "fixed_depth_k": depth,
        "acc_start": round(float(first["accuracy"]), 4),
        "acc_end": round(float(last["accuracy"]), 4),
        "acc_delta": round(float(last["accuracy"]) - float(first["accuracy"]), 4),
        "f1_end": round(float(last.get("f1", float("nan"))), 4),
        "roc_auc_end": round(float(last.get("roc_auc", float("nan"))), 4),
        "depth_start": int(first["depth"]),
        "depth_end": int(last["depth"]),
        "n_commits": int(last["n_commits"]),
        "n_rollbacks": int(last["n_rollbacks"]),
    }


def plot(rows, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    ks = [r["fixed_depth_k"] for r in rows]
    acc = [r["acc_end"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, acc, marker="o", linewidth=1.8)
    ax.set(title="Fixed depth k (khong doi) - accuracy cuoi vs k",
          xlabel="depth k (co dinh)", ylabel="accuracy")
    ax.grid(alpha=0.3)
    for k, a in zip(ks, acc):
        ax.text(k, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["simulate", "pennylane"], default="simulate")
    ap.add_argument("--dataset", default="iris")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--depths", default="2,4,6,8,10,12,16",
                    help="danh sach depth k co dinh, phan cach boi dau phay")
    ap.add_argument("--cost-type", default="local")
    ap.add_argument("--topology", default="linear")
    ap.add_argument("--entangler", choices=["cnot", "cz"], default="cnot")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    out_dir = Path(args.out or ROOT / "outputs" / "comparison" / args.backend)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args)
    print(f"[fixed] {len(depths)} depth | backend={args.backend} "
          f"seed={args.seed} epochs={args.epochs} depths={depths}")

    rows = []
    for k in depths:
        name = f"fixed_k{k}"
        print(f"[fixed] deploy {name}")
        df = deploy(args, data, depth=k)
        df.to_csv(out_dir / f"deploy_log_{name}.csv", index=False)
        rows.append(summarize(name, df, args, k))

    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "comparison_fixed.csv", index=False)
    plot(rows, out_dir / "comparison_fixed.png")

    cols = ["scheduler", "fixed_depth_k", "acc_start", "acc_end", "acc_delta",
            "f1_end", "depth_start", "depth_end"]
    print(comp[cols].to_string(index=False))
    print(f"[fixed] saved -> {out_dir}")
    print(f"[fixed] de gop vao comparison.csv (rule/greedy/ppo/dqn), doc "
          f"outputs/comparison/{args.backend}/comparison_fixed.csv va "
          f"chon dong fixed_k tot nhat (hoac gop het) truoc khi ve chung bieu do.")


if __name__ == "__main__":
    main()
