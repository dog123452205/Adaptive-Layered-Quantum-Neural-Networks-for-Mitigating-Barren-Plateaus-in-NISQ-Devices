"""
Run Stage 3 — Adaptive Classification Training.

--backend simulate : chạy ngay, không cần pennylane (test orchestration).
--backend pennylane: mạch thật (cần pennylane + dữ liệu MNIST PCA).

Ví dụ:
    python scripts/03_run_adaptive_training.py \
        --backend simulate \
        --calibration outputs/stage1c/calibration_rules.json \
        --epochs 100 --out outputs/training/training_log.parquet
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.synthetic_bp.stage2.rule_based_scheduler import RuleBasedScheduler, RuleBasedConfig
from src.synthetic_bp.stage3.trainer import AdaptiveTrainer


def build_backend(args):
    if args.backend == "simulate":
        from src.synthetic_bp.stage3.circuit_backend import SimulateBackend
        return SimulateBackend(
            n_qubits=args.n_qubits, initial_depth=args.initial_depth,
            max_depth=args.max_depth, cost_type=args.cost_type,
            topology=args.topology, seed=args.seed)
    # pennylane: cần dữ liệu MNIST PCA->angle
    from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend
    from src.synthetic_bp.stage3.data import load_mnist_pca_angle
    X_tr, y_tr, X_va, y_va = load_mnist_pca_angle(n_qubits=args.n_qubits, seed=args.seed)
    return PennyLaneBackend(
        X_tr, y_tr, X_va, y_va, n_qubits=args.n_qubits,
        initial_depth=args.initial_depth, max_depth=args.max_depth,
        topology=args.topology, cost_type=args.cost_type, seed=args.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["simulate", "pennylane"], default="simulate")
    ap.add_argument("--calibration", default="outputs/stage1c/calibration_rules.json")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--initial-depth", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=16)
    ap.add_argument("--cost-type", default="local")
    ap.add_argument("--topology", default="linear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/training/training_log.parquet")
    args = ap.parse_args()

    backend = build_backend(args)
    scheduler = RuleBasedScheduler(
        calibration=args.calibration if Path(args.calibration).exists() else None,
        cfg=RuleBasedConfig(target_depth=6),
        warmup_epochs=10, action_interval_epochs=5, cooldown_epochs=5)

    trainer = AdaptiveTrainer(
        backend=backend, scheduler=scheduler,
        n_qubits=args.n_qubits, cost_type=args.cost_type, topology=args.topology,
        epochs=args.epochs)

    df = trainer.run()
    out = trainer.save_log(args.out)

    # tóm tắt
    first, last = df.iloc[0], df.iloc[-1]
    print(f"[Stage 3] backend={args.backend} | epochs={args.epochs}")
    print(f"[Stage 3] loss   {first['validation_loss']:.3f} -> {last['validation_loss']:.3f}")
    print(f"[Stage 3] acc    {first['accuracy']:.3f} -> {last['accuracy']:.3f}")
    print(f"[Stage 3] depth  {int(first['depth'])} -> {int(last['depth'])}")
    print(f"[Stage 3] mutations: commit={int(last['n_commits'])}, "
          f"rollback={int(last['n_rollbacks'])}")
    acts = df["scheduler_action"].value_counts().to_dict()
    print(f"[Stage 3] actions: {acts}")
    print(f"[Stage 3] log saved -> {out}")


if __name__ == "__main__":
    main()
