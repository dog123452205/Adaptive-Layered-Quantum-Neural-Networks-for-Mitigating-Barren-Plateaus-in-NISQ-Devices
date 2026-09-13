"""
Stage 3 — Adaptive Classification Trainer (Execution Engine).

Ghép: CircuitBackend + Scheduler (Stage 2) + MutationEngine + ParameterRegistry.
Mỗi epoch: train -> đánh giá -> tiến triển mask đang chạy -> (đúng nhịp) hỏi scheduler
-> thực thi mutation -> ghi log. Kết thúc xuất training_log.parquet.

Trainer này KHÔNG biết scheduler là rule-based hay RL (nhờ SchedulerBase) -> dùng chung
cho cả paper 1 (rule) và V2 (RL). Đồng thời là RUNTIME cho RL backend.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from src.synthetic_bp.stage3.circuit_backend import CircuitBackend
from src.synthetic_bp.stage3.parameter_registry import ParameterRegistry
from src.synthetic_bp.stage3.mutation_engine import MutationEngine
from src.synthetic_bp.stage2.scheduler_base import SchedulerBase, RuntimeState
from src.synthetic_bp.stage3.telemetry_engine import TelemetryEngine, TeleConfig

_DESTRUCTIVE = {"add_layer", "propose_prune_layer", "propose_reduce_entanglement"}


class AdaptiveTrainer:
    def __init__(self, backend: CircuitBackend, scheduler: SchedulerBase,
                 n_qubits: int, cost_type: str, topology: str,
                 epochs: int = 100, mask_epochs_T: int = 5,
                 max_relative_loss_increase: float = 0.02,
                 log_telemetry: bool = True, telemetry_guard: bool = True,
                 telemetry_cfg=None, offline_risk: float = 0.5):
        self.backend = backend
        self.scheduler = scheduler
        self.n_qubits = n_qubits
        self.cost_type = cost_type
        self.topology = topology
        self.epochs = epochs
        # TE deploy-time: log_telemetry chỉ ghi nhãn; telemetry_guard mới veto hành động.
        # Mặc định CẢ HAI BẬT (khớp mô hình hệ thống trong báo cáo: TE luôn hoạt động).
        # Truyền log_telemetry=False, telemetry_guard=False tường minh để tái lập baseline cũ.
        self.log_telemetry = log_telemetry or telemetry_guard
        self.telemetry_guard = telemetry_guard
        self.offline_risk = offline_risk
        self.te = TelemetryEngine(telemetry_cfg or TeleConfig()) if self.log_telemetry else None
        self.registry = ParameterRegistry(backend)
        self.mutation = MutationEngine(
            backend, self.registry, mask_epochs_T=mask_epochs_T,
            max_relative_loss_increase=max_relative_loss_increase)
        self.log: list[dict[str, Any]] = []

    def _runtime_state(self, epoch: int, eval_metrics: dict,
                       last_mut: int) -> RuntimeState:
        g = self.backend.global_metrics()
        return RuntimeState(
            epoch=epoch,
            n_qubits=self.n_qubits,
            entanglement_topology=self.topology,
            cost_type=self.cost_type,
            depth=g["depth"],
            n_entangling_gates=g["n_entangling_gates"],
            validation_loss=eval_metrics["validation_loss"],
            training_loss=g["training_loss"],
            grad_norm=g["grad_norm"],
            spatial_grad_variance=g["spatial_grad_variance"],
            near_zero_ratio=g["near_zero_ratio"],
            layers=self.backend.layer_metrics(),
            last_mutation_epoch=last_mut,
        )

    def run(self) -> pd.DataFrame:
        self.scheduler.reset()
        if self.te is not None:
            self.te.reset()
        # checkpoint khởi tạo
        ev = self.backend.evaluate()
        self.registry.commit_checkpoint(ev["validation_loss"])
        last_mut = -10_000

        for epoch in range(self.epochs):
            self.backend.train_epoch()
            ev = self.backend.evaluate()
            vloss = ev["validation_loss"]

            # tiến triển mask đang chạy (nếu có)
            mut_status = self.mutation.on_epoch(vloss)

            # chẩn đoán TE (nếu bật) - dùng cho log và/hoặc guard, áp cho MỌI scheduler
            te_out, proposed = None, None
            if self.te is not None:
                g0 = self.backend.global_metrics()
                lnzr = [l["layer_near_zero_ratio"] for l in self.backend.layer_metrics()]
                te_out = self.te.step(sgv=g0["spatial_grad_variance"], nzr=g0["near_zero_ratio"],
                                      vloss=vloss, layer_nzr=lnzr, depth=g0["depth"],
                                      offline_risk=self.offline_risk)

            # đúng nhịp + không bận mask -> hỏi scheduler
            decided = "keep"
            if not self.mutation.busy and self.scheduler.should_act(epoch, last_mut):
                state = self._runtime_state(epoch, ev, last_mut)
                req = self.scheduler.decide(state)
                decided = req.action
                proposed = req.action
                target = req.target_layer
                # guardrail: solved rồi mà đòi hành động phá hoại -> ép về keep
                if (self.telemetry_guard and te_out is not None
                        and te_out.labels["is_solved"] and req.action in _DESTRUCTIVE):
                    decided, target = "keep", None
                s = self.mutation.submit(decided, target, vloss)
                if s in ("in_progress", "committed") and decided != "keep":
                    last_mut = epoch
                if s != "none":
                    mut_status = s

            g = self.backend.global_metrics()
            row = {
                "epoch": epoch,
                "training_loss": g["training_loss"],
                "validation_loss": vloss,
                "accuracy": ev["accuracy"],
                "f1": ev["f1"],
                "roc_auc": ev["roc_auc"],
                "grad_norm": g["grad_norm"],
                "spatial_grad_variance": g["spatial_grad_variance"],
                "near_zero_ratio": g["near_zero_ratio"],
                "depth": g["depth"],
                "n_entangling_gates": g["n_entangling_gates"],
                "entangler_strength": g["entangler_strength"],
                "scheduler_action": decided,
                "mutation_status": mut_status,
                "n_commits": self.registry.n_commits,
                "n_rollbacks": self.registry.n_rollbacks,
            }
            if te_out is not None:
                # 5 cờ chẩn đoán cho training_log + đếm mutation phá hoại sau solved
                row["proposed_action"] = proposed if proposed is not None else "keep"
                row["vetoed"] = bool(proposed in _DESTRUCTIVE and decided == "keep"
                                     and self.telemetry_guard and te_out.labels["is_solved"])
                for k, v in te_out.labels.items():
                    row[k] = v if not isinstance(v, list) else str(v)
                row["te_p_solved"] = te_out.features["p_solved"]
            self.log.append(row)

        return pd.DataFrame(self.log)

    def save_log(self, path: str | Path) -> Path:
        df = pd.DataFrame(self.log)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
        return path
