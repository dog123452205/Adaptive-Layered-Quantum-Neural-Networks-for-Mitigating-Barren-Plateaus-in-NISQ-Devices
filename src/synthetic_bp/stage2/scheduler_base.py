"""
Stage 2 — Trainability Scheduler: interface chung.

Định nghĩa hợp đồng (contract) mà MỌI scheduler phải tuân theo, để Stage 3 gọi
thống nhất dù scheduler là Rule-based (V1) hay RL (V2). Nhờ vậy có thể thay
"bộ não ra quyết định" mà không sửa Execution Engine.

Scheduler chỉ phát ra MACRO-decision (Mutation Request). Vòng đời nhiều bước
(soft_mask -> commit/rollback) do Stage 3 đảm nhiệm theo config template.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

# Enum thật của dự án (fallback khi chạy độc lập)
try:
    from src.synthetic_bp.constants import SchedulerAction, DiagnosisState
    A_KEEP = SchedulerAction.KEEP.value
    A_ADD = SchedulerAction.ADD_LAYER.value
    A_STOP = SchedulerAction.STOP_GROWTH.value
    A_PRUNE = SchedulerAction.PROPOSE_PRUNE_LAYER.value
    A_REDUCE = SchedulerAction.PROPOSE_REDUCE_ENTANGLEMENT.value
    DS_HEALTHY = DiagnosisState.HEALTHY.value
except Exception:
    A_KEEP, A_ADD, A_STOP = "keep", "add_layer", "stop_growth"
    A_PRUNE, A_REDUCE = "propose_prune_layer", "propose_reduce_entanglement"
    DS_HEALTHY = "healthy"

# Tập macro-action mà scheduler được phép phát (5). Vòng đời còn lại do Stage 3 lo.
MACRO_ACTIONS = [A_KEEP, A_ADD, A_STOP, A_PRUNE, A_REDUCE]


@dataclass
class RuntimeState:
    """Ảnh chụp trạng thái huấn luyện mà Stage 3 cung cấp cho scheduler tại mỗi tick."""
    epoch: int
    # cấu hình mạch hiện tại (để tra calibration)
    n_qubits: int
    entanglement_topology: str
    cost_type: str
    depth: int
    n_entangling_gates: int
    # chỉ số toàn cục runtime
    validation_loss: float
    training_loss: float
    grad_norm: float
    spatial_grad_variance: float
    near_zero_ratio: float
    # chỉ số theo lớp: list dict, mỗi dict gồm
    #   layer_id, layer_status, layer_grad_norm, layer_grad_variance,
    #   layer_near_zero_ratio, mask_value
    layers: list[dict[str, Any]] = field(default_factory=list)
    # lịch sử để chống mutate liên tục
    last_mutation_epoch: int = -10_000


@dataclass
class MutationRequest:
    """Quyết định scheduler gửi cho Stage 3."""
    action: str                       # 1 trong MACRO_ACTIONS
    target_layer: Optional[int] = None
    reason: str = ""
    diagnosis: str = DS_HEALTHY
    risk_level: str = "LOW"

    def is_noop(self) -> bool:
        return self.action == A_KEEP and self.target_layer is None


class SchedulerBase(ABC):
    """
    Hợp đồng chung cho mọi scheduler Stage 2.

    Lifecycle do Stage 3 gọi:
        scheduler.reset()
        ... mỗi epoch:
            if scheduler.should_act(epoch):
                req = scheduler.decide(runtime_state)
                stage3.apply(req)
    """

    def __init__(self, warmup_epochs: int = 10, action_interval_epochs: int = 5,
                 cooldown_epochs: int = 5):
        self.warmup_epochs = warmup_epochs
        self.action_interval_epochs = action_interval_epochs
        self.cooldown_epochs = cooldown_epochs

    def reset(self) -> None:
        """Đặt lại trạng thái nội bộ (override nếu cần)."""

    def should_act(self, epoch: int, last_mutation_epoch: int = -10_000) -> bool:
        """Chỉ ra quyết định sau warmup, đúng nhịp interval, và ngoài cooldown."""
        if epoch < self.warmup_epochs:
            return False
        if epoch % self.action_interval_epochs != 0:
            return False
        if epoch - last_mutation_epoch < self.cooldown_epochs:
            return False
        return True

    @abstractmethod
    def decide(self, state: RuntimeState) -> MutationRequest:
        """Trả về MutationRequest dựa trên trạng thái runtime."""
        raise NotImplementedError
