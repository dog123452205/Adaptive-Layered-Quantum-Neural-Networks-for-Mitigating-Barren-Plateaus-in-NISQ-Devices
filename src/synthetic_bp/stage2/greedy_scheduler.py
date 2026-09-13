"""
Stage 2 — Greedy Gradient-Guided Scheduler (baseline không học, kiểu ADAPT).

Khác Rule-Based ở chỗ: không dùng ngưỡng độ sâu tĩnh mà quyết định theo tín hiệu
gradient ĐO ĐƯỢC tại runtime — mọc lớp khi còn dư địa gradient trên τ, tỉa đúng
lớp có đóng góp gradient thấp nhất, dừng khi phương sai chạm sàn τ. Cùng contract
SchedulerBase nên cắm thẳng vào Stage 3 / bảng so sánh như rule-based, không cần train.

Tham chiếu tinh thần: ADAPT-VQE (Grimsley et al. 2019) — mọc theo gradient, dừng khi tụt.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.synthetic_bp.stage2.scheduler_base import (
    SchedulerBase, RuntimeState, MutationRequest,
    A_KEEP, A_ADD, A_STOP, A_PRUNE, A_REDUCE, DS_HEALTHY,
)
from src.synthetic_bp.stage2.rule_based_scheduler import _load_calibration, _lookup

try:
    from src.synthetic_bp.constants import LayerStatus
    ST_ACTIVE = LayerStatus.ACTIVE.value
except Exception:
    ST_ACTIVE = "active"


@dataclass
class GreedyConfig:
    grow_margin: float = 3.0    # add khi variance > τ * grow_margin (còn dư địa)
    stop_margin: float = 1.0    # dừng khi variance <= τ * stop_margin (chạm sàn)
    prune_ratio: float = 0.15   # lớp có grad_var < prune_ratio * trung bình -> coi là chết
    healthy_nzr: float = 0.40   # near_zero toàn cục dưới mức này mới cho mọc
    reduce_nzr: float = 0.40    # near_zero >= -> BP toàn cục, giảm vướng víu
    reduce_min_ent: int = 6
    min_depth: int = 1
    max_depth: int = 16


class GreedyScheduler(SchedulerBase):
    def __init__(self, calibration: dict | str | Path | None = None,
                 cfg: GreedyConfig | None = None,
                 warmup_epochs: int = 10, action_interval_epochs: int = 5,
                 cooldown_epochs: int = 5):
        super().__init__(warmup_epochs, action_interval_epochs, cooldown_epochs)
        self.cal = _load_calibration(calibration)
        self.cfg = cfg or GreedyConfig()

    def _active(self, layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [l for l in layers if l.get("layer_status", ST_ACTIVE) == ST_ACTIVE]

    def _deadest_layer(self, active: list[dict[str, Any]]):
        """Lớp active có grad_variance thấp nhất, kèm tỉ lệ so với trung bình."""
        if not active:
            return None, 1.0
        gvs = [l.get("layer_grad_variance", 0.0) for l in active]
        mean_gv = sum(gvs) / len(gvs) if gvs else 0.0
        worst = min(active, key=lambda l: l.get("layer_grad_variance", 1e9))
        ratio = (worst.get("layer_grad_variance", 0.0) / mean_gv) if mean_gv > 0 else 1.0
        return worst, ratio

    def decide(self, state: RuntimeState) -> MutationRequest:
        cfg = self.cfg
        rule = _lookup(self.cal, state.n_qubits, state.entanglement_topology,
                       state.cost_type, state.depth)
        tau = float(rule["thresholds"].get("tau_empirical", 0.0))
        tau_eff = max(tau, 1e-6)
        var = state.spatial_grad_variance
        diag, risk = rule.get("diagnosis", DS_HEALTHY), rule.get("risk_level", "LOW")

        def req(action, layer=None, reason=""):
            return MutationRequest(action=action, target_layer=layer,
                                   reason=reason, diagnosis=diag, risk_level=risk)

        active = self._active(state.layers)
        depth = len(active) if active else state.depth
        worst, ratio = self._deadest_layer(active)

        # 1) có lớp gần như không đóng góp gradient -> tỉa đúng lớp đó
        if worst is not None and depth > cfg.min_depth and ratio < cfg.prune_ratio:
            return req(A_PRUNE, int(worst["layer_id"]),
                       f"greedy prune: layer contributes {ratio:.0%} of mean grad")

        # 2) BP toàn cục + nhiều cổng vướng víu -> giảm vướng víu
        if state.near_zero_ratio >= cfg.reduce_nzr and state.n_entangling_gates >= cfg.reduce_min_ent:
            return req(A_REDUCE, reason=f"global BP (nzr={state.near_zero_ratio:.2f})")

        # 3) phương sai gradient chạm sàn τ -> dừng mọc
        if var <= tau_eff * cfg.stop_margin and depth > cfg.min_depth:
            return req(A_STOP, reason=f"variance {var:.2e} <= τ*{cfg.stop_margin}")

        # 4) còn dư địa gradient trên τ và mạch còn lành -> mọc thêm
        if (var > tau_eff * cfg.grow_margin and state.near_zero_ratio < cfg.healthy_nzr
                and depth < cfg.max_depth):
            return req(A_ADD, reason=f"headroom: variance {var:.2e} > τ*{cfg.grow_margin}")

        return req(A_KEEP, reason="greedy: no better move")
