"""
Stage 2 — Rule-Based Trainability Scheduler (V1).

Bộ điều phối theo LUẬT TĨNH: tra ngưỡng tau_empirical + mức rủi ro từ
calibration_rules.json (Stage 1C), kết hợp vài luật về độ sâu / vướng víu,
để phát Mutation Request (KEEP / ADD / STOP / PROPOSE_PRUNE / PROPOSE_REDUCE).

Đây là scheduler cho paper 1. RL scheduler (V2) sẽ kế thừa cùng SchedulerBase,
thay phần decide() bằng policy học được, nhưng giữ nguyên contract với Stage 3.

Độ phức tạp tra cứu O(số rule) (số rule nhỏ) — đúng tinh thần "lookup" của V1.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.synthetic_bp.stage2.scheduler_base import (
    SchedulerBase, RuntimeState, MutationRequest,
    A_KEEP, A_ADD, A_STOP, A_PRUNE, A_REDUCE, DS_HEALTHY,
)

try:
    from src.synthetic_bp.constants import LayerStatus
    ST_ACTIVE = LayerStatus.ACTIVE.value
except Exception:
    ST_ACTIVE = "active"


@dataclass
class RuleBasedConfig:
    target_depth: int = 6        # đạt độ sâu này -> dừng tăng trưởng
    min_depth: int = 1           # không cắt dưới mức này
    max_depth: int = 16          # không thêm quá mức này
    dead_nzr: float = 0.60       # near_zero_ratio theo lớp >= -> lớp "chết"
    global_bp_nzr: float = 0.40  # near_zero toàn cục >= -> BP nặng
    reduce_min_ent: int = 6      # số cổng vướng víu tối thiểu để xét giảm
    loss_divergence: float = 0.05  # val_loss tăng tương đối > -> cẩn trọng, không ADD


def _load_calibration(calibration: dict | str | Path | None) -> dict:
    if calibration is None:
        return {"rules": [], "global_default": {"tau_empirical": 0.0,
                                                 "near_zero_grad_threshold": 1e-6}}
    if isinstance(calibration, (str, Path)):
        return json.loads(Path(calibration).read_text(encoding="utf-8"))
    return calibration


def _lookup(cal: dict, n_qubits: int, topo: str, cost: str, depth: int) -> dict:
    for r in cal.get("rules", []):
        k = r["key"]
        if (k["n_qubits"] == n_qubits and k["entanglement_topology"] == topo
                and k["cost_type"] == cost and k["depth"] == depth):
            return r
    return {"thresholds": cal.get("global_default", {"tau_empirical": 0.0}),
            "diagnosis": DS_HEALTHY, "risk_level": "LOW", "fallback": True}


class RuleBasedScheduler(SchedulerBase):
    def __init__(self, calibration: dict | str | Path | None = None,
                 cfg: RuleBasedConfig | None = None,
                 warmup_epochs: int = 10, action_interval_epochs: int = 5,
                 cooldown_epochs: int = 5):
        super().__init__(warmup_epochs, action_interval_epochs, cooldown_epochs)
        self.cal = _load_calibration(calibration)
        self.cfg = cfg or RuleBasedConfig()
        self._prev_val_loss: float | None = None

    def reset(self) -> None:
        self._prev_val_loss = None

    def _weakest_active_layer(self, layers: list[dict[str, Any]]) -> int | None:
        """Lớp active có grad_variance thấp nhất (yếu nhất) -> ứng viên prune."""
        active = [l for l in layers if l.get("layer_status", ST_ACTIVE) == ST_ACTIVE]
        if not active:
            return None
        worst = min(active, key=lambda l: l.get("layer_grad_variance", 1e9))
        return int(worst["layer_id"])

    def decide(self, state: RuntimeState) -> MutationRequest:
        cfg = self.cfg
        rule = _lookup(self.cal, state.n_qubits, state.entanglement_topology,
                       state.cost_type, state.depth)
        tau = float(rule["thresholds"].get("tau_empirical", 0.0))
        risk = rule.get("risk_level", "LOW")
        diag = rule.get("diagnosis", DS_HEALTHY)

        def req(action, layer=None, reason=""):
            return MutationRequest(action=action, target_layer=layer,
                                   reason=reason, diagnosis=diag, risk_level=risk)

        # đo xu hướng loss (để tránh ADD khi đang xấu đi)
        loss_rising = False
        if self._prev_val_loss is not None and self._prev_val_loss > 0:
            rel = (state.validation_loss - self._prev_val_loss) / self._prev_val_loss
            loss_rising = rel > cfg.loss_divergence
        self._prev_val_loss = state.validation_loss

        active_layers = [l for l in state.layers
                         if l.get("layer_status", ST_ACTIVE) == ST_ACTIVE]
        depth = len(active_layers) if active_layers else state.depth
        weakest = self._weakest_active_layer(state.layers)

        # ---- LUẬT (ưu tiên từ trên xuống) -------------------------------
        # 1) Rủi ro cao + có lớp chết + còn cắt được -> đề xuất cắt lớp yếu nhất
        if risk == "HIGH" and weakest is not None and depth > cfg.min_depth:
            wl = next((l for l in state.layers if l["layer_id"] == weakest), None)
            if wl and wl.get("layer_near_zero_ratio", 0.0) >= cfg.dead_nzr:
                return req(A_PRUNE, weakest,
                           f"HIGH risk + dead layer (nzr={wl['layer_near_zero_ratio']:.2f})")

        # 2) BP toàn cục nặng + nhiều cổng vướng víu -> giảm vướng víu
        if (state.grad_norm < tau and state.near_zero_ratio >= cfg.global_bp_nzr
                and state.n_entangling_gates >= cfg.reduce_min_ent):
            return req(A_REDUCE, reason=f"global BP (grad_norm<{tau:.3f}, "
                                        f"nzr={state.near_zero_ratio:.2f})")

        # 3) Trainability ổn (grad_norm>=tau) & mạch còn nông & loss không xấu -> thêm lớp
        if (state.grad_norm >= tau and depth < cfg.target_depth
                and depth < cfg.max_depth and not loss_rising):
            return req(A_ADD, reason=f"healthy & shallow (depth={depth})")

        # 4) Đã đạt độ sâu mục tiêu & ổn định -> dừng tăng trưởng
        if depth >= cfg.target_depth and state.grad_norm >= tau:
            return req(A_STOP, reason=f"reached target depth ({depth})")

        # 5) Mặc định: giữ nguyên
        return req(A_KEEP, reason="no rule triggered")
