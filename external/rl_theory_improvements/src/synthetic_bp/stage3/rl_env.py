"""
Stage 3 — RL Training Environment.

Môi trường Gym-style để TRAIN agent (PPO/DQN) trên Stage 3.
Chọn backend qua tham số `backend`:
  - "simulate" (mặc định): SimulateBackend, numpy thuần, nhanh — prototype thuật toán.
  - "pennylane": PennyLaneBackend, mạch lượng tử THẬT — cần `data` (X/y train+val).

Cả hai cùng interface CircuitBackend nên MutationEngine/ParameterRegistry/encoder
dùng chung, agent train ở đây deploy thẳng làm scheduler.

Mỗi step = MỘT quyết định (không còn quét lần lượt từng layer trong 1 "round" rồi
mới train + phát reward ở step cuối). Trước đây agent ra `depth` action/round nhưng
chỉ action ở step cuối round nhận reward != 0 -> credit assignment rất thưa/trễ, đặc
biệt hại cho DQN (TD 1 bước phải bootstrap qua nhiều step reward=0 mới lan được tín
hiệu). Nay: mỗi step đều train k_epochs + phát reward ngay, target layer (cho PRUNE)
lấy theo layer_grad_variance thấp nhất (weakest_layer_pos) — khớp đúng cách
RLScheduler chọn layer lúc deploy, tránh lệch phân phối train/deploy.
"""
from __future__ import annotations
import numpy as np

from src.synthetic_bp.stage3.circuit_backend import SimulateBackend
from src.synthetic_bp.stage3.parameter_registry import ParameterRegistry
from src.synthetic_bp.stage3.mutation_engine import MutationEngine
from src.synthetic_bp.stage3.telemetry_engine import TelemetryEngine, TeleConfig
from src.synthetic_bp.stage2.scheduler_base import (
    MACRO_ACTIONS, A_KEEP, A_ADD, A_STOP, A_PRUNE, A_REDUCE,
)
from src.synthetic_bp.stage2.rl_scheduler import (
    encode_layer_state, STATE_DIM, A_IDX, weakest_layer_pos,
)

DEFAULT_W = dict(w1=0.5, w2=5.0, w3=0.05, w4=0.02, w5=1.0, w6=0.5)


class Stage3RLEnv:
    def __init__(self, n_qubits=4, initial_depth=2, max_depth=16, min_depth=1,
                 cost_type="local", topology="linear", tau=0.1, calibration=None,
                 k_epochs=5, total_epochs=60, mask_epochs_T=5,
                 max_relative_loss_increase=0.02,
                 weights=None, seed=0,
                 backend="simulate", data=None,
                 entangler_type="cnot", lr=0.01, surrogate=None,
                 use_telemetry=True, telemetry_cfg=None):
        self.cfg = dict(n_qubits=n_qubits, initial_depth=initial_depth,
                        max_depth=max_depth, cost_type=cost_type,
                        topology=topology, seed=seed)
        self.min_depth, self.max_depth = min_depth, max_depth
        self.tau_default = tau
        self.cost_type_val, self.topology_val, self.n_qubits_val = cost_type, topology, n_qubits
        self._calibration = self._load_calibration(calibration)
        self.tau = self._resolve_tau(initial_depth)
        self.k_epochs, self.total_epochs = k_epochs, total_epochs
        self.mask_epochs_T = mask_epochs_T
        self.max_relative_loss_increase = max_relative_loss_increase
        self.w = dict(DEFAULT_W if weights is None else weights)
        self.rng = np.random.default_rng(seed)
        # --- lựa chọn backend ---
        self.backend = backend
        self.data = data
        self.entangler_type = entangler_type
        self.lr = lr
        self.surrogate = surrogate
        # Telemetry Engine: mặc định BẬT (khớp mô hình hệ thống trong báo cáo: TE luôn
        # hoạt động). Truyền use_telemetry=False tường minh để tái lập baseline cũ.
        self.te = TelemetryEngine(telemetry_cfg or TeleConfig()) if use_telemetry else None
        self._te_out = None
        if backend == "pennylane" and data is None:
            raise ValueError(
                "backend='pennylane' cần data=dict(X_train,y_train,X_val,y_val). "
                "Dùng dataset.load_binary_dataset(...) để tạo.")
        self.reset()

    def _make_backend(self):
        """Tạo backend đúng loại. SimulateBackend không cần data; PennyLaneBackend cần."""
        if self.backend == "pennylane":
            from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend
            d = self.data
            return PennyLaneBackend(
                d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                n_qubits=self.cfg["n_qubits"], initial_depth=self.cfg["initial_depth"],
                max_depth=self.cfg["max_depth"], topology=self.cfg["topology"],
                entangler_type=self.entangler_type, cost_type=self.cfg["cost_type"],
                lr=self.lr, seed=self.cfg["seed"])
        return SimulateBackend(**self.cfg, surrogate=self.surrogate)

    def reset(self):
        self.b = self._make_backend()
        self.reg = ParameterRegistry(self.b)
        ev = self.b.evaluate()
        self.reg.commit_checkpoint(ev["validation_loss"])
        self.eng = MutationEngine(self.b, self.reg, mask_epochs_T=self.mask_epochs_T,
                                  max_relative_loss_increase=self.max_relative_loss_increase)
        self.epoch = 0
        self._growth_locked = False
        if self.te is not None:
            self.te.reset()
        self._te_out = None
        self._last_rb = 0
        self._prev = self._metrics()
        return self._encode(), self._mask()

    def _metrics(self):
        g = self.b.global_metrics()
        return dict(sgv=g["spatial_grad_variance"], vloss=self.b.evaluate()["validation_loss"],
                    depth=g["depth"], nent=g["n_entangling_gates"], nzr=g["near_zero_ratio"])

    def _mask(self):
        m = np.ones(len(MACRO_ACTIONS), dtype=bool)
        if self.eng.busy:
            # đang trong cửa sổ masking (ramp) của 1 mutation chưa resolve -> submit()
            # mới sẽ bị eng lặng lẽ bỏ qua (xem MutationEngine.submit: `if self.busy: return`).
            # Trước đây mask không phản ánh điều này -> agent có thể chọn PRUNE/ADD/REDUCE
            # trong lúc bận, action bị drop nhưng vẫn được lưu vào buffer như thể có tác
            # dụng -> nhiễu dữ liệu train. Chỉ cho KEEP cho tới khi mutation hiện tại resolve.
            m[:] = False
            m[A_IDX[A_KEEP]] = True
            return m
        depth = self.b.depth
        if depth >= self.max_depth or self._growth_locked:
            m[A_IDX[A_ADD]] = False
        if depth <= self.min_depth:
            m[A_IDX[A_PRUNE]] = False
        # guardrail TE: chặn hành động phá hoại khi chẩn đoán đã hội tụ / kẹt
        if self._te_out is not None:
            tm = self._te_out.mask
            if not tm["add_layer"]:
                m[A_IDX[A_ADD]] = False
            if not tm["propose_prune_layer"]:
                m[A_IDX[A_PRUNE]] = False
            if not tm["propose_reduce_entanglement"]:
                m[A_IDX[A_REDUCE]] = False
        return m

    @staticmethod
    def _load_calibration(calibration):
        if calibration is None:
            return None
        if isinstance(calibration, dict):
            return calibration
        import json
        from pathlib import Path
        p = Path(calibration)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _resolve_tau(self, depth):
        """τ thật từ Stage 1C theo cấu hình + depth hiện tại; else tau_default."""
        if self._calibration is None:
            return self.tau_default
        try:
            from src.synthetic_bp.stage1c.calibration import lookup_rule
            rule = lookup_rule(self._calibration, self.n_qubits_val,
                               self.topology_val, self.cost_type_val, int(depth))
            return float(rule["thresholds"].get("tau_empirical", self.tau_default))
        except Exception:
            gd = self._calibration.get("global_default", {})
            return float(gd.get("tau_empirical", self.tau_default))

    def _resolve_risk(self, depth):
        """offline_risk in [0,1] từ Stage 1C theo cấu hình + depth; else 0.5 trung tính."""
        if self._calibration is None:
            return 0.5
        try:
            from src.synthetic_bp.stage1c.calibration import lookup_rule
            rule = lookup_rule(self._calibration, self.n_qubits_val,
                               self.topology_val, self.cost_type_val, int(depth))
            level = str(rule.get("bp_risk", rule.get("risk", "medium"))).lower()
            return {"low": 0.2, "medium": 0.5, "high": 0.85}.get(level, 0.5)
        except Exception:
            return 0.5

    def _encode(self):
        g = self.b.global_metrics()
        g["validation_loss"] = self.b.evaluate()["validation_loss"]
        self.tau = self._resolve_tau(g.get("depth", self.b.depth))  # τ động theo depth
        layers = self.b.layer_metrics()
        pos = weakest_layer_pos(layers)
        return encode_layer_state(layers, g, self.tau, pos, self.epoch,
                                  self.total_epochs, self._growth_locked)

    def step(self, action_idx):
        action = MACRO_ACTIONS[action_idx]
        layers = self.b.layer_metrics()
        vloss = self.b.evaluate()["validation_loss"]

        if action == A_STOP:
            self._growth_locked = True
        # đưa quyết định cho mutation engine (nếu không bận) — target luôn là lớp
        # yếu nhất (khớp encode/deploy), _mask() đã đảm bảo action != KEEP chỉ được
        # chọn khi eng KHÔNG bận nên submit() ở đây không bao giờ bị lặng lẽ bỏ qua.
        if not self.eng.busy:
            tgt = layers[weakest_layer_pos(layers)]["layer_id"] if action == A_PRUNE else None
            self.eng.submit(action, tgt, vloss)

        # train k epoch + tiến triển mutation — MỖI step, không đợi quét hết layer.
        for _ in range(self.k_epochs):
            self.b.train_epoch()
            self.eng.on_epoch(self.b.evaluate()["validation_loss"])
        self.epoch += self.k_epochs

        cur = self._metrics()
        info = {"action": action}
        if self.te is not None:
            lnzr = [l["layer_near_zero_ratio"] for l in self.b.layer_metrics()]
            self._te_out = self.te.step(
                sgv=cur["sgv"], nzr=cur["nzr"], vloss=cur["vloss"],
                layer_nzr=lnzr, depth=cur["depth"],
                offline_risk=self._resolve_risk(cur["depth"]))
            info.update(self._te_out.labels)
            info["te_p_solved"] = self._te_out.features["p_solved"]
            info["te_p_converged"] = self._te_out.features["p_converged"]
        reward = self._reward(self._prev, cur)
        info.update(cur)
        info["rollbacks"] = self.reg.n_rollbacks
        self._prev = cur
        done = self.epoch >= self.total_epochs
        return self._encode(), reward, done, self._mask(), info

    def _reward(self, prev, cur):
        w = self.w
        d_rollback = max(0, self.reg.n_rollbacks - getattr(self, "_last_rb", 0))
        self._last_rb = self.reg.n_rollbacks
        # gate: khi TE báo đã hội tụ, tắt dần thưởng variance/nzr (chống phá mô hình tốt)
        g = self._te_out.gate if self._te_out is not None else {"w1_var": 1.0, "w6_nzr": 1.0}
        # nzr dùng DELTA (cur - prev) như 5 số hạng còn lại, KHÔNG dùng giá trị tuyệt đối.
        # Trước đây `- w6 * cur["nzr"]` phạt cố định mỗi step theo MỨC near-zero-ratio
        # hiện tại (thường ~0.3-0.5, không bao giờ về 0 thật) bất kể agent đang cải
        # thiện hay không -> kéo return âm dai dẳng ngay cả khi policy đã hội tụ tốt
        # (khớp với return_last100 âm quan sát được trên mọi run baseline/telemetry).
        # Delta đúng ngữ nghĩa "thưởng khi nzr giảm, phạt khi nzr tăng" như các số hạng khác.
        r = (g["w1_var"] * w["w1"] * (np.log10(max(cur["sgv"], 1e-12)) - np.log10(max(prev["sgv"], 1e-12)))
             - w["w2"] * (cur["vloss"] - prev["vloss"])
             - w["w3"] * (cur["depth"] - prev["depth"])
             - w["w4"] * (cur["nent"] - prev["nent"])
             - w["w5"] * d_rollback
             - g["w6_nzr"] * w["w6"] * (cur["nzr"] - prev["nzr"]))
        return float(r)
