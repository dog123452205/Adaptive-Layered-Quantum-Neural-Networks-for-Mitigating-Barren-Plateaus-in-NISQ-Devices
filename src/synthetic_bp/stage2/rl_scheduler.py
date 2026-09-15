"""
Stage 2 — RL Scheduler.

Bọc một agent RL (PPO/DQN) vào interface SchedulerBase, để policy học được điều khiển
kiến trúc mạch ở Stage 3 — đúng vị trí mà Rule-based scheduler đang đứng.

Cùng một agent: train trên Stage3RLEnv (SimulateBackend, nhanh) -> deploy làm scheduler
trên PennyLane Stage 3 (thật). Encoder dùng CHUNG đảm bảo state nhất quán giữa train & deploy.

Mỗi tick, scheduler quét các lớp active, mã hóa state từng lớp, hỏi agent, và chọn
(lớp, macro-action) ưu tiên nhất -> trả MutationRequest (gộp per-layer về 1 quyết định/tick,
khớp interface Stage 2).

Ghi chú xuất xứ (2026-09-16): nội dung file này là bản của Lê Hoàng Nam
(github.com/LeNam123456/AL_QNN), KHÔNG phải bản do Lam Vuong tự viết, dù đây
thuộc phần "RL scheduler (PPO/DQN)" ghi nhận là trách nhiệm Lam Vuong trong
bảng nhóm (project_management.tex, tab:team, Approach 4-5). Dùng bản này vì
nó đúng là code đã tạo ra tab:ai4i/tab:heart. Bản gốc do Lam Vuong viết
(weakest-layer targeting, deploy greedy) được lưu ở
external/rl_theory_improvements/src/synthetic_bp/stage2/rl_scheduler.py.
"""
from __future__ import annotations
import numpy as np

from src.synthetic_bp.stage2.scheduler_base import (
    SchedulerBase, RuntimeState, MutationRequest, MACRO_ACTIONS,
    A_KEEP, A_ADD, A_STOP, A_PRUNE, A_REDUCE,
)

STATE_DIM = 16  # khớp agent torch (agent_dqn_torch / agent_ppo_torch)
A_IDX = {a: i for i, a in enumerate(MACRO_ACTIONS)}


def _safe_log10(x, eps=1e-12):
    return float(np.log10(max(float(x), eps)))


def encode_layer_state(layers: list[dict], g: dict, tau: float, layer_idx: int,
                       epoch: int, total_epochs: int, growth_locked: bool) -> np.ndarray:
    """RuntimeState (cho 1 lớp) -> vector 16 chiều. Dùng chung train & deploy."""
    L = max(len(layers), 1)
    i = min(layer_idx, L - 1)
    lyr = layers[i]
    gvs = np.array([x.get("layer_grad_variance", 0.0) for x in layers])
    med = float(np.median(gvs)) if len(gvs) else 0.0
    rel_weak = 1.0 if lyr.get("layer_grad_variance", 1e9) <= med else 0.0

    n_params = max(g.get("n_params", L * 12), 1)
    norm_gn = g.get("grad_norm", 0.0) / np.sqrt(n_params)

    local = [
        _safe_log10(lyr.get("layer_grad_variance", 0.0)),
        lyr.get("layer_grad_norm", 0.0),
        lyr.get("layer_mean_abs_grad", 0.0),
        lyr.get("layer_near_zero_ratio", 0.0),
        lyr.get("mask_value", 1.0),
        rel_weak,
    ]
    glob = [
        _safe_log10(g.get("spatial_grad_variance", 0.0)),
        norm_gn,
        g.get("near_zero_ratio", 0.0),
        g.get("depth", L) / 16.0,
        g.get("n_entangling_gates", 0) / 64.0,
        g.get("validation_loss", 0.69),
    ]
    ctx = [i / L, epoch / max(total_epochs, 1), float(growth_locked)]
    gap = [lyr.get("layer_grad_norm", 0.0) - tau]
    s = np.array(local + glob + ctx + gap, dtype=np.float32)
    return np.nan_to_num(s, nan=0.0, posinf=10.0, neginf=-10.0)


class RLScheduler(SchedulerBase):
    """
    Scheduler điều khiển bằng agent RL.

    agent: bất kỳ đối tượng nào có .act(state, mask) -> action_idx (DQN/PPO).
           (PPO torch trả tuple; truyền greedy_fn nếu cần lấy action xác định.)
    """

    def __init__(self, agent, tau: float = 0.1, total_epochs: int = 100,
                 warmup_epochs: int = 10, action_interval_epochs: int = 5,
                 cooldown_epochs: int = 5, act_fn=None, calibration=None):
        """
        calibration: đường dẫn calibration_rules.json (Stage 1C) HOẶC dict đã load.
                     Nếu có -> tau tra ĐỘNG theo cấu hình mạch (n_qubits, topology,
                     cost_type, depth). Nếu None -> dùng `tau` cố định (tương thích cũ).
        """
        super().__init__(warmup_epochs, action_interval_epochs, cooldown_epochs)
        self.agent = agent
        self.tau_default = tau           # fallback khi không tra được
        self.total_epochs = total_epochs
        self.act_fn = act_fn or self._default_act
        self._growth_locked = False
        self._calibration = self._load_calibration(calibration)

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

    def _resolve_tau(self, state) -> float:
        """Tra tau_empirical thật từ Stage 1C theo cấu hình hiện tại; else tau_default."""
        if self._calibration is None:
            return self.tau_default
        try:
            from src.synthetic_bp.stage1c.calibration import lookup_rule
            rule = lookup_rule(self._calibration, state.n_qubits,
                               state.entanglement_topology, state.cost_type, state.depth)
            return float(rule["thresholds"].get("tau_empirical", self.tau_default))
        except Exception:
            gd = self._calibration.get("global_default", {})
            return float(gd.get("tau_empirical", self.tau_default))

    @staticmethod
    def _default_act(agent, state, mask):
        out = agent.act(state, mask)
        return int(out[0]) if isinstance(out, tuple) else int(out)

    def reset(self) -> None:
        self._growth_locked = False

    def _valid_mask(self, layers, depth, min_depth=1, max_depth=16):
        m = np.ones(len(MACRO_ACTIONS), dtype=bool)
        if depth >= max_depth or self._growth_locked:
            m[A_IDX[A_ADD]] = False
        if depth <= min_depth:
            m[A_IDX[A_PRUNE]] = False
        return m

    def decide(self, state: RuntimeState) -> MutationRequest:
        layers = [l for l in state.layers
                  if l.get("layer_status", "active") == "active"]
        if not layers:
            return MutationRequest(action=A_KEEP, reason="no active layer")
        g = {
            "spatial_grad_variance": state.spatial_grad_variance,
            "grad_norm": state.grad_norm,
            "near_zero_ratio": state.near_zero_ratio,
            "depth": state.depth,
            "n_entangling_gates": state.n_entangling_gates,
            "validation_loss": state.validation_loss,
            "n_params": state.depth * state.n_qubits * 3,
        }
        depth = len(layers)
        mask = self._valid_mask(state.layers, depth)
        tau = self._resolve_tau(state)   # τ thật từ Stage 1C (hoặc fallback)

        # Quét từng lớp, hỏi agent; chọn quyết định "mạnh nhất" khác keep.
        best = None  # (priority, layer_id, action_idx)
        for li, lyr in enumerate(layers):
            s = encode_layer_state(state.layers, g, tau, li,
                                   state.epoch, self.total_epochs, self._growth_locked)
            a = self.act_fn(self.agent, s, mask)
            if MACRO_ACTIONS[a] == A_KEEP:
                continue
            # ưu tiên: lớp yếu nhất (grad_variance thấp) được ưu tiên hành động
            priority = -lyr.get("layer_grad_variance", 0.0)
            if best is None or priority > best[0]:
                best = (priority, lyr["layer_id"], a)

        if best is None:
            return MutationRequest(action=A_KEEP, reason="agent chose keep")

        _, target_layer, a = best
        action = MACRO_ACTIONS[a]
        if action == A_STOP:
            self._growth_locked = True
        tgt = target_layer if action in (A_PRUNE,) else None
        return MutationRequest(action=action, target_layer=tgt,
                               reason=f"RL agent (layer {target_layer})")
