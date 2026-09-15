"""
Stage 2 — RL Scheduler.

Bọc một agent RL (PPO/DQN) vào interface SchedulerBase, để policy học được điều khiển
kiến trúc mạch ở Stage 3 — đúng vị trí mà Rule-based scheduler đang đứng.

Cùng một agent: train trên Stage3RLEnv (SimulateBackend, nhanh) -> deploy làm scheduler
trên PennyLane Stage 3 (thật). Encoder dùng CHUNG đảm bảo state nhất quán giữa train & deploy.

Mỗi tick, scheduler quét các lớp active, mã hóa state từng lớp, hỏi agent, và chọn
(lớp, macro-action) ưu tiên nhất -> trả MutationRequest (gộp per-layer về 1 quyết định/tick,
khớp interface Stage 2).
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


def weakest_layer_pos(layers: list[dict]) -> int:
    """Vị trí (0-based, trong danh sách truyền vào) của lớp có layer_grad_variance
    thấp nhất -> khả năng dính barren plateau nặng nhất, ưu tiên xét hành động.
    Dùng CHUNG giữa Stage3RLEnv (train) và RLScheduler (deploy) để tránh lệch phân
    phối state giữa hai pha."""
    if not layers:
        return 0
    gvs = [l.get("layer_grad_variance", float("inf")) for l in layers]
    return int(np.argmin(gvs))


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
           (PPO torch trả tuple.) Nếu agent.act() nhận thêm tham số `greedy`
           (PPOAgent/DQNAgent trong RL/PPO, RL/DQN đều hỗ trợ) thì mặc định
           RLScheduler gọi greedy=True: deploy phải quyết định ổn định/tái lập
           được, không nên còn sample ngẫu nhiên (PPO) hay epsilon dư (DQN,
           eps_end mặc định 0.05 không bao giờ về 0) như lúc train.
    """

    def __init__(self, agent, tau: float = 0.1, total_epochs: int = 100,
                 warmup_epochs: int = 10, action_interval_epochs: int = 5,
                 cooldown_epochs: int = 5, act_fn=None, calibration=None,
                 greedy: bool = True):
        """
        calibration: đường dẫn calibration_rules.json (Stage 1C) HOẶC dict đã load.
                     Nếu có -> tau tra ĐỘNG theo cấu hình mạch (n_qubits, topology,
                     cost_type, depth). Nếu None -> dùng `tau` cố định (tương thích cũ).
        greedy: True -> hỏi agent action xác định (argmax) thay vì sample/epsilon.
                Đặt False nếu cố ý muốn deploy stochastic (vd đánh giá exploration).
        """
        super().__init__(warmup_epochs, action_interval_epochs, cooldown_epochs)
        self.agent = agent
        self.tau_default = tau           # fallback khi không tra được
        self.total_epochs = total_epochs
        self.act_fn = act_fn or self._default_act
        self.greedy = greedy
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
    def _default_act(agent, state, mask, greedy=True):
        try:
            out = agent.act(state, mask, greedy=greedy)
        except TypeError:
            # agent tùy biến không nhận tham số `greedy` (vd GreedyPolicy đã tự argmax
            # sẵn ở bên trong) -> gọi kiểu cũ, không phải lỗi.
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

        # Một quyết định/tick, nhìn lớp YẾU NHẤT — khớp đúng cách Stage3RLEnv mã hóa
        # state lúc train (weakest_layer_pos), tránh agent gặp state ngoài phân phối
        # đã học lúc deploy (trước đây: quét TỪNG lớp hỏi agent, agent chưa từng thấy
        # state của lớp không-yếu-nhất lúc train -> lệch phân phối).
        pos = weakest_layer_pos(layers)
        s = encode_layer_state(state.layers, g, tau, pos,
                               state.epoch, self.total_epochs, self._growth_locked)
        a = self.act_fn(self.agent, s, mask, self.greedy)
        action = MACRO_ACTIONS[a]

        if action == A_KEEP:
            return MutationRequest(action=A_KEEP, reason="agent chose keep")
        if action == A_STOP:
            self._growth_locked = True
        target_layer = layers[pos]["layer_id"]
        tgt = target_layer if action == A_PRUNE else None
        return MutationRequest(action=action, target_layer=tgt,
                               reason=f"RL agent (layer {target_layer})")
