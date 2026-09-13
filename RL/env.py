"""
env.py — Môi trường RL (Gym-style) cho AL-QNN Scheduler.

Sequential per-layer: mỗi step agent đứng tại 1 layer, chọn 1 macro-action.
Hết các layer active -> backend train k_epochs -> tính reward (delayed) -> episode mới.

State encoder bám đúng tên cột schema thật. State chiều CỐ ĐỊNH (mỗi step chỉ
nhìn 1 layer + vài chỉ số global), nên depth L thay đổi không làm vỡ vector.
"""
from __future__ import annotations
import numpy as np
from backend import QNNBackend, MACRO_ACTIONS, N_ACTIONS

# Trọng số reward (sweep ở P3)
DEFAULT_W = dict(w1=2.0, w2=3.0, w3=0.05, w4=0.02, w5=1.0, w6=0.5)


STATE_DIM = 16


def _safe_log10(x, eps=1e-12):
    return np.log10(max(float(x), eps))


class ALQNNSchedulerEnv:
    def __init__(self, backend: QNNBackend, k_epochs: int = 5,
                 max_steps_per_episode: int = 64, weights: dict | None = None,
                 total_epochs: int = 100):
        self.b = backend
        self.k_epochs = k_epochs
        self.max_steps = max_steps_per_episode
        self.w = dict(DEFAULT_W if weights is None else weights)
        self.total_epochs = total_epochs
        self.reset()

    # ---- encoding --------------------------------------------------------
    def _encode(self, layer_idx: int) -> np.ndarray:
        layers = self.b.get_layer_metrics()
        g = self.b.get_global_metrics()
        th = self.b.get_thresholds()
        L = max(len(layers), 1)
        li = min(layer_idx, L - 1)
        lyr = layers[li]

        local = [
            _safe_log10(lyr["layer_grad_variance"]),
            lyr["layer_grad_norm"],
            lyr["layer_mean_abs_grad"],
            lyr["layer_near_zero_ratio"],
            lyr["mask_value"],
        ]
        glob = [
            _safe_log10(g["spatial_grad_variance"]),
            g["normalized_grad_norm"],
            g["near_zero_ratio"],
            g["depth"] / 16.0,
            g["n_entangling_gates"] / 64.0,
            g["validation_loss"],
        ]
        ctx = [
            li / L,
            self.epoch / max(self.total_epochs, 1),
            float(getattr(self, "_growth_locked", 0)),
        ]
        gap = [lyr["layer_grad_norm"] - th["tau_empirical"]]
        s = np.array(local + glob + ctx + gap, dtype=np.float32)
        return np.nan_to_num(s, nan=0.0, posinf=10.0, neginf=-10.0)

    def _metrics_vector(self):
        g = self.b.get_global_metrics()
        return dict(
            sgv=g["spatial_grad_variance"],
            vloss=g["validation_loss"],
            depth=g["depth"],
            nent=g["n_entangling_gates"],
            nzr=g["near_zero_ratio"],
        )

    # ---- gym API ---------------------------------------------------------
    def reset(self):
        self.b.reset()
        self.layer_ptr = 0
        self.steps = 0
        self.epoch = 0
        self._growth_locked = False
        self._prev = self._metrics_vector()
        return self._encode(self.layer_ptr), self.action_mask()

    def action_mask(self) -> np.ndarray:
        return self.b.valid_action_mask(self.layer_ptr)

    def step(self, action_idx: int):
        macro = MACRO_ACTIONS[action_idx]
        self.b.submit_mutation(self.layer_ptr, macro)
        if macro == "stop_growth":
            self._growth_locked = True

        L = len(self.b.get_layer_metrics())
        self.layer_ptr += 1
        self.steps += 1
        reward = 0.0
        done = False
        info = {"macro": macro, "mutation_status": "none"}

        # hết các layer active -> train + reward delayed
        if self.layer_ptr >= L:
            out = self.b.step_training(self.k_epochs)
            self.epoch += self.k_epochs
            cur = self._metrics_vector()
            reward = self._reward(self._prev, cur, out)
            info["mutation_status"] = out.get("mutation_status", "none")
            info["rollback"] = out.get("rollback_triggered", 0)
            info.update(cur)
            self._prev = cur
            self.layer_ptr = 0
            if self.epoch >= self.total_epochs or self.steps >= self.max_steps:
                done = True

        next_state = self._encode(self.layer_ptr)
        return next_state, reward, done, self.action_mask(), info

    def _reward(self, prev, cur, out):
        w = self.w
        r = (
            w["w1"] * (_safe_log10(cur["sgv"]) - _safe_log10(prev["sgv"]))
            - w["w2"] * (cur["vloss"] - prev["vloss"])
            - w["w3"] * (cur["depth"] - prev["depth"])
            - w["w4"] * (cur["nent"] - prev["nent"])
            - w["w5"] * out.get("rollback_triggered", 0)
            - w["w6"] * cur["nzr"]
        )
        return float(r)
