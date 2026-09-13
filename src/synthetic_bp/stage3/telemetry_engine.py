"""
Stage 3 — Telemetry Engine (chẩn đoán mềm 5+1 trạng thái).

Thay logic cứng "grad nhỏ => Barren Plateau" bằng nhiều điểm tin cậy liên tục,
cross-check giữa spatial_grad_variance, validation_loss, near_zero_ratio (toàn cục
và theo lớp) và offline_risk từ Stage 1C. Chỉ dùng tín hiệu backend ĐÃ có sẵn.

Ba đầu ra, ba mục đích:
  .features     -> vector liên tục (đưa vào state RL nếu muốn, KHÔNG one-hot)
  .labels       -> nhãn cứng multi-label, để log vào training_log + vẽ hình luận văn
  .mask/.gate   -> guardrail an toàn cho scheduler và hệ số nhân reward

Ví dụ:
    te = TelemetryEngine()
    out = te.step(sgv=3.8e-6, nzr=0.4, vloss=0.30, layer_nzr=[0.2,0.9], depth=6,
                  offline_risk=0.3)
    print(out.labels["is_converged"], out.gate["w1_var"])
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np


def _sig(x, T=1.0):
    z = np.clip(np.asarray(x, float) / max(T, 1e-9), -60, 60)
    out = 1.0 / (1.0 + np.exp(-z))
    return float(out) if out.ndim == 0 else out


@dataclass
class TeleConfig:
    drop_decades: float = 1.0     # sgv tụt >= ngần này decade so với đầu episode = sụp đổ
    T_grad: float = 0.30
    loss_target: float = 0.35     # vloss dưới mức này coi như đã giải xong (acc ~0.5+)
    T_loss: float = 0.05
    stall_window: int = 5
    stall_eps: float = 2e-3       # |slope vloss| dưới mức này = đi ngang
    dead_nzr: float = 0.85        # per-layer nzr trên mức này = lớp chết
    global_nzr_threshold: float = 0.7 # [Fix 1] nzr toàn cục trên mức này coi như gradient bão hòa
    stall_scale: float = 5.0      # [Fix 3] hệ số khuếch đại cho hàm sigmoid của stall
    ema: float = 0.6
    dwell: int = 2                # phải giữ >0.5 liên tiếp k lần mới bật nhãn cứng


@dataclass
class TeleOutput:
    features: dict
    labels: dict
    mask: dict            # tên action -> True nếu được phép
    gate: dict            # tên hệ số reward -> nhân [0,1]
    dead_layer_ids: list


class TelemetryEngine:
    def __init__(self, cfg: TeleConfig | None = None):
        self.cfg = cfg or TeleConfig()
        self.reset()

    def reset(self):
        self.sgv0 = None
        self.last_depth = None    # [Fix 2] Theo dõi sự thay đổi kiến trúc
        self.loss_hist = deque(maxlen=self.cfg.stall_window)
        self._ema = {}
        self._streak = {}

    def _smooth(self, k, v):
        a = self.cfg.ema
        cur = a * self._ema.get(k, v) + (1 - a) * v
        self._ema[k] = cur
        return cur

    def _hard(self, k, s):
        self._streak[k] = self._streak.get(k, 0) + 1 if s > 0.5 else 0
        return self._streak[k] >= self.cfg.dwell

    def step(self, sgv, nzr, vloss, layer_nzr, depth, offline_risk=0.5):
        c = self.cfg
        sgv = max(float(sgv), 1e-30)
        
        # [Fix 2] Điểm neo động: Reset sgv0 nếu kiến trúc (depth) thay đổi
        if self.sgv0 is None or self.last_depth != depth:
            self.sgv0 = sgv
            self.last_depth = depth
            
        g_log = float(np.log10(sgv / self.sgv0))          # <0 = variance tụt so với đầu

        # loss đi ngang?
        self.loss_hist.append(float(vloss))
        if len(self.loss_hist) >= 3:
            y = np.array(self.loss_hist)
            slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
        else:
            slope = -1.0
            
        # [Fix 3] Khuếch đại giá trị đỉnh của s_stall để tiến gần 1.0 khi slope = 0
        s_stall = _sig(-c.stall_scale * (abs(slope) - c.stall_eps) / max(c.stall_eps, 1e-9), 1.0)

        # mệnh đề cơ sở
        # [Fix 1] Kết hợp SGV và NZR toàn cục để xác định sụp đổ
        s_collapse_sgv = _sig(-(g_log + c.drop_decades), c.T_grad)
        s_collapse_nzr = _sig((float(nzr) - c.global_nzr_threshold) / 0.1, 1.0)
        s_collapse = float(np.maximum(s_collapse_sgv, s_collapse_nzr))
        
        s_solved = _sig((c.loss_target - vloss), c.T_loss)       # giải xong?

        ln = np.asarray(layer_nzr, float).ravel()
        dead_ids = np.where(ln > c.dead_nzr)[0].tolist()
        p_dead = float(np.max(_sig((ln - c.dead_nzr) / 0.05, 1.0))) if ln.size else 0.0

        # tổ hợp multi-label (KHÔNG loại trừ nhau).
        # solved: đã giải xong bài toán (dùng cho gate + guardrail an toàn).
        # converged: định nghĩa chặt theo doc = grad sụp đổ HỢP LÝ khi đã solved (để log/vẽ).
        raw = dict(
            solved=s_solved,
            converged=s_collapse * s_solved,
            true_bp=s_collapse * (1 - s_solved) * offline_risk,
            local_min=(1 - s_solved) * s_stall * (1 - s_collapse),
            ambiguous=s_collapse * (1 - s_solved) * (1 - offline_risk),
            dead_layer=p_dead,
        )
        sc = {k: self._smooth(k, v) for k, v in raw.items()}
        labels = {f"is_{k}": self._hard(k, v) for k, v in sc.items()}
        labels["dead_layers"] = dead_ids

        features = dict(g_log=g_log, nzr=float(nzr), loss_slope=slope,
                        offline_risk=float(offline_risk),
                        **{f"p_{k}": v for k, v in sc.items()})

        # guardrail: đừng phá mô hình ĐÃ GIẢI XONG (khóa vào solved, không cần grad sụp đổ)
        mask = {
            "add_layer": not labels["is_solved"],
            "propose_prune_layer": not labels["is_solved"],
            # [Fix 4] Mở khóa hành động giảm vướng víu ở Local Minima để giúp mô hình thoát bẫy
            "propose_reduce_entanglement": not labels["is_solved"],
            "keep": True, "stop_growth": True,
        }
        # tắt dần thưởng variance/nzr khi đã giải xong (chống bơm variance phá mô hình tốt)
        gate = {"w1_var": 1.0 - sc["solved"], "w6_nzr": 1.0 - sc["solved"]}
        return TeleOutput(features, labels, mask, gate, dead_ids)


if __name__ == "__main__":
    scen = {
        "converged":  dict(sgv=1e-7, vloss=0.15, nzr=0.3, ln=[0.3], risk=0.2),
        "true_bp":    dict(sgv=1e-7, vloss=0.60, nzr=0.9, ln=[0.9], risk=0.9),
        "local_min":  dict(sgv=4e-6, vloss=0.58, nzr=0.2, ln=[0.1], risk=0.2),
        "ambiguous":  dict(sgv=1e-7, vloss=0.60, nzr=0.9, ln=[0.9], risk=0.1),
    }
    for name, s in scen.items():
        te = TelemetryEngine()
        for ep in range(6):
            sgv = 4e-6 if ep == 0 else s["sgv"]
            out = te.step(sgv, s["nzr"], s["vloss"], s["ln"], depth=6, offline_risk=s["risk"])
        top = sorted(((v, k) for k, v in out.features.items() if k.startswith("p_")), reverse=True)[0]
        print(f"{name:11s} -> {top[1]:14s}={top[0]:.2f} | "
              f"ADD={out.mask['add_layer']} PRUNE={out.mask['propose_prune_layer']} "
              f"gate_var={out.gate['w1_var']:.2f}")
