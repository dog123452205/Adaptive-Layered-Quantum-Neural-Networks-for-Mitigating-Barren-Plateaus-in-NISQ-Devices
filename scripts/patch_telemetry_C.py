"""
PATCH C — Nối Telemetry Engine vào REWARD (không đổi STATE_DIM).

Mục tiêu: gate["w1_var"] và gate["w6_nzr"] nhân vào term variance & nzr của
reward, để KHÔNG thưởng bơm-variance khi mô hình đã giải xong (solved).
Đây là ablation: bật/tắt bằng cờ use_telemetry -> so sánh có/không.

KHÔNG đổi STATE_DIM (state vẫn 16-dim) -> không phải sửa mạng, không regress
bug shape mismatch. Chỉ đổi cách tính reward.

CÁCH ÁP DỤNG: 4 sửa nhỏ trong src/synthetic_bp/stage3/rl_env.py

────────────────────────────────────────────────────────────────
[1] import + thêm tham số use_telemetry vào __init__
────────────────────────────────────────────────────────────────
Đầu file, thêm:

    from src.synthetic_bp.stage3.telemetry_engine import TelemetryEngine

Trong __init__(...), thêm tham số (cuối danh sách, mặc định False = giữ hành vi cũ):

    def __init__(self, ..., surrogate=None, use_telemetry=False):
        ...
        self.use_telemetry = use_telemetry
        self.te = TelemetryEngine() if use_telemetry else None
        ...
        self.reset()

────────────────────────────────────────────────────────────────
[2] reset() — reset telemetry cùng môi trường
────────────────────────────────────────────────────────────────
Trong reset(), sau khi tạo self.eng, thêm:

        if self.te is not None:
            self.te.reset()

────────────────────────────────────────────────────────────────
[3] _metrics() — bổ sung layer_nzr + offline_risk cho telemetry
────────────────────────────────────────────────────────────────
Thay _metrics() bằng:

    def _metrics(self):
        g = self.b.global_metrics()
        layers = self.b.layer_metrics()
        layer_nzr = [l.get("layer_near_zero_ratio", 0.0) for l in layers]
        # offline_risk: khoảng cách variance tới tau (từ Stage 1C). Nếu không có,
        # dùng 0.5 (trung tính). tau càng gần variance -> rủi ro BP càng cao.
        sgv = g["spatial_grad_variance"]
        offline_risk = 0.5
        if getattr(self, "tau", None):
            offline_risk = float(np.clip(self.tau / max(sgv, 1e-12), 0.0, 1.0))
        return dict(sgv=sgv,
                    vloss=self.b.evaluate()["validation_loss"],
                    depth=g["depth"], nent=g["n_entangling_gates"],
                    nzr=g["near_zero_ratio"],
                    layer_nzr=layer_nzr, offline_risk=offline_risk)

────────────────────────────────────────────────────────────────
[4] _reward() — nhân gate vào term w1 (variance) và w6 (nzr)
────────────────────────────────────────────────────────────────
Thay _reward() bằng:

    def _reward(self, prev, cur):
        w = self.w
        d_rollback = max(0, self.reg.n_rollbacks - getattr(self, "_last_rb", 0))
        self._last_rb = self.reg.n_rollbacks

        # ---- telemetry gate (mặc định = 1.0 nếu tắt) ----
        gate_var, gate_nzr = 1.0, 1.0
        if self.te is not None:
            out = self.te.step(
                sgv=cur["sgv"], nzr=cur["nzr"], vloss=cur["vloss"],
                layer_nzr=cur.get("layer_nzr", [cur["nzr"]]),
                depth=cur["depth"], offline_risk=cur.get("offline_risk", 0.5))
            gate_var = out.gate["w1_var"]     # ~0 khi đã solved -> tắt thưởng variance
            gate_nzr = out.gate["w6_nzr"]
            self._last_te = out              # để log/vẽ nếu cần

        r = (w["w1"] * gate_var * (np.log10(max(cur["sgv"], 1e-12)) - np.log10(max(prev["sgv"], 1e-12)))
             - w["w2"] * (cur["vloss"] - prev["vloss"])
             - w["w3"] * (cur["depth"] - prev["depth"])
             - w["w4"] * (cur["nent"] - prev["nent"])
             - w["w5"] * d_rollback
             - w["w6"] * gate_nzr * cur["nzr"])
        return float(r)

────────────────────────────────────────────────────────────────
[5] TRUYỀN cờ từ script train (04_train_rl_scheduler.py)
────────────────────────────────────────────────────────────────
Thêm argparse:  ap.add_argument("--use-telemetry", action="store_true")
Khi tạo env:    Stage3RLEnv(..., use_telemetry=args.use_telemetry)

────────────────────────────────────────────────────────────────
LƯU Ý QUAN TRỌNG
────────────────────────────────────────────────────────────────
- STATE_DIM KHÔNG đổi -> checkpoint cũ vẫn nạp được, mạng không đổi.
- use_telemetry=False -> gate=1.0 -> reward y HỆT bản cũ (kiểm chứng: chạy
  1 config với False, so seed với kết quả cũ, phải khớp tới chữ số).
- Chỉ khi =True mới đổi hành vi -> đó là nhánh ablation.
- offline_risk ở đây là xấp xỉ (tau/sgv). Nếu Stage 1C cho tau đúng ở n=12
  thì tốt; nếu không, telemetry vẫn chạy với risk trung tính.
"""
