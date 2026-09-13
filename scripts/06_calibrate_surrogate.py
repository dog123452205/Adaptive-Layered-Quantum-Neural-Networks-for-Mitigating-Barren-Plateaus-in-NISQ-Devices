"""
Hiệu chỉnh SimulateBackend từ PennyLane THẬT (surrogate calibration).

Quét lưới (depth x entangler_strength) trên PennyLaneBackend, đo loss floor và
gradient variance thật, rồi fit 2 đường cong bằng scipy.least_squares:
    floor(d)     = f_min + (f_max - f_min) * exp(-a * (d - 1))
    log var(d,e) = lv0 - b*d + c*e
Lưu surrogate_fit.json để SimulateBackend nạp (thay công thức bịa tay).

Chạy (cần pennylane):
    python scripts/06_calibrate_surrogate.py --n-qubits 6 --cost-type global ^
        --dataset mnist --max-samples 200 --depths 1,2,3,4,5,6,7,8 ^
        --entanglers 0.4,0.7,1.0 --epochs 40

Chỉ fit lại từ CSV đã thu (không chạy PennyLane):
    python scripts/06_calibrate_surrogate.py --fit-only outputs/surrogate/surrogate_data.csv
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def collect(args):
    from dataset import load_binary_dataset
    from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend

    d = load_binary_dataset(args.dataset, n_qubits=args.n_qubits,
                            max_samples=args.max_samples, seed=42) 
    print(f"[data] {d['name']}: train={d['n_train']} val={d['n_val']}")

    depths = [int(x) for x in args.depths.split(",")]
    rows = []
    for depth in depths:
        b = PennyLaneBackend(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            n_qubits=args.n_qubits, initial_depth=depth, max_depth=depth,
            topology=args.topology, entangler_type=args.entangler,
            cost_type=args.cost_type, seed=args.seed)
        var0 = b.sample_grad_variance(args.var_samples, seed=args.seed)

        best = float("inf")
        for ep in range(args.epochs):
            m = b.train_epoch()
            if args.floor_source == "train":
                best = min(best, m["training_loss"])
            elif (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
                best = min(best, b.evaluate()["validation_loss"])

        ev = b.evaluate()
        if args.floor_source == "val":
            best = min(best, ev["validation_loss"])
        rows.append({"depth": depth, "loss_floor": best,
                     "grad_var": var0, "accuracy": ev["accuracy"]})
        print(f"  depth={depth}: floor={best:.3f} var={var0:.2e} acc={ev['accuracy']:.3f}")
    return pd.DataFrame(rows)


def fit(df):
    depth = df["depth"].to_numpy(float)
    floor = df["loss_floor"].to_numpy(float)
    logvar = np.log(np.clip(df["grad_var"].to_numpy(float), 1e-12, None))

    # floor(d) = f_min + (f_max - f_min) * exp(-a (d-1))
    def r_floor(p):
        f_min, f_max, a = p
        return f_min + (f_max - f_min) * np.exp(-a * (depth - 1)) - floor
    p0 = [floor.min(), floor.max(), 0.3]
    sol_f = least_squares(r_floor, p0, bounds=([0.0, 0.0, 0.01], [0.7, 0.75, 5.0]))
    f_min, f_max, a = sol_f.x

    # log var(d) = lv0 - b*d   (tuyến tính; c giữ 0 vì entangler_strength không nối vào qnode)
    # Bỏ mạch rất nông (depth < var_fit_min_depth) khi fit variance: ở n lớn, mạch
    # 1-2 lớp chưa đủ vướng víu nên phân bố gradient khác chế độ ổn định, kéo lệch fit.
    # Agent train từ initial_depth>=2 và mọc lên nên gần như không hoạt động ở vùng này.
    VAR_FIT_MIN_DEPTH = 3
    m = depth >= VAR_FIT_MIN_DEPTH
    if m.sum() < 3:                       # không đủ điểm -> dùng hết
        m = np.ones_like(depth, dtype=bool)
    A = np.column_stack([np.ones_like(depth[m]), -depth[m]])
    coef, *_ = np.linalg.lstsq(A, logvar[m], rcond=None)
    lv0, b = coef

    fit_dict = {
        "floor": {"f_min": float(f_min), "f_max": float(f_max), "a": float(a)},
        "var": {"lv0": float(lv0), "b": float(b), "c": 0.0,
                "fit_min_depth": int(VAR_FIT_MIN_DEPTH),
                "n_points_used": int(m.sum())},
    }
    floor_pred = f_min + (f_max - f_min) * np.exp(-a * (depth - 1))
    # RMSE variance tính TRÊN vùng đã fit (phản ánh đúng độ sạch ở vùng agent làm việc)
    var_pred_fit = np.exp(lv0 - b * depth[m])
    var_pred_all = np.exp(lv0 - b * depth)
    fit_dict["rmse"] = {
        "floor": float(np.sqrt(np.mean((floor_pred - floor) ** 2))),
        "logvar": float(np.sqrt(np.mean((np.log(var_pred_fit) - logvar[m]) ** 2))),
        "logvar_all_depths": float(np.sqrt(np.mean((np.log(var_pred_all) - logvar) ** 2))),
    }
    return fit_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--cost-type", default="global")
    ap.add_argument("--topology", default="linear")
    ap.add_argument("--entangler", choices=["cnot", "cz"], default="cnot")
    ap.add_argument("--dataset", default="mnist",
                    help="tên dataset trong dataset.py (iris, mnist, cancer, predictive_maintenance)")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--depths", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--var-samples", type=int, default=1,
                    help="số lần init ngẫu nhiên để đo gradient variance (>1 giảm nhiễu, đúng chuẩn BP)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="chỉ evaluate() mỗi N epoch khi floor-source=val (giảm chi phí)")
    ap.add_argument("--floor-source", choices=["val", "train"], default="val",
                    help="val: floor từ validation_loss; train: từ training_loss "
                         "(bỏ hẳn evaluate() trong vòng lặp -> nhanh nhất, hơi lạc quan)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fit-only", default=None, help="CSV đã thu -> chỉ fit lại")
    args = ap.parse_args()

    out_dir = Path(args.out or ROOT / "outputs" / "surrogate")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fit_only:
        df = pd.read_csv(args.fit_only)
    else:
        df = collect(args)
        df.to_csv(out_dir / "surrogate_data.csv", index=False)

    fit_dict = fit(df)
    fit_dict["meta"] = {"n_qubits": args.n_qubits, "cost_type": args.cost_type,
                        "topology": args.topology, "dataset": args.dataset}
    (out_dir / "surrogate_fit.json").write_text(
        json.dumps(fit_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[fit] floor:", fit_dict["floor"])
    print("[fit] var  :", fit_dict["var"])
    print("[fit] rmse :", fit_dict["rmse"])
    print(f"[saved] -> {out_dir / 'surrogate_fit.json'}")


if __name__ == "__main__":
    main()