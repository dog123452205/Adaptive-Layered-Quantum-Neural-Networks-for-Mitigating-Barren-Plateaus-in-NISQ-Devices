"""
dataset.py — Dataloader cho PennyLaneBackend (Stage 3).

Trả về X đã scale -> [0, pi] (khớp angle encoding RX/RY trong PennyLaneBackend),
y in {0,1}. Hỗ trợ:
  - "iris"  : nhẹ, offline (sklearn bundle) -> chạy thử real backend nhanh.
  - "mnist" : lớp 0/1, PCA -> n_qubits (giống dataset thật của dự án). Lần đầu tải
              qua fetch_openml (cần mạng); subsample cho nhẹ vì PennyLane rất chậm.

Chỉ cần khi backend = pennylane. SimulateBackend không dùng file này.

Split: mặc định VẪN 2 chiều train/val (test_frac=0.0) — KHÔNG đổi hành vi pipeline
RL hiện tại (04_train_rl_scheduler.py chỉ cần X_train/X_val để train + cho scheduler
đánh giá accept/rollback mutation). Truyền test_frac>0 để có thêm X_test/y_test —
tách RIÊNG, seed cố định, KHÔNG lọt vào train hay val -> dùng cho việc EXPORT số liệu
demo (export_predictions.py), vì X_val vẫn được scheduler nhìn thấy liên tục lúc
train (quyết định accept/rollback), không phải "chưa từng thấy" theo đúng nghĩa test.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _find_csv(fname: str) -> Path:
    here = Path(__file__).resolve().parent
    for c in [here.parent / "data" / fname, here.parent / fname,
              here / fname, Path.cwd() / fname]:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Không thấy {fname}. Đặt file vào {here.parent / 'data' / fname} "
        f"(tải từ UCI AI4I 2020 Predictive Maintenance hoặc Kaggle).")


def load_binary_dataset(name: str = "iris", n_qubits: int = 4,
                        val_frac: float = 0.3, test_frac: float = 0.0,
                        max_samples: int | None = None,
                        seed: int = 42, smote: bool = True) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.decomposition import PCA

    if name == "iris":
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        m = y < 2                       # binary subset: lớp 0 & 1
        X, y = X[m], y[m]
    elif name == "cancer":
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)   # 30 feature, nhị phân ác/lành
        if max_samples and len(X) > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=max_samples, replace=False)
            X, y = X[idx], y[idx]
    elif name == "mnist":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist.data.astype(float)
        y = mnist.target.astype(int)
        m = (y == 0) | (y == 1)
        X, y = X[m], y[m]
        n = max_samples or 400          # PennyLane chậm -> giới hạn mẫu
        if len(X) > n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=n, replace=False)
            X, y = X[idx], y[idx]
    elif name == "predictive_maintenance":
        import pandas as pd
        df = pd.read_csv(_find_csv("predictive_maintenance.csv"))
        num = ["Air temperature [K]", "Process temperature [K]",
               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        Xnum = df[num].to_numpy(float)
        type_oh = pd.get_dummies(df["Type"]).to_numpy(float)  # L/M/H -> 3 cột
        X = np.hstack([Xnum, type_oh])                        # 8 feature -> PCA linh hoạt
        y = df["Target"].to_numpy(int)
        # data gốc chỉ 3.4% lỗi (9661 vs 339). KHÔNG undersample majority nữa —
        # vứt ~96% dữ liệu thật chỉ để cân bằng là quá lãng phí. Giữ nguyên tỉ lệ
        # gốc ở đây (subsample có stratify nếu cần giới hạn max_samples); cân bằng
        # lớp thiểu số bằng SMOTE CHỈ trên train, sau khi split + scale (khối SMOTE
        # phía dưới) để val/test vẫn phản ánh đúng phân phối thật.
        if max_samples and max_samples < len(y):
            X, _, y, _ = train_test_split(
                X, y, train_size=max_samples, random_state=seed, stratify=y)

    elif name == "german_credit":
        import pandas as pd
        from pathlib import Path
        csv_path = Path(r"C:\Users\ACER\Documents\PROJECT\KLTN\al_qnn_project\data\german_credit.csv")
        df = pd.read_csv(csv_path)
        feat = [f"f{i}" for i in range(8)]
        X = df[feat].to_numpy(float)
        y = df["Target"].to_numpy(int)
        if max_samples and max_samples < len(y):
            rng = np.random.default_rng(seed)
            idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
            per = min(len(idx0), len(idx1), max_samples // 2)
            keep = np.concatenate([rng.choice(idx0, per, replace=False),
                                   rng.choice(idx1, per, replace=False)])
            rng.shuffle(keep)
            X, y = X[keep], y[keep]
    elif name == "pima":
        import pandas as pd
        df = pd.read_csv(_find_csv("pima.csv"))
        feat = [f"f{i}" for i in range(8)]        # 8 feature 1-to-1, KHÔNG PCA
        X = df[feat].to_numpy(float)
        y = df["Target"].to_numpy(int)
        if max_samples and max_samples < len(y):
            rng = np.random.default_rng(seed)
            idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
            per = min(len(idx0), len(idx1), max_samples // 2)
            keep = np.concatenate([rng.choice(idx0, per, replace=False),
                                   rng.choice(idx1, per, replace=False)])
            rng.shuffle(keep); X, y = X[keep], y[keep]

    elif name == "heart":
        import pandas as pd
        df = pd.read_csv(_find_csv("heart.csv"))
        feat = [f"f{i}" for i in range(8)]
        X = df[feat].to_numpy(float)
        y = df["Target"].to_numpy(int)
        if max_samples and max_samples < len(y):
            rng = np.random.default_rng(seed)
            idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
            per = min(len(idx0), len(idx1), max_samples // 2)
            keep = np.concatenate([rng.choice(idx0, per, replace=False),
                                   rng.choice(idx1, per, replace=False)])
            rng.shuffle(keep); X, y = X[keep], y[keep]

    elif name == "parkinsons":
        import pandas as pd
        df = pd.read_csv(_find_csv("parkinsons.csv"))
        feat = [f"f{i}" for i in range(8)]
        X = df[feat].to_numpy(float)
        y = df["Target"].to_numpy(int)
        # tập nhỏ -> thường KHÔNG cần max_samples, để nguyên
        if max_samples and max_samples < len(y):
            rng = np.random.default_rng(seed)
            idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
            per = min(len(idx0), len(idx1), max_samples // 2)
            keep = np.concatenate([rng.choice(idx0, per, replace=False),
                                   rng.choice(idx1, per, replace=False)])
            rng.shuffle(keep); X, y = X[keep], y[keep]
    else:
        raise ValueError(f"dataset không hỗ trợ: {name!r} ")

    

    y = y.astype(int)

    if test_frac and test_frac > 0:
        # Tách TEST trước, cố định theo seed -> không đụng lại trong suốt training
        # (không góp gradient, không được scheduler nhìn để quyết định accept/rollback).
        X_fit, X_test, y_fit, y_test = train_test_split(
            X, y, test_size=test_frac, random_state=seed, stratify=y)
        # val_frac vẫn giữ nghĩa "tỉ lệ trên TOÀN BỘ dữ liệu gốc" như trước khi có
        # test_frac -> quy đổi lại thành tỉ lệ trên phần CÒN LẠI (sau khi bớt test).
        rel_val = min(max(val_frac / max(1e-9, 1.0 - test_frac), 0.0), 0.9)
        Xtr, Xva, ytr, yva = train_test_split(
            X_fit, y_fit, test_size=rel_val, random_state=seed, stratify=y_fit)
    else:
        X_test, y_test = np.empty((0, X.shape[1])), np.empty((0,), dtype=int)
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=val_frac, random_state=seed, stratify=y)

    has_test = len(X_test) > 0

    # đưa về đúng n_qubits feature (PCA fit trên train), pad nếu thiếu
    if Xtr.shape[1] > n_qubits:
        # Z-score TRƯỚC PCA (fit-on-train): để PCA không bị biến scale lớn
        # (vd Credit Amount) lấn át các biến one-hot / scale nhỏ.
        zsc = StandardScaler().fit(Xtr)
        Xtr, Xva = zsc.transform(Xtr), zsc.transform(Xva)
        if has_test:
            X_test = zsc.transform(X_test)
        pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr)
        Xtr, Xva = pca.transform(Xtr), pca.transform(Xva)
        if has_test:
            X_test = pca.transform(X_test)
    elif Xtr.shape[1] < n_qubits:
        pad = n_qubits - Xtr.shape[1]
        Xtr = np.hstack([Xtr, np.zeros((len(Xtr), pad))])
        Xva = np.hstack([Xva, np.zeros((len(Xva), pad))])
        if has_test:
            X_test = np.hstack([X_test, np.zeros((len(X_test), pad))])

    sc = MinMaxScaler(feature_range=(0.0, np.pi)).fit(Xtr)  # fit-on-train-only
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    if has_test:
        X_test = sc.transform(X_test)

    if name == "predictive_maintenance" and smote:
        # Oversample CHỈ train bằng SMOTE, nội suy trong không gian đã PCA/pad +
        # scale-[0,pi] (khoảng cách KNN có ý nghĩa, mẫu synthetic tự nằm trong
        # [0,pi] vì là tổ hợp lồi của hàng xóm cùng lớp) — KHÔNG đụng X_val/X_test.
        # smote=False (chỉ dùng để so sánh/minh hoạ, xem data/*.ipynb) trả lại
        # đúng data gốc mất cân bằng ở bước này, mọi thứ khác giữ nguyên.
        from imblearn.over_sampling import SMOTE
        n_minority = min(np.bincount(ytr, minlength=2))
        if n_minority >= 2:
            k = min(5, n_minority - 1)
            Xtr, ytr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xtr, ytr)

    return {
        "X_train": Xtr.astype(float), "y_train": ytr.astype(float),
        "X_val": Xva.astype(float), "y_val": yva.astype(float),
        "X_test": X_test.astype(float), "y_test": y_test.astype(float),
        "name": name, "n_qubits": n_qubits,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(X_test)),
    }