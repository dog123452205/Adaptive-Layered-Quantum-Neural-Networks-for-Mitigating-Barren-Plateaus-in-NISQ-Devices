"""
dataset.py — Dataloader cho PennyLaneBackend (Stage 3).

Trả về X đã scale -> [0, pi] (khớp angle encoding RX/RY trong PennyLaneBackend),
y in {0,1}. Hỗ trợ:
  - "iris"  : nhẹ, offline (sklearn bundle) -> chạy thử real backend nhanh.
  - "mnist" : lớp 0/1, PCA -> n_qubits (giống dataset thật của dự án). Lần đầu tải
              qua fetch_openml (cần mạng); subsample cho nhẹ vì PennyLane rất chậm.

Chỉ cần khi backend = pennylane. SimulateBackend không dùng file này.

Ghi chú xuất xứ (2026-09-16): nội dung file này là bản của Lê Hoàng Nam
(github.com/LeNam123456/AL_QNN), KHÔNG phải bản do Lam Vuong tự viết — dùng ở
đây vì đây đúng là code đã tạo ra kết quả cuối tab:ai4i/tab:heart. Bản gốc do
Lam Vuong viết (SMOTE thay vì cyclic partitioned undersampling) được lưu ở
external/rl_theory_improvements/scripts/dataset.py.
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


class BalancedBatchSampler:
    """
    Cyclic Partitioned Undersampling (Epoch-wise Rotating Chunks).
    Giữ nguyên 100% mẫu lỗi (minority), chia mẫu bình thường (majority) thành N chunks.
    Mỗi epoch ghép 1 chunk bình thường mới với toàn bộ mẫu lỗi -> batch cân bằng 50:50.
    Sau ~28-30 epochs, mô hình sẽ quét qua trọn vẹn 100% dữ liệu gốc mà không sinh mẫu ảo!
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.idx_min = np.where(self.y == 1)[0]
        self.idx_maj = np.where(self.y == 0)[0]
        self.n_min = len(self.idx_min)
        self.n_maj = len(self.idx_maj)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.shuffled_maj = self.rng.permutation(self.idx_maj)
        self.n_chunks = max(1, int(np.ceil(self.n_maj / self.n_min)))
        self.current_chunk = 0
        print(f"  [CyclicSampler] Minority (failures): {self.n_min} | Majority (normal): {self.n_maj}")
        print(f"  [CyclicSampler] Divided into {self.n_chunks} chunks (each epoch pairs {self.n_min} failures + {self.n_min} normal = {2*self.n_min} balanced samples)")

    def next_batch(self):
        start = (self.current_chunk * self.n_min) % self.n_maj
        end = start + self.n_min
        if end <= self.n_maj:
            chunk_maj = self.shuffled_maj[start:end]
        else:
            chunk_maj = np.concatenate([self.shuffled_maj[start:], self.shuffled_maj[:end - self.n_maj]])

        batch_idx = np.concatenate([self.idx_min, chunk_maj])
        self.rng.shuffle(batch_idx)

        self.current_chunk = (self.current_chunk + 1) % self.n_chunks
        if self.current_chunk == 0:
            self.shuffled_maj = self.rng.permutation(self.idx_maj)

        return self.X[batch_idx], self.y[batch_idx]


def load_binary_dataset(name: str = "iris", n_qubits: int = 4,
                        val_frac: float = 0.15, test_frac: float = 0.15, max_samples: int | None = None,
                        seed: int = 42, **kwargs) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA

    aliases = {
        "heart": "heart_disease",
        "pima": "pima_diabetes",
        "bcw": "cancer",
        "breast_cancer": "cancer",
        "pm": "predictive_maintenance",
        "ai4i": "predictive_maintenance",
    }
    name = aliases.get(name.lower(), name.lower())

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
        use_cyclic = kwargs.get("cyclic_chunks", True)
        if not use_cyclic:
            # Fallback: Undersample cũ nếu tắt cyclic_chunks
            rng = np.random.default_rng(seed)
            idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
            per = min(len(idx0), len(idx1))
            if max_samples:
                per = min(per, max_samples // 2)
            keep = np.concatenate([rng.choice(idx0, per, replace=False),
                                   rng.choice(idx1, per, replace=False)])
            rng.shuffle(keep)
            X, y = X[keep], y[keep]
        elif max_samples and max_samples < len(X):
            # Nếu có giới hạn max_samples, subsample có phân tầng
            rng = np.random.default_rng(seed)
            idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
            ratio = len(idx1) / len(X)
            n1 = max(10, int(max_samples * ratio))
            n0 = max_samples - n1
            keep = np.concatenate([rng.choice(idx0, min(n0, len(idx0)), replace=False),
                                   rng.choice(idx1, min(n1, len(idx1)), replace=False)])
            rng.shuffle(keep)
            X, y = X[keep], y[keep]
        else:
            print(f"[data] predictive_maintenance: Keeping all {len(X)} original samples (Failures: {np.sum(y==1)}, Normal: {np.sum(y==0)}) for Cyclic Partitioned Undersampling")
    elif name == "pima_diabetes":
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name='diabetes', version=1, as_frame=False, parser='auto')
        X = np.asarray(data.data, dtype=np.float64)
        y_str = np.asarray(data.target)
        y = np.where(y_str == 'tested_positive', 1, 0).astype(np.int64)
        # cân bằng: undersample về bằng lớp thiểu số
        rng = np.random.default_rng(seed)
        idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
        per = min(len(idx0), len(idx1))
        if max_samples:
            per = min(per, max_samples // 2)
        keep = np.concatenate([rng.choice(idx0, per, replace=False),
                               rng.choice(idx1, per, replace=False)])
        rng.shuffle(keep)
        X, y = X[keep], y[keep]
    elif name == "heart_disease":
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name='heart', version=1, as_frame=False, parser='auto')
        raw_x = data.data.toarray() if hasattr(data.data, 'toarray') else data.data
        X = np.asarray(raw_x, dtype=np.float64)
        raw_y = np.asarray(data.target, dtype=np.int64)
        # OpenML heart target: 1=absence (0), 2=presence (1)
        y = np.where(raw_y == 2, 1, 0).astype(np.int64) if np.max(raw_y) == 2 else (raw_y > 0).astype(np.int64)
    elif name == "parkinsons":
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.decomposition import PCA

        data = fetch_openml(name='parkinsons', version=1, as_frame=False, parser='auto')
        raw_x = data.data.toarray() if hasattr(data.data, 'toarray') else data.data
        X = np.asarray(raw_x, dtype=np.float64)
        y_str = np.asarray(data.target)
        # OpenML 1430: '2' là PD (Positive), '1' là Khỏe mạnh (Negative)
        y = np.where(y_str == '2', 1, 0).astype(np.int64)

        # Cân bằng: Random Oversampling
        rng = np.random.default_rng(seed)
        idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
        max_len = max(len(idx0), len(idx1))
        if max_samples:
            max_len = min(max_len, max_samples // 2)

        keep0 = rng.choice(idx0, max_len, replace=True)
        keep1 = rng.choice(idx1, max_len, replace=True)
        keep = np.concatenate([keep0, keep1])
        rng.shuffle(keep)
        X, y = X[keep], y[keep]
    elif name == "german_credit":
        from sklearn.datasets import fetch_openml
        import pandas as pd
        # Fetch ID 31, return dataframe to keep categoricals
        data = fetch_openml(data_id=31, as_frame=True, parser='auto')
        df_X = data.data
        y_str = np.asarray(data.target)
        # Target: 'good' -> 0, 'bad' -> 1
        y = np.where(y_str == 'bad', 1, 0).astype(np.int64)

        # One-hot encode categoricals
        df_X = pd.get_dummies(df_X, drop_first=True)
        X = df_X.to_numpy(float)

        # Cân bằng: Undersampling
        rng = np.random.default_rng(seed)
        idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
        per = min(len(idx0), len(idx1))
        if max_samples:
            per = min(per, max_samples // 2)
        keep = np.concatenate([rng.choice(idx0, per, replace=False),
                               rng.choice(idx1, per, replace=False)])
        rng.shuffle(keep)
        X, y = X[keep], y[keep]
    elif name in ("synthetic_ts", "synthetic_teacher"):
        from src.synthetic_bp.synthetic_data import generate_teacher_student_dataset
        teacher_depth = kwargs.get("teacher_depth", 4)
        topology = kwargs.get("topology", "circular")
        n_samples = max_samples or 500
        return generate_teacher_student_dataset(
            n_qubits=n_qubits, teacher_depth=teacher_depth, n_samples=n_samples,
            topology=topology, val_frac=val_frac, test_frac=test_frac, seed=seed
        )
    else:
        raise ValueError(f"dataset không hỗ trợ: {name!r} "
                         f"(dùng 'iris', 'cancer', 'mnist', 'predictive_maintenance', 'pima_diabetes', 'heart_disease', 'parkinsons', 'german_credit', hoặc 'synthetic_ts')")

    y = y.astype(int)

    # 3-way split
    if test_frac > 0:
        val_test_frac = val_frac + test_frac
        Xtr, X_temp, ytr, y_temp = train_test_split(
            X, y, test_size=val_test_frac, random_state=seed, stratify=y)

        rel_test_frac = test_frac / val_test_frac
        Xva, Xte, yva, yte = train_test_split(
            X_temp, y_temp, test_size=rel_test_frac, random_state=seed, stratify=y_temp)
    else:
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=val_frac, random_state=seed, stratify=y)
        Xte, yte = np.empty((0, X.shape[1])), np.empty((0,))

    # đưa về đúng n_qubits feature (PCA fit trên train), pad nếu thiếu
    if Xtr.shape[1] > n_qubits:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        # Chuẩn hóa Z-score trước PCA
        z_scaler = StandardScaler().fit(Xtr)
        Xtr_scaled = z_scaler.transform(Xtr)
        Xva_scaled = z_scaler.transform(Xva)
        if len(Xte) > 0: Xte_scaled = z_scaler.transform(Xte)

        pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr_scaled)
        Xtr = pca.transform(Xtr_scaled)
        Xva = pca.transform(Xva_scaled)
        if len(Xte) > 0: Xte = pca.transform(Xte_scaled)
    elif Xtr.shape[1] < n_qubits:
        pad = n_qubits - Xtr.shape[1]
        Xtr = np.hstack([Xtr, np.zeros((len(Xtr), pad))])
        Xva = np.hstack([Xva, np.zeros((len(Xva), pad))])
        if len(Xte) > 0: Xte = np.hstack([Xte, np.zeros((len(Xte), pad))])

    sc = MinMaxScaler(feature_range=(0.0, np.pi)).fit(Xtr)  # fit-on-train-only
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    if len(Xte) > 0: Xte = sc.transform(Xte)

    res = {
        "X_train": Xtr.astype(float), "y_train": ytr.astype(float),
        "X_val": Xva.astype(float), "y_val": yva.astype(float),
        "name": name, "n_qubits": n_qubits,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
    }
    if test_frac > 0:
        res["X_test"] = Xte.astype(float)
        res["y_test"] = yte.astype(float)
        res["n_test"] = int(len(Xte))

    if name == "predictive_maintenance" and kwargs.get("cyclic_chunks", True):
        sampler = BalancedBatchSampler(Xtr, ytr, seed=seed)
        res["batch_sampler"] = sampler
        res["X_train_full"] = Xtr.astype(float)
        res["y_train_full"] = ytr.astype(float)
        res["n_train_full"] = int(len(Xtr))
        res["n_chunks"] = sampler.n_chunks
        # Khởi tạo X_train mặc định là batch đầu tiên cân bằng 50:50
        Xtr_init, ytr_init = sampler.next_batch()
        res["X_train"] = Xtr_init
        res["y_train"] = ytr_init
        res["n_train"] = int(len(Xtr_init))

    return res
