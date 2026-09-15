"""
Stage 3 — Circuit Backend.

Trừu tượng hóa mạch lượng tử để Trainer/MutationEngine không phụ thuộc trực tiếp
PennyLane. Hai hiện thực:

  - SimulateBackend  : numpy thuần, mô phỏng động lực huấn luyện + BP. Chạy được NGAY,
                       không cần pennylane -> dùng để test orchestration (mutation/registry/scheduler).
  - PennyLaneBackend : mạch THẬT, tái dùng stage1a/ansatz.py (apply_hea). Chạy ở máy có pennylane.

Cả hai giữ TOÀN BỘ trạng thái model (params, depth, mask theo lớp, cường độ entangler)
để Parameter Registry chụp/khôi phục được.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import copy
import numpy as np


class CircuitBackend(ABC):
    """Hợp đồng chung cho mọi backend mạch ở Stage 3."""

    n_qubits: int
    depth: int

    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def train_epoch(self) -> dict[str, Any]: ...
    @abstractmethod
    def evaluate(self) -> dict[str, Any]: ...
    @abstractmethod
    def layer_metrics(self) -> list[dict[str, Any]]: ...
    @abstractmethod
    def global_metrics(self) -> dict[str, Any]: ...
    @abstractmethod
    def add_layer(self) -> None: ...
    @abstractmethod
    def set_layer_mask(self, layer_id: int, alpha: float) -> None: ...
    @abstractmethod
    def remove_layer(self, layer_id: int) -> None: ...
    @abstractmethod
    def set_entangler_strength(self, alpha: float) -> None: ...
    @abstractmethod
    def snapshot(self) -> dict[str, Any]: ...
    @abstractmethod
    def restore(self, snap: dict[str, Any]) -> None: ...


# ===========================================================================
class SimulateBackend(CircuitBackend):
    """
    Mô phỏng động lực học (KHÔNG cần pennylane).

    Quy luật bám vật lý định tính:
      - mỗi lớp có 'quality' ngẫu nhiên; lớp khỏe giúp giảm loss.
      - phương sai gradient suy giảm theo độ sâu (Barren Plateau): var ~ base * 2^(-depth/k).
      - mask alpha<1 làm lớp đóng góp ít hơn (soft-mask).
      - cường độ entangler thấp -> giảm vướng víu -> bớt BP nhưng cũng bớt capacity.
    """

    def __init__(self, n_qubits=4, initial_depth=2, max_depth=16,
                 cost_type="local", topology="linear", seed=0, surrogate=None):
        self.n_qubits = n_qubits
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.cost_type = cost_type
        self.topology = topology
        self.seed = seed
        self._fit = self._load_surrogate(surrogate)
        if self._fit:
            fit_nq = self._fit.get("meta", {}).get("n_qubits")
            if fit_nq is not None and fit_nq != n_qubits:
                import warnings
                warnings.warn(
                    f"surrogate fit ở n_qubits={fit_nq} nhưng backend n_qubits={n_qubits}. "
                    f"Landscape BP không khớp -> nên calibrate lại đúng qubit.",
                    stacklevel=2)
        self.reset()

    @staticmethod
    def _load_surrogate(surrogate):
        """Nạp hệ số surrogate (fit từ PennyLane thật) từ path/dict. None -> dùng công thức mặc định."""
        if surrogate is None:
            return None
        if isinstance(surrogate, (str, Path)):
            import json
            p = Path(surrogate)
            if not p.exists():
                return None
            surrogate = json.loads(p.read_text(encoding="utf-8"))
        return surrogate

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.depth = self.initial_depth
        self.entangler_strength = 1.0
        self.epoch = 0
        # mỗi lớp: theta (params giả), mask alpha, quality
        self.layers = [self._new_layer(i) for i in range(self.depth)]
        self._val_loss = 0.69 + 0.02 * self.rng.standard_normal()
        self._train_loss = self._val_loss + 0.02

    def _new_layer(self, lid):
        return {
            "layer_id": lid,
            "theta": self.rng.uniform(-np.pi, np.pi, size=(self.n_qubits, 3)),
            "mask": 1.0,
            "quality": float(np.clip(0.5 + 0.3 * self.rng.standard_normal(), 0.05, 1.0)),
        }

    # ---- động lực học mô phỏng ----
    def _n_qubits_decay(self) -> float:
        """Hệ số BP theo SỐ QUBIT — trước đây _trainability() không phụ thuộc n_qubits
        chút nào (chỉ depth/cost_type), sai với lý thuyết: BP còn nặng theo n_qubits,
        đặc biệt với cost toàn cục (McClean et al. 2018: Var ~ 2^-n, mũ theo n_qubits).
        Cost cục bộ ít bị ảnh hưởng bởi n_qubits hơn nhiều (Cerezo et al. 2021:
        cost-function-dependent BP, chỉ suy giảm đa thức ~1/sqrt(n)).
        Chuẩn hóa để n_qubits=8 -> hệ số=1.0 (giữ nguyên mọi cấu hình 8-qubit đã hiệu
        chỉnh trước đó, ví dụ K=0.28/p=0.75 trong train_epoch())."""
        n = max(self.n_qubits, 1)
        if self.cost_type == "global":
            return float(2.0 ** (-(n - 8)))
        return float((8.0 / n) ** 0.5)

    def _trainability(self) -> float:
        """Phương sai gradient hữu hiệu (fallback heuristic), giảm theo depth, n_qubits, cost."""
        base = 0.05
        bp_decay = 2.0 ** (-(self.depth * (2 if self.cost_type == "global" else 1)) / 4.0)
        ent = 0.4 + 0.6 * self.entangler_strength
        return base * bp_decay * self._n_qubits_decay() * ent

    def _spatial_var(self) -> float:
        """Phương sai gradient toàn cục. Ưu tiên surrogate fit từ PennyLane thật."""
        if self._fit and "var" in self._fit:
            v = self._fit["var"]
            return float(np.exp(v["lv0"] - v["b"] * self.depth
                                + v["c"] * self.entangler_strength))
        return float(self._trainability() ** 1.5)

    def _grad_mag(self) -> float:
        """Độ lớn gradient hữu hiệu, nhất quán với variance (var = grad_mag**1.5)."""
        return float(self._spatial_var() ** (1.0 / 1.5))

    def _effective_capacity(self) -> float:
        return sum(l["mask"] * l["quality"] for l in self.layers)

    def _loss_floor(self) -> float:
        """Sàn loss phụ thuộc depth — giới hạn biểu đạt của mạch nông.

        Ưu tiên surrogate fit từ PennyLane thật; nếu không có thì dùng exp-decay heuristic.
        """
        if self._fit and "floor" in self._fit:
            f = self._fit["floor"]
            return float(f["f_min"] + (f["f_max"] - f["f_min"])
                         * np.exp(-f["a"] * (self.depth - 1)))
        decay = 2.0 / max(self.n_qubits, 2)
        return float(0.05 + 0.60 * np.exp(-decay * (self.depth - 1)))

    def train_epoch(self) -> dict[str, Any]:
        self.epoch += 1
        gm = self._grad_mag()
        floor = self._loss_floor()
        # step TỈ LỆ VỚI PHẦN CÒN LẠI (val_loss - floor) -> hội tụ TIỆM CẬN (giống gradient
        # descent thật gần local minimum: bước tự nhiên nhỏ dần khi gần hội tụ), thay vì
        # step CỐ ĐỊNH mỗi epoch (bản trước — dù đã sửa mũ 0.25->0.75, step cố định vẫn tạo
        # ra hiện tượng "ngưỡng nhị phân": 100 epoch có đủ ngân sách step×epoch để vượt hết
        # khoảng cách tới floor hay không gần như là CÓ/KHÔNG, tạo "vực" đột ngột giữa các
        # depth thay vì suy giảm mượt — vd trước đây depth=12 vẫn hội tụ đầy đủ (87%) nhưng
        # depth=14 đã kẹt hẳn, không có điểm nào ở giữa thể hiện suy giảm dần).
        # rate = tốc độ đóng khoảng cách MỖI EPOCH (0=đứng yên, 1=chạm floor ngay lập tức),
        # phụ thuộc gm (trainability) theo depth/n_qubits/cost_type như cũ. Hằng số K=19.84,
        # p=1.10 hiệu chỉnh để n_qubits=8/depth 2-8 khớp gần đúng số cũ đã kiểm chứng (chưa
        # đổi), còn depth 10->16 giờ suy giảm MƯỢT (75%->67%->46%->27%->14%) thay vì rơi
        # thẳng từ 87% xuống 45% giữa depth 12 và 16.
        rate = min(19.8401 * (gm ** 1.1018), 0.95)
        step = rate * (self._val_loss - floor)
        new_loss = self._val_loss - step + 0.008 * self.rng.standard_normal()
        self._val_loss = float(max(floor, new_loss))
        self._train_loss = self._val_loss + 0.02
        return self.global_metrics()

    def evaluate(self) -> dict[str, Any]:
        ln2 = 0.6931
        acc = float(np.clip(1.0 - self._val_loss / ln2, 0.0, 1.0))
        return {"validation_loss": self._val_loss, "accuracy": acc,
                "f1": float(np.clip(acc * 0.98, 0, 1)),
                "roc_auc": float(np.clip(0.5 + 0.5 * acc, 0.5, 1)), "is_proxy": False}

    def layer_metrics(self) -> list[dict[str, Any]]:
        gm = self._grad_mag()
        sv = self._spatial_var()
        out = []
        for l in self.layers:
            gnorm = gm * l["mask"] * l["quality"] * (1 + 0.1 * self.rng.standard_normal())
            gvar = sv * l["mask"] * l["quality"]
            nzr = float(np.clip(1.0 - l["mask"] * l["quality"] - 0.3 * gm, 0, 1))
            out.append({
                "layer_id": l["layer_id"],
                "layer_status": "active" if l["mask"] > 0.99 else
                                ("masked" if l["mask"] < 0.01 else "masking"),
                "layer_grad_norm": float(abs(gnorm)),
                "layer_grad_variance": float(abs(gvar)),
                "layer_mean_abs_grad": float(abs(gnorm) * 0.8),
                "layer_near_zero_ratio": nzr,
                "mask_value": l["mask"],
            })
        return out

    def global_metrics(self) -> dict[str, Any]:
        gm = self._grad_mag()
        sv = self._spatial_var()
        nzr = float(np.clip(1.0 - self._effective_capacity() / max(self.depth, 1)
                            - 0.3 * gm, 0, 1))
        n_ent = self._n_entangling()
        return {
            "training_loss": self._train_loss,
            "validation_loss": self._val_loss,
            "grad_norm": float(gm * self._effective_capacity()),
            "spatial_grad_variance": float(sv),
            "near_zero_ratio": nzr,
            "depth": self.depth,
            "n_params": self.depth * self.n_qubits * 3,
            "n_entangling_gates": n_ent,
            "entangler_strength": self.entangler_strength,
        }

    def _n_entangling(self) -> int:
        per = {"none": 0, "linear": self.n_qubits - 1,
               "circular": self.n_qubits, "full": self.n_qubits * (self.n_qubits - 1) // 2}
        base = per.get(self.topology, self.n_qubits - 1)
        return int(round(base * self.depth * self.entangler_strength))

    # ---- mutation primitives ----
    def add_layer(self) -> None:
        if self.depth >= self.max_depth:
            return
        self.layers.append(self._new_layer(self.depth))
        self.depth += 1

    def set_layer_mask(self, layer_id: int, alpha: float) -> None:
        for l in self.layers:
            if l["layer_id"] == layer_id:
                l["mask"] = float(np.clip(alpha, 0.0, 1.0))

    def remove_layer(self, layer_id: int) -> None:
        self.layers = [l for l in self.layers if l["layer_id"] != layer_id]
        for i, l in enumerate(self.layers):
            l["layer_id"] = i
        self.depth = len(self.layers)

    def set_entangler_strength(self, alpha: float) -> None:
        self.entangler_strength = float(np.clip(alpha, 0.0, 1.0))

    def snapshot(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "entangler_strength": self.entangler_strength,
            "layers": copy.deepcopy(self.layers),
            "val_loss": self._val_loss,
            "train_loss": self._train_loss,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.depth = snap["depth"]
        self.entangler_strength = snap["entangler_strength"]
        self.layers = copy.deepcopy(snap["layers"])
        self._val_loss = snap["val_loss"]
        self._train_loss = snap["train_loss"]


# ===========================================================================
class PennyLaneBackend(CircuitBackend):
    """
    Mạch THẬT — tái dùng stage1a/ansatz.py. Chạy ở máy có pennylane.

    Phân loại nhị phân: expval(observable) -> sigmoid -> BCE loss.
    Soft-mask lớp i: nhân tham số lớp đó với alpha (theta_eff = alpha * theta),
    đúng cơ chế θ -> α(t)θ của tài liệu V1=.

    LƯU Ý: chưa chạy/kiểm thử trong sandbox này (thiếu pennylane). Cấu trúc bám đúng
    chữ ký apply_hea(params, n_qubits, depth, entanglement_topology, entangler_type).
    """

    def __init__(self, X_train, y_train, X_val, y_val, n_qubits=4, initial_depth=2,
                 max_depth=16, topology="linear", entangler_type="ising_zz",
                 cost_type="local", lr=0.01, seed=0, device_name="default.qubit",
                 diff_method="backprop"):
        import pennylane as qml
        from pennylane import numpy as pnp
        from src.synthetic_bp.stage1a.ansatz import apply_hea
        from src.synthetic_bp.stage1a.observables import build_observable
        self.qml, self.pnp = qml, pnp
        self._apply_hea = apply_hea
        self._build_obs = build_observable

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.n_qubits = n_qubits
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.topology = topology
        self.entangler_type = entangler_type
        self.cost_type = cost_type
        self.lr = lr
        self.seed = seed
        self.device_name = device_name
        self.diff_method = diff_method
        self.reset()

    def reset(self) -> None:
        pnp = self.pnp
        rng = np.random.default_rng(self.seed)
        self.depth = self.initial_depth
        self.entangler_strength = 1.0
        self.masks = np.ones(self.depth)           # alpha mỗi lớp
        self.theta = pnp.array(
            rng.uniform(-np.pi, np.pi, size=(self.depth, self.n_qubits, 3)),
            requires_grad=True)
        self.opt = self.qml.AdamOptimizer(self.lr)
        self.device = self.qml.device(self.device_name, wires=self.n_qubits)
        self._last_metrics = {}

    def _make_qnode(self):
        qml = self.qml
        obs = self._build_obs(self.n_qubits, self.cost_type)

        @qml.qnode(self.device, diff_method=self.diff_method)
        def circuit(theta, x):
            # encoding: Angle Embedding (RX/RY) — khớp mô tả MNIST PCA->angle
            for w in range(self.n_qubits):
                qml.RX(x[w], wires=w)
                qml.RY(x[w], wires=w)
            # áp mask: theta_eff = alpha * theta theo từng lớp
            masked = theta * self.pnp.array(self.masks).reshape(-1, 1, 1)
            self._apply_hea(masked, self.n_qubits, self.depth,
                            self.topology, self.entangler_type)
            return qml.expval(obs)
        return circuit

    def _loss(self, theta, X, y):
        qnode = self._make_qnode()
        preds = self.pnp.stack([(qnode(theta, x) + 1) / 2 for x in X])  # [-1,1]->[0,1]
        eps = 1e-7
        preds = self.pnp.clip(preds, eps, 1 - eps)
        return -self.pnp.mean(y * self.pnp.log(preds) + (1 - y) * self.pnp.log(1 - preds))

    def train_epoch(self) -> dict[str, Any]:
        self.theta, _ = self.opt.step_and_cost(
            lambda t: self._loss(t, self.X_train, self.y_train), self.theta)
        return self.global_metrics()

    def evaluate(self) -> dict[str, Any]:
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
        qnode = self._make_qnode()
        probs = np.array([float((qnode(self.theta, x) + 1) / 2) for x in self.X_val])
        preds = (probs >= 0.5).astype(int)
        vloss = float(self._loss(self.theta, self.X_val, self.y_val))
        try:
            auc = float(roc_auc_score(self.y_val, probs))
        except Exception:
            auc = float("nan")
        return {"validation_loss": vloss,
                "accuracy": float(accuracy_score(self.y_val, preds)),
                "f1": float(f1_score(self.y_val, preds, zero_division=0)),
                "roc_auc": auc, "is_proxy": False}

    def _grads(self):
        g = self.qml.grad(lambda t: self._loss(t, self.X_train, self.y_train))(self.theta)
        return np.array(g)  # shape (depth, n_qubits, 3)

    def sample_grad_variance(self, n_samples=1, seed=0):
        """Phương sai gradient ĐÚNG định nghĩa barren plateau: Var_theta[dC/dtheta_i],
        tức với MỖI vị trí tham số i, tính variance của gradient tại i QUA n_samples lần
        khởi tạo ngẫu nhiên khác nhau — rồi lấy trung bình qua các vị trí tham số.

        (Bản trước ở đây SAI: tính variance GIỮA CÁC THAM SỐ trong 1 lần khởi tạo, rồi
        lấy trung bình qua các lần khởi tạo — một đại lượng "spatial" khác hẳn, không đo
        đúng cái BP theory nói tới, và không thể hiện xu hướng giảm sạch theo depth khi
        thử calibrate thật — n_qubits=8 cho b gần 0 hoặc âm thay vì dương như lý thuyết.)
        n_samples=1 -> chỉ 1 điểm, không có variance thật, coi như fallback cũ.
        """
        if n_samples <= 1:
            return float(self.global_metrics()["spatial_grad_variance"])
        rng = np.random.default_rng(seed)
        saved = self.theta
        grads_per_init = []
        for _ in range(n_samples):
            self.theta = self.pnp.array(
                rng.uniform(-np.pi, np.pi, size=np.array(saved).shape),
                requires_grad=True)
            grads_per_init.append(self._grads().ravel())
        self.theta = saved
        G = np.array(grads_per_init)              # (n_samples, n_params)
        var_per_param = np.var(G, axis=0)          # variance QUA CÁC LẦN KHỞI TẠO, mỗi tham số
        return float(np.mean(var_per_param))       # trung bình qua các vị trí tham số

    def layer_metrics(self) -> list[dict[str, Any]]:
        g = self._grads()
        out = []
        for i in range(self.depth):
            gl = g[i].ravel()
            out.append({
                "layer_id": i,
                "layer_status": "active" if self.masks[i] > 0.99 else
                                ("masked" if self.masks[i] < 0.01 else "masking"),
                "layer_grad_norm": float(np.linalg.norm(gl)),
                "layer_grad_variance": float(np.var(gl)),
                "layer_mean_abs_grad": float(np.mean(np.abs(gl))),
                "layer_near_zero_ratio": float(np.mean(np.abs(gl) < 1e-6)),
                "mask_value": float(self.masks[i]),
            })
        return out

    def global_metrics(self) -> dict[str, Any]:
        g = self._grads().ravel()
        per = {"none": 0, "linear": self.n_qubits - 1, "circular": self.n_qubits,
               "full": self.n_qubits * (self.n_qubits - 1) // 2}
        n_ent = int(round(per.get(self.topology, self.n_qubits - 1)
                          * self.depth * self.entangler_strength))
        tl = float(self._loss(self.theta, self.X_train, self.y_train))
        return {
            "training_loss": tl, "validation_loss": float("nan"),
            "grad_norm": float(np.linalg.norm(g)),
            "spatial_grad_variance": float(np.var(g)),
            "near_zero_ratio": float(np.mean(np.abs(g) < 1e-6)),
            "depth": self.depth, "n_params": int(self.theta.size),
            "n_entangling_gates": n_ent, "entangler_strength": self.entangler_strength,
        }

    def add_layer(self) -> None:
        if self.depth >= self.max_depth:
            return
        pnp = self.pnp
        rng = np.random.default_rng(self.seed + self.depth)
        new = rng.uniform(-np.pi, np.pi, size=(1, self.n_qubits, 3))
        self.theta = pnp.array(np.concatenate([np.array(self.theta), new], axis=0),
                               requires_grad=True)
        self.masks = np.concatenate([self.masks, [1.0]])
        self.depth += 1
        self.opt = self.qml.AdamOptimizer(self.lr)  # reset state cho param mới

    def set_layer_mask(self, layer_id: int, alpha: float) -> None:
        if 0 <= layer_id < self.depth:
            self.masks[layer_id] = float(np.clip(alpha, 0.0, 1.0))

    def remove_layer(self, layer_id: int) -> None:
        keep = [i for i in range(self.depth) if i != layer_id]
        self.theta = self.pnp.array(np.array(self.theta)[keep], requires_grad=True)
        self.masks = self.masks[keep]
        self.depth = len(keep)
        self.opt = self.qml.AdamOptimizer(self.lr)

    def set_entangler_strength(self, alpha: float) -> None:
        self.entangler_strength = float(np.clip(alpha, 0.0, 1.0))

    def snapshot(self) -> dict[str, Any]:
        return {"depth": self.depth, "entangler_strength": self.entangler_strength,
                "theta": np.array(self.theta).copy(), "masks": self.masks.copy()}

    def restore(self, snap: dict[str, Any]) -> None:
        self.depth = snap["depth"]
        self.entangler_strength = snap["entangler_strength"]
        self.theta = self.pnp.array(snap["theta"].copy(), requires_grad=True)
        self.masks = snap["masks"].copy()
        self.opt = self.qml.AdamOptimizer(self.lr)
