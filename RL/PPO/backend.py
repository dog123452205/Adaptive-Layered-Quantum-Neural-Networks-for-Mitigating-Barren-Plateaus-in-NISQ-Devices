"""
backend.py — Interface contract giữa RL Scheduler (V2) và lõi QNN (V1).

Hai hiện thực:
  - Stage1aReplayBackend : đọc CSV Stage 1A thật -> môi trường offline (dùng NGAY).
  - PennyLaneRuntimeBackend : nhóm Q hiện thực sau (bọc Stage 1/3 thật). Stub ở cuối file.

Tên cột bám đúng schema thật (schema.py / stage1a_smoke_layers.csv / runs.csv).
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
import pandas as pd

# Macro-action space (đúng R1 trong đặc tả): agent chỉ ra quyết định mức cao.
MACRO_ACTIONS = [
    "keep",
    "add_layer",
    "stop_growth",
    "propose_prune_layer",
    "propose_reduce_entanglement",
]
N_ACTIONS = len(MACRO_ACTIONS)


@runtime_checkable
class QNNBackend(Protocol):
    def reset(self) -> dict: ...
    def get_layer_metrics(self) -> list[dict]: ...
    def get_global_metrics(self) -> dict: ...
    def submit_mutation(self, layer_idx: int, macro_action: str) -> None: ...
    def step_training(self, k_epochs: int) -> dict: ...
    def get_thresholds(self) -> dict: ...
    def valid_action_mask(self, layer_idx: int) -> np.ndarray: ...


class Stage1aReplayBackend:
    """
    Môi trường offline build từ output Stage 1A thật.

    Mỗi 'config' (n_qubits, depth, entanglement_topology, cost_type) là một circuit.
    layers.csv cung cấp metrics theo layer; runs.csv cung cấp metrics global.
    Vì Stage 1A là data-free benchmark, ta MÔ PHỎNG hiệu ứng mutation bằng mô hình
    đơn giản nhưng bám quy luật vật lý (Var giảm theo depth/entanglement) để prototype
    thuật toán RL. Khi Stage 3 thật xong -> thay bằng PennyLaneRuntimeBackend.
    """

    def __init__(self, layers_csv: str, runs_csv: str,
                 max_depth: int = 16, min_depth: int = 1, seed: int = 0):
        self.layers = pd.read_csv(layers_csv)
        self.runs = pd.read_csv(runs_csv)
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.rng = np.random.default_rng(seed)

        # Khóa nhóm 1 circuit-config để replay
        self.group_cols = ["n_qubits", "entanglement_topology", "cost_type"]
        self.configs = (
            self.layers[self.group_cols].drop_duplicates().reset_index(drop=True)
        )
        self._cur = None          # dict mô tả circuit hiện tại
        self._layer_state = None  # list dict per active layer (mutable)

    # ---- helpers ---------------------------------------------------------
    def _sample_config(self) -> dict:
        row = self.configs.iloc[self.rng.integers(len(self.configs))]
        return {c: row[c] for c in self.group_cols}

    def _layers_for(self, cfg: dict, depth: int) -> list[dict]:
        m = self.layers
        sub = m[(m.n_qubits == cfg["n_qubits"]) &
                (m.entanglement_topology == cfg["entanglement_topology"]) &
                (m.cost_type == cfg["cost_type"]) &
                (m.depth == depth)]
        if len(sub) == 0:  # fallback: lấy depth gần nhất có sẵn
            sub = m[(m.n_qubits == cfg["n_qubits"]) &
                    (m.cost_type == cfg["cost_type"])]
        out = []
        # gom theo layer_id, lấy trung bình qua seed/run
        for lid, g in sub.groupby("layer_id"):
            out.append({
                "layer_id": int(lid),
                "layer_grad_variance": float(g.layer_grad_variance.mean()),
                "layer_grad_norm": float(g.layer_grad_norm.mean()),
                "layer_mean_abs_grad": float(g.layer_mean_abs_grad.mean()),
                "layer_near_zero_ratio": float(g.layer_near_zero_ratio.mean()),
                "layer_max_abs_grad": float(g.layer_max_abs_grad.mean()),
                "layer_status": "active",
                "mask_value": 1.0,
                "age": 0,
            })
        return out

    def _global_for(self, cfg: dict, depth: int) -> dict:
        r = self.runs
        sub = r[(r.n_qubits == cfg["n_qubits"]) &
                (r.entanglement_topology == cfg["entanglement_topology"]) &
                (r.cost_type == cfg["cost_type"]) &
                (r.depth == depth)]
        if len(sub) == 0:
            sub = r[(r.n_qubits == cfg["n_qubits"]) & (r.cost_type == cfg["cost_type"])]
        depth_eff = len(self._layer_state) if self._layer_state else depth
        n_ent = int(sub.n_entangling_gates.mean()) if len(sub) else depth_eff * cfg["n_qubits"]
        return {
            "spatial_grad_variance": float(sub.spatial_grad_variance.mean()) if len(sub) else 1e-3,
            "normalized_grad_norm": float(sub.normalized_grad_norm.mean()) if len(sub) else 0.1,
            "near_zero_ratio": float(sub.near_zero_ratio.mean()) if len(sub) else 0.3,
            "n_params": int(sub.n_params.mean()) if len(sub) else depth_eff * cfg["n_qubits"] * 3,
            "n_entangling_gates": n_ent,
            "depth": depth_eff,
            # telemetry runtime giả lập (Stage 3 chưa có) — sẽ thay bằng số thật ở P4
            "training_loss": float(self._train_loss),
            "validation_loss": float(self._val_loss),
            "grad_norm": float(np.mean([l["layer_grad_norm"] for l in self._layer_state])) if self._layer_state else 0.1,
        }

    # ---- contract --------------------------------------------------------
    def reset(self) -> dict:
        self._cur = self._sample_config()
        depth0 = 2
        self._layer_state = self._layers_for(self._cur, depth0)
        if not self._layer_state:
            self._layer_state = self._layers_for(self._cur, 1)
        self._growth_locked = False
        self._pending_prune = None
        self._pending_reduce = False
        # loss giả lập phải set TRƯỚC khi gọi _global_for (vì _global_for đọc nó)
        self._val_loss = 0.69 + 0.1 * self.rng.standard_normal()  # ~ln2 (binary)
        self._train_loss = self._val_loss + 0.02
        return self.snapshot()

    def get_layer_metrics(self) -> list[dict]:
        return [dict(l) for l in self._layer_state]

    def get_global_metrics(self) -> dict:
        return self._global_for(self._cur, len(self._layer_state))

    def valid_action_mask(self, layer_idx: int) -> np.ndarray:
        mask = np.ones(N_ACTIONS, dtype=bool)
        depth = len(self._layer_state)
        if depth >= self.max_depth or self._growth_locked:
            mask[MACRO_ACTIONS.index("add_layer")] = False
        if depth <= self.min_depth:
            mask[MACRO_ACTIONS.index("propose_prune_layer")] = False
        # reduce-entanglement chỉ hợp lệ khi topology có entangler
        if self._cur["entanglement_topology"] in ("none",):
            mask[MACRO_ACTIONS.index("propose_reduce_entanglement")] = False
        return mask

    def submit_mutation(self, layer_idx: int, macro_action: str) -> None:
        """Mô phỏng hiệu ứng mutation (sẽ do Stage 3 thật xử lý ở P4)."""
        layer_idx = min(layer_idx, len(self._layer_state) - 1)
        if macro_action == "add_layer" and len(self._layer_state) < self.max_depth and not self._growth_locked:
            base = dict(self._layer_state[-1])
            base["layer_id"] = len(self._layer_state)
            base["age"] = 0
            # lớp mới: gradient mạnh hơn chút (chưa bị BP), nhưng tăng depth
            base["layer_grad_norm"] *= 1.15
            base["layer_grad_variance"] *= 1.15
            self._layer_state.append(base)
        elif macro_action == "stop_growth":
            self._growth_locked = True
        elif macro_action == "propose_prune_layer" and len(self._layer_state) > self.min_depth:
            # soft-mask layer yếu nhất rồi loại (Stage 3 thật sẽ anneal cosine 5 bước)
            self._pending_prune = layer_idx
        elif macro_action == "propose_reduce_entanglement":
            self._pending_reduce = True
        self._last_action = macro_action

    def step_training(self, k_epochs: int) -> dict:
        """Train k epoch (giả lập). Trả metrics mới + mutation_status."""
        g = self.get_global_metrics()
        gv = max(g["spatial_grad_variance"], 1e-9)
        nzr = g["near_zero_ratio"]
        # loss giảm tỉ lệ với trainability (var cao, near-zero thấp -> giảm nhanh)
        progress = 0.05 * k_epochs * (gv ** 0.25) * (1 - nzr)
        self._val_loss = max(0.05, self._val_loss - progress + 0.01 * self.rng.standard_normal())
        self._train_loss = self._val_loss + 0.02

        status = "none"
        # xử lý prune đang chờ
        if getattr(self, "_pending_prune", None) is not None:
            idx = self._pending_prune
            # acceptance: nếu loss không tệ đi quá 2% thì commit, ngược lại rollback
            if self.rng.random() < 0.75 and len(self._layer_state) > self.min_depth:
                del self._layer_state[idx]
                for j, l in enumerate(self._layer_state):
                    l["layer_id"] = j
                status = "committed"
            else:
                status = "rolled_back"
            self._pending_prune = None
        if getattr(self, "_pending_reduce", False):
            self._pending_reduce = False
            status = "committed" if status == "none" else status

        for l in self._layer_state:
            l["age"] += 1
        out = self.get_global_metrics()
        out["mutation_status"] = status
        out["rollback_triggered"] = int(status == "rolled_back")
        return out

    def get_thresholds(self) -> dict:
        # Stage 1C chưa có -> ước lượng tau theo công thức đặc tả: beta*sqrt(N*Var_1A)
        g = self.get_global_metrics()
        beta = 1.0
        tau = beta * np.sqrt(max(g["n_params"], 1) * max(g["spatial_grad_variance"], 1e-12))
        return {"tau_empirical": float(tau), "near_zero_grad_threshold": 1e-6}

    def snapshot(self) -> dict:
        return {"layers": self.get_layer_metrics(),
                "global": self.get_global_metrics(),
                "thresholds": self.get_thresholds()}


# =====================================================================
# PennyLaneRuntimeBackend — hiện thực THẬT (P4).
# Bọc một QNN PennyLane (HEA / IsingZZ) huấn luyện trên Iris.
# Cùng 7 method của QNNBackend, cùng tên cột state -> drop-in cho env.py.
#
# Yêu cầu: pip install pennylane scikit-learn
# Thiết kế bám adaptive_alqnn_template.yaml:
#   - n_qubits=4, initial_depth=2, min_depth=1, max_depth=16
#   - ansatz HEA: mỗi layer = (RX,RY,RZ) mỗi qubit + entangler IsingZZ theo topology
#   - encoding: angle (RY(x_i)) một lần ở đầu mạch (reuploading_layers=1)
#   - readout: <Z_0> -> p=(1+<Z>)/2 ; loss = BCE
#   - acceptance: validation_loss tăng <= max_relative_loss_increase thì commit
# =====================================================================
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    _PL_OK = True
except Exception:  # pennylane chưa cài
    qml = None
    pnp = None
    _PL_OK = False


# Số cặp entangling theo topology (mỗi layer)
def _pairs_for(topology: str, n_qubits: int) -> list[tuple[int, int]]:
    if topology == "none":
        return []
    if topology == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if topology == "circular":
        return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    raise ValueError(f"topology không hỗ trợ: {topology}")


# Thứ tự bậc topology để 'reduce_entanglement' đi xuống một nấc
_TOPO_ORDER = ["circular", "linear", "none"]


class PennyLaneRuntimeBackend:
    """
    Backend thật cho AL-QNN Scheduler. Mỗi 'layer' là một block HEA tham số hóa;
    scheduler thêm/cắt layer hoặc giảm entanglement, backend train mạch thật và
    đo gradient variance / loss thật để trả về cho env.

    Param được giữ ở một vector phẳng `theta` (pennylane.numpy, requires_grad)
    kèm `self.slices`: list (rot_slice, ent_slice) cho từng layer -> tách gradient
    theo layer dễ dàng (phục vụ near-zero-ratio, layer_grad_variance...).
    """

    def __init__(self,
                 n_qubits: int = 4,
                 initial_depth: int = 2,
                 min_depth: int = 1,
                 max_depth: int = 16,
                 topology: str = "linear",
                 lr: float = 0.01,
                 batch_size: int = 16,
                 near_zero_grad_threshold: float = 1e-6,
                 max_relative_loss_increase: float = 0.02,
                 seed: int = 42,
                 device_name: str = "default.qubit"):
        if not _PL_OK:
            raise ImportError(
                "PennyLaneRuntimeBackend cần PennyLane. Cài: pip install pennylane scikit-learn"
            )
        self.n_qubits = n_qubits
        self.init_depth = initial_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.topology = topology
        self.init_topology = topology
        self.lr = lr
        self.batch_size = batch_size
        self.nz_thr = near_zero_grad_threshold
        self.max_rel_inc = max_relative_loss_increase
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.dev = qml.device(device_name, wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev,
                               interface="autograd", diff_method="backprop")

        self._load_iris(seed)
        self.reset()

    # ---- dữ liệu ---------------------------------------------------------
    def _load_iris(self, seed: int):
        """Iris binary (lớp 0,1), minmax -> [0, pi], split train/val + 1 fixed batch."""
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        X, y = load_iris(return_X_y=True)
        mask = y < 2                      # lấy lớp 0 và 1
        X, y = X[mask], y[mask]
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y)
        sc = MinMaxScaler(feature_range=(0.0, np.pi)).fit(Xtr)  # fit on train only
        self.Xtr = pnp.array(sc.transform(Xtr), requires_grad=False)
        self.Xva = pnp.array(sc.transform(Xva), requires_grad=False)
        self.ytr = pnp.array(ytr.astype(float), requires_grad=False)
        self.yva = pnp.array(yva.astype(float), requires_grad=False)
        # fixed batch để đo gradient variance (benchmark BP)
        idx = self.rng.choice(len(self.Xtr),
                              size=min(self.batch_size, len(self.Xtr)), replace=False)
        self.Xb = self.Xtr[idx]
        self.yb = self.ytr[idx]

    # ---- mạch lượng tử ---------------------------------------------------
    def _build_slices(self, depth: int):
        """Tạo bản đồ slice cho theta theo depth & topology hiện tại."""
        pairs = _pairs_for(self.topology, self.n_qubits)
        n_rot = self.n_qubits * 3
        n_ent = len(pairs)
        slices, cur = [], 0
        for _ in range(depth):
            rot = slice(cur, cur + n_rot); cur += n_rot
            ent = slice(cur, cur + n_ent); cur += n_ent
            slices.append((rot, ent))
        return slices, cur, pairs

    def _init_theta(self, depth: int):
        self.slices, total, self.pairs = self._build_slices(depth)
        theta = self.rng.uniform(-np.pi, np.pi, size=total)
        self.theta = pnp.array(theta, requires_grad=True)
        self.depth = depth
        # mask mềm prune theo layer (1 = active). Stage 3 anneal cosine; draft giữ nhị phân.
        self.rot_mask = [1.0] * depth

    def _circuit(self, theta):
        # encoding angle (RY) một lần
        for q in range(self.n_qubits):
            qml.RY(self._x_cache[q], wires=q)
        # các layer HEA
        for li, (rot_sl, ent_sl) in enumerate(self.slices):
            m = self.rot_mask[li]
            rot = theta[rot_sl].reshape(self.n_qubits, 3)
            for q in range(self.n_qubits):
                qml.RX(m * rot[q, 0], wires=q)
                qml.RY(m * rot[q, 1], wires=q)
                qml.RZ(m * rot[q, 2], wires=q)
            ent = theta[ent_sl]
            for k, (a, b) in enumerate(self.pairs):
                qml.IsingZZ(ent[k], wires=[a, b])
        return qml.expval(qml.PauliZ(0))

    def _proba(self, theta, x):
        self._x_cache = x
        z = self.qnode(theta)
        return (1.0 + z) / 2.0          # <Z> in [-1,1] -> p in [0,1]

    def _bce(self, theta, X, Y):
        eps = 1e-7
        loss = 0.0
        for i in range(len(X)):
            p = self._proba(theta, X[i])
            p = pnp.clip(p, eps, 1 - eps)
            loss = loss - (Y[i] * pnp.log(p) + (1 - Y[i]) * pnp.log(1 - p))
        return loss / len(X)

    def _accuracy(self, theta, X, Y):
        correct = 0
        for i in range(len(X)):
            p = self._proba(theta, X[i])
            correct += int((float(p) >= 0.5) == bool(Y[i]))
        return correct / len(X)

    # ---- đo gradient (barren-plateau metrics) ----------------------------
    def _recompute_metrics(self):
        """Tính gradient thật trên fixed batch -> cache layer & global metrics."""
        grad_fn = qml.grad(lambda t: self._bce(t, self.Xb, self.yb))
        g = np.asarray(grad_fn(self.theta), dtype=float)

        # global
        self._val_loss = float(self._bce(self.theta, self.Xva, self.yva))
        self._train_loss = float(self._bce(self.theta, self.Xtr, self.ytr))
        self._acc = self._accuracy(self.theta, self.Xva, self.yva)
        n_params = g.size
        self._global = dict(
            spatial_grad_variance=float(np.var(g)) if n_params else 0.0,
            normalized_grad_norm=float(np.linalg.norm(g) / np.sqrt(max(n_params, 1))),
            near_zero_ratio=float(np.mean(np.abs(g) < self.nz_thr)) if n_params else 1.0,
            n_params=int(n_params),
            n_entangling_gates=len(self.pairs) * self.depth,
            depth=self.depth,
            training_loss=self._train_loss,
            validation_loss=self._val_loss,
            grad_norm=float(np.linalg.norm(g)),
            accuracy=self._acc,
        )

        # per-layer
        layers = []
        for li, (rot_sl, ent_sl) in enumerate(self.slices):
            gl = np.concatenate([g[rot_sl], g[ent_sl]]) if len(self.pairs) else g[rot_sl]
            layers.append(dict(
                layer_id=li,
                layer_grad_variance=float(np.var(gl)) if gl.size else 0.0,
                layer_grad_norm=float(np.linalg.norm(gl)),
                layer_mean_abs_grad=float(np.mean(np.abs(gl))) if gl.size else 0.0,
                layer_near_zero_ratio=float(np.mean(np.abs(gl) < self.nz_thr)) if gl.size else 1.0,
                layer_max_abs_grad=float(np.max(np.abs(gl))) if gl.size else 0.0,
                layer_status="active",
                mask_value=self.rot_mask[li],
                age=0,
            ))
        self._layers = layers

    # ---- huấn luyện ------------------------------------------------------
    def _train_epochs(self, k_epochs: int):
        opt = qml.AdamOptimizer(stepsize=self.lr)
        cost = lambda t: self._bce(t, self.Xtr, self.ytr)
        for _ in range(k_epochs):
            self.theta = opt.step(cost, self.theta)

    # ---- contract (7 method) --------------------------------------------
    def reset(self) -> dict:
        self.topology = self.init_topology   # khôi phục topology gốc mỗi episode
        self._init_theta(self.init_depth)
        self._growth_locked = False
        self._pending_prune = None
        self._pending_reduce = False
        self._recompute_metrics()
        return self.snapshot()

    def get_layer_metrics(self) -> list[dict]:
        return [dict(l) for l in self._layers]

    def get_global_metrics(self) -> dict:
        return dict(self._global)

    def get_thresholds(self) -> dict:
        g = self._global
        beta = 1.0
        tau = beta * np.sqrt(max(g["n_params"], 1) * max(g["spatial_grad_variance"], 1e-12))
        return {"tau_empirical": float(tau), "near_zero_grad_threshold": self.nz_thr}

    def valid_action_mask(self, layer_idx: int) -> np.ndarray:
        mask = np.ones(N_ACTIONS, dtype=bool)
        if self.depth >= self.max_depth or self._growth_locked:
            mask[MACRO_ACTIONS.index("add_layer")] = False
        if self.depth <= self.min_depth:
            mask[MACRO_ACTIONS.index("propose_prune_layer")] = False
        if self.topology == "none":
            mask[MACRO_ACTIONS.index("propose_reduce_entanglement")] = False
        return mask

    def submit_mutation(self, layer_idx: int, macro_action: str) -> None:
        layer_idx = min(layer_idx, self.depth - 1)
        if macro_action == "add_layer" and self.depth < self.max_depth and not self._growth_locked:
            self._grow_one_layer()
            self._recompute_metrics()          # depth đổi ngay -> env đọc đúng L
        elif macro_action == "stop_growth":
            self._growth_locked = True
        elif macro_action == "propose_prune_layer" and self.depth > self.min_depth:
            self._pending_prune = layer_idx     # commit/rollback ở step_training
        elif macro_action == "propose_reduce_entanglement" and self.topology != "none":
            self._pending_reduce = True
        self._last_action = macro_action

    def step_training(self, k_epochs: int) -> dict:
        status = "none"
        # snapshot để rollback
        snap = self._snapshot_state()
        vloss_before = self._val_loss

        applied_prune = self._pending_prune is not None
        applied_reduce = self._pending_reduce

        if applied_prune:
            self._drop_layer(self._pending_prune)
            self._pending_prune = None
        if applied_reduce:
            self._reduce_topology()
            self._pending_reduce = False

        self._train_epochs(k_epochs)
        self._recompute_metrics()

        # acceptance: chỉ áp cho mutation (add/keep không cần rollback)
        if applied_prune or applied_reduce:
            rel_inc = (self._val_loss - vloss_before) / (abs(vloss_before) + 1e-9)
            if rel_inc <= self.max_rel_inc:
                status = "committed"
            else:
                self._restore_state(snap)        # khôi phục kiến trúc + param
                self._train_epochs(0)            # không train thêm
                self._recompute_metrics()
                status = "rolled_back"

        out = self.get_global_metrics()
        out["mutation_status"] = status
        out["rollback_triggered"] = int(status == "rolled_back")
        return out

    # ---- thao tác kiến trúc ---------------------------------------------
    def _grow_one_layer(self):
        """Thêm một layer HEA mới (rotation gần 0 -> khởi đầu gần identity)."""
        n_rot = self.n_qubits * 3
        n_ent = len(self.pairs)
        new = self.rng.normal(0.0, 0.1, size=n_rot + n_ent)   # centered, biên độ nhỏ
        theta_np = np.asarray(self.theta, dtype=float)
        theta_np = np.concatenate([theta_np, new])
        self.depth += 1
        self.rot_mask.append(1.0)
        self.slices, _, self.pairs = self._build_slices(self.depth)
        self.theta = pnp.array(theta_np, requires_grad=True)

    def _drop_layer(self, idx: int):
        n_rot = self.n_qubits * 3
        n_ent = len(self.pairs)
        per = n_rot + n_ent
        theta_np = np.asarray(self.theta, dtype=float)
        keep = np.concatenate([theta_np[:idx * per], theta_np[(idx + 1) * per:]])
        self.depth -= 1
        del self.rot_mask[idx]
        self.slices, _, self.pairs = self._build_slices(self.depth)
        self.theta = pnp.array(keep, requires_grad=True)

    def _reduce_topology(self):
        """Hạ một nấc: circular -> linear -> none. Rebuild entangler params."""
        i = _TOPO_ORDER.index(self.topology)
        if i + 1 < len(_TOPO_ORDER):
            new_topo = _TOPO_ORDER[i + 1]
            # giữ rotation params, dựng lại entangler cho topology mới
            rot_per = self.n_qubits * 3
            old_pairs = len(self.pairs)
            theta_np = np.asarray(self.theta, dtype=float)
            rots = []
            cur = 0
            for _ in range(self.depth):
                rots.append(theta_np[cur:cur + rot_per]); cur += rot_per + old_pairs
            self.topology = new_topo
            self.slices, _, self.pairs = self._build_slices(self.depth)
            new_ent = len(self.pairs)
            blocks = []
            for r in rots:
                blocks.append(r)
                if new_ent:
                    blocks.append(self.rng.normal(0.0, 0.1, size=new_ent))
            self.theta = pnp.array(np.concatenate(blocks) if blocks else np.array([]),
                                   requires_grad=True)

    # ---- snapshot/restore (rollback) ------------------------------------
    def _snapshot_state(self) -> dict:
        return dict(
            theta=np.asarray(self.theta, dtype=float).copy(),
            depth=self.depth,
            topology=self.topology,
            rot_mask=list(self.rot_mask),
        )

    def _restore_state(self, snap: dict):
        self.topology = snap["topology"]
        self.depth = snap["depth"]
        self.rot_mask = list(snap["rot_mask"])
        self.slices, _, self.pairs = self._build_slices(self.depth)
        self.theta = pnp.array(snap["theta"], requires_grad=True)

    def snapshot(self) -> dict:
        return {"layers": self.get_layer_metrics(),
                "global": self.get_global_metrics(),
                "thresholds": self.get_thresholds()}
