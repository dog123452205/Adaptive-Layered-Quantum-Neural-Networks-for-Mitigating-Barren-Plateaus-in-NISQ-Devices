"""
Stage 1B — Task-Aware BP Benchmark (Application Map).

BP do dữ liệu gây ra: |psi(x,theta)> = U_ansatz(theta) U_encode(x)|0>.
Đo Var_seed[dL/dtheta_i] với L là supervised loss.

objective : supervised_loss
encoding  : angle | amplitude | data_reuploading
batch     : fixed benchmark batch (giới hạn V1)
output    : task_aware_bp_summary.parquet, task_aware_bp_param_variance.parquet
            (+ task_aware_bp_layers.parquet cho scheduler/1C dùng per-layer)

Tái dùng stage1a/ansatz.py + observables.py. Backend: pennylane | simulate.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Stage1BConfig:
    n_qubits_list: tuple[int, ...] = (4,)
    depth_list: tuple[int, ...] = (2, 4, 6)
    topology_list: tuple[str, ...] = ("linear", "circular")
    cost_type_list: tuple[str, ...] = ("local", "global")
    encoding_list: tuple[str, ...] = ("angle",)     # angle|amplitude|data_reuploading
    entangler_type: str = "cnot"
    n_seeds: int = 20
    batch_size: int = 16
    near_zero_threshold: float = 1e-6
    backend: str = "simulate"
    seed: int = 0


def save_outputs(result: dict, out_dir) -> dict:
    """Lưu theo tên chuẩn spec. Trả dict {tên: đường dẫn}."""
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    names = {
        "summary": "task_aware_bp_summary.parquet",
        "param_variance": "task_aware_bp_param_variance.parquet",
        "layers": "task_aware_bp_layers.parquet",
    }
    paths = {}
    for key, fname in names.items():
        if key not in result:
            continue
        p = od / fname
        try:
            result[key].to_parquet(p, index=False)
        except Exception:
            p = p.with_suffix(".csv"); result[key].to_csv(p, index=False)
        paths[key] = str(p)
    return paths


class Stage1BEngine:
    def __init__(self, config: Stage1BConfig, data: dict | None = None):
        self.cfg = config
        self.data = data
        self.run_records: list[dict] = []
        self.layer_records: list[dict] = []
        self.param_records: list[dict] = []
        self.run_id = 0

    def _get_batch(self, rng, n_qubits):
        if self.data is not None:
            X, y = np.asarray(self.data["X_train"]), np.asarray(self.data["y_train"])
            k = min(self.cfg.batch_size, len(X))
            idx = rng.choice(len(X), size=k, replace=False)
            return X[idx], y[idx]
        X = rng.uniform(0, np.pi, size=(self.cfg.batch_size, n_qubits))
        y = rng.integers(0, 2, size=self.cfg.batch_size).astype(float)
        return X, y

    def _grads(self, n_qubits, depth, topo, cost, encoding, seed):
        rng = np.random.default_rng(seed)
        X, y = self._get_batch(rng, n_qubits)
        if self.cfg.backend == "pennylane":
            return self._pennylane_grads(n_qubits, depth, topo, cost, encoding, X, y, seed)
        bp = 2.0 ** (-(depth * (2 if cost == "global" else 1)) / 4.0)
        ent = {"none": .4, "linear": .7, "circular": .85, "full": 1.}.get(topo, .7)
        enc = {"angle": 1.0, "amplitude": 0.9, "data_reuploading": 1.1}.get(encoding, 1.0)
        data_factor = 0.5 + 0.5 * np.tanh(float(np.std(X)))
        scale = 0.05 * bp * ent * enc * data_factor
        return rng.normal(0, scale, size=(depth, n_qubits, 3))

    def _pennylane_grads(self, n_qubits, depth, topo, cost, encoding, X, y, seed):
        import pennylane as qml
        from pennylane import numpy as pnp
        from src.synthetic_bp.stage1a.ansatz import apply_hea
        from src.synthetic_bp.stage1a.observables import build_observable
        dev = qml.device("default.qubit", wires=n_qubits)
        obs = build_observable(n_qubits, cost)
        reup = (encoding == "data_reuploading")

        @qml.qnode(dev, diff_method="backprop")
        def circuit(theta, x):
            for w in range(n_qubits):
                qml.RX(x[w], wires=w); qml.RY(x[w], wires=w)
            for d in range(depth):
                if reup and d > 0:
                    for w in range(n_qubits):
                        qml.RX(x[w], wires=w)
                apply_hea(theta[d:d+1], n_qubits, 1, topo, self.cfg.entangler_type)
            return qml.expval(obs)

        rng = np.random.default_rng(seed)
        theta = pnp.array(rng.uniform(-np.pi, np.pi, size=(depth, n_qubits, 3)),
                          requires_grad=True)

        def loss(theta):
            preds = pnp.stack([(circuit(theta, x) + 1) / 2 for x in X])
            eps = 1e-7; preds = pnp.clip(preds, eps, 1 - eps)
            return -pnp.mean(y * pnp.log(preds) + (1 - y) * pnp.log(1 - preds))

        return np.array(qml.grad(loss)(theta))

    def run(self) -> dict[str, pd.DataFrame]:
        cfg = self.cfg
        for nq, depth, topo, cost, enc in product(
                cfg.n_qubits_list, cfg.depth_list, cfg.topology_list,
                cfg.cost_type_list, cfg.encoding_list):
            self._run_structure(nq, depth, topo, cost, enc)
        return {
            "summary": pd.DataFrame(self.run_records),
            "param_variance": pd.DataFrame(self.param_records),
            "layers": pd.DataFrame(self.layer_records),
        }

    def _run_structure(self, n_qubits, depth, topo, cost, encoding):
        cfg = self.cfg
        n_params = depth * n_qubits * 3
        seeds_flat = [self._grads(n_qubits, depth, topo, cost, encoding, cfg.seed + s).reshape(-1)
                      for s in range(cfg.n_seeds)]
        arr = np.array(seeds_flat)
        var_per_param = np.var(arr, axis=0)
        spatial_var = float(np.mean(var_per_param))
        grad_norm = float(np.mean(np.linalg.norm(arr, axis=1)))
        near_zero = float(np.mean(np.abs(arr) < cfg.near_zero_threshold))
        self.run_id += 1

        self.run_records.append({
            "benchmark_mode": "task_aware", "objective_type": "supervised_loss",
            "ansatz_type": "hea", "encoding_type": encoding,
            "entanglement_topology": topo, "entangler_type": cfg.entangler_type,
            "cost_type": cost, "n_qubits": n_qubits, "depth": depth,
            "run_id": self.run_id, "n_params": n_params,
            "spatial_grad_variance": spatial_var, "grad_norm": grad_norm,
            "normalized_grad_norm": grad_norm / np.sqrt(max(n_params, 1)),
            "near_zero_ratio": near_zero,
            "log10_spatial_grad_variance": float(np.log10(max(spatial_var, 1e-30))),
            "batch_size": cfg.batch_size, "n_seeds": cfg.n_seeds,
        })
        for pi in range(n_params):
            self.param_records.append({
                "n_qubits": n_qubits, "depth": depth, "entanglement_topology": topo,
                "cost_type": cost, "encoding_type": encoding,
                "param_index": pi, "layer_id": pi // (n_qubits * 3),
                "param_grad_variance_across_seeds": float(var_per_param[pi]),
                "log10_param_grad_variance": float(np.log10(max(var_per_param[pi], 1e-30))),
            })
        for li in range(depth):
            cols = arr[:, li*n_qubits*3:(li+1)*n_qubits*3]
            self.layer_records.append({
                "n_qubits": n_qubits, "depth": depth, "entanglement_topology": topo,
                "cost_type": cost, "encoding_type": encoding, "layer_id": li,
                "layer_grad_variance": float(np.mean(np.var(cols, axis=0))),
                "layer_grad_norm": float(np.mean(np.linalg.norm(cols, axis=1))),
                "layer_mean_abs_grad": float(np.mean(np.abs(cols))),
                "layer_max_abs_grad": float(np.max(np.abs(cols))),
                "layer_near_zero_ratio": float(np.mean(np.abs(cols) < cfg.near_zero_threshold)),
            })
