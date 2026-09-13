"""
Stage 1B task-aware barren plateau benchmark engine.

This engine computes:

    Var_seed[dL_B(θ)/dθ_k]

where:
    L_B(θ) = mean_i loss(fθ(x_i), y_i)

Stage 1B v1 uses:
    - fixed benchmark batch,
    - angle encoding,
    - binary supervised loss,
    - HEA ansatz,
    - Welford online variance.

Important limitation:
    Fixed-batch Stage 1B isolates initialization-induced variance but does not
    fully capture data-induced gradient variability. Multi-batch analysis is
    reserved for v2/v3.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import pennylane as qml
import pennylane.numpy as pnp
from tqdm import tqdm

from src.synthetic_bp.constants import BenchmarkMode, ObjectiveType
from src.synthetic_bp.stage1a.ansatz import apply_hea
from src.synthetic_bp.stage1a.budget import count_hea_parameters, plan_effective_depth
from src.synthetic_bp.stage1a.metrics import iter_gradient_components
from src.synthetic_bp.stage1a.online_stats import OnlineGradientStats
from src.synthetic_bp.stage1b.data import (
    FixedBatch,
    build_fixed_batches,
    prepare_task_dataset,
    save_batch_manifest,
)
from src.synthetic_bp.stage1b.encoding import apply_data_encoding
from src.synthetic_bp.stage1b.io_utils import ensure_plot_dir, save_table
from src.synthetic_bp.stage1b.losses import compute_supervised_loss
from src.synthetic_bp.stage1b.metrics import (
    compute_layer_metrics_task,
    compute_prediction_metrics,
    compute_task_run_metrics,
)
from src.synthetic_bp.stage1b.readout import (
    build_readout_observable,
    z_expectation_to_probability,
)

class Stage1BTaskAwareBPEngine:
    '''
    Task-aware BP benchmark runner.

    Args:
        config:
            Resolved and validated Stage 1B config.

    Outputs:
        - run-level metrics,
        - layer-level metrics,
        - optional raw gradient records,
        - parameter variance across seeds,
        - summary table,
        - fixed batch manifest.
    '''

    def __init__(self, config: dict[str, Any]):
        self.config = config

        self.experiment = config['experiment']
        self.benchmark = config['benchmark']
        self.dataset_cfg = config['dataset']
        self.encoding_cfg = config['encoding']
        self.ansatz = config["ansatz"]
        self.sweep = config["sweep"]
        self.loss_cfg = config["loss"]
        self.readout_cfg = config['readout']
        self.batching_cfg = config["batching"]
        self.initialization = config['initialization']
        self.device_config = config['device']
        self.metrics_config = config['metrics']
        self.logging_config = config['logging']
        self.outputs = config['outputs']

        self.run_records: list[dict[str, Any]] = []
        self.layer_records: list[dict[str, Any]] = []
        self.raw_gradient_records: list[dict[str, Any]] = []

        self.gradient_stats: dict[tuple[Any, ...], OnlineGradientStats] = defaultdict(
            OnlineGradientStats
        )
        
        self.run_id = 0
    
    def run(self) -> dict[str, pd.DataFrame]:
        '''
        Execute the full Stage 1B benchmark.

        Returns:
            Dictionary of output DataFrames.
        '''

        self._validate_stage1b_assumption()

        dataset = prepare_task_dataset(self.config)
        fixed_batches = build_fixed_batches(dataset, self.config)

        if "batch_manifest_path" in self.outputs:
            save_batch_manifest(
                batches=fixed_batches,
                dataset=dataset,
                output_path=self.outputs['batch_manifest_path']
            )
        
        configs = list(self._iter_benchmark_config())

        for config_item, fixed_batch in tqdm(
            list(itertools.product(configs, fixed_batches)),
            desc="Stage 1B task-aware BP"
        ):
            self._run_single_structure(
                config_item = config_item,
                fixed_batch = fixed_batch
            )
        
        run_df = pd.DataFrame(self.run_records)
        layer_df = pd.DataFrame(self.layer_records)
        param_variance_df = self._build_param_variance_df()
        summary_df = self._build_summary_df(run_df, layer_df, param_variance_df)

        self._save_outputs(run_df, layer_df, param_variance_df, summary_df)

        result = {
            "runs": run_df,
            "layers": layer_df,
            "param_variance": param_variance_df,
            "summary": summary_df,
        }

        if self.logging_config['save_raw_gradients']:
            result['raw_gradients'] = pd.DataFrame(self.raw_gradient_records)
        
        return result
    
    def _validate_stage1b_assumption(self) -> None:
        """
        Enforce Stage 1B task-aware assumptions.

        Raises:
            ValueError:
                If config is not a Stage 1B supervised-loss benchmark.
        """
        if self.benchmark['mode'] != BenchmarkMode.TASK_AWARE_BP.value:
            raise ValueError("Stage1BTaskAwareBPEngine requires benchmark.mode=task_aware_bp.")
        
        if self.benchmark["objective_type"] != ObjectiveType.SUPERVISED_LOSS.value:
            raise ValueError(
                "Stage1BTaskAwareBPEngine requires objective_type=supervised_loss."
            )

    def _iter_benchmark_config(self):
        """
        Iterate over benchmark structures, excluding seed and batch.

        Yields:
            Dictionary describing one circuit structure.
        """
        n_qubits_list = self.sweep['n_qubits']
        depths = self.sweep['depths']
        topologies = self.ansatz['entanglement_topologies']

        budget_normalization = self.benchmark.get("budget_normalization", "same_depth")

        for n_qubits, reference_depth, topology in itertools.product(
            n_qubits_list,
            depths,
            topologies
        ):
            depth_plan = plan_effective_depth(
                n_qubits=int(n_qubits),
                reference_depth=int(reference_depth),
                topology=topology,
                budget_normalization=budget_normalization
            )

            yield {
                "n_qubits": int(n_qubits),
                "reference_depth": int(reference_depth),
                "depth": int(depth_plan.effective_depth),
                "entanglement_topology": topology,
                "target_entangling_budget": int(depth_plan.target_entangling_budget),
                "actual_entangling_budget": int(depth_plan.actual_entangling_budget),
                "gates_per_layer": int(depth_plan.gates_per_layer),
            }
        
    def _run_single_structure(
        self,
        config_item: dict[str, Any],
        fixed_batch: FixedBatch
    ) -> None:
        """
        Run all seeds for one circuit structure and fixed batch.

        Args:
            config_item:
                Structure config excluding seed.
            fixed_batch:
                Fixed benchmark batch.
        """
        n_qubits = config_item['n_qubits']
        depth = config_item['depth']
        topology = config_item['entanglement_topology']

        qnode = self._build_qnode(
            n_qubits = n_qubits,
            depth = depth,
            topology = topology
        )

        loss_fn = self._build_batch_loss_function(
            qnode = qnode,
            fixed_batch = fixed_batch
        )

        seeds = range(int(self.sweep['seeds']))

        for seed in seeds:
            params = self._initialize_params(
                n_qubits = n_qubits,
                depth=depth,
                seed=seed
            )

            loss_value = float(loss_fn(params))
            grad = qml.grad(loss_fn)(params)

            probabilities = self._compute_batch_probabilities(
                qnode=qnode,
                params=params,
                fixed_batch=fixed_batch
            )

            self._record_run(
                config_item=config_item,
                fixed_batch=fixed_batch,
                seed=seed,
                loss_value=loss_value,
                probabilities=probabilities,
                grad=grad
            )
    
    def _build_qnode(
        self,
        n_qubits: int, 
        depth: int,
        topology: str,
    ):
        """
        Build QNode for one task-aware circuit structure.

        Args:
            n_qubits:
                Number of qubits.
            depth:
                Effective ansatz depth.
            topology:
                Entanglement topology.

        Returns:
            PennyLane QNode returning a single expectation value.
        """
        device = qml.device(
            self.device_config['name'],
            wires = n_qubits,
            shots = self.device_config["shots"]
        )

        observable = build_readout_observable(self.readout_cfg)

        @qml.qnode(device, diff_method = self.device_config["diff_method"])
        def circuit(params, features):
            apply_data_encoding(
                features= features,
                n_qubits=n_qubits,
                encoding_type=self.encoding_cfg['encoding_type'],
                feature_map=self.encoding_cfg.get("feature_map", "ry_angle")
            )

            apply_hea(
                params=params,
                n_qubits=n_qubits,
                depth=depth,
                entanglement_topology=topology,
                entangler_type=self.ansatz['entangler_type']
            )

            return qml.expval(observable)
        
        return circuit
    
    def _build_batch_loss_function(self, qnode, fixed_batch: FixedBatch):
        """
        Build differentiable batch loss function for a fixed batch.

        Args:
            qnode:
                PennyLane QNode.
            fixed_batch:
                Fixed benchmark batch.

        Returns:
            Function loss_fn(params) -> scalar loss.
        """
        X_batch = fixed_batch.X
        y_batch = fixed_batch.y

        loss_type = self.loss_cfg['type']
        eps = float(self.metrics_config.get("loss_epsilon", 1e-7))

        def loss_fn(params):
            losses = []

            for features, target in zip(X_batch, y_batch):
                z = qnode(params, features)
                p = z_expectation_to_probability(z, eps=eps)
                loss = compute_supervised_loss(
                    probability=p,
                    target=target,
                    loss_type=loss_type,
                    eps=eps
                )
                losses.append(loss)
            
            return pnp.mean(pnp.stack(losses))
        
        return loss_fn
    
    def _compute_batch_probabilities(
        self,
        qnode,
        params,
        fixed_batch: FixedBatch,
    ):
        """
        Compute probabilities for the fixed benchmark batch.

        Args:
            qnode:
                PennyLane QNode.
            params:
                Circuit parameters.
            fixed_batch:
                Fixed benchmark batch.

        Returns:
            Numpy array of probabilities for class 1.
        """

        eps = float(self.metrics_config.get('loss_epsilon', 1e-7))
        probs = []

        for features in fixed_batch.X:
            z = qnode(params, features)
            p = z_expectation_to_probability(z, eps=eps)
            probs.append(float(p))
        
        return np.asarray(probs, dtype=np.float64)
    
    def _initialize_params(
        self, 
        n_qubits: int,
        depth: int,
        seed: int
    ):
        """
        Initialize HEA parameters.

        Args:
            n_qubits:
                Number of qubits.
            depth:
                Effective depth.
            seed:
                Random seed.

        Returns:
            PennyLane numpy tensor with requires_grad=True.
        """
        rng = np.random.default_rng(seed)

        distribution = self.initialization["distribution"]
        low = float(self.initialization['low'])
        high = float(self.initialization['high'])

        shape = (depth, n_qubits, 3)
        
        if distribution == "uniform":
            values = rng.uniform(low=low, high=high, size=shape)
        elif distribution == 'normal':
            mean = 0.5 * (low + high)
            std = (high - low) / 6.0
            values = rng.normal(loc=mean, scale=std, size=shape)
        else:
            raise ValueError(f"Unsupported initialization distribution: {distribution}")
        
        return pnp.array(values)
    
    def _record_run(
        self,
        config_item: dict[str, Any],
        fixed_batch: FixedBatch,
        seed: int,
        loss_value: float,
        probabilities: np.ndarray,
        grad,
    ) -> None:
        """
        Record run-level, layer-level, and parameter-level statistics.

        Args:
            config_item:
                Circuit structure config.
            fixed_batch:
                Fixed benchmark batch.
            seed:
                Random seed.
            loss_value:
                Supervised batch loss.
            probabilities:
                Predicted class-1 probabilities on the fixed batch.
            grad:
                Gradient tensor with shape (depth, n_qubits, 3).
        """
        near_zero_threshold = float(self.metrics_config["near_zero_grad_threshold"])
        log_epsilon = float(self.metrics_config["log_epsilon"])

        n_qubits = config_item["n_qubits"]
        depth = config_item["depth"]
        n_params = count_hea_parameters(n_qubits, depth)

        run_metrics = compute_task_run_metrics(
            grad=grad,
            near_zero_threshold=near_zero_threshold,
            log_epsilon=log_epsilon,
        )

        prediction_metrics = compute_prediction_metrics(
            probabilities=probabilities,
            labels=fixed_batch.y,
        )

        current_run_id = self.run_id
        self.run_id += 1

        base_record = self._base_record(config_item, fixed_batch)

        run_record = {
            **base_record,
            "run_id": int(current_run_id),
            "seed": int(seed),
            "hilbert_dim": int(2**n_qubits),
            "n_params": int(n_params),
            "n_entangling_gates": int(config_item["actual_entangling_budget"]),
            "objective_value": float(loss_value),
            "loss_value": float(loss_value),
            **run_metrics,
            **prediction_metrics,
        }

        self.run_records.append(run_record)

        layer_records = compute_layer_metrics_task(
            run_id=current_run_id,
            grad=grad,
            near_zero_threshold=near_zero_threshold,
        )

        for record in layer_records:
            self.layer_records.append({**base_record, **record})

        self._update_online_param_stats(
            run_id=current_run_id,
            config_item=config_item,
            fixed_batch=fixed_batch,
            seed=seed,
            grad=grad,
            near_zero_threshold=near_zero_threshold,
        )

    def _base_record(
        self,
        config_item: dict[str, Any],
        fixed_batch: FixedBatch,
    ) -> dict[str, Any]:
        """
        Build common metadata record.

        Args:
            config_item:
                Circuit structure config.
            fixed_batch:
                Fixed benchmark batch.

        Returns:
            Metadata dictionary.
        """
        return {
            "benchmark_mode": self.benchmark["mode"],
            "objective_type": self.benchmark["objective_type"],
            "budget_normalization": self.benchmark.get("budget_normalization", "same_depth"),
            "experiment_group": self.experiment["group"],
            "dataset_name": self.dataset_cfg["name"],
            "target_mode": self.dataset_cfg.get("target_mode", "native"),
            "batch_id": int(fixed_batch.batch_id),
            "fixed_batch_seed": int(fixed_batch.fixed_batch_seed),
            "batch_size": int(len(fixed_batch.y)),
            "batch_class_counts": dict(fixed_batch.class_counts),
            "ansatz_type": self.ansatz["type"],
            "gate_set": self.ansatz["gate_set"],
            "encoding_type": self.encoding_cfg["encoding_type"],
            "feature_map": self.encoding_cfg.get("feature_map", "ry_angle"),
            "entanglement_topology": config_item["entanglement_topology"],
            "entangler_type": self.ansatz["entangler_type"],
            "loss_type": self.loss_cfg["type"],
            "readout_type": self.readout_cfg["type"],
            "n_qubits": int(config_item["n_qubits"]),
            "reference_depth": int(config_item["reference_depth"]),
            "depth": int(config_item["depth"]),
            "target_entangling_budget": int(config_item["target_entangling_budget"]),
            "actual_entangling_budget": int(config_item["actual_entangling_budget"]),
            "gates_per_layer": int(config_item["gates_per_layer"]),
        }

    def _update_online_param_stats(
        self,
        run_id: int,
        config_item: dict[str, Any],
        fixed_batch: FixedBatch,
        seed: int,
        grad,
        near_zero_threshold: float,
    ) -> None:
        """
        Update Welford statistics for every parameter component.

        Args:
            run_id:
                Current run ID.
            config_item:
                Circuit structure config.
            fixed_batch:
                Fixed benchmark batch.
            seed:
                Random seed.
            grad:
                Gradient tensor.
            near_zero_threshold:
                Near-zero threshold.
        """
        base_record = self._base_record(config_item, fixed_batch)

        for (
            layer_id,
            qubit_id,
            axis_id,
            axis_name,
            param_index,
            grad_value,
        ) in iter_gradient_components(grad):
            param_id = self._make_param_id(
                layer_id=layer_id,
                qubit_id=qubit_id,
                axis_name=axis_name,
            )

            key = (
                base_record["benchmark_mode"],
                base_record["objective_type"],
                base_record["experiment_group"],
                base_record["dataset_name"],
                base_record["target_mode"],
                base_record["batch_id"],
                base_record["ansatz_type"],
                base_record["gate_set"],
                base_record["encoding_type"],
                base_record["feature_map"],
                base_record["entanglement_topology"],
                base_record["entangler_type"],
                base_record["loss_type"],
                base_record["readout_type"],
                base_record["n_qubits"],
                base_record["reference_depth"],
                base_record["depth"],
                base_record["budget_normalization"],
                layer_id,
                qubit_id,
                axis_name,
                axis_id,
                param_index,
                param_id,
            )

            self.gradient_stats[key].update(
                value=grad_value,
                near_zero_threshold=near_zero_threshold,
            )

            if self.logging_config["save_raw_gradients"]:
                self.raw_gradient_records.append(
                    {
                        **base_record,
                        "run_id": int(run_id),
                        "seed": int(seed),
                        "layer_id": int(layer_id),
                        "qubit_id": int(qubit_id),
                        "gate_type": axis_name,
                        "axis_id": int(axis_id),
                        "param_index": int(param_index),
                        "param_id": param_id,
                        "grad_value": float(grad_value),
                        "abs_grad": float(abs(grad_value)),
                        "is_near_zero": bool(abs(grad_value) < near_zero_threshold),
                    }
                )

    @staticmethod
    def _make_param_id(layer_id: int, qubit_id: int, axis_name: str) -> str:
        """
        Create stable parameter ID.

        Args:
            layer_id:
                Layer index.
            qubit_id:
                Qubit index.
            axis_name:
                RX, RY, or RZ.

        Returns:
            Stable parameter ID string.
        """
        return f"layer_{layer_id:03d}_wire_{qubit_id:03d}_{axis_name}"

    def _build_param_variance_df(self) -> pd.DataFrame:
        """
        Build parameter-level variance DataFrame.

        Returns:
            DataFrame containing Var_seed[dL_B/dθ_k].
        """
        records: list[dict[str, Any]] = []

        for key, stats in self.gradient_stats.items():
            (
                benchmark_mode,
                objective_type,
                experiment_group,
                dataset_name,
                target_mode,
                batch_id,
                ansatz_type,
                gate_set,
                encoding_type,
                feature_map,
                topology,
                entangler_type,
                loss_type,
                readout_type,
                n_qubits,
                reference_depth,
                depth,
                budget_normalization,
                layer_id,
                qubit_id,
                axis_name,
                axis_id,
                param_index,
                param_id,
            ) = key

            records.append(
                {
                    "benchmark_mode": benchmark_mode,
                    "objective_type": objective_type,
                    "experiment_group": experiment_group,
                    "dataset_name": dataset_name,
                    "target_mode": target_mode,
                    "batch_id": int(batch_id),
                    "ansatz_type": ansatz_type,
                    "gate_set": gate_set,
                    "encoding_type": encoding_type,
                    "feature_map": feature_map,
                    "entanglement_topology": topology,
                    "entangler_type": entangler_type,
                    "loss_type": loss_type,
                    "readout_type": readout_type,
                    "n_qubits": int(n_qubits),
                    "reference_depth": int(reference_depth),
                    "depth": int(depth),
                    "budget_normalization": budget_normalization,
                    "layer_id": int(layer_id),
                    "qubit_id": int(qubit_id),
                    "gate_type": axis_name,
                    "axis_id": int(axis_id),
                    "param_index": int(param_index),
                    "param_id": param_id,
                    "n_seeds": int(stats.count),
                    "param_grad_mean": float(stats.mean),
                    "param_grad_variance_across_seeds": float(stats.variance),
                    "param_mean_abs_grad": float(stats.mean_abs),
                    "param_near_zero_ratio": float(stats.near_zero_ratio),
                }
            )

        return pd.DataFrame(records)

    def _build_summary_df(
        self,
        run_df: pd.DataFrame,
        layer_df: pd.DataFrame,
        param_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build structure-level summary table.

        Args:
            run_df:
                Run-level DataFrame.
            layer_df:
                Layer-level DataFrame.
            param_df:
                Parameter variance DataFrame.

        Returns:
            Summary DataFrame.
        """
        group_cols = [
            "benchmark_mode",
            "objective_type",
            "experiment_group",
            "dataset_name",
            "target_mode",
            "batch_id",
            "ansatz_type",
            "gate_set",
            "encoding_type",
            "feature_map",
            "entanglement_topology",
            "entangler_type",
            "loss_type",
            "readout_type",
            "n_qubits",
            "reference_depth",
            "depth",
            "budget_normalization",
        ]

        param_summary = (
            param_df.groupby(group_cols, dropna=False)
            .agg(
                mean_param_grad_variance_across_seeds=(
                    "param_grad_variance_across_seeds",
                    "mean",
                ),
                median_param_grad_variance_across_seeds=(
                    "param_grad_variance_across_seeds",
                    "median",
                ),
                min_param_grad_variance_across_seeds=(
                    "param_grad_variance_across_seeds",
                    "min",
                ),
                max_param_grad_variance_across_seeds=(
                    "param_grad_variance_across_seeds",
                    "max",
                ),
                mean_param_abs_grad=("param_mean_abs_grad", "mean"),
                mean_param_near_zero_ratio=("param_near_zero_ratio", "mean"),
                n_params=("param_id", "nunique"),
                n_seeds=("n_seeds", "max"),
            )
            .reset_index()
        )

        log_eps = float(self.metrics_config["log_epsilon"])

        param_summary["log10_mean_param_grad_variance_across_seeds"] = param_summary[
            "mean_param_grad_variance_across_seeds"
        ].apply(lambda x: math.log10(max(float(x), log_eps)))

        param_summary["log10_median_param_grad_variance_across_seeds"] = param_summary[
            "median_param_grad_variance_across_seeds"
        ].apply(lambda x: math.log10(max(float(x), log_eps)))

        run_summary = (
            run_df.groupby(group_cols, dropna=False)
            .agg(
                n_runs=("run_id", "nunique"),
                mean_loss_value=("loss_value", "mean"),
                mean_batch_accuracy=("batch_accuracy", "mean"),
                mean_spatial_grad_variance=("spatial_grad_variance", "mean"),
                mean_grad_norm=("grad_norm", "mean"),
                mean_normalized_grad_norm=("normalized_grad_norm", "mean"),
                mean_abs_grad=("mean_abs_grad", "mean"),
                mean_near_zero_ratio=("near_zero_ratio", "mean"),
                mean_n_entangling_gates=("n_entangling_gates", "mean"),
                mean_pred_probability=("mean_pred_probability", "mean"),
                std_pred_probability=("std_pred_probability", "mean"),
            )
            .reset_index()
        )

        layer_summary = (
            layer_df.groupby(group_cols, dropna=False)
            .agg(
                mean_layer_grad_norm=("layer_grad_norm", "mean"),
                max_layer_near_zero_ratio=("layer_near_zero_ratio", "max"),
                mean_layer_near_zero_ratio=("layer_near_zero_ratio", "mean"),
            )
            .reset_index()
        )

        summary = param_summary.merge(run_summary, on=group_cols, how="left")
        summary = summary.merge(layer_summary, on=group_cols, how="left")

        return summary

    def _save_outputs(
        self,
        run_df: pd.DataFrame,
        layer_df: pd.DataFrame,
        param_variance_df: pd.DataFrame,
        summary_df: pd.DataFrame,
    ) -> None:
        """
        Save Stage 1B output artifacts.

        Args:
            run_df:
                Run-level table.
            layer_df:
                Layer-level table.
            param_variance_df:
                Parameter variance table.
            summary_df:
                Summary table.
        """
        save_table(run_df, self.outputs["run_level_path"])
        save_table(layer_df, self.outputs["layer_level_path"])
        save_table(param_variance_df, self.outputs["param_variance_path"])
        save_table(summary_df, self.outputs["summary_path"])

        ensure_plot_dir(self.outputs["plot_dir"])

        if self.logging_config["save_raw_gradients"]:
            raw_df = pd.DataFrame(self.raw_gradient_records)
            save_table(raw_df, self.outputs["raw_gradient_path"])