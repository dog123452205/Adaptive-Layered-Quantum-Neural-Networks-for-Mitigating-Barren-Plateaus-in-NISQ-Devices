'''
Stage 1A random-parameter barren plateau benchmark engine

This engine computes:
    Var_seed[dE(θ)/dθ_k]

where:
    E(θ) = <0| U†(θ) O U(θ) |0>

The engine uses Welford online aggregation and does not require
raw gradient CSV in full benchmark mode.
'''

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

from src.synthetic_bp.constants import (
    BenchmarkMode,
    EncodingType,
    ObjectiveType,
)

from src.synthetic_bp.stage1a.ansatz import apply_hea
from src.synthetic_bp.stage1a.budget import (
    count_hea_parameters,
    plan_effective_depth
)

from src.synthetic_bp.stage1a.io_utils import save_table, ensure_plot_dir
from src.synthetic_bp.stage1a.metrics import (
    compute_gradient_metrics,
    compute_layerwise_metrics,
    iter_gradient_components
)

from src.synthetic_bp.stage1a.observables import build_observable
from src.synthetic_bp.stage1a.online_stats import OnlineGradientStats


class Stage1ARandomBPEngine:
    """
    Random-parameter BP benchmark runner.

    Args:
        config:
            Resolved and validated Stage 1A config.

    Outputs:
        - run-level table
        - layer-level table
        - optional raw-gradient table
        - parameter variance table
        - summary table
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

        self.experiment = config["experiment"]
        self.benchmark = config["benchmark"]
        self.ansatz = config["ansatz"]
        self.sweep = config["sweep"]
        self.initialization = config["initialization"]
        self.device_config = config["device"]
        self.metrics_config = config["metrics"]
        self.logging_config = config["logging"]
        self.outputs = config["outputs"]

        self.run_records: list[dict[str, Any]] = []
        self.layer_records: list[dict[str, Any]] = []
        self.raw_gradient_records: list[dict[str, Any]] = []

        self.gradient_stats: dict[tuple[Any, ...], OnlineGradientStats] = defaultdict(
            OnlineGradientStats
        )

        self.run_id = 0

    def run(self) -> dict[str, pd.DataFrame]:
        """
        Execute the full Stage 1A benchmark.

        Returns:
            Dictionary of output DataFrames.
        """
        self._validate_stage1a_assumptions()

        configs = list(self._iter_benchmark_configs())

        for config_item in tqdm(configs, desc="Stage 1A random BP"):
            self._run_single_structure(config_item)

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

        if self.logging_config["save_raw_gradients"]:
            result["raw_gradients"] = pd.DataFrame(self.raw_gradient_records)

        return result

    def _validate_stage1a_assumptions(self) -> None:
        """
        Enforce Stage 1A assumptions.

        Stage 1A must be:
            - random_bp
            - expectation_value objective
            - encoding_type none
        """
        if self.benchmark["mode"] != BenchmarkMode.RANDOM_BP.value:
            raise ValueError("Stage1ARandomBPEngine requires benchmark.mode=random_bp.")

        if self.benchmark["objective_type"] != ObjectiveType.EXPECTATION_VALUE.value:
            raise ValueError(
                "Stage1ARandomBPEngine requires objective_type=expectation_value."
            )

        if self.ansatz["encoding_type"] != EncodingType.NONE.value:
            raise ValueError("Stage 1A must use ansatz.encoding_type=none.")

    def _iter_benchmark_configs(self):
        """
        Iterate over benchmark structures, excluding seed.

        Yields:
            Dictionary describing one circuit structure.
        """
        n_qubits_list = self.sweep["n_qubits"]
        depths = self.sweep["depths"]
        cost_types = self.sweep["cost_types"]
        topologies = self.ansatz["entanglement_topologies"]

        for n_qubits, reference_depth, cost_type, topology in itertools.product(
            n_qubits_list,
            depths,
            cost_types,
            topologies,
        ):
            depth_plan = plan_effective_depth(
                n_qubits=n_qubits,
                reference_depth=reference_depth,
                topology=topology,
                budget_normalization=self.benchmark["budget_normalization"],
            )

            yield {
                "n_qubits": int(n_qubits),
                "reference_depth": int(reference_depth),
                "depth": int(depth_plan.effective_depth),
                "cost_type": cost_type,
                "entanglement_topology": topology,
                "target_entangling_budget": int(depth_plan.target_entangling_budget),
                "actual_entangling_budget": int(depth_plan.actual_entangling_budget),
                "gates_per_layer": int(depth_plan.gates_per_layer),
            }

    def _run_single_structure(self, config_item: dict[str, Any]) -> None:
        """
        Run all seeds for one circuit structure.

        Args:
            config_item:
                Structure config excluding seed.
        """
        n_qubits = config_item["n_qubits"]
        depth = config_item["depth"]
        cost_type = config_item["cost_type"]
        topology = config_item["entanglement_topology"]

        qnode = self._build_qnode(
            n_qubits=n_qubits,
            depth=depth,
            cost_type=cost_type,
            topology=topology,
        )

        seeds = range(int(self.sweep["seeds"]))

        for seed in seeds:
            params = self._initialize_params(
                n_qubits=n_qubits,
                depth=depth,
                seed=seed,
            )

            objective_value = float(qnode(params))
            grad = qml.grad(qnode)(params)

            self._record_run(
                config_item=config_item,
                seed=seed,
                objective_value=objective_value,
                grad=grad,
            )

    def _build_qnode(
        self,
        n_qubits: int,
        depth: int,
        cost_type: str,
        topology: str,
    ):
        """
        Build a QNode for one circuit structure.

        Args:
            n_qubits:
                Number of qubits.
            depth:
                Effective depth.
            cost_type:
                Cost observable type.
            topology:
                Entanglement topology.

        Returns:
            PennyLane QNode.
        """
        device = qml.device(
            self.device_config["name"],
            wires=n_qubits,
            shots=self.device_config["shots"],
        )

        observable = build_observable(n_qubits, cost_type)

        @qml.qnode(device, diff_method=self.device_config["diff_method"])
        def circuit(params):
            apply_hea(
                params=params,
                n_qubits=n_qubits,
                depth=depth,
                entanglement_topology=topology,
                entangler_type=self.ansatz["entangler_type"],
            )
            return qml.expval(observable)

        return circuit

    def _initialize_params(
        self,
        n_qubits: int,
        depth: int,
        seed: int,
    ):
        """
        Initialize random HEA parameters.

        Args:
            n_qubits:
                Number of qubits.
            depth:
                Effective depth.
            seed:
                Random seed.

        Returns:
            PennyLane numpy array with requires_grad=True.
        """
        rng = np.random.default_rng(seed)

        distribution = self.initialization["distribution"]
        low = float(self.initialization["low"])
        high = float(self.initialization["high"])

        shape = (depth, n_qubits, 3)

        if distribution == "uniform":
            values = rng.uniform(low=low, high=high, size=shape)
        elif distribution == "normal":
            mean = 0.5 * (low + high)
            std = (high - low) / 6.0
            values = rng.normal(loc=mean, scale=std, size=shape)
        else:
            raise ValueError(f"Unsupported initialization distribution: {distribution}")

        return pnp.array(values, requires_grad=True)

    def _record_run(
        self,
        config_item: dict[str, Any],
        seed: int,
        objective_value: float,
        grad,
    ) -> None:
        """
        Record run-level, layer-level, and parameter-level statistics.

        Args:
            config_item:
                Structure config.
            seed:
                Random seed.
            objective_value:
                Scalar expectation value.
            grad:
                Gradient tensor.
        """
        near_zero_threshold = float(self.metrics_config["near_zero_grad_threshold"])
        log_epsilon = float(self.metrics_config["log_epsilon"])

        n_qubits = config_item["n_qubits"]
        depth = config_item["depth"]

        n_params = count_hea_parameters(n_qubits, depth)

        run_metrics = compute_gradient_metrics(
            grad=grad,
            near_zero_threshold=near_zero_threshold,
            log_epsilon=log_epsilon,
        )

        current_run_id = self.run_id
        self.run_id += 1

        base_record = self._base_record(config_item)

        run_record = {
            **base_record,
            "run_id": int(current_run_id),
            "seed": int(seed),
            "hilbert_dim": int(2**n_qubits),
            "n_params": int(n_params),
            "n_entangling_gates": int(config_item["actual_entangling_budget"]),
            "objective_value": float(objective_value),
            **run_metrics,
        }

        self.run_records.append(run_record)

        layer_metrics = compute_layerwise_metrics(
            run_id=current_run_id,
            grad=grad,
            near_zero_threshold=near_zero_threshold,
        )

        for record in layer_metrics:
            self.layer_records.append({**base_record, **record})

        self._update_online_param_stats(
            run_id=current_run_id,
            config_item=config_item,
            seed=seed,
            grad=grad,
            near_zero_threshold=near_zero_threshold,
        )

    def _base_record(self, config_item: dict[str, Any]) -> dict[str, Any]:
        """
        Build common metadata record for outputs.

        Args:
            config_item:
                Structure config.

        Returns:
            Metadata dictionary.
        """
        return {
            "benchmark_mode": self.benchmark["mode"],
            "objective_type": self.benchmark["objective_type"],
            "budget_normalization": self.benchmark["budget_normalization"],
            "experiment_group": self.experiment["group"],
            "ansatz_type": self.ansatz["type"],
            "gate_set": self.ansatz["gate_set"],
            "encoding_type": self.ansatz["encoding_type"],
            "entanglement_topology": config_item["entanglement_topology"],
            "entangler_type": self.ansatz["entangler_type"],
            "cost_type": config_item["cost_type"],
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
        seed: int,
        grad,
        near_zero_threshold: float,
    ) -> None:
        """
        Update Welford stats for every parameter component.

        Args:
            run_id:
                Current run ID.
            config_item:
                Structure config.
            seed:
                Random seed.
            grad:
                Gradient tensor.
            near_zero_threshold:
                Near-zero threshold.
        """
        base_record = self._base_record(config_item)

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
                base_record["ansatz_type"],
                base_record["gate_set"],
                base_record["encoding_type"],
                base_record["entanglement_topology"],
                base_record["entangler_type"],
                base_record["cost_type"],
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
            Stable parameter ID.
        """
        return f"layer_{layer_id:03d}_wire_{qubit_id:03d}_{axis_name}"

    def _build_param_variance_df(self) -> pd.DataFrame:
        """
        Build parameter-level variance DataFrame from Welford states.

        Returns:
            Parameter variance DataFrame.
        """
        records: list[dict[str, Any]] = []

        for key, stats in self.gradient_stats.items():
            (
                benchmark_mode,
                objective_type,
                experiment_group,
                ansatz_type,
                gate_set,
                encoding_type,
                topology,
                entangler_type,
                cost_type,
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
                    "ansatz_type": ansatz_type,
                    "gate_set": gate_set,
                    "encoding_type": encoding_type,
                    "entanglement_topology": topology,
                    "entangler_type": entangler_type,
                    "cost_type": cost_type,
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
        Build config-level summary table.

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
            "ansatz_type",
            "gate_set",
            "encoding_type",
            "entanglement_topology",
            "entangler_type",
            "cost_type",
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
                mean_objective_value=("objective_value", "mean"),
                mean_spatial_grad_variance=("spatial_grad_variance", "mean"),
                mean_grad_norm=("grad_norm", "mean"),
                mean_normalized_grad_norm=("normalized_grad_norm", "mean"),
                mean_abs_grad=("mean_abs_grad", "mean"),
                mean_near_zero_ratio=("near_zero_ratio", "mean"),
                mean_n_entangling_gates=("n_entangling_gates", "mean"),
                mean_actual_entangling_budget=("actual_entangling_budget", "mean"),
                mean_target_entangling_budget=("target_entangling_budget", "mean"),
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
        Save Stage 1A artifacts.

        Args:
            run_df:
                Run-level DataFrame.
            layer_df:
                Layer-level DataFrame.
            param_variance_df:
                Parameter variance DataFrame.
            summary_df:
                Summary DataFrame.
        """
        save_table(run_df, self.outputs["run_level_path"])
        save_table(layer_df, self.outputs["layer_level_path"])
        save_table(param_variance_df, self.outputs["param_variance_path"])
        save_table(summary_df, self.outputs["summary_path"])

        ensure_plot_dir(self.outputs["plot_dir"])

        if self.logging_config["save_raw_gradients"]:
            raw_df = pd.DataFrame(self.raw_gradient_records)
            save_table(raw_df, self.outputs["raw_gradient_path"])