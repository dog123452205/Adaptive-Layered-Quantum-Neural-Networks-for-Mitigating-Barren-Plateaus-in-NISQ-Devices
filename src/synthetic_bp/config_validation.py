"""
Lightweight config validators for AL-QNN.

Validators check:
- required keys,
- enum values,
- scalar ranges,
- logical consistency,
- output paths.

Validators must not:
- import PennyLane,
- build QNodes,
- call qml.state(),
- call qml.density_matrix(),
- call qml.matrix(),
- construct full unitaries or Hamiltonian matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.synthetic_bp.constants import (
    AnsatzType,
    BatchBenchmarkMode,
    BenchmarkMode,
    BudgetNormalizationMode,
    CostType,
    DatasetName,
    DeviceName,
    DiffFallbackPolicy,
    DiffMethod,
    EncodingType,
    EntanglerType,
    EntanglementTopology,
    FileFormat,
    GateSet,
    GradientLoggingMode,
    InitDistribution,
    InitStrategy,
    LossType,
    MaskMode,
    MaskSchedule,
    ObjectiveType,
    PruneMode,
    ReadoutType,
    ResourceLimitPolicy,
    SchedulerAction,
    SchedulerType,
    SimulationProfile,
    enum_values,
)


def require_keys(section: dict[str, Any], keys: Iterable[str], context: str) -> None:
    """
    Ensure required keys exist in a config section.

    Args:
        section: Config section.
        keys: Required key names.
        context: Human-readable section name.

    Raises:
        ValueError: If keys are missing.
    """
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(f"{context} missing required keys: {missing}")


def validate_enum(value: Any, allowed: set[str], field_name: str) -> None:
    """
    Validate scalar enum value.

    Args:
        value: Value to validate.
        allowed: Allowed string values.
        field_name: Human-readable field name.
    """
    if value not in allowed:
        raise ValueError(f"{field_name}={value!r} not in {sorted(allowed)}")


def validate_enum_list(values: Any, allowed: set[str], field_name: str) -> None:
    """
    Validate list of enum values.

    Args:
        values: List of values.
        allowed: Allowed string values.
        field_name: Human-readable field name.
    """
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list.")

    invalid = set(values) - allowed
    if invalid:
        raise ValueError(f"{field_name} contains invalid values: {sorted(invalid)}")


def validate_positive_int(value: Any, field_name: str) -> None:
    """Validate positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def validate_non_negative_int(value: Any, field_name: str) -> None:
    """Validate non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def validate_positive_float(value: Any, field_name: str) -> None:
    """Validate positive finite number."""
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be a positive finite number.")


def validate_probability(value: Any, field_name: str) -> None:
    """Validate value in [0, 1]."""
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be in [0, 1].")


def validate_config(config: dict[str, Any]) -> str:
    """
    Validate a resolved config and infer its type.

    Args:
        config: Resolved config.

    Returns:
        Config type:
            - stage1a_random_bp
            - stage1b_task_aware_bp
            - adaptive_alqnn

    Raises:
        ValueError: If config is invalid.
    """
    validate_common_sections(config)

    if "benchmark" in config:
        mode = config["benchmark"].get("mode")

        if mode == BenchmarkMode.RANDOM_BP.value:
            validate_stage1a_random_bp_config(config)
            return "stage1a_random_bp"

        if mode == BenchmarkMode.TASK_AWARE_BP.value:
            validate_stage1b_task_aware_bp_config(config)
            return "stage1b_task_aware_bp"

    if "scheduler" in config and "architecture" in config:
        validate_adaptive_alqnn_config(config)
        return "adaptive_alqnn"

    raise ValueError("Could not infer config type.")


def validate_common_sections(config: dict[str, Any]) -> None:
    """Validate sections common to all configs."""
    require_keys(config, ["experiment", "safety", "device", "outputs"], "config")

    validate_experiment(config["experiment"])
    validate_safety(config["safety"])
    validate_device(config["device"])


def validate_experiment(section: dict[str, Any]) -> None:
    """Validate experiment section."""
    require_keys(section, ["name", "group"], "experiment")

    for key in ["name", "group"]:
        if not isinstance(section[key], str) or not section[key].strip():
            raise ValueError(f"experiment.{key} must be a non-empty string.")


def validate_safety(section: dict[str, Any]) -> None:
    """Validate safety section."""
    require_keys(
        section,
        [
            "simulation_profile",
            "enforce_resource_limits",
            "resource_limit_policy",
        ],
        "safety",
    )

    validate_enum(
        section["simulation_profile"],
        enum_values(SimulationProfile),
        "safety.simulation_profile",
    )
    validate_enum(
        section["resource_limit_policy"],
        enum_values(ResourceLimitPolicy),
        "safety.resource_limit_policy",
    )

    if not isinstance(section["enforce_resource_limits"], bool):
        raise ValueError("safety.enforce_resource_limits must be boolean.")


def validate_device(section: dict[str, Any]) -> None:
    """Validate device section."""
    require_keys(
        section,
        [
            "name",
            "diff_method",
            "diff_fallback_policy",
            "fallback_diff_method",
            "shots",
        ],
        "device",
    )

    validate_enum(section["name"], enum_values(DeviceName), "device.name")
    validate_enum(section["diff_method"], enum_values(DiffMethod), "device.diff_method")
    validate_enum(
        section["diff_fallback_policy"],
        enum_values(DiffFallbackPolicy),
        "device.diff_fallback_policy",
    )
    validate_enum(
        section["fallback_diff_method"],
        enum_values(DiffMethod),
        "device.fallback_diff_method",
    )

    shots = section["shots"]
    if shots is not None:
        validate_positive_int(shots, "device.shots")


def validate_stage1a_random_bp_config(config: dict[str, Any]) -> None:
    """Validate Stage 1A random-parameter BP config."""
    require_keys(
        config,
        [
            "benchmark",
            "ansatz",
            "sweep",
            "initialization",
            "metrics",
            "logging",
            "outputs",
        ],
        "Stage 1A config",
    )

    benchmark = config["benchmark"]
    validate_enum(benchmark["mode"], {BenchmarkMode.RANDOM_BP.value}, "benchmark.mode")
    validate_enum(
        benchmark["objective_type"],
        {ObjectiveType.EXPECTATION_VALUE.value},
        "benchmark.objective_type",
    )
    validate_enum(
        benchmark["budget_normalization"],
        enum_values(BudgetNormalizationMode),
        "benchmark.budget_normalization",
    )

    validate_ansatz(config["ansatz"], stage="stage1a")
    validate_stage1a_sweep(config["sweep"], config["ansatz"])
    validate_initialization(config["initialization"])
    validate_metrics(config["metrics"])
    validate_logging(config["logging"])
    validate_stage1a_outputs(config["outputs"], config["logging"])

    if config["ansatz"]["encoding_type"] != EncodingType.NONE.value:
        raise ValueError("Stage 1A random_bp must use ansatz.encoding_type='none'.")


def validate_stage1b_task_aware_bp_config(config: dict[str, Any]) -> None:
    """Validate Stage 1B task-aware BP config."""
    require_keys(
        config,
        [
            "benchmark",
            "dataset",
            "preprocessing",
            "encoding",
            "ansatz",
            "sweep",
            "loss",
            "readout",
            "batching",
            "initialization",
            "metrics",
            "logging",
            "outputs",
        ],
        "Stage 1B config",
    )

    benchmark = config["benchmark"]
    validate_enum(benchmark["mode"], {BenchmarkMode.TASK_AWARE_BP.value}, "benchmark.mode")
    validate_enum(
        benchmark["objective_type"],
        {ObjectiveType.SUPERVISED_LOSS.value},
        "benchmark.objective_type",
    )

    validate_dataset(config["dataset"])
    validate_encoding(config["encoding"])
    validate_ansatz(config["ansatz"], stage="stage1b")
    validate_stage1b_sweep(config["sweep"], config["ansatz"])
    validate_loss(config["loss"])
    validate_readout(config["readout"])
    validate_batching(config["batching"])
    validate_initialization(config["initialization"])
    validate_metrics(config["metrics"])
    validate_logging(config["logging"])
    validate_stage1a_outputs(config["outputs"], config["logging"])


def validate_adaptive_alqnn_config(config: dict[str, Any]) -> None:
    """Validate Stage 2/3 adaptive AL-QNN config."""
    require_keys(
        config,
        [
            "dataset",
            "architecture",
            "training",
            "telemetry",
            "scheduler",
            "logging",
          
        ],
        "adaptive AL-QNN config",
    )

    validate_dataset(config["dataset"])
    validate_adaptive_architecture(config["architecture"])
    validate_training(config["training"])
    validate_scheduler(config["scheduler"], config["architecture"])
    

def validate_ansatz(section: dict[str, Any], stage: str) -> None:
    """Validate ansatz section."""
    require_keys(
        section,
        [
            "type",
            "gate_set",
            "encoding_type",
            "entanglement_topologies",
            "entangler_type",
            "shallow_depth",
            "max_depth",
        ],
        f"{stage}.ansatz",
    )

    validate_enum(section["type"], enum_values(AnsatzType), "ansatz.type")
    validate_enum(section["gate_set"], enum_values(GateSet), "ansatz.gate_set")
    validate_enum(section["encoding_type"], enum_values(EncodingType), "ansatz.encoding_type")
    validate_enum_list(
        section["entanglement_topologies"],
        enum_values(EntanglementTopology),
        "ansatz.entanglement_topologies",
    )
    validate_enum(section["entangler_type"], enum_values(EntanglerType), "ansatz.entangler_type")

    validate_positive_int(section["shallow_depth"], "ansatz.shallow_depth")
    validate_positive_int(section["max_depth"], "ansatz.max_depth")

    if section["shallow_depth"] > section["max_depth"]:
        raise ValueError("ansatz.shallow_depth cannot exceed ansatz.max_depth.")


def validate_stage1a_sweep(section: dict[str, Any], ansatz: dict[str, Any]) -> None:
    """Validate Stage 1A sweep."""
    require_keys(section, ["n_qubits", "depths", "cost_types", "seeds"], "sweep")

    validate_positive_int(section["seeds"], "sweep.seeds")
    if section["seeds"] <= 1:
        raise ValueError("sweep.seeds must be > 1 for ensemble variance.")

    validate_positive_int_list(section["n_qubits"], "sweep.n_qubits")
    validate_positive_int_list(section["depths"], "sweep.depths")
    validate_enum_list(section["cost_types"], enum_values(CostType), "sweep.cost_types")

    if max(section["depths"]) > ansatz["max_depth"]:
        raise ValueError("sweep.depths cannot exceed ansatz.max_depth.")


def validate_stage1b_sweep(section: dict[str, Any], ansatz: dict[str, Any]) -> None:
    """Validate Stage 1B sweep."""
    require_keys(section, ["n_qubits", "depths", "seeds"], "sweep")

    validate_positive_int(section["seeds"], "sweep.seeds")
    if section["seeds"] <= 1:
        raise ValueError("sweep.seeds must be > 1 for ensemble variance.")

    validate_positive_int_list(section["n_qubits"], "sweep.n_qubits")
    validate_positive_int_list(section["depths"], "sweep.depths")

    if max(section["depths"]) > ansatz["max_depth"]:
        raise ValueError("sweep.depths cannot exceed ansatz.max_depth.")


def validate_positive_int_list(values: Any, field_name: str) -> None:
    """Validate non-empty list of positive integers."""
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list.")

    for value in values:
        validate_positive_int(value, field_name)


def validate_initialization(section: dict[str, Any]) -> None:
    """Validate initialization section."""
    require_keys(section, ["strategy", "distribution", "low", "high"], "initialization")

    validate_enum(section["strategy"], enum_values(InitStrategy), "initialization.strategy")
    validate_enum(section["distribution"], enum_values(InitDistribution), "initialization.distribution")

    low = section["low"]
    high = section["high"]

    if not isinstance(low, (int, float)) or not np.isfinite(low):
        raise ValueError("initialization.low must be finite.")

    if not isinstance(high, (int, float)) or not np.isfinite(high):
        raise ValueError("initialization.high must be finite.")

    if low >= high:
        raise ValueError("initialization.low must be smaller than initialization.high.")

    if high - low > 4 * np.pi:
        raise ValueError("Initialization range is unusually large.")


def validate_metrics(section: dict[str, Any]) -> None:
    """Validate metrics section."""
    require_keys(section, ["log_epsilon", "near_zero_grad_threshold"], "metrics")
    validate_positive_float(section["log_epsilon"], "metrics.log_epsilon")
    validate_positive_float(
        section["near_zero_grad_threshold"],
        "metrics.near_zero_grad_threshold",
    )


def validate_logging(section: dict[str, Any]) -> None:
    """Validate logging section."""
    require_keys(
        section,
        ["gradient_logging_mode", "save_raw_gradients", "raw_gradient_format", "online_aggregation"],
        "logging",
    )

    validate_enum(
        section["gradient_logging_mode"],
        enum_values(GradientLoggingMode),
        "logging.gradient_logging_mode",
    )
    validate_enum(
        section["raw_gradient_format"],
        {FileFormat.CSV.value, FileFormat.PARQUET.value},
        "logging.raw_gradient_format",
    )

    if not isinstance(section["save_raw_gradients"], bool):
        raise ValueError("logging.save_raw_gradients must be boolean.")

    if not isinstance(section["online_aggregation"], bool):
        raise ValueError("logging.online_aggregation must be boolean.")

    if section["gradient_logging_mode"] == GradientLoggingMode.ONLINE_SUMMARY.value:
        if section["save_raw_gradients"]:
            raise ValueError("online_summary mode must not save full raw gradients.")


def validate_stage1a_outputs(outputs: dict[str, Any], logging: dict[str, Any]) -> None:
    """Validate Stage 1 output paths."""
    required = [
        "run_level_path",
        "layer_level_path",
        "param_variance_path",
        "summary_path",
        "plot_dir",
    ]
    require_keys(outputs, required, "outputs")

    if logging.get("save_raw_gradients", False):
        require_keys(outputs, ["raw_gradient_path"], "outputs")

    for key, path in outputs.items():
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"outputs.{key} must be a non-empty string.")

    validate_suffix(outputs["run_level_path"], [".csv"], "outputs.run_level_path")
    validate_suffix(outputs["layer_level_path"], [".csv"], "outputs.layer_level_path")
    validate_suffix(outputs["summary_path"], [".csv", ".parquet"], "outputs.summary_path")
    validate_suffix(outputs["param_variance_path"], [".csv", ".parquet"], "outputs.param_variance_path")

    if "raw_gradient_path" in outputs:
        validate_suffix(outputs["raw_gradient_path"], [".csv", ".parquet"], "outputs.raw_gradient_path")


def validate_suffix(path: str, allowed_suffixes: list[str], field_name: str) -> None:
    """Validate file suffix."""
    suffix = Path(path).suffix
    if suffix not in allowed_suffixes:
        raise ValueError(f"{field_name} must end with {allowed_suffixes}, got {path}")


def validate_dataset(section: dict[str, Any]) -> None:
    """Validate dataset section."""
    require_keys(section, ["name", "task_type", "random_state"], "dataset")
    validate_enum(section["name"], enum_values(DatasetName), "dataset.name")

    if not isinstance(section["random_state"], int):
        raise ValueError("dataset.random_state must be integer.")


def validate_encoding(section: dict[str, Any]) -> None:
    """Validate encoding section."""
    require_keys(section, ["encoding_type"], "encoding")
    validate_enum(section["encoding_type"], enum_values(EncodingType), "encoding.encoding_type")

    if section["encoding_type"] == EncodingType.NONE.value:
        raise ValueError("Stage 1B task-aware benchmark cannot use encoding_type='none'.")


def validate_loss(section: dict[str, Any]) -> None:
    """Validate loss section."""
    require_keys(section, ["type"], "loss")
    validate_enum(section["type"], enum_values(LossType), "loss.type")


def validate_readout(section: dict[str, Any]) -> None:
    """Validate readout section."""
    require_keys(section, ["type"], "readout")
    validate_enum(section["type"], enum_values(ReadoutType), "readout.type")


def validate_batching(section: dict[str, Any]) -> None:
    """Validate fixed-batch benchmark protocol."""
    require_keys(
        section,
        ["benchmark_batch_mode", "batch_size", "n_fixed_batches", "fixed_batch_seed"],
        "batching",
    )

    validate_enum(
        section["benchmark_batch_mode"],
        enum_values(BatchBenchmarkMode),
        "batching.benchmark_batch_mode",
    )
    validate_positive_int(section["batch_size"], "batching.batch_size")
    validate_positive_int(section["n_fixed_batches"], "batching.n_fixed_batches")

    if not isinstance(section["fixed_batch_seed"], int):
        raise ValueError("batching.fixed_batch_seed must be integer.")


def validate_adaptive_architecture(section: dict[str, Any]) -> None:
    """Validate adaptive architecture section."""
    require_keys(
        section,
        [
            "model_type",
            "ansatz_type",
            "gate_set",
            "encoding_type",
            "n_qubits",
            "initial_depth",
            "min_depth",
            "max_depth",
            "initial_entanglement_topology",
            "allowed_entanglement_topologies",
            "entangler_type",
        ],
        "architecture",
    )

    validate_enum(section["ansatz_type"], enum_values(AnsatzType), "architecture.ansatz_type")
    validate_enum(section["gate_set"], enum_values(GateSet), "architecture.gate_set")
    validate_enum(section["encoding_type"], enum_values(EncodingType), "architecture.encoding_type")
    validate_enum(
        section["initial_entanglement_topology"],
        enum_values(EntanglementTopology),
        "architecture.initial_entanglement_topology",
    )
    validate_enum_list(
        section["allowed_entanglement_topologies"],
        enum_values(EntanglementTopology),
        "architecture.allowed_entanglement_topologies",
    )
    validate_enum(section["entangler_type"], enum_values(EntanglerType), "architecture.entangler_type")

    validate_positive_int(section["n_qubits"], "architecture.n_qubits")
    validate_positive_int(section["initial_depth"], "architecture.initial_depth")
    validate_positive_int(section["min_depth"], "architecture.min_depth")
    validate_positive_int(section["max_depth"], "architecture.max_depth")

    if section["min_depth"] > section["initial_depth"]:
        raise ValueError("architecture.min_depth cannot exceed initial_depth.")

    if section["initial_depth"] > section["max_depth"]:
        raise ValueError("architecture.initial_depth cannot exceed max_depth.")


def validate_training(section: dict[str, Any]) -> None:
    """Validate training section."""
    require_keys(
        section,
        ["epochs", "batch_size", "optimizer", "base_learning_rate", "random_seed"],
        "training",
    )

    validate_positive_int(section["epochs"], "training.epochs")
    validate_positive_int(section["batch_size"], "training.batch_size")
    validate_positive_float(section["base_learning_rate"], "training.base_learning_rate")

    if not isinstance(section["random_seed"], int):
        raise ValueError("training.random_seed must be integer.")


def validate_scheduler(section: dict[str, Any], architecture: dict[str, Any]) -> None:
    """Validate scheduler section."""
    require_keys(
        section,
        [
            "enabled",
            "type",
            "action_interval_epochs",
            "warmup_epochs",
            "allowed_actions",
            "prune",
        ],
        "scheduler",
    )

    if not isinstance(section["enabled"], bool):
        raise ValueError("scheduler.enabled must be boolean.")

    validate_enum(section["type"], enum_values(SchedulerType), "scheduler.type")
    validate_enum_list(
        section["allowed_actions"],
        enum_values(SchedulerAction),
        "scheduler.allowed_actions",
    )

    validate_positive_int(section["action_interval_epochs"], "scheduler.action_interval_epochs")
    validate_non_negative_int(section["warmup_epochs"], "scheduler.warmup_epochs")

    validate_prune_section(section["prune"])


def validate_prune_section(section: dict[str, Any]) -> None:
    """Validate scheduler.prune section."""
    require_keys(section, ["enabled", "mode", "mask"], "scheduler.prune")

    if not isinstance(section["enabled"], bool):
        raise ValueError("scheduler.prune.enabled must be boolean.")

    validate_enum(section["mode"], enum_values(PruneMode), "scheduler.prune.mode")

    mask = section["mask"]
    require_keys(mask, ["mask_mode", "mask_schedule", "initial_value", "final_value", "anneal_steps"], "scheduler.prune.mask")

    validate_enum(mask["mask_mode"], enum_values(MaskMode), "scheduler.prune.mask.mask_mode")
    validate_enum(mask["mask_schedule"], enum_values(MaskSchedule), "scheduler.prune.mask.mask_schedule")
    validate_probability(mask["initial_value"], "scheduler.prune.mask.initial_value")
    validate_probability(mask["final_value"], "scheduler.prune.mask.final_value")
    validate_positive_int(mask["anneal_steps"], "scheduler.prune.mask.anneal_steps")

    if mask["mask_mode"] == MaskMode.READOUT_CLASSICAL_MASK.value:
        raise ValueError("READOUT_CLASSICAL_MASK is not valid for quantum layer pruning.")
    

def validate_preprocessing(section: dict[str, Any]) -> None:
    """
    Validate Stage 1B preprocessing section.
    """
    if section.get("scaler", "minmax") != "minmax":
        raise ValueError("Stage 1B v1.1 supports scaler='minmax' only.")

    feature_selection = section.get("feature_selection", {})
    method = feature_selection.get("method", "first_n")

    allowed_methods = {
        "first_n",
        "variance_threshold",
        "variance_then_selectkbest",
    }

    if method not in allowed_methods:
        raise ValueError(f"Unsupported feature_selection.method: {method}")

    n_features = feature_selection.get("n_features", section.get("n_features"))
    if n_features is None:
        raise ValueError("feature_selection.n_features or preprocessing.n_features is required.")

    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("feature_selection.n_features must be a positive integer.")

    if "angle_margin_epsilon" in section:
        eps = float(section["angle_margin_epsilon"])
        if eps <= 0:
            raise ValueError("preprocessing.angle_margin_epsilon must be positive.")


def validate_batching(section: dict[str, Any]) -> None:
    """
    Validate fixed-batch benchmark protocol.
    """
    require_keys(
        section,
        [
            "benchmark_batch_mode",
            "batch_size",
            "n_fixed_batches",
            "fixed_batch_seed",
        ],
        "batching",
    )

    allowed_sampling_modes = {
        "independent_stratified",
        "stratified_partition",
    }

    sampling_mode = section.get("sampling_mode", "stratified_partition")
    if sampling_mode not in allowed_sampling_modes:
        raise ValueError(f"Unsupported batching.sampling_mode: {sampling_mode}")

    validate_positive_int(section["batch_size"], "batching.batch_size")
    validate_positive_int(section["n_fixed_batches"], "batching.n_fixed_batches")

    min_per_class = int(section.get("min_per_class", 1))
    if min_per_class < 0:
        raise ValueError("batching.min_per_class must be non-negative.")

    if section["batch_size"] < 2 * min_per_class:
        raise ValueError("batching.batch_size is too small for binary min_per_class constraint.")

    if not isinstance(section["fixed_batch_seed"], int):
        raise ValueError("batching.fixed_batch_seed must be integer.")