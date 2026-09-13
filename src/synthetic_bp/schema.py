"""
Output schema definitions for AL-QNN artifacts.

These schemas are used for post-run validation in later stages.
Stage 0 only defines them.
"""

STAGE1_RUN_REQUIRED_COLUMNS = [
    "run_id",
    "benchmark_mode",
    "objective_type",
    "experiment_group",
    "n_qubits",
    "hilbert_dim",
    "depth",
    "ansatz_type",
    "gate_set",
    "encoding_type",
    "entanglement_topology",
    "entangler_type",
    "n_params",
    "n_entangling_gates",
    "seed",
    "objective_value",
    "mean_abs_grad",
    "spatial_grad_variance",
    "log10_spatial_grad_variance",
    "grad_norm",
    "normalized_grad_norm",
    "near_zero_ratio",
]

STAGE1_PARAM_VARIANCE_REQUIRED_COLUMNS = [
    "benchmark_mode",
    "objective_type",
    "experiment_group",
    "ansatz_type",
    "gate_set",
    "encoding_type",
    "entanglement_topology",
    "entangler_type",
    "n_qubits",
    "depth",
    "layer_id",
    "qubit_id",
    "gate_type",
    "param_id",
    "param_index",
    "n_seeds",
    "param_grad_mean",
    "param_grad_variance_across_seeds",
    "param_mean_abs_grad",
    "param_near_zero_ratio",
]

CALIBRATION_RULES_REQUIRED_KEYS = [
    "version",
    "threshold_priority",
    "gradient_threshold_mode",
    "default_beta",
    "fallback_near_zero_grad_threshold",
]

PARAMETER_REGISTRY_REQUIRED_COLUMNS = [
    "epoch",
    "param_id",
    "layer_id",
    "wire_id",
    "gate_type",
    "status",
    "created_at_epoch",
    "mutation_id",
]

MUTATION_TRACE_REQUIRED_COLUMNS = [
    "mutation_id",
    "epoch",
    "action",
    "status",
    "target_layer",
    "mask_value",
    "validation_loss_before",
    "validation_loss_current",
]