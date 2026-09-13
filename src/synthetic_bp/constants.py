'''
Project-wide constants and enums for AL-QNN

This module is the source of truth for:
- allowed config values
- benchmark modes,
- scheduler actions,
- masking modes,
- operational safety defaults.

Important:
    Do not import PennyLane here.
    Do not instantiate quantum circuits here.
    Do not validate statevectores, density matrices, or full unitaries.
'''
from __future__ import annotations

try: from enum import StrEnum
except ImportError: 
    from enum import Enum

    class StrEnum(str, Enum):
        pass


def enum_values(enum_cls: type[StrEnum]) -> set[str]:
    """
    Return all string values of a StrEnum class.

    Args:
        enum_cls: A StrEnum class.

    Returns:
        Set of valid string values.
    """
    return {item.value for item in enum_cls}


# ============================================================
# Benchmark modes
# ============================================================

class BenchmarkMode(StrEnum):
    '''Benchmark phase identifier.'''
    RANDOM_BP = 'random_bp'
    TASK_AWARE_BP = 'task_aware_bp'

class ObjectiveType(StrEnum):
    """Objective family used by the benchmark"""
    EXPECTATION_VALUE = 'expectation_value'
    SUPERVISED_LOSS = 'supervised_loss'

class BudgetNormalizationMode(StrEnum):
    '''
    Budget normalization mode for fair benchmark comparision.

    SAME_ENTANGLING_GATE_COUNT is especially important when comparing
    LINEAR, CIRCULAR, and FULL topologies.
    '''
    NONE = 'none'
    SAME_DEPTH = 'same_depth'
    SAME_PARAMETER_COUNT = 'same_parameter_count'
    SAME_ENTANGLING_GATE_COUNT = 'same_entangling_gate_count'
    SAME_ENTANGLING_GATES_PER_QUBIT = 'same_entangling_gates_per_qubit'
    SAME_COMPUTE_BUDGET = 'same_compute_budget'

# ============================================================
# Quantum model design 
# ============================================================

class CostType(StrEnum):
    """
    Observable / cost objective used in synthetic barren plateau benchmarks.
    This is not the same as supervised learning loss.
    """
    LOCAL = "local"
    GLOBAL = "global"
    HAMILTONIAN_LOCAL = "hamiltonian_local"


class AnsatzType(StrEnum):
    """
    Ansatz families supported by the project.
    Stage 1 uses HEA first.
    Other ansatz types are optional extensions.
    """
    HEA = "hea"
    ALTERNATING_LAYERED = "alternating_layered"
    TREE_NETWORK = "tree_network"
    DATA_REUPLOADING = "data_reuploading"


class GateSet(StrEnum):
    """
    Supported gate-set families.

    RX_RY_RZ_CNOT:
        Standard hardware-efficient ansatz with hard CNOT entanglers.

    RX_RY_RZ_PARAM_ENT:
        Ansatz with parameterized entanglers, better for soft entanglement masking.
    """
    RX_RY_RZ_CNOT = "RX_RY_RZ_CNOT"
    RX_RY_RZ_PARAM_ENT = "RX_RY_RZ_PARAM_ENT"


class EncodingType(StrEnum):
    """
    Classical-to-quantum data encoding strategy.
    """
    NONE = "none"
    ANGLE = "angle"
    AMPLITUDE = "amplitude"
    DATA_REUPLOADING = "data_reuploading"


class EntanglementTopology(StrEnum):
    """
    Topology of entangling connections.
    """
    NONE = "none"
    LINEAR = "linear"
    CIRCULAR = "circular"
    TREE = "tree"
    FULL = "full"


class EntanglerType(StrEnum):
    """
    Entangling gate type.

    CNOT and CZ are hard, non-parameterized gates.
    CRZ and Ising gates support continuous strength control.
    """
    CNOT = "cnot"
    CZ = "cz"
    CRZ = "crz"
    ISING_XX = "ising_xx"
    ISING_YY = "ising_yy"
    ISING_ZZ = "ising_zz"

# ============================================================
# Task-aware benchmark
# ============================================================

class DatasetName(StrEnum):
    """Supported benchmark datasets."""
    IRIS = 'iris'
    WINE = 'wine'
    BREAST_CANCER = 'breast_cancer'
    DIGITS = 'digits'
    MNIST_REDUCED = 'mnist_reduced'

class LossType(StrEnum):
    '''Supervised loss functions'''
    MSE = 'mse'
    BINARY_CROSS_ENTROPY = 'binary_cross_entropy'
    CROSS_ENTROPY = 'cross_entropy'

class ReadoutType(StrEnum):
    '''Readout mapping from quantum measurements to prediction.'''
    SINGLE_QUBIT_Z = 'single_qubit_z'
    MULTI_QUBIT_Z = 'multi_qubit_z'
    PARITY = 'parity'

class BatchBenchmarkMode(StrEnum):
    """
    Batch protocol for Stage 1B.

    FIXED:
        Use a fixed benchmark batch across seeds to isolate initialization variance.

    MULTI_BATCH:
        Use multiple fixed batches to estimate data-induced variability.
    """
    FIXED = 'fixed'
    MULTI_BATCH = 'multi_batch'

# ============================================================
# Initialization / device / differentiation
# ============================================================

class InitStrategy(StrEnum):
    """
    Parameter initialization strategy.
    """
    FULL_RANDOM_CENTERED = "full_random_centered"      # e.g. Uniform(-pi, pi)
    FULL_RANDOM_POSITIVE = "full_random_positive"      # e.g. Uniform(0, 2pi)
    IDENTITY_NEAR_ZERO = "identity_near_zero"          # e.g. Uniform(-eps, eps)


class InitDistribution(StrEnum):
    """
    Distribution used for parameter initialization.
    """
    UNIFORM = "uniform"
    NORMAL = "normal"


class DeviceName(StrEnum):
    """
    Supported PennyLane backend names.
    """
    DEFAULT_QUBIT = "default.qubit"
    LIGHTNING_QUBIT = "lightning.qubit"
    DEFAULT_MIXED = "default.mixed"


class DiffMethod(StrEnum):
    """
    Gradient computation method.
    """
    BACKPROP = "backprop"
    ADJOINT = "adjoint"
    PARAMETER_SHIFT = "parameter-shift"
    FINITE_DIFF = "finite-diff"

class DiffFallbackPolicy(StrEnum):
    """
    Policy for invalid device/diff_method combinations.

    ERROR:
        Raise an exception.

    WARN_AND_FALLBACK:
        Resolve to fallback_diff_method and write config_resolved.yaml.
    """
    ERROR = 'error'
    WARN_AND_FALLBACK = 'warn_and_fallback'
# ============================================================
# Logging / artifact format
# ============================================================

class GradientLoggingMode(StrEnum):
    """
    Gradient logging mode.

    RAW:
        Save raw gradient components. Use only for smoke/debug.

    ONLINE_SUMMARY:
        Do online aggregation such as Welford variance.
        Use for full benchmark.
    """
    RAW = "raw"
    ONLINE_SUMMARY = "online_summary"


class FileFormat(StrEnum):
    """
    Output file formats.
    """
    CSV = "csv"
    PARQUET = "parquet"
    JSONL = "jsonl"
    YAML = 'yaml'
    JSON = 'json'

# ============================================================
# Scheduler / adaptive architecture actions
# ============================================================

class SchedulerType(StrEnum):
    """
    Scheduler implementation type.
    MVP should use RULE_BASED.
    Learned/RL schedulers are future work.
    """
    RULE_BASED = "rule_based"
    SUPERVISED = "supervised"
    RL = "rl"


class SchedulerAction(StrEnum):
    """
    Safe action space for adaptive AL-QNN.

    Pruning and entanglement reduction are represented as multi-step
    procedures, not instant hard mutations.
    """
    KEEP = "keep"
    ADD_LAYER = "add_layer"
    STOP_GROWTH = "stop_growth"

    SOFT_FREEZE_LAYER = "soft_freeze_layer"
    UNFREEZE_LAYER = "unfreeze_layer"

    PROPOSE_PRUNE_LAYER = "propose_prune_layer"
    SOFT_MASK_LAYER = "soft_mask_layer"
    COMMIT_PRUNE_LAYER = "commit_prune_layer"
    ROLLBACK_PRUNE = "rollback_prune"

    PROPOSE_REDUCE_ENTANGLEMENT = "propose_reduce_entanglement"
    SOFT_MASK_ENTANGLEMENT = "soft_mask_entanglement"
    COMMIT_REDUCE_ENTANGLEMENT = "commit_reduce_entanglement"
    ROLLBACK_ENTANGLEMENT = "rollback_entanglement"

    ROLLBACK = "rollback"


class PruneMode(StrEnum):
    """
    How pruning is executed.
    """
    HARD_REMOVE_IMMEDIATE = "hard_remove_immediate"
    SOFT_MASK_THEN_COMMIT = "soft_mask_then_commit"
    DISABLED = "disabled"


class EntanglementReductionMode(StrEnum):
    """
    How entanglement reduction is executed.
    """
    HARD_REMOVE_EDGE_IMMEDIATE = "hard_remove_edge_immediate"
    EDGE_MASK_THEN_COMMIT = "edge_mask_then_commit"
    PARAMETERIZED_STRENGTH_ANNEALING = "parameterized_strength_annealing"
    DISABLED = "disabled"


class MaskMode(StrEnum):
    """
    Mask mechanism for layer/gate/edge control.
    """
    NONE = "none"
    ROTATION_PARAMETER_MASK = "rotation_parameter_mask"
    ENTANGLER_EDGE_MASK = "entangler_edge_mask"
    ENTANGLER_STRENGTH_MASK = "entangler_strength_mask"
    READOUT_CLASSICAL_MASK = 'readout_classical_mask'

class MaskSchedule(StrEnum):
    """
    Schedule for reducing mask value from 1 to 0.
    """
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    STEP = "step"


class MutationStatus(StrEnum):
    """
    Runtime status of an adaptive architecture mutation.
    """
    NONE = "none"
    PROPOSED = "proposed"
    IN_PROGRESS = "in_progress"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    COMMITTED = "committed"


class LayerStatus(StrEnum):
    """
    Runtime status of a quantum layer.
    """
    ACTIVE = "active"
    SOFT_FROZEN = "soft_frozen"
    MASKING = "masking"
    MASKED = "masked"
    PRUNED = "pruned"

class ParameterStatus(StrEnum):
    """Runtime status of a parameter tracked by ParameterRegistry."""
    ACTIVE = "active"
    MASKING = "masking"
    MASKED = "masked"
    PRUNED = "pruned"
    ARCHIVED = 'archived'

class OptimizerStatePolicy(StrEnum):
    '''How optimizer state should be restored after mutation'''
    RESTORE_BY_PARAM_ID = 'restore_by_param_id'
    RESET_NEW_PARAMS = "reset_new_params"
    ARCHIVE_REMOVED_PARAMS = 'archived_removed_params'

class AcceptanceMetric(StrEnum):
    """
    Metric used to accept or reject pruning/reduction.
    """
    TRAINING_LOSS = "training_loss"
    VALIDATION_LOSS = "validation_loss"
    VALIDATION_ACCURACY = "validation_accuracy"
    VALIDATION_F1 = "validation_f1"
    GRADIENT_HEALTH = "gradient_health"


# ============================================================
# Diagnostics / calibration
# ============================================================

class TrainabilityMetric(StrEnum):
    """
    Metrics used for barren plateau and gradient health analysis.
    """
    GRAD_NORM = "grad_norm"
    NORMALIZED_GRAD_NORM = 'normalized_grad_norm'
    MEAN_ABS_GRAD = "mean_abs_grad"
    NEAR_ZERO_RATIO = "near_zero_ratio"
    SPATIAL_GRAD_VARIANCE = "spatial_grad_variance"
    PARAM_GRAD_VARIANCE_ACROSS_SEEDS = "param_grad_variance_across_seeds"
    LAYER_GRAD_NORM = "layer_grad_norm"
    LAYER_NEAR_ZERO_RATIO = "layer_near_zero_ratio"


class DiagnosisState(StrEnum):
    """Scheduler-level trainability diagnosis."""
    HEALTHY = "healthy"
    CONVERGED = 'converged'
    SUSPECTED_BARREN_PLATEAU = 'suspected_barren_plateau'
    SUSPECTED_LOCAL_MINIMUM = 'suspected_local_minimum'
    LAYERWISE_DEAD_ZONE = 'layerwise_dead_zone'
    AMBIGUOUS_GRADIENT_COLLAPSE = 'ambiguous_gradient_collapse'

class GradientThresholdMode(StrEnum):
    """Gradient threshold calibration strategy."""
    FIXED = "fixed"
    N_QUBIT_SCALED_STD = "n_qubit_scaled_std"
    EMPIRICAL_STAGE1A = "empirical_stage1a"
    EMPIRICAL_STAGE1B = "empirical_stage1b"
    EMPIRICAL_COMBINED = "empirical_combined"


class ThresholdStatistic(StrEnum):
    """Statistic used to derive empirical thresholds."""
    MEAN = "mean"
    MEDIAN = "median"
    QUANTILE = "quantile"

# ============================================================
# Operational safeguards
# ============================================================

class SimulationProfile(StrEnum):
    """
    Compute profile used by validators to apply resource guards.
    """
    DEBUG = "debug"
    MAIN = "main"
    STRESS = "stress"
    NOISE = "noise"
    HARDWARE_DEMO = "hardware_demo"

class ResourceLimitPolicy(StrEnum):
    '''How resource guard violations should be handled.'''
    STRICT = 'strict'
    WARN_ONLY = 'warn_only'
    OVERRIDABLE = 'overridable'

class ResourceLimitSource(StrEnum):
    '''Where the resource limit came from'''
    CONSTANTS_DEFAULT = 'constants_default'
    YAML_OVERRIDE = 'yaml_override'
    ENV_OVERRIDE = 'env_override'

# ============================================================
# Default safety caps
# ============================================================

DEFAULT_MAX_STAGE1_QUBITS = 12
STRESS_MAX_STATEVECTOR_QUBITS = 16

HARD_MAX_STATEVECTOR_QUBITS = 18
HARD_MAX_BACKPROP_QUBITS = 14
HARD_MAX_DENSITY_MATRIX_QUBITS = 8
HARD_MAX_HARDWARE_DEMO_QUBITS = 5

DEFAULT_MAX_CIRCUIT_DEPTH = 16
STRESS_MAX_CIRCUIT_DEPTH = 20
HARD_MAX_CIRCUIT_DEPTH = 32

DEFAULT_LOG_EPSILON = 1e-30
DEFAULT_NEAR_ZERO_GRAD_THRESHOLD = 1e-6
DEFAULT_FINITE_DIFF_STEP = 1e-5
DEFAULT_IDENTITY_INIT_EPSILON = 1e-2
DEFAULT_RELATIVE_LOSS_TOLERANCE = 0.02

DEFAULT_MASK_INITIAL_VALUE = 1.0
DEFAULT_MASK_FINAL_VALUE = 0.0
DEFAULT_MASK_ANNEAL_STEPS = 5

DEFAULT_OLD_LAYER_LR_MULTIPLIER = 0.1
DEFAULT_NEW_LAYER_LR_MULTIPLIER = 1.0