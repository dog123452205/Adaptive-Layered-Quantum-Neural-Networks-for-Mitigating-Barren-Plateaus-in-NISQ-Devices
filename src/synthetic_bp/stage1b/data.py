"""
Stage 1B dataset utilities v1.

This module prepares small classical datasets for task-aware BP benchmarking.

Key design rules:
    1. Split before preprocessing to avoid leakage.
    2. Fit feature selector on train only.
    3. Fit scaler on train only.
    4. Build fixed benchmark batches with optional non-overlap constraints.
    5. Audit boundary concentration after angle scaling.

Stage 1B v1 still uses fixed-batch task-aware benchmarking.
Multi-batch data-induced gradient decomposition is reserved for v2/v3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.synthetic_bp.constants import DatasetName
from src.synthetic_bp.path_utils import ensure_parent_dir

@dataclass(frozen = True)
class FeatureSelectionReport:
    '''
    Metadata describing the feature selection process.

    Attributes:
        method:
            Feature selection method.
        selected_indices:
            Original feature indices selected after train-only feature selection.
        selected_feature_names:
            Names of selected features.
        variance_threshold:
            VarianceThreshold value.
        score_func:
            Supervised score function used by SelectKBest.
    '''
    method: str
    selected_indices: list[int]
    selected_feature_names: list[str]
    variance_threshold: float | None
    score_func: str | None

@dataclass(frozen = True)
class BoundaryAuditReport:
    """
    Report for checking whether scaled features concentrate at angle boundaries.

    Attributes:
        enabled:
            Whether boundary audit was enabled.
        lower:
            Lower feature range bound.
        upper:
            Upper feature range bound.
        tolerance:
            Boundary tolerance.
        lower_boundary_fraction:
            Fraction of train feature values close to lower bound.
        upper_boundary_fraction:
            Fraction of train feature values close to upper bound.
        total_boundary_fraction:
            Fraction of train feature values close to either boundary.
    """
    enabled: bool
    lower: float
    upper: float
    tolerance: float
    lower_boundary_fraction: float
    upper_boundary_fraction: float
    total_boundary_fraction: float


@dataclass(frozen = True)
class TaskDataset:
    '''
    Container for a supervised classification dataset.

    Attributes:
        X_train:
            Preprocessed training features.
        y_train:
            Training labels.
        X_val:
            Preprocessed validation features.
        y_val:
            Validation labels.
        X_test:
            Preprocessed test features.
        y_test:
            Test labels.
        feature_names:
            Feature names after optional feature selection.
        class_names:
            Class names after optional binary remapping.
        scaler_feature_range:
            Feature range used by MinMaxScaler.
    '''
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    scaler_feature_range: tuple[float, float]
    feature_selection_report: FeatureSelectionReport
    boundary_audit_report: BoundaryAuditReport

@dataclass(frozen = True)
class FixedBatch:
    '''
    Fixed benchmark batch for Stage 1B.

    Attributes:
        batch_id:
            Integer batch ID.
        X:
            Batch features.
        y:
            Batch labels.
        indices:
            Indices selected from the training set.
        fixed_batch_seed:
            Seed used to sample the batch.
        class_counts:
            Label counts inside the batch.
    '''
    batch_id: int
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    fixed_batch_seed: int
    class_counts: dict[int, int]

def prepare_task_dataset(config: dict[str, Any]) -> TaskDataset:
    """
    Report for checking whether scaled features concentrate at angle boundaries.

    Attributes:
        enabled:
            Whether boundary audit was enabled.
        lower:
            Lower feature range bound.
        upper:
            Upper feature range bound.
        tolerance:
            Boundary tolerance.
        lower_boundary_fraction:
            Fraction of train feature values close to lower bound.
        upper_boundary_fraction:
            Fraction of train feature values close to upper bound.
        total_boundary_fraction:
            Fraction of train feature values close to either boundary.
    """
    dataset_cfg = config['dataset']
    preprocessing_cfg = config['preprocessing']

    X_raw, y_raw, feature_names, class_names = load_raw_dataset(dataset_cfg)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_dataset(
        X=X_raw,
        y=y_raw,
        dataset_cfg=dataset_cfg,
    )

    (
        X_train_selected,
        X_val_selected,
        X_test_selected,
        feature_report,
    ) = select_features(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        feature_names=feature_names,
        preprocessing_cfg=preprocessing_cfg,
    )

    X_train, X_val, X_test, feature_range = scale_features_train_only(
        X_train_raw=X_train_selected,
        X_val_raw=X_val_selected,
        X_test_raw=X_test_selected,
        preprocessing_cfg = preprocessing_cfg,
    )

    boundary_report = audit_scaled_feature_boundaries(
        X_train = X_train,
        preprocessing_cfg = preprocessing_cfg,
        feature_range = feature_range
    )

    return TaskDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_report.selected_feature_names,
        class_names=class_names,
        scaler_feature_range=feature_range,
        feature_selection_report=feature_report,
        boundary_audit_report=boundary_report
    )

def load_raw_dataset(dataset_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    '''
    Load a supported sklearn dataset and optionally convert to binary subset.

    Args:
        dataset_cfg:
            Dataset configuration section.

    Returns:
        Tuple:
            X, y, feature_names, class_names.

    Raises:
        ValueError:
            If dataset is unsupported.
    '''
    name = dataset_cfg['name']

    if name == DatasetName.IRIS.value:
        data = load_iris()
    elif name == DatasetName.WINE.value:
        data = load_wine()
    elif name == DatasetName.BREAST_CANCER.value:
        data = load_breast_cancer()
    elif name == DatasetName.DIGITS.value:
        data = load_digits()
    else:
        raise ValueError(
            f"Unsupported Stage 1B v1 dataset: {name}. "
            "Supported: iris, wine, breast_cancer, digits."
        )
    
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)

    feature_names = (
        list(data.feature_names)
        if hasattr(data, 'feature_names')
        else [f"feature_{i}" for i in range(X.shape[1])]
    )

    class_names = (
        [str(c) for c in data.target_names]
        if hasattr(data, "target_names")
        else [str(c) for c in sorted(np.unique(y))]
    )

    target_mode = dataset_cfg.get('target_mode', 'native')

    if target_mode == "binary_subset":
        selected = dataset_cfg.get("selected_classes")
        if not isinstance(selected, list) or len(selected) != 2:
            raise ValueError("dataset.selected_classes must contain exactly two classes.")

        selected = [int(c) for c in selected]
        mask = np.isin(y, selected)

        X = X[mask]
        y_selected = y[mask]

        y_binary = np.zeros_like(y_selected)
        y_binary[y_selected == selected[1]] = 1

        return X, y_binary.astype(np.int64), feature_names, [str(selected[0]), str(selected[1])]

    if target_mode != "native":
        raise ValueError(f"Unsupported dataset.target_mode: {target_mode}")

    unique = sorted(np.unique(y))
    if len(unique) != 2:
        raise ValueError(
            "Stage 1B v1 expects binary classification. "
            "Use dataset.target_mode='binary_subset' for multiclass datasets."
        )

    y_binary = np.zeros_like(y)
    y_binary[y == unique[1]] = 1

    return X, y_binary.astype(np.int64), feature_names, [str(unique[0]), str(unique[1])]

def select_features(
        X_train_raw: np.ndarray,
        y_train: np.ndarray,
        X_val_raw: np.ndarray,
        X_test_raw: np.ndarray,
        feature_names: list[str],
        preprocessing_cfg: dict[str, Any], 
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, FeatureSelectionReport]:
    """
    Fit feature selector on train only and transform all splits.

    Supported methods:
        first_n:
            Keep first n features. Provided only for debugging.
        variance_threshold:
            Remove near-constant features, then keep first n surviving features.
        variance_then_selectkbest:
            Remove near-constant features, then select top-k supervised features.

    Args:
        X_train_raw:
            Raw training features.
        y_train:
            Training labels.
        X_val_raw:
            Raw validation features.
        X_test_raw:
            Raw test features.
        feature_names:
            Original feature names.
        preprocessing_cfg:
            Preprocessing config.

    Returns:
        X_train_selected, X_val_selected, X_test_selected, feature selection report.

    Raises:
        ValueError:
            If feature selection config is invalid.
    """
    selection_cfg = preprocessing_cfg.get("feature_selection", {})
    method = selection_cfg.get("method", "first_n")
    n_features = int(preprocessing_cfg.get("n_features", preprocessing_cfg.get("n_features", X_train_raw.shape[1])))

    if n_features <= 0:
        raise ValueError(f"preprocessing.n_features must be positive. Got {n_features}.")
    
    if n_features > X_train_raw.shape[1]:
        raise ValueError(
            f"Requested n_features={n_features}, but dataset has {X_train_raw.shape[1]} features."
        )
    
    if method == "first_n": 
        selected_indices = list(range(n_features))

        report = FeatureSelectionReport(
            method=method,
            selected_indices=selected_indices,
            selected_feature_names=[feature_names[i] for i in selected_indices],
            variance_threshold=None,
            score_func=None
        )

        return (
            X_train_raw[:, selected_indices],
            X_val_raw[:, selected_indices],
            X_test_raw[:, selected_indices],
            report
        )
    
    variance_threshold = float(selection_cfg.get("variance_threshold", 1e-12))

    vt = VarianceThreshold(threshold=variance_threshold)
    X_train_vt = vt.fit_transform(X_train_raw)
    X_val_vt = vt.transform(X_val_raw)
    X_test_vt = vt.transform(X_test_raw)

    surviving_indices = np.where(vt.get_support())[0].tolist()

    if len(surviving_indices) == 0:
        raise ValueError(
            f"VarianceThreshold removed all features. "
            f"Try lowering preprocessing.feature_selection.variance_threshold={variance_threshold}."
        )
    
    if n_features > len(surviving_indices):
        raise ValueError(
            f"Requested n_features={n_features}, but only {len(surviving_indices)} features survived VarianceThreshold."
        )
    
    if method == "variance_threshold":
        selected_survivor_positions = list(range(n_features))
        selected_indices = [surviving_indices[pos] for pos in selected_survivor_positions]

        report = FeatureSelectionReport(
            method=method,
            selected_indices=selected_indices,
            selected_feature_names=[feature_names[i] for i in selected_indices],
            variance_threshold=variance_threshold,
            score_func=None,
        )

        return (
            X_train_vt[:, selected_survivor_positions],
            X_val_vt[:, selected_survivor_positions],
            X_test_vt[:, selected_survivor_positions],
            report,
        )
    
    if method == "variance_then_selectkbest":
        score_func_name = selection_cfg.get("score_func", "f_classif")
        score_func = get_score_func(score_func_name)

        selector = SelectKBest(score_func=score_func, k=n_features)
        X_train_selected = selector.fit_transform(X_train_vt, y_train)
        X_val_selected = selector.transform(X_val_vt)
        X_test_selected = selector.transform(X_test_vt)

        selected_survivor_positions = np.where(selector.get_support())[0].tolist()
        selected_indices = [surviving_indices[pos] for pos in selected_survivor_positions]

        report = FeatureSelectionReport(
            method=method,
            selected_indices=selected_indices,
            selected_feature_names=[feature_names[i] for i in selected_indices],
            variance_threshold=variance_threshold,
            score_func=score_func_name,
        )

        return X_train_selected, X_val_selected, X_test_selected, report

    raise ValueError(
        "Unsupported feature_selection.method. "
        "Use first_n, variance_threshold, or variance_then_selectkbest."
    )
                        
def get_score_func(name: str):
    """
    Return sklearn score function by name.

    Args:
        name:
            f_classif or mutual_info_classif.

    Returns:
        Callable sklearn score function.

    Raises:
        ValueError:
            If score function is unsupported.
    """
    if name == "f_classif":
        return f_classif

    if name == "mutual_info_classif":
        return mutual_info_classif

    raise ValueError(f"Unsupported feature_selection.score_func: {name}")

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    dataset_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train/val/test sets.

    Args:
        X:
            Feature matrix.
        y:
            Label vector.
        dataset_cfg:
            Dataset configuration section.

    Returns:
        Tuple:
            X_train, X_val, X_test, y_train, y_val, y_test.

    Raises:
        ValueError:
            If split ratios are invalid.
    """
    random_state = int(dataset_cfg.get("random_state", 42))
    test_size = float(dataset_cfg.get("test_size", 0.2))
    validation_size = float(dataset_cfg.get("validation_size", 0.2))

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size = test_size,
        random_state = random_state,
        stratify=y,
    )

    relative_val_size = validation_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size = relative_val_size,
        random_state = random_state,
        stratify = y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features_train_only(
    X_train_raw: np.ndarray, 
    X_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    preprocessing_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    '''
    Fit MinMaxScaler on training data and transform train/val/test sets.

    Args:
        X_train_raw:
            Raw training features.
        X_val_raw:
            Raw validation features.
        X_test_raw:
            Raw test features.
        preprocessing_cfg:
            Preprocessing config.

    Returns:
        Tuple:
            X_train, X_val, X_test, feature_range.

    Raises:
        ValueError:
            If unsupported scaler is requested.
    '''
    scaler_name = preprocessing_cfg.get("scaler", "minmax")

    if scaler_name != "minmax":
        raise ValueError("Stage 1B v1 currently supports scaler='minmax' only.")
    
    feature_range_raw = preprocessing_cfg.get("feature_range")

    if feature_range_raw is None: 
        margin = float(preprocessing_cfg.get("angle_margin_epsilon", 1e-3))
        feature_range = (margin, float(np.pi-margin))
    else:
        if not isinstance(feature_range_raw, list) or len(feature_range_raw) != 2:
            raise ValueError("preprocessing.feature_range must be [low, high].")
        
        feature_range = (float(feature_range_raw[0]), float(feature_range_raw[1]))

    if feature_range[0] >= feature_range[1]:
        raise ValueError(
            f"Invalid preprocessing.feature_range: {feature_range}. "
            "Must satisfy low < high."
        )

    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(X_train_raw)

    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_val, X_test, feature_range

def audit_scaled_feature_boundaries(
    X_train: np.ndarray,
    preprocessing_cfg: dict[str, Any],
    feature_range: tuple[float, float]
)   -> BoundaryAuditReport:
    """
    Audit concentration of scaled features near angle boundaries.

    Args:
        X_train:
            Scaled training features.
        preprocessing_cfg:
            Preprocessing config.
        feature_range:
            Feature range used by scaler.

    Returns:
        BoundaryAuditReport.
    """

    audit_cfg = preprocessing_cfg.get("boundary_audit", {})
    enabled = bool(audit_cfg.get("enabled", True))
    tolerance = float(audit_cfg.get("tolerance", 1e-6))

    lower, upper = feature_range

    if not enabled:
        return BoundaryAuditReport(
            enabled=False,
            lower=lower,
            upper=upper,
            tolerance=tolerance,
            lower_boundary_fraction=0.0,
            upper_boundary_fraction=0.0,
            total_boundary_fraction=0.0
        )
    
    lower_mask = np.abs(X_train -lower) <= tolerance
    upper_mask = np.abs(X_train -upper) <= tolerance

    lower_fraction = float(np.mean(lower_mask))
    upper_fraction = float(np.mean(upper_mask))
    total_fraction = float(np.mean(lower_mask | upper_mask))

    return BoundaryAuditReport(
        enabled=True,
        lower=lower,
        upper=upper,
        tolerance=tolerance,
        lower_boundary_fraction=lower_fraction,
        upper_boundary_fraction=upper_fraction,
        total_boundary_fraction=total_fraction
    )

def build_fixed_batches(
    dataset: TaskDataset,
    config: dict[str, Any],
) -> list[FixedBatch]:
    """
    Build fixed benchmark batches from the training set.

    Args:
        dataset:
            Prepared TaskDataset.
        config:
            Resolved Stage 1B config.

    Returns:
        List of FixedBatch objects.

    Raises:
        ValueError:
            If batch construction violates class or overlap constraints.
    """
    batching_cfg = config["batching"]

    sampling_mode = batching_cfg.get("sampling_mode", "stratified_partition")
    batch_size = int(batching_cfg["batch_size"])
    n_fixed_batches = int(batching_cfg["n_fixed_batches"])
    fixed_seed = int(batching_cfg["fixed_batch_seed"])
    allow_overlap = bool(batching_cfg.get("allow_overlap", False))
    min_per_class = int(batching_cfg.get("min_per_class", 1))

    if batch_size <= 0:
        raise ValueError("batching.batch_size must be positive.")

    if n_fixed_batches <= 0:
        raise ValueError("batching.n_fixed_batches must be positive.")

    if min_per_class < 0:
        raise ValueError("batching.min_per_class must be non-negative.")

    if sampling_mode == "independent_stratified":
        batches = build_independent_stratified_batches(
            dataset=dataset,
            batch_size=batch_size,
            n_fixed_batches=n_fixed_batches,
            fixed_seed=fixed_seed,
            min_per_class=min_per_class,
        )
    elif sampling_mode == "stratified_partition":
        batches = build_stratified_partition_batches(
            dataset=dataset,
            batch_size=batch_size,
            n_fixed_batches=n_fixed_batches,
            fixed_seed=fixed_seed,
            min_per_class=min_per_class,
            allow_overlap=allow_overlap,
        )
    else:
        raise ValueError(
            "Unsupported batching.sampling_mode. "
            "Use independent_stratified or stratified_partition."
        )

    validate_batch_class_counts(batches, min_per_class=min_per_class)

    if not allow_overlap:
        validate_no_batch_overlap(batches)

    return batches


def build_independent_stratified_batches(
    dataset: TaskDataset,
    batch_size: int,
    n_fixed_batches: int,
    fixed_seed: int,
    min_per_class: int,
) -> list[FixedBatch]:
    """
    Build batches by independently sampling from the full train set.

    This mode may create overlap across batches. It is useful for quick smoke
    tests but not recommended for multi-batch data-variability analysis.

    Args:
        dataset:
            Prepared TaskDataset.
        batch_size:
            Batch size.
        n_fixed_batches:
            Number of fixed batches.
        fixed_seed:
            Base random seed.
        min_per_class:
            Minimum number of samples per class per batch.

    Returns:
        List of FixedBatch objects.
    """
    batches: list[FixedBatch] = []

    for batch_id in range(n_fixed_batches):
        rng = np.random.default_rng(fixed_seed + batch_id)
        indices = stratified_sample_indices(
            y=dataset.y_train,
            batch_size=batch_size,
            rng=rng,
            min_per_class=min_per_class,
            candidate_indices=None,
        )
        batches.append(make_fixed_batch(dataset, batch_id, indices, fixed_seed + batch_id))

    return batches


def build_stratified_partition_batches(
    dataset: TaskDataset,
    batch_size: int,
    n_fixed_batches: int,
    fixed_seed: int,
    min_per_class: int,
    allow_overlap: bool,
) -> list[FixedBatch]:
    """
    Build approximately non-overlapping stratified batches.

    Args:
        dataset:
            Prepared TaskDataset.
        batch_size:
            Batch size.
        n_fixed_batches:
            Number of fixed batches.
        fixed_seed:
            Base random seed.
        min_per_class:
            Minimum samples per class per batch.
        allow_overlap:
            Whether overlap is allowed if there are insufficient samples.

    Returns:
        List of FixedBatch objects.

    Raises:
        ValueError:
            If non-overlap is required but impossible.
    """
    y_train = dataset.y_train
    total_required = batch_size * n_fixed_batches

    if not allow_overlap and total_required > len(y_train):
        raise ValueError(
            f"Cannot build {n_fixed_batches} non-overlapping batches of size {batch_size} "
            f"from train size {len(y_train)}."
        )

    rng = np.random.default_rng(fixed_seed)
    remaining_indices = np.arange(len(y_train), dtype=np.int64)
    batches: list[FixedBatch] = []

    for batch_id in range(n_fixed_batches):
        indices = stratified_sample_indices(
            y=y_train,
            batch_size=batch_size,
            rng=rng,
            min_per_class=min_per_class,
            candidate_indices=remaining_indices,
        )

        batches.append(make_fixed_batch(dataset, batch_id, indices, fixed_seed + batch_id))

        if not allow_overlap:
            remaining_indices = np.setdiff1d(remaining_indices, indices, assume_unique=False)

    return batches


def stratified_sample_indices(
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    min_per_class: int = 1,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sample a binary stratified batch from candidate indices.

    Args:
        y:
            Full training labels.
        batch_size:
            Desired batch size.
        rng:
            Numpy random generator.
        min_per_class:
            Minimum number of samples per class.
        candidate_indices:
            Optional subset of indices eligible for sampling.

    Returns:
        Selected indices.

    Raises:
        ValueError:
            If binary stratified sampling is impossible.
    """
    if batch_size < 2 * min_per_class:
        raise ValueError(
            f"batch_size={batch_size} is too small for min_per_class={min_per_class} "
            "in binary classification."
        )

    if candidate_indices is None:
        candidate_indices = np.arange(len(y), dtype=np.int64)

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    y_candidate = y[candidate_indices]

    classes = sorted(np.unique(y))
    if len(classes) != 2:
        raise ValueError("stratified_sample_indices expects binary labels.")

    class_to_candidates = {
        cls: candidate_indices[y_candidate == cls]
        for cls in classes
    }

    for cls in classes:
        if len(class_to_candidates[cls]) < min_per_class:
            raise ValueError(
                f"Not enough candidate samples for class {cls}. "
                f"Need at least {min_per_class}, have {len(class_to_candidates[cls])}."
            )

    original_class_ratio = {
        cls: len(np.where(y == cls)[0]) / len(y)
        for cls in classes
    }

    n_class_1 = int(round(batch_size * original_class_ratio[classes[1]]))
    n_class_1 = max(min_per_class, n_class_1)
    n_class_0 = batch_size - n_class_1

    if n_class_0 < min_per_class:
        n_class_0 = min_per_class
        n_class_1 = batch_size - n_class_0

    n_by_class = {
        classes[0]: n_class_0,
        classes[1]: n_class_1,
    }

    for cls in classes:
        if n_by_class[cls] > len(class_to_candidates[cls]):
            raise ValueError(
                f"Not enough candidate samples in class {cls}: "
                f"need {n_by_class[cls]}, have {len(class_to_candidates[cls])}."
            )

    selected_parts = []
    for cls in classes:
        selected = rng.choice(
            class_to_candidates[cls],
            size=n_by_class[cls],
            replace=False,
        )
        selected_parts.append(selected)

    indices = np.concatenate(selected_parts)
    rng.shuffle(indices)

    return indices.astype(np.int64)


def make_fixed_batch(
    dataset: TaskDataset,
    batch_id: int,
    indices: np.ndarray,
    fixed_batch_seed: int,
) -> FixedBatch:
    """
    Create FixedBatch object from selected indices.

    Args:
        dataset:
            Prepared TaskDataset.
        batch_id:
            Batch ID.
        indices:
            Selected train indices.
        fixed_batch_seed:
            Seed used for the batch.

    Returns:
        FixedBatch.
    """
    X_batch = dataset.X_train[indices]
    y_batch = dataset.y_train[indices]

    unique, counts = np.unique(y_batch, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    return FixedBatch(
        batch_id=batch_id,
        X=X_batch,
        y=y_batch,
        indices=indices,
        fixed_batch_seed=fixed_batch_seed,
        class_counts=class_counts,
    )


def validate_batch_class_counts(
    batches: list[FixedBatch],
    min_per_class: int,
) -> None:
    """
    Validate that each batch contains both classes.

    Args:
        batches:
            Fixed batches.
        min_per_class:
            Minimum count per class.

    Raises:
        ValueError:
            If a batch violates the class count constraint.
    """
    for batch in batches:
        for cls in [0, 1]:
            count = int(batch.class_counts.get(cls, 0))
            if count < min_per_class:
                raise ValueError(
                    f"Batch {batch.batch_id} has class {cls} count={count}, "
                    f"below min_per_class={min_per_class}."
                )


def validate_no_batch_overlap(batches: list[FixedBatch]) -> None:
    """
    Validate that fixed batches do not overlap.

    Args:
        batches:
            Fixed batches.

    Raises:
        ValueError:
            If any pair of batches overlaps.
    """
    seen: set[int] = set()

    for batch in batches:
        current = set(int(i) for i in batch.indices.tolist())
        overlap = seen.intersection(current)

        if overlap:
            raise ValueError(
                f"Batch overlap detected. Batch {batch.batch_id} overlaps "
                f"with previous batches by {len(overlap)} samples."
            )

        seen.update(current)


def compute_batch_overlap_matrix(batches: list[FixedBatch]) -> list[list[int]]:
    """
    Compute pairwise overlap counts between fixed batches.

    Args:
        batches:
            Fixed batches.

    Returns:
        Matrix of pairwise overlap counts.
    """
    matrix: list[list[int]] = []

    sets = [set(int(i) for i in batch.indices.tolist()) for batch in batches]

    for i, set_i in enumerate(sets):
        row = []
        for j, set_j in enumerate(sets):
            row.append(len(set_i.intersection(set_j)))
        matrix.append(row)

    return matrix


def save_batch_manifest(
    batches: list[FixedBatch],
    dataset: TaskDataset,
    output_path: str | Path,
) -> None:
    """
    Save fixed batch metadata to JSON.

    Args:
        batches:
            Fixed benchmark batches.
        dataset:
            Prepared dataset.
        output_path:
            JSON output path.
    """
    output_path = Path(output_path)
    ensure_parent_dir(output_path)

    manifest = {
        "feature_names": dataset.feature_names,
        "class_names": dataset.class_names,
        "scaler_feature_range": list(dataset.scaler_feature_range),
        "feature_selection_report": asdict(dataset.feature_selection_report),
        "boundary_audit_report": asdict(dataset.boundary_audit_report),
        "batch_overlap_matrix": compute_batch_overlap_matrix(batches),
        "batches": [
            {
                "batch_id": batch.batch_id,
                "indices": batch.indices.tolist(),
                "y": batch.y.tolist(),
                "fixed_batch_seed": batch.fixed_batch_seed,
                "class_counts": batch.class_counts,
            }
            for batch in batches
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)