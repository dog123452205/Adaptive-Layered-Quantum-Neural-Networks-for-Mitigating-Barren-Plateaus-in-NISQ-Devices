
"""
Threshold calibration utilities for Stage 1C.

This module turns Stage 1A/1B gradient artifacts into runtime thresholds
for Stage 2 Scheduler.

Core outputs:
    - component_grad_threshold
    - normalized_grad_norm_threshold
    - near_zero_threshold_dynamic

Important:
    Thresholds are empirical calibration aids, not mathematical proofs of
    barren plateau.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_GROUP_CANDIDATES = [
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
    "cost_type",
    "loss_type",
    "readout_type",
    "n_qubits",
    "reference_depth",
    "depth",
    "budget_normalization",
]

def infer_group_columns(df: pd.DataFrame) -> list[str]:
    """
    Infer calibration grouping columns from available columns.

    Args:
        df:
            Input DataFrame.

    Returns:
        List of columns used to group calibration records.

    Notes:
        Stage 1A and Stage 1B have slightly different metadata columns.
        This function only keeps columns that actually exist.
    """
    return [col for col in DEFAULT_GROUP_CANDIDATES if col in df.columns]

def safe_log10(value: float, epsilon: float) -> float:
    """
    Compute stable log10.

    Args:
        value:
            Input value.
        epsilon:
            Lower numerical floor.

    Returns:
        log10(max(value, epsilon)).
    """
    return math.log10(max(float(value), float(epsilon)))

def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    """
    Compute stable log10.

    Args:
        value:
            Input value.
        epsilon:
            Lower numerical floor.

    Returns:
        log10(max(value, epsilon)).
    """
    missing = [col for col in columns if col not in df.columns]
    if missing: 
        raise ValueError(f"{table_name} missing required columns: {missing}")

def build_threshold_table(
    param_df: pd.DataFrame,
    run_df: pd.DataFrame,
    source_name: str,
    calibration_cfg: dict[str, Any]
) -> pd.DataFrame:
    """
    Build threshold calibration table from Stage 1 parameter and run artifacts.

    Args:
        param_df:
            Parameter-level DataFrame containing
            param_grad_variance_across_seeds.
        run_df:
            Run-level DataFrame containing normalized_grad_norm.
            Can be empty; fallback threshold will then be used.
        source_name:
            Calibration source name, e.g. "stage1a" or "stage1b".
        calibration_cfg:
            calibration section from Stage 1C config.

    Returns:
        Threshold calibration DataFrame.

    Raises:
        ValueError:
            If param_df lacks required columns.
    """
    require_columns(
        param_df,
        [
            "param_grad_variance_across_seeds",
            "param_mean_abs_grad",
            "param_near_zero_ratio",
        ],
        table_name=f"{source_name}.param_variance",
    )

    beta = float(calibration_cfg.get("component_threshold_beta", 0.5))
    quantile = float(calibration_cfg.get("threshold_quantile", 0.05))
    epsilon = float(calibration_cfg.get("variance_epsilon", 1e-30))
    min_component_threshold = float(
        calibration_cfg.get("min_component_grad_threshold", 1e-12)
    )

    min_norm_threshold = float(
        calibration_cfg.get("min_normalized_grad_norm_threshold", 1e-12)
    )

    group_cols = infer_group_columns(param_df)

    param_agg = (
        param_df.groupby(group_cols, dropna=False)
        .agg(
            n_params=("param_id", "nunique"),
            median_param_grad_variance_across_seeds = (
                "param_grad_variance_across_seeds",
                "median",
            ),
            mean_param_grad_variance_across_seeds = (
                "param_grad_variance_across_seeds",
                "mean",
            ),
            q_param_mean_abs_grad = (
                "param_mean_abs_grad",
                lambda s: float(s.quantile(quantile))
            ),
            median_param_mean_abs_grad=("param_mean_abs_grad", "median"),
            mean_param_near_zero_ratio=("param_near_zero_ratio", "mean"),
        )
        .reset_index()
    )

    param_agg["component_grad_threshold_variance_scaled"] = param_agg[
        "median_param_grad_variance_across_seeds"
    ].apply(lambda x: max(min_component_threshold, beta * math.sqrt(max(float(x), epsilon))))
    
    param_agg["component_grad_threshold_abs_quantile"] = param_agg[
        "q_param_mean_abs_grad"
    ].apply(lambda x: max(min_component_threshold, float(x)))

    threshold_mode = calibration_cfg.get(
        "component_threshold_mode",
        "variance_scaled",
    )

    if threshold_mode == "variance_scaled":
        param_agg["component_grad_threshold"] = param_agg["component_grad_threshold_variance_scaled"]
    
    elif threshold_mode == "abs_quantile":
        param_agg["component_grad_threshold"] = param_agg["component_grad_threshold_abs_quantile"]

    elif threshold_mode == "min_of_bth":
        param_agg["component_grad_threshold"] = param_agg[[
            "component_grad_threshold_variance_scaled", 
            "component_grad_threshold_abs_quantile"]    
        ].min(axis=1)
    
    elif threshold_mode == "max_of_both":
        param_agg["component_grad_threshold"] = param_agg[
            [
                "component_grad_threshold_variance_scaled",
                "component_grad_threshold_abs_quantile",
            ]
        ].max(axis=1)
    else:
        raise ValueError(
            "Unsupported calibration.component_threshold_mode. "
            "Use variance_scaled, abs_quantile, min_of_both, or max_of_both."
        )
    
    param_agg["near_zero_threshold_dynamic"] = param_agg[
        "component_grad_threshold"
    ]

    norm_threshold_df = build_normalized_grad_norm_thresholds(
        run_df=run_df,
        group_reference_df=param_agg,
        source_name=source_name,
        quantile=quantile,
        min_norm_threshold=min_norm_threshold,
    )

    output = param_agg.merge(
        norm_threshold_df,
        on=group_cols,
        how="left",
    )

    output["normalized_grad_norm_threshold"] = output[
        "normalized_grad_norm_threshold"
    ].fillna(output["component_grad_threshold"])

    output["normalized_grad_norm_threshold"] = output[
        "normalized_grad_norm_threshold"
    ].apply(lambda x: max(min_norm_threshold, float(x)))

    output["calibration_source"] = source_name
    output["component_threshold_mode"] = threshold_mode
    output["threshold_quantile"] = quantile
    output["component_threshold_beta"] = beta

    output["log10_median_param_grad_variance_across_seeds"] = output[
        "median_param_grad_variance_across_seeds"
    ].apply(lambda x: safe_log10(float(x), epsilon))

    output["log10_mean_param_grad_variance_across_seeds"] = output[
        "mean_param_grad_variance_across_seeds"
    ].apply(lambda x: safe_log10(float(x), epsilon))

    return output

def build_normalized_grad_norm_thresholds(
    run_df: pd.DataFrame,
    group_reference_df: pd.DataFrame,
    source_name: str,
    quantile: float,
    min_norm_threshold: float,
) -> pd.DataFrame:
    """
    Build normalized gradient norm thresholds from run-level artifacts.

    Args:
        run_df:
            Run-level DataFrame. Can be empty.
        group_reference_df:
            Parameter aggregation table used to infer grouping columns.
        source_name:
            Calibration source name.
        quantile:
            Quantile used for threshold.
        min_norm_threshold:
            Minimum threshold floor.

    Returns:
        DataFrame with normalized_grad_norm_threshold.

    Notes:
        If run_df is unavailable or does not have normalized_grad_norm, returns
        a table with NaN thresholds. The caller applies fallback.
    """

    group_cols = infer_group_columns(group_reference_df)

    if run_df is None or run_df.empty or 'normalized_grad_norm' not in run_df.columns:
        fallback = group_reference_df[group_cols].copy()
        fallback["normalized_grad_norm_threshold"] = np.nan
        fallback["normalized_grad_norm_threshold_source"] = "missing_run_artifact"
        return fallback
    
    run_group_cols = [col for col in group_cols if col in run_df.columns]

    run_threshold = (
        run_df.groupby(run_group_cols, dropna=False)
        .agg(
            normalized_grad_norm_threshold=(
                "normalized_grad_norm",
                lambda s: max(min_norm_threshold, float(s.quantile(quantile))),
            ),
            median_normalized_grad_norm=("normalized_grad_norm", "median"),
            mean_normalized_grad_norm=("normalized_grad_norm", "mean"),
        )
        .reset_index()
    )

    run_threshold["normalized_grad_norm_threshold_source"] = (
        f"{source_name}_run_quantile"
    )

    return run_threshold