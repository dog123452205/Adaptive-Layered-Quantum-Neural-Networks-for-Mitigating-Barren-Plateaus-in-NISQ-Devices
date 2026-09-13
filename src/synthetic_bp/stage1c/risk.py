"""
BP risk estimation utilities for Stage 1C.

Risk levels in v1 are heuristic and calibration-oriented.

They should be interpreted as:
    - low: low offline warning
    - medium: moderate offline warning
    - high: strong offline warning

They are not formal proofs of barren plateau.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_bp_risk_table(
    summary_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    source_name: str,
    calibration_cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Build BP risk table from summary and threshold artifacts.

    Args:
        summary_df:
            Stage 1 summary DataFrame.
        threshold_df:
            Threshold calibration table.
        source_name:
            stage1a or stage1b.
        calibration_cfg:
            calibration config section.

    Returns:
        BP risk DataFrame.

    Raises:
        ValueError:
            If required columns are unavailable.
    """
    if summary_df is None or summary_df.empty:
        raise ValueError(f"{source_name}.summary artifact is empty.")

    df = summary_df.copy()

    ensure_log_variance_columns(df, calibration_cfg)

    high_near_zero = float(calibration_cfg.get("high_risk_near_zero_ratio", 0.85))
    medium_near_zero = float(calibration_cfg.get("medium_risk_near_zero_ratio", 0.70))

    high_quantile = float(calibration_cfg.get("high_risk_variance_quantile", 0.25))
    medium_quantile = float(calibration_cfg.get("medium_risk_variance_quantile", 0.50))

    log_col = "log10_median_param_grad_variance_across_seeds"

    high_cut = float(df[log_col].quantile(high_quantile))
    medium_cut = float(df[log_col].quantile(medium_quantile))

    risk_levels = []
    risk_reasons = []

    for _, row in df.iterrows():
        log_var = float(row[log_col])
        near_zero = float(row.get("mean_param_near_zero_ratio", 0.0))

        if log_var <= high_cut or near_zero >= high_near_zero:
            risk_levels.append("high")
            risk_reasons.append(
                build_reason(
                    log_var=log_var,
                    near_zero=near_zero,
                    high_cut=high_cut,
                    medium_cut=medium_cut,
                    high_near_zero=high_near_zero,
                    medium_near_zero=medium_near_zero,
                    level="high",
                )
            )
        elif log_var <= medium_cut or near_zero >= medium_near_zero:
            risk_levels.append("medium")
            risk_reasons.append(
                build_reason(
                    log_var=log_var,
                    near_zero=near_zero,
                    high_cut=high_cut,
                    medium_cut=medium_cut,
                    high_near_zero=high_near_zero,
                    medium_near_zero=medium_near_zero,
                    level="medium",
                )
            )
        else:
            risk_levels.append("low")
            risk_reasons.append("gradient variance and near-zero ratio are not in warning region")

    df["bp_risk_level"] = risk_levels
    df["bp_risk_reason"] = risk_reasons
    df["calibration_source"] = source_name
    df["risk_high_log10_variance_cut"] = high_cut
    df["risk_medium_log10_variance_cut"] = medium_cut
    df["risk_high_near_zero_cut"] = high_near_zero
    df["risk_medium_near_zero_cut"] = medium_near_zero

    merge_cols = infer_merge_columns(df, threshold_df)

    if merge_cols:
        threshold_subset_cols = merge_cols + [
            col
            for col in [
                "component_grad_threshold",
                "normalized_grad_norm_threshold",
                "near_zero_threshold_dynamic",
            ]
            if col in threshold_df.columns
        ]

        df = df.merge(
            threshold_df[threshold_subset_cols].drop_duplicates(),
            on=merge_cols,
            how="left",
        )

    return df


def ensure_log_variance_columns(
    df: pd.DataFrame,
    calibration_cfg: dict[str, Any],
) -> None:
    """
    Ensure summary DataFrame has log10 median variance column.

    Args:
        df:
            Summary DataFrame.
        calibration_cfg:
            Calibration config.

    Raises:
        ValueError:
            If neither log nor raw median variance column exists.
    """
    epsilon = float(calibration_cfg.get("variance_epsilon", 1e-30))

    if "log10_median_param_grad_variance_across_seeds" in df.columns:
        return

    if "median_param_grad_variance_across_seeds" not in df.columns:
        raise ValueError(
            "Summary artifact must contain either "
            "log10_median_param_grad_variance_across_seeds or "
            "median_param_grad_variance_across_seeds."
        )

    df["log10_median_param_grad_variance_across_seeds"] = df[
        "median_param_grad_variance_across_seeds"
    ].apply(lambda x: np.log10(max(float(x), epsilon)))


def build_reason(
    log_var: float,
    near_zero: float,
    high_cut: float,
    medium_cut: float,
    high_near_zero: float,
    medium_near_zero: float,
    level: str,
) -> str:
    """
    Build human-readable risk reason.

    Args:
        log_var:
            log10 median variance.
        near_zero:
            Mean near-zero ratio.
        high_cut:
            High-risk log variance cutoff.
        medium_cut:
            Medium-risk log variance cutoff.
        high_near_zero:
            High-risk near-zero cutoff.
        medium_near_zero:
            Medium-risk near-zero cutoff.
        level:
            Assigned risk level.

    Returns:
        Reason string.
    """
    reasons = []

    if level == "high":
        if log_var <= high_cut:
            reasons.append("log10 median gradient variance is in high-risk lower quantile")
        if near_zero >= high_near_zero:
            reasons.append("near-zero ratio exceeds high-risk threshold")

    if level == "medium":
        if log_var <= medium_cut:
            reasons.append("log10 median gradient variance is in medium-risk region")
        if near_zero >= medium_near_zero:
            reasons.append("near-zero ratio exceeds medium-risk threshold")

    if not reasons:
        reasons.append("heuristic risk assignment")

    return "; ".join(reasons)


def infer_merge_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """
    Infer shared columns for merging risk and threshold tables.

    Args:
        left:
            Left DataFrame.
        right:
            Right DataFrame.

    Returns:
        Shared merge columns.
    """
    candidates = [
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

    return [col for col in candidates if col in left.columns and col in right.columns]