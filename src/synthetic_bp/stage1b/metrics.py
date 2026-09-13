"""
Stage 1B task-aware gradient metrics.

These metrics mirror Stage 1A but replace the expectation-value objective
with supervised batch loss:

    L_B(θ) = mean_i loss(fθ(x_i), y_i)

Important:
    param_grad_variance_across_seeds remains the main offline evidence.
    spatial_grad_variance remains a within-run diagnostic only.
"""

from __future__ import annotations

import math
from typing import Any

import pennylane.numpy as pnp


def compute_task_run_metrics(
    grad,
    near_zero_threshold: float,
    log_epsilon: float,
) -> dict[str, float]:
    """
    Compute run-level gradient metrics for task-aware loss.

    Args:
        grad:
            Gradient tensor with shape (depth, n_qubits, 3).
        near_zero_threshold:
            Threshold used to count near-zero gradients.
        log_epsilon:
            Small positive value for safe log10.

    Returns:
        Dictionary of scalar metrics.
    """
    grad_flat = pnp.ravel(grad)
    abs_grad = pnp.abs(grad_flat)

    spatial_var = float(pnp.var(grad_flat))
    grad_norm = float(pnp.linalg.norm(grad_flat))
    n_params = int(grad_flat.shape[0])

    return {
        "mean_abs_grad": float(pnp.mean(abs_grad)),
        "spatial_grad_variance": spatial_var,
        "log10_spatial_grad_variance": float(math.log10(max(spatial_var, log_epsilon))),
        "grad_norm": grad_norm,
        "normalized_grad_norm": float(grad_norm / math.sqrt(max(n_params, 1))),
        "max_abs_grad": float(pnp.max(abs_grad)),
        "min_abs_grad": float(pnp.min(abs_grad)),
        "near_zero_ratio": float(pnp.mean(abs_grad < near_zero_threshold)),
        "target_grad": float(grad[0, 0, 0]),
    }


def compute_prediction_metrics(
    probabilities,
    labels,
) -> dict[str, float]:
    """
    Compute simple prediction metrics on the fixed benchmark batch.

    Args:
        probabilities:
            Predicted probabilities for class 1.
        labels:
            Binary labels.

    Returns:
        Dictionary with batch accuracy and mean probability.
    """
    probs = pnp.asarray(probabilities)
    y = pnp.asarray(labels)

    preds = (probs >= 0.5).astype(int)

    return {
        "batch_accuracy": float(pnp.mean(preds == y)),
        "mean_pred_probability": float(pnp.mean(probs)),
        "std_pred_probability": float(pnp.std(probs)),
    }


def compute_layer_metrics_task(
    run_id: int,
    grad,
    near_zero_threshold: float,
) -> list[dict[str, Any]]:
    """
    Compute layer-level gradient metrics for Stage 1B.

    Args:
        run_id:
            Unique run ID.
        grad:
            Gradient tensor with shape (depth, n_qubits, 3).
        near_zero_threshold:
            Near-zero gradient threshold.

    Returns:
        List of layer-level records.
    """
    records: list[dict[str, Any]] = []

    for layer_id in range(grad.shape[0]):
        layer_grad = pnp.ravel(grad[layer_id])
        abs_grad = pnp.abs(layer_grad)

        records.append(
            {
                "run_id": int(run_id),
                "layer_id": int(layer_id),
                "layer_mean_abs_grad": float(pnp.mean(abs_grad)),
                "layer_spatial_grad_variance": float(pnp.var(layer_grad)),
                "layer_grad_norm": float(pnp.linalg.norm(layer_grad)),
                "layer_max_abs_grad": float(pnp.max(abs_grad)),
                "layer_min_abs_grad": float(pnp.min(abs_grad)),
                "layer_near_zero_ratio": float(pnp.mean(abs_grad < near_zero_threshold)),
            }
        )

    return records