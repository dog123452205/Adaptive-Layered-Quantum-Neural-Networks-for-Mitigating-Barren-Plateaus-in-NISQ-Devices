"""
Stage 1A gradient metrics.

Important distinction:
    spatial_grad_variance:
        Variance across parameters within one circuit run.
        This is a runtime diagnostic.

    param_grad_variance_across_seeds:
        Variance of a fixed parameter's gradient across random seeds.
        This is the primary BP evidence.
"""

import math
from typing import Any

from pennylane import numpy as pnp
import numpy as np

from src.synthetic_bp.stage1a.ansatz import ROTATION_AXES

def compute_gradient_metrics(
        grad, 
        near_zero_threshold: float,
        log_epsilon: float) -> dict[str, float]:
    """
    Compute run-level gradient diagnostics.

    Args:
        grad:
            Gradient tensor with shape (depth, n_qubits, 3).
        near_zero_threshold:
            Threshold for near-zero gradient components.
        log_epsilon:
            Small positive value used to avoid log10(0).

    Returns:
        Dictionary of scalar metrics.
    """
    grad_flat = np.ravel(grad)
    abs_grad  = np.abs(grad_flat)

    spatial_var = float(np.var(grad_flat))
    grad_norm = float(np.linalg.norm(grad_flat))
    n_params = int(grad_flat.shape[0])

    return {
        "mean_abs_grad": float(np.mean(abs_grad)),
        "spatial_grad_variance": spatial_var,
        "log10_spatial_grad_variance": float(math.log10(max(spatial_var, log_epsilon))),
        "grad_norm": grad_norm,
        "normalized_grad_norm": float(grad_norm / math.sqrt(max(n_params,1))),
        "max_abs_grad": float(np.max(abs_grad)),
        "min_abs_grad": float(np.min(abs_grad)),
        "near_zero_ratio": float(np.mean(abs_grad < near_zero_threshold)),
        "target_grad": float(grad[0, 0, 0]),
    }

def compute_layerwise_metrics(
        run_id: int,
        grad,
        near_zero_threshold: float) -> list[dict[str, Any]]:
    """
    Compute layer-level gradient diagnostics.

    Args:
        run_id:
            Unique run ID.
        grad:
            Gradient tensor with shape (depth, n_qubits, 3).
        near_zero_threshold:
            Threshold for near-zero gradient components.

    Returns:
        List of layer-level metric records.
    """
    layer_records: list[dict[str, Any]] = []

    for layer_id in range(grad.shape[0]):
        layer_grad = np.ravel(grad[layer_id])
        abs_grad = np.abs(layer_grad)

        layer_records.append({
            "run_id": int(run_id),
            "layer_id": int(layer_id),
            "layer_mean_abs_grad": float(np.mean(abs_grad)),
            "layer_grad_variance": float(np.var(layer_grad)),
            "layer_grad_norm": float(np.linalg.norm(layer_grad)),
            "layer_max_abs_grad": float(np.max(abs_grad)),
            "layer_min_abs_grad": float(np.min(abs_grad)),
            "layer_near_zero_ratio": float(
                np.mean(abs_grad < near_zero_threshold)
            ),
        })

    return layer_records

def iter_gradient_components(grad):
    '''
    Iterate over gradient tensor components.

    Args:
        grad:
            Gradient tensor with shape (depth, n_qubits, 3).

    Yields:
        Tuple:
            layer_id, qubit_id, axis_id, axis_name, param_index, grad_value.
    '''
    param_index = 0

    for layer_id in range(grad.shape[0]):
        for qubit_id in range(grad.shape[1]):
            for axis_id, axis_name in enumerate(ROTATION_AXES):
                yield (
                    int(layer_id),
                    int(qubit_id),
                    int(axis_id),
                    axis_name,
                    int(param_index),
                    float(grad[layer_id, qubit_id, axis_id]),
                )
                param_index += 1