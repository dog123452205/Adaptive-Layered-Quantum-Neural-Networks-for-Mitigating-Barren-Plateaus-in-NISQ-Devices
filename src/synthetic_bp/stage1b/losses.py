"""
Supervised loss functions for Stage 1B.

Stage 1B v1 supports binary cross entropy.
"""

from __future__ import annotations

import pennylane.numpy as pnp
import numpy as np

from src.synthetic_bp.constants import LossType

def binary_cross_entropy(probability, target, eps: float):
    '''
    Compute binary cross entropy.

    Args:
        probability:
            Predicted probability for class 1.
        target:
            Binary target, 0 or 1.
        eps:
            Numerical epsilon for stable log.

    Returns:
        Scalar loss.
    '''
    p = pnp.clip(probability, eps, 1.0 - eps)
    y = pnp.array(target)

    return -(y * pnp.log(p) + (1.0 - y) * pnp.log(1.0 - p))


def compute_supervised_loss(probability, target, loss_type: str, eps: float):
    """
    Dispatch supervised loss by type.

    Args:
        probability:
            Predicted probability for class 1.
        target:
            Binary target.
        loss_type:
            Loss type.
        eps:
            Numerical epsilon.

    Returns:
        Scalar loss.

    Raises:
        ValueError:
            If loss type is unsupported.
    """
    if loss_type == LossType.BINARY_CROSS_ENTROPY.value:
        return binary_cross_entropy(probability, target, eps)

    if loss_type == LossType.MSE.value:
        y = pnp.array(target)
        return (probability - y) ** 2

    raise ValueError(
        f"Stage 1B v1 supports binary_cross_entropy and mse. Got: {loss_type}"
    )