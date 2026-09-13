"""
Readout utilities for Stage 1B.

Stage 1B v1 uses a single-qubit Pauli-Z expectation readout.

If z = <Z>, then z lies in [-1, 1]. We map it to a probability by:

    p(y=1 | x) = (1 - z) / 2

This convention means:
    z = +1 -> class 0 probability high
    z = -1 -> class 1 probability high
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as pnp
import numpy as np

from src.synthetic_bp.constants import ReadoutType

def build_readout_observable(readout_cfg: dict):
    """
    Build the readout observable.

    Args:
        readout_cfg:
            Readout config section.

    Returns:
        PennyLane observable.

    Raises:
        ValueError:
            If readout type is unsupported.
    """
    readout_type = readout_cfg["type"]

    if readout_type == ReadoutType.SINGLE_QUBIT_Z.value:
        wire = int(readout_cfg.get("readout_wire", 0))

        return qml.PauliZ(wire)
    
    raise ValueError(
        f"Stage 1B v1 supports readout.type='single_qubit_z' only. "
        f"Got: {readout_type}"
    )

def z_expectation_to_probability(z_expectation, eps: float):
    """
    Convert Pauli-Z expectation value to binary class-1 probability.

    Args:
        z_expectation:
            Expectation value in [-1, 1].
        eps:
            Numerical epsilon for clipping.

    Returns:
        Probability tensor in [eps, 1 - eps].
    """
    probability = 0.5 * (1.0 - z_expectation)
    return pnp.clip(probability, eps, 1.0 - eps)
