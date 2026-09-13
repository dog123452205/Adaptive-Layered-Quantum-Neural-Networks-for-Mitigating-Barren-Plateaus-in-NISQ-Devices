"""
Quantum data encoding utilities for Stage 1B.

Stage 1B v1 supports angle encoding with RY rotations:

    x_j -> RY(x_j)

The input features should already be scaled into a suitable angle range,
typically [0, pi].
"""

from __future__ import annotations

import pennylane as qml

from src.synthetic_bp.constants import EncodingType

def apply_data_encoding(
    features,
    n_qubits: int, 
    encoding_type: str,
    feature_map: str = 'ry_angle'
) -> None:
    """
    Apply classical-to-quantum data encoding.

    Args:
        features:
            One feature vector with shape (n_features,).
        n_qubits:
            Number of qubits.
        encoding_type:
            Encoding type. Stage 1B v1 supports angle.
        feature_map:
            Angle feature map. Supported:
            ry_angle, rx_angle, rz_angle.

    Raises:
        ValueError:
            If encoding type or feature map is unsupported.
    """
    if encoding_type == EncodingType.ANGLE.value:
        apply_angle_encoding(features, n_qubits, feature_map)
        return
    
    raise ValueError("Unsupported encoding type or feature map.")

def apply_angle_encoding(features, 
                         n_qubits: int,
                         feature_map: str) -> None:
    '''
    Apply angle encoding to the first n_features qubits.

    Args:
        features:
            Feature vector.
        n_qubits:
            Number of qubits.
        feature_map:
            ry_angle, rx_angle, or rz_angle.

    Notes:
        If n_features < n_qubits, remaining qubits are left in |0>.
        If n_features > n_qubits, an error is raised.
    '''
    n_features = len(features)

    if n_features > n_qubits:
        raise ValueError(
            f"Angle encoding requires n_features <= n_qubits. "
            f"Got n_features={n_features}, n_qubits={n_qubits}."
        )
    
    for wire, value in enumerate(features):
        if feature_map == 'ry_angle':
            qml.RY(value, wires=wire)
        elif feature_map == 'rx_angle':
            qml.RX(value, wires=wire)
        elif feature_map == 'rz_angle':
            qml.RZ(value, wires=wire)
        else:
            raise ValueError("Unsupported feature map.")
        