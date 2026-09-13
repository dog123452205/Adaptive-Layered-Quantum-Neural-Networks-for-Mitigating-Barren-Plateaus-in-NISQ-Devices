"""
Stage 1A ansatz builders.

This module implements random-parameter HEA circuits for barren plateau
benchmarking.

Important:
    Stage 1A does not use data encoding.
    Stage 1A does not claim Haar randomness.
"""

from __future__ import annotations

import pennylane as qml
from src.synthetic_bp.constants import EntanglementTopology, EntanglerType

ROTATION_AXES = ('RX', 'RY', 'RZ')

def apply_hea(
    params, 
    n_qubits: int,
    depth: int,
    entanglement_topology: str,
    entangler_type: str
) -> None:
    """
    Apply a hardware-efficient ansatz.

    Each layer applies RX, RY, RZ rotations on every qubit,
    followed by an entanglement block.

    Args:
        params:
            Trainable rotation parameters with shape:
            (depth, n_qubits, 3).
        n_qubits:
            Number of qubits.
        depth:
            Number of HEA layers.
        entanglement_topology:
            Entanglement topology. Supported:
            none, linear, circular, tree, full.
        entangler_type:
            Entangler gate type. Stage 1A primarily uses cnot.

    Raises:
        ValueError:
            If input dimensions or topology are invalid.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    
    if depth <= 0:
        raise ValueError("depth must be positive")
    
    expected_shape = (depth, n_qubits, 3)
    if tuple(params.shape) != expected_shape:
        raise ValueError(
            f"Expected params shape {expected_shape}, got {tuple(params.shape)}"
        )
    
    for layer_id in range(depth):
        apply_rotation_block(params[layer_id], n_qubits)
        apply_entanglement_block(
            n_qubits = n_qubits,
            topology = entanglement_topology,
            entangler_type = entangler_type
        )

def apply_rotation_block(layer_params, n_qubit) -> None:
    '''
    Apply RX/RY/RZ rotations for one HEA layer.

    Args:
        layer_params:
            Parameters with shape (n_qubits, 3).
        n_qubits:
            Number of qubits.
    '''
    for wire in range(n_qubit):
        qml.RX(layer_params[wire, 0], wires=wire)
        qml.RZ(layer_params[wire, 0], wires=wire)
        qml.RY(layer_params[wire, 1], wires=wire)
        qml.RZ(layer_params[wire, 2], wires=wire)

def apply_entanglement_block(
    n_qubits: int,
    topology: str,
    entangler_type: str,
) -> None:
    """
    Apply an entanglement block.

    Args:
        n_qubits:
            Number of qubits.
        topology:
            Entanglement topology.
        entangler_type:
            Entangling gate type.

    Notes:
        CNOT and CZ are hard non-parameterized entanglers.
        Parameterized entanglers are reserved mainly for adaptive Stage 2/3.
    """
    edges = build_entanglement_edges(n_qubits, topology)

    for control, target in edges:
        apply_entangler(control, target, entangler_type)

def build_entanglement_edges(
    n_qubits: int,
    topology: str,
) -> list[tuple[int, int]]:
    '''
    Build entanglement edges for a given topology.

    Args:
        n_qubits:
            Number of qubits.
        topology:
            none, linear, circular, tree, full.

    Returns:
        List of directed wire pairs.

    Raises:
        ValueError:
            If topology is unsupported.
    '''
    if topology == EntanglementTopology.NONE.value: return []
    if topology == EntanglementTopology.LINEAR.value: 
        return [(i, i + 1) for i in range(n_qubits - 1)]
    
    if topology == EntanglementTopology.CIRCULAR.value:
        edges = [(i, i+  1) for i in range(n_qubits - 1)]
        if n_qubits > 2:
            edges.append((n_qubits-1,0))
        return edges
    
    if topology == EntanglementTopology.TREE.value:
        return [(i, 2 * i + 1) for i in range(n_qubits) if 2 * i + 1 < n_qubits] + [
            (i, 2 * i + 2) for i in range(n_qubits) if 2 * i + 2 < n_qubits
        ]

    if topology == EntanglementTopology.FULL.value:
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    raise ValueError(f"Unsupported entanglement topology: {topology}")

def apply_entangler(control: int, target: int, entangler_type: str) -> None:
    """
    Apply a supported non-parameterized entangler.

    Args:
        control:
            Control wire.
        target:
            Target wire.
        entangler_type:
            cnot or cz.

    Raises:
        ValueError:
            If a parameterized entangler is requested in Stage 1A.
    """
    if entangler_type == EntanglerType.CNOT.value:
        qml.CNOT(wires=[control, target])
        return
    
    if entangler_type == EntanglerType.CZ.value:
        qml.CZ(wires=[control, target])
        return
    
    raise ValueError(
        "Stage 1A engine currently supports hard entanglers cnot/cz only. "
        f"Got entangler_type={entangler_type}."
    )

