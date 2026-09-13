
"""
Stage 1A observables.

These observables define the random-parameter BP benchmark objective:

    E(θ) = <0| U†(θ) O U(θ) |0>

Supported objectives:
    local:
        Mean single-qubit Z observable.
    global:
        Product of Z over all qubits.
    hamiltonian_local:
        Mean nearest-neighbor ZZ Hamiltonian.
"""
from __future__ import annotations

import pennylane as qml

from src.synthetic_bp.constants import CostType

def build_observable(n_qubits: int, cost_type: str):
    """
    Build the observable for the synthetic barren plateau benchmark.

    cost_type = local:
        C(θ) = (1/n) * Σ_i <Z_i>

    cost_type = global:
        C(θ) = <Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_{n-1}>

    cost_type = hamiltonian_local:
        C(θ) = (1/(n-1)) * Σ_i <Z_i Z_{i+1}>

    Args:
        n_qubits:
            Number of qubits.
        cost_type:
            local, global, or hamiltonian_local.

    Returns:
        PennyLane observable.

    Raises:
        ValueError:
            If n_qubits or cost_type is invalid.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    
    if cost_type == CostType.LOCAL.value:
        coeffs = [1.0 / n_qubits for _ in range(n_qubits)]
        observables = [qml.PauliZ(i) for i in range(n_qubits)]
        return qml.Hamiltonian(coeffs, observables)
    
    if cost_type == CostType.GLOBAL.value:
        return qml.prod(*[qml.PauliZ(i) for i in range(n_qubits)])

    if cost_type == CostType.HAMILTONIAN_LOCAL.value:
        if n_qubits < 2:
            raise ValueError("Hamiltonian local requires at least 2 qubits")
        
        coeffs = [1.0 / (n_qubits -1) for _ in range(n_qubits -1)]
        obs = [
            qml.prod(qml.PauliZ(i), qml.PauliZ(i  + 1))
            for i in range(n_qubits -1)
        ]
        return qml.Hamiltonian(coeffs, obs)

    raise ValueError(f"Unknown cost_type: {cost_type}")