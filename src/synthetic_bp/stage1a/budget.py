"""
Budget normalization utilities for Stage 1A.

When comparing topologies such as LINEAR, CIRCULAR, and FULL,
same-depth comparison can be misleading because the number of entangling
gates per layer differs significantly.

This module supports entangling-gate-count normalization.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.synthetic_bp.constants import BudgetNormalizationMode, EntanglementTopology
from src.synthetic_bp.stage1a.ansatz import build_entanglement_edges

@dataclass(frozen=True)
class DepthPlan:
    """
    Effective depth plan for one benchmark configuration.

    Attributes:
        reference_depth:
            Depth requested by the sweep config.
        effective_depth:
            Actual depth used after budget normalization.
        target_entangling_budget:
            Target number of entangling gates.
        actual_entangling_budget:
            Actual number of entangling gates used.
        gates_per_layer:
            Number of entangling gates per layer for this topology.
    """

    reference_depth: int
    effective_depth: int
    target_entangling_budget: int
    actual_entangling_budget: int
    gates_per_layer: int

def entangling_gates_per_layer(n_qubits: int, topology: str) -> int:
    '''
    Count entangling gates per layer for a topology.

    Args:
        n_qubits:
            Number of qubits.
        topology:
            Entanglement topology.

    Returns:
        Number of entangling gates per layer.
    '''
    return len(build_entanglement_edges(n_qubits, topology))

def plan_effective_depth(
    n_qubits: int,
    reference_depth: int, 
    topology: str,
    budget_normalization: str,
    reference_topology: str = EntanglementTopology.LINEAR.value
) -> DepthPlan:
    '''
    Resolve effective depth under a budget normalization rule.

    Args:
        n_qubits:
            Number of qubits.
        reference_depth:
            Depth from YAML sweep.
        topology:
            Topology being evaluated.
        budget_normalization:
            Budget normalization mode.
        reference_topology:
            Topology used to define target entangling budget.

    Returns:
        DepthPlan.

    Notes:
        For same_entangling_gate_count, the target budget is computed from
        reference_topology. The effective depth is rounded to the nearest
        positive integer. Exact equality may not be possible for small circuits.
    '''
    if reference_depth <= 0:
        raise ValueError("reference_depth must be positive.")
    
    gates_this = entangling_gates_per_layer(n_qubits, topology)
    gates_ref = entangling_gates_per_layer(n_qubits, reference_topology)

    if budget_normalization == BudgetNormalizationMode.SAME_DEPTH.value:
        effective_depth = reference_depth
        target_budget = reference_depth * gates_this
        actual_budget = effective_depth * gates_this
        return DepthPlan(
            reference_depth=reference_depth,
            effective_depth=effective_depth,
            target_entangling_budget=target_budget,
            actual_entangling_budget=actual_budget,
            gates_per_layer=gates_this
        )
    
    if budget_normalization == BudgetNormalizationMode.SAME_ENTANGLING_GATE_COUNT.value:
        target_budget = reference_depth * gates_ref
    
        if gates_this == 0:
            effective_depth = reference_depth
            actual_budget = 0
        else:
            effective_depth = max(1, round(target_budget / gates_this))
            actual_budget = effective_depth * gates_this
        
        return DepthPlan(
            reference_depth=reference_depth,
            effective_depth=effective_depth,
            target_entangling_budget=target_budget,
            actual_entangling_budget=actual_budget,
            gates_per_layer=gates_this
        )
    
    if budget_normalization == BudgetNormalizationMode.NONE.value:
        effective_depth = reference_depth
        target_budget = reference_depth * gates_this
        actual_budget = effective_depth * gates_this
        return DepthPlan(
            reference_depth=reference_depth,
            effective_depth=effective_depth,
            target_entangling_budget=target_budget,
            actual_entangling_budget=actual_budget,
            gates_per_layer=gates_this,
        )
    
    raise ValueError(
        "Stage 1A engine currently supports budget_normalization in "
        "{none, same_depth, same_entangling_gate_count}. "
        f"Got: {budget_normalization}"
    )

def count_hea_parameters(n_qubits: int, depth: int) -> int:
    """
    Count trainable rotation parameters for ZYZ (RZ/RY/RZ) HEA.

    Args:
        n_qubits:
            Number of qubits.
        depth:
            Number of HEA layers.

    Returns:
        Number of trainable parameters.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    
    if depth <= 0:
        raise ValueError("depth must be positive")
    
    return depth * n_qubits * 3
