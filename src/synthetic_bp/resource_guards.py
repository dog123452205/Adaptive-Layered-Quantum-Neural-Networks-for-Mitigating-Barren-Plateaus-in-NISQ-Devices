"""
Operational resource guards for AL-QNN.

This module prevents unsafe configurations before quantum simulation starts.

It only inspects scalar config values.
It does not instantiate devices or quantum circuits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.synthetic_bp.constants import (
    DeviceName,
    DiffMethod,
    HARD_MAX_BACKPROP_QUBITS,
    HARD_MAX_DENSITY_MATRIX_QUBITS,
    HARD_MAX_HARDWARE_DEMO_QUBITS,
    HARD_MAX_STATEVECTOR_QUBITS,
    ResourceLimitPolicy,
    ResourceLimitSource,
    SimulationProfile,
    STRESS_MAX_STATEVECTOR_QUBITS,
)

@dataclass(frozen=True)
class ResourceLimit:
    '''
    Resource limit result.

    Attributes:
        max_qubits: Maximum allowed qubits.
        source: Where the limit came from.
        reason: Human-readable explanation.
    '''
    max_qubits: int
    source: ResourceLimitSource
    reason: str

def extract_max_requested_qubits(config: dict[str, Any]) -> int:
    '''
    Extract maximum requested qubit count from config.

    Supports:
    - Stage 1 configs with sweep.n_qubits,
    - adaptive configs with architecture.n_qubits if present.

    Args:
        config: Resolved config.

    Returns:
        Maximum requested qubit count.

    Raises:
        ValueError: If no qubit information is found.
    '''
    if "sweep" in config and "n_qubits" in config["sweep"]:
        n_values = config['sweep']['n_qubits']
        if not isinstance(n_values, list) or not n_values:
            raise ValueError("sweep.n_qubits must be a non-empty list.")
        return max(int(n) for n in n_values)
    
    architecture = config.get('architecture', {})
    if "n_qubits"in architecture:
        return int(architecture["n_qubits"])
    
    raise ValueError("Could not determine requested n_qubits from config.")

def resolve_max_qubit_limit(config: dict[str, Any]) -> ResourceLimit:
    """
    Resolve maximum allowed qubits using constants, YAML override, or env override.

    Priority:
        1. Environment variable if resource policy is OVERRIDABLE.
        2. YAML resource_overrides if policy is OVERRIDABLE.
        3. Conservative default constants.

    Args:
        config: Resolved config.

    Returns:
        ResourceLimit object.
    """
    env_limit = read_env_int("ALQNN_MAX_QUBITS")
    if env_limit is not None:
        return ResourceLimit(
            max_qubits=env_limit,
            source=ResourceLimitSource.ENV_OVERRIDE,
            reason="ALQNN_MAX_QUBITS environment override.",
        )
    
    safety = config.get("safety", {})
    device = config.get("device", {})
    policy = safety.get("resource_limit_policy", ResourceLimitPolicy.STRICT.value)
    profile = safety.get("simulation_profile", SimulationProfile.MAIN.value)
    yaml_overrides = safety.get("resource_overrides", {}) or {}

    if policy == ResourceLimitPolicy.OVERRIDABLE.value:
        

        yaml_limit = yaml_overrides.get("max_qubits")
        if yaml_limit is not None:
            return ResourceLimit(
                max_qubits=int(yaml_limit),
                source=ResourceLimitSource.YAML_OVERRIDE,
                reason="YAML safety.resource_overrides.max_qubits override.",
            )

    device_name = device.get("name")
    diff_method = device.get("diff_method")

    if profile == SimulationProfile.HARDWARE_DEMO.value:
        return ResourceLimit(
            max_qubits=HARD_MAX_HARDWARE_DEMO_QUBITS,
            source=ResourceLimitSource.CONSTANTS_DEFAULT,
            reason="Hardware demo profile.",
        )

    if device_name == DeviceName.DEFAULT_MIXED.value:
        return ResourceLimit(
            max_qubits=HARD_MAX_DENSITY_MATRIX_QUBITS,
            source=ResourceLimitSource.CONSTANTS_DEFAULT,
            reason="Density-matrix simulation scales as O(4^n).",
        )

    if diff_method == DiffMethod.BACKPROP.value:
        return ResourceLimit(
            max_qubits=HARD_MAX_BACKPROP_QUBITS,
            source=ResourceLimitSource.CONSTANTS_DEFAULT,
            reason="Backprop stores computation graph and is memory intensive.",
        )

    if profile == SimulationProfile.STRESS.value:
        return ResourceLimit(
            max_qubits=STRESS_MAX_STATEVECTOR_QUBITS,
            source=ResourceLimitSource.CONSTANTS_DEFAULT,
            reason="Stress profile statevector cap.",
        )

    return ResourceLimit(
        max_qubits=HARD_MAX_STATEVECTOR_QUBITS,
        source=ResourceLimitSource.CONSTANTS_DEFAULT,
        reason="Default statevector hard cap.",
    )


def read_env_int(name: str) -> int | None:
    """
    Read an integer environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Integer value or None.

    Raises:
        ValueError: If env var exists but is not an integer.
    """
    raw = os.environ.get(name)

    if raw is None or raw.strip() == "":
        return None

    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def enforce_resource_guards(config: dict[str, Any]) -> list[str]:
    """
    Enforce resource limits.

    Args:
        config: Resolved config.

    Returns:
        Warning messages.

    Raises:
        ValueError: If config violates strict resource limits.
    """
    warnings: list[str] = []

    safety = config.get("safety", {})
    enforce = safety.get("enforce_resource_limits", True)

    if not enforce:
        warnings.append("Resource limits are disabled by config.")
        return warnings

    policy = safety.get("resource_limit_policy", ResourceLimitPolicy.STRICT.value)

    requested = extract_max_requested_qubits(config)
    limit = resolve_max_qubit_limit(config)

    # 1. Kiểm tra giới hạn Qubit
    if requested > limit.max_qubits:
        message = (
            f"Requested n_qubits = {requested} exceeds max_qubits={limit.max_qubits}"
            f"Limit source = {limit.source.value}. Reason={limit.reason}"
        )
        if policy == ResourceLimitPolicy.WARN_ONLY.value:
            warnings.append(message)
        else:
            raise ValueError(message)
    
    # 2. Kiểm tra tích hợp bộ nhớ động với Batch Size
    batch_size = 1
    if "training" in config and "batch_size" in config['training']:
        batch_size = int(config["training"]["batch_size"])
    elif "batching" in config and 'batch_size' in config["batching"]:
        batch_size = int(config['batching']['batch_size'])
    
    # Ngưỡng cảnh báo: Nếu dung lượng thô của Batch Statevectors vượt quá giới hạn an toàn vật lý
    # Giả định một ngưỡng danh định phối hợp (ví dụ: nếu > 28 qubits hệ đơn lẻ, hoặc tổ hợp vượt ngưỡng)
    device = config.get("device", {})
    if device.get("diff_method") == DiffMethod.BACKPROP.value and requested >= 20:
        # Ước lượng dung lượng RAM thô của batch (chưa tính đồ thị lưu vết đạo hàm)
        approx_batch_mem_gb = ( (2 ** requested) * 16 * batch_size ) / (1024 ** 3)
        
        # Nếu dung lượng vượt quá 48GB RAM (Cấu hình tới hạn thông thường của một Node)
        if approx_batch_mem_gb > 48.0:
            mem_message = (
                f"Risk of Catastrophic OOM: Multiplied footprint (n_qubits={requested} "
                f"x batch_size={batch_size}) requires approx {approx_batch_mem_gb:.2f} GB RAM "
                f"for forward statevectors alone during backpropagation."
            )
            if policy == ResourceLimitPolicy.STRICT.value:
                raise ValueError(mem_message)
            else:
                warnings.append(mem_message)

    return warnings