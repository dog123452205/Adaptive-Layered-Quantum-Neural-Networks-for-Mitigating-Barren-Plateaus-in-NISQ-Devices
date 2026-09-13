"""
Config resolver for Stage 0.

Responsibilities:
- merge YAML config with CLI overrides,
- resolve device/diff_method conflicts,
- attach reproducibility metadata,
- compute config hash,
- write config_resolved.yaml.

Important:
    If fallback changes diff_method, it must be recorded in config_resolved.yaml.
    Silent fallback is forbidden.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.synthetic_bp.config_loader import deep_merge, dump_yaml, parse_cli_overrides
from src.synthetic_bp.constants import (
    DeviceName,
    DiffFallbackPolicy,
    DiffMethod
)
from src.synthetic_bp.path_utils import atomic_write_text, ensure_parent_dir


def resolve_config(
    raw_config: dict[str, Any],
    cli_overrides: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Resolve a raw config into an executable config.

    Args:
        raw_config: Config loaded from YAML.
        cli_overrides: Optional CLI overrides in dot notation.

    Returns:
        Tuple of:
            resolved config,
            resolver messages.

    Raises:
        ValueError: If a conflict is found and fallback policy is ERROR.
    """
    messages: list[str] = []

    override_dict = parse_cli_overrides(cli_overrides)
    resolved = deep_merge(raw_config, override_dict)

    if override_dict:
        messages.append(f"Applied CLI overrides: {override_dict}")
    
    resolve_device_diff_method(resolved, messages)

    config_hash = compute_config_hash(resolved)

    attach_metadata(resolved, messages, config_hash)

    return resolved, messages

def resolve_device_diff_method(config: dict[str, Any], messages: list[str]) -> None:
    """
    Resolve device/diff_method compatibility.

    Rule:
        default.mixed + adjoint is invalid.

    If fallback policy is ERROR:
        raise ValueError.

    If fallback policy is WARN_AND_FALLBACK:
        set diff_method to fallback_diff_method and record the change.

    Args:
        config: Mutable resolved config.
        messages: Resolver message list.

    Raises:
        ValueError: On invalid combination under ERROR policy.
    """
    device = config.setdefault("device", {})
    device_name = device.get("name")
    diff_method = device.get("diff_method")

    fallback_policy = device.get("diff_fallback_policy", DiffFallbackPolicy.ERROR.value)
    fallback_diff = device.get("fallback_diff_method", DiffMethod.PARAMETER_SHIFT.value)

    invalid_mixed_adjoint = (
        device_name == DeviceName.DEFAULT_MIXED.value
        and diff_method == DiffMethod.ADJOINT.value
    )

    if not invalid_mixed_adjoint: return

    if fallback_policy == DiffFallbackPolicy.ERROR.value:
        raise ValueError(
            "Invalid config: device.name='default.mixed' is incompatible with "
            "diff_method='adjoint'. Set device.diff_fallback_policy='warn_and_fallback' "
            "if you explicitly want automatic fallback."
        )

    if fallback_policy == DiffFallbackPolicy.WARN_AND_FALLBACK.value:
        device['diff_method'] = fallback_diff
        messages.append(
            "Resolved invalid diff_method: default.mixed + adjoint "
            f"-> diff_method={fallback_diff}"
        )
        return
    
    raise ValueError(f"Unknown diff_fallback_policy: {fallback_policy}")

def attach_metadata(config: dict[str, Any], messages: list[str], config_hash: str) -> None:
    '''
    Attach reproducibility metadata to resolved config

    Args:
        config: Mutable resolved config
        messages: Resolver messages
    '''
    metadata = config.setdefault("metadata", {})

    metadata['resolved_at_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['python_version'] = sys.version
    metadata['platform'] = platform.platform()
    metadata['resolver_messages'] = list(messages)

    for package in ["pennylane", "numpy", "pandas", "pyyaml"]:
        metadata[f"{package}_version"] = safe_package_version(package)
    
    metadata["config_hash"] = config_hash


def safe_package_version(package_name:str) -> str | None:
    '''
    Return package version if installed

    Args:
        package_name: Package name.
    
    Returns:
        Version string or None
    '''
    try: 
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None

def compute_config_hash(
    config: dict[str, Any],
    exclude_metadata_hash: bool = True
) -> str:
    '''
    Compute a stable SHA256 hash of the resolved config.

    Args:
        config: Config dictionary.
        exclude_metadata_hash: Whether to exclude metadata.config_hash
            to avoid self-referential hashing.

    Returns:
        Hex digest string.
    '''
    config_copy = deepcopy(config)

    if exclude_metadata_hash:
        metadata = config_copy.get("metadata", {})
        metadata.pop("config_hash", None)
    
    text = dump_yaml(config_copy)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def write_resolved_config(config: dict[str, Any], output_path: str | Path) -> None:
    '''
    Write config_resolved.yaml atomically

    Args:
        config: Resolved config.
        output_path: Destination YAML path.

    Raises:
        RuntimeError: If the file cannot be written.
    '''
    output_path = Path(output_path)
    ensure_parent_dir(output_path)

    text = dump_yaml(config)
    atomic_write_text(output_path, text)

    if not output_path.exists():
        raise RuntimeError(f"Failed to write resolved config: {output_path}")
    
