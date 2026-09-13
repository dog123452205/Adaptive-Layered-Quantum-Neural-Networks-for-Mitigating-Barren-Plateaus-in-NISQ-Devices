"""
YAML config loading, CLI override parsing, and immutable config utilities.

Stage 0 principle:
    This module only manipulates dictionaries and YAML files.
    It must not import PennyLane or create quantum circuits.
"""

from __future__ import annotations

import copy 
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from src.synthetic_bp.path_utils import require_file_exists

def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file

    Args:
        path: YAML file path.

    Returns:
        Config dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML file is empty or not a dictionary.
    """
    require_file_exists(path, "config")

    config_path = Path(path)
    with config_path.open("r", encoding= "utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a dictionary at the top level.")
    
    return config

def dump_yaml(config: Mapping[str, Any]) -> str:
    """
    Convert a mapping to stable YAML text.

    Args:
        config: Config mapping.

    Returns:
        YAML string.
    """
    return yaml.safe_dump(
        dict(config),
        sort_keys= True,
        allow_unicode=True,
        default_flow_style=False
    )

def parse_cli_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """
    Parse CLI overrides in dot notation.

    Example:
        ["device.diff_method=adjoint", "sweep.seeds=30"]

    Args:
        overrides: List of key=value override strings.

    Returns:
        Nested dictionary of overrides.
    """
    result: dict[str, Any] = {}
    if not overrides: return result

    for raw_items in overrides:
        if "=" not in raw_items:
            raise ValueError(f"Invalid override format: {raw_items}. Expected key=value")
        
        raw_key, raw_value = raw_items.split("=", 1)
        key = raw_key.strip()
        value = coerce_override_value(raw_value.strip())

        if not key:
            raise ValueError(f"Invalid override key in: {raw_items}")
        
        deep_set(result, key.split("."), value)
    
    return result

def coerce_override_value(value: str) -> Any:
    """
    Convert CLI override string to a Python value using YAML parsing.

    Args:
        value: Raw string value.

    Returns:
        Parsed Python value.
    """
    if value.lower() in ("none", "null", "false", "true"):
        if value.lower() == "true": return True
        if value.lower() == "false": return False
        return value.lower()
    
    try: return value if yaml.safe_load(value) is None else yaml.safe_load(value)
    except yaml.YAMLError: return value

def deep_set(target: dict[str, Any], path: list[str], value: Any) -> None:
    '''
    Set a nested dictionary field.

    Args:
        target: Target dictionary
        path: Key path
        value: Value to set
    '''
    cursor = target

    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    
    cursor[path[-1]] = value

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge override into base without redundant deepcopy penalty.

    Optimization: Clones the base structural tree exactly once at the entry point,
    then mutates the branches in-place during recursive traversal.
    """
    # Bản sao duy nhất được tạo ra ở tầng root của hàm gọi
    merged = copy.deepcopy(base)

    def _merge_in_place(target_dict: dict[str, Any], source_dict: dict[str, Any]) -> None:
        for key, value in source_dict.items():
            if (
                key in target_dict
                and isinstance(target_dict[key], dict)
                and isinstance(value, dict)
            ):
                # Khử đệ quy tạo bản sao mới, sửa đổi trực tiếp trên cây đã nhân bản
                _merge_in_place(target_dict[key], value)
            else:
                target_dict[key] = copy.deepcopy(value)

    _merge_in_place(merged, override)
    return merged

def freeze_mapping(obj: Any) -> Any:
    """
    Recursively freeze a config object.

    Dicts become MappingProxyType.
    Lists become tuples.

    Args:
        obj: Object to freeze.

    Returns:
        Immutable representation.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: freeze_mapping(v) for k, v in obj.items()})

    if isinstance(obj, list):
        return tuple(freeze_mapping(v) for v in obj)

    return obj
