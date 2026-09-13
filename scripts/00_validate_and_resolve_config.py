"""
Validate and resolve AL-QNN config.

This script is mandatory before running Stage 1/2/3.

It produces:
    outputs/resolved_configs/config_resolved.yaml

Run:
    python scripts/00_validate_and_resolve_config.py \
        --config configs/stage1a_random_bp_smoke.yaml

With overrides:
    python scripts/00_validate_and_resolve_config.py \
        --config configs/stage1a_random_bp_smoke.yaml \
        --override sweep.seeds=5 \
        --override device.diff_method=adjoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.synthetic_bp.config_loader import load_yaml_config
from src.synthetic_bp.config_resolver import resolve_config, write_resolved_config
from src.synthetic_bp.config_validation import validate_config
from src.synthetic_bp.resource_guards import enforce_resource_guards


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Input YAML config path.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config using dot notation, e.g. sweep.seeds=10",
    )
    parser.add_argument(
        "--resolved-output",
        default="outputs/resolved_configs/config_resolved.yaml",
        help="Output path for resolved config.",
    )

    args = parser.parse_args()

    raw_config = load_yaml_config(args.config)

    resolved_config, resolver_messages = resolve_config(
        raw_config=raw_config,
        cli_overrides=args.override,
    )

    config_type = validate_config(resolved_config)
    guard_warnings = enforce_resource_guards(resolved_config)

    write_resolved_config(resolved_config, args.resolved_output)

    print(f"Config type: {config_type}")
    print(f"Resolved config written to: {args.resolved_output}")

    if resolver_messages:
        print("Resolver messages:")
        for message in resolver_messages:
            print(f"  - {message}")

    if guard_warnings:
        print("Resource guard warnings:")
        for warning in guard_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()