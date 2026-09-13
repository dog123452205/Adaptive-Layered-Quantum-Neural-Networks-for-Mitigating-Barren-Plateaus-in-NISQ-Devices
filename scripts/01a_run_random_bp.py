"""
Run Stage 1A random-parameter BP benchmark.

Prerequisite:
    Run Stage 0 first to create a resolved config.

Example:
    python scripts/01a_run_random_bp.py \
        --config outputs/resolved_configs/stage1a_config_resolved.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.synthetic_bp.config_loader import load_yaml_config
from src.synthetic_bp.config_validation import validate_config
from src.synthetic_bp.resource_guards import enforce_resource_guards
from src.synthetic_bp.stage1a.random_bp_engine import Stage1ARandomBPEngine


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Resolved Stage 1A YAML config path.",
    )

    args = parser.parse_args()

    config = load_yaml_config(args.config)

    config_type = validate_config(config)
    if config_type != "stage1a_random_bp":
        raise ValueError(
            f"Expected stage1a_random_bp config, got config_type={config_type}"
        )

    enforce_resource_guards(config)

    engine = Stage1ARandomBPEngine(config)
    result = engine.run()

    print("Stage 1A random-parameter BP benchmark completed.")
    print(f"runs: {result['runs'].shape}")
    print(f"layers: {result['layers'].shape}")
    print(f"param_variance: {result['param_variance'].shape}")
    print(f"summary: {result['summary'].shape}")


if __name__ == "__main__":
    main()