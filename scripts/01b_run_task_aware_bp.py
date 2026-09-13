"""
Run Stage 1B task-aware BP benchmark.

Prerequisite:
    Run Stage 0 first to create a resolved Stage 1B config.

Example:
    python scripts/01b_run_task_aware_bp.py \
        --config outputs/resolved_configs/stage1b_config_resolved.yaml
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
from src.synthetic_bp.stage1b.task_aware_bp_engine import Stage1BTaskAwareBPEngine

def main() -> None:
    """
    CLI entry point for Stage 1B.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Resolved Stage 1B YAML config path.",
    )

    args = parser.parse_args()

    config = load_yaml_config(args.config)

    config_type = validate_config(config)
    if config_type != "stage1b_task_aware_bp":
        raise ValueError(
            f"Expected stage1b_task_aware_bp config, got config_type={config_type}"
        )

    enforce_resource_guards(config)

    engine = Stage1BTaskAwareBPEngine(config)
    result = engine.run()

    print("Stage 1B task-aware BP benchmark completed.")
    print(f"runs: {result['runs'].shape}")
    print(f"layers: {result['layers'].shape}")
    print(f"param_variance: {result['param_variance'].shape}")
    print(f"summary: {result['summary'].shape}")


if __name__ == "__main__":
    main()