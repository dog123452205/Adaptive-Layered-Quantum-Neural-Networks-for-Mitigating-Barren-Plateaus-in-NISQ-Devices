"""
Run Stage 1C Calibration Artifact Builder.

Example:
    python scripts/01c_build_calibration.py \
        --config configs/stage1c_calibration_v1.yaml

This script does not instantiate quantum circuits.
It only reads Stage 1A/1B artifacts and writes compact calibration artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.synthetic_bp.config_loader import load_yaml_config
from src.synthetic_bp.stage1c.calibration_builder import Stage1CCalibrationBuilder


def main() -> None:
    """
    CLI entry point for Stage 1C.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Stage 1C calibration YAML config path.",
    )

    args = parser.parse_args()

    config = load_yaml_config(args.config)

    builder = Stage1CCalibrationBuilder(config)
    result = builder.run()

    print("Stage 1C calibration artifacts built.")
    print(f"threshold_calibration_table: {result['threshold_calibration_table'].shape}")
    print(f"bp_risk_table: {result['bp_risk_table'].shape}")
    print(f"calibration_report: {result['calibration_report'].shape}")


if __name__ == "__main__":
    main()