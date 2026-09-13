"""
Stage 1C Calibration Artifact Builder.

This module converts Stage 1A and Stage 1B artifacts into scheduler-readable
calibration artifacts.

Inputs:
    Stage 1A:
        runs.csv/parquet
        param_variance.csv/parquet
        summary.csv/parquet

    Stage 1B:
        runs.csv/parquet
        param_variance.csv/parquet
        summary.csv/parquet

Outputs:
    calibration_rules.json
    bp_risk_table.parquet/csv
    threshold_calibration_table.parquet/csv
    calibration_report.csv

Design:
    Scheduler must not directly depend on raw Stage 1 artifacts.
    It should read the compact calibration artifacts from Stage 1C.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.synthetic_bp.stage1c.io_utils import load_table, save_table, write_json
from src.synthetic_bp.stage1c.thresholds import build_threshold_table
from src.synthetic_bp.stage1c.risk import build_bp_risk_table


class Stage1CCalibrationBuilder:
    """
    Build Stage 1C calibration artifacts.

    Args:
        config:
            Stage 1C config dictionary.

    Main outputs:
        - calibration_rules.json
        - bp_risk_table
        - threshold_calibration_table
        - calibration_report
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.experiment = config["experiment"]
        self.inputs = config["inputs"]
        self.outputs = config["outputs"]
        self.calibration_cfg = config["calibration"]

    def run(self) -> dict[str, pd.DataFrame]:
        """
        Execute Stage 1C calibration building.

        Returns:
            Dictionary with:
                threshold_calibration_table,
                bp_risk_table,
                calibration_report.
        """
        self._validate_config_minimal()

        stage1a_artifacts = self._load_source_artifacts("stage1a")
        stage1b_artifacts = self._load_source_artifacts("stage1b")

        threshold_tables = []
        risk_tables = []
        report_records = []

        if stage1a_artifacts["enabled"]:
            stage1a_thresholds = build_threshold_table(
                param_df=stage1a_artifacts["param_variance"],
                run_df=stage1a_artifacts["runs"],
                source_name="stage1a",
                calibration_cfg=self.calibration_cfg,
            )

            stage1a_risk = build_bp_risk_table(
                summary_df=stage1a_artifacts["summary"],
                threshold_df=stage1a_thresholds,
                source_name="stage1a",
                calibration_cfg=self.calibration_cfg,
            )

            threshold_tables.append(stage1a_thresholds)
            risk_tables.append(stage1a_risk)
            report_records.append(self._build_source_report("stage1a", stage1a_artifacts))

        if stage1b_artifacts["enabled"]:
            stage1b_thresholds = build_threshold_table(
                param_df=stage1b_artifacts["param_variance"],
                run_df=stage1b_artifacts["runs"],
                source_name="stage1b",
                calibration_cfg=self.calibration_cfg,
            )

            stage1b_risk = build_bp_risk_table(
                summary_df=stage1b_artifacts["summary"],
                threshold_df=stage1b_thresholds,
                source_name="stage1b",
                calibration_cfg=self.calibration_cfg,
            )

            threshold_tables.append(stage1b_thresholds)
            risk_tables.append(stage1b_risk)
            report_records.append(self._build_source_report("stage1b", stage1b_artifacts))

        if not threshold_tables:
            raise ValueError("No enabled Stage 1 source artifacts were loaded.")

        threshold_calibration_table = pd.concat(
            threshold_tables,
            ignore_index=True,
            sort=False,
        )

        bp_risk_table = pd.concat(
            risk_tables,
            ignore_index=True,
            sort=False,
        )

        calibration_report = pd.DataFrame(report_records)

        calibration_rules = self._build_calibration_rules(
            threshold_table=threshold_calibration_table,
            risk_table=bp_risk_table,
            report_df=calibration_report,
        )

        self._save_outputs(
            threshold_table=threshold_calibration_table,
            risk_table=bp_risk_table,
            report_df=calibration_report,
            calibration_rules=calibration_rules,
        )

        return {
            "threshold_calibration_table": threshold_calibration_table,
            "bp_risk_table": bp_risk_table,
            "calibration_report": calibration_report,
        }

    def _validate_config_minimal(self) -> None:
        """
        Validate required Stage 1C config sections.

        Raises:
            ValueError:
                If required config fields are missing.
        """
        required_sections = ["experiment", "inputs", "outputs", "calibration"]
        missing = [section for section in required_sections if section not in self.config]

        if missing:
            raise ValueError(f"Stage 1C config missing sections: {missing}")

        required_outputs = [
            "calibration_rules_path",
            "bp_risk_table_path",
            "threshold_calibration_table_path",
            "calibration_report_path",
        ]

        missing_outputs = [
            key for key in required_outputs if key not in self.outputs
        ]

        if missing_outputs:
            raise ValueError(f"Stage 1C outputs missing keys: {missing_outputs}")

    def _load_source_artifacts(self, source_name: str) -> dict[str, Any]:
        """
        Load Stage 1A or Stage 1B artifacts.

        Args:
            source_name:
                "stage1a" or "stage1b".

        Returns:
            Dictionary containing enabled flag and DataFrames.

        Raises:
            ValueError:
                If config for source is invalid.
        """
        source_cfg = self.inputs.get(source_name, {})
        enabled = bool(source_cfg.get("enabled", False))

        if not enabled:
            return {
                "enabled": False,
                "runs": pd.DataFrame(),
                "param_variance": pd.DataFrame(),
                "summary": pd.DataFrame(),
            }

        runs_path = source_cfg.get("runs_path")
        param_variance_path = source_cfg.get("param_variance_path")
        summary_path = source_cfg.get("summary_path")

        if not param_variance_path or not summary_path:
            raise ValueError(
                f"{source_name} requires param_variance_path and summary_path."
            )

        runs = load_table(runs_path, required=False) if runs_path else pd.DataFrame()
        param_variance = load_table(param_variance_path, required=True)
        summary = load_table(summary_path, required=True)

        return {
            "enabled": True,
            "runs": runs,
            "param_variance": param_variance,
            "summary": summary,
            "paths": {
                "runs_path": runs_path,
                "param_variance_path": param_variance_path,
                "summary_path": summary_path,
            },
        }

    def _build_source_report(
        self,
        source_name: str,
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a compact source report row.

        Args:
            source_name:
                stage1a or stage1b.
            artifacts:
                Loaded artifact dictionary.

        Returns:
            Report record.
        """
        runs = artifacts["runs"]
        param_variance = artifacts["param_variance"]
        summary = artifacts["summary"]

        return {
            "source": source_name,
            "runs_rows": int(len(runs)),
            "param_variance_rows": int(len(param_variance)),
            "summary_rows": int(len(summary)),
            "has_runs": bool(not runs.empty),
            "has_param_variance": bool(not param_variance.empty),
            "has_summary": bool(not summary.empty),
            "runs_path": artifacts.get("paths", {}).get("runs_path"),
            "param_variance_path": artifacts.get("paths", {}).get("param_variance_path"),
            "summary_path": artifacts.get("paths", {}).get("summary_path"),
        }

    def _build_calibration_rules(
        self,
        threshold_table: pd.DataFrame,
        risk_table: pd.DataFrame,
        report_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Build scheduler-readable calibration rules.

        Args:
            threshold_table:
                Threshold calibration table.
            risk_table:
                BP risk table.
            report_df:
                Calibration report DataFrame.

        Returns:
            JSON-serializable calibration rules dictionary.
        """
        threshold_priority = self.calibration_cfg.get(
            "threshold_priority",
            [
                "stage1b_task_aware_matched",
                "stage1a_random_matched",
                "n_qubit_scaled_fallback",
                "fixed_fallback",
            ],
        )

        matching_keys = self.calibration_cfg.get(
            "matching_keys",
            [
                "benchmark_mode",
                "dataset_name",
                "encoding_type",
                "ansatz_type",
                "gate_set",
                "entanglement_topology",
                "entangler_type",
                "n_qubits",
                "depth",
                "loss_type",
                "cost_type",
            ],
        )

        risk_counts = (
            risk_table["bp_risk_level"].value_counts().to_dict()
            if "bp_risk_level" in risk_table.columns
            else {}
        )

        return {
            "version": self.calibration_cfg.get("version", "stage1c_v1"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_name": self.experiment.get("name"),
            "experiment_group": self.experiment.get("group"),
            "pipeline_scope": "v1_simulator_first",
            "threshold_priority": threshold_priority,
            "matching_keys": matching_keys,
            "component_threshold_mode": self.calibration_cfg.get(
                "component_threshold_mode",
                "variance_scaled",
            ),
            "threshold_quantile": float(
                self.calibration_cfg.get("threshold_quantile", 0.05)
            ),
            "component_threshold_beta": float(
                self.calibration_cfg.get("component_threshold_beta", 0.5)
            ),
            "fallbacks": {
                "fixed_component_grad_threshold": float(
                    self.calibration_cfg.get("fixed_component_grad_threshold", 1e-6)
                ),
                "fixed_normalized_grad_norm_threshold": float(
                    self.calibration_cfg.get("fixed_normalized_grad_norm_threshold", 1e-6)
                ),
                "variance_epsilon": float(
                    self.calibration_cfg.get("variance_epsilon", 1e-30)
                ),
            },
            "runtime_diagnosis_rule": {
                "suspected_barren_plateau": (
                    "offline_bp_risk_high AND runtime_gradient_collapse AND task_unsolved"
                ),
                "converged": (
                    "runtime_gradient_collapse AND task_solved"
                ),
                "suspected_local_minimum": (
                    "task_unsolved AND NOT runtime_gradient_collapse"
                ),
            },
            "artifact_paths": {
                "bp_risk_table_path": self.outputs["bp_risk_table_path"],
                "threshold_calibration_table_path": self.outputs[
                    "threshold_calibration_table_path"
                ],
                "calibration_report_path": self.outputs["calibration_report_path"],
            },
            "risk_counts": risk_counts,
            "source_report": report_df.to_dict(orient="records"),
            "limitations": [
                "Stage 1C v1 uses fixed-batch Stage 1B calibration if Stage 1B is enabled.",
                "Stage 1C v1 does not decompose data-induced gradient variability across multiple batches.",
                "Thresholds are empirical scheduler aids, not formal proofs of barren plateau.",
                "Runtime diagnosis must combine offline risk, runtime gradient collapse, and task state.",
            ],
        }

    def _save_outputs(
        self,
        threshold_table: pd.DataFrame,
        risk_table: pd.DataFrame,
        report_df: pd.DataFrame,
        calibration_rules: dict[str, Any],
    ) -> None:
        """
        Save all Stage 1C artifacts.

        Args:
            threshold_table:
                Threshold calibration table.
            risk_table:
                BP risk table.
            report_df:
                Calibration report.
            calibration_rules:
                JSON calibration rules.
        """
        save_table(
            threshold_table,
            self.outputs["threshold_calibration_table_path"],
        )
        save_table(
            risk_table,
            self.outputs["bp_risk_table_path"],
        )
        save_table(
            report_df,
            self.outputs["calibration_report_path"],
        )
        write_json(
            calibration_rules,
            self.outputs["calibration_rules_path"],
        )