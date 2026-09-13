"""
Stage 1C — Calibration Artifact Builder.

Input : artifact Stage 1A (data-free) + Stage 1B (task-aware).
Output: calibration_rules.json
        bp_risk_table.parquet            (phân loại rủi ro theo cấu hình)
        threshold_calibration_table.parquet  (ngưỡng tau theo từng cấu hình + mode)

Ngưỡng thực nghiệm:  tau_empirical = beta * sqrt(N_theta * Var)
  - mode empirical_stage1a  : Var lấy từ 1A (cấu trúc thuần)
  - mode empirical_stage1b  : Var lấy từ 1B (data-induced)
  - mode empirical_combined : Var = min(Var_1A, Var_1B)  (thận trọng nhất)

Phân loại DiagnosisState -> risk LOW/MEDIUM/HIGH, dùng cho Rule-based & RL scheduler.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import numpy as np
import pandas as pd

try:
    from src.synthetic_bp.constants import (
        DiagnosisState, GradientThresholdMode, ThresholdStatistic)
    DS_HEALTHY = DiagnosisState.HEALTHY.value
    DS_BP = DiagnosisState.SUSPECTED_BARREN_PLATEAU.value
    DS_DEAD = DiagnosisState.LAYERWISE_DEAD_ZONE.value
    DS_AMBIG = DiagnosisState.AMBIGUOUS_GRADIENT_COLLAPSE.value
    STAT_MEDIAN = ThresholdStatistic.MEDIAN.value
    STAT_MEAN = ThresholdStatistic.MEAN.value
except Exception:
    DS_HEALTHY, DS_BP = "healthy", "suspected_barren_plateau"
    DS_DEAD, DS_AMBIG = "layerwise_dead_zone", "ambiguous_gradient_collapse"
    STAT_MEDIAN, STAT_MEAN = "median", "mean"

MODE_1A, MODE_1B, MODE_COMB = "empirical_stage1a", "empirical_stage1b", "empirical_combined"


@dataclass
class CalibrationConfig:
    beta: float = 1.0
    statistic: str = STAT_MEDIAN
    near_zero_grad_threshold: float = 1e-6
    nzr_dead: float = 0.60
    nzr_bp: float = 0.40
    log10_var_flat: float = -4.0


def _agg(s, stat):
    return float(s.mean() if stat == STAT_MEAN else s.median())


def _diagnose(nzr, log10_var, cfg):
    flat = log10_var <= cfg.log10_var_flat
    if nzr >= cfg.nzr_dead:
        return DS_DEAD
    if nzr >= cfg.nzr_bp and flat:
        return DS_BP
    if flat and nzr < cfg.nzr_bp:
        return DS_AMBIG
    return DS_HEALTHY


def _risk(diag):
    return {DS_HEALTHY: "LOW", DS_AMBIG: "MEDIUM", DS_BP: "HIGH", DS_DEAD: "HIGH"}.get(diag, "MEDIUM")


GROUP = ["n_qubits", "entanglement_topology", "cost_type", "depth"]


def _prep(df):
    """Chuẩn hóa: đảm bảo có các cột cần; trả None nếu df rỗng/thiếu."""
    if df is None or len(df) == 0:
        return None
    need = set(GROUP) | {"spatial_grad_variance", "grad_norm",
                         "near_zero_ratio", "n_params"}
    if not need.issubset(df.columns):
        return None
    return df


def build_calibration(runs_1a: pd.DataFrame | None,
                      runs_1b: pd.DataFrame | None,
                      cfg: CalibrationConfig | None = None):
    """Trả (calibration_dict, bp_risk_df, threshold_df)."""
    cfg = cfg or CalibrationConfig()
    a, b = _prep(runs_1a), _prep(runs_1b)
    if a is None and b is None:
        raise ValueError("Cần ít nhất một trong Stage 1A / 1B artifact.")

    # union các cấu hình xuất hiện ở 1A hoặc 1B
    keys = set()
    for df in (a, b):
        if df is not None:
            keys |= set(map(tuple, df[GROUP].drop_duplicates().to_numpy()))

    def lookup(df, key):
        if df is None:
            return None
        mask = np.ones(len(df), dtype=bool)
        for col, val in zip(GROUP, key):
            mask &= (df[col] == val)
        sub = df[mask]
        return sub if len(sub) else None

    rules, risk_rows, thr_rows = [], [], []
    for key in sorted(keys):
        nq, topo, cost, depth = key
        sa, sb = lookup(a, key), lookup(b, key)

        def stats(sub):
            if sub is None:
                return None
            return dict(
                var=_agg(sub["spatial_grad_variance"], cfg.statistic),
                gnorm=_agg(sub["grad_norm"], cfg.statistic),
                nzr=_agg(sub["near_zero_ratio"], cfg.statistic),
                nparams=int(_agg(sub["n_params"], STAT_MEAN)))

        st_a, st_b = stats(sa), stats(sb)
        # Var cho từng mode
        var_modes = {}
        if st_a: var_modes[MODE_1A] = st_a["var"]
        if st_b: var_modes[MODE_1B] = st_b["var"]
        if st_a and st_b: var_modes[MODE_COMB] = min(st_a["var"], st_b["var"])

        # n_params & nzr ưu tiên 1B (task-aware) nếu có, else 1A
        ref = st_b or st_a
        n_theta = ref["nparams"]; nzr = ref["nzr"]; gnorm = ref["gnorm"]

        # tau theo từng mode
        taus = {m: cfg.beta * math.sqrt(max(n_theta, 1) * max(v, 0.0))
                for m, v in var_modes.items()}
        primary_mode = MODE_COMB if MODE_COMB in taus else (MODE_1B if MODE_1B in taus else MODE_1A)
        tau = taus[primary_mode]
        var_primary = var_modes[primary_mode]
        log10_var = math.log10(max(var_primary, 1e-30))
        diag = _diagnose(nzr, log10_var, cfg)
        risk = _risk(diag)

        keyd = {"n_qubits": int(nq), "entanglement_topology": str(topo),
                "cost_type": str(cost), "depth": int(depth)}
        rules.append({
            "key": keyd,
            "thresholds": {"tau_empirical": tau,
                           "tau_by_mode": taus,
                           "near_zero_grad_threshold": cfg.near_zero_grad_threshold},
            "stats": {"spatial_grad_variance": var_primary,
                      "log10_spatial_grad_variance": log10_var,
                      "grad_norm": gnorm, "near_zero_ratio": nzr,
                      "n_params": n_theta, "primary_mode": primary_mode,
                      "has_1a": st_a is not None, "has_1b": st_b is not None},
            "diagnosis": diag, "risk_level": risk})
        risk_rows.append({**keyd, "diagnosis": diag, "risk_level": risk,
                          "near_zero_ratio": nzr,
                          "log10_spatial_grad_variance": log10_var,
                          "primary_mode": primary_mode})
        for m, tv in taus.items():
            thr_rows.append({**keyd, "threshold_mode": m,
                             "spatial_grad_variance": var_modes[m],
                             "tau_empirical": tv, "beta": cfg.beta,
                             "n_params": n_theta})

    all_tau = [r["thresholds"]["tau_empirical"] for r in rules]
    calibration = {
        "schema_version": "1c.v2",
        "source_stages": [s for s, df in [("stage1a", a), ("stage1b", b)] if df is not None],
        "calibration_config": asdict(cfg),
        "risk_legend": {
            "LOW": "Trainability lành mạnh — không cần can thiệp.",
            "MEDIUM": "Tín hiệu mơ hồ — theo dõi, có thể giảm vướng víu.",
            "HIGH": "Nghi ngờ BP / vùng chết — ưu tiên cắt lớp hoặc giảm entanglement."},
        "global_default": {
            "tau_empirical": float(np.median(all_tau)) if all_tau else 0.0,
            "near_zero_grad_threshold": cfg.near_zero_grad_threshold},
        "n_rules": len(rules), "rules": rules,
    }
    return calibration, pd.DataFrame(risk_rows), pd.DataFrame(thr_rows)


def lookup_rule(cal, n_qubits, topology, cost_type, depth):
    for r in cal.get("rules", []):
        k = r["key"]
        if (k["n_qubits"] == n_qubits and k["entanglement_topology"] == topology
                and k["cost_type"] == cost_type and k["depth"] == depth):
            return r
    return {"key": {"n_qubits": n_qubits, "entanglement_topology": topology,
                    "cost_type": cost_type, "depth": depth},
            "thresholds": cal["global_default"], "diagnosis": DS_HEALTHY,
            "risk_level": "LOW", "fallback": True}
