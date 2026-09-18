"""
build_ai4i_named.py — Rebuild predictions_predictive_maintenance_named.csv: the
SAME rows as predictions_predictive_maintenance.csv (produced by
export_predictions.py), but with the 8 feature columns replaced by their RAW
values (Kelvin, rpm, Nm, minutes, and the 3 one-hot Type_H/L/M columns)
instead of the [0, pi] angle-encoded values fed to the circuit.

Why this needs reconstruction instead of just reading
data/predictive_maintenance.csv: dataset.py's load_binary_dataset ALWAYS
MinMaxScales every feature to [0, pi] for angle encoding (even when no PCA is
applied), so predictions_predictive_maintenance.csv's f0..f7 columns are
already encoded, not raw. This script re-derives the raw 8-column matrix (5
numeric sensor readings + 3 one-hot Type columns — exactly 8, so at
n_qubits=8 no PCA/padding touches it) the same way dataset.py builds it, then
re-applies the SAME 2-step train/val/test split dataset.py uses (full 10,000
rows, no max_samples cap — AI4I now uses the Cyclic Partitioned Undersampling
path, see BalancedBatchSampler in scripts/dataset.py) to align row-for-row
with the current predictions_predictive_maintenance.csv TEST split.

MUST be run AFTER export_predictions.py predictive_maintenance (uses its
SEED/VAL_FRAC/TEST_FRAC — keep in sync if either script's constants change),
and BEFORE export_classical_baseline.py predictive_maintenance (which borrows
this file's raw feature columns).

Verifies alignment itself: asserts the true_label sequence it reconstructs
matches predictions_predictive_maintenance.csv's true_label column exactly,
and refuses to write the output otherwise.

RUN FROM misc/ (same folder as the other export_*.py scripts):
    python build_ai4i_named.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# must match export_predictions.py's SEED/VAL_FRAC/TEST_FRAC exactly
SEED = 123
VAL_FRAC = 0.3
TEST_FRAC = 0.2

df = pd.read_csv(PROJECT_ROOT / "data" / "predictive_maintenance.csv")
num_cols = ["Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
Xnum = df[num_cols].to_numpy(float)
type_oh = pd.get_dummies(df["Type"])  # columns: H, L, M (alphabetical) — must match dataset.py
X_raw = np.hstack([Xnum, type_oh.to_numpy(float)])
y = df["Target"].to_numpy(int)
raw_cols = ["air_temperature_K", "process_temperature_K", "rotational_speed_rpm",
            "torque_Nm", "tool_wear_min"] + [f"type_{c}" for c in type_oh.columns]
X_raw_df = pd.DataFrame(X_raw, columns=raw_cols)
print(f"  raw AI4I data: {len(y)} rows, columns={raw_cols}")

# reproduce dataset.py's 2-step split exactly (no max_samples cap — full data):
#   1) carve out (val_frac+test_frac) combined from the full set
#   2) split that combined chunk into val vs test
val_test_frac = VAL_FRAC + TEST_FRAC
X_tr, X_temp, y_tr, y_temp = train_test_split(
    X_raw_df, y, test_size=val_test_frac, random_state=SEED, stratify=y)
rel_test_frac = TEST_FRAC / val_test_frac
X_va, X_test_raw, y_va, y_test = train_test_split(
    X_temp, y_temp, test_size=rel_test_frac, random_state=SEED, stratify=y_temp)
X_test_raw = X_test_raw.reset_index(drop=True)

preds = pd.read_csv(HERE / "predictions_predictive_maintenance.csv")
assert len(X_test_raw) == len(preds), \
    f"TEST row count mismatch: reconstructed {len(X_test_raw)} vs predictions_predictive_maintenance.csv {len(preds)}"
assert (y_test == preds["true_label"].to_numpy()).all(), \
    "true_label sequence mismatch — check SEED/VAL_FRAC/TEST_FRAC match dataset.py's split"
print(f"  TEST split verified: {len(y_test)} rows, true_label matches predictions_predictive_maintenance.csv exactly")

out = pd.concat(
    [X_test_raw.round(3), preds[["true_label", "pred", "confidence", "proba_class1"]].reset_index(drop=True)],
    axis=1)
out_path = HERE / "predictions_predictive_maintenance_named.csv"
out.to_csv(out_path, index=False)
print(f"  -> {out_path.name}  ({len(out)} rows)")
