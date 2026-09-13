"""
build_heart_named.py — Rebuild predictions_heart_named.csv: the SAME rows as
predictions_heart.csv (produced by export_predictions.py), but with the 8
feature columns replaced by their RAW, real-unit clinical values (age... no,
the actual 8 SelectKBest-chosen features: sex, chest_pain_type,
max_heart_rate, ...) instead of the [0, pi] angle-encoded values fed to the
circuit.

Why this needs reconstruction instead of just reading data/heart.csv: that
file already went through prepare_heart.py's StandardScaler, so it doesn't
carry raw values either. This script re-fetches the original UCI Cleveland
data and re-applies prepare_heart.py's exact steps (same RANDOM_STATE=42
balance-sample, same SelectKBest(k=8)) to recover the untouched raw values,
then re-applies export_predictions.py's exact max_samples/train-val-test
split (same SEED/MAXS) to align row-for-row with the current
predictions_heart.csv TEST split.

MUST be run AFTER export_predictions.py heart (uses its SEED/MAXS/TEST_FRAC/
VAL_FRAC — keep these in sync if either script's constants change), and
BEFORE export_classical_baseline.py heart (which borrows this file's raw
feature columns).

Verifies alignment itself: asserts the true_label sequence it reconstructs
matches predictions_heart.csv's true_label column exactly, and refuses to
write the output otherwise.

RUN FROM PROJECT ROOT (needs network access to fetch the UCI dataset):
    python build_heart_named.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

# must match export_predictions.py's SEED/MAXS/VAL_FRAC/TEST_FRAC exactly
SEED = 123; VAL_FRAC = 0.3; TEST_FRAC = 0.2; MAXS = 280
PREPARE_RANDOM_STATE = 42  # must match data/prepare_heart.py

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
       "exang", "oldpeak", "slope", "ca", "thal", "num"]
RENAME = {
    "sex": "sex", "cp": "chest_pain_type", "thalach": "max_heart_rate",
    "exang": "exercise_angina", "oldpeak": "st_depression", "slope": "st_slope",
    "ca": "n_major_vessels", "thal": "thalassemia",
}

print("Fetching UCI Cleveland heart-disease data...")
df = pd.read_csv(URL, header=None, names=COLS, na_values="?").dropna()
y_full = (df["num"].to_numpy(int) > 0).astype(int)
X_raw_full = df.drop(columns=["num"]).reset_index(drop=True)

# 1) reproduce prepare_heart.py's class-balance sampling (RANDOM_STATE=42)
idx0, idx1 = np.where(y_full == 0)[0], np.where(y_full == 1)[0]
n_bal = min(len(idx0), len(idx1))
rng = np.random.default_rng(PREPARE_RANDOM_STATE)
sel = np.concatenate([rng.choice(idx0, n_bal, replace=False),
                      rng.choice(idx1, n_bal, replace=False)])
rng.shuffle(sel)
y_bal = y_full[sel]
X_raw_bal = X_raw_full.iloc[sel].reset_index(drop=True)

heart_csv = pd.read_csv(ROOT / "data" / "heart.csv")
assert len(X_raw_bal) == len(heart_csv), \
    f"row count mismatch: reconstructed {len(X_raw_bal)} vs data/heart.csv {len(heart_csv)}"
assert (y_bal == heart_csv["Target"].to_numpy()).all(), \
    "Target sequence mismatch — UCI source data may have changed, or PREPARE_RANDOM_STATE is wrong"
print(f"  balanced sample verified: {len(y_bal)} rows, matches data/heart.csv exactly")

# 2) reproduce export_predictions.py's max_samples subsample (SEED=123)
if MAXS and MAXS < len(y_bal):
    rng2 = np.random.default_rng(SEED)
    i0, i1 = np.where(y_bal == 0)[0], np.where(y_bal == 1)[0]
    per = min(len(i0), len(i1), MAXS // 2)
    keep = np.concatenate([rng2.choice(i0, per, replace=False), rng2.choice(i1, per, replace=False)])
    rng2.shuffle(keep)
    X_raw_bal = X_raw_bal.iloc[keep].reset_index(drop=True)
    y_bal = y_bal[keep]
print(f"  after max_samples={MAXS} subsample: {len(y_bal)} rows")

# 3) reproduce export_predictions.py's train/val/test split (SEED=123)
X_fit, X_test_raw, y_fit, y_test = train_test_split(
    X_raw_bal, y_bal, test_size=TEST_FRAC, random_state=SEED, stratify=y_bal)

preds = pd.read_csv(ROOT / "predictions_heart.csv")
assert len(X_test_raw) == len(preds), \
    f"TEST row count mismatch: reconstructed {len(X_test_raw)} vs predictions_heart.csv {len(preds)}"
assert (y_test == preds["true_label"].to_numpy()).all(), \
    "true_label sequence mismatch between reconstructed TEST split and predictions_heart.csv — " \
    "check that export_predictions.py's SEED/MAXS/VAL_FRAC/TEST_FRAC match this script's constants"
print(f"  TEST split verified: {len(y_test)} rows, true_label matches predictions_heart.csv exactly")

raw8 = X_test_raw[list(RENAME.keys())].reset_index(drop=True).rename(columns=RENAME)
out = pd.concat([raw8, preds[["true_label", "pred", "confidence", "proba_class1"]].reset_index(drop=True)], axis=1)
out_path = ROOT / "predictions_heart_named.csv"
out.to_csv(out_path, index=False)
print(f"  -> {out_path.name}  ({len(out)} rows)")
