"""
prepare_pima.py — Pima Indians Diabetes -> 8 feature cho 8-qubit (KHÔNG cần PCA).

Điểm mạnh: Pima có ĐÚNG 8 feature y khoa -> map 1-to-1 vào 8 qubit bằng angle
encoding, giữ nguyên 100% thông tin, không nén PCA.

Chạy TRÊN MÁY BẠN (có mạng), một lần:
    python prepare_pima.py
Kết quả: pima.csv (8 feature f0..f7 + Target), đặt vào data/ của project.
Yêu cầu: pip install pandas scikit-learn numpy
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

OUT = "pima.csv"; RANDOM_STATE = 42
# 8 feature gốc: pregnancies, glucose, bp, skin, insulin, bmi, dpf, age
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLS = ["pregnancies","glucose","bp","skin","insulin","bmi","dpf","age","label"]

print("Tai Pima Indians Diabetes...")
try:
    df = pd.read_csv(URL, header=None, names=COLS)
    print(f"  tai xong: {len(df)} mau, 8 feature")
except Exception as ex:
    print(f"LOI tai: {ex}\nTai tay: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    raise SystemExit(1)

X = df[COLS[:-1]].to_numpy(float)   # đúng 8 feature -> KHÔNG SelectKBest, KHÔNG PCA
y = df["label"].to_numpy(int)

# cân bằng 50/50 (data gốc ~65/35)
idx0, idx1 = np.where(y==0)[0], np.where(y==1)[0]
n = min(len(idx0), len(idx1)); rng = np.random.default_rng(RANDOM_STATE)
sel = np.concatenate([rng.choice(idx0,n,replace=False), rng.choice(idx1,n,replace=False)])
rng.shuffle(sel); X, y = X[sel], y[sel]
print(f"Can bang: {len(y)} mau ({n} moi lop)")

Xs = StandardScaler().fit_transform(X)
out = pd.DataFrame(Xs, columns=[f"f{i}" for i in range(8)]); out["Target"] = y
out.to_csv(OUT, index=False)
print(f"LUU -> {OUT} ({len(out)} mau, 8 feature 1-to-1 + Target)")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sep = cross_val_score(LogisticRegression(max_iter=500), Xs, y, cv=5).mean()
print(f"Do tach (LogReg CV): {sep:.3f}  ({'VUA PHAI - tot' if 0.6<sep<0.9 else 'qua de' if sep>=0.9 else 'qua kho'})")
print("\nTIEP: them nhanh 'pima' vao scripts/dataset.py (xem dataset_branches.txt), roi chay run_new_datasets.ps1")
