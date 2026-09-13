"""
prepare_heart.py — Heart Disease (Cleveland) -> 8 feature cho 8-qubit.

Lưu ý trung thực: Heart có 13 feature -> CẦN chọn/nén về 8 (SelectKBest), nên KHÔNG
giữ được ưu điểm "1-to-1 không PCA" như Pima. Điểm mạnh của Heart là dữ liệu nhiều
chiều, phi tuyến -> chứng minh khả năng "xử lý nhiễu/đa chiều" của mạch thích nghi.

Chạy: python prepare_heart.py   ->  heart.csv (f0..f7 + Target)
Yêu cầu: pip install pandas scikit-learn numpy
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

OUT="heart.csv"; N=8; RANDOM_STATE=42
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLS=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]

print("Tai Heart Disease (Cleveland)...")
try:
    df = pd.read_csv(URL, header=None, names=COLS, na_values="?")
    print(f"  tai xong: {len(df)} mau, 13 feature")
except Exception as ex:
    print(f"LOI tai: {ex}\nTai tay: https://archive.ics.uci.edu/dataset/45/heart+disease")
    raise SystemExit(1)

df = df.dropna()                      # bo ~6 dong thieu gia tri
y = (df["num"].to_numpy(int) > 0).astype(int)   # 0=khong benh, 1-4 -> 1 (co benh)
X = df.drop(columns=["num"]).to_numpy(float)    # 13 feature

# can bang 50/50
idx0, idx1 = np.where(y==0)[0], np.where(y==1)[0]
n=min(len(idx0),len(idx1)); rng=np.random.default_rng(RANDOM_STATE)
sel=np.concatenate([rng.choice(idx0,n,replace=False), rng.choice(idx1,n,replace=False)])
rng.shuffle(sel); X,y=X[sel],y[sel]
print(f"Can bang: {len(y)} mau ({n} moi lop)")

# 13 -> 8 feature manh nhat
sk=SelectKBest(f_classif, k=N); Xk=sk.fit_transform(X,y)
print(f"Chon {N}/13 feature: cot {list(sk.get_support(indices=True))}")
Xs=StandardScaler().fit_transform(Xk)
out=pd.DataFrame(Xs, columns=[f"f{i}" for i in range(N)]); out["Target"]=y
out.to_csv(OUT,index=False); print(f"LUU -> {OUT} ({len(out)} mau, {N} feature + Target)")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sep=cross_val_score(LogisticRegression(max_iter=500), Xs, y, cv=5).mean()
print(f"Do tach (LogReg CV): {sep:.3f}  ({'VUA PHAI - tot' if 0.6<sep<0.9 else 'qua de' if sep>=0.9 else 'qua kho'})")
print("\nTIEP: them nhanh 'heart' vao scripts/dataset.py (xem dataset_branches.txt)")
