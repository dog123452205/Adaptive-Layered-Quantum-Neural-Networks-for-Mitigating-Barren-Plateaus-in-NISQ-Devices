"""
prepare_parkinsons.py — Parkinson's (Oxford) -> 8 feature cho 8-qubit.

Lưu ý trung thực: bộ Oxford là 22 ĐẶC TRƯNG ÂM HỌC ĐÃ TRÍCH XUẤT (jitter, shimmer,
HNR, RPDE...) — BẢNG TĨNH, KHÔNG phải raw time-series/signal. Đừng nói "xử lý
chuỗi thời gian" khi bảo vệ. Nói đúng: "phân loại từ đặc trưng âm học trích xuất".
22 feature -> nén về 8 (SelectKBest). Data mất cân bằng (147 PD / 48 khoẻ) -> undersample.

Chạy: python prepare_parkinsons.py  ->  parkinsons.csv (f0..f7 + Target)
Yêu cầu: pip install pandas scikit-learn numpy
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

OUT="parkinsons.csv"; N=8; RANDOM_STATE=42
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

print("Tai Parkinson's (Oxford)...")
try:
    df = pd.read_csv(URL)   # co header, cot 'name' + 'status' (nhan)
    print(f"  tai xong: {len(df)} mau, {df.shape[1]-2} feature")
except Exception as ex:
    print(f"LOI tai: {ex}\nTai tay: https://archive.ics.uci.edu/dataset/174/parkinsons")
    raise SystemExit(1)

y = df["status"].to_numpy(int)                       # 1=PD, 0=khoe
X = df.drop(columns=["name","status"]).to_numpy(float)  # 22 feature am hoc

# can bang 50/50 (undersample lop PD da so)
idx0, idx1 = np.where(y==0)[0], np.where(y==1)[0]
n=min(len(idx0),len(idx1)); rng=np.random.default_rng(RANDOM_STATE)
sel=np.concatenate([rng.choice(idx0,n,replace=False), rng.choice(idx1,n,replace=False)])
rng.shuffle(sel); X,y=X[sel],y[sel]
print(f"Can bang: {len(y)} mau ({n} moi lop) — LUU Y: chi {n} moi lop, tap NHO")

sk=SelectKBest(f_classif, k=N); Xk=sk.fit_transform(X,y)
print(f"Chon {N}/22 feature am hoc: cot {list(sk.get_support(indices=True))}")
Xs=StandardScaler().fit_transform(Xk)
out=pd.DataFrame(Xs, columns=[f"f{i}" for i in range(N)]); out["Target"]=y
out.to_csv(OUT,index=False); print(f"LUU -> {OUT} ({len(out)} mau, {N} feature + Target)")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sep=cross_val_score(LogisticRegression(max_iter=500), Xs, y, cv=5).mean()
print(f"Do tach (LogReg CV): {sep:.3f}  ({'VUA PHAI' if 0.6<sep<0.9 else 'qua de' if sep>=0.9 else 'qua kho'})")
print("CANH BAO: tap rat nho (~48 moi lop) -> accuracy se dao dong manh theo seed, chay >=3 seed.")
print("\nTIEP: them nhanh 'parkinsons' vao scripts/dataset.py (xem dataset_branches.txt)")
