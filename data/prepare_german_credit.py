"""
prepare_german_credit.py — tải German Credit, tiền xử lý thành 8 feature cho 8-qubit.

Chạy TRÊN MÁY BẠN (có mạng), một lần:
    python prepare_german_credit.py

Kết quả:
    german_credit.csv  — 8 feature đã chuẩn hóa + nhãn nhị phân (0/1), cân bằng
    -> đặt vào thư mục data của project, rồi thêm nhánh vào dataset.py (xem cuối file).

Yêu cầu: pip install pandas scikit-learn numpy
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

OUT = "german_credit.csv"
N_FEATURES = 8   # = số qubit
RANDOM_STATE = 42

# ---- 1. TẢI DATA ----
# German Credit (Statlog) từ UCI. File 'german.data' — 20 feature + nhãn.
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# tên cột theo tài liệu UCI (A1..A20 + label)
COLS = [f"attr{i}" for i in range(1, 21)] + ["label"]

print("Tai German Credit tu UCI...")
try:
    df = pd.read_csv(URL, sep=r"\s+", header=None, names=COLS)
    print(f"  tai xong: {len(df)} mau, {len(COLS)-1} feature")
except Exception as ex:
    print(f"LOI tai: {ex}")
    print("Neu mang chan, tai tay tu:")
    print("  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data")
    print("  luu 'german.data' cung thu muc roi doi URL thanh 'german.data'")
    raise SystemExit(1)

# ---- 2. MÃ HÓA feature phân loại ----
# German Credit có feature dạng chữ (A11, A12...) — chuyển thành số.
# Ép TỪNG cột: thử chuyển số; cột nào có chữ -> mã hóa Categorical.
for c in df.columns[:-1]:
    num = pd.to_numeric(df[c], errors="coerce")
    if num.isna().any():
        df[c] = pd.Categorical(df[c].astype(str)).codes   # cột có chữ
    else:
        df[c] = num                                        # cột số

# nhãn: 1=good, 2=bad -> chuyen thanh 0/1 (1 = bad credit = rủi ro cao)
df["label"] = (df["label"] == 2).astype(int)

X = df.drop(columns=["label"]).values.astype(float)
y = df["label"].values

# ---- 3. CÂN BẰNG 50/50 (undersample lớp đa số) ----
idx0 = np.where(y == 0)[0]
idx1 = np.where(y == 1)[0]
n = min(len(idx0), len(idx1))
rng = np.random.default_rng(RANDOM_STATE)
sel = np.concatenate([rng.choice(idx0, n, replace=False),
                      rng.choice(idx1, n, replace=False)])
rng.shuffle(sel)
X, y = X[sel], y[sel]
print(f"Can bang: {len(y)} mau ({n} moi lop)")

# ---- 4. CHỌN 8 FEATURE MẠNH NHẤT (SelectKBest, giống BCW) ----
selector = SelectKBest(f_classif, k=N_FEATURES)
X_sel = selector.fit_transform(X, y)
kept = selector.get_support(indices=True)
print(f"Chon {N_FEATURES} feature manh nhat: cot {list(kept)}")

# ---- 5. CHUẨN HÓA ----
X_scaled = StandardScaler().fit_transform(X_sel)

# ---- 6. LƯU CSV ----
out = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(N_FEATURES)])
out["Target"] = y
out.to_csv(OUT, index=False)
print(f"\nLUU -> {OUT}  ({len(out)} mau, {N_FEATURES} feature + Target)")

# kiem tra do tach (separability nhanh)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sep = cross_val_score(LogisticRegression(max_iter=500), X_scaled, y, cv=5).mean()
print(f"Do tach (LogReg CV accuracy): {sep:.3f}")
if sep > 0.9:
    print("  -> qua de (giong BCW), scheduler kho phan biet")
elif sep < 0.6:
    print("  -> qua kho, co the accuracy thap")
else:
    print("  -> VUA PHAI, tot cho so sanh scheduler (giong predictive_maintenance ~0.72)")

print("""
════════════════════════════════════════════════════════════
BUOC TIEP: them nhanh vao dataset.py
════════════════════════════════════════════════════════════
Trong ham load_binary_dataset(name, n_qubits, ...), them:

    elif name == "german_credit":
        df = pd.read_csv("data/german_credit.csv")   # sua duong dan cho dung
        X = df[[f"f{i}" for i in range(n_qubits)]].values   # 8 feature -> 8 qubit
        y = df["Target"].values
        # (angle encoding + chia train/val giong cac dataset khac)

Roi chay pipeline nhu predictive_maintenance, doi --dataset german_credit --n-qubits 8:

  # 1. surrogate
  python scripts\\06_calibrate_surrogate.py --dataset german_credit --n-qubits 8 --cost-type local ...
  # 2. train agent
  python scripts\\04_train_rl_scheduler.py --dataset german_credit --n-qubits 8 --cost-type local ...
  # 3. compare 4 scheduler
  python scripts\\05_compare_schedulers.py --dataset german_credit --n-qubits 8 --backend pennylane ...
════════════════════════════════════════════════════════════
""")