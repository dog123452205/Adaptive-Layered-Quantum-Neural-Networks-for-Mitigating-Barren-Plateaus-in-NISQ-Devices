"""
Stage 3 — Data loader (MNIST 0/1, PCA -> Angle encoding).

PCA nén 28x28 -> n_qubits chiều, scale về [0, pi] cho Angle Embedding (RX/RY).
Dùng bởi PennyLaneBackend. Cần scikit-learn.

Lưu ý no-leakage: fit PCA/scaler CHỈ trên train, transform val bằng cùng bộ.
"""
from __future__ import annotations
import numpy as np


def load_mnist_pca_angle(n_qubits: int = 4, n_train: int = 200, n_val: int = 100,
                         seed: int = 0):
    """
    Trả X_train, y_train, X_val, y_val. X scale về [0, pi] (Angle Embedding).
    Nhãn nhị phân 0/1.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:n_train + n_val], y[n_train:n_train + n_val]

    # no-leakage: fit trên train, transform val
    pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr)
    Xtr_p, Xva_p = pca.transform(Xtr), pca.transform(Xva)
    scaler = MinMaxScaler((0, np.pi)).fit(Xtr_p)
    return (scaler.transform(Xtr_p), ytr.astype(float),
            scaler.transform(Xva_p), yva.astype(float))
