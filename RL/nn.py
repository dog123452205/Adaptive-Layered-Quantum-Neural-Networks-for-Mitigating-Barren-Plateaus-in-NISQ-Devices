"""
nn.py — MLP nhỏ bằng numpy thuần (forward + backward), dùng chung cho DQN & PPO.
Không phụ thuộc torch để chạy ngay. Cấu trúc đủ rõ để port sang torch ở P4.
"""
from __future__ import annotations
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0, out_activation="linear"):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            self.W.append(rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / fan_in))
            self.b.append(np.zeros(sizes[i + 1]))
        self.out_activation = out_activation
        self._cache = None

    def forward(self, x):
        x = np.atleast_2d(x).astype(np.float64)
        acts = [x]
        pre = []
        h = x
        for i in range(len(self.W) - 1):
            z = h @ self.W[i] + self.b[i]
            pre.append(z)
            h = np.tanh(z)
            acts.append(h)
        z = h @ self.W[-1] + self.b[-1]
        pre.append(z)
        if self.out_activation == "softmax":
            z = z - z.max(axis=1, keepdims=True)
            out = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        else:
            out = z
        acts.append(out)
        self._cache = (acts, pre)
        return out

    def backward(self, grad_out, lr, clip=5.0):
        """grad_out: dL/d(output pre-activation hoặc output). SGD step in-place."""
        acts, pre = self._cache
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        delta = grad_out  # với linear out: dL/dz_out ; với softmax+CE: (p - y)
        for i in reversed(range(len(self.W))):
            h_prev = acts[i]
            grads_W[i] = h_prev.T @ delta / delta.shape[0]
            grads_b[i] = delta.mean(axis=0)
            if i > 0:
                dh = delta @ self.W[i].T
                delta = dh * (1 - np.tanh(pre[i - 1]) ** 2)
        for i in range(len(self.W)):
            gW = np.clip(grads_W[i], -clip, clip)
            gb = np.clip(grads_b[i], -clip, clip)
            self.W[i] -= lr * gW
            self.b[i] -= lr * gb

    def copy_from(self, other, tau=1.0):
        for i in range(len(self.W)):
            self.W[i] = tau * other.W[i] + (1 - tau) * self.W[i]
            self.b[i] = tau * other.b[i] + (1 - tau) * self.b[i]

    def params(self):
        return [w.copy() for w in self.W] + [b.copy() for b in self.b]
