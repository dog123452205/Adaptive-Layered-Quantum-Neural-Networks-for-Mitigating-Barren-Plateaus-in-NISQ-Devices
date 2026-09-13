"""
agent_dqn_torch.py — Double-DQN (PyTorch) cho AL-QNN Scheduler.
Drop-in với env.py: state STATE_DIM chiều, N_ACTIONS hành động, action masking.

Yêu cầu: pip install torch
"""
from __future__ import annotations
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-contained: STATE_DIM (16 chiều state) và N_ACTIONS (5 macro-action) là
# hằng số cố định của scheduler, đồng nhất giữa Branch A (RL/PPO) và Branch B
# (Stage3RLEnv). Nếu backend.py/env.py có sẵn trên path thì lấy từ đó; nếu không
# (vd chạy --rl-path RL/DQN) thì fallback về hằng số -> không còn ModuleNotFoundError.
try:
    from backend import N_ACTIONS  # noqa: F401
    from env import STATE_DIM  # noqa: F401
except ModuleNotFoundError:
    STATE_DIM = 16
    N_ACTIONS = 5



NEG_INF = -1e9


class QNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=(128, 128)):
        super().__init__()
        layers, d = [], state_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, cap=50000):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, done, mask2):
        self.buf.append((s, a, r, s2, done, mask2))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d, m2 = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32), np.array(m2))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.95, hidden=(128, 128), seed=0,
                 eps_start=1.0, eps_end=0.05, eps_decay=4000, tau=0.01,
                 device=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(hidden=hidden).to(self.device)
        self.qt = QNet(hidden=hidden).to(self.device)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.tau = gamma, tau
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.buf = ReplayBuffer()
        self.t = 0

    def eps(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.t / self.eps_decay)

    @torch.no_grad()
    def act(self, state, mask, greedy=False):
        """greedy=False (mặc định, dùng lúc TRAIN): epsilon-greedy để explore.
        greedy=True (dùng lúc DEPLOY): argmax Q thuần, bỏ qua eps() — eps() không
        bao giờ decay về đúng 0 (eps_end=0.05 mặc định) nên nếu không có cờ này,
        policy đã train xong vẫn ngẫu nhiên ~5% mỗi quyết định lúc deploy thật."""
        if not greedy:
            self.t += 1
            if random.random() < self.eps():
                return int(np.random.choice(np.where(mask)[0]))
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(s).squeeze(0).cpu().numpy()
        q[~np.asarray(mask)] = NEG_INF
        return int(np.argmax(q))

    def update(self, batch_size=64):
        if len(self.buf) < batch_size:
            return None
        s, a, r, s2, d, m2 = self.buf.sample(batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)
        m2 = torch.as_tensor(m2, dtype=torch.bool, device=self.device)

        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double-DQN: chọn action bằng q online, đánh giá bằng target
            q_next_online = self.q(s2).clone()
            q_next_online[~m2] = NEG_INF
            a_star = q_next_online.argmax(dim=1, keepdim=True)
            q_next = self.qt(s2).gather(1, a_star).squeeze(1)
            target = r + self.gamma * (1 - d) * q_next

        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        # soft update target
        with torch.no_grad():
            for tp, p in zip(self.qt.parameters(), self.q.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)
        return float(loss.item())
