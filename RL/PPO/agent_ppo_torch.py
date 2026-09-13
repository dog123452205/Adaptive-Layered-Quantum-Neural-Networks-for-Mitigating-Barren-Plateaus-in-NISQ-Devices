"""
agent_ppo_torch.py — PPO (PyTorch) cho AL-QNN Scheduler.
Categorical policy + action masking + GAE + clipped surrogate + entropy + value clipping.
Drop-in với env.py.

Yêu cầu: pip install torch
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from backend import N_ACTIONS
from env import STATE_DIM

NEG_INF = -1e9


class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        h = self.body(x)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, NEG_INF)
        value = self.v(h).squeeze(-1)
        return logits, value


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.s, self.a, self.r, self.done = [], [], [], []
        self.logp, self.v, self.mask = [], [], []

    def add(self, s, a, r, done, logp, v, mask):
        self.s.append(s); self.a.append(a); self.r.append(r); self.done.append(done)
        self.logp.append(logp); self.v.append(v); self.mask.append(mask)

    def __len__(self):
        return len(self.s)


class PPOAgent:
    def __init__(self, lr=3e-4, gamma=0.95, lam=0.95, clip=0.2, ent_coef=0.01,
                 vf_coef=0.5, epochs=10, minibatch=64, hidden=128, seed=0, device=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(hidden=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.epochs, self.minibatch = epochs, minibatch
        self.buf = RolloutBuffer()

    @torch.no_grad()
    def act(self, state, mask, greedy=False):
        """greedy=False (mặc định, dùng lúc TRAIN): sample từ Categorical để explore.
        greedy=True (dùng lúc DEPLOY): argmax logits — quyết định của policy đã học
        phải ổn định, không nên còn ngẫu nhiên khi đã đưa vào scheduler thật."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device).unsqueeze(0)
        logits, value = self.ac(s, m)
        dist = Categorical(logits=logits)
        a = logits.argmax(dim=-1) if greedy else dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def store(self, s, a, r, done, logp, v, mask):
        self.buf.add(s, a, r, done, logp, v, np.asarray(mask))

    def _gae(self, last_v):
        r = np.array(self.buf.r, dtype=np.float32)
        done = np.array(self.buf.done, dtype=np.float32)
        v = np.array(self.buf.v, dtype=np.float32)
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            next_v = last_v if t == len(r) - 1 else v[t + 1]
            delta = r[t] + self.gamma * next_v * (1 - done[t]) - v[t]
            gae = delta + self.gamma * self.lam * (1 - done[t]) * gae
            adv[t] = gae
        ret = adv + v
        return adv, ret

    def update(self, last_v=0.0):
        if len(self.buf) == 0:
            return None
        dev = self.device
        S = torch.as_tensor(np.array(self.buf.s, dtype=np.float32), device=dev)
        A = torch.as_tensor(np.array(self.buf.a), dtype=torch.long, device=dev)
        OLD = torch.as_tensor(np.array(self.buf.logp, dtype=np.float32), device=dev)
        M = torch.as_tensor(np.array(self.buf.mask), dtype=torch.bool, device=dev)
        adv_np, ret_np = self._gae(last_v)
        ADV = torch.as_tensor(adv_np, device=dev)
        RET = torch.as_tensor(ret_np, device=dev)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        n = len(self.buf)
        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch):
                b = idx[start:start + self.minibatch]
                bi = torch.as_tensor(b, dtype=torch.long, device=dev)
                logits, value = self.ac(S[bi], M[bi])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(A[bi])
                ratio = torch.exp(logp - OLD[bi])
                surr1 = ratio * ADV[bi]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV[bi]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = (value - RET[bi]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()

        info = dict(pi_loss=float(pi_loss.item()), v_loss=float(v_loss.item()),
                    entropy=float(ent.item()))
        self.buf.clear()
        return info
