"""
Stage 3 — Parameter Registry.

Bộ nhớ đệm cổ điển lưu trạng thái model + optimizer trước mỗi mutation, để:
  - Commit : chốt checkpoint mới khi mutation thành công.
  - Rollback: nạp lại checkpoint cũ khi mutation làm sụt hiệu năng.

Tách khỏi backend để logic an toàn (registry/rollback) độc lập với loại mạch.
"""
from __future__ import annotations
from typing import Any
from src.synthetic_bp.stage3.circuit_backend import CircuitBackend


class ParameterRegistry:
    def __init__(self, backend: CircuitBackend):
        self.backend = backend
        self._checkpoint: dict[str, Any] | None = None
        self._checkpoint_val_loss: float | None = None
        self.n_commits = 0
        self.n_rollbacks = 0

    def commit_checkpoint(self, val_loss: float) -> None:
        """Chốt trạng thái hiện tại làm mốc an toàn."""
        self._checkpoint = self.backend.snapshot()
        self._checkpoint_val_loss = val_loss

    @property
    def checkpoint_val_loss(self) -> float | None:
        return self._checkpoint_val_loss

    def rollback(self) -> None:
        """Khôi phục backend về checkpoint gần nhất."""
        if self._checkpoint is None:
            return
        self.backend.restore(self._checkpoint)
        self.n_rollbacks += 1

    def mark_commit(self) -> None:
        self.n_commits += 1
