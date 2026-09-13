"""
Stage 2 — Fixed-Depth Scheduler (baseline không mutate, dùng để đo depth=k cố định).

Không tra rule, không tra gradient — không bao giờ phát ADD/PRUNE/REDUCE/STOP,
chỉ KEEP. Dùng để so depth mọc thích nghi (rule/greedy/ppo/dqn) với các mốc
depth k cố định (chạy trainer nhiều lần với initial_depth=max_depth=k).
"""
from __future__ import annotations

from src.synthetic_bp.stage2.scheduler_base import (
    SchedulerBase, RuntimeState, MutationRequest, A_KEEP, DS_HEALTHY,
)


class FixedScheduler(SchedulerBase):
    def should_act(self, epoch: int, last_mutation_epoch: int = -10_000) -> bool:
        return False

    def decide(self, state: RuntimeState) -> MutationRequest:
        return MutationRequest(action=A_KEEP, reason="fixed depth: no-op",
                               diagnosis=DS_HEALTHY, risk_level="LOW")
