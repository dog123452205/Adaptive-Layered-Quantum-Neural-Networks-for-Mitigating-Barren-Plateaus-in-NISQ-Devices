"""
Stage 3 — Mutation Engine (Execution Engine).

Nhận MutationRequest từ Scheduler (Stage 2) và thực thi VÒNG ĐỜI nhiều bước,
đúng cơ chế tài liệu V1:

  propose_prune_layer / propose_reduce_entanglement:
      PROPOSED -> MASKING (cosine anneal alpha 1->0 trong T epoch) -> kiểm acceptance
               -> COMMITTED (loại lớp / chốt giảm entangler) | ROLLED_BACK (nạp lại registry)

  add_layer  : CÙNG vòng đời an toàn như trên, nhưng alpha ramp NGƯỢC (0->1): lớp mới
               thêm vào ở mask=0 (chưa đóng góp gì), tăng dần lên 1 trong T epoch. Hết
               cửa sổ mới kiểm acceptance -> COMMITTED (giữ lớp) | ROLLED_BACK (gỡ lớp,
               nạp lại registry). Trước bản sửa này add_layer commit ngay lập tức, không
               có lưới an toàn -> một lần add làm hại (BP nặng hơn) không thể hoàn tác.
  stop_growth: khóa tăng trưởng.
  keep       : không làm gì.

Soft-mask: alpha(t) = 1/2 (1 + cos(pi * t / T)), t = 0..T  -> alpha đi từ 1 về 0.
Tham số lớp mục tiêu suy giảm: theta_eff = alpha(t) * theta (do backend áp).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

from src.synthetic_bp.stage3.circuit_backend import CircuitBackend
from src.synthetic_bp.stage3.parameter_registry import ParameterRegistry

try:
    from src.synthetic_bp.constants import MutationStatus, SchedulerAction
    M_NONE = MutationStatus.NONE.value
    M_MASKING = MutationStatus.IN_PROGRESS.value
    M_COMMIT = MutationStatus.COMMITTED.value
    M_ROLLBACK = MutationStatus.ROLLED_BACK.value
    A_ADD = SchedulerAction.ADD_LAYER.value
    A_STOP = SchedulerAction.STOP_GROWTH.value
    A_PRUNE = SchedulerAction.PROPOSE_PRUNE_LAYER.value
    A_REDUCE = SchedulerAction.PROPOSE_REDUCE_ENTANGLEMENT.value
    A_KEEP = SchedulerAction.KEEP.value
except Exception:
    M_NONE, M_MASKING, M_COMMIT, M_ROLLBACK = "none", "in_progress", "committed", "rolled_back"
    A_ADD, A_STOP, A_PRUNE = "add_layer", "stop_growth", "propose_prune_layer"
    A_REDUCE, A_KEEP = "propose_reduce_entanglement", "keep"


def cosine_alpha(t: int, T: int) -> float:
    """alpha(t) = 1/2 (1 + cos(pi t / T)); t=0 -> 1.0, t=T -> 0.0."""
    if T <= 0:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * min(t, T) / T))


@dataclass
class _ActiveMutation:
    kind: str            # A_PRUNE | A_REDUCE | A_ADD
    target_layer: Optional[int]
    t: int               # epoch đã trôi qua trong quá trình mask
    T: int               # tổng epoch mask
    start_val_loss: float
    peak_val_loss: float  # loss cao nhất quan sát trong cửa sổ (cho acceptance)


class MutationEngine:
    def __init__(self, backend: CircuitBackend, registry: ParameterRegistry,
                 mask_epochs_T: int = 5, max_relative_loss_increase: float = 0.02,
                 growth_locked: bool = False):
        self.backend = backend
        self.registry = registry
        self.T = mask_epochs_T
        self.max_rel_loss = max_relative_loss_increase
        self.growth_locked = growth_locked
        self.active: Optional[_ActiveMutation] = None
        self.last_status = M_NONE
        self.mutation_count = 0

    @property
    def busy(self) -> bool:
        return self.active is not None

    # ---- nhận quyết định từ scheduler ----
    def submit(self, action: str, target_layer: Optional[int],
               current_val_loss: float) -> str:
        """Bắt đầu một mutation. Trả status tức thời."""
        if self.busy:
            return M_MASKING  # đang bận, bỏ qua request mới
        if action == A_KEEP:
            self.last_status = M_NONE
            return M_NONE
        if action == A_STOP:
            self.growth_locked = True
            self.last_status = M_NONE
            return M_NONE
        if action == A_ADD:
            if self.growth_locked:
                self.last_status = M_NONE
                return M_NONE
            # chốt checkpoint TRƯỚC khi add (mốc an toàn để rollback nếu lớp mới làm hại)
            self.registry.commit_checkpoint(current_val_loss)
            depth_before = self.backend.depth
            self.backend.add_layer()
            if self.backend.depth == depth_before:
                # đã ở max_depth -> add_layer no-op, không có gì để theo dõi
                self.last_status = M_NONE
                return M_NONE
            new_layer_id = self.backend.depth - 1
            self.backend.set_layer_mask(new_layer_id, 0.0)  # ramp 0->1, chưa đóng góp ngay
            self.active = _ActiveMutation(kind=A_ADD, target_layer=new_layer_id,
                                          t=0, T=self.T, start_val_loss=current_val_loss,
                                          peak_val_loss=current_val_loss)
            self.last_status = M_MASKING
            return M_MASKING
        if action in (A_PRUNE, A_REDUCE):
            # chốt checkpoint TRƯỚC khi mask (để rollback được)
            self.registry.commit_checkpoint(current_val_loss)
            self.active = _ActiveMutation(kind=action, target_layer=target_layer,
                                          t=0, T=self.T, start_val_loss=current_val_loss,
                                          peak_val_loss=current_val_loss)
            self.last_status = M_MASKING
            return M_MASKING
        self.last_status = M_NONE
        return M_NONE

    # ---- gọi mỗi epoch để tiến triển mask ----
    def on_epoch(self, current_val_loss: float) -> str:
        """Tiến một bước soft-mask; khi xong thì commit/rollback. Trả status."""
        if not self.busy:
            return M_NONE
        m = self.active
        m.t += 1
        # theo dõi loss ĐỈNH trong cửa sổ: damage giữa cửa sổ có thể được model
        # thích nghi bù lại vào cuối, nên chỉ nhìn loss cuối sẽ bỏ sót (bug §IX-F).
        m.peak_val_loss = max(m.peak_val_loss, current_val_loss)
        alpha = cosine_alpha(m.t, m.T)

        if m.kind == A_PRUNE:
            self.backend.set_layer_mask(m.target_layer, alpha)
        elif m.kind == A_REDUCE:
            self.backend.set_entangler_strength(alpha)
        else:  # A_ADD: ramp NGƯỢC — lớp mới đi từ mask=0 lên 1
            self.backend.set_layer_mask(m.target_layer, 1.0 - alpha)

        if m.t < m.T:
            return M_MASKING

        # ---- hết T epoch: kiểm acceptance bằng PEAK loss trong cửa sổ ----
        # so đỉnh loss (thời điểm rủi ro nhất), không phải loss cuối cửa sổ.
        rel = ((m.peak_val_loss - m.start_val_loss) / m.start_val_loss
               if m.start_val_loss > 0 else 0.0)
        accept = rel <= self.max_rel_loss

        if accept:
            if m.kind == A_PRUNE:
                self.backend.remove_layer(m.target_layer)
            # với REDUCE: đã ở alpha thấp, giữ nguyên (đã giảm)
            # với ADD: mask đã ramp tới 1.0, lớp mới giữ nguyên full-strength
            self.registry.commit_checkpoint(current_val_loss)
            self.registry.mark_commit()
            self.mutation_count += 1
            status = M_COMMIT
        else:
            self.registry.rollback()        # nạp lại param + cấu trúc cũ (gỡ lớp mới nếu ADD)
            status = M_ROLLBACK

        self.active = None
        self.last_status = status
        return status