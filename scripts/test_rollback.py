"""
Unit test cho cơ chế commit/rollback (Stage 3).

Chứng minh safety envelope hoạt động ĐÚNG mà không phụ thuộc agent có tình cờ
tạo mutation xấu hay không — dựng thẳng các tình huống và kiểm quyết định.

Chạy:  python scripts/test_rollback.py
Không cần PennyLane/torch. Chạy trong 1 giây.

Kiểm 4 tình huống:
  1. Prune lớp thừa (loss phẳng)              -> COMMIT
  2. Prune lớp quan trọng (loss vọt giữa cửa sổ, hồi cuối) -> ROLLBACK
     (đây là ca bug cũ BỎ SÓT: chỉ nhìn loss cuối sẽ accept nhầm)
  3. Prune làm hỏng hẳn (loss cao suốt)       -> ROLLBACK
  4. Prune tốt (loss giảm)                    -> COMMIT
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


def cosine_alpha(t: int, T: int) -> float:
    """Mask coefficient: 1.0 -> 0.0 theo cosine qua T bước."""
    return 0.5 * (1 + math.cos(math.pi * min(t, T) / T)) if T > 0 else 0.0


@dataclass
class _ActiveMutation:
    kind: str
    target_layer: Optional[int]
    t: int
    T: int
    start_val_loss: float
    peak_val_loss: float          # FIX: theo dõi đỉnh loss trong cửa sổ


class FakeBackend:
    """Backend giả: ghi lại lớp bị gỡ và số lần khôi phục."""
    def __init__(self):
        self.removed = []
        self.restored = 0

    def remove_layer(self, layer):
        self.removed.append(layer)

    def restore(self, ckpt):
        self.restored += 1


class ParameterRegistry:
    """Registry rút gọn: snapshot / commit / rollback."""
    def __init__(self, backend):
        self.backend = backend
        self.n_commits = 0
        self.n_rollbacks = 0
        self._checkpoint = None

    def commit_checkpoint(self, vloss):
        self._checkpoint = {"vloss": vloss}

    def mark_commit(self):
        self.n_commits += 1

    def rollback(self):
        self.backend.restore(self._checkpoint)
        self.n_rollbacks += 1


class MutationEngine:
    """Engine rút gọn với FIX peak-tracking (khớp mutation_engine_fixed.py)."""
    def __init__(self, backend, registry, T=5, max_rel_loss=0.05):
        self.backend = backend
        self.registry = registry
        self.T = T
        self.max_rel_loss = max_rel_loss
        self.active = None

    @property
    def busy(self):
        return self.active is not None

    def submit(self, action, current_val_loss, target_layer=0):
        self.registry.commit_checkpoint(current_val_loss)
        self.active = _ActiveMutation(action, target_layer, 0, self.T,
                                      current_val_loss, current_val_loss)
        return "in_progress"

    def on_epoch(self, current_val_loss):
        if not self.busy:
            return "none"
        m = self.active
        m.t += 1
        # theo dõi loss ĐỈNH: damage giữa cửa sổ có thể được bù vào cuối (bug §IX-F)
        m.peak_val_loss = max(m.peak_val_loss, current_val_loss)
        if m.t < m.T:
            return "in_progress"
        # hết cửa sổ: kiểm acceptance bằng PEAK, không phải loss cuối
        rel = ((m.peak_val_loss - m.start_val_loss) / m.start_val_loss
               if m.start_val_loss > 0 else 0.0)
        if rel <= self.max_rel_loss:
            self.backend.remove_layer(m.target_layer)
            self.registry.mark_commit()
            status = "committed"
        else:
            self.registry.rollback()
            status = "rolled_back"
        self.active = None
        return status


def run_case(name, window, expected, T=5, max_rel=0.05):
    b = FakeBackend()
    r = ParameterRegistry(b)
    e = MutationEngine(b, r, T=T, max_rel_loss=max_rel)
    e.submit("prune", window[0])
    status = None
    for loss in window[1:]:
        status = e.on_epoch(loss)
    ok = status == expected
    # cách bug CŨ (loss cuối) sẽ quyết định gì, để đối chiếu
    rel_old = (window[-1] - window[0]) / window[0]
    old = "committed" if rel_old <= max_rel else "rolled_back"
    peak = max(window)
    rel_new = (peak - window[0]) / window[0]
    print(f"  [{('PASS' if ok else 'FAIL')}] {name}")
    print(f"        window peak={peak:.3f} (rel {rel_new:+.2f}) -> FIX: {status}")
    print(f"        window end ={window[-1]:.3f} (rel {rel_old:+.2f}) -> OLD BUG: {old}"
          + ("   <-- bug would MISS this" if old != expected else ""))
    return ok


def main():
    print("=" * 62)
    print("UNIT TEST: commit / rollback safety envelope (peak-loss fix)")
    print("=" * 62)
    cases = [
        ("redundant layer (flat loss) -> COMMIT",
         [0.50, 0.502, 0.503, 0.501, 0.499, 0.500], "committed"),
        ("important layer (spike mid, recover end) -> ROLLBACK",
         [0.50, 0.62, 0.66, 0.58, 0.52, 0.505], "rolled_back"),
        ("severe damage (high throughout) -> ROLLBACK",
         [0.50, 0.55, 0.60, 0.62, 0.61, 0.60], "rolled_back"),
        ("good prune (loss drops) -> COMMIT",
         [0.50, 0.49, 0.48, 0.485, 0.48, 0.478], "committed"),
    ]
    results = [run_case(n, w, e, T=len(w) - 1) for n, w, e in cases]
    print("=" * 62)
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} passed")
    if passed == len(results):
        print("\nCo che rollback hoat dong dung:")
        print("  - COMMIT khi mutation an toan (loss trong nguong)")
        print("  - ROLLBACK khi mutation lam hong loss")
        print("  - Ca 'spike mid, recover end' bi bug cu BO SOT, nay bat duoc")
    else:
        print("\nCON LOI - can kiem tra lai logic")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
