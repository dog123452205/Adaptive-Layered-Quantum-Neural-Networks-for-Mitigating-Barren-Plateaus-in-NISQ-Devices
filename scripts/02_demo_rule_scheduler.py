"""
Demo Stage 2 — chạy RuleBasedScheduler trên chuỗi runtime metric mô phỏng
+ calibration_rules.json thật (Stage 1C). Dùng để verify scheduler phát action hợp lý.

Chạy:
    python scripts/02_demo_rule_scheduler.py \
        --calibration outputs/stage1c/calibration_rules.json
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.synthetic_bp.stage2.scheduler_base import RuntimeState
from src.synthetic_bp.stage2.rule_based_scheduler import (
    RuleBasedScheduler, RuleBasedConfig,
)


def fake_layers(depth, nzr_weak=0.7):
    """Tạo metric theo lớp: lớp cuối là 'yếu' để thử kịch bản prune."""
    layers = []
    for i in range(depth):
        weak = (i == depth - 1)
        layers.append({
            "layer_id": i,
            "layer_status": "active",
            "layer_grad_norm": 0.05 if weak else 0.3,
            "layer_grad_variance": 1e-5 if weak else 1e-2,
            "layer_near_zero_ratio": nzr_weak if weak else 0.25,
            "mask_value": 1.0,
        })
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    args = ap.parse_args()

    sched = RuleBasedScheduler(
        calibration=args.calibration,
        cfg=RuleBasedConfig(target_depth=5),
        warmup_epochs=10, action_interval_epochs=5, cooldown_epochs=5,
    )
    sched.reset()

    print(f"{'epoch':>5} {'act?':>5} {'action':>26} {'risk':>6}  reason")
    print("-" * 78)

    last_mut = -100
    depth = 1          # bắt đầu ở cấu hình rủi ro HIGH (global cost, depth=1)
    n_ent = 8          # nhiều cổng vướng víu để thử nhánh reduce
    for epoch in range(0, 41):
        # mô phỏng: BP nặng dần (near_zero tăng), val_loss giảm chậm
        nzr_global = min(0.45 + 0.01 * epoch, 0.75)
        state = RuntimeState(
            epoch=epoch,
            n_qubits=4, entanglement_topology="circular", cost_type="global",
            depth=depth, n_entangling_gates=n_ent,
            validation_loss=max(0.69 - 0.01 * epoch, 0.2),
            training_loss=max(0.69 - 0.01 * epoch, 0.2),
            grad_norm=max(0.4 - 0.01 * epoch, 0.05),
            spatial_grad_variance=max(1e-2 - 2e-4 * epoch, 1e-4),
            near_zero_ratio=nzr_global,
            layers=fake_layers(depth, nzr_weak=nzr_global + 0.05),
            last_mutation_epoch=last_mut,
        )
        if not sched.should_act(epoch, last_mut):
            print(f"{epoch:>5} {'-':>5} {'(warmup/interval/cooldown)':>26}")
            continue
        req = sched.decide(state)
        print(f"{epoch:>5} {'YES':>5} {req.action:>26} {req.risk_level:>6}  {req.reason}")
        # áp dụng thô để demo thay đổi cấu hình
        if req.action == "add_layer":
            depth += 1; n_ent += 1
        elif req.action == "propose_prune_layer":
            depth = max(1, depth - 1)
        elif req.action == "propose_reduce_entanglement":
            n_ent = max(0, n_ent - 1)
        last_mut = epoch


if __name__ == "__main__":
    main()
