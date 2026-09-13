# ĐẶC TẢ NGẮN — STAGE 2: RULE-BASED TRAINABILITY SCHEDULER

> Bộ điều phối kiến trúc theo luật tĩnh cho AL-QNN V1 (paper 1). Thiết kế interface
> chung `SchedulerBase` để RL scheduler (V2) sau này cắm vào cùng vị trí mà không
> sửa Stage 3.

## 1. Vai trò
Tại mỗi nhịp huấn luyện, Stage 2 nhận chỉ số runtime (gradient norm, validation loss,
metric theo lớp), tra ngưỡng từ `calibration_rules.json` (Stage 1C), và phát một
**Mutation Request** cho Stage 3 thực thi. Scheduler KHÔNG tự sửa tham số — chỉ ra quyết định.

## 2. Không gian quyết định (macro, 5)
`keep | add_layer | stop_growth | propose_prune_layer | propose_reduce_entanglement`

Vòng đời nhiều bước (`soft_mask → commit/rollback`) do **Stage 3** đảm nhiệm theo config
template (`prune.mask`, `reduce_entanglement.mask`, `acceptance`, `rollback`). Scheduler
chỉ phát quyết định mức cao — giữ ranh giới Stage 2/Stage 3 rõ ràng.

## 3. Nhịp ra quyết định (khớp config template)
- `warmup_epochs` (mặc định 10): chưa can thiệp trong giai đoạn khởi động.
- `action_interval_epochs` (mặc định 5): chỉ quyết định mỗi 5 epoch.
- `cooldown_epochs` (mặc định 5): sau một mutation, chờ trước khi mutation tiếp.

## 4. Luật quyết định (ưu tiên từ trên xuống)
1. **Rủi ro HIGH + có lớp chết** (near_zero_ratio lớp ≥ ngưỡng) & còn cắt được → `propose_prune_layer` (lớp yếu nhất).
2. **BP toàn cục nặng** (grad_norm < τ_empirical & near_zero toàn cục cao) + nhiều cổng vướng víu → `propose_reduce_entanglement`.
3. **Trainability ổn** (grad_norm ≥ τ) & mạch còn nông & loss không xấu đi → `add_layer`.
4. **Đã đạt độ sâu mục tiêu** & ổn định → `stop_growth`.
5. Còn lại → `keep`.

Ngưỡng `τ_empirical` và `risk_level` lấy từ Stage 1C (`lookup` theo n_qubits, topology,
cost_type, depth). Nếu cấu hình ngoài bảng → dùng `global_default`.

## 5. Hợp đồng dữ liệu (Stage 3 phải cung cấp mỗi tick)
`RuntimeState`: epoch; cấu hình mạch (n_qubits, topology, cost_type, depth,
n_entangling_gates); chỉ số toàn cục (validation_loss, training_loss, grad_norm,
spatial_grad_variance, near_zero_ratio); danh sách lớp (layer_id, layer_status,
layer_grad_norm, layer_grad_variance, layer_near_zero_ratio, mask_value).

`MutationRequest` trả về: action, target_layer, reason, diagnosis, risk_level.

## 6. Cấu trúc file
| File | Nội dung |
|---|---|
| `stage2/scheduler_base.py` | `SchedulerBase` (interface), `RuntimeState`, `MutationRequest` |
| `stage2/rule_based_scheduler.py` | `RuleBasedScheduler` + `RuleBasedConfig` |
| `scripts/02_demo_rule_scheduler.py` | demo chạy trên calibration thật |

## 7. Đường nâng cấp lên RL (V2)
RL scheduler chỉ cần kế thừa `SchedulerBase` và override `decide()` bằng policy học được
(DQN/PPO), giữ nguyên `should_act()`, `RuntimeState`, `MutationRequest`. Stage 3 không phải
biết scheduler là rule hay RL → thay "bộ não" không phá lõi.
