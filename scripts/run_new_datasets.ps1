# ══════════════════════════════════════════════════════════════════════
# run_new_datasets.ps1 — chạy full pipeline cho Pima / Heart / Parkinson's
# Đặt file này trong thư mục gốc project (cạnh scripts\, data\).
# Điều kiện: đã chạy prepare_*.py, đã đặt *.csv vào data\, đã thêm nhánh vào dataset.py
# Chạy:  .\run_new_datasets.ps1
# Mỗi dataset ~ vài chục phút đến vài giờ tùy máy (deploy PennyLane chậm).
# ══════════════════════════════════════════════════════════════════════

$datasets = @("heart","parkinsons")   # bỏ bớt nếu chỉ chạy 1-2 cái
$NQ = 8
$COST = "local"
$DEPLOY_EPOCHS = 40
$SEEDS = "0,1,2"

foreach ($ds in $datasets) {
    Write-Host "`n==================== $ds ====================" -ForegroundColor Cyan

    # 1) Calibrate surrogate
    Write-Host "[1/4] Calibrate surrogate..." -ForegroundColor Yellow
    python scripts\06_calibrate_surrogate.py --dataset $ds --n-qubits $NQ --cost-type $COST --out outputs\surrogate\${ds}_q8_local

    $SUR = "outputs\surrogate\${ds}_q8_local\surrogate_fit.json"

    # 2) Train PPO baseline (3 seed)
    Write-Host "[2/4] Train PPO..." -ForegroundColor Yellow
    python scripts\04_train_rl_scheduler.py --rl-path RL\PPO --agent ppo --episodes 800 --backend simulate --deploy-backend pennylane --dataset $ds --max-samples 200 --n-qubits $NQ --cost-type $COST --initial-depth 2 --max-depth 8 --surrogate $SUR --deploy-epochs $DEPLOY_EPOCHS --seeds $SEEDS --results-dir outputs\rl_results\simulate\ppo_${ds}_local_baseline

    # 3) Train DQN baseline (3 seed)
    Write-Host "[3/4] Train DQN..." -ForegroundColor Yellow
    python scripts\04_train_rl_scheduler.py --rl-path RL\DQN --agent dqn --episodes 1000 --backend simulate --deploy-backend pennylane --dataset $ds --max-samples 200 --n-qubits $NQ --cost-type $COST --initial-depth 2 --max-depth 8 --surrogate $SUR --deploy-epochs $DEPLOY_EPOCHS --seeds $SEEDS --results-dir outputs\rl_results\simulate\dqn_${ds}_local_baseline

    # 4) So sánh 4 scheduler trên PennyLane (seed 0 làm đại diện)
    Write-Host "[4/4] Compare 4 schedulers..." -ForegroundColor Yellow
    python scripts\05_compare_schedulers.py --backend pennylane --dataset $ds --max-samples 200 --n-qubits $NQ --cost-type $COST --epochs $DEPLOY_EPOCHS --ppo-ckpt outputs\rl_results\simulate\ppo_${ds}_local_baseline\seed_0\agent.pt --dqn-ckpt outputs\rl_results\simulate\dqn_${ds}_local_baseline\seed_0\agent.pt --out outputs\comparison\${ds}_q8_local

    Write-Host "==================== $ds XONG ====================" -ForegroundColor Green
}

Write-Host "`nTAT CA XONG. Chay: python summarize_datasets.py  de tong hop output." -ForegroundColor Cyan
