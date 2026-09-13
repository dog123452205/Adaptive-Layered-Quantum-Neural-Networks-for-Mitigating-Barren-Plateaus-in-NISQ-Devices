# ============================================================
#  run_all.ps1 - chay pipeline n=12 mot mach, treo may khong can canh
#
#  Thu tu: 06 (calibrate) -> 04 PPO 3 seed -> 04 DQN 3 seed -> 05 (so sanh)
#  Dung --results-dir de ep ten thu muc CO DINH -> noi duoc agent.pt sang 05.
#
#  Chay:  .\run_all.ps1
#         .\run_all.ps1 -Nqubits 12 -MaxDepth 16 -Seeds "0,1,2"
#
#  Log ghi ra run_all.log. Moi buoc kiem ket qua truoc khi sang buoc sau;
#  loi o buoc nao thi DUNG, khong chay tiep tren dau vao hong.
# ============================================================
param(
  [int]$Nqubits    = 12,
  [int]$MaxDepth   = 16,
  [string]$Cost    = "global",
  [string]$Dataset = "predictive_maintenance",
  [int]$MaxSamples = 200,
  [int]$VarSamples = 40,
  [int]$DeployEpochs = 40,
  [int]$Episodes   = 600,
  [string]$Seeds   = "0,1,2",
  [switch]$SkipCalibrate,          # bo qua 06 neu da co surrogate
  [switch]$SkipCompare             # chi train, khong chay bang so sanh
)

$ErrorActionPreference = "Stop"
$tag  = "q${Nqubits}_${Cost}_d${MaxDepth}"
$root = "outputs"
$surDir = "$root\surrogate\$tag"
$surFit = "$surDir\surrogate_fit.json"
$ppoDir = "$root\rl_results\simulate\ppo_$tag"
$dqnDir = "$root\rl_results\simulate\dqn_$tag"
$cmpDir = "$root\comparison\$tag"
$log = "run_all_$tag.log"

function Log($m){
  $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
  Write-Host $line -ForegroundColor Cyan
  Add-Content $log $line
}
function Need($path, $what){
  if(!(Test-Path $path)){ Log "LOI: khong thay $what tai $path -> DUNG."; exit 1 }
}
function FirstSeedDir($parent){
  $seed0 = ($Seeds -split ",")[0].Trim()
  return "$parent\seed_$seed0\agent.pt"
}

Log "===== BAT DAU pipeline $tag ====="
$t0 = Get-Date

# ---------- Buoc 1: calibrate surrogate (06) ----------
if($SkipCalibrate -and (Test-Path $surFit)){
  Log "Bo qua calibrate, dung surrogate co san: $surFit"
} else {
  Log "Buoc 1/4: calibrate surrogate (n=$Nqubits, depths toi $MaxDepth, var-samples=$VarSamples)"
  $depths = (1..$MaxDepth | Where-Object { $_ -eq 1 -or $_ % 2 -eq 0 }) -join ","  # 1,2,4,6,...
  python scripts\06_calibrate_surrogate.py --n-qubits $Nqubits --cost-type $Cost `
    --dataset $Dataset --max-samples $MaxSamples --depths $depths `
    --var-samples $VarSamples --floor-source train --out $surDir 2>&1 | Tee-Object -Append $log
  Need $surFit "surrogate_fit.json"
  Log "  -> surrogate xong. KIEM b va rmse.logvar truoc khi tin (xem $surFit)."
}

# ---------- Buoc 2: train PPO 3 seed (04) ----------
Log "Buoc 2/4: train PPO $Seeds"
python scripts\04_train_rl_scheduler.py --rl-path RL\PPO --agent ppo --episodes $Episodes `
  --backend simulate --use-telemetry --deploy-backend pennylane --dataset $Dataset --max-samples $MaxSamples `
  --n-qubits $Nqubits --cost-type $Cost --initial-depth 2 --max-depth $MaxDepth `
  --surrogate $surFit --deploy-epochs $DeployEpochs --seeds $Seeds `
  --results-dir $ppoDir 2>&1 | Tee-Object -Append $log
Need "$ppoDir\aggregate.json" "PPO aggregate.json"
$ppoCkpt = FirstSeedDir $ppoDir
Need $ppoCkpt "PPO agent.pt"

# ---------- Buoc 3: train DQN 3 seed (04) ----------
Log "Buoc 3/4: train DQN $Seeds"
python scripts\04_train_rl_scheduler.py --rl-path RL\DQN --agent dqn --episodes $Episodes `
  --backend simulate --use-telemetry --deploy-backend pennylane --dataset $Dataset --max-samples $MaxSamples `
  --n-qubits $Nqubits --cost-type $Cost --initial-depth 2 --max-depth $MaxDepth `
  --surrogate $surFit --deploy-epochs $DeployEpochs --seeds $Seeds `
  --results-dir $dqnDir 2>&1 | Tee-Object -Append $log
Need "$dqnDir\aggregate.json" "DQN aggregate.json"
$dqnCkpt = FirstSeedDir $dqnDir
Need $dqnCkpt "DQN agent.pt"

# ---------- Buoc 4: bang so sanh 4 scheduler (05) ----------
if($SkipCompare){
  Log "Bo qua buoc so sanh (theo yeu cau)."
} else {
  Log "Buoc 4/4: so sanh 4 scheduler"
  # baseline: ghi nhan de dem, KHONG veto
  python scripts\05_compare_schedulers.py --backend pennylane --dataset $Dataset `
    --max-samples $MaxSamples --n-qubits $Nqubits --max-depth $MaxDepth --cost-type $Cost `
    --epochs $DeployEpochs --ppo-ckpt $ppoCkpt --dqn-ckpt $dqnCkpt `
    --out "${cmpDir}_base" --log-telemetry 2>&1 | Tee-Object -Append $log
  Need "${cmpDir}_base\comparison.csv" "comparison.csv (base)"

  # co guardrail: veto
  python scripts\05_compare_schedulers.py --backend pennylane --dataset $Dataset `
    --max-samples $MaxSamples --n-qubits $Nqubits --max-depth $MaxDepth --cost-type $Cost `
    --epochs $DeployEpochs --ppo-ckpt $ppoCkpt --dqn-ckpt $dqnCkpt `
    --out "${cmpDir}_te" --log-telemetry --telemetry-guard 2>&1 | Tee-Object -Append $log
  Need "${cmpDir}_te\comparison.csv" "comparison.csv (te)"
}

$dt = [int]((Get-Date) - $t0).TotalMinutes
Log "===== XONG $tag trong $dt phut ====="
Log "Ket qua:"
Log "  surrogate : $surFit"
Log "  PPO       : $ppoDir\aggregate.json"
Log "  DQN       : $dqnDir\aggregate.json"
if(!$SkipCompare){ Log "  so sanh   : $cmpDir\comparison.csv" }
Log "Gui cho Claude: 2 aggregate.json + comparison.csv + surrogate_fit.json"
