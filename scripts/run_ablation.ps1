# ============================================================
#  run_ablation.ps1  (v2 - splatting, khong dung backtick)
#
#  Chay 4 nhanh mot mach (cam may qua dem):
#     PPO baseline | PPO telemetry | DQN baseline | DQN telemetry
#  Moi nhanh 3 seed (0,1,2). Local n=12.
#
#  Chay:  powershell -ExecutionPolicy Bypass -File .\scripts\run_ablation.ps1
#
#  Splatting: tham so trong mang, PowerShell tu noi -> khong gay dong nhu backtick.
# ============================================================
param(
  [int]$Nqubits      = 12,
  [string]$Cost      = "local",
  [int]$MaxDepth     = 12,
  [string]$Surrogate = "outputs\surrogate\q12_local_d16\surrogate_fit.json",
  [string]$Seeds     = "0,1,2",
  [int]$DeployEpochs = 40,
  [int]$Episodes     = 600
)
$ErrorActionPreference = "Stop"
$log = "run_ablation_q${Nqubits}_${Cost}.log"
function Log($m){ $l="[{0}] {1}" -f (Get-Date -Format HH:mm:ss),$m; Write-Host $l -ForegroundColor Cyan; Add-Content $log $l }

if(!(Test-Path $Surrogate)){ Log "LOI: khong thay surrogate: $Surrogate -> DUNG."; exit 1 }

$common = @(
  "--episodes", $Episodes,
  "--backend", "simulate",
  "--deploy-backend", "pennylane",
  "--dataset", "predictive_maintenance",
  "--max-samples", "200",
  "--n-qubits", $Nqubits,
  "--cost-type", $Cost,
  "--initial-depth", "2",
  "--max-depth", $MaxDepth,
  "--surrogate", $Surrogate,
  "--deploy-epochs", $DeployEpochs,
  "--seeds", $Seeds
)

$runs = @(
  @{agent="ppo"; path="RL\PPO"; tele=$false; dir="ppo_q${Nqubits}_${Cost}_baseline"},
  @{agent="ppo"; path="RL\PPO"; tele=$true;  dir="ppo_q${Nqubits}_${Cost}_telemetry"},
  @{agent="dqn"; path="RL\DQN"; tele=$false; dir="dqn_q${Nqubits}_${Cost}_baseline"},
  @{agent="dqn"; path="RL\DQN"; tele=$true;  dir="dqn_q${Nqubits}_${Cost}_telemetry"}
)

Log "===== ABLATION: 4 nhanh (PPO/DQN x baseline/telemetry), q$Nqubits $Cost ====="
$t0 = Get-Date
$i = 0
foreach($r in $runs){
  $i++
  $outdir = "outputs\rl_results\simulate\$($r.dir)"
  $teleFlag = if($r.tele){ "telemetry ON" } else { "baseline" }
  Log "[$i/4] $($r.agent) $teleFlag -> $outdir"

  $pyargs = @("scripts\04_train_rl_scheduler.py", "--rl-path", $r.path, "--agent", $r.agent) + $common + @("--results-dir", $outdir)
  if($r.tele){ $pyargs += "--use-telemetry" }

  & python @pyargs 2>&1 | Add-Content $log
  if($LASTEXITCODE -ne 0){
    Log "  LOI: nhanh nay that bai (exit $LASTEXITCODE). Xem $log. TIEP TUC nhanh sau."
  } else {
    if(Test-Path "$outdir\aggregate.json"){ Log "  OK -> $outdir\aggregate.json" }
    else { Log "  CANH BAO: khong thay aggregate.json du exit=0" }
  }
}

$dt = [int]((Get-Date) - $t0).TotalMinutes
Log "===== XONG 4 nhanh trong $dt phut ====="
Log "So cac cap: PPO baseline vs telemetry | DQN baseline vs telemetry"
Log "Nhin: depth_end std, acc_end, f1_end, overshoot. Gui Claude 4 aggregate.json."