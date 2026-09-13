"""
Train PPO (hoặc DQN) làm RL Scheduler, chọn được backend Simulate / PennyLane thật,
VÀ LƯU KẾT QUẢ TÁCH RIÊNG THEO BACKEND.

Kết quả lưu vào:  outputs/rl_results/<backend>/<agent>_<timestamp>/
  -> dễ phân biệt số từ SimulateBackend vs PennyLaneBackend.

Cờ mới:
  --backend         simulate|pennylane : backend dùng để TRAIN agent (env).  [mặc định simulate]
  --deploy-backend  simulate|pennylane|same : backend cho phần DEPLOY/đánh giá. [mặc định same]
  --dataset         iris|mnist : dataset cho PennyLaneBackend (bỏ qua nếu simulate). [mặc định iris]
  --max-samples     giới hạn số mẫu (mnist rất chậm với PennyLane).

Ví dụ:
  # train nhanh trên simulate, đánh giá thật trên pennylane (KHUYẾN NGHỊ):
  python scripts/04_train_rl_scheduler.py --rl-path ...\\RL\\PPO --agent ppo ^
      --episodes 600 --cost-type global --topology circular ^
      --backend simulate --deploy-backend pennylane --dataset iris

  # train THẲNG trên pennylane thật (rất chậm — giảm episodes/total-epochs):
  python scripts/04_train_rl_scheduler.py --rl-path ...\\RL\\PPO --agent ppo ^
      --episodes 30 --total-epochs 20 --backend pennylane --dataset iris
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))  # để import dataset.py cạnh script

import numpy as np
import pandas as pd
from src.synthetic_bp.stage3.rl_env import Stage3RLEnv


def load_data_if_needed(args, needed: bool):
    """Chỉ nạp dataset khi có backend pennylane."""
    if not needed:
        return None
    from dataset import load_binary_dataset
    d = load_binary_dataset(args.dataset, n_qubits=args.n_qubits,
                            max_samples=args.max_samples, seed=42)
    print(f"  [data] {d['name']}: train={d['n_train']} val={d['n_val']} "
          f"n_qubits={d['n_qubits']}")
    return d


def make_env(args, seed, backend, data, calibration=None):
    return Stage3RLEnv(n_qubits=args.n_qubits, initial_depth=args.initial_depth,
                       max_depth=args.max_depth, cost_type=args.cost_type,
                       topology=args.topology, total_epochs=args.total_epochs,
                       seed=seed, backend=backend, data=data,
                       entangler_type=args.entangler, calibration=calibration,
                       surrogate=getattr(args, "surrogate", None),
                       use_telemetry=not getattr(args, "no_telemetry", False),
                       mask_epochs_T=args.mask_epochs_t,
                       max_relative_loss_increase=args.max_rel_loss)


def build_backend(kind, args, data, seed):
    """Dựng một backend rời (cho phần deploy)."""
    from src.synthetic_bp.stage3.circuit_backend import SimulateBackend
    if kind == "pennylane":
        from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend
        return PennyLaneBackend(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_qubits=args.n_qubits, initial_depth=args.initial_depth,
            max_depth=args.max_depth, topology=args.topology,
            entangler_type=args.entangler, cost_type=args.cost_type, seed=seed)
    return SimulateBackend(n_qubits=args.n_qubits, initial_depth=args.initial_depth,
                           max_depth=args.max_depth, cost_type=args.cost_type,
                           topology=args.topology, seed=seed)


def _log_every(args):
    """Cadence in log: nhỏ episode -> in dày hơn (để run PennyLane ngắn vẫn thấy tiến độ)."""
    return max(1, args.episodes // 20)


def _progress(tag, ep, args, rets, t_ep, t_start, log_every):
    if (ep + 1) % log_every == 0 or ep == args.episodes - 1:
        dt = time.time() - t_ep
        avg = (time.time() - t_start) / (ep + 1)
        eta = avg * (args.episodes - ep - 1)
        print(f"  [{tag}] ep {ep+1}/{args.episodes} "
              f"| return(last100)={np.mean(rets[-100:]):.3f} "
              f"| {dt:.1f}s/ep | avg {avg:.1f}s | ETA ~{eta/60:.1f} phút")


def _maybe_ckpt(agent, out_dir, ep, args):
    """Lưu agent.pt định kỳ (ghi đè -> luôn là bản mới nhất). Crash/Ctrl-C không mất công."""
    if args.ckpt_every and (ep + 1) % args.ckpt_every == 0:
        if save_agent(agent, out_dir):
            print(f"    [ckpt] agent.pt @ ep {ep+1}")


def train_ppo(args, PPOAgent, data, out_dir, seed):
    env = make_env(args, seed, args.backend, data, args.calibration)
    agent = PPOAgent(seed=seed)
    steps, rets = 0, []
    log_every = _log_every(args)
    t_start = time.time()
    for ep in range(args.episodes):
        t_ep = time.time()
        s, mask = env.reset()
        done, R = False, 0.0
        while not done:
            a, logp, v = agent.act(s, mask)
            s2, r, done, mask2, info = env.step(a)
            agent.store(s, a, r, float(done), logp, v, mask)
            s, mask = s2, mask2
            R += r; steps += 1
            if steps % 512 == 0:
                last_v = 0.0 if done else agent.act(s, mask)[2]
                agent.update(last_v=last_v)
        rets.append(R)
        _progress("PPO", ep, args, rets, t_ep, t_start, log_every)
        _maybe_ckpt(agent, out_dir, ep, args)
    agent.update()
    return agent, rets


def train_dqn(args, DQNAgent, data, out_dir, seed):
    env = make_env(args, seed, args.backend, data, args.calibration)
    agent = DQNAgent(seed=seed)
    rets = []
    log_every = _log_every(args)
    t_start = time.time()
    for ep in range(args.episodes):
        t_ep = time.time()
        s, mask = env.reset()
        done, R = False, 0.0
        while not done:
            a = agent.act(s, mask)
            s2, r, done, mask2, info = env.step(a)
            agent.buf.push(s, a, r, s2, float(done), np.asarray(mask2))
            agent.update(batch_size=64)
            s, mask = s2, mask2
            R += r
        rets.append(R)
        _progress("DQN", ep, args, rets, t_ep, t_start, log_every)
        _maybe_ckpt(agent, out_dir, ep, args)
    return agent, rets


def save_learning_curve(rets, out_dir):
    df = pd.DataFrame({"episode": range(1, len(rets) + 1), "return": rets})
    df["return_ma100"] = df["return"].rolling(100, min_periods=1).mean()
    df.to_csv(out_dir / "learning_curve.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(df["episode"], df["return"], alpha=0.3, label="return")
        plt.plot(df["episode"], df["return_ma100"], label="moving avg (100)")
        plt.xlabel("Episode"); plt.ylabel("Return"); plt.legend()
        plt.title("RL Scheduler — Learning Curve"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(out_dir / "learning_curve.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"  (bỏ qua vẽ biểu đồ: {e})")


def save_agent(agent, out_dir):
    try:
        import torch
        net = getattr(agent, "ac", None) or getattr(agent, "q", None)
        if net is not None:
            torch.save(net.state_dict(), out_dir / "agent.pt")
            return True
    except Exception as e:
        print(f"  (bỏ qua lưu model: {e})")
    return False


def run_once(args, seed, data, deploy_backend, needs_data, out_dir):
    from src.synthetic_bp.stage2.rl_scheduler import RLScheduler
    from src.synthetic_bp.stage3.trainer import AdaptiveTrainer

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[seed {seed}] train {args.agent.upper()} | backend={args.backend} "
          f"| deploy={deploy_backend} | {args.episodes} episode")
    if args.agent == "ppo":
        from agent_ppo_torch import PPOAgent
        agent, rets = train_ppo(args, PPOAgent, data, out_dir, seed)
    else:
        from agent_dqn_torch import DQNAgent
        agent, rets = train_dqn(args, DQNAgent, data, out_dir, seed)

    save_learning_curve(rets, out_dir)
    saved = save_agent(agent, out_dir)

    b = build_backend(deploy_backend, args, data, seed=seed)
    dep_ep = args.deploy_epochs
    warmup = min(10, max(1, dep_ep // 3))
    print(f"[seed {seed}] deploy {dep_ep} epoch trên {deploy_backend} (warmup={warmup})")
    sched = RLScheduler(agent, tau=0.1, total_epochs=dep_ep,
                        warmup_epochs=warmup, calibration=args.calibration)
    te_on = not getattr(args, "no_telemetry", False)
    tr = AdaptiveTrainer(b, sched, n_qubits=args.n_qubits, cost_type=args.cost_type,
                         topology=args.topology, epochs=dep_ep,
                         mask_epochs_T=args.mask_epochs_t,
                         max_relative_loss_increase=args.max_rel_loss,
                         log_telemetry=te_on, telemetry_guard=te_on)
    df = tr.run()
    df.to_csv(out_dir / f"deploy_log_{deploy_backend}.csv", index=False)

    summary = {
        "agent": args.agent,
        "seed": seed,
        "backend": args.backend,
        "deploy_backend": deploy_backend,
        "dataset": (args.dataset if needs_data else None),
        "episodes": args.episodes,
        "config": {"n_qubits": args.n_qubits, "cost_type": args.cost_type,
                   "topology": args.topology, "max_depth": args.max_depth},
        "training": {
            "return_first100": float(np.mean(rets[:100])),
            "return_last100": float(np.mean(rets[-100:])),
            "return_best": float(np.max(rets)),
        },
        "deploy": {
            "accuracy_start": float(df.iloc[0]["accuracy"]),
            "accuracy_end": float(df.iloc[-1]["accuracy"]),
            "f1_end": float(df.iloc[-1].get("f1", float("nan"))),
            "roc_auc_end": float(df.iloc[-1].get("roc_auc", float("nan"))),
            "depth_start": int(df.iloc[0]["depth"]),
            "depth_end": int(df.iloc[-1]["depth"]),
            "n_commits": int(df.iloc[-1]["n_commits"]),
            "n_rollbacks": int(df.iloc[-1]["n_rollbacks"]),
            "actions": df["scheduler_action"].value_counts().to_dict(),
        },
        "model_saved": saved,
        "timestamp": stamp,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    d = summary["deploy"]
    print(f"[seed {seed}] acc {d['accuracy_start']:.3f} -> {d['accuracy_end']:.3f} "
          f"| depth {d['depth_start']}->{d['depth_end']} "
          f"| commits {d['n_commits']} rollbacks {d['n_rollbacks']}")
    return summary


def aggregate(summaries, out_dir):
    rows = []
    for s in summaries:
        rows.append({
            "seed": s["seed"],
            "return_last100": s["training"]["return_last100"],
            "return_best": s["training"]["return_best"],
            "acc_start": s["deploy"]["accuracy_start"],
            "acc_end": s["deploy"]["accuracy_end"],
            "f1_end": s["deploy"].get("f1_end", float("nan")),
            "depth_end": s["deploy"]["depth_end"],
            "n_commits": s["deploy"]["n_commits"],
            "n_rollbacks": s["deploy"]["n_rollbacks"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "seeds.csv", index=False)

    metrics = [c for c in df.columns if c != "seed"]
    stats = {m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=0))}
             for m in metrics}
    (out_dir / "aggregate.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[multiseed] {len(summaries)} seed, mean±std:")
    for m in metrics:
        print(f"  {m:15s} {stats[m]['mean']:.4f} ± {stats[m]['std']:.4f}")
    return df, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-path", required=True)
    ap.add_argument("--agent", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--initial-depth", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=16)
    ap.add_argument("--total-epochs", type=int, default=60)
    ap.add_argument("--cost-type", default="local")
    ap.add_argument("--topology", default="linear")
    ap.add_argument("--results-dir", default=None)
    # --- chọn backend ---
    ap.add_argument("--backend", choices=["simulate", "pennylane"], default="simulate",
                    help="backend train agent (env)")
    ap.add_argument("--deploy-backend", choices=["simulate", "pennylane", "same"],
                    default="same", help="backend cho phần deploy/đánh giá")
    ap.add_argument("--dataset", default="iris",
                    help="tên dataset trong dataset.py (iris, mnist, cancer, predictive_maintenance)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--entangler", choices=["cnot", "cz"], default="cnot",
                    help="entangler cho PennyLaneBackend (engine Stage1A chỉ hỗ trợ cnot/cz)")
    ap.add_argument("--calibration", default="outputs/stage1c/calibration_rules.json",
                    help="τ thật từ Stage 1C; nếu không tồn tại -> fallback tau=0.1")
    # --- chi phí/an toàn cho run PennyLane dài ---
    ap.add_argument("--deploy-epochs", type=int, default=100,
                    help="số epoch phần deploy/đánh giá (trước bị hard-code 100). "
                         "Giảm mạnh khi deploy trên pennylane.")
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="lưu agent.pt mỗi N episode (0=tắt). An toàn khi run dài crash.")
    ap.add_argument("--seed", type=int, default=0, help="seed cho 1 lần chạy")
    ap.add_argument("--seeds", default=None,
                    help="danh sách seed cách nhau dấu phẩy để chạy multi-seed, vd 0,1,2,3,4")
    ap.add_argument("--surrogate", default=None,
                    help="surrogate_fit.json hiệu chỉnh SimulateBackend từ PennyLane thật")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="TẮT Telemetry Engine (mask + reward gate + nhãn chẩn đoán) cho cả "
                         "train (Stage3RLEnv) lẫn deploy (AdaptiveTrainer log+guard). Mặc định "
                         "TE BẬT (khớp mô hình hệ thống trong báo cáo). Dùng cờ này để tái lập "
                         "baseline không-TE (ablation).")
    ap.add_argument("--max-rel-loss", type=float, default=0.02,
                    help="nguong chap nhan mutation (add/prune/reduce): peak val-loss "
                         "trong cua so mask_epochs_T tang qua ti le nay -> rollback. "
                         "Mac dinh 0.02 (2%%) co the qua chat cho circuit PennyLane that "
                         "(nhieu gradient/loss tu nhien) -> thu 0.05 neu thay depth khong "
                         "bao gio tang duoc du agent co de xuat add_layer.")
    ap.add_argument("--mask-epochs-t", type=int, default=5,
                    help="so epoch cua cua so soft-mask (ramp alpha) truoc khi kiem "
                         "accept/rollback cho add/prune/reduce. Mac dinh 5.")
    args = ap.parse_args()

    # calibration path -> None nếu không tồn tại (fallback an toàn)
    _cal = args.calibration if Path(args.calibration).exists() else None
    args.calibration = _cal   # gắn lại để make_env/train dùng
    if _cal:
        print(f"  [calib] dùng τ thật từ {_cal}")
    else:
        print(f"  [calib] không thấy calibration -> tau mặc định 0.1")

    sys.path.append(args.rl_path)
    deploy_backend = args.backend if args.deploy_backend == "same" else args.deploy_backend
    needs_data = ("pennylane" in (args.backend, deploy_backend))
    if needs_data and args.backend == "pennylane":
        print("  [!] Train trực tiếp trên PennyLaneBackend rất CHẬM. "
              "Cân nhắc --backend simulate --deploy-backend pennylane, "
              "hoặc giảm --episodes/--total-epochs.")

    data = load_data_if_needed(args, needs_data)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]

    if len(seeds) == 1:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.results_dir or (ROOT / "outputs" / "rl_results" /
                                            args.backend / f"{args.agent}_{stamp}"))
        out_dir.mkdir(parents=True, exist_ok=True)
        run_once(args, seeds[0], data, deploy_backend, needs_data, out_dir)
        print(f"\n[Saved] -> {out_dir}")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = Path(args.results_dir or (ROOT / "outputs" / "rl_results" /
                                       args.backend / f"{args.agent}_multiseed_{stamp}"))
    parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sd in seeds:
        sub = parent / f"seed_{sd}"
        sub.mkdir(exist_ok=True)
        summaries.append(run_once(args, sd, data, deploy_backend, needs_data, sub))
    aggregate(summaries, parent)
    print(f"\n[Saved] -> {parent}")


if __name__ == "__main__":
    main()