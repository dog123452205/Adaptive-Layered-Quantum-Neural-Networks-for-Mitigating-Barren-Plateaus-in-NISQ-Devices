"""
Deploy Rule / Greedy / DQN / PPO trên cùng backend + seed + config để so sánh.
Rule và Greedy chạy được trên simulate không cần torch; DQN/PPO cần torch + ckpt agent.pt (từ script 04).

Chạy:
    # chỉ rule và greedy (không cần torch)
    python scripts/05_compare_schedulers.py --backend simulate

    # đủ 4 agents
    python scripts/05_compare_schedulers.py --backend pennylane \
        --ppo-ckpt outputs/rl_results/bcw_6q_ppo/agent.pt \
        --dqn-ckpt outputs/rl_results/bcw_6q_dqn/agent.pt

Ghi chú xuất xứ (2026-09-16): nội dung file này là bản của Lê Hoàng Nam
(github.com/LeNam123456/AL_QNN), KHÔNG phải bản do Lam Vuong tự viết, dù
"Deployment: run learned policy on PennyLane; compare against rule and greedy"
ghi nhận thuộc trách nhiệm Dao Quang Minh trong WBS (project_management.tex).
Dùng bản này vì nó đúng là script đã tạo ra tab:ai4i/tab:heart. Bản gốc do
Lam Vuong viết được lưu ở external/rl_theory_improvements/scripts/05_compare_schedulers.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.synthetic_bp.stage2.rule_based_scheduler import RuleBasedScheduler
from src.synthetic_bp.stage2.greedy_scheduler import GreedyScheduler
from src.synthetic_bp.stage2.rl_scheduler import RLScheduler
from src.synthetic_bp.stage3.trainer import AdaptiveTrainer

MACROS = ["keep", "add_layer", "stop_growth",
          "propose_prune_layer", "propose_reduce_entanglement"]


def load_data(args):
    if args.backend != "pennylane":
        return None
    from dataset import load_binary_dataset
    d = load_binary_dataset(args.dataset, n_qubits=args.n_qubits,
                            max_samples=args.max_samples, seed=42)
    print(f"[data] {d['name']}: train={d['n_train']} val={d['n_val']}")
    return d


def build_backend(args, data, seed):
    from src.synthetic_bp.stage3.circuit_backend import SimulateBackend
    if args.backend == "pennylane":
        from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend
        return PennyLaneBackend(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_qubits=args.n_qubits, initial_depth=args.initial_depth,
            max_depth=args.max_depth, topology=args.topology,
            entangler_type=args.entangler, cost_type=args.cost_type, seed=seed,
            device_name=getattr(args, 'device_name', 'lightning.gpu'), diff_method="best",
            batch_sampler=data.get("batch_sampler", None))
    return SimulateBackend(n_qubits=args.n_qubits, initial_depth=args.initial_depth,
                           max_depth=args.max_depth, cost_type=args.cost_type,
                           topology=args.topology, seed=seed)


class GreedyPolicy:
    """Bọc net đã train thành policy argmax để RLScheduler gọi .act(state, mask)."""
    def __init__(self, net, torch, kind):
        self.net = net.eval()
        self.torch = torch
        self.kind = kind

    def act(self, state, mask):
        t = self.torch
        s = t.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        mask = np.asarray(mask, dtype=bool)
        with t.no_grad():
            if self.kind == "ppo":
                logits, _ = self.net(s, t.as_tensor(mask).unsqueeze(0))
                return int(logits.argmax(dim=1).item())
            q = self.net(s).squeeze(0).cpu().numpy()
            q[~mask] = -1e9
            return int(np.argmax(q))


def load_policy(kind, ckpt, rl_dir, hidden):
    import torch
    sys.path.append(str(rl_dir))
    if kind == "ppo":
        from agent_ppo_torch import ActorCritic
        net = ActorCritic(hidden=hidden)
    else:
        from agent_dqn_torch import QNet
        net = QNet(hidden=hidden)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return GreedyPolicy(net, torch, kind)


def deploy(name, scheduler, args, data):
    b = build_backend(args, data, seed=args.seed)
    tr = AdaptiveTrainer(b, scheduler, n_qubits=args.n_qubits,
                         cost_type=args.cost_type, topology=args.topology,
                         epochs=args.epochs,
                         log_telemetry=getattr(args, "log_telemetry", False),
                         telemetry_guard=getattr(args, "telemetry_guard", False))
    return tr.run()


def summarize(name, df, args):
    first, last = df.iloc[0], df.iloc[-1]
    counts = df["scheduler_action"].value_counts().to_dict()
    row = {
        "scheduler": name,
        "backend": args.backend,
        "seed": args.seed,
        "epochs": args.epochs,
        "acc_start": round(float(first["accuracy"]), 4),
        "acc_end": round(float(last["accuracy"]), 4),
        "acc_delta": round(float(last["accuracy"]) - float(first["accuracy"]), 4),
        "f1_end": round(float(last.get("f1", float("nan"))), 4),
        "roc_auc_end": round(float(last.get("roc_auc", float("nan"))), 4),
        "depth_start": int(first["depth"]),
        "depth_end": int(last["depth"]),
        "n_commits": int(last["n_commits"]),
        "n_rollbacks": int(last["n_rollbacks"]),
    }
    for a in MACROS:
        row[f"act_{a}"] = int(counts.get(a, 0))
    for a, c in counts.items():
        if a not in MACROS:
            row[f"act_{a}"] = int(c)
    return row


def plot(results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ((ax_acc, ax_depth), (ax_bar, ax_cr)) = plt.subplots(2, 2, figsize=(12, 8))
    for name, df in results:
        ax_acc.plot(df["epoch"], df["accuracy"], label=name, linewidth=1.8)
        ax_depth.step(df["epoch"], df["depth"], where="post", label=name, linewidth=1.8)
    ax_acc.set(title="Accuracy theo epoch", xlabel="epoch", ylabel="accuracy")
    ax_depth.set(title="Depth theo epoch", xlabel="epoch", ylabel="depth")
    for ax in (ax_acc, ax_depth):
        ax.grid(alpha=0.3); ax.legend()

    names = [n for n, _ in results]
    acc_end = [float(df.iloc[-1]["accuracy"]) for _, df in results]
    ax_bar.bar(names, acc_end)
    ax_bar.set(title="Accuracy cuoi", ylabel="accuracy", ylim=(0, 1))
    for i, v in enumerate(acc_end):
        ax_bar.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax_bar.grid(alpha=0.3, axis="y")

    commits = [int(df.iloc[-1]["n_commits"]) for _, df in results]
    rollbacks = [int(df.iloc[-1]["n_rollbacks"]) for _, df in results]
    x = np.arange(len(names))
    ax_cr.bar(x - 0.19, commits, 0.38, label="commits")
    ax_cr.bar(x + 0.19, rollbacks, 0.38, label="rollbacks")
    ax_cr.set(title="Commit / Rollback")
    ax_cr.set_xticks(x); ax_cr.set_xticklabels(names)
    ax_cr.grid(alpha=0.3, axis="y"); ax_cr.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def collect_schedulers(args):
    cal = args.calibration if Path(args.calibration).exists() else None
    scheds = [("rule", RuleBasedScheduler(calibration=cal))]
    if not args.no_greedy:
        scheds.append(("greedy", GreedyScheduler(calibration=cal)))

    for kind, ckpt in (("ppo", args.ppo_ckpt), ("dqn", args.dqn_ckpt)):
        if not ckpt:
            continue
        if not Path(ckpt).exists():
            print(f"[compare] bo qua {kind}: khong thay {ckpt}")
            continue
        try:
            rl_dir = args.rl_root / kind.upper()
            hidden = args.ppo_hidden if kind == "ppo" \
                else tuple(int(x) for x in str(args.dqn_hidden).split(","))
            pol = load_policy(kind, ckpt, rl_dir, hidden)
        except ImportError:
            print(f"[compare] bo qua {kind}: chua co torch")
            continue
        except Exception as e:
            print(f"[compare] bo qua {kind}: {type(e).__name__}: {e}")
            continue
        scheds.append((kind, RLScheduler(pol, tau=0.1, total_epochs=args.epochs,
                                         warmup_epochs=args.warmup, calibration=cal)))
    return scheds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["simulate", "pennylane"], default="simulate")
    ap.add_argument("--device-name", default="lightning.gpu")
    ap.add_argument("--dataset", default="cancer",
                    help="tên dataset trong dataset.py (iris, mnist, cancer, predictive_maintenance)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--initial-depth", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--cost-type", default="local")
    ap.add_argument("--topology", default="linear")
    ap.add_argument("--entangler", choices=["cnot", "cz"], default="cnot")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--calibration", default="outputs/stage1c/calibration_rules.json")
    ap.add_argument("--no-greedy", action="store_true", help="bỏ greedy scheduler")
    ap.add_argument("--ppo-ckpt", default=None)
    ap.add_argument("--dqn-ckpt", default=None)
    ap.add_argument("--rl-root", default=None)
    ap.add_argument("--ppo-hidden", type=int, default=128)
    ap.add_argument("--dqn-hidden", default="128,128")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log-telemetry", action="store_true",
                    help="ghi 5 cờ chẩn đoán TE vào log của MỌI scheduler (không đổi hành vi)")
    ap.add_argument("--telemetry-guard", action="store_true",
                    help="bật guardrail TE: veto add/prune/reduce khi đã solved (áp cho mọi scheduler)")
    args = ap.parse_args()

    args.rl_root = Path(args.rl_root) if args.rl_root else ROOT / "RL"
    out_dir = Path(args.out or ROOT / "outputs" / "comparison" / args.backend)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args)
    scheds = collect_schedulers(args)
    print(f"[compare] {len(scheds)} scheduler | backend={args.backend} "
          f"seed={args.seed} epochs={args.epochs}")

    results, rows = [], []
    for name, sched in scheds:
        print(f"[compare] deploy {name}")
        df = deploy(name, sched, args, data)
        df.to_csv(out_dir / f"deploy_log_{name}.csv", index=False)
        results.append((name, df))
        rows.append(summarize(name, df, args))

    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "comparison.csv", index=False)
    plot(results, out_dir / "comparison.png")

    cols = ["scheduler", "acc_start", "acc_end", "acc_delta", "f1_end",
            "depth_start", "depth_end", "n_commits", "n_rollbacks"]
    print(comp[cols].to_string(index=False))
    print(f"[compare] saved -> {out_dir}")
    if len(scheds) == 1:
        print("[compare] moi co rule. Them --ppo-ckpt/--dqn-ckpt (va torch) de du 3.")


if __name__ == "__main__":
    main()
