"""
Deploy Rule / Greedy / Fixed-depth / DQN / PPO trên cùng backend + seed + config để so sánh.
Rule/Greedy/Fixed chạy được trên simulate không cần torch; DQN/PPO cần torch + ckpt
agent.pt (từ script 04).

Chạy:
    # chỉ rule + greedy + fixed-depth (không cần torch)
    python scripts/05_compare_schedulers.py --backend simulate --fixed-depths 2,4,8

    # đủ cả 5 (rule, greedy, fixed-depth, ppo, dqn)
    python scripts/05_compare_schedulers.py --backend simulate --fixed-depths 2,4,8 \
        --ppo-ckpt outputs/rl_results/simulate/ppo_XXXX/agent.pt \
        --dqn-ckpt outputs/rl_results/simulate/dqn_YYYY/agent.pt

--fixed-depths k1,k2,...: mỗi k chạy AdaptiveTrainer với FixedScheduler (không mutate,
chỉ KEEP) tại initial_depth=max_depth=k -> baseline "kiến trúc cố định depth k" để so
sánh với các scheduler thích nghi (rule/greedy/ppo/dqn) trong CÙNG 1 bảng/1 hình,
không cần chạy script 06 riêng rồi tự gộp CSV nữa.
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
from src.synthetic_bp.stage2.fixed_scheduler import FixedScheduler
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


def build_backend(args, data, seed, depth_override=None):
    """depth_override: dùng cho baseline fixed-depth -> initial_depth=max_depth=k,
    ghi đè args.initial_depth/args.max_depth chỉ cho lần build này."""
    init_d = args.initial_depth if depth_override is None else depth_override
    max_d = args.max_depth if depth_override is None else depth_override
    from src.synthetic_bp.stage3.circuit_backend import SimulateBackend
    if args.backend == "pennylane":
        from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend
        return PennyLaneBackend(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_qubits=args.n_qubits, initial_depth=init_d,
            max_depth=max_d, topology=args.topology,
            entangler_type=args.entangler, cost_type=args.cost_type, seed=seed)
    return SimulateBackend(n_qubits=args.n_qubits, initial_depth=init_d,
                           max_depth=max_d, cost_type=args.cost_type,
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


def deploy(name, scheduler, args, data, depth_override=None):
    b = build_backend(args, data, seed=args.seed, depth_override=depth_override)
    tr = AdaptiveTrainer(b, scheduler, n_qubits=args.n_qubits,
                         cost_type=args.cost_type, topology=args.topology,
                         epochs=args.epochs,
                         mask_epochs_T=args.mask_epochs_t,
                         max_relative_loss_increase=args.max_rel_loss,
                         log_telemetry=not getattr(args, "no_telemetry", False),
                         telemetry_guard=not getattr(args, "no_telemetry", False))
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
    ax_acc.set(title="Accuracy over epoch", xlabel="epoch", ylabel="accuracy")
    ax_depth.set(title="Depth over epoch", xlabel="epoch", ylabel="depth")
    for ax in (ax_acc, ax_depth):
        ax.grid(alpha=0.3); ax.legend()

    names = [n for n, _ in results]
    acc_end = [float(df.iloc[-1]["accuracy"]) for _, df in results]
    ax_bar.bar(names, acc_end)
    ax_bar.set(title="Final accuracy", ylabel="accuracy", ylim=(0, 1))
    ax_bar.tick_params(axis="x", rotation=30)
    for i, v in enumerate(acc_end):
        ax_bar.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax_bar.grid(alpha=0.3, axis="y")

    commits = [int(df.iloc[-1]["n_commits"]) for _, df in results]
    rollbacks = [int(df.iloc[-1]["n_rollbacks"]) for _, df in results]
    x = np.arange(len(names))
    ax_cr.bar(x - 0.19, commits, 0.38, label="commits")
    ax_cr.bar(x + 0.19, rollbacks, 0.38, label="rollbacks")
    ax_cr.set(title="Commits / Rollbacks")
    ax_cr.set_xticks(x); ax_cr.set_xticklabels(names, rotation=30)
    ax_cr.grid(alpha=0.3, axis="y"); ax_cr.legend()

    fig.suptitle("Adaptive vs. Fixed-Depth Scheduler Comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def collect_schedulers(args):
    """Trả về list (name, scheduler, depth_override). depth_override=None -> dùng
    args.initial_depth/args.max_depth chung (scheduler thích nghi); khác None ->
    fixed-depth baseline (initial_depth=max_depth=depth_override, không mutate)."""
    cal = args.calibration if Path(args.calibration).exists() else None
    scheds = [("rule", RuleBasedScheduler(calibration=cal), None)]
    if not args.no_greedy:
        scheds.append(("greedy", GreedyScheduler(calibration=cal), None))

    fixed_depths = [int(x) for x in str(args.fixed_depths).split(",") if x.strip()]
    for k in fixed_depths:
        scheds.append((f"fixed_k{k}", FixedScheduler(), k))

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
                                         warmup_epochs=args.warmup, calibration=cal), None))
    return scheds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["simulate", "pennylane"], default="simulate")
    ap.add_argument("--dataset", default="iris",
                    help="tên dataset trong dataset.py (iris, mnist, cancer, predictive_maintenance)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--n-qubits", type=int, default=4)
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
    ap.add_argument("--fixed-depths", default="",
                    help="danh sach depth k co dinh (fixed, khong mutate) de so sanh, "
                         "vd '2,4,8'. Rong = bo qua (mac dinh).")
    ap.add_argument("--max-rel-loss", type=float, default=0.02,
                    help="nguong accept/rollback cho add/prune/reduce (mac dinh 0.02).")
    ap.add_argument("--mask-epochs-t", type=int, default=5,
                    help="so epoch cua so soft-mask truoc khi kiem accept/rollback (mac dinh 5).")
    ap.add_argument("--ppo-ckpt", default=None)
    ap.add_argument("--dqn-ckpt", default=None)
    ap.add_argument("--rl-root", default=None)
    ap.add_argument("--ppo-hidden", type=int, default=128)
    ap.add_argument("--dqn-hidden", default="128,128")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-telemetry", action="store_true",
                    help="TẮT Telemetry Engine (log chẩn đoán + guardrail veto add/prune/reduce "
                         "khi đã solved) cho MỌI scheduler. Mặc định TE BẬT (khớp mô hình hệ "
                         "thống trong báo cáo). Dùng cờ này để tái lập baseline không-TE.")
    args = ap.parse_args()

    args.rl_root = Path(args.rl_root) if args.rl_root else ROOT / "RL"
    out_dir = Path(args.out or ROOT / "outputs" / "comparison" / args.backend)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args)
    scheds = collect_schedulers(args)
    print(f"[compare] {len(scheds)} scheduler | backend={args.backend} "
          f"seed={args.seed} epochs={args.epochs}")

    results, rows = [], []
    for name, sched, depth_override in scheds:
        print(f"[compare] deploy {name}"
              + (f" (fixed depth={depth_override})" if depth_override is not None else ""))
        df = deploy(name, sched, args, data, depth_override=depth_override)
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