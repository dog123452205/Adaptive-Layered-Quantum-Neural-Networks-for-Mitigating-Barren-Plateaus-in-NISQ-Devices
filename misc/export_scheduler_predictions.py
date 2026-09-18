"""
export_scheduler_predictions.py — Export per-sample VQC predictions for the
GREEDY / PPO / DQN schedulers, on the EXACT SAME held-out TEST split as
export_predictions.py's RuleBasedScheduler ("quantum") export — same
SEED/VAL_FRAC/TEST_FRAC/MAXS_BY_DATASET — so the demo can show Rule / Greedy /
PPO / DQN as directly comparable "AL-QNN variant" choices for the same rows.

Only Heart and AI4I (predictive_maintenance) are in scope — both run on
n_qubits=8, so a single pair of trained PPO/DQN agents (from
outputs/rl_results/ppo_q8/agent.pt and dqn_q8/agent.pt, trained via
scripts/04_train_rl_scheduler.py --backend simulate --deploy-backend simulate)
is reused for both datasets — the agent is a scheduling POLICY over generic
circuit-state features, not something tied to one dataset's raw values.

Output per (dataset, scheduler): predictions_<dataset>_<scheduler>.csv (same
columns as predictions_<dataset>.csv: f0..f7, true_label, pred, confidence,
proba_class1), and a "<scheduler>" block merged into stats_<dataset>.json
with the same shape as the existing "quantum" block.

RUN FROM misc/ (same folder as export_predictions.py):
    python export_scheduler_predictions.py                    # all datasets x all schedulers
    python export_scheduler_predictions.py heart               # only heart, all schedulers
    python export_scheduler_predictions.py heart:ppo            # only heart PPO
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from dataset import load_binary_dataset
from src.synthetic_bp.stage3.circuit_backend import PennyLaneBackend
from src.synthetic_bp.stage3.trainer import AdaptiveTrainer
from src.synthetic_bp.stage2.greedy_scheduler import GreedyScheduler
from src.synthetic_bp.stage2.rl_scheduler import RLScheduler

# must match export_predictions.py exactly, or the TEST split won't line up
SEED = 123; EPOCHS = 40; COST = "local"; TOPO = "linear"
VAL_FRAC = 0.3; TEST_FRAC = 0.2
CALIBRATION = str(ROOT / "outputs" / "stage1c" / "calibration_rules.json")

DATASETS = {
    "heart": {"load_name": "heart", "n_qubits": 8, "friendly": "Heart Disease (Cleveland)", "maxs": 280},
    "predictive_maintenance": {"load_name": "predictive_maintenance", "n_qubits": 8,
                               "friendly": "AI4I Predictive Maintenance", "maxs": None},
}

AGENT_CKPT = {
    "ppo": ROOT / "outputs" / "rl_results" / "ppo_q8" / "agent.pt",
    "dqn": ROOT / "outputs" / "rl_results" / "dqn_q8" / "agent.pt",
}

STATS_LABEL = {
    "greedy": "AL-QNN (Greedy Scheduler)",
    "ppo": "AL-QNN (PPO Scheduler)",
    "dqn": "AL-QNN (DQN Scheduler)",
}


def load_ppo_agent():
    sys.path.append(str(ROOT / "RL" / "PPO"))
    from agent_ppo_torch import PPOAgent
    import torch
    agent = PPOAgent(seed=SEED)
    agent.ac.load_state_dict(torch.load(AGENT_CKPT["ppo"], map_location=agent.device))
    agent.ac.eval()
    return agent


def load_dqn_agent():
    sys.path.append(str(ROOT / "RL" / "DQN"))
    from agent_dqn_torch import DQNAgent
    import torch
    agent = DQNAgent(seed=SEED)
    agent.q.load_state_dict(torch.load(AGENT_CKPT["dqn"], map_location=agent.device))
    agent.q.eval()
    return agent


def _greedy_act_fn(agent, state, mask):
    # RLScheduler.__init__ dropped its `greedy` kwarg (teammate refactor);
    # _default_act now calls agent.act(state, mask) with NO greedy flag,
    # which defaults to PPOAgent/DQNAgent's own greedy=False (stochastic /
    # epsilon-greedy) — wrong for deploy, which must be deterministic and
    # reproducible. Supply our own act_fn (RLScheduler still accepts that)
    # that forces greedy=True, without touching the shared rl_scheduler.py.
    out = agent.act(state, mask, greedy=True)
    return int(out[0]) if isinstance(out, tuple) else int(out)


def make_scheduler(name: str):
    if name == "greedy":
        return GreedyScheduler(calibration=None)
    if name == "ppo":
        agent = load_ppo_agent()
        return RLScheduler(agent, tau=0.1, total_epochs=EPOCHS, warmup_epochs=10,
                            calibration=CALIBRATION, act_fn=_greedy_act_fn)
    if name == "dqn":
        agent = load_dqn_agent()
        return RLScheduler(agent, tau=0.1, total_epochs=EPOCHS, warmup_epochs=10,
                            calibration=CALIBRATION, act_fn=_greedy_act_fn)
    raise ValueError(f"unknown scheduler: {name}")


def parse_args(argv):
    if not argv:
        return [(k, s) for k in DATASETS for s in ("greedy", "ppo", "dqn")]
    out = []
    for tok in argv:
        if ":" in tok:
            k, s = tok.split(":", 1)
            out.append((k, s))
        else:
            out.extend((tok, s) for s in ("greedy", "ppo", "dqn"))
    return out


jobs = parse_args(sys.argv[1:])

for name, sched_name in jobs:
    cfg = DATASETS[name]
    nq, friendly, maxs = cfg["n_qubits"], cfg["friendly"], cfg["maxs"]
    print(f"\n=== {friendly} ({name}) x {sched_name} scheduler ({nq}q, max_samples={maxs}) ===")
    try:
        data = load_binary_dataset(name=cfg["load_name"], n_qubits=nq, val_frac=VAL_FRAC,
                                   test_frac=TEST_FRAC, seed=SEED, max_samples=maxs)
    except Exception as e:
        print(f"  SKIP (load fail): {e}"); continue
    if data["n_test"] == 0:
        print(f"  SKIP (n_test=0)"); continue
    try:
        b = PennyLaneBackend(data["X_train"], data["y_train"],
                             data["X_val"], data["y_val"],
                             n_qubits=nq, initial_depth=2, max_depth=8,
                             topology=TOPO, entangler_type="cnot",
                             cost_type=COST, seed=SEED,
                             batch_sampler=data.get("batch_sampler"))
        sched = make_scheduler(sched_name)
        tr = AdaptiveTrainer(b, sched, n_qubits=nq, cost_type=COST,
                             topology=TOPO, epochs=EPOCHS)
        tr.run()
        qnode = b._make_qnode()
        Xt, yt = data["X_test"], data["y_test"]
        qnode(b.theta, Xt[0])  # warm-up
        rows, correct, infer_times = [], 0, []
        for x, y in zip(Xt, yt):
            t0 = time.perf_counter()
            p = float((qnode(b.theta, x) + 1) / 2)
            infer_times.append(time.perf_counter() - t0)
            pred = int(p > 0.5); correct += int(pred == int(y))
            rec = {f"f{i}": round(float(v), 5) for i, v in enumerate(x)}
            rec.update(true_label=int(y), pred=pred,
                       confidence=round(max(p, 1 - p), 4), proba_class1=round(p, 4))
            rows.append(rec)
        out = HERE / f"predictions_{name}_{sched_name}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        acc = correct / max(len(yt), 1)
        avg_ms = 1000 * float(np.mean(infer_times))
        total_ms = 1000 * float(np.sum(infer_times))
        print(f"  -> {out.name}  ({len(rows)} samples, TEST acc {acc:.3f}, final depth {b.depth}, "
              f"avg inference {avg_ms:.2f} ms/sample, total {total_ms:.1f} ms)")

        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        yp = [r["pred"] for r in rows]; yt_list = [r["true_label"] for r in rows]
        yprob = [r["proba_class1"] for r in rows]
        stats_path = HERE / f"stats_{name}.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        stats[sched_name] = {
            "label": STATS_LABEL[sched_name],
            "accuracy": round(acc, 4),
            "precision": round(precision_score(yt_list, yp, zero_division=0), 4),
            "recall": round(recall_score(yt_list, yp, zero_division=0), 4),
            "f1": round(f1_score(yt_list, yp, zero_division=0), 4),
            "auc": round(roc_auc_score(yt_list, yprob), 4) if len(set(yt_list)) > 1 else None,
            "final_depth": int(b.depth),
            "n_qubits": nq,
            "avg_infer_ms": round(avg_ms, 4),
            "total_infer_ms": round(total_ms, 2),
        }
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {stats_path.name} updated ({sched_name} stats)")
    except Exception as e:
        print(f"  SKIP (deploy fail): {type(e).__name__}: {e}")

print("\nDone.")
