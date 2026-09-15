# RL-theory improvements (not yet re-validated by an end-to-end run)

This folder holds this project's own earlier redesign of the Stage 2/3 RL
pipeline — kept here, not deleted, because it addresses real, documented RL
correctness issues, but it is not what produced any currently-reported result
(`tab:ai4i`, `tab:heart`, or any other table in Section 6).

## Why it's here instead of live in `scripts/`/`src/`

As of 2026-09-16 this repository's active `scripts/`/`src/synthetic_bp/stage2`
and `stage3` pipeline was reset to match the teammate (Nam)'s implementation
(<https://github.com/LeNam123456/AL_QNN>) exactly, so the committed pipeline
can reproduce `tab:ai4i` and `tab:heart`'s numbers without an expensive
PennyLane re-run. The files below are what this repository's pipeline looked
like *before* that reset — a set of RL-design fixes made and reasoned
through, but never run end-to-end on real PennyLane at a reportable qubit
count to produce a table in the report. Keeping both codebases live at once
without duplication meant picking one; this one lost only because it lacks a
validated result behind it, not because it's judged wrong.

## What changed here, and why (each addresses a specific, documented problem)

| File | Change | Problem it fixes |
|---|---|---|
| `src/synthetic_bp/stage3/rl_env.py` | One decision = one reward, every step (was: one reward per full round of `depth` decisions) | Sparse/delayed credit assignment — especially harmful to DQN's 1-step TD bootstrapping, which needs many zero-reward steps to propagate a signal back |
| `src/synthetic_bp/stage3/rl_env.py` | Targets/encodes the lowest-gradient-variance ("weakest") layer, shared with `RLScheduler` at deploy | Matches train-time state distribution to deploy-time state distribution (previously they could diverge) |
| `src/synthetic_bp/stage3/rl_env.py` | Action mask forces KEEP while a mutation is still resolving | Previously a masked-out action could still be silently dropped by `MutationEngine.submit()` yet get logged as if it had an effect — noisy training data |
| `src/synthetic_bp/stage3/rl_env.py` | Reward's near-zero-ratio term uses a delta (`cur - prev`) like every other term | The absolute-value version penalized every step at whatever the current near-zero ratio was (rarely 0 in practice) regardless of whether the policy was improving, dragging return negative even after convergence |
| `src/synthetic_bp/stage3/mutation_engine.py` | `add_layer` goes through the same ramp (mask 0→1 over T epochs) + rollback safety net as prune/reduce | Previously a single harmful `add_layer` (one that worsens barren-plateau behaviour) could never be undone |
| `src/synthetic_bp/stage3/circuit_backend.py` | `SimulateBackend`'s trainability heuristic depends on `n_qubits` (McClean et al. 2018 / Cerezo et al. 2021 cost-dependent BP scaling); convergence step is proportional to the remaining loss gap instead of a fixed step per epoch | The old heuristic didn't depend on qubit count at all, and a fixed step created an unrealistic "cliff" between depths where training either fully converged or got stuck, with no smooth degradation in between |
| `src/synthetic_bp/stage3/circuit_backend.py` | `PennyLaneBackend.sample_grad_variance()` computes variance-per-parameter-across-random-inits, then averages over parameters | The previous formula computed variance-across-parameters within one init and averaged over inits — a different "spatial" quantity that doesn't match the barren-plateau definition and gave nonsensical (near-zero or negative) calibration fits |
| `src/synthetic_bp/stage2/rl_scheduler.py` | Deploy-time `decide()` looks at only the weakest layer (matching `rl_env.py`'s training encoding) and asks the agent for a deterministic/greedy action | Matches train/deploy state distribution; a stochastic/epsilon-influenced deploy action is not reproducible |
| `src/synthetic_bp/stage3/trainer.py`, `rl_env.py` | Telemetry Engine (`use_telemetry`/`log_telemetry`/`telemetry_guard`) defaults to on | Matches the system model described in the report (Section 4.4.1): the Telemetry Engine is meant to always be active, not an opt-in ablation |
| `scripts/dataset.py` | AI4I imbalance corrected with SMOTE (oversample train split only, after PCA/pad + `[0,π]` scaling) | Avoids discarding real majority-class rows the way naive undersampling did, without depending on a teammate's separate implementation |

## Status

None of this has been run end-to-end on real PennyLane at a reportable qubit
count since these changes were made — doing so is future work (mentioned in
`report/contents/conclusion.tex` §7.4 as one open direction). If it ever is,
this is the code to start from.
