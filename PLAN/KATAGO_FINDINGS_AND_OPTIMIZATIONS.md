# KataGo Findings and Optimization Plan

## Scope
This document focuses only on KataGo-related behavior in the training system:

1. How KataGo is configured and launched.
2. How KataGo is used during GRPO / full-game GRPO.
3. Where throughput bottlenecks exist.
4. What optimizations are feasible, prioritized by impact and effort.
5. A concrete benchmark and rollout plan.

Repository root: `/Users/phoenix/go-test`

---

## 1) Current KataGo Setup

### 1.1 Install and runtime defaults

KataGo setup is done in:
- `/Users/phoenix/go-test/scripts/setup_environment.sh`

What it currently does:
- Downloads CUDA KataGo binary (`katago-v1.15.3-cuda11.8-linux-x64.zip`).
- Downloads one neural net file.
- Writes `analysis.cfg` with these defaults:
  - `numSearchThreads = 8`
  - `nnCacheSizePowerOfTwo = 21`
  - `nnMutexPoolSizePowerOfTwo = 16`
  - `numNNServerThreadsPerModel = 2`
  - `nnMaxBatchSize = 16`
  - `maxVisits = 100`
  - `maxPlayouts = 300`
- Exports paths in `.env`:
  - `KATAGO_BIN`
  - `KATAGO_MODEL`
  - `KATAGO_CONFIG`

### 1.2 KataGo client implementation

KataGo client is implemented in:
- `/Users/phoenix/go-test/go_rl/katago.py`

Behavior:
- Starts one subprocess: `katago analysis -model ... -config ...`
- Uses a synchronized request path with an `RLock`.
- Writes one JSON request to stdin and waits for one stdout line response.
- `analyze_position(board)` returns:
  - `winrate`
  - optional `score_lead`
  - optional `best_move`

Important performance property:
- Current client path is blocking request-response, effectively serial from caller perspective.

---

## 2) How KataGo Is Used in Training

## 2.1 Move-level GRPO (`scripts/run_grpo.py`)

KataGo use pattern per sampled completion (typical path):

1. Parse board from prompt.
2. Run base analysis on current board (`katago.analyze_position`).
3. Normalize completion to a move.
4. Validate move legality.
5. If legal, apply move on candidate board.
6. Run candidate analysis (`katago.analyze_position` again).
7. Compute winrate delta -> reward shaping.

So in common legal case: ~2 KataGo analyses per completion.

With `rollout.num_samples = 8` (default in `config/grpo_9x9.yaml`), the KataGo call volume per batch can be substantial.

## 2.2 Full-game GRPO (`scripts/run_grpo_fullgame.py` + `go_rl/fullgame_rollout.py`)

KataGo is used more heavily:
- Pre/post checks for model turns.
- Opponent move selection (`choose_move`) every opponent turn.
- Final position analysis for terminal outcome.

This is much higher analysis call volume than move-level GRPO.

---

## 3) Distributed Behavior and KataGo Scaling Limit

In both GRPO runners:
- Reward computation is constrained to rank 0.
- KataGo is started only on rank 0.
- Worker ranks do not run KataGo and rely on trainer broadcast path.

Files:
- `/Users/phoenix/go-test/scripts/run_grpo.py`
- `/Users/phoenix/go-test/scripts/run_grpo_fullgame.py`

Implication:
- Multi-GPU training can still bottleneck on a single rank’s reward engine throughput.
- Even if model-side DDP scales, reward-side throughput may not.

---

## 4) Primary KataGo Bottlenecks

1. **Single reward process bottleneck**
- One KataGo process on rank 0 handles reward calls.

2. **Synchronous request path**
- Request then wait; no pipeline/queue depth in client.

3. **Potential duplicate analyses**
- Repeated prompts may cause repeated base analysis.
- Repeated completions may cause repeated candidate analysis.

4. **Expensive training-time search settings**
- `maxVisits=100` and `maxPlayouts=300` can be costly for reward shaping.
- These settings may be over-provisioned for RL reward estimation compared with final evaluation.

5. **GPU contention risk**
- Rank-0 training and KataGo can compete on the same GPU unless explicitly separated.

---

## 5) Optimization Opportunities (KataGo-Only)

Ordered by expected ROI.

## 5.1 High impact, low-to-medium effort

1. **Base-position analysis cache**
- Cache by board-state hash for the reward function scope.
- Reuse base analysis when the same prompt/position appears repeatedly.

2. **Candidate analysis cache**
- Cache by `(board-state hash, move)`.
- Reuse candidate analysis for repeated completions.

3. **Training-specific lighter KataGo config**
- Keep strong config for eval; use cheaper config for reward loops.
- Typical knob reductions:
  - lower `maxVisits`
  - lower `maxPlayouts`
  - tune `numSearchThreads` and `nnMaxBatchSize` for hardware

4. **Dedicated GPU for KataGo**
- Isolate KataGo from rank-0 training process to reduce contention.

## 5.2 Medium-to-high effort, high potential

5. **Asynchronous/pipelined KataGo requests**
- Submit multiple analysis requests before blocking on responses.
- Exploit request IDs and maintain an in-flight map.

6. **Multi-process KataGo worker pool**
- Run multiple KataGo analysis processes and shard requests.
- Potentially assign workers to different GPUs.

7. **Distributed reward service design**
- Move away from strict rank-0 bottleneck.
- Requires careful synchronization and determinism strategy.

---

## 6) Practical Tuning Targets for `analysis.cfg`

Current defaults are quality-leaning. For training-time reward estimation, create a separate config, for example:

- `maxVisits`: reduce first (largest direct cost lever)
- `maxPlayouts`: reduce or cap below current default
- `numSearchThreads`: tune to avoid CPU overhead and context switching
- `nnMaxBatchSize`: increase only if queue depth exists and GPU has headroom
- `numNNServerThreadsPerModel`: tune with measured GPU occupancy

Key rule:
- Maintain separate configs for:
  - **training reward** (fast, noisier acceptable)
  - **evaluation / gating** (higher quality, slower acceptable)

---

## 7) Benchmark Plan for KataGo Optimizations

For each optimization, measure before/after on fixed seeds and same prompt subset.

Track:
1. KataGo requests/sec
2. Reward function wall time / batch
3. GRPO steps/sec
4. Rank-0 GPU utilization
5. Rank-0 CPU utilization
6. End quality proxy:
- eval winrate / Elo on fixed evaluation protocol

Minimal benchmark matrix:
1. Baseline (current config, no cache)
2. + base cache
3. + candidate cache
4. + lighter training KataGo config
5. + dedicated KataGo GPU
6. + async pipeline

Stop criteria:
- Accept optimization if throughput gain is meaningful and evaluation drop stays within agreed tolerance.

---

## 8) Implementation Roadmap (KataGo-Focused)

## Phase 1: Fast wins
1. Add base and candidate caching in GRPO reward path.
2. Split KataGo configs into training vs eval.
3. Tune training config to reduce visits/playouts.
4. Run A/B benchmark and verify no severe quality regression.

## Phase 2: Throughput scaling
5. Add async request pipeline in KataGo client.
6. Add worker-pool mode (multiple KataGo processes).
7. Re-benchmark multi-GPU scaling efficiency after bottleneck relief.

## Phase 3: Advanced architecture
8. If still bottlenecked, design distributed reward-serving architecture beyond single rank.

---

## 9) Risks and Mitigations

1. **Noisier rewards after lowering KataGo search**
- Mitigation: hold-out eval protocol with stronger KataGo settings.

2. **Cache correctness bugs**
- Mitigation: strict board hash design + validation tests.

3. **Async ordering issues**
- Mitigation: request ID tracking + deterministic response matching.

4. **Resource contention in multi-process mode**
- Mitigation: explicit CPU/GPU affinity and process budgeting.

---

## 10) File Index (KataGo-Relevant)

Setup and defaults:
- `/Users/phoenix/go-test/scripts/setup_environment.sh`

KataGo client:
- `/Users/phoenix/go-test/go_rl/katago.py`

Move-level GRPO reward loop:
- `/Users/phoenix/go-test/scripts/run_grpo.py`

Full-game GRPO reward loop:
- `/Users/phoenix/go-test/scripts/run_grpo_fullgame.py`
- `/Users/phoenix/go-test/go_rl/fullgame_rollout.py`

Reward shaping and legality:
- `/Users/phoenix/go-test/go_rl/rewards.py`
- `/Users/phoenix/go-test/go_rl/board.py`
- `/Users/phoenix/go-test/go_rl/prompting.py`

Configs:
- `/Users/phoenix/go-test/config/grpo_9x9.yaml`
- `/Users/phoenix/go-test/config/grpo_19x19.yaml`
- `/Users/phoenix/go-test/config/grpo_fullgame_19x19.yaml`

