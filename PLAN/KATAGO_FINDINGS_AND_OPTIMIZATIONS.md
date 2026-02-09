# KataGo Optimization Change Plan (Implementation-Focused)

## Goal
Increase GRPO throughput by removing KataGo as a reward-path bottleneck while keeping Elo/winrate quality stable.

This document is intentionally focused on **what to change**, in code-level detail.

---

## 1) Current Constraints That Must Be Addressed

1. Reward computation is rank-0 only in GRPO and full-game GRPO.
2. Rank 0 starts one KataGo process only.
3. KataGo client request path is synchronous and lock-serialized.
4. Reward functions perform many repeated analyses of identical positions.
5. Training-time KataGo search budget (`maxVisits`, `maxPlayouts`) is expensive for high-volume reward scoring.
6. Training and KataGo may contend on the same GPU.

Relevant files:
- `/Users/phoenix/go-test/scripts/run_grpo.py`
- `/Users/phoenix/go-test/scripts/run_grpo_fullgame.py`
- `/Users/phoenix/go-test/go_rl/katago.py`
- `/Users/phoenix/go-test/scripts/setup_environment.sh`

---

## 2) Change Set A: Reward-Side Caching (Immediate ROI)

### A1. Add base-position analysis cache (move-level GRPO)

### What to change
In `/Users/phoenix/go-test/scripts/run_grpo.py`, inside `build_reward_fn(...)`:

1. Compute a deterministic key for parsed board positions.
2. Cache `base_analysis` per board key.
3. Reuse base analysis across repeated prompts in the same reward batch.

### Code-level design
Add helper in `run_grpo.py`:

```python
def board_key(board: BoardState) -> tuple:
    return (board.size, tuple(board.moves))
```

Inside reward function closure:

```python
base_cache: dict[tuple, KataGoResult] = {}

key = board_key(board)
if key not in base_cache:
    base_cache[key] = katago.analyze_position(board)
base_analysis = base_cache[key]
```

### Why it helps
GRPO often generates multiple completions from the same prompt (`num_generations`), so base position analysis is duplicated heavily.

### Risk
Very low. Cache invalidation is trivial because board state is immutable within that iteration.

---

### A2. Add candidate `(position, move)` cache (move-level GRPO)

### What to change
Still in `build_reward_fn(...)`:

1. Cache candidate analysis by `(board_key, normalized_move)`.
2. Reuse candidate analysis when sampled completion repeats.

### Code-level design

```python
candidate_cache: dict[tuple, KataGoResult] = {}

cand_key = (key, move)
if cand_key not in candidate_cache:
    candidate = BoardState(size=board.size, moves=list(board.moves))
    candidate.apply_move(player, move)
    candidate_cache[cand_key] = katago.analyze_position(candidate)
candidate_analysis = candidate_cache[cand_key]
```

### Why it helps
Repeated completions are common with bounded `max_tokens` and temperature sampling.

### Risk
Low. Key must use normalized move string and canonical board state representation.

---

### A3. Extend caching to full-game GRPO

### What to change
In `/Users/phoenix/go-test/go_rl/fullgame_rollout.py` and/or reward closure in `/Users/phoenix/go-test/scripts/run_grpo_fullgame.py`:

1. Cache analyses for board snapshots encountered repeatedly in simulation loops.
2. Reuse scorelead/winrate reads for same exact board.

### Design note
Use same `board_key(board)` strategy and keep cache scope per reward batch (or bounded LRU across batches if memory allows).

### Risk
Medium-low due to larger state space in full games. Must bound cache size.

---

## 3) Change Set B: Split Training vs Evaluation KataGo Configs

### B1. Introduce dedicated fast training KataGo config

### What to change
Add separate config file(s):
- `/Users/phoenix/go-test/config/katago_analysis_train.cfg`
- `/Users/phoenix/go-test/config/katago_analysis_eval.cfg`

### Proposed initial training values (starting point)
- `maxVisits = 25` (from 100)
- `maxPlayouts = 80` (from 300)
- `numSearchThreads = 4-8` (hardware-dependent)
- `nnMaxBatchSize = 16` (keep initially)

### Wiring changes
1. Keep eval using high-quality config.
2. GRPO scripts use training config by default for reward path.
3. Eval scripts keep strict config for gating/final metrics.

### Files to update
- `/Users/phoenix/go-test/config/grpo_9x9.yaml`
- `/Users/phoenix/go-test/config/grpo_19x19.yaml`
- `/Users/phoenix/go-test/config/grpo_fullgame_19x19.yaml`
- optionally setup helper in `/Users/phoenix/go-test/scripts/setup_environment.sh`

### Risk
Medium. Reward noise increases. Must validate with fixed-seed A/B.

---

## 4) Change Set C: GPU Isolation for KataGo

### C1. Pin KataGo to dedicated GPU

### What to change
At KataGo process launch site (`go_rl/katago.py`), support explicit environment override.

### Code-level design
Add optional field to `KataGoConfig` in `/Users/phoenix/go-test/go_rl/config.py`:

```python
katago_cuda_visible_devices: Optional[str] = None
```

In `KataGoAnalysisClient.start()`:

```python
env = os.environ.copy()
if self.config.katago_cuda_visible_devices:
    env["CUDA_VISIBLE_DEVICES"] = self.config.katago_cuda_visible_devices
self._process = subprocess.Popen(..., env=env)
```

### Ops recommendation
- Run model training on one device set.
- Run KataGo on a separate device set.

### Risk
Low. Mostly operational; requires hardware with >=2 GPUs for benefit.

---

## 5) Change Set D: Asynchronous KataGo Client

### D1. Add non-blocking request/response multiplexer

### What to change
Refactor `/Users/phoenix/go-test/go_rl/katago.py` to support:
1. Background stdout reader thread.
2. Pending map by request `id`.
3. API that submits requests without immediate block.
4. API to wait for specific IDs or gather batch responses.

### New API proposal

```python
class KataGoAnalysisClient:
    def submit_position(self, board: BoardState) -> int: ...
    def await_result(self, req_id: int, timeout: float | None = None) -> KataGoResult: ...
    def analyze_positions(self, boards: list[BoardState]) -> list[KataGoResult]: ...
```

### Integration changes
- In reward functions, submit N requests then await all.
- In full-game rollout, pipeline pre/post analyses when possible.

### Risk
Medium-high. Concurrency correctness is critical.

### Test requirements
Add tests in:
- `/Users/phoenix/go-test/tests/test_katago.py`

Test cases:
1. Out-of-order responses map to correct request IDs.
2. Timeout path behavior.
3. Graceful close with pending requests.
4. Backward-compatible `analyze_position` wrapper.

---

## 6) Change Set E: KataGo Worker Pool

### E1. Multi-process analysis workers on rank 0

### What to change
Implement lightweight worker pool service (new module recommended):
- `/Users/phoenix/go-test/go_rl/katago_pool.py`

### Responsibilities
1. Spawn K `KataGoAnalysisClient` instances.
2. Route requests round-robin or least-loaded.
3. Return merged results preserving request order.

### API proposal

```python
class KataGoPool:
    def __init__(self, configs: list[KataGoConfig]): ...
    def analyze_position(self, board: BoardState) -> KataGoResult: ...
    def analyze_positions(self, boards: list[BoardState]) -> list[KataGoResult]: ...
    def close(self) -> None: ...
```

### Integration point
- Replace single `KataGoAnalysisClient` object in GRPO scripts with pool-backed object.

### Risk
Medium-high due to process orchestration and host resource contention.

---

## 7) Change Set F: Rank-0 Bottleneck Reduction Strategy

### F1. Preserve correctness first; parallelize within rank 0 before cross-rank redesign

Recommended order:
1. Caching.
2. Async client.
3. Worker pool.

Only after these:
4. Consider distributed reward serving beyond rank 0.

Reason:
- Cross-rank reward decentralization is more invasive and easier to destabilize.

---

## 8) Concrete File-by-File Change Matrix

1. `/Users/phoenix/go-test/scripts/run_grpo.py`
- Add board key helper.
- Add base and candidate caches in reward function.
- Optional batch submission if async client exists.

2. `/Users/phoenix/go-test/scripts/run_grpo_fullgame.py`
- Add or wire bounded cache usage for repeated full-game analyses.
- Keep rank-0 safety checks.

3. `/Users/phoenix/go-test/go_rl/fullgame_rollout.py`
- Add optional analysis cache injection to avoid repeated board analyses.

4. `/Users/phoenix/go-test/go_rl/katago.py`
- Add async multiplexer APIs while preserving existing sync API for compatibility.
- Add env override support for GPU pinning.

5. `/Users/phoenix/go-test/go_rl/config.py`
- Extend `KataGoConfig` with optional runtime controls:
  - `katago_cuda_visible_devices`
  - `request_timeout_seconds`
  - optional pool size if using pool architecture

6. `/Users/phoenix/go-test/config/*.yaml`
- Add training-specific KataGo config path.
- Optional GPU pinning fields for training jobs.

7. `/Users/phoenix/go-test/tests/test_katago.py`
- Add async/client-pool tests.
- Add cache behavior tests via controlled mock responses.

---

## 9) Benchmark Protocol (Mandatory for Each Change)

## Dataset and run setup
- Fix random seeds.
- Use fixed prompt subset for timing benchmarks.
- Use same hardware allocation for A/B.

## Metrics to collect
1. Reward function wall time per call.
2. KataGo requests/sec.
3. Training steps/sec.
4. Rank-0 GPU utilization.
5. Rank-0 CPU utilization.
6. Eval winrate/Elo deltas from fixed eval procedure.

## Acceptance gates
1. Throughput improvement target per phase:
- Phase A: +20% steps/sec target.
- Phase D/E combined: +40-100% in reward throughput target (hardware dependent).

2. Quality guardrail:
- No significant regression beyond agreed threshold in eval winrate/Elo.

---

## 10) Phased Execution Plan (Recommended)

### Phase 1: Low-risk gains (1-2 days)
1. Implement base and candidate caches in move-level GRPO.
2. Introduce training-vs-eval KataGo config split.
3. Run benchmark A/B and verify quality guardrails.

### Phase 2: Medium complexity (2-5 days)
1. Add KataGo GPU pinning support.
2. Implement async `analyze_positions` client path.
3. Integrate in move-level GRPO reward function.
4. Benchmark and validate.

### Phase 3: High complexity scaling (3-7 days)
1. Implement KataGo worker pool.
2. Integrate with GRPO and full-game GRPO reward paths.
3. Benchmark throughput and system contention.

### Phase 4: Architecture decision
1. Reassess whether rank-0-only design is still limiting.
2. If needed, design distributed reward service.

---

## 11) Immediate Next 3 Changes to Implement First

1. **Caching in `run_grpo.py` reward function** (fastest safe speedup).
2. **Training/eval KataGo config split** (large speed leverage at low code risk).
3. **KataGo GPU pinning in `go_rl/katago.py`** (reduces contention and variance).

These three changes should be completed and benchmarked before async/pool refactors.

---

## 12) Risks and Mitigations

1. Reward quality drift from lighter search
- Mitigation: strict eval config retained; compare Elo/winrate after each change.

2. Async correctness bugs
- Mitigation: request-ID based mapping and exhaustive unit tests.

3. Cache memory growth
- Mitigation: bounded LRU caches and per-batch cache reset.

4. Process over-subscription in worker pool
- Mitigation: hard cap workers, CPU thread tuning, GPU affinity.

---

## 13) Reference Files

Core KataGo paths:
- `/Users/phoenix/go-test/go_rl/katago.py`
- `/Users/phoenix/go-test/scripts/run_grpo.py`
- `/Users/phoenix/go-test/scripts/run_grpo_fullgame.py`
- `/Users/phoenix/go-test/go_rl/fullgame_rollout.py`
- `/Users/phoenix/go-test/scripts/setup_environment.sh`

Config paths:
- `/Users/phoenix/go-test/config/grpo_9x9.yaml`
- `/Users/phoenix/go-test/config/grpo_19x19.yaml`
- `/Users/phoenix/go-test/config/grpo_fullgame_19x19.yaml`

Testing paths:
- `/Users/phoenix/go-test/tests/test_katago.py`

