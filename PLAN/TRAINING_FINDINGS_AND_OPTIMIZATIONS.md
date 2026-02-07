# Training Findings and Optimization Plan

## Scope
This document consolidates findings from the current codebase about:

1. End-to-end training implementation (SFT, GRPO, full-game GRPO).
2. How Unsloth is used.
3. Distributed training behavior (Accelerate/DDP vs alternatives).
4. Precision / quantization behavior.
5. LoRA vs full fine-tuning feasibility.
6. KataGo reward-path bottlenecks and optimization opportunities.
7. Compile-path status and performance implications.
8. Prioritized optimization roadmap.

Repository root: `/Users/phoenix/go-test`

---

## 1) High-Level Pipeline (Current Implementation)

Top-level orchestration script:
- `/Users/phoenix/go-test/scripts/run_full_pipeline.sh`

Default stage order:
1. Environment setup.
2. Dataset preparation.
3. SFT 9x9 warm start.
4. Eval vs KataGo (optional).
5. GRPO 9x9.
6. Promotion-gate eval.
7. GRPO 19x19.
8. Promotion-gate eval for full-game stage.
9. Full-game GRPO 19x19 (conditional).
10. Final eval.
11. Optional HF upload.

Wrapper scripts:
- SFT wrapper: `/Users/phoenix/go-test/scripts/train_sft.sh`
- GRPO wrapper: `/Users/phoenix/go-test/scripts/train_grpo.sh`
- Full-game GRPO wrapper: `/Users/phoenix/go-test/scripts/train_grpo_fullgame.sh`

Python entrypoints:
- SFT: `/Users/phoenix/go-test/scripts/run_sft.py`
- GRPO: `/Users/phoenix/go-test/scripts/run_grpo.py`
- Full-game GRPO: `/Users/phoenix/go-test/scripts/run_grpo_fullgame.py`

---

## 2) Data Pipeline and Training Data Shape

### 2.1 SGF parsing and dataset build

Primary parser path is Go-based (Python SGF parser is deprecated):
- Deprecated SGF Python path: `/Users/phoenix/go-test/go_rl/sgf_utils.py`
- Go parser binary and implementation:
  - `/Users/phoenix/go-test/go_parser/main.go`
  - `/Users/phoenix/go-test/go_parser/dataset.go`
  - `/Users/phoenix/go-test/go_parser/sgf.go`
  - `/Users/phoenix/go-test/go_parser/board.go`

Dataset builder:
- `/Users/phoenix/go-test/scripts/prepare_datasets.py`

### 2.2 Dataset schema consumed by trainers

Go parser emits records with:
- `prompt`
- `response`
- `board_size`

In `prepare_datasets.py`, these are split into:
- **SFT dataset** (`data/sft_<tag>.jsonl`): keeps `prompt`, `response`, `board_size`
- **RL prompts dataset** (`data/rl_prompts_<tag>.jsonl`): keeps only `prompt`

Reference:
- `/Users/phoenix/go-test/scripts/prepare_datasets.py:90`
- `/Users/phoenix/go-test/scripts/prepare_datasets.py:95`

Implication:
- SFT is supervised next-token training on prompt+answer pairs.
- GRPO generates completions online and scores them via reward function.

---

## 3) SFT Implementation Details

Entrypoint:
- `/Users/phoenix/go-test/scripts/run_sft.py`

Configuration:
- `/Users/phoenix/go-test/config/sft_warm_start.yaml`
- `/Users/phoenix/go-test/config/sft_warm_start_19x19.yaml`
- Dataclass validation: `/Users/phoenix/go-test/go_rl/config.py` (`SftConfig`)

### 3.1 Core flow

1. Parse CLI flags and load config.
2. Import `datasets`, TRL `SFTTrainer`, and Unsloth `FastLanguageModel`.
3. Configure torch speed settings through shared utility.
4. Load model via Unsloth `FastLanguageModel.from_pretrained(...)`.
5. Attach LoRA via `FastLanguageModel.get_peft_model(...)`.
6. Load JSON dataset and optional smoke truncation.
7. Optional train/eval split (`eval_split_ratio`).
8. Build TRL `SFTConfig` and `SFTTrainer`.
9. Train, optional eval, log W&B metrics/perplexity.
10. Save model to configured output dir.

Key references:
- Unsloth model load: `/Users/phoenix/go-test/scripts/run_sft.py:164`
- LoRA attach: `/Users/phoenix/go-test/scripts/run_sft.py:176`
- Train call: `/Users/phoenix/go-test/scripts/run_sft.py:278`
- Save model: `/Users/phoenix/go-test/scripts/run_sft.py:309`

### 3.2 Formatting function

SFT examples are concatenated as:
- `prompt + "\n" + response`

Reference:
- `/Users/phoenix/go-test/scripts/run_sft.py:30`

---

## 4) GRPO Implementation Details (Move-Level)

Entrypoint:
- `/Users/phoenix/go-test/scripts/run_grpo.py`

Configs:
- 9x9: `/Users/phoenix/go-test/config/grpo_9x9.yaml`
- 19x19: `/Users/phoenix/go-test/config/grpo_19x19.yaml`
- Dataclass validation: `/Users/phoenix/go-test/go_rl/config.py` (`TrainingConfig`, `KataGoConfig`, `RewardConfig`, `RolloutConfig`)

### 4.1 Core flow

1. Parse args and load YAML sections into typed configs.
2. Import `datasets`, TRL `GRPOTrainer`, Unsloth model loader.
3. Configure torch speed settings.
4. Load model via Unsloth (with `fast_inference=True`).
5. Optionally attach LoRA.
6. Load prompt-only RL dataset.
7. Start KataGo only on rank 0.
8. Construct reward function that parses prompt/completion, calls KataGo, computes reward.
9. Create `GRPOConfig` and `GRPOTrainer`.
10. Train, log reward/loss metrics, save model, close KataGo.

Key references:
- Unsloth load with fast inference: `/Users/phoenix/go-test/scripts/run_grpo.py:271`
- LoRA attach: `/Users/phoenix/go-test/scripts/run_grpo.py:285`
- Rank check and rank-0 KataGo startup: `/Users/phoenix/go-test/scripts/run_grpo.py:321`, `/Users/phoenix/go-test/scripts/run_grpo.py:333`
- Trainer construction: `/Users/phoenix/go-test/scripts/run_grpo.py:410`
- Train call: `/Users/phoenix/go-test/scripts/run_grpo.py:421`

### 4.2 Reward function mechanics

Defined by `build_reward_fn(...)` in:
- `/Users/phoenix/go-test/scripts/run_grpo.py:47`

Per `(prompt, completion)`:
1. Parse board from prompt text.
2. Determine player-to-move.
3. Analyze base position using KataGo.
4. Normalize completion into a move string.
5. Validate coordinates and legality.
6. If legal: apply move on copied board and re-analyze with KataGo.
7. Compute winrate delta and feed to reward shaping.
8. If invalid/illegal/failed analysis: fallback penalties still computed.

References:
- Prompt parse: `/Users/phoenix/go-test/go_rl/prompting.py:10`
- Move normalization: `/Users/phoenix/go-test/go_rl/rollout.py:49`
- Legality + reward breakdown: `/Users/phoenix/go-test/go_rl/rewards.py:74`, `/Users/phoenix/go-test/go_rl/rewards.py:114`
- KataGo client analysis call: `/Users/phoenix/go-test/go_rl/katago.py:103`

---

## 5) Full-Game GRPO Variant

Entrypoint:
- `/Users/phoenix/go-test/scripts/run_grpo_fullgame.py`

Config:
- `/Users/phoenix/go-test/config/grpo_fullgame_19x19.yaml`

Difference vs move-level GRPO:
- Reward is based on simulated continuation of a game (model + KataGo opponent), not just single-move delta.
- Uses `simulate_full_game(...)`:
  - `/Users/phoenix/go-test/go_rl/fullgame_rollout.py:140`

Notable behavior:
- Configurable truncation limits (`max_plies`, `max_model_turns`).
- Alternating model color possible.
- Terminal reward from final scorelead sign.
- Illegal move can terminate episode depending on config.

---

## 6) How Unsloth Is Used

Unsloth appears in all training entrypoints through:

1. `FastLanguageModel.from_pretrained(...)` for model loading.
2. `FastLanguageModel.get_peft_model(...)` for LoRA adaptation.
3. Unsloth-specific GRPO memory knobs passed into TRL config fields.
4. Unsloth-style gradient checkpointing mode (`"unsloth"`).

References:
- SFT model load and LoRA: `/Users/phoenix/go-test/scripts/run_sft.py:164`, `/Users/phoenix/go-test/scripts/run_sft.py:176`
- GRPO model load and LoRA: `/Users/phoenix/go-test/scripts/run_grpo.py:271`, `/Users/phoenix/go-test/scripts/run_grpo.py:285`
- GRPO Unsloth memory fields: `/Users/phoenix/go-test/scripts/run_grpo.py:395`
- Unsloth gradient checkpointing mode (SFT/GRPO):
  - `/Users/phoenix/go-test/scripts/run_sft.py:259`
  - `/Users/phoenix/go-test/scripts/run_grpo.py:405`

---

## 7) Distributed Training: Accelerate/DDP vs Torchrun

Current wrapper behavior:
- Uses `accelerate launch --config_file=...` when `--multi-gpu` is set.
- Wrapper references:
  - `/Users/phoenix/go-test/scripts/train_sft.sh:161`
  - `/Users/phoenix/go-test/scripts/train_grpo.sh:181`

Accelerate config files specify DDP-style multi-GPU with fp16 mixed precision:
- `/Users/phoenix/go-test/config/accelerate/ddp_2gpu.yaml`
- `/Users/phoenix/go-test/config/accelerate/ddp_4gpu.yaml`
- `/Users/phoenix/go-test/config/accelerate/ddp_8gpu.yaml`

Observation:
- This is a launcher/config management choice. Torchrun could also launch DDP, but current repo standardizes on Accelerate YAMLs.

Important GRPO constraint:
- Reward path is rank-0 only; this can cap scaling efficiency regardless of launcher.

---

## 8) Precision and Quantization (Current)

### 8.1 Model load quantization

Handled by `quantization_kwargs(...)`:
- `/Users/phoenix/go-test/go_rl/training_utils.py:10`

Modes:
- Default in configs: 8-bit.
- Optional CLI toggles: dynamic 4-bit and fp8-style bnb mode.

Default config examples:
- SFT default 8-bit: `/Users/phoenix/go-test/config/sft_warm_start.yaml:14`
- GRPO default 8-bit: `/Users/phoenix/go-test/config/grpo_9x9.yaml:19`

### 8.2 Mixed precision under multi-GPU launch

Accelerate YAMLs set:
- `mixed_precision: fp16`

References:
- `/Users/phoenix/go-test/config/accelerate/ddp_2gpu.yaml:5`
- `/Users/phoenix/go-test/config/accelerate/ddp_4gpu.yaml:5`
- `/Users/phoenix/go-test/config/accelerate/ddp_8gpu.yaml:5`

### 8.3 Additional math/backends

`configure_torch_for_speed(...)` sets:
- TF32 matmul enabled.
- cuDNN benchmark enabled.
- high matmul precision preference.
- logs selected attention backend.

Reference:
- `/Users/phoenix/go-test/go_rl/training_utils.py:42`

---

## 9) Base Model Resolution / Download Behavior

Current SFT config uses:
- `model_name: qwen-3-8b`
- `/Users/phoenix/go-test/config/sft_warm_start.yaml:2`

GRPO 9x9 starts from local SFT output:
- `training.model_name: outputs/sft_warm_start`
- `/Users/phoenix/go-test/config/grpo_9x9.yaml:3`

Download behavior:
- There is no explicit predownload step in repo scripts; model retrieval happens through `from_pretrained(...)` path when not already cached.

Open decision (deferred in conversation):
- Exact base/instruct model selection should be explicitly revisited and pinned to a fully qualified model identifier.

---

## 10) LoRA vs Full Fine-Tuning Feasibility

## 10.1 Why LoRA is currently the practical default

Current stack is strongly designed for parameter-efficient finetuning:
- `use_lora: true` across GRPO configs.
- 8-bit loading by default.
- 8-bit optimizer enabled by default.
- Unsloth memory-saving GRPO knobs used.

This minimizes memory and improves stability for expensive RL loops.

## 10.2 Does LoRA work for GRPO here?

Yes. It is the implemented and tested default path:
- LoRA-wrapped model is passed to `GRPOTrainer` in `/Users/phoenix/go-test/scripts/run_grpo.py`.

## 10.3 Is full fine-tuning feasible?

Feasible on sufficient hardware (for example H100s), but not a trivial toggle in this repo:
- Requires disabling LoRA and likely changing quantization/optimizer strategy.
- DDP-only may become limiting for full FT at long context and larger global batch.
- For higher-scale full FT, FSDP/ZeRO-style sharding is typically considered.

## 10.4 Would full FT necessarily be better quality?

Not guaranteed. Potentially higher ceiling, but depends on:
- Reward signal quality/noise.
- RL stability and hyperparameters.
- Effective sample budget.
- Distributed efficiency and bottlenecks.

Recommendation:
- Keep LoRA for near-term velocity.
- Run targeted ablations before committing to full-FT migration.

---

## 11) KataGo Bottleneck Analysis

### 11.1 Current architecture

KataGo is launched as one analysis subprocess on rank 0:
- `/Users/phoenix/go-test/scripts/run_grpo.py:333`

Client request path is synchronized with a lock and blocking request-response:
- `/Users/phoenix/go-test/go_rl/katago.py:93`
- `/Users/phoenix/go-test/go_rl/katago.py:97`

### 11.2 Is KataGo GPU-accelerated currently?

Environment setup installs CUDA KataGo binary (`cuda11.8` artifact):
- `/Users/phoenix/go-test/scripts/setup_environment.sh:91`

So yes, GPU-enabled binary is intended.

### 11.3 Why throughput can still be poor

1. Reward path is rank-0 only.
2. Single KataGo process.
3. Blocking one-request-at-a-time client behavior.
4. Potential repeated analyses for repeated prompts/completions.
5. Training and KataGo may compete for same GPU if not explicitly separated.

### 11.4 Current KataGo config defaults

Generated `analysis.cfg` includes:
- `numSearchThreads = 8`
- `numNNServerThreadsPerModel = 2`
- `nnMaxBatchSize = 16`
- `maxVisits = 100`
- `maxPlayouts = 300`

Reference:
- `/Users/phoenix/go-test/scripts/setup_environment.sh:100`

These are quality-leaning settings; they may be expensive for reward-only scoring loops.

---

## 12) Compile Path Status

CLI exposes compile flag in trainers:
- `--enable-compile` in SFT and GRPO scripts.

However, current utility only logs compile intent and does not call `torch.compile(model)`.
- `/Users/phoenix/go-test/go_rl/training_utils.py:81`

Conclusion:
- Compile is not fully implemented for actual model graph compilation today.
- Current speedups from this flag are unlikely to materialize as expected.

---

## 13) Optimization Opportunities (Comprehensive)

## 13.1 Highest impact, lowest risk (do first)

1. **Base position caching in GRPO reward_fn**
- Cache KataGo analysis for identical prompt-derived board states within a training step.
- Avoid duplicate `base_analysis` calls for repeated prompts.
- Expected impact: high in multi-generation GRPO.

2. **Candidate move caching**
- Cache `(board_state_hash, move)` -> candidate analysis result.
- Expected impact: medium-high when completions repeat.

3. **Reduce KataGo search budget for reward path**
- Lower `maxVisits` and/or `maxPlayouts` for training-time reward estimation.
- Keep stronger settings for evaluation matches.
- Expected impact: high, with quality/noise tradeoff to validate.

4. **Separate KataGo onto dedicated GPU**
- Reduce contention between training kernels and reward inference.
- Expected impact: medium-high depending on hardware.

## 13.2 Medium effort, high potential

5. **Asynchronous request pipeline to KataGo**
- Queue multiple requests before waiting for results.
- Exploit KataGo’s request IDs to improve throughput.

6. **Multiple KataGo worker processes**
- Shard reward requests across workers (possibly per GPU).
- Requires careful process management and aggregation logic.

7. **Improve reward parallel architecture beyond rank-0 bottleneck**
- This is the key scaling limiter in multi-GPU GRPO today.
- Requires nontrivial redesign to preserve determinism and correctness.

## 13.3 Training stack improvements

8. **True compile implementation**
- Implement real `torch.compile(model)` path with safe fallback.
- Benchmark per stage (SFT vs GRPO) because RL workload may benefit less.

9. **Attention backend explicit tuning by hardware**
- Validate `sdpa` vs `flash_attention_2` with actual throughput tests.
- Current logic is heuristic-based.

10. **Dataloader tuning**
- Validate worker counts and pin-memory effects under real loads.

## 13.4 If/when pursuing full fine-tuning

11. **Add full-FT profile configs**
- Explicit non-LoRA configs, precision strategy, optimizer settings.

12. **Adopt stronger sharding strategy for full FT**
- Evaluate FSDP or ZeRO when memory/throughput demand exceeds DDP comfort.

13. **Structured ablation framework**
- Compare LoRA vs full FT under equalized compute budgets.

---

## 14) Prioritized Execution Plan

### Phase 1 (fast wins)
1. Add base + candidate KataGo caching in GRPO reward path.
2. Introduce separate training vs evaluation KataGo configs.
3. Tune training KataGo visits/playouts downward and benchmark reward-noise impact.
4. Pin KataGo to dedicated GPU where available.

### Phase 2 (scaling wins)
5. Implement async/batched KataGo request handling.
6. Add multi-worker KataGo service mode.
7. Measure multi-GPU scaling efficiency before/after rank-0 bottleneck mitigations.

### Phase 3 (quality frontier)
8. Establish controlled LoRA vs full-FT ablation matrix.
9. If justified, introduce full-FT distributed strategy changes.

---

## 15) Metrics to Track for Every Optimization

1. **Throughput**
- training steps/sec
- reward evaluations/sec
- KataGo requests/sec

2. **Utilization**
- GPU utilization for training process
- GPU utilization for KataGo process
- rank-0 vs non-rank idle time

3. **Quality**
- eval winrate vs KataGo
- Elo delta from eval pipeline
- legal-move rate

4. **Stability**
- reward variance across batches
- loss spikes/divergence events
- run failure frequency

---

## 16) Open Decisions

1. **Base model selection (deferred)**
- Revisit and pin exact model identity (base vs instruct variant).
- Use fully qualified model identifier in configs.

2. **Long-term strategy**
- Continue optimizing LoRA+GRPO stack first, then reassess full FT only after plateau evidence.

---

## 17) Key File Index

Training entrypoints:
- `/Users/phoenix/go-test/scripts/run_sft.py`
- `/Users/phoenix/go-test/scripts/run_grpo.py`
- `/Users/phoenix/go-test/scripts/run_grpo_fullgame.py`

Wrappers and launch:
- `/Users/phoenix/go-test/scripts/train_sft.sh`
- `/Users/phoenix/go-test/scripts/train_grpo.sh`
- `/Users/phoenix/go-test/config/accelerate/single_gpu.yaml`
- `/Users/phoenix/go-test/config/accelerate/ddp_2gpu.yaml`
- `/Users/phoenix/go-test/config/accelerate/ddp_4gpu.yaml`
- `/Users/phoenix/go-test/config/accelerate/ddp_8gpu.yaml`

Core libraries:
- `/Users/phoenix/go-test/go_rl/config.py`
- `/Users/phoenix/go-test/go_rl/training_utils.py`
- `/Users/phoenix/go-test/go_rl/katago.py`
- `/Users/phoenix/go-test/go_rl/rewards.py`
- `/Users/phoenix/go-test/go_rl/prompting.py`
- `/Users/phoenix/go-test/go_rl/rollout.py`
- `/Users/phoenix/go-test/go_rl/fullgame_rollout.py`

Data prep:
- `/Users/phoenix/go-test/scripts/prepare_datasets.py`
- `/Users/phoenix/go-test/go_parser/dataset.go`

Setup and KataGo defaults:
- `/Users/phoenix/go-test/scripts/setup_environment.sh`

