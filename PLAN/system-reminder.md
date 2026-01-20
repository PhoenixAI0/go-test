# RL Plan: Train Qwen-3-8B to Play Go with GRPO + Unsloth

## Assumptions
- Hardware: 8x H100/H200 (80GB/141GB), multi-node OK.
- Output format: single GTP move coordinate (e.g., D4) only.
- Rollouts: partial rollouts with engine evaluation, plus periodic full-game outcomes.
- Inference: vLLM for rollout generation.
- Reward signals: engine winrate delta + full-game outcome bonus.
- Safety: reward hacking guardrails required.

## Recommended Environment
- KataGo as engine/evaluator (strong, open-source, supports 9x9/19x19, rich tooling).

## Phase 0: Infra & Environment Setup
1) Install stack
   - Unsloth + vLLM + PyTorch nightly/2.x + CUDA 12.x.
   - KataGo binary + nets for evaluation.
2) Build rollout service
   - vLLM serving Qwen-3-8B for fast sampling.
   - KataGo running in analysis mode for batch evaluations.
3) Data plumbing
   - Position sampler (from SGF and self-play positions).
   - State serializer (board -> model prompt with consistent template).

## Phase 1: Minimal Format SFT (Warm Start)
Goal: teach format + legality without prompting.
1) Dataset
   - Extract positions from public SGF (KGS/OGS/etc.).
   - For each position, label top-1 KataGo move.
   - Keep data minimal (e.g., 50k–200k positions) to avoid overfitting.
2) SFT
   - Train on “position -> GTP move” pairs.
   - Strict format: output single coordinate only.
3) Validation
   - Format compliance rate (>99%).
   - Legal move rate (>99%).
   - Basic policy accuracy vs KataGo top-1.

## Phase 2: RL w/ GRPO on 9x9
Goal: improve policy via RLVR-style rewards.
1) Rollout
   - Sample N moves per position (e.g., 8–16) using vLLM.
   - Use KataGo analysis to score each move’s winrate delta.
2) Reward shaping
   - R = +winrate_delta (scaled).
   - +legal move bonus, -illegal move penalty.
   - Small length penalty if output not minimal.
   - Periodic endgame reward: +1/-1 for full-game outcome vs baseline.
3) GRPO config
   - Use Unsloth GRPO with multiple generations per prompt.
   - Advantage: group-relative Z-score across samples.
4) Safety/anti-hacking
   - Verify output regex strictly.
   - Penalize invalid coordinates or illegal moves.
   - Filter repeated or pass-spam.
   - Cross-check rule consistency (komi, ruleset).
5) Eval
   - Elo vs baseline on 9x9 (self-play + KataGo match engine).

## Phase 3: Curriculum to 19x19
Gate: Elo +200 vs baseline on 9x9 (adjustable).
1) Mixed-board training
   - Gradually mix 9x9/19x19 (e.g., 70/30 -> 50/50 -> 20/80).
2) Update KataGo configs
   - Ruleset/komi standardized (e.g., Chinese, 7.5).
3) Continue GRPO with same reward scheme.

## Phase 4: 19x19 Scaling
1) Increase rollout depth
   - Longer context for board serialization.
   - More samples per position if stable.
2) Evaluate
   - Elo vs baseline and vs fixed KataGo net.
   - Track legality, resignation policy, and winrate stability.

## Phase 5: Logging & Reproducibility
1) Log
   - Reward components per step.
   - Elo curve per checkpoint.
   - Rollout stats (entropy, illegal rate).
2) Checkpointing
   - Save at each curriculum milestone.
   - Retain best-performing snapshot for evaluation.

## Milestone Deliverables
- Warm-start SFT model with high legality + format compliance.
- 9x9 GRPO model with Elo improvement over baseline.
- 19x19 curriculum model with stable winrate gains.

## Notes & References
- Unsloth GRPO RL guide: https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide
- Unsloth GRPO long-context blog: https://unsloth.ai/blog/grpo
- KataGo engine: https://github.com/lightvector/KataGo
- DeepSeek-R1 paper on RL reasoning: https://arxiv.org/abs/2501.12948
