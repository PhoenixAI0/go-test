# Plan: Reasoning Token Handling (SFT + GRPO, All Model Families)

## Summary
Create one production-complete implementation that lets models think freely while training/reward only depends on final answer tokens.  
This applies to:
1. SFT label masking (reasoning tokens masked out, final output tokens trained).
2. GRPO token loss masking (reasoning tokens excluded from policy loss).
3. Reward parsing (reasoning never rewarded/penalized directly).
4. Move-level and full-game paths.
5. Qwen-3, GPT-OSS, and GLM-4.7-Flash profiles under a shared abstraction.

This plan is intended to be written as a single file at:
`/Users/ruichen/go-test/PLAN/REASONING_TOKEN_HANDLING_IMPLEMENTATION_PLAN.md`

---

## Locked Decisions
1. Model scope: support **all three** families now (`qwen3`, `gpt_oss`, `glm47`).
2. SFT formatting: migrate to **chat-template conversational format**.
3. GRPO policy: reasoning must be **ignored for optimization target**; only final-answer tokens optimized.
4. Reward policy: reasoning text is **not graded** (no reward/penalty based on reasoning content itself).
5. Malformed reasoning boundary (started but not closed): **mask all completion tokens** and use deterministic fallback move-scoring behavior.
6. Delivery bar: **no stubs, no placeholders, no partial hooks**.

---

## Current-State Audit (from repo)
1. `scripts/run_sft.py` currently uses plain `prompt + response` and does not do reasoning-aware masking.
2. `scripts/run_grpo.py` and `scripts/run_grpo_fullgame.py` use vanilla `GRPOTrainer` with completion-wide masks, not post-reasoning masks.
3. Move-level parsing (`go_rl/rollout.py`) is first-token based and can penalize reasoning-prefixed completions.
4. Full-game parsing (`go_rl/fullgame_rollout.py` + `go_rl/prompting.py`) is more tolerant but still not explicit reasoning-aware optimization.
5. Config schema (`go_rl/config.py`) has no reasoning policy controls today.

---

## Research Findings to Encode
1. GPT-OSS commonly separates reasoning/final channels (Harmony-style structures).
2. Qwen-3 commonly uses reasoning tags (`<think>...</think>`) in thinking-capable variants; newer 2507 family includes explicit thinking/non-thinking variants.
3. GLM-4.7-Flash supports preserved thinking mode with separate reasoning/final content semantics.
4. Unsloth/TRL natively support role/completion-level masking but not guaranteed strict “post-reasoning-boundary only” across all paths without custom logic.

---

## Public API / Config Changes

## New dataclass
Add `ReasoningConfig` in `go_rl/config.py` with:
1. `enabled: bool = True`
2. `profile: str = "auto"` (`auto|qwen3|gpt_oss|glm47|plain`)
3. `optimize_final_only: bool = True`
4. `malformed_policy: str = "mask_all"` (`mask_all` only for now)
5. `qwen_enable_thinking: bool = True`
6. `gpt_oss_reasoning_effort: str = "medium"` (`low|medium|high`)
7. `glm_thinking_type: str = "enabled"` (`enabled|disabled`)
8. `glm_clear_thinking: bool = False`
9. `future_reasoning_length_penalty: float = 0.0` (declared, intentionally inactive)

## Attach config
1. `SftConfig.reasoning: ReasoningConfig`
2. `TrainingConfig.reasoning: ReasoningConfig` (used by GRPO + fullgame GRPO)

## YAML compatibility
All new fields optional with safe defaults so existing configs continue running.

---

## Implementation Design

## 1) Shared reasoning module
Create `go_rl/reasoning.py` implementing:
1. `detect_reasoning_profile(model_name: str, tokenizer: object, configured_profile: str) -> str`
2. `extract_final_answer(completion: object, profile: str) -> tuple[str, bool, bool]`
3. `strip_reasoning_for_reward(completion: object, profile: str) -> str`
4. `build_final_only_mask(...) -> list[int]` for token-level masking over completion tokens.
5. `apply_malformed_policy(...)` enforcing `mask_all`.

Behavior by profile:
1. `qwen3`: parse `<think>...</think>` style reasoning boundaries.
2. `gpt_oss`: parse structured channel output and use final channel only.
3. `glm47`: use final `content` while ignoring `reasoning_content` when present.
4. `plain`: no reasoning boundary; entire completion treated as final (backward fallback).

Malformed boundary behavior:
1. If reasoning starts but final boundary is missing: return empty final text + malformed flag.
2. Mask builder returns all zeros for completion mask in this case.

## 2) SFT path (strict final-only supervision)
Update `scripts/run_sft.py`:
1. Replace `format_example(prompt + response)` with conversational dataset mapping (`messages`).
2. Use chat template application path compatible with model tokenizer.
3. Ensure dataset/tokenization path can carry mask metadata.
4. Apply final-only label masking: reasoning tokens set to `-100`, only final-answer tokens remain trainable.
5. Keep malformed samples with all answer labels masked (`-100`) per policy.

## 3) GRPO trainer (strict final-only policy loss)
Create `go_rl/reasoning_grpo_trainer.py`:
1. Subclass TRL GRPO trainer.
2. Override completion-mask assembly to intersect default mask with reasoning final-only mask.
3. Preserve existing truncation and tool-call mask semantics.
4. Emit metrics:
   - masked_token_ratio
   - malformed_completion_count
   - final_span_mean_len

Wire trainer into:
1. `scripts/run_grpo.py`
2. `scripts/run_grpo_fullgame.py`

## 4) Reward integration (ignore reasoning text)
Update reward parse paths:
1. `scripts/run_grpo.py` reward function:
   - parse move from stripped final answer only.
   - unchanged legality/winrate mechanics on parsed final output.
2. `go_rl/fullgame_rollout.py`:
   - apply same stripping for initial completion and generated moves before parse.
3. No reasoning-length reward term now.

## 5) Model-specific runtime knobs
Expose profile options through generation/chat-template kwargs:
1. Qwen: `enable_thinking` handling where supported.
2. GPT-OSS: reasoning effort passthrough.
3. GLM: thinking mode + clear_thinking passthrough when serving stack supports it.

---

## File-by-File Change Plan

## New files
1. `go_rl/reasoning.py`
2. `go_rl/reasoning_grpo_trainer.py`
3. `tests/test_reasoning.py`
4. `tests/test_reasoning_masks.py`
5. `tests/test_reasoning_grpo_masking.py` (or equivalent trainer-focused integration test)

## Modified files
1. `go_rl/config.py`
2. `scripts/run_sft.py`
3. `scripts/run_grpo.py`
4. `scripts/run_grpo_fullgame.py`
5. `go_rl/fullgame_rollout.py`
6. Relevant config YAMLs in `/Users/ruichen/go-test/config/` to include `reasoning` block examples/defaults.

---

## Test Cases and Scenarios

## Unit tests
1. Qwen extraction:
   - `<think>analysis</think> D4` -> final `D4`, malformed false.
2. Qwen malformed:
   - `<think>analysis D4` -> malformed true, final empty.
3. GPT-OSS structured output:
   - analysis+final channels -> final extracted only.
4. GLM structured output:
   - reasoning_content + content -> content extracted.
5. Mask builder:
   - reasoning tokens masked 0, final tokens masked 1.
   - malformed -> all 0.

## Integration tests
1. SFT label masking:
   - verify labels are `-100` over reasoning span and valid on final move tokens only.
2. Move-level GRPO reward:
   - `D4` and reasoning-prefixed equivalent score identically when final move same.
3. Fullgame GRPO rollout:
   - reasoning-prefixed generated text parses/scored by final answer only.
4. GRPO trainer:
   - per-token loss mask excludes reasoning tokens.
5. Config loading:
   - new reasoning fields parse with defaults and explicit overrides.

## Smoke tests
1. `./scripts/train_sft.sh --board-size 9x9 --smoke`
2. `./scripts/train_grpo.sh --board-size 9x9 --smoke`
3. full-game smoke path using existing script/config entrypoint.

---

## No-Stubs / Full-Implementation Requirements

## Hard constraints
1. No `TODO`, `pass`, `NotImplementedError`, placeholder branches, or unused “future hook” code in changed files.
2. Any declared support path (`qwen3`, `gpt_oss`, `glm47`, malformed policy) must be fully wired and tested.
3. All new config fields must be consumed in runtime logic or removed.
4. Feature is not accepted if only partially enabled via flags without full behavior.

## Verification gate
Run grep checks on changed files before completion:
1. no placeholder tokens (`TODO|FIXME|NotImplementedError|pass #`).
2. no dead config fields (field declared but never referenced).

---

## Acceptance Criteria
1. SFT: reasoning tokens never contribute to loss; only final-output tokens train.
2. GRPO: reasoning tokens never contribute to policy loss; only final-output tokens optimize.
3. Reward: reasoning content itself is not rewarded or penalized.
4. Malformed reasoning boundary: zero training signal on completion, deterministic fallback scoring.
5. Existing non-reasoning outputs continue to work as before.
6. All new and touched tests pass.

---

## Assumptions and Defaults
1. Training stack will continue using current TRL/Unsloth-compatible environment.
2. Chat-template migration is mandatory for model-family-consistent behavior.
3. `future_reasoning_length_penalty` remains inactive now (value must be zero).
4. If model profile cannot be confidently detected under `auto`, fallback is `plain` with explicit warning log.

---

## External References
1. [Unsloth GPT-OSS docs](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune)
2. [Unsloth Qwen3 docs](https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune)
3. [Unsloth GLM docs](https://docs.unsloth.ai/models/z-ai-glm-4.5-how-to-run-and-fine-tune)
4. [Unsloth RL guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)
5. [Unsloth RL advanced docs](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation)
6. [OpenAI open-models announcement](https://openai.com/index/open-models/)
7. [OpenAI Harmony format cookbook](https://cookbook.openai.com/articles/openai-harmony)
8. [Qwen quickstart docs](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)
9. [GLM-4.7-Flash model card](https://huggingface.co/zai-org/GLM-4.7-Flash)
10. [Z.ai preserved thinking mode](https://docs.z.ai/guides/preserved_thinking_mode)
