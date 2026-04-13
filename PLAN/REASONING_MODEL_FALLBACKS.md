# Reasoning Model Fallbacks and Integration Findings

Date: 2026-04-13

## Scope

This note revises the single-GPU fallback list to include only reasoning-capable OSS models and evaluates how easily each model can be integrated into this repo's current SFT + GRPO stack.

Key constraints from the repo:

- Single-GPU LoRA / QLoRA flow via `unsloth.FastLanguageModel.from_pretrained(...)` and `FastLanguageModel.get_peft_model(...)`
- No model sharding; multi-GPU launch is DDP throughput scaling rather than model partitioning
- Reasoning support is routed through three local profiles only: `qwen3`, `gpt_oss`, and `glm47`
- RL grading strips reasoning before move parsing, but only when the raw completion format matches the extractor assumptions

Relevant local code:

- `go_rl/reasoning.py`
- `go_rl/reasoning_grpo_trainer.py`
- `scripts/run_grpo.py`
- `go_rl/fullgame_rollout.py`
- `go_rl/prompting.py`
- `go_rl/config.py`

## Revised Fallback Chain

This list is ordered as a practical single-GPU fallback ladder from most ambitious to safest small model.

1. `zai-org/GLM-4.5-Air`
2. `openai/gpt-oss-120b`
3. `Qwen/Qwen3-Next-80B-A3B-Thinking`
4. `Qwen/Qwen3-30B-A3B-Thinking-2507`
5. `Qwen/Qwen3-32B`
6. `Qwen/Qwen3-14B`
7. `Qwen/Qwen3-8B`

Why these remain:

- `GLM-4.5-Air` is an official hybrid reasoning model with thinking and non-thinking modes.
- `gpt-oss-120b` is an official reasoning model with configurable reasoning effort and confirmed single-GPU Unsloth QLoRA support.
- `Qwen3-Next-80B-A3B-Thinking` and `Qwen3-30B-A3B-Thinking-2507` are official thinking-only Qwen releases.
- `Qwen3-32B`, `Qwen3-14B`, and `Qwen3-8B` are official Qwen3 models with `enable_thinking=True` support.

Why some earlier candidates were removed:

- `gpt-oss-20b` was removed per request.
- Qwen Coder models were removed per request.
- Non-thinking Qwen variants such as `Qwen3-Next-80B-A3B-Instruct` and `Qwen3-30B-A3B-Instruct-2507` were removed because this list is now reasoning-only.
- `Qwen3-235B-A22B-Thinking-2507` was not kept because it is not a practical single-GPU fallback for this repo.

## External Source Notes

### GLM-4.5-Air

- Official model card: `106B total / 12B active`
- Official description: hybrid reasoning model with thinking and non-thinking modes
- Source: <https://huggingface.co/zai-org/GLM-4.5-Air>

### gpt-oss-120b

- Official model card: `117B / 5.1B active`
- Official description: reasoning model, Harmony-format outputs, single-80GB inference, single-H100 fine-tuning
- Unsloth documentation: QLoRA fine-tuning on about `65GB VRAM`
- Sources:
  - <https://huggingface.co/openai/gpt-oss-120b>
  - <https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune>
  - <https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning>

### Qwen3-Next-80B-A3B-Thinking

- Official model card: `80B total / 3B active`
- Official description: thinking-only; the default chat template may emit only `</think>` without an explicit opening `<think>` tag
- Source: <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking>

### Qwen3-30B-A3B-Thinking-2507

- Official model card: `30.5B total / 3.3B active`
- Official description: thinking-only; the default chat template may emit only `</think>` without an explicit opening `<think>` tag
- Source: <https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507>

### Qwen3-32B / 14B / 8B

- Official description: thinking is enabled by default with `enable_thinking=True`
- Official format: generates `<think>...</think>` followed by final response
- Sources:
  - <https://huggingface.co/Qwen/Qwen3-32B>
  - <https://huggingface.co/Qwen/Qwen3-14B>
  - <https://huggingface.co/Qwen/Qwen3-8B>

### Unsloth support

- Unsloth docs explicitly support Qwen3, Qwen3 MOE, gpt-oss, and GLM model families.
- Source: <https://unsloth.ai/docs/new>

## Integration Ease by Model

### 1. Qwen3-32B

Integration ease: Easy

Why:

- `go_rl/reasoning.py` already has a dedicated `qwen3` profile.
- `build_chat_template_kwargs(...)` and `build_generation_kwargs(...)` already pass `enable_thinking`.
- The current extractor expects `<think>...</think>` blocks, which matches the standard Qwen3 thinking-mode cards.

Main repo fit:

- Best drop-in replacement today if we want the least code churn and strongest confidence.

### 2. Qwen3-14B

Integration ease: Easy

Why:

- Same reasoning format and repo integration path as `Qwen3-32B`.
- Lower risk for memory and runtime than `Qwen3-32B`.

### 3. Qwen3-8B

Integration ease: Easy

Why:

- Same reasoning format and repo integration path as `Qwen3-32B`.
- Smallest and safest fallback baseline.

### 4. Qwen3-30B-A3B-Thinking-2507

Integration ease: Medium

Why:

- Auto-detection will still route this model to the existing `qwen3` profile because the model name contains `qwen`.
- Unsloth support for Qwen3 2507 is already public.
- The main blocker is output parsing: the official card says it is normal for output to contain only `</think>` without an explicit opening `<think>`.

Repo-specific risk:

- `_extract_qwen3(...)` currently only treats text as reasoning when it sees `<think>` first.
- If a completion arrives as `"reasoning ... </think> D4"` without an opening tag, current code will not strip the reasoning span.

Impact:

- RL grading can mis-score reasoning text as part of the move.
- GRPO final-only masking can keep too many tokens instead of masking the thinking prefix.

### 5. Qwen3-Next-80B-A3B-Thinking

Integration ease: Medium to High

Why:

- It is still Qwen-family and should route to the existing `qwen3` profile.
- Unsloth publicly states support for Qwen3 MOE models.
- Same suffix-only `</think>` issue as `Qwen3-30B-A3B-Thinking-2507`.

Additional risk:

- It uses the newer `qwen3_next` architecture, so repo compatibility depends on sufficiently new `transformers` and a working Unsloth path for that architecture.
- This is more likely to be a tooling/version issue than a reasoning-format issue.

### 6. gpt-oss-120b

Integration ease: Medium to High

Why:

- The repo already has a `gpt_oss` reasoning profile.
- `ReasoningConfig` already supports `gpt_oss_reasoning_effort`.
- Unsloth explicitly supports single-GPU QLoRA fine-tuning for `gpt-oss-120b`.

Main blocker:

- The official model card says `gpt-oss` should be used with Harmony format.
- The current repo prompt path uses a generic user-message chat template in `build_generation_inputs(...)`; that may be enough for prompting if the tokenizer template encodes Harmony, but the RL grading path still assumes plain decoded strings.
- `_extract_gpt_oss(...)` strips reasoning only when completions arrive as structured channel data such as `{"channel": "analysis"}` and `{"channel": "final"}`.
- In GRPO and full-game rollouts, completions are effectively plain decoded strings, so the `gpt_oss` extractor falls back to returning the whole string unchanged.

Impact:

- Current RL grading will not reliably remove `gpt-oss` reasoning traces.
- Current GRPO final-only masking will also be unreliable unless completions are converted back into Harmony-style structure or parsed from Harmony text.

### 7. GLM-4.5-Air

Integration ease: Medium to High

Why:

- The repo already has a `glm47` reasoning profile and config switches for `thinking_type` and `clear_thinking`.
- Unsloth publicly lists GLM family support.

Main blocker:

- `_extract_glm47(...)` only strips reasoning when completion arrives in a dict-like shape with separate `reasoning_content` and `content`.
- The current GRPO and full-game paths work on decoded text strings, not structured GLM response objects.

Impact:

- Current RL grading will not reliably strip GLM reasoning traces if the generated text is flattened into a plain string.
- Current final-only GRPO masking has the same limitation.

## Current Repo Behavior for Reasoning Trace Filtering

### What already exists

Move-level RL grading:

- `scripts/run_grpo.py` calls `strip_reasoning_for_reward(...)` before `normalize_move(...)`

Full-game RL grading:

- `go_rl/fullgame_rollout.py` calls `strip_reasoning_for_reward(...)` inside `_extract_reward_move(...)`

Final-only GRPO masking:

- `go_rl/reasoning_grpo_trainer.py` calls `build_final_only_mask(...)` and overwrites `completion_mask`

Qwen base thinking-mode support:

- `_extract_qwen3(...)` strips `<think>...</think>` blocks correctly

Existing tests:

- `tests/test_reasoning.py`
- `tests/test_reasoning_grpo_masking.py`

### Where the current logic is strong

- Standard Qwen3 models that emit full `<think>...</think>` blocks

### Where the current logic is weak

1. Qwen thinking-only 2507 / Next models

- Cards explicitly warn that the model may output only `</think>` with no opening `<think>`
- Current extractor does not handle that suffix-only format

2. gpt-oss

- Current extractor expects structured Harmony channels, but RL grading receives flattened strings

3. GLM-4.5-Air

- Current extractor expects separate `reasoning_content` and `content`, but RL grading receives flattened strings

## Recommended Changes Before Using the Harder Models

### Minimum parser fixes

1. Extend the `qwen3` extractor to support suffix-only `</think>` boundaries

Suggested rule:

- If text contains `</think>` but no `<think>`, treat everything before the last `</think>` as reasoning and everything after it as final answer.

This is the most important fix for:

- `Qwen3-30B-A3B-Thinking-2507`
- `Qwen3-Next-80B-A3B-Thinking`

2. Add a plain-text Harmony parser for `gpt_oss`

Suggested rule:

- Parse the decoded text emitted by the tokenizer/model and recover `analysis` vs `final` content before reward scoring and final-only masking.
- Do not rely on structured dict/list channels during GRPO, because the current trainer path mostly receives strings.

3. Add a plain-text fallback parser for `glm47`

Suggested rule:

- If a structured dict with `reasoning_content` and `content` is unavailable, split the decoded string using the official GLM reasoning format or its transformers reasoning parser conventions.

### Recommended tests

Add unit tests for:

1. Qwen suffix-only closing-tag output

- Example: `"long reasoning ... </think> D4"`

2. gpt-oss flattened Harmony text

- Example should verify that `strip_reasoning_for_reward(...)` returns only the final move.

3. GLM flattened text completion

- Example should verify that `strip_reasoning_for_reward(...)` returns only the final move.

4. GRPO masking for each supported family

- Verify `build_final_only_mask(...)` and `ReasoningGRPOTrainer` keep only final-answer tokens.

## Practical Recommendation Order

If we want the easiest near-term path with minimal repo changes:

1. `Qwen3-32B`
2. `Qwen3-14B`
3. `Qwen3-8B`

If we are willing to patch the Qwen parser for the newer thinking-only format:

4. `Qwen3-30B-A3B-Thinking-2507`
5. `Qwen3-Next-80B-A3B-Thinking`

If we are willing to add model-family-specific reasoning-text parsers:

6. `gpt-oss-120b`
7. `GLM-4.5-Air`

That ordering is about integration ease, not raw model quality.

## Bottom Line

- The revised reasoning-only fallback chain is valid, but the repo is not equally ready for all of those models.
- Today, base Qwen3 thinking models are the easiest drop-ins.
- Newer Qwen thinking-only releases are close, but need a parser fix for outputs that contain only `</think>`.
- `gpt-oss-120b` and `GLM-4.5-Air` are supported in principle, but their RL grading path is not production-safe yet because current reasoning stripping relies on structured outputs that GRPO does not currently preserve.
