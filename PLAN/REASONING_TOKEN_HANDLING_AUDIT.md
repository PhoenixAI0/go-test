# Reasoning Token Handling Audit (SFT + GRPO)

## Scope
This audit answers:

- Does the current codebase support "reasoning mode" where `<think>...</think>` tokens are ignored?
- Do we train only on final outputs (moves), or also on reasoning text?

The analysis covers dataset generation, SFT training, GRPO training (move-level and full-game), and the actual masking/loss behavior used by TRL + Unsloth in this repo.

---

## Executive Answer
Short answer: **No, the codebase does not implement explicit reasoning-tag masking today.**

From a Unsloth-native capability perspective (validated against docs + installed source):

- **SFT**: Unsloth does provide `train_on_responses_only(...)`, which natively masks non-assistant spans.
- **SFT with "train only after `</think>` inside assistant turn"**: not a first-class one-flag feature; this requires custom masking unless your data/template keeps reasoning outside the supervised span.
- **GRPO**: current GRPO path masks by completion span (and optionally truncated completions), not by "post-`</think>` token span" out of the box.

- In **SFT**, there is no `<think>`-specific stripping or mask logic in repo code.
- In **GRPO**, there is no `<think>`-specific masking in repo code; GRPO loss is computed over completion tokens.
- In **move-level GRPO** (`scripts/run_grpo.py`), reasoning text before the move can actively hurt reward because move extraction uses first-token normalization.
- In **full-game GRPO** (`scripts/run_grpo_fullgame.py` + `go_rl/fullgame_rollout.py`), move extraction is more robust and can recover a move embedded in reasoning text, but completion tokens are still part of GRPO optimization.

So the behavior is **not** "ignore thinking tags and only train on final move output".

---

## What The Dataset Actually Contains

### SFT labels are SGF next-move targets (not reasoning text)
The Go parser writes records as:

- `prompt`: board state text
- `response`: next move in GTP format (e.g., `D4`, `PASS`)

Evidence:

- `go_parser/dataset.go:15-20` (`Sample` schema)
- `go_parser/dataset.go:191-204` (`Response: gtp`)
- `go_parser/dataset.go:218-241` (JSONL writer)
- `scripts/prepare_datasets.py:90-95` (SFT gets `prompt` + `response`, RL gets prompt only)

Prompt format itself is deterministic board text ending with `move:`:

- `go_rl/board.py:231-237`

No reasoning field is present in dataset generation.

---

## Rationale for Ignoring Thinking Tags

We evaluate the model based on its outputs rather than its reasoning. We do not train on/evaluate the thought traces because:
- It is difficult to generate thought traces for these games without spending significant computation in distilling from a larger model
- We would like the model to learn its own thought traces and figure out methods to reason through game-playing strategies by itself.
Right now by doing SFT without considering the thought tags, we eventually train the model to stop thinking, which is not ideal for performance.
Instead, we should mask the thinking traces and only grade the output for SFT./

---

## Unsloth Support + Implementation Design (What To Build)

This section is implementation-first: what Unsloth gives natively, what it does not, and exactly how that maps to this repo.

### 1) What Unsloth natively supports for this requirement

1. **Assistant-only SFT masking exists natively**
   - Source: `unsloth_zoo/dataset_utils.py:184-385` (`train_on_responses_only`)
   - It locates instruction/response template token boundaries, then writes labels as `-100` outside assistant response spans.
   - It updates mapped datasets and data collator automatically.

2. **Reasoning-aware chat templates exist**
   - Source: `unsloth/chat_templates.py:1036-1040`, `1814-1817`, `1966-1968`.
   - Templates can parse `</think>` and split reasoning vs content at formatting time.
   - `enable_thinking` support exists in templates (`unsloth/chat_templates.py:1085-1087`) and docs for Qwen3 usage.

3. **GRPO token masking is completion-level by default**
   - Source: `trl/trainer/grpo_trainer.py:1382-1385`, `1730-1738`.
   - `completion_mask` is initialized to 1 for all generated completion tokens, then used in GRPO loss.
   - `mask_truncated_completions` can zero whole clipped completions (`trl/trainer/grpo_trainer.py:1391-1395`), but this is not a `</think>` boundary mask.

### 2) What is not native (and matters for your policy)

Target policy:
- "Model may emit `<think> ... </think>`."
- "Only supervise/optimize final output tokens after end-thinking token."

Gap:
- Unsloth native SFT helper masks by role/segment boundaries, not by an internal delimiter inside the assistant segment.
- GRPOTrainer masks full completion, not "suffix after delimiter" unless you customize mask construction.

### 3) Implementation blueprint for this repo

### SFT path (scripts/run_sft.py)

Current path:
- Uses plain `format_example = prompt + "\\n" + response` (`scripts/run_sft.py:30-31`).
- Instantiates `SFTTrainer` directly (`scripts/run_sft.py:267-274`).
- No Unsloth `train_on_responses_only` call.

Implementation options:

1. **Minimum-change option (native + data design)**
   - Keep reasoning out of the supervised assistant target text entirely.
   - Format so assistant target contains only final move.
   - Use `train_on_responses_only(...)` to ignore prompt/instruction.
   - Result: no learning signal on reasoning tokens because they are absent from supervised label text.

2. **Strict "post-`</think>` only" option (custom label mask)**
   - Keep assistant text as `<think>...</think>FINAL`.
   - After tokenization, build label mask that sets labels `-100` up to and including the end-think token sequence.
   - Keep labels only for suffix tokens (final output).
   - This is the exact behavior you described and is the robust way if reasoning tokens are present in target text.

Implementation insertion point:
- Add mask transform after dataset tokenization / before trainer consumption in `scripts/run_sft.py`.
- If using Unsloth helper as base, apply helper first for assistant-only masking, then apply per-example end-think masking.

### GRPO path (scripts/run_grpo.py, scripts/run_grpo_fullgame.py)

Current path:
- `GRPOTrainer(... reward_funcs=...)` only (`scripts/run_grpo.py:503-509`, `scripts/run_grpo_fullgame.py:475-489`).
- No per-token reasoning mask argument is passed.
- Move-level reward currently uses first-token normalization (`scripts/run_grpo.py:139-141` with `go_rl/rollout.py:49-58`), which penalizes reasoning-prefixed completions.
- Full-game path can parse moves from anywhere in text (`go_rl/fullgame_rollout.py:86-97`, `go_rl/prompting.py:36-45`).

Implementation options:

1. **Reward-only gating (no trainer subclass)**
   - In reward function, split completion at `</think>`.
   - Score only suffix text.
   - Add explicit penalty if end-think token missing or final move not parseable.
   - Pros: simple, low risk.
   - Cons: policy gradient still applies to all completion tokens under shared advantage, so this is not strict token-level optimization isolation.

2. **Strict token-level GRPO masking (trainer customization)**
   - Subclass/patch GRPO training step to build custom `completion_mask`:
     - 0 for tokens before `</think>` and delimiter itself
     - 1 for tokens after delimiter
   - Keep existing reward function parsing on suffix.
   - This gives true "optimize final output tokens only" behavior in policy loss.
   - This is the GRPO-equivalent of strict SFT suffix masking.

Implementation insertion points:
- Either wrap `GRPOTrainer` in a local subclass module and swap constructor in:
  - `scripts/run_grpo.py:503-509`
  - `scripts/run_grpo_fullgame.py:475-489`
- Or apply a local patched trainer utility imported by both scripts.

### 4) Why current behavior diverges from your intended policy

1. SFT data mostly has short move targets, so the model learns brevity by distribution, not by explicit "ignore think span" masking.
2. Move-level GRPO can punish reasoning-prefixed outputs because parser is first-token based.
3. Full-game GRPO is parsing-tolerant, but GRPO optimization mask is still completion-global.

### 5) Practical acceptance criteria for the implementation

1. Given completion `<think>analysis</think> D4`, supervised/GRPO loss should be non-zero only on tokens representing `D4` (and any intentional trailing format tokens), not on `<think>analysis</think>`.
2. Given completion missing `</think>`, reward path should enforce deterministic fallback/penalty behavior.
3. Given purely final output (`D4`), behavior should remain valid and equivalent to baseline parser/reward path.

### 6) Web + source references used

- Unsloth troubleshooting/FAQ (`train_on_responses_only` usage):
  - https://docs.unsloth.ai/basics/troubleshooting-and-faqs
- Unsloth RL guide:
  - https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide
- Unsloth advanced RL docs (`mask_truncated_completions`):
  - https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation
- Unsloth reasoning/Qwen docs (`enable_thinking` usage):
  - https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune
- Local installed sources (line-cited above):
  - `/Users/phoenix/Library/Python/3.13/lib/python/site-packages/unsloth_zoo/dataset_utils.py`
  - `/Users/phoenix/Library/Python/3.13/lib/python/site-packages/unsloth/chat_templates.py`
  - `/Users/phoenix/Library/Python/3.13/lib/python/site-packages/trl/trainer/grpo_trainer.py`

---

## SFT Implementation: Are Thinking Tokens Ignored?

### Local training script behavior
SFT constructs one plain text sequence:

- `format_example = prompt + "\n" + response`
- `scripts/run_sft.py:30-31`

Trainer is instantiated with:

- `SFTTrainer(..., formatting_func=format_example, args=SFTConfig(...))`
- `scripts/run_sft.py:267-274`

There is no local `<think>` parsing/stripping/masking in SFT script.

### TRL behavior actually used here (from local installed `trl==0.24.0`)
In this environment, SFT default behavior depends on dataset schema:

- `completion_only_loss` is auto-enabled only when dataset has `prompt` + `completion` keys.
- `.../trl/trainer/sft_trainer.py:733-739`

Your SFT pipeline uses `formatting_func`, which maps each row to `{"text": ...}`:

- `.../sft_trainer.py:903-910`

Then tokenization goes through language-modeling path (`input_ids` from `text`) rather than prompt-completion mask path:

- `.../sft_trainer.py:1002-1019`

Masking in the data collator only applies `completion_mask` if such a mask exists:

- `.../sft_trainer.py:176-177`, `211-216`

For this run path, no `completion_mask` is created, so there is no special final-answer-only masking.

### Practical implication for SFT
Because your labels are SGF move strings and not reasoning traces, SFT still tends to train short outputs. But this is because of target data distribution, **not** because `<think>` tokens are explicitly ignored.

---

## GRPO Implementation: Are Thinking Tokens Ignored?

### Move-level GRPO (`scripts/run_grpo.py`)
Reward function converts completion to a move using:

- `normalize_move(completion, stop_at_pass=...)`
- `scripts/run_grpo.py:139-141`

`normalize_move` is first-token based:

- `go_rl/rollout.py:49-58`

So if completion starts with reasoning text, first token may be non-move (`"I"`, `"<THINK>..."`, `"MOVE:"`), then legality/parse fails and reward is penalized.

Relevant reward path:

- Parse failure/invalid coords handling: `scripts/run_grpo.py:151-165`
- Fallback reward scoring on invalid move text: `scripts/run_grpo.py:196-203`
- Reward components include legality and length penalties: `go_rl/rewards.py:27-35`, `114-132`

There is no `<think>` stripping in this path.

### Full-game GRPO (`scripts/run_grpo_fullgame.py` + `go_rl/fullgame_rollout.py`)
Initial and generated completions are parsed via `_parse_completion_to_move`:

- `go_rl/fullgame_rollout.py:86-97`

This tries:

1. `parse_move` regex search anywhere in text (`D4` can be found inside prose) 
2. fallback to `normalize_move`

Evidence:

- `go_rl/prompting.py:36-45` (`parse_move` finds move or PASS via regex)
- `go_rl/fullgame_rollout.py:90-95`

So full-game rollout is tolerant of reasoning text that contains a valid move somewhere.

### GRPO loss masking behavior (from local installed `trl==0.24.0`)
GRPO constructs a `completion_mask` that marks generated completion tokens, then computes loss using `(per_token_loss * completion_mask)`:

- completion mask creation: `.../trl/trainer/grpo_trainer.py:1382-1385`
- loss application: `.../grpo_trainer.py:1729-1739`

There is no built-in `<think>`-tag mask in your pipeline configuration.

So reasoning tokens in completion are **not ignored by GRPO loss** unless you add custom masking logic.

---

## Runtime Sanity Checks Performed

### Parser behavior check
Input vs parser outputs:

- Input: `<think>I should play center</think> D4`
  - `normalize_move` -> `<THINK>I`
  - `parse_move` -> `D4`
  - `_parse_completion_to_move` (full-game) -> `D4`

- Input: `I think D4 is best`
  - `normalize_move` -> `I`
  - `parse_move` -> `D4`
  - `_parse_completion_to_move` -> `D4`

This confirms the move-level vs full-game parsing difference.

### Reward impact check (move-level GRPO)
Using `build_reward_fn` from `scripts/run_grpo.py` with default `RewardConfig`:

- completion `D4` -> reward `0.2`
- completion `I think D4 is best` -> reward `-1.0`
- completion `<think>consider</think> D4` -> reward `-1.05`

This confirms reasoning-prefixed outputs are penalized in move-level GRPO.

### Full-game check
Using `simulate_full_game` with one-turn config:

- `D4`, `I think D4 is best`, `<think>consider</think> D4` all parse to legal move and produced same reward (`0.2`) in that controlled setup.

---

## Current Effective Behavior By Stage

1. **Data prep**
   - SFT targets are clean move labels (`response`), not reasoning.

2. **SFT**
   - No explicit reasoning-tag filtering/masking.
   - Training text is `prompt + response`; behavior toward concise output comes from labels, not reasoning-mask logic.

3. **GRPO (move-level)**
   - No reasoning-token ignore mask.
   - Reasoning text before move usually harms reward due first-token parsing.

4. **GRPO (full-game)**
   - Move extraction is robust to extra text.
   - Still no explicit masking of reasoning tokens in GRPO optimization.

---

## Bottom Line
If your intended policy is:

- "Allow model reasoning internally"
- "Ignore `<think>...</think>` for training objective and reward"
- "Optimize only final move output"

then the current codebase **does not implement that policy end-to-end**.

It is currently a mixed setup:

- SFT data encourages concise move answers.
- Move-level GRPO penalizes reasoning-like prefixed text.
- Full-game GRPO can parse through reasoning text, but GRPO optimization still includes completion tokens globally.
