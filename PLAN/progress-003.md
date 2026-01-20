# Progress Log 003

## Summary
- Core Go RL scaffolding implemented with data prep, reward shaping, KataGo analysis, and GRPO/SFT runners.
- Configuration templates added for 9x9 and 19x19 training.

## Files Created
- `PLAN/system-reminder.md`
- `PLAN/progress-001.md`
- `PLAN/progress-002.md`
- `PLAN/progress-003.md`
- `go_rl/__init__.py`
- `go_rl/board.py`
- `go_rl/config.py`
- `go_rl/data_prep.py`
- `go_rl/katago.py`
- `go_rl/prompting.py`
- `go_rl/rewards.py`
- `go_rl/rollout.py`
- `go_rl/sgf_utils.py`
- `scripts/build_rl_prompts.py`
- `scripts/build_sft_dataset.py`
- `scripts/eval_elo.py`
- `scripts/run_grpo.py`
- `scripts/run_sft.py`
- `config/grpo_9x9.yaml`
- `config/grpo_19x19.yaml`
- `config/sft_warm_start.yaml`

## Tooling Documentation
- `scripts/build_sft_dataset.py` — Convert SGF data into JSONL SFT samples.
  - Args: `--sgf-dir`, `--output`, `--max-positions`.
- `scripts/build_rl_prompts.py` — Build JSONL prompts for GRPO from SGF data.
  - Args: `--sgf-dir`, `--output`, `--max-positions`.
- `scripts/run_sft.py` — Run Unsloth SFT warm-start.
  - Args: `--config` (YAML, see `config/sft_warm_start.yaml`).
- `scripts/run_grpo.py` — Run GRPO training with KataGo-based reward.
  - Args: `--config` (YAML, see `config/grpo_9x9.yaml`).
- `scripts/eval_elo.py` — Run KataGo match engine for Elo evaluation.
  - Args: `--katago-bin`, `--match-config`, `--log-file`, `--sgf-dir`.

## Further Changes Needed
- Wire actual KataGo paths in `config/grpo_9x9.yaml` and `config/grpo_19x19.yaml`.
- Prepare SGF datasets (9x9/19x19) and generate `data/sft_9x9.jsonl` and `data/rl_prompts_9x9.jsonl`.
- Confirm vLLM endpoint access and model name alignment for rollouts.
- Decide gate threshold for 9x9 -> 19x19 (Elo delta) and update configs.
- Add optional legality checks against full rules (ko, suicide) if needed.
