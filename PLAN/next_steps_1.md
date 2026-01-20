# Next Steps 01

## Context
Status update 03 shows core Go RL scaffolding, config templates, and runner scripts are in place. The repo currently has no tests, no commit history, and a single scratch `test.py`. Key scripts live under `scripts/` and rely on YAML configs in `config/`.

## Immediate Priorities
1. Populate KataGo paths in `config/grpo_9x9.yaml` and `config/grpo_19x19.yaml` (binary, model, config).
2. Prepare SGF datasets and generate JSONL files:
   - `python scripts/build_sft_dataset.py --sgf-dir data/sgf_9x9 --output data/sft_9x9.jsonl`
   - `python scripts/build_rl_prompts.py --sgf-dir data/sgf_9x9 --output data/rl_prompts_9x9.jsonl`
3. Run a small SFT warm start and GRPO smoke test to validate end-to-end flow and artifact outputs.

## Validation & Quality
- Add a minimal `tests/` directory with `pytest` smoke coverage for data prep and prompt parsing.
- Capture a short run log for SFT/GRPO to confirm dependency versions and GPU requirements.

## Experiment Tracking
- Define a gate from 9x9 to 19x19 (Elo delta or winrate threshold) and reflect it in config notes.
- Standardize run output locations (e.g., `outputs/` and `logs/`) and document them.

## Optional Enhancements
- Evaluate stricter move legality checks (ko/suicide) in rewards if rollout errors are frequent.
- Add a brief README for setup, dependencies (Unsloth/TRL/datasets), and dataset expectations.
