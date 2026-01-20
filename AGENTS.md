# Repository Guidelines

## Project Structure & Module Organization
- `go_rl/` holds the core Go RL library (board logic, KataGo integration, data prep, rewards, rollouts).
- `scripts/` contains runnable entry points for dataset building, training, and evaluation.
- `config/` provides YAML configs for SFT/GRPO training presets.
- `PLAN/` tracks work-in-progress notes.
- `test.py` is a scratch file, not a test suite.

## Build, Test, and Development Commands
This repo is Python-first; there is no build step.
- `python scripts/build_sft_dataset.py --sgf-dir data/sgf --output data/sft.jsonl` builds SFT training data from SGF files.
- `python scripts/build_rl_prompts.py --sgf-dir data/sgf --output data/rl_prompts.jsonl` creates prompt-only RL samples.
- `python scripts/run_sft.py --config config/sft_warm_start.yaml` runs supervised fine-tuning.
- `python scripts/run_grpo.py --config config/grpo_9x9.yaml` runs GRPO training with reward shaping.
- `python scripts/eval_elo.py --katago-bin /path/to/katago --match-config /path/to/match.cfg` runs a KataGo match for Elo evaluation.

## Coding Style & Naming Conventions
- Use 4-space indentation and PEP8-style naming (`snake_case` for functions/files, `CamelCase` for classes).
- Prefer explicit type hints and small, focused functions.
- Keep filenames and config keys descriptive (e.g., `grpo_9x9.yaml`).
- No formatter or linter is configured; keep edits consistent with existing style.

## Testing Guidelines
- There is no formal test framework yet. If you add tests, place them in `tests/` and use `pytest` naming (`test_*.py`).
- For smoke checks, run the relevant script with a small SGF dataset and confirm outputs are generated.

## Commit & Pull Request Guidelines
- The repository has no commit history yet, so no existing conventions. Use short, imperative messages (e.g., "Add GRPO reward normalization").
- PRs should include: a concise description, configs used for runs, and any relevant logs or metrics. Attach sample outputs or screenshots if you change prompts or evaluation outputs.

## Configuration Notes
- Training scripts expect YAML configs with paths to KataGo binaries/models; keep these paths local and do not commit large artifacts.
- Output directories (e.g., model checkpoints) should be excluded from version control unless explicitly requested.
