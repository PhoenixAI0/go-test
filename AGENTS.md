# Repository Guidelines

## Project Structure & Module Organization
- `go_rl/` holds the core Python library (board logic, KataGo integration, data prep, rewards, rollouts, prompting).
- `go_parser/` contains the Go SGF parser used for dataset building.
- `scripts/` contains runnable entry points for setup, data prep, training, evaluation, and uploads.
- `config/` provides YAML configs for SFT/GRPO, plus `config/accelerate/` for multi-GPU.
- `data/` stores input SGFs and generated JSONL datasets.
- `outputs/` contains model checkpoints.
- `logs/` stores run logs.
- `tests/` is the pytest suite.
- `PLAN/` tracks work-in-progress notes.

## Build, Test, and Development Commands
This repo is Python-first for training/inference code. SGF dataset parsing relies on the Go SGF parser (`go_parser/sgf_parser`); scripts will install Go and compile the parser when needed.

Setup and validation:
- `./scripts/setup_environment.sh` creates `.venv`, installs deps, downloads KataGo, and writes `.env` (Linux + NVIDIA GPU).
- `./scripts/verify_setup.sh` validates your environment.

Dataset prep:
- `./scripts/prepare_data.sh` builds `data/sft_*.jsonl` and `data/rl_prompts_*.jsonl`.
- `uv run python scripts/prepare_datasets.py --sgf-dir data/sgf_9x9 --tag 9x9`
- `uv run python scripts/build_sft_dataset.py --sgf-dir data/sgf_9x9 --output data/sft_9x9.jsonl` (compiles the Go parser if needed).
- `uv run python scripts/download_sgf.py --manual-instructions --board-size 9`

Training:
- `./scripts/train_sft.sh --board-size 9x9 --smoke`
- `./scripts/train_grpo.sh --board-size 9x9 --smoke`
- `./scripts/run_full_pipeline.sh --smoke`
- `./scripts/run_full_pipeline.sh --upload-to-hub --hf-repo-id user/repo`
- Multi-GPU uses `./scripts/train_sft.sh --multi-gpu 2` (configs in `config/accelerate/`).

Evaluation:
- `uv run python scripts/create_match_config.py --output config/my_match.cfg --model-command "uv run python ..."`
- `uv run python scripts/eval_elo.py --match-config config/my_match.cfg`

Model upload:
- `uv run python scripts/upload_to_hub.py --model outputs/grpo_9x9 --repo-id user/repo --board-size 9 --push-only`

Testing:
- `uv run pytest`
- `uv run pytest -m "not katago"` to skip KataGo integration tests.
- `uv run pytest -m katago` to run KataGo integration tests (requires `.env`).

## Coding Style & Naming Conventions
- Use 4-space indentation and PEP8-style naming (`snake_case` for functions/files, `CamelCase` for classes).
- Prefer explicit type hints and small, focused functions.
- Keep filenames and config keys descriptive (e.g., `grpo_9x9.yaml`).
- No formatter or linter is configured; keep edits consistent with existing style.

## Testing Guidelines
- Tests live in `tests/` and use `pytest` naming (`test_*.py`).
- KataGo integration tests are tagged with the `katago` marker in `pytest.ini`.
- For smoke checks, run a small SGF dataset and confirm outputs are generated in `data/` and `outputs/`.

## Commit & Pull Request Guidelines
- Use short, imperative messages (e.g., "Add GRPO reward normalization").
- PRs should include: a concise description, configs used for runs, and any relevant logs or metrics. Attach sample outputs or screenshots if you change prompts or evaluation outputs.

## Configuration Notes
- Training scripts expect YAML configs with paths to KataGo binaries/models (usually via `.env` created by setup).
- Multi-GPU uses `config/accelerate/*.yaml` with `uv run accelerate launch`.
- Output directories (e.g., `outputs/`, `logs/`, generated datasets) should be excluded from version control unless explicitly requested.
