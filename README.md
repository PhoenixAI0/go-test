# Go RL Training Utilities

This repo provides Go RL data prep, prompting utilities, and training entry points
for SFT and GRPO using KataGo-based rewards. Dataset parsing is handled by the Go
`sgf_parser` tool.

**Repo layout**
- `go_rl/` core Python library.
- `go_parser/` Go SGF parser (builds a `sgf_parser` binary).
- `scripts/` setup, data prep, training, evaluation, and utilities.
- `config/` training YAML configs, plus `config/accelerate/` for multi-GPU.
- `data/` SGF inputs and generated JSONL datasets.
- `outputs/` model checkpoints.
- `logs/` run logs.
- `tests/` pytest suite.

## Quick Start (Linux + NVIDIA GPU)

1. Setup environment (creates `.venv`, installs deps, downloads KataGo, writes `.env`):
```bash
./scripts/setup_environment.sh
```

2. Add SGF files to `data/sgf_9x9/` or `data/sgf_19x19/`.

3. Build datasets (compiles and uses the Go SGF parser):
```bash
./scripts/prepare_data.sh
```

4. Run smoke tests:
```bash
./scripts/train_sft.sh --smoke
./scripts/train_grpo.sh --smoke
```

5. Run the full pipeline (default includes 9x9 GRPO -> promotion gate -> 19x19 GRPO -> 19x19 full-game GRPO when the second gate passes):
```bash
./scripts/run_full_pipeline.sh \
  --model-command-template "python /path/to/your_gtp_engine.py --model {MODEL_DIR}"
```

To skip all evaluation and promotion-gate checks, pass `--no-eval`.
To skip the full-game stage only, pass `--skip-fullgame`.

6. Run the full pipeline with automatic Hub publish:
```bash
./scripts/run_full_pipeline.sh \
  --model-command-template "python /path/to/your_gtp_engine.py --model {MODEL_DIR}" \
  --upload-to-hub \
  --hf-repo-id username/go-rl-grpo
```

If you want to verify your setup at any point:
```bash
./scripts/verify_setup.sh
```

## Manual Setup

Requirements:
- Python 3.10+
- NVIDIA GPU with CUDA for training (setup script is optimized for Linux)
- KataGo binary + model + config (the setup script creates `.env` with paths)

Manual install with `uv`:
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync --extra dev
```

## Data Preparation

Expected SGF directories:
- `data/sgf_9x9/`
- `data/sgf_19x19/`

Dataset helpers:
```bash
uv run python scripts/prepare_datasets.py --sgf-dir data/sgf_9x9 --output-dir data --tag 9x9
uv run python scripts/prepare_datasets.py --sgf-dir data/sgf_19x19 --output-dir data --tag 19x19
```

Individual builders:
```bash
uv run python scripts/build_sft_dataset.py --sgf-dir data/sgf_9x9 --output data/sft_9x9.jsonl
uv run python scripts/build_rl_prompts.py --sgf-dir data/sgf_9x9 --output data/rl_prompts_9x9.jsonl
```

Fast builder:
```bash
uv run python scripts/fast_build_dataset.py --sgf-dir data/sgf_9x9 --output data/sft_9x9.jsonl
```

Manual Go parser compile:
```bash
./scripts/compile_go_parser.sh
```

SGF download and validation helpers:
```bash
./scripts/download_sample_sgf.sh --board-size 9x9
uv run python scripts/download_sgf.py --manual-instructions --board-size 9
uv run python scripts/download_sgf.py --validate-only --board-size 9
```

## Training

Shell wrappers (recommended):
```bash
./scripts/train_sft.sh --board-size 9x9
./scripts/train_grpo.sh --board-size 9x9
./scripts/train_grpo_fullgame.sh --smoke --fullgame-smoke-plies 20
```

Common flags:
- `--smoke` for short sanity runs.
- `--wandb` and `--wandb-project` for W&B logging.
- `--debug` for verbose logs.
- `--gradient-checkpointing` for memory savings.
- `--multi-gpu 2|4|8|auto` for DDP via `accelerate`.
- SFT now logs train/val loss and perplexity to W&B (validation split controlled by `eval_split_ratio` and `eval_steps` in SFT config).

Direct Python entry points:
```bash
uv run python scripts/run_sft.py --config config/sft_warm_start.yaml
uv run python scripts/run_sft.py --config config/sft_warm_start_19x19.yaml
uv run python scripts/run_grpo.py --config config/grpo_9x9.yaml
uv run python scripts/run_grpo.py --config config/grpo_19x19.yaml
uv run python scripts/run_grpo_fullgame.py --config config/grpo_fullgame_19x19.yaml
```

### Four-stage curriculum
1. SFT warm start on SGF data.
2. GRPO 9x9 (move-level reward shaping).
3. GRPO 19x19 curriculum stage.
4. Full-game GRPO 19x19 vs KataGo (interactive rollouts, margin + terminal outcome reward).

The full-game stage is gated by `promotion_gate_fullgame` in `config/grpo_19x19.yaml`.
Default gate: winrate >= 0.55 over 200 evaluation games.

Multi-GPU helpers:
```bash
uv run python scripts/check_gpu_setup.py
uv run python scripts/validate_multi_gpu.py --verbose
```

## Evaluation

Create a KataGo match config and run evaluation:
```bash
uv run python scripts/create_match_config.py \
  --output config/match_9x9.cfg \
  --board-size 9 \
  --num-games 200 \
  --model-command "/path/to/your_gtp_engine --model outputs/grpo_9x9"

uv run python scripts/eval_elo.py --katago-bin $KATAGO_BIN --match-config config/match_9x9.cfg
```

`eval_elo.py` can also write structured metrics and log to W&B:
```bash
uv run python scripts/eval_elo.py \
  --katago-bin $KATAGO_BIN \
  --match-config config/match_9x9.cfg \
  --summary-json logs/matches/match_9x9_summary.json \
  --wandb --wandb-project go-rl
```

## Upload to Hugging Face Hub

Upload a trained model or LoRA adapter:
```bash
uv run python scripts/upload_to_hub.py --model outputs/grpo_9x9 --repo-id username/go-rl-9x9-grpo --board-size 9 --push-only
```

## Tests

Run the full test suite:
```bash
uv run pytest
```

Skip KataGo integration tests:
```bash
uv run pytest -m "not katago"
```

Run only KataGo integration tests (requires `.env`):
```bash
uv run pytest -m katago
```

## Outputs and Logs

- Model outputs are stored under `outputs/`.
  - Includes `outputs/grpo_fullgame_19x19/` when step-4 full-game training runs.
- Run logs are stored under `logs/`.
- Datasets are written to `data/`.

These folders are generated artifacts and should not be committed unless explicitly requested.
