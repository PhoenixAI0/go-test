# Go RL Training Utilities

This repo provides Go RL data prep, prompting utilities, and training entry points
for SFT and GRPO using KataGo-based rewards. It also includes optional Go tooling
for fast SGF parsing, self-play generation, and evaluation helpers.

**Repo layout**
- `go_rl/` core Python library.
- `go_parser/` optional Go SGF parser (builds a `sgf_parser` binary).
- `scripts/` setup, data prep, training, self-play, evaluation, and utilities.
- `config/` training and self-play YAML configs, plus `config/accelerate/` for multi-GPU.
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

3. Build datasets (will attempt to install Go + compile the Go SGF parser if needed):
```bash
./scripts/prepare_data.sh
```

4. Run smoke tests:
```bash
./scripts/train_sft.sh --smoke
./scripts/train_grpo.sh --smoke
```

5. Run the full pipeline (optional):
```bash
./scripts/run_full_pipeline.sh
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
uv pip install -r requirements.txt
uv pip install -e .
```

## Data Preparation

Expected SGF directories:
- `data/sgf_9x9/`
- `data/sgf_19x19/`

Dataset helpers:
```bash
python scripts/prepare_datasets.py --sgf-dir data/sgf_9x9 --output-dir data --tag 9x9
python scripts/prepare_datasets.py --sgf-dir data/sgf_19x19 --output-dir data --tag 19x19
```

Individual builders:
```bash
python scripts/build_sft_dataset.py --sgf-dir data/sgf_9x9 --output data/sft_9x9.jsonl
python scripts/build_rl_prompts.py --sgf-dir data/sgf_9x9 --output data/rl_prompts_9x9.jsonl
```

Fast builder (auto-uses Go parser if present):
```bash
python scripts/fast_build_dataset.py --sgf-dir data/sgf_9x9 --output data/sft_9x9.jsonl
```

Optional Go parser build (scripts will attempt install/build automatically if Go is missing):
```bash
cd go_parser
go build -o sgf_parser main.go
```

SGF download and validation helpers:
```bash
./scripts/download_sample_sgf.sh --board-size 9x9
python scripts/download_sgf.py --manual-instructions --board-size 9
python scripts/download_sgf.py --validate-only --board-size 9
```

## Training

Shell wrappers (recommended):
```bash
./scripts/train_sft.sh --board-size 9x9
./scripts/train_grpo.sh --board-size 9x9
```

Common flags:
- `--smoke` for short sanity runs.
- `--wandb` and `--wandb-project` for W&B logging.
- `--debug` for verbose logs.
- `--gradient-checkpointing` for memory savings.
- `--multi-gpu 2|4|8|auto` for DDP via `accelerate`.

Direct Python entry points:
```bash
python scripts/run_sft.py --config config/sft_warm_start.yaml
python scripts/run_sft.py --config config/sft_warm_start_19x19.yaml
python scripts/run_grpo.py --config config/grpo_9x9.yaml
python scripts/run_grpo.py --config config/grpo_19x19.yaml
```

Multi-GPU helpers:
```bash
python scripts/check_gpu_setup.py
python scripts/validate_multi_gpu.py --verbose
```

## Self-Play

Generate self-play SGFs:
```bash
python scripts/run_self_play.py --config config/self_play_9x9.yaml
```

Filter or validate self-play games:
```bash
python scripts/filter_self_play_games.py --input data/self_play/9x9 --output data/self_play/9x9_filtered
```

## Evaluation

Create a KataGo match config and run evaluation:
```bash
python scripts/create_match_config.py \
  --output config/match_9x9.cfg \
  --board-size 9 \
  --num-games 200 \
  --model-command "/path/to/your_gtp_engine --model outputs/grpo_9x9/final"

python scripts/eval_elo.py --katago-bin $KATAGO_BIN --match-config config/match_9x9.cfg
```

## Interactive Play

Play against a trained model in the terminal:
```bash
python scripts/play_interactive.py --model outputs/grpo_9x9/final
```

## Upload to Hugging Face Hub

Upload a trained model or LoRA adapter:
```bash
python scripts/upload_to_hub.py --model outputs/grpo_9x9/final --repo-id username/go-rl-9x9-grpo --board-size 9
```

## Tests

Run the full test suite:
```bash
pytest
```

Skip KataGo integration tests:
```bash
pytest -m "not katago"
```

Run only KataGo integration tests (requires `.env`):
```bash
pytest -m katago
```

## Outputs and Logs

- Model outputs are stored under `outputs/`.
- Run logs are stored under `logs/`.
- Datasets are written to `data/`.

These folders are generated artifacts and should not be committed unless explicitly requested.
