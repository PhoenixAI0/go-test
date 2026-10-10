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

## Run the entire training curriculum on CUDA

The same command supports **DGX Spark (Linux aarch64)** and **conventional NVIDIA
GPUs on Linux x86_64 / WSL2**. Setup selects the locked packages automatically:

| Host | PyTorch | CUDA runtime | KataGo evaluator |
| --- | --- | --- | --- |
| DGX Spark / aarch64 | 2.9.1 | 13.0 | Native Eigen CPU build |
| x86_64 NVIDIA machine | 2.9.1 | 12.8 | Native Eigen CPU build |

Use a compatible NVIDIA driver; Spark's CUDA 13 runtime requires driver 580 or
newer. CUDA runtime libraries come with the Python packages; a separately installed
CUDA toolkit is not needed. Setup checks actual attention backward and optimizer
kernels, and the launcher additionally tests the selected quantization kernels.

On either platform, run from the repository root:

```bash
./scripts/train_cuda.sh
```

This installs the locked Python 3.12 environment, installs Go if needed, builds
SGF datasets, checks CUDA and KataGo analysis, and trains these stages in order.
If either SGF directory is empty, it automatically downloads real games from the
[KataGo bulk archive](https://katagoarchive.org/kata1/traininggames/index.html),
verifies the archive checksum, validates the SGFs, and separates 9x9 and 19x19 games.
Existing games in `data/sgf_9x9/` and `data/sgf_19x19/` are reused.

The training stages are:

1. 9x9 supervised warm start.
2. 9x9 position GRPO, loading the SFT adapter.
3. 19x19 position GRPO, loading the 9x9 adapter.
4. 19x19 full-game GRPO, loading the 19x19 adapter.

Defaults use `Qwen/Qwen3-8B`, 4-bit LoRA, gradient checkpointing, a per-device
training batch of one, and non-thinking move completions. The model is downloaded
from Hugging Face on first use. GPU memory needs depend on the model and sequence
length; use `--model MODEL_ID` to select a smaller compatible model if needed.

To check installation and one training update per stage without supplying games:

```bash
./scripts/train_cuda.sh --smoke --model Qwen/Qwen3-0.6B
```

Smoke mode generates tiny synthetic SGFs in its own run directory. These fixtures
are only for testing the pipeline, not training a useful player. Smoke runs still
load the selected model and require a CUDA GPU.

Useful options:

```bash
./scripts/train_cuda.sh --dry-run              # No installation or GPU needed
./scripts/train_cuda.sh --skip-setup           # Reuse the installed environment
./scripts/train_cuda.sh --sgf-9x9 /games/9 --sgf-19x19 /games/19
./scripts/train_cuda.sh --skip-fullgame        # Stop after position GRPO
./scripts/train_cuda.sh --no-download-data     # Require existing local SGFs
./scripts/train_cuda.sh --download-games 1000  # Cap imported games per board size
CUDA_VISIBLE_DEVICES=1 ./scripts/train_cuda.sh --skip-setup
./scripts/train_cuda.sh --quantization-bits 16 --optimizer adamw  # BF16/FP16 LoRA
```

Each run writes its datasets, resolved configs, logs, and adapters under a new
`outputs/cuda_TIMESTAMP/` directory (`outputs/smoke_TIMESTAMP/` for smoke mode).
Use `--output-dir PATH` to choose an empty destination. The final adapter is in
`grpo_fullgame_19x19/`, or `grpo_19x19/` with `--skip-fullgame`. A failed stage stops
the pipeline; it does not train subsequent stages from stale checkpoints. This
launcher starts fresh runs, not optimizer-state resumes.

Setup uses KataGo's CPU Eigen build to reserve GPU memory for the language model
and avoid a separate CUDA/cuDNN installation for the evaluator. CPU evaluation
can make full-game training slow. Existing `KATAGO_BIN` and `KATAGO_MODEL` settings
in `.env` are preserved, so an installed GPU KataGo can also be used. The launcher
sources `.env` automatically. Both platforms build KataGo from pinned source with CMake,
Eigen3 and zlib. Setup installs missing build packages automatically on Debian/Ubuntu
when running as root or with passwordless sudo. Otherwise install them once:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libeigen3-dev zlib1g-dev
```

Standard Linux runtime libraries (including libgomp) and network access for initial
downloads are required. When moving a checkout between machines, recreate `.venv`
and remove stale machine-specific KataGo paths from `.env`. Keep your SGFs and
checkpoints. Architecture-specific KataGo installs are stored separately.

Training uses Transformers, PEFT and TRL on both platforms, with SDPA attention and
bitsandbytes NF4 by default. It no longer depends on Unsloth, custom FlashAttention
builds, or vLLM. Remove the obsolete `grpo_mini_batch` and
`grpo_logit_chunk_multiplier` keys from custom configs. Legacy FP8/dynamic-4bit
flags are rejected explicitly. `--quantization-bits 16 --optimizer adamw` uses
BF16 where supported and FP16 otherwise; it needs more memory than 4-bit LoRA.
The whole-curriculum launcher uses one GPU. Individual stage scripts support DDP;
it does not orchestrate a cluster of linked Sparks.

The default data source is the pinned `2025-01-01sgfs.tar.bz2` public self-play
archive (about 21 MiB compressed). Up to 10,000 games per size are imported;
the default import contains 461 9x9 games and 10,000 19x19 games. This is a
starter corpus, so the resulting 9x9 dataset may be smaller than `--max-positions`.
The games include varied rules and komi. The current prompts represent board
positions and side to move; GRPO evaluation uses the configured Chinese rules
and 7.5 komi. Use your own curated SGFs for a larger or rule-specific SFT corpus.
The Go parser stops a game before an illegal move under its training legality checks.

Downloads are cached in `data/downloads/` (override with `--data-cache`). Each
installed corpus includes `source.json` with its source, SHA-256, and file list;
the run's `run.json` records data provenance. Completed imports are reused and
interrupted downloads/imports never appear as complete datasets. Downloads fail
explicitly on network or checksum errors. Smoke mode continues to use synthetic
fixtures and never downloads the game archive.

W&B is off unless `--wandb` is supplied. Match evaluation, promotion gates, and
Hub publishing are separate from this training-only launcher. The existing
`run_full_pipeline.sh` supports those workflows with a user-supplied GTP model
engine via `--model-command-template`; it can also run with `--no-eval`.

## Validation

```bash
uv sync --locked --extra dev
uv run pytest -m "not katago"
```

The suite includes actual one-step TRL/PEFT training and adapter saves using a tiny
local CPU model, using the production loader and checking adapter weights after
reload. Only KataGo is substituted in these trainer tests. Both Linux dependency
plans have been resolved, but CUDA execution needs verification on the target
hardware: run `./scripts/train_cuda.sh --smoke --model Qwen/Qwen3-0.6B` on each
machine before a full run. KataGo integration tests
require the paths from `.env` (`set -a; source .env; set +a`).

The integrations follow the pinned [TRL 0.24 trainer APIs](https://huggingface.co/docs/trl/v0.24.0/grpo_trainer),
[PyTorch 2.9 ARM/CUDA support](https://pytorch.org/blog/pytorch-2-9/),
[bitsandbytes 0.49.2 platform matrix](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.49.2/docs/source/installation.mdx),
and [KataGo analysis protocol](https://github.com/lightvector/KataGo/blob/v1.15.3/docs/Analysis_Engine.md).

## Manual Setup

Requirements:
- Python 3.12 or 3.13 (the launcher uses 3.12)
- NVIDIA GPU with CUDA for training (setup script is optimized for Linux)
- KataGo binary + model + config (the setup script creates `.env` with paths)

Manual install with `uv`:
```bash
uv venv .venv --python 3.12
uv sync --extra dev
```

(`uv run ...` uses the project `.venv` automatically, so manual activation is optional.)

## Data Preparation

Expected SGF directories:
- `data/sgf_9x9/`
- `data/sgf_19x19/`

Download real games separately (no GPU needed after installing the Python environment):

```bash
uv run python scripts/download_sgf.py --download --board-size 9
uv run python scripts/download_sgf.py --download --board-size 19
```

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
