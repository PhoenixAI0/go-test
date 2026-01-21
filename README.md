# Go RL Training Utilities

This repo provides Go RL data prep, prompting utilities, and training entry points
for SFT and GRPO using KataGo-based rewards.

## 🚀 Quick Start

**New to this project? Start here:** [**QUICKSTART.md**](QUICKSTART.md)

The quick start guide provides step-by-step instructions for setting up and training on a fresh server using automated scripts.

### Automated Setup (Recommended)

```bash
# 1. Setup environment (installs everything)
./setup_environment.sh

# 2. Add SGF files to data/sgf_9x9/ or data/sgf_19x19/
# See download_sample_sgf.sh --help for sources

# 3. Prepare datasets
./prepare_data.sh

# 4. Train (SFT + GRPO in sequence)
./run_full_pipeline.sh --smoke  # smoke test first
./run_full_pipeline.sh          # full training
```

Or train individually:
```bash
./train_sft.sh --smoke    # SFT warm start
./train_grpo.sh --smoke   # RL training
```

## Manual Setup

If you prefer manual setup:

### Requirements
- Python 3.10+
- Dependencies: `datasets`, `trl`, `unsloth`, `sgfmill`, `pyyaml`, plus `torch`
- KataGo: install KataGo locally and export paths for configs:
  - `KATAGO_BIN`
  - `KATAGO_MODEL`
  - `KATAGO_CONFIG`

### Installation

```bash
# Using uv (recommended)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

# Or using pip
pip install -r requirements.txt
pip install -e .
```

## Data Expectations
- SGF files should live under a directory like `data/sgf_9x9` or `data/sgf_19x19`.
- Dataset builders write JSONL files under `data/`.

Build datasets:
```bash
python scripts/build_sft_dataset.py --sgf-dir data/sgf_9x9 --output data/sft_9x9.jsonl
python scripts/build_rl_prompts.py --sgf-dir data/sgf_9x9 --output data/rl_prompts_9x9.jsonl
```

Or use the combined helper:
```bash
python scripts/prepare_datasets.py --sgf-dir data/sgf_9x9 --tag 9x9
```

## Training
SFT warm start:
```bash
python scripts/run_sft.py --config config/sft_warm_start.yaml
```

GRPO training:
```bash
python scripts/run_grpo.py --config config/grpo_9x9.yaml
```

Smoke checks (short runs for wiring verification):
```bash
python scripts/run_sft.py --config config/sft_warm_start.yaml --smoke
python scripts/run_grpo.py --config config/grpo_9x9.yaml --smoke
```

Verbose logging and W&B reporting:
```bash
python scripts/run_sft.py --config config/sft_warm_start.yaml --log-level DEBUG
python scripts/run_sft.py --config config/sft_warm_start.yaml --wandb --wandb-project go-rl
```

### Multi-GPU Training

Train on multiple GPUs using Distributed Data Parallel (DDP):

```bash
# Check available GPUs
nvidia-smi
python scripts/check_gpu_setup.py

# SFT Training with multiple GPUs
./train_sft.sh --multi-gpu ddp-2    # 2 GPUs
./train_sft.sh --multi-gpu ddp-4    # 4 GPUs
./train_sft.sh --multi-gpu ddp-8    # 8 GPUs

# GRPO Training with multiple GPUs
./train_grpo.sh --multi-gpu ddp-4

# With gradient checkpointing for memory efficiency
./train_sft.sh --multi-gpu ddp-4 --gradient-checkpointing
```

**Performance:** Near-linear speedup (2 GPUs ≈ 2x faster, 4 GPUs ≈ 4x faster)

**Requirements:**
- Homogeneous GPUs recommended (same model/memory)
- Works with Unsloth's 4-bit quantization
- FSDP not supported (Unsloth limitation)

See [Multi-GPU Guide](docs/MULTI_GPU_GUIDE.md) for detailed usage, performance tuning, and troubleshooting.

## Outputs and Logs
- Model outputs live under `outputs/` (configurable via `output_dir`).
- Run logs are JSON files under `logs/` (configurable via `log_dir`) and include
  dependency versions plus GPU availability.

## Experiment Gate
- The 9x9 config includes a promotion gate (Elo delta or winrate threshold).
- Run 19x19 after the 9x9 gate is satisfied.

## Rewards
- `reward.strict_legality` enables ko/suicide checks in move legality scoring.
