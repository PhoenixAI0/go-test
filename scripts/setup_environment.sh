#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != Linux || ! "$(uname -m)" =~ ^(x86_64|aarch64)$ ]]; then
    echo "CUDA setup requires Linux x86_64 (including WSL2) or aarch64 (DGX Spark)." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
    echo "Install an NVIDIA driver before running CUDA setup (nvidia-smi is missing)." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl --fail --location --silent --show-error https://astral.sh/uv/install.sh | sh
fi
uv sync --locked --python 3.12
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi
uv run --locked python -m go_rl.cuda_check --quantization-bits 16 --optimizer adamw
uv run --locked python scripts/setup_katago.py
./scripts/compile_go_parser.sh
mkdir -p data/sgf_9x9 data/sgf_19x19 outputs logs
echo "Setup complete. Start training with ./scripts/run_full_pipeline.sh --skip-setup"
