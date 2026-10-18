#!/usr/bin/env bash
# One command for setup, data preparation, and all four curriculum stages.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
SKIP_SETUP=false
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi
for arg in "$@"; do
    case "$arg" in
        --skip-setup) SKIP_SETUP=true ;;
        --help|-h|--dry-run) exec python3 scripts/train_cuda.py "$@" ;;
    esac
done
# Validate arguments and data paths before downloading the training environment.
python3 scripts/train_cuda.py --preflight-only "$@"
if [[ "$SKIP_SETUP" == false ]]; then
    ./scripts/setup_environment.sh
    # Setup may have just created the KataGo settings.
    set -a
    source .env
    set +a
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec uv run --locked python scripts/train_cuda.py "$@"
