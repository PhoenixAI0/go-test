#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Go RL: Full-Game GRPO Training (19x19)"
echo "=========================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SMOKE_TEST=false
USE_WANDB=false
WANDB_PROJECT="go-rl"
LOG_LEVEL="INFO"
MULTI_GPU=""
GRADIENT_CHECKPOINTING=false
FULLGAME_SMOKE_PLIES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            SMOKE_TEST=true
            shift
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --debug)
            LOG_LEVEL="DEBUG"
            shift
            ;;
        --multi-gpu)
            MULTI_GPU="$2"
            shift 2
            ;;
        --gradient-checkpointing)
            GRADIENT_CHECKPOINTING=true
            shift
            ;;
        --fullgame-smoke-plies)
            FULLGAME_SMOKE_PLIES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoke                      Run smoke test"
            echo "  --wandb                      Enable Weights & Biases logging"
            echo "  --wandb-project NAME         Set W&B project name (default: go-rl)"
            echo "  --debug                      Enable debug logging"
            echo "  --multi-gpu N               Number of GPUs (1-8; supported: 1,2,4,8)"
            echo "  --multi-gpu auto            Auto-detect GPU count"
            echo "  --gradient-checkpointing    Enable gradient checkpointing"
            echo "  --fullgame-smoke-plies N    Override full-game max plies (useful for smoke)"
            echo "  -h, --help                  Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

if [ -f .env ]; then
    source .env
fi

if [ -z "${KATAGO_BIN:-}" ]; then
    echo -e "${RED}Error: KATAGO_BIN environment variable not set${NC}"
    echo "Please source .env or run ./scripts/setup_environment.sh"
    exit 1
fi

if [ ! -f "$KATAGO_BIN" ]; then
    echo -e "${RED}Error: KataGo binary not found: $KATAGO_BIN${NC}"
    echo "Please run ./scripts/setup_environment.sh to install KataGo"
    exit 1
fi

echo -e "${GREEN}✓ KataGo binary found: $KATAGO_BIN${NC}"

if ! $KATAGO_BIN version &> /dev/null; then
    echo -e "${RED}Error: KataGo test failed${NC}"
    exit 1
fi

DATASET_FILE="data/rl_prompts_19x19.jsonl"
if [ ! -f "$DATASET_FILE" ]; then
    echo -e "${RED}Error: Dataset not found: $DATASET_FILE${NC}"
    echo ""
    echo "Please run ./scripts/prepare_data.sh first to create the dataset."
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "$DATASET_FILE")
echo -e "${GREEN}Found dataset: $DATASET_FILE ($SAMPLE_COUNT samples)${NC}"

CONFIG_FILE="config/grpo_fullgame_19x19.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

if [ -z "$MULTI_GPU" ]; then
    CMD=(uv run python scripts/run_grpo_fullgame.py --config "$CONFIG_FILE" --log-level "$LOG_LEVEL")
else
    if [ "$MULTI_GPU" = "auto" ]; then
        if command -v nvidia-smi &>/dev/null; then
            MULTI_GPU=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
        else
            MULTI_GPU=$(uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
        fi
        if [ "$MULTI_GPU" -eq 0 ]; then
            echo -e "${RED}Error: Auto-detect found no GPUs${NC}"
            echo "Install NVIDIA drivers or pass --multi-gpu with an explicit count"
            exit 1
        fi
    fi

    if ! [[ "$MULTI_GPU" =~ ^[1-8]$ ]]; then
        echo -e "${RED}Error: Invalid --multi-gpu value: $MULTI_GPU${NC}"
        echo "Expected a number from 1-8 or 'auto' (supported counts: 1, 2, 4, 8)"
        exit 1
    fi

    case "$MULTI_GPU" in
        1) ACCELERATE_CONFIG="config/accelerate/single_gpu.yaml" ;;
        2) ACCELERATE_CONFIG="config/accelerate/ddp_2gpu.yaml" ;;
        4) ACCELERATE_CONFIG="config/accelerate/ddp_4gpu.yaml" ;;
        8) ACCELERATE_CONFIG="config/accelerate/ddp_8gpu.yaml" ;;
        *)
            echo -e "${RED}Error: Unsupported GPU count: $MULTI_GPU${NC}"
            echo "Supported counts: 1, 2, 4, 8"
            exit 1
            ;;
    esac

    if [ ! -f "$ACCELERATE_CONFIG" ]; then
        echo -e "${RED}Error: Accelerate config not found: $ACCELERATE_CONFIG${NC}"
        exit 1
    fi

    CMD=(uv run accelerate launch --config_file="$ACCELERATE_CONFIG" scripts/run_grpo_fullgame.py --config "$CONFIG_FILE" --log-level "$LOG_LEVEL")
    echo -e "${GREEN}Multi-GPU training enabled: ${MULTI_GPU} GPU(s)${NC}"
fi

if [ "$GRADIENT_CHECKPOINTING" = true ]; then
    CMD+=(--gradient-checkpointing)
    echo -e "${YELLOW}Gradient checkpointing enabled${NC}"
fi

if [ "$SMOKE_TEST" = true ]; then
    CMD+=(--smoke)
    echo -e "${YELLOW}Running in SMOKE TEST mode (minimal training for verification)${NC}"
fi

if [ -n "$FULLGAME_SMOKE_PLIES" ]; then
    CMD+=(--max-plies "$FULLGAME_SMOKE_PLIES")
    echo -e "${YELLOW}Overriding full-game max plies: $FULLGAME_SMOKE_PLIES${NC}"
fi

if [ "$USE_WANDB" = true ]; then
    CMD+=(--wandb --wandb-project "$WANDB_PROJECT")
    echo -e "${YELLOW}Weights & Biases logging enabled (project: $WANDB_PROJECT)${NC}"
fi

echo ""
echo "Configuration:"
echo "  - Stage: Full-game GRPO (19x19)"
echo "  - Config: $CONFIG_FILE"
echo "  - Dataset: $DATASET_FILE"
echo "  - Samples: $SAMPLE_COUNT"
echo "  - Smoke test: $SMOKE_TEST"
echo "  - W&B enabled: $USE_WANDB"
echo "  - KataGo: $KATAGO_BIN"
echo ""
echo -e "${YELLOW}Starting full-game GRPO training...${NC}"
echo "Command: ${CMD[*]}"
echo ""

# Run training; under set -e only an `if` can observe the failure.
if "${CMD[@]}"; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Full-game GRPO Training Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "Model saved to: outputs/grpo_fullgame_19x19/"
    echo "Logs saved to: logs/grpo_fullgame_19x19/"
    echo ""
else
    echo -e "${RED}Training failed! Check the logs above for errors.${NC}"
    exit 1
fi
