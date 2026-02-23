#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Go RL: GRPO Training"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
SMOKE_TEST=false
USE_WANDB=false
WANDB_PROJECT="go-rl"
BOARD_SIZE="9x9"
LOG_LEVEL="INFO"
MULTI_GPU=""
GRADIENT_CHECKPOINTING=false

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
        --board-size)
            BOARD_SIZE="$2"
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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoke              Run smoke test (minimal steps for verification)"
            echo "  --wandb              Enable Weights & Biases logging"
            echo "  --wandb-project NAME Set W&B project name (default: go-rl)"
            echo "  --board-size SIZE    Board size: 9x9 or 19x19 (default: 9x9)"
            echo "  --debug              Enable debug logging"
            echo "  --multi-gpu N        Number of GPUs (1-8; supported: 1, 2, 4, 8)"
            echo "  --multi-gpu auto     Auto-detect GPU count"
            echo "  --gradient-checkpointing  Enable gradient checkpointing"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Ensure virtual environment is activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Verify KataGo is available
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

# Test KataGo
if ! $KATAGO_BIN version &> /dev/null; then
    echo -e "${RED}Error: KataGo test failed${NC}"
    exit 1
fi

# Check if dataset exists
DATASET_FILE="data/rl_prompts_${BOARD_SIZE}.jsonl"
if [ ! -f "$DATASET_FILE" ]; then
    echo -e "${RED}Error: Dataset not found: $DATASET_FILE${NC}"
    echo ""
    echo "Please run ./scripts/prepare_data.sh first to create the dataset."
    exit 1
fi

# Count samples
SAMPLE_COUNT=$(wc -l < "$DATASET_FILE")
echo -e "${GREEN}Found dataset: $DATASET_FILE ($SAMPLE_COUNT samples)${NC}"

# Determine config file
if [ "$BOARD_SIZE" = "9x9" ]; then
    CONFIG_FILE="config/grpo_9x9.yaml"
elif [ "$BOARD_SIZE" = "19x19" ]; then
    CONFIG_FILE="config/grpo_19x19.yaml"
else
    echo -e "${RED}Error: Invalid board size: $BOARD_SIZE${NC}"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Build command
if [ -z "$MULTI_GPU" ]; then
    # Single GPU mode (backward compatible)
    CMD="uv run python scripts/run_grpo.py --config $CONFIG_FILE --log-level $LOG_LEVEL"
else
    if [ "$MULTI_GPU" = "auto" ]; then
        if command -v nvidia-smi &>/dev/null; then
            MULTI_GPU=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
        else
            MULTI_GPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
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

    # Multi-GPU mode with accelerate
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
    CMD="accelerate launch --config_file=$ACCELERATE_CONFIG scripts/run_grpo.py --config $CONFIG_FILE --log-level $LOG_LEVEL"
    echo -e "${GREEN}Multi-GPU training enabled: ${MULTI_GPU} GPU(s)${NC}"
fi

if [ "$GRADIENT_CHECKPOINTING" = true ]; then
    CMD="$CMD --gradient-checkpointing"
    echo -e "${YELLOW}Gradient checkpointing enabled${NC}"
fi

if [ "$SMOKE_TEST" = true ]; then
    CMD="$CMD --smoke"
    echo -e "${YELLOW}Running in SMOKE TEST mode (minimal training for verification)${NC}"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --wandb --wandb-project $WANDB_PROJECT"
    echo -e "${YELLOW}Weights & Biases logging enabled (project: $WANDB_PROJECT)${NC}"
fi

echo ""
echo "Configuration:"
echo "  - Board size: $BOARD_SIZE"
echo "  - Config: $CONFIG_FILE"
echo "  - Dataset: $DATASET_FILE"
echo "  - Samples: $SAMPLE_COUNT"
echo "  - Smoke test: $SMOKE_TEST"
echo "  - W&B enabled: $USE_WANDB"
echo "  - KataGo: $KATAGO_BIN"
echo ""
echo -e "${YELLOW}Starting GRPO training...${NC}"
echo "Command: $CMD"
echo ""

# Run training
$CMD

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "GRPO Training Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "Model saved to: outputs/grpo_${BOARD_SIZE}/"
    echo "Logs saved to: logs/grpo_${BOARD_SIZE}/"
    echo ""
    if [ "$SMOKE_TEST" = false ]; then
        echo "Next steps:"
        echo "1. Evaluate the trained model with KataGo matches"
        echo "2. Run: uv run python scripts/eval_elo.py --katago-bin \$KATAGO_BIN ..."
        echo ""
        if [ "$BOARD_SIZE" = "9x9" ]; then
            echo "Consider promoting to 19x19 if Elo threshold is met!"
        fi
    else
        echo "This was a smoke test. To do full training, run without --smoke flag."
        echo ""
    fi
else
    echo -e "${RED}Training failed! Check the logs above for errors.${NC}"
    exit 1
fi
