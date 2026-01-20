#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "Go RL: Supervised Fine-Tuning (SFT)"
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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoke              Run smoke test (1 epoch, minimal data)"
            echo "  --wandb              Enable Weights & Biases logging"
            echo "  --wandb-project NAME Set W&B project name (default: go-rl)"
            echo "  --board-size SIZE    Board size: 9x9 or 19x19 (default: 9x9)"
            echo "  --debug              Enable debug logging"
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

# Check if dataset exists
DATASET_FILE="data/sft_${BOARD_SIZE}.jsonl"
if [ ! -f "$DATASET_FILE" ]; then
    echo -e "${RED}Error: Dataset not found: $DATASET_FILE${NC}"
    echo ""
    echo "Please run ./prepare_data.sh first to create the dataset."
    exit 1
fi

# Count samples
SAMPLE_COUNT=$(wc -l < "$DATASET_FILE")
echo -e "${GREEN}Found dataset: $DATASET_FILE ($SAMPLE_COUNT samples)${NC}"

# Determine config file
if [ "$BOARD_SIZE" = "9x9" ]; then
    CONFIG_FILE="config/sft_warm_start.yaml"
elif [ "$BOARD_SIZE" = "19x19" ]; then
    # For 19x19, we'd need a separate config, but using same one for now
    CONFIG_FILE="config/sft_warm_start.yaml"
    echo -e "${YELLOW}Note: Using 9x9 config for 19x19. Consider creating config/sft_warm_start_19x19.yaml${NC}"
else
    echo -e "${RED}Error: Invalid board size: $BOARD_SIZE${NC}"
    exit 1
fi

# Build command
CMD="python scripts/run_sft.py --config $CONFIG_FILE --log-level $LOG_LEVEL"

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
echo ""
echo -e "${YELLOW}Starting SFT training...${NC}"
echo "Command: $CMD"
echo ""

# Run training
$CMD

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "SFT Training Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "Model saved to: outputs/sft_warm_start/"
    echo "Logs saved to: logs/sft_warm_start/"
    echo ""
    if [ "$SMOKE_TEST" = false ]; then
        echo "Next steps:"
        echo "1. Evaluate the trained model"
        echo "2. Run GRPO training: ./train_grpo.sh"
        echo ""
    else
        echo "This was a smoke test. To do full training, run without --smoke flag."
        echo ""
    fi
else
    echo -e "${RED}Training failed! Check the logs above for errors.${NC}"
    exit 1
fi
