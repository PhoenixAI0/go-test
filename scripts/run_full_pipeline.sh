#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Go RL: Full Training Pipeline"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Set up environment"
echo "  2. Prepare datasets"
echo "  3. Run SFT training"
echo "  4. Run GRPO training"
echo ""
echo "This may take several hours to complete."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
SKIP_SETUP=false
SKIP_DATA=false
SKIP_SFT=false
SMOKE_TEST=false
USE_WANDB=false
BOARD_SIZE="9x9"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-sft)
            SKIP_SFT=true
            shift
            ;;
        --smoke)
            SMOKE_TEST=true
            shift
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --board-size)
            BOARD_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-setup         Skip environment setup"
            echo "  --skip-data          Skip data preparation"
            echo "  --skip-sft           Skip SFT training (go straight to GRPO)"
            echo "  --smoke              Run smoke tests instead of full training"
            echo "  --wandb              Enable Weights & Biases logging"
            echo "  --board-size SIZE    Board size: 9x9 or 19x19 (default: 9x9)"
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

# Confirm before starting
if [ "$SMOKE_TEST" = false ]; then
    echo -e "${YELLOW}⚠️  WARNING: This will run FULL training which may take hours.${NC}"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

START_TIME=$(date +%s)

# Step 1: Environment Setup
if [ "$SKIP_SETUP" = false ]; then
    echo ""
    echo -e "${YELLOW}=========================================="
    echo "STEP 1/4: Environment Setup"
    echo "==========================================${NC}"
    echo ""
    ./scripts/setup_environment.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}Setup failed!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping environment setup${NC}"
fi

# Activate environment for subsequent steps
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source .venv/bin/activate
fi
if [ -f .env ]; then
    source .env
fi

# Step 2: Data Preparation
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo -e "${YELLOW}=========================================="
    echo "STEP 2/4: Data Preparation"
    echo "==========================================${NC}"
    echo ""
    ./scripts/prepare_data.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}Data preparation failed!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping data preparation${NC}"
fi

# Step 3: SFT Training
if [ "$SKIP_SFT" = false ]; then
    echo ""
    echo -e "${YELLOW}=========================================="
    echo "STEP 3/4: SFT Training (Warm Start)"
    echo "==========================================${NC}"
    echo ""
    
    SFT_CMD="./scripts/train_sft.sh --board-size $BOARD_SIZE"
    if [ "$SMOKE_TEST" = true ]; then
        SFT_CMD="$SFT_CMD --smoke"
    fi
    if [ "$USE_WANDB" = true ]; then
        SFT_CMD="$SFT_CMD --wandb --wandb-project go-rl"
    fi
    
    $SFT_CMD
    if [ $? -ne 0 ]; then
        echo -e "${RED}SFT training failed!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping SFT training${NC}"
fi

# Step 4: GRPO Training
echo ""
echo -e "${YELLOW}=========================================="
echo "STEP 4/4: GRPO Training (Reinforcement Learning)"
echo "==========================================${NC}"
echo ""

GRPO_CMD="./scripts/train_grpo.sh --board-size $BOARD_SIZE"
if [ "$SMOKE_TEST" = true ]; then
    GRPO_CMD="$GRPO_CMD --smoke"
fi
if [ "$USE_WANDB" = true ]; then
    GRPO_CMD="$GRPO_CMD --wandb --wandb-project go-rl"
fi

$GRPO_CMD
if [ $? -ne 0 ]; then
    echo -e "${RED}GRPO training failed!${NC}"
    exit 1
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo -e "${GREEN}=========================================="
echo "🎉 FULL PIPELINE COMPLETE! 🎉"
echo "==========================================${NC}"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results:"
echo "  - SFT model: outputs/sft_warm_start/"
echo "  - GRPO model: outputs/grpo_${BOARD_SIZE}/"
echo "  - Logs: logs/"
echo ""
echo "Next steps:"
echo "  1. Evaluate your model with KataGo matches"
echo "  2. Export to GGUF for deployment"
echo "  3. Run more training iterations to improve"
echo ""
if [ "$BOARD_SIZE" = "9x9" ]; then
    echo "  Consider promoting to 19x19 if Elo threshold is met!"
    echo ""
fi
