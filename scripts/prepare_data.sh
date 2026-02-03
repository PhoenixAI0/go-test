#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Go RL Dataset Preparation"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure virtual environment is activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Load KataGo environment variables
if [ -f .env ]; then
    source .env
fi

# Check if SGF files exist
SGF_9x9_COUNT=$(find data/sgf_9x9 -name "*.sgf" 2>/dev/null | wc -l)
SGF_19x19_COUNT=$(find data/sgf_19x19 -name "*.sgf" 2>/dev/null | wc -l)

echo "Found $SGF_9x9_COUNT SGF files in data/sgf_9x9/"
echo "Found $SGF_19x19_COUNT SGF files in data/sgf_19x19/"

if [ "$SGF_9x9_COUNT" -eq 0 ] && [ "$SGF_19x19_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No SGF files found!${NC}"
    echo ""
    echo "Please add SGF files to one of these directories:"
    echo "  - data/sgf_9x9/    (for 9x9 Go games)"
    echo "  - data/sgf_19x19/  (for 19x19 Go games)"
    echo ""
    echo "You can download SGF files from:"
    echo "  - https://homepages.cwi.nl/~aeb/go/games/ (historical games)"
    echo "  - https://www.u-go.net/gamerecords/ (9x9 games)"
    echo "  - https://www.online-go.com/ (download your own games)"
    exit 1
fi

# Prepare 9x9 datasets
if [ "$SGF_9x9_COUNT" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Preparing 9x9 datasets...${NC}"
    python scripts/prepare_datasets.py \
        --sgf-dir data/sgf_9x9 \
        --output-dir data \
        --tag 9x9 \
        --max-positions 200000 \
        --log-level INFO
    
    if [ -f data/sft_9x9.jsonl ] && [ -f data/rl_prompts_9x9.jsonl ]; then
        SFT_LINES=$(wc -l < data/sft_9x9.jsonl)
        RL_LINES=$(wc -l < data/rl_prompts_9x9.jsonl)
        echo -e "${GREEN}✓ 9x9 datasets created:${NC}"
        echo "  - data/sft_9x9.jsonl ($SFT_LINES samples)"
        echo "  - data/rl_prompts_9x9.jsonl ($RL_LINES samples)"
    else
        echo -e "${RED}Error: Failed to create 9x9 datasets${NC}"
        exit 1
    fi
fi

# Prepare 19x19 datasets
if [ "$SGF_19x19_COUNT" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Preparing 19x19 datasets...${NC}"
    python scripts/prepare_datasets.py \
        --sgf-dir data/sgf_19x19 \
        --output-dir data \
        --tag 19x19 \
        --max-positions 200000 \
        --log-level INFO
    
    if [ -f data/sft_19x19.jsonl ] && [ -f data/rl_prompts_19x19.jsonl ]; then
        SFT_LINES=$(wc -l < data/sft_19x19.jsonl)
        RL_LINES=$(wc -l < data/rl_prompts_19x19.jsonl)
        echo -e "${GREEN}✓ 19x19 datasets created:${NC}"
        echo "  - data/sft_19x19.jsonl ($SFT_LINES samples)"
        echo "  - data/rl_prompts_19x19.jsonl ($RL_LINES samples)"
    else
        echo -e "${RED}Error: Failed to create 19x19 datasets${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Dataset Preparation Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Review the generated datasets in data/"
echo "2. Run: ./scripts/train_sft.sh (for supervised fine-tuning warm start)"
echo "3. Run: ./scripts/train_grpo.sh (for RL training)"
echo ""
