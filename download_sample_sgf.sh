#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "Download Sample SGF Files"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BOARD_SIZE="9x9"

while [[ $# -gt 0 ]]; do
    case $1 in
        --board-size)
            BOARD_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --board-size SIZE    Board size: 9x9 or 19x19 (default: 9x9)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "This script downloads sample SGF files for testing."
            echo "For production training, you should obtain a larger dataset."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create directory
TARGET_DIR="data/sgf_${BOARD_SIZE}"
mkdir -p "$TARGET_DIR"

echo -e "${YELLOW}Preparing to download SGF files for ${BOARD_SIZE}...${NC}"
echo ""

# Check if Python is available
if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    echo "Python is required to run the download script."
    exit 1
fi

# Use python3 if available, fallback to python
PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
    PYTHON_CMD="python"
fi

# Extract numeric board size (9 from "9x9", etc.)
NUMERIC_SIZE=$(echo "$BOARD_SIZE" | cut -d'x' -f1)

# Run Python download script (shows manual instructions)
$PYTHON_CMD scripts/download_sgf.py \
    --board-size "$NUMERIC_SIZE" \
    --output-dir "$TARGET_DIR" \
    --manual-instructions

echo ""

# Check if any files were added
COUNT=$(find "$TARGET_DIR" -name "*.sgf" 2>/dev/null | wc -l)

if [ "$COUNT" -gt 0 ]; then
    echo -e "${GREEN}Found $COUNT SGF files in $TARGET_DIR${NC}"
    echo ""
    echo "Next step: Run ./prepare_data.sh to convert these to training data"
else
    echo -e "${YELLOW}No SGF files found yet in $TARGET_DIR${NC}"
    echo ""
    echo "After downloading SGF files, run ./prepare_data.sh"
fi
