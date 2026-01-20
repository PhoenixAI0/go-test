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

echo -e "${YELLOW}Downloading sample SGF files for ${BOARD_SIZE}...${NC}"
echo ""

if [ "$BOARD_SIZE" = "9x9" ]; then
    # For 9x9, we'll provide instructions since there's no single public repository
    echo "Unfortunately, there's no single public API for downloading 9x9 SGF files."
    echo ""
    echo -e "${YELLOW}Please manually download SGF files from these sources:${NC}"
    echo ""
    echo "1. U-Go.net (9x9 games):"
    echo "   https://www.u-go.net/gamerecords/"
    echo ""
    echo "2. Online-Go.com (your own games):"
    echo "   - Go to https://online-go.com/"
    echo "   - Play or download games"
    echo "   - Click 'Download SGF' on game pages"
    echo ""
    echo "3. Little Golem (correspondence Go):"
    echo "   http://www.littlegolem.net/jsp/game/"
    echo ""
    echo -e "Then copy them to: ${GREEN}${TARGET_DIR}/${NC}"
    echo ""
    echo "Example:"
    echo "  cp ~/Downloads/*.sgf ${TARGET_DIR}/"
    echo ""
    
elif [ "$BOARD_SIZE" = "19x19" ]; then
    # For 19x19, we can download from GoKifu's public games
    echo "Downloading professional games from Waltheri's database..."
    echo ""
    
    # Note: This is a placeholder - you'd need actual URLs or API access
    echo -e "${YELLOW}Recommended sources for 19x19 SGF files:${NC}"
    echo ""
    echo "1. GoGoD Database (commercial, high quality):"
    echo "   https://gogodonline.co.uk/"
    echo ""
    echo "2. Waltheri's Go Pattern Search:"
    echo "   https://ps.waltheri.net/"
    echo "   - Search for games"
    echo "   - Download individual SGF files"
    echo ""
    echo "3. KGS Game Archives:"
    echo "   https://www.gokgs.com/gameArchives.html"
    echo ""
    echo "4. GoKifu (free professional games):"
    echo "   https://gokifu.com/"
    echo ""
    echo -e "Then copy them to: ${GREEN}${TARGET_DIR}/${NC}"
    echo ""
else
    echo -e "${RED}Invalid board size: $BOARD_SIZE${NC}"
    exit 1
fi

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
