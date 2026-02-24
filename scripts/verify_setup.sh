#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Go RL Setup Verification"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

echo ""
echo "Checking setup..."
echo ""

# Check 1: Virtual environment
echo -n "1. Virtual environment: "
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "   Run: ./scripts/setup_environment.sh"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Python packages (if venv exists)
if [ -d ".venv" ]; then
    echo -n "2. Python packages: "
    source .venv/bin/activate 2>/dev/null || true
    
    if python -c "import torch, transformers, datasets, trl, sgfmill" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ Missing packages${NC}"
        echo "   Run: source .venv/bin/activate && uv sync --locked"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check CUDA
    echo -n "3. PyTorch CUDA: "
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        echo -e "${GREEN}✓ (CUDA ${CUDA_VERSION})${NC}"
    else
        echo -e "${RED}✗ Not available${NC}"
        echo "   PyTorch may not be installed with CUDA support"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check 4: KataGo
echo -n "4. KataGo binary: "
if [ -f .env ]; then
    source .env 2>/dev/null || true
fi

if [ -n "${KATAGO_BIN:-}" ] && [ -f "$KATAGO_BIN" ]; then
    if $KATAGO_BIN version &>/dev/null; then
        KATAGO_VER=$($KATAGO_BIN version | head -n1)
        echo -e "${GREEN}✓ (${KATAGO_VER})${NC}"
    else
        echo -e "${YELLOW}⚠ Binary found but not working${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${RED}✗ Not found${NC}"
    echo "   Run: ./scripts/setup_environment.sh"
    ERRORS=$((ERRORS + 1))
fi

# Check 5: KataGo model
echo -n "5. KataGo model: "
if [ -n "${KATAGO_MODEL:-}" ] && [ -f "$KATAGO_MODEL" ]; then
    SIZE=$(du -h "$KATAGO_MODEL" | cut -f1)
    echo -e "${GREEN}✓ (${SIZE})${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "   Run: ./scripts/setup_environment.sh"
    ERRORS=$((ERRORS + 1))
fi

# Check 6: KataGo config
echo -n "6. KataGo config: "
if [ -n "${KATAGO_CONFIG:-}" ] && [ -f "$KATAGO_CONFIG" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "   Run: ./scripts/setup_environment.sh"
    ERRORS=$((ERRORS + 1))
fi

# Check 7: SGF files
echo -n "7. SGF files (9x9): "
SGF_9x9_COUNT=$(find data/sgf_9x9 -name "*.sgf" 2>/dev/null | wc -l | tr -d ' ')
if [ "$SGF_9x9_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ ($SGF_9x9_COUNT files)${NC}"
else
    echo -e "${YELLOW}⚠ No files found${NC}"
    echo "   Add SGF files to data/sgf_9x9/ or run: ./scripts/download_sample_sgf.sh"
    WARNINGS=$((WARNINGS + 1))
fi

echo -n "8. SGF files (19x19): "
SGF_19x19_COUNT=$(find data/sgf_19x19 -name "*.sgf" 2>/dev/null | wc -l | tr -d ' ')
if [ "$SGF_19x19_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ ($SGF_19x19_COUNT files)${NC}"
else
    echo -e "${YELLOW}⚠ No files found${NC}"
    echo "   Add SGF files to data/sgf_19x19/ (optional for 19x19 training)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 9: Training datasets
echo -n "9. Training datasets: "
DATASETS=0
[ -f "data/sft_9x9.jsonl" ] && DATASETS=$((DATASETS + 1))
[ -f "data/rl_prompts_9x9.jsonl" ] && DATASETS=$((DATASETS + 1))
[ -f "data/sft_19x19.jsonl" ] && DATASETS=$((DATASETS + 1))
[ -f "data/rl_prompts_19x19.jsonl" ] && DATASETS=$((DATASETS + 1))

if [ $DATASETS -gt 0 ]; then
    echo -e "${GREEN}✓ ($DATASETS found)${NC}"
elif [ $SGF_9x9_COUNT -gt 0 ] || [ $SGF_19x19_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠ Not prepared yet${NC}"
    echo "   Run: ./scripts/prepare_data.sh"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${YELLOW}⚠ No datasets or SGF files${NC}"
    WARNINGS=$((WARNINGS + 1))
fi

# Optional artifact: full-game GRPO model
echo -n "9b. Full-game GRPO artifact (optional): "
if [ -d "outputs/grpo_fullgame_19x19" ]; then
    echo -e "${GREEN}✓ Found outputs/grpo_fullgame_19x19${NC}"
else
    echo -e "${YELLOW}⚠ Not found (optional)${NC}"
fi

# Check 10: GPU
echo -n "10. NVIDIA GPU: "
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo -e "${GREEN}✓ (${GPU_NAME}, ${GPU_MEM})${NC}"
else
    echo -e "${RED}✗ nvidia-smi not found${NC}"
    echo "   GPU is required for training"
    ERRORS=$((ERRORS + 1))
fi

# Check 11: Executable scripts
echo -n "11. Shell scripts: "
EXEC_COUNT=0
for script in scripts/setup_environment.sh scripts/prepare_data.sh scripts/train_sft.sh scripts/train_grpo.sh scripts/run_full_pipeline.sh; do
    [ -x "$script" ] && EXEC_COUNT=$((EXEC_COUNT + 1))
done

for script in scripts/train_grpo_fullgame.sh; do
    [ -x "$script" ] && EXEC_COUNT=$((EXEC_COUNT + 1))
done

if [ $EXEC_COUNT -eq 6 ]; then
    echo -e "${GREEN}✓ All executable${NC}"
else
    echo -e "${YELLOW}⚠ Some not executable${NC}"
    echo "   Run: chmod +x scripts/*.sh"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 12: Config files
echo -n "12. Config files: "
if [ -f "config/sft_warm_start.yaml" ] && [ -f "config/grpo_9x9.yaml" ] && [ -f "config/grpo_fullgame_19x19.yaml" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ Missing configs${NC}"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "=========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ Setup is complete!${NC}"
    echo ""
    echo "You can now run training:"
    echo "  ./scripts/train_sft.sh --smoke    # Test SFT"
    echo "  ./scripts/train_grpo.sh --smoke   # Test GRPO"
    echo "  ./scripts/train_grpo_fullgame.sh --smoke --fullgame-smoke-plies 20   # Test full-game GRPO"
    echo "  ./scripts/run_full_pipeline.sh    # Full pipeline"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Setup mostly complete with $WARNINGS warning(s)${NC}"
    echo ""
    echo "You can proceed with training, but consider:"
    echo "  - Adding more SGF files for better training"
    echo "  - Running ./scripts/prepare_data.sh to create datasets"
else
    echo -e "${RED}✗ Setup incomplete: $ERRORS error(s), $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before training."
    echo "Start with: ./scripts/setup_environment.sh"
    exit 1
fi

echo "=========================================="
