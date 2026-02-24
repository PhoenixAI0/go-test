#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Go RL Training Environment Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Linux. Some features may not work on other systems.${NC}"
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. This training requires an NVIDIA GPU.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Installing uv package manager...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo -e "${GREEN}✓ uv is installed${NC}"
uv --version

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
uv venv .venv --python 3.12

# Install project environment from the committed lockfile (includes PyTorch via uv source config)
echo -e "${YELLOW}Syncing project dependencies...${NC}"
if [ ! -f "uv.lock" ]; then
    echo -e "${RED}Error: uv.lock not found in repository root.${NC}"
    echo "This setup script requires a committed lockfile."
    exit 1
fi
uv sync --locked

# Verify PyTorch installation
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

if ! uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}Error: PyTorch CUDA not available. Please check your CUDA installation.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Project dependencies synced (PyTorch CUDA available)${NC}"

# Create necessary directories
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p data/sgf_9x9
mkdir -p data/sgf_19x19
mkdir -p outputs
mkdir -p logs

echo -e "${GREEN}✓ Directories created${NC}"

# Download and install KataGo
echo -e "${YELLOW}Setting up KataGo...${NC}"
KATAGO_VERSION="1.15.3"
KATAGO_DIR="$HOME/katago"

if [ ! -d "$KATAGO_DIR" ]; then
    mkdir -p "$KATAGO_DIR"
    cd "$KATAGO_DIR"
    
    # Download KataGo binary
    echo "Downloading KataGo v${KATAGO_VERSION}..."
    wget -q https://github.com/lightvector/KataGo/releases/download/v${KATAGO_VERSION}/katago-v${KATAGO_VERSION}-cuda11.8-linux-x64.zip
    unzip -q katago-v${KATAGO_VERSION}-cuda11.8-linux-x64.zip
    chmod +x katago
    
    # Download a neural network model (b18c384nbt is good for 9x9 and 19x19)
    echo "Downloading KataGo neural network model..."
    wget -q https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s7709731328-d3715293823.bin.gz

    cd -
    echo -e "${GREEN}✓ KataGo installed to ${KATAGO_DIR}${NC}"
else
    echo -e "${GREEN}✓ KataGo already installed at ${KATAGO_DIR}${NC}"
fi

# Create split analysis configs for train/eval workloads.
cat > "${KATAGO_DIR}/analysis_train.cfg" << 'EOF'
# Fast training-time KataGo analysis config
numSearchThreads = 4
nnCacheSizePowerOfTwo = 21
nnMutexPoolSizePowerOfTwo = 16
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
maxVisits = 25
maxPlayouts = 80
EOF

cat > "${KATAGO_DIR}/analysis_eval.cfg" << 'EOF'
# Higher-quality evaluation KataGo analysis config
numSearchThreads = 8
nnCacheSizePowerOfTwo = 21
nnMutexPoolSizePowerOfTwo = 16
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
maxVisits = 100
maxPlayouts = 300
EOF

# Backward-compatible alias expected by older scripts.
cp "${KATAGO_DIR}/analysis_eval.cfg" "${KATAGO_DIR}/analysis.cfg"

# Create .env file for KataGo paths
cat > .env << EOF
# KataGo Configuration
export KATAGO_BIN="${KATAGO_DIR}/katago"
export KATAGO_MODEL="${KATAGO_DIR}/kata1-b18c384nbt-s7709731328-d3715293823.bin.gz"
export KATAGO_TRAIN_CONFIG="${KATAGO_DIR}/analysis_train.cfg"
export KATAGO_EVAL_CONFIG="${KATAGO_DIR}/analysis_eval.cfg"
export KATAGO_CONFIG="\${KATAGO_EVAL_CONFIG}"
EOF

echo -e "${GREEN}✓ Environment variables written to .env${NC}"

# Source the environment variables
source .env

# Test KataGo
echo -e "${YELLOW}Testing KataGo installation...${NC}"
if $KATAGO_BIN version &> /dev/null; then
    echo -e "${GREEN}✓ KataGo is working${NC}"
    $KATAGO_BIN version
else
    echo -e "${RED}Error: KataGo test failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Add SGF files to data/sgf_9x9/ or data/sgf_19x19/"
echo "2. Run: ./scripts/prepare_data.sh"
echo "3. Run: ./scripts/train_sft.sh (for supervised fine-tuning)"
echo "4. Run: ./scripts/train_grpo.sh (for RL training)"
echo ""
echo "Scripts use the project environment automatically via uv (no manual activation required)."
echo "If you need env vars for evaluation/training helpers, run:"
echo "  source .env"
echo ""
