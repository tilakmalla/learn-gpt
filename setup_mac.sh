#!/bin/bash
# =============================================================================
# GPT Learning Project - Mac M1 Setup Script
# =============================================================================
# Run this script from the project directory:
#   chmod +x setup_mac.sh && ./setup_mac.sh
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "GPT Learning Environment Setup for Mac M1"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Step 1: Check Python version
# -----------------------------------------------------------------------------
echo "Step 1: Checking Python installation..."

# Check if python3 exists
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    echo "Install via Homebrew:"
    echo "  brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "Found Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "❌ Python 3.9+ required. You have $PYTHON_VERSION"
    echo "Install via Homebrew:"
    echo "  brew install python@3.11"
    exit 1
fi

echo "✅ Python version OK"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Check if running on Apple Silicon
# -----------------------------------------------------------------------------
echo "Step 2: Checking architecture..."

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "✅ Running on Apple Silicon (arm64) - MPS acceleration available!"
else
    echo "⚠️  Running on $ARCH - MPS may not be available"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 3: Create virtual environment
# -----------------------------------------------------------------------------
echo "Step 3: Creating virtual environment..."

if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf .venv
fi

python3 -m venv .venv
echo "✅ Virtual environment created at .venv/"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Activate and install dependencies
# -----------------------------------------------------------------------------
echo "Step 4: Installing dependencies..."

source .venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install from requirements.txt (PyTorch + other dependencies)
echo "Installing from requirements.txt (this may take a few minutes)..."
pip install -r requirements.txt

echo "✅ Dependencies installed"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Download Shakespeare dataset
# -----------------------------------------------------------------------------
echo "Step 5: Downloading Shakespeare dataset..."

mkdir -p data

if [ -f "data/input.txt" ]; then
    echo "Dataset already exists."
else
    curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    echo "✅ Dataset downloaded to data/input.txt"
fi

# Show dataset info
LINES=$(wc -l < data/input.txt)
CHARS=$(wc -c < data/input.txt)
echo "   Lines: $LINES"
echo "   Characters: $CHARS"
echo ""

# -----------------------------------------------------------------------------
# Step 6: Verify MPS acceleration
# -----------------------------------------------------------------------------
echo "Step 6: Verifying MPS (GPU) acceleration..."

python3 << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"Python executable: {sys.executable}")

# Check MPS availability
mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()

print(f"MPS built: {mps_built}")
print(f"MPS available: {mps_available}")

if mps_available:
    # Quick test
    device = torch.device("mps")
    x = torch.randn(1000, 1000, device=device)
    y = torch.matmul(x, x)
    print(f"✅ MPS tensor test passed! Matrix multiplication on GPU works.")
    print(f"   Device: {device}")
else:
    print("⚠️  MPS not available. Will use CPU (slower but still works).")
    device = torch.device("cpu")
    
print(f"\nRecommended device for training: {device}")
EOF

echo ""

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo "=============================================="
echo "✅ Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  source .venv/bin/activate"
echo ""
echo "Next step: Run the verification script"
echo "  python verify_setup.py"
echo ""
echo "Then start Phase 2: Data Pipeline"
echo ""
