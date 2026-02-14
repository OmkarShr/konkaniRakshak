#!/bin/bash
# Setup STT Service Environment
# Creates a separate conda environment for NeMo to avoid dependency conflicts

set -e

echo "=========================================="
echo "Setup: NeMo STT Service Environment"
echo "=========================================="
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Install Anaconda/Miniconda first."
    exit 1
fi

source $(conda info --base)/etc/profile.d/conda.sh

# Create STT environment
if conda env list | grep -q "konkani-stt"; then
    echo "✓ Environment 'konkani-stt' already exists"
    echo "  To recreate: conda env remove -n konkani-stt && ./setup_stt_env.sh"
else
    echo "Creating 'konkani-stt' environment..."
    conda create -n konkani-stt python=3.10 -y
fi

echo ""
echo "Activating environment..."
conda activate konkani-stt

echo ""
echo "Installing PyTorch 2.1.0..."
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing NeMo toolkit..."
pip install nemo-toolkit[asr]==2.6.1

echo ""
echo "Installing Flask and utilities..."
pip install flask aiohttp loguru numpy

echo ""
echo "=========================================="
echo "✓ STT Environment Ready!"
echo "=========================================="
echo ""
echo "To start the STT service:"
echo "  ./start_stt_service.sh"
echo ""
echo "STT service will run on: http://localhost:50051"
echo ""
