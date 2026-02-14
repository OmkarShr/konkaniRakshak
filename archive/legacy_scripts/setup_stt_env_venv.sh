#!/bin/bash
# Setup STT Service Environment using Python venv
# Creates a separate virtual environment for NeMo to avoid dependency conflicts

set -e

echo "=========================================="
echo "Setup: NeMo STT Service Environment (venv)"
echo "=========================================="
echo ""

# Create virtual environment
if [ -d "venv-stt" ]; then
    echo "✓ Virtual environment 'venv-stt' already exists"
    echo "  To recreate: rm -rf venv-stt && ./setup_stt_env_venv.sh"
else
    echo "Creating 'venv-stt' virtual environment..."
    python3 -m venv venv-stt
fi

echo ""
echo "Activating environment..."
source venv-stt/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

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
