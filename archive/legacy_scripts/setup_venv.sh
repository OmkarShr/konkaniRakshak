#!/bin/bash
# Setup script for Konkani Conversational AI Agent using Python venv
# This creates a virtual environment instead of using conda

set -e

echo "=========================================="
echo "Konkani Agent Setup (venv-based)"
echo "=========================================="
echo ""

# Create virtual environment
if [ -d "venv" ]; then
    echo "✓ Virtual environment 'venv' already exists"
else
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate environment
echo "Activating environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install system dependencies (if on Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing system dependencies..."
    sudo apt-get update -y
    sudo apt-get install -y portaudio19-dev 2>/dev/null || echo "Note: portaudio19-dev requires manual installation if not available"
fi

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch 2.1.0 with CUDA 12.1..."
echo "This may take 10-15 minutes depending on your connection..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install Pipecat and core dependencies
echo ""
echo "Installing Pipecat and dependencies..."
pip install pipecat-ai==0.0.49 pyaudio sounddevice numpy scipy loguru python-dotenv

# Install Google Generative AI
echo "Installing Google Generative AI..."
pip install google-generativeai

# Install NeMo (for STT)
echo ""
echo "Installing NeMo toolkit (this may take a while)..."
pip install nemo-toolkit[asr]==1.23.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Install TTS (Coqui)
echo ""
echo "Installing Coqui TTS (this may take a while)..."
pip install TTS==0.22.0

# Install additional dependencies from requirements.txt
echo "Installing additional dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Download STT model: python -m konkani_agent.utils.model_download"
echo "  3. Set Gemini API key: export GEMINI_API_KEY=your_key"
echo "  4. Run the agent: python -m konkani_agent"
echo ""
