#!/bin/bash
# Start STT Service

set -e

echo "Starting NeMo STT Service..."
echo ""

# Check if we're in the right directory
if [ ! -f "services/stt_service.py" ]; then
    echo "Error: stt_service.py not found. Run from project root."
    exit 1
fi

# Check for model
if [ ! -f "models/indicconformer_stt_kok_hybrid_rnnt_large.nemo" ]; then
    echo "Error: IndicConformer model not found!"
    echo "Download it first: python -m konkani_agent.utils.model_download"
    exit 1
fi

# Activate environment with NeMo
source $(conda info --base)/etc/profile.d/conda.sh

# Use a separate environment for STT service (NeMo with protobuf 5.x)
if conda env list | grep -q "konkani-stt"; then
    echo "Activating konkani-stt environment..."
    conda activate konkani-stt
else
    echo "Creating konkani-stt environment for STT service..."
    conda create -n konkani-stt python=3.10 -y
    conda activate konkani-stt
    
    echo "Installing dependencies..."
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    pip install nemo-toolkit[asr]==2.6.1
    pip install flask loguru numpy
fi

# Set environment variables
export STT_MODEL_PATH="models/indicconformer_stt_kok_hybrid_rnnt_large.nemo"
export STT_HOST="0.0.0.0"
export STT_PORT="50051"
export STT_DEVICE="cuda"
export STT_LANGUAGE="kok"

echo ""
echo "Configuration:"
echo "  Model: $STT_MODEL_PATH"
echo "  Device: $STT_DEVICE"
echo "  Port: $STT_PORT"
echo ""

# Start service
python services/stt_service.py
