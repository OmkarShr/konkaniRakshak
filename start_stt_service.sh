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
if [ -d "venv-stt" ]; then
    echo "Activating venv-stt environment..."
    source venv-stt/bin/activate
elif [ -d "venv" ]; then
    echo "Activating venv environment..."
    source venv/bin/activate
else
    echo "Error: No virtual environment found (venv-stt or venv)."
    echo "Please run: ./setup_stt_env_venv.sh"
    exit 1
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
