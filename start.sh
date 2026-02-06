#!/bin/bash
# Quick start script for development

# Check environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "konkani-agent" ]; then
    echo "Activating konkani-agent environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate konkani-agent
fi

# Check for Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Warning: GEMINI_API_KEY not set"
    echo "Please set it: export GEMINI_API_KEY=your_key_here"
fi

# Check for STT model
MODEL_PATH="models/indicconformer_stt_kok_hybrid_rnnt_large.nemo"
if [ ! -f "$MODEL_PATH" ]; then
    echo "STT model not found. Downloading..."
    python -m konkani_agent.utils.model_download
fi

echo "Starting Konkani Agent..."
python -m konkani_agent
