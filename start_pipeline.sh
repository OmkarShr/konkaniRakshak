#!/bin/bash
# Start the main Pipecat pipeline (uses STT client, not direct NeMo)

set -e

echo "Starting Konkani Agent Pipeline..."
echo ""

# Check if STT service is running
if ! curl -s http://localhost:50051/health > /dev/null 2>&1; then
    echo "⚠ WARNING: STT service not running on localhost:50051"
    echo "   Start it first: ./start_stt_service.sh (in another terminal)"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Activate main environment (Pipecat, no NeMo)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate konkani-agent

# Set Python path to include src directory
export PYTHONPATH="/home/omkar/ArchDrive/omk/Projects/GoaPolice/konkani/src:$PYTHONPATH"

# Check for Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠ WARNING: GEMINI_API_KEY not set"
    echo "   Set it with: export GEMINI_API_KEY=your_key"
    echo ""
fi

echo "Starting pipeline..."
echo "  STT: Service at localhost:50051"
echo "  LLM: Gemini API"
echo "  TTS: Coqui XTTSv2 (optional)"
echo ""

# Run pipeline
python src/konkani_agent/main.py
