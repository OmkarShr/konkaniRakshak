#!/bin/bash
# Start the main Pipecat pipeline (uses STT client, not direct NeMo)

set -e

echo "Starting Konkani Agent Pipeline..."
echo ""

# Check if STT service is running
STT_URL="${STT_SERVICE_URL:-http://konkani-stt-1:50051}"
RETRY_COUNT=0
MAX_RETRIES=30
while ! curl -s "${STT_URL}/health" > /dev/null 2>&1 && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq 1 ]; then
        echo "⚠ Waiting for STT service at ${STT_URL}..."
    fi
    sleep 1
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "⚠ WARNING: STT service not responding after 30 seconds"
    echo "   Service may still be initializing, but continuing anyway..."
fi

# Activate virtual environment if it exists, otherwise use current Python
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set Python path to include src directory (for Docker, use /app)
if [ -d "/app" ]; then
    export PYTHONPATH="/app:/app/src:$PYTHONPATH"
else
    export PYTHONPATH="/home/btech/Music/NagarRakshakKonkani/konkaniRakshak/src:$PYTHONPATH"
fi

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

# Run WebSocket pipeline for browser connections
python /app/ws_pipeline.py
