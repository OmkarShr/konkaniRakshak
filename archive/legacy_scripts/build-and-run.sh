#!/bin/bash
# Build and deploy Konkani Agent with AI4Bharat Parler-TTS

set -e

echo "=========================================="
echo "Konkani Agent - AI4Bharat TTS Edition"
echo "=========================================="
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not installed"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "Warning: .env not found. Creating from example..."
    cp .env.example .env 2>/dev/null || echo "HF_TOKEN=your_token_here" > .env
    echo "Please edit .env and add your HF_TOKEN and GEMINI_API_KEY"
fi

# Load environment
export $(grep -v '^#' .env | xargs)

# Check for model
if [ ! -f "models/indicconformer_stt_kok_hybrid_rnnt_large.nemo" ]; then
    echo "Downloading STT model..."
    python -m konkani_agent.utils.model_download
fi

echo "Building Docker images..."
echo "This takes 10-15 minutes on first run..."
echo ""

# Build STT service
echo "[1/2] Building STT Service..."
docker build -f Dockerfile.stt -t konkani-stt:latest . || {
    echo "Error: STT build failed"
    exit 1
}

# Build pipeline
echo "[2/2] Building Pipeline (with AI4Bharat TTS)..."
docker build -f Dockerfile.pipeline -t konkani-pipeline:latest . || {
    echo "Error: Pipeline build failed"
    exit 1
}

echo ""
echo "✓ Build complete!"
echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "=========================================="
echo "✓ Konkani Agent is running!"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  STT Service: http://localhost:50051"
echo "  TTS Model: AI4Bharat Indic-Parler-TTS"
echo "  Voice: ${TTS_VOICE:-female}"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Test audio samples:"
echo "  python tests/test_audio_samples.py"
echo ""
echo "For production (2x RTX 4000):"
echo "  docker-compose -f docker-compose.prod.yml up -d"
echo ""
