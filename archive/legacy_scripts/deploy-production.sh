#!/bin/bash
# Production deployment script for RTX Ada 4000 machines
# Run this on your production server with 2x RTX 4000 (20GB each)

set -e

PROD_DIR="/opt/konkani-agent"
BACKUP_DIR="/opt/konkani-agent-backup-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo "Konkani Agent - Production Deployment"
echo "Target: 2x RTX Ada 4000 (20GB each)"
echo "=========================================="
echo ""

# Check system requirements
echo "[1/6] Checking system requirements..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not installed"
    echo "Install: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check NVIDIA Docker runtime
if ! docker info | grep -q "nvidia"; then
    echo "Warning: NVIDIA Docker runtime may not be configured"
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPUs not detected"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "✓ Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "Warning: Expected 2 GPUs for optimal performance"
    echo "Will run with $GPU_COUNT GPU(s)"
fi

# Check environment variables
echo ""
echo "[2/6] Checking environment..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set"
    echo "Set it: export GEMINI_API_KEY=your_key_here"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set (needed for TTS model)"
    echo "Set it: export HF_TOKEN=your_token_here"
fi

echo "✓ Environment configured"

# Create production directory
echo ""
echo "[3/6] Setting up production directory..."
if [ -d "$PROD_DIR" ]; then
    echo "Existing installation found, creating backup..."
    sudo mkdir -p "$BACKUP_DIR"
    sudo cp -r "$PROD_DIR" "$BACKUP_DIR/" 2>/dev/null || true
    echo "✓ Backup created: $BACKUP_DIR"
fi

sudo mkdir -p "$PROD_DIR"
sudo chown $USER:$USER "$PROD_DIR"

# Copy project files
echo ""
echo "[4/6] Copying project files..."
cp -r . "$PROD_DIR/"
cd "$PROD_DIR"

# Download models if not present
if [ ! -f "models/indicconformer_stt_kok_hybrid_rnnt_large.nemo" ]; then
    echo ""
    echo "[5/6] Downloading STT model..."
    python3 -c "
import urllib.request
import os

url = 'https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_kok_hybrid_rnnt_large.nemo'
model_path = 'models/indicconformer_stt_kok_hybrid_rnnt_large.nemo'

os.makedirs('models', exist_ok=True)
print('Downloading model...')
urllib.request.urlretrieve(url, model_path)
print('✓ Model downloaded')
"
fi

# Build and start services
echo ""
echo "[6/6] Building and starting services..."
echo "This will take 15-20 minutes on first run..."

# Pull base images first (faster)
docker pull nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Build images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Wait for health checks
echo ""
echo "Waiting for services to be healthy..."
sleep 30

# Check health
for i in {1..2}; do
    PORT=$((50050 + i))
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✓ STT Service $i: Healthy (port $PORT)"
    else
        echo "✗ STT Service $i: Not responding"
    fi
done

echo ""
echo "=========================================="
echo "✓ Production Deployment Complete!"
echo "=========================================="
echo ""
echo "Services running:"
echo "  - STT Service #1: http://localhost:50051 (GPU 0)"
echo "  - Pipeline #1: Running (GPU 0)"
echo "  - STT Service #2: http://localhost:50052 (GPU 1)"
echo "  - Pipeline #2: Running (GPU 1)"
echo ""
echo "Concurrent sessions: 2"
echo "GPU allocation: 1 per session"
echo "VRAM per session: ~5GB (fits in 20GB)"
echo ""
echo "Management commands:"
echo "  View logs:     docker-compose -f docker-compose.prod.yml logs -f"
echo "  Stop services: docker-compose -f docker-compose.prod.yml down"
echo "  Restart:       docker-compose -f docker-compose.prod.yml restart"
echo "  Update:        ./deploy-production.sh"
echo ""
echo "Installation directory: $PROD_DIR"
echo "Backup directory: $BACKUP_DIR"
echo ""
