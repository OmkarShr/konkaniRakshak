#!/bin/bash
# Build and start all services with Docker

set -e

echo "=========================================="
echo "Konkani Agent - Docker Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found. Please install Docker first."
    echo "Visit: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check if docker compose is installed
if ! command -v docker compose &> /dev/null; then
    echo "Error: docker compose not found. Please install it."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check for model
if [ ! -f "models/indicconformer_stt_kok_hybrid_rnnt_large.nemo" ]; then
    echo "Error: IndicConformer model not found!"
    echo "Download it first: python -m konkani_agent.utils.model_download"
    exit 1
fi

# Check for .env
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found"
    echo "Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Please edit .env and add your GEMINI_API_KEY"
    fi
fi

# Build and start services
echo "Building Docker images..."
echo "This will take 10-15 minutes on first run..."
echo ""

docker compose build

echo ""
echo "Starting services..."
echo ""

# Start in detached mode
docker compose up -d

echo ""
echo "=========================================="
echo "✓ Services started!"
echo "=========================================="
echo ""
echo "STT Service: http://localhost:50051"
echo "Pipeline: Running in container 'konkani-pipeline'"
echo ""
echo "View logs:"
echo "  docker compose logs -f stt-service"
echo "  docker compose logs -f pipeline"
echo ""
echo "Stop services:"
echo "  docker compose down"
echo ""
echo "Access pipeline shell:"
echo "  docker compose exec pipeline bash"
echo ""
