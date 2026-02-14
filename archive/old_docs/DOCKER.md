# Docker Setup Guide

## 🐳 Why Docker?

Docker completely isolates the dependency conflicts:
- **STT Service Container**: NeMo 1.21.0 + all its dependencies (protobuf 5.x compatible)
- **Pipeline Container**: Pipecat 0.0.49 + protobuf 4.x
- **No conflicts**: Each container has its own isolated Python environment
- **Reproducible**: Works the same on any machine with Docker

## 🚀 Quick Start with Docker

### Prerequisites

1. **Install Docker**: https://docs.docker.com/engine/install/
2. **Install Docker Compose**: https://docs.docker.com/compose/install/
3. **NVIDIA Docker Runtime** (for GPU support):
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### Step 1: Download Model

```bash
conda activate konkani-agent  # or use your existing environment
python -m konkani_agent.utils.model_download
```

### Step 2: Configure

```bash
# Copy and edit .env
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Step 3: Start with Docker

```bash
# One command to start everything
./docker-start.sh

# Or manually:
docker-compose up -d
```

This will:
1. Build both Docker images (STT service + Pipeline)
2. Start containers with proper GPU access
3. Connect them via internal network
4. Expose STT service on port 50051

### Step 4: Monitor

```bash
# View logs
./docker-logs.sh           # All services
./docker-logs.sh stt       # STT service only
./docker-logs.sh pipeline  # Pipeline only

# Check status
docker-compose ps

# Test STT service
curl http://localhost:50051/health
```

### Step 5: Stop

```bash
./docker-stop.sh
# Or: docker-compose down
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                           │
│                                                             │
│  ┌─────────────────────┐      ┌──────────────────────┐     │
│  │   stt-service       │      │    pipeline          │     │
│  │   (NeMo 1.21.0)     │◀────▶│    (Pipecat)         │     │
│  │   Port: 50051       │ HTTP │    Audio I/O         │     │
│  │   GPU: cuda:0       │      │    GPU: cuda:0       │     │
│  └─────────────────────┘      └──────────────────────┘     │
│           │                            │                    │
│           │                            │                    │
│     localhost:50051              Host Audio                │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Docker Files

- `Dockerfile.stt` - STT service with NeMo
- `Dockerfile.pipeline` - Pipecat pipeline
- `docker-compose.yml` - Orchestrates both services
- `docker-start.sh` - One-click start script
- `docker-stop.sh` - Stop script
- `docker-logs.sh` - Log viewer
- `.dockerignore` - Excludes unnecessary files from build

## 🔧 Manual Docker Commands

### Build Images

```bash
# Build STT service
docker build -f Dockerfile.stt -t konkani-stt .

# Build pipeline
docker build -f Dockerfile.pipeline -t konkani-pipeline .
```

### Run Containers

```bash
# Run STT service
docker run -d \
  --name stt-service \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -p 50051:50051 \
  -v $(pwd)/models:/app/models:ro \
  konkani-stt

# Run pipeline
docker run -it \
  --name pipeline \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e STT_SERVICE_URL=http://host.docker.internal:50051 \
  --device /dev/snd \
  -v $(pwd):/app:ro \
  konkani-pipeline
```

### Debug Containers

```bash
# Enter STT container
docker exec -it stt-service bash

# Check STT service
curl http://localhost:50051/health
curl http://localhost:50051/info

# View GPU usage
nvidia-smi

# Check logs
docker logs stt-service
docker logs pipeline
```

## 🐛 Troubleshooting

### "Cannot connect to Docker daemon"
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### "NVIDIA Docker runtime not found"
```bash
# Install nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### "GPU not available in container"
```bash
# Test NVIDIA runtime
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### "STT service fails to start"
```bash
# Check logs
docker-compose logs stt-service

# Rebuild
docker-compose down
docker-compose build --no-cache stt-service
docker-compose up -d
```

### "Audio not working in container"
```bash
# Grant permissions
sudo usermod -aG audio $USER

# Or run privileged (less secure)
docker run --privileged --device /dev/snd ...
```

## 📊 Resource Usage

**STT Service Container:**
- Base image: ~8GB
- Model: 499MB
- Runtime VRAM: ~2GB

**Pipeline Container:**
- Base image: ~6GB
- Runtime VRAM: ~3-4GB (STT client, VAD, LLM, TTS)

**Total for both:**
- Disk: ~14GB
- VRAM: ~6GB

## 🎯 Production Deployment

For production on your 2x RTX Ada 4000 machines:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  stt-service-1:
    build:
      context: .
      dockerfile: Dockerfile.stt
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0  # GPU 0
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "50051:50051"
    
  stt-service-2:
    build:
      context: .
      dockerfile: Dockerfile.stt
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=1  # GPU 1
      - CUDA_VISIBLE_DEVICES=1
    ports:
      - "50052:50051"
    
  pipeline-1:
    build:
      context: .
      dockerfile: Dockerfile.pipeline
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - STT_SERVICE_URL=http://stt-service-1:50051
    depends_on:
      - stt-service-1
      
  pipeline-2:
    build:
      context: .
      dockerfile: Dockerfile.pipeline
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - STT_SERVICE_URL=http://stt-service-2:50051
    depends_on:
      - stt-service-2
```

This runs 2 concurrent sessions (1 per GPU).

## ✅ Advantages Over Conda

| Feature | Conda | Docker |
|---------|-------|--------|
| Dependency isolation | ❌ Conflicts | ✅ Complete isolation |
| Reproducibility | ⚠️ Machine-specific | ✅ Same everywhere |
| Deployment | ❌ Complex setup | ✅ Single command |
| Rollback | ❌ Hard | ✅ Easy (docker images) |
| Multiple versions | ❌ Hard | ✅ Easy (tags) |
| Production scaling | ⚠️ Manual | ✅ Kubernetes-ready |

## 🚀 Next Steps

1. **Install Docker** on your system
2. **Run** `./docker-start.sh`
3. **Wait** 10-15 minutes for first build
4. **Test** with `./docker-logs.sh`
5. **Deploy** to RTX Ada 4000 machines with same commands!

**Ready to try Docker?**
