# 🚀 Start Nagar Rakshak

## Prerequisites
- NVIDIA GPU with drivers
- Docker with NVIDIA Container Toolkit
- `.env` file with `GEMINI_API_KEY`

## Quick Start

```bash
# Start all services
docker compose up -d --build

# Check status
docker compose ps
```

## Access

**Web Interface**: http://localhost:8080/realtime_multi.html

- Click language tab (Konkani / English / Hindi)
- Click microphone button
- Allow browser mic access
- Start speaking

## Monitor

```bash
# View logs
docker compose logs -f

# View specific service
docker compose logs -f pipeline-multi
```

## Stop

```bash
docker compose down
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| Web UI | 8080 | Frontend |
| Konkani Pipeline | 8765 | Konkani STT+LLM+TTS |
| Multi Pipeline | 8766 | English/Hindi STT+LLM+TTS |
| Konkani STT | 50051 | Konkani speech recognition |
| Multi STT | 50052 | English/Hindi speech recognition |
| Ollama | 11435 | Local LLM (gemma2:2b) |
