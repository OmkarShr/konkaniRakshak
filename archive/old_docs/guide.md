# Nagar Rakshak -- Start/Stop Guide

## Architecture Overview

```
Browser (localhost:8080/realtime.html)
   |
   |-- HTTP --> web_backend.py (host, port 8080) -- serves HTML files
   |
   |-- WebSocket --> ws_pipeline.py (inside konkani-pipeline-1, port 8765)
                        |
                        +--> Silero VAD (voice activity detection)
                        +--> STT (konkani-stt-1, port 50051)
                        +--> Gemini LLM (Google API)
                        +--> Indic Parler-TTS (GPU)
```

**Three things need to be running:**
1. `konkani-stt-1` -- STT container (GPU 0)
2. `konkani-pipeline-1` -- Pipeline container running ws_pipeline.py (GPU 0)
3. `web_backend.py` -- Host process serving the HTML UI on port 8080

**Optional (for 2 concurrent sessions):**
- `konkani-stt-2` + `konkani-pipeline-2` on GPU 1

---

## Quick Stop (when not using)

Stop everything with one command:

```bash
cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak
docker compose -f docker-compose.prod.yml stop
pkill -f web_backend.py
```

This **stops** containers but keeps them (fast restart). Frees all GPU VRAM and RAM.

To also free disk space from container state (not needed normally):

```bash
docker compose -f docker-compose.prod.yml down
```

---

## Quick Start (bring it back up)

### Step 1: Start containers

```bash
cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak
docker compose -f docker-compose.prod.yml start stt-service-1 pipeline-1
```

Wait for pipeline to be ready (~1-2 min):

```bash
docker logs -f konkani-pipeline-1
# Wait until you see: "Waiting for browser connections ..."
# Press Ctrl+C to stop watching
```

### Step 2: Start web backend

```bash
cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak
nohup python3 web_backend.py > /tmp/web_backend.log 2>&1 &
```

### Step 3: Fix Bluetooth mic (if using Sony WH-1000XM4)

The headset defaults to A2DP mode (output only, no mic). Switch to HFP:

```bash
pactl set-card-profile bluez_card.80_99_E7_5E_45_84 headset-head-unit-msbc
pactl set-default-source bluez_input.80_99_E7_5E_45_84.0
```

Verify mic is available:

```bash
pactl list sources short
# Should show: bluez_input.80_99_E7_5E_45_84.0 ... RUNNING
```

### Step 4: Open the UI

Open in browser: **http://localhost:8080/realtime.html**

> Must use `localhost`, not an IP address -- browser requires HTTPS
> for microphone access on non-localhost origins.

---

## Full Start (from scratch / after reboot / after `down`)

If containers were removed (via `docker compose down`) or after a reboot:

```bash
cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak
docker compose -f docker-compose.prod.yml up -d stt-service-1 pipeline-1
```

This recreates containers. The pipeline container will:
- Auto-login to HuggingFace (using HF_TOKEN from .env)
- Auto-download the TTS model on first run (~4GB, takes ~7 min)
- Auto-start `ws_pipeline.py`

Then follow Steps 2-4 from Quick Start above.

To start ALL services (including GPU 1 for 2-session support):

```bash
docker compose -f docker-compose.prod.yml up -d
```

---

## Restarting Just the Pipeline (after code changes to ws_pipeline.py)

Since `ws_pipeline.py` is mounted read-only into the container, edits
on the host are immediately visible. Just restart the container:

```bash
docker compose -f docker-compose.prod.yml restart pipeline-1
```

---

## Checking Status

```bash
# Container status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep konkani

# GPU memory usage
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Pipeline logs (live)
docker logs -f konkani-pipeline-1

# Pipeline logs (last 20 lines)
docker logs konkani-pipeline-1 --tail 20

# Web backend running?
pgrep -a -f web_backend.py

# Web backend logs
tail -20 /tmp/web_backend.log
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Connecting..." in browser | Pipeline container not ready | `docker logs -f konkani-pipeline-1` and wait for "Waiting for browser connections" |
| Page won't load | Web backend not running | Start web_backend.py (Step 2) |
| "Processing..." hangs | Gemini 429 rate limit | Wait 30s and try again |
| No audio playback | TTS error | `docker logs konkani-pipeline-1 --tail 30` |
| "Mic permission denied" | Bluetooth in A2DP mode / no mic | Run the Bluetooth mic fix (Step 3) |
| "Mic permission denied" | Not using localhost | Use `http://localhost:8080/realtime.html` |
| Pipeline container keeps restarting | Model download or dep error | `docker logs konkani-pipeline-1` for details |

---

## Key Ports

| Port | Service | Location |
|------|---------|----------|
| 8080 | Web UI (Flask) | Host |
| 8765 | WebSocket pipeline 1 | Container (mapped to host) |
| 8766 | WebSocket pipeline 2 | Container (mapped to host, if running) |
| 50051 | STT service 1 | Container (mapped to host) |
| 50052 | STT service 2 | Container (mapped to host, if running) |
