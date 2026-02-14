# 🐳 Run Nagar Rakshak on Docker

This guide explains how to run the entire **Nagar Rakshak** system (Multilingual AI + Web UI) using Docker.

## 📋 Prerequisites

1.  **NVIDIA GPU** with drivers installed.
2.  **Docker Desktop** or **Docker Engine** installed.
3.  **NVIDIA Container Toolkit** installed (for GPU support).

## 🚀 Quick Start

### 1. Setup Environment
Ensure you have a `.env` file with your API keys:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
nano .env
```

### 2. Start Services
Run the following command to build and start all services (STT, Backend, Web UI):

```bash
docker compose up -d --build
```

**Note:** The first run will take some time to download base images and build the containers.

### 3. Verify Status
Check if all containers are running:

```bash
docker compose ps
```

You should see:
- `konkani-stt-service` (Port 50051)
- `multilingual-stt-service` (Port 50052)
- `konkani-pipeline` (Port 8765)
- `konkani-pipeline-multi` (Port 8766)
- `konkani-web-ui` (Port 8080)

### 4. Access the Application
Open your browser and navigate to:

👉 **http://localhost:8080/realtime_multi.html**

- Select **Konkani** or **English/Hindi** tabs.
- Click the **Microphone** button to start talking.

## 🔍 Monitoring Logs

To see what's happening inside the containers:

**View All Logs:**
```bash
docker compose logs -f
```

**View Specific Service Logs:**
```bash
# Multilingual Pipeline (Logic & LLM)
docker compose logs -f pipeline-multi

# Web UI (Frontend Server)
docker compose logs -f web-ui

# STT Service (Speech-to-Text)
docker compose logs -f multilingual-stt
```

## 🛑 Stopping the System

To stop all services:

```bash
docker compose down
```

## 🛠️ Troubleshooting

- **GPU Issues**: If containers fail to start with GPU errors, ensure `nvidia-smi` works on your host and the NVIDIA Container Toolkit is installed.
- **Port Conflicts**: Ensure ports `8080`, `8765`, `8766`, `50051`, and `50052` are free.
- **Microphone**: Ensure your browser has permission to access the microphone. HTTP sites might block mic access on some browsers; use `localhost` or set up HTTPS.

---
**Enjoy your Multilingual Voice Agent!** 🎤🤖
