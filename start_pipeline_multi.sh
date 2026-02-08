#!/bin/bash
# Startup script for multi-language pipeline container (English + Hindi)
# Dependencies are pre-installed in the Docker image.

set -e

echo "=== Multi-language Pipeline container starting ==="

# Login to HuggingFace if token is set
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null || true
fi

echo "Starting ws_pipeline_multi.py..."
exec python3 /app/ws_pipeline_multi.py
