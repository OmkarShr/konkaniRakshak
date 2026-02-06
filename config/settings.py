"""Configuration for Konkani Agent"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# Audio Configuration
AUDIO_CONFIG = {
    "sample_rate": 16000,  # Standard for STT
    "channels": 1,
    "chunk_duration_ms": 20,  # 20ms = 320 samples at 16kHz
    "frame_size": 320,
}

# STT Configuration (IndicConformer)
STT_CONFIG = {
    "model_path": str(MODELS_DIR / "indicconformer_stt_kok_hybrid_rnnt_large.nemo"),
    "language": "kok",  # Konkani
    "device": "cuda",  # or "cpu"
    "fp16": True,  # Use FP16 for lower VRAM
}

# LLM Configuration
LLM_CONFIG = {
    # Development: Gemini API
    "dev": {
        "provider": "gemini",
        "model": "gemini-2.5-flash",  # or gemini-2.0-flash-exp
        "temperature": 0.7,
        "max_tokens": 512,
    },
    # Production: Ollama Local
    "prod": {
        "provider": "ollama",
        "model": "gpt-oss-20b",  # Placeholder - adjust based on actual model
        "temperature": 0.7,
        "max_tokens": 512,
        "base_url": "http://localhost:11434",
    },
}

# TTS Configuration (XTTSv2)
TTS_CONFIG = {
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "language": "mr",  # Marathi as proxy for Konkani
    "device": "cuda",
    "fp16": True,
    "streaming": True,
}

# VAD Configuration (Silero)
VAD_CONFIG = {
    "model": "silero_vad",
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 500,
    "sample_rate": 16000,
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    "max_sessions": 2,  # 2 sessions across 2x RTX Ada 4000
    "session_timeout_seconds": 600,  # 10 minutes
    "allow_interruptions": True,
    "latency_target_ms": 1000,  # <1s time-to-first-audio
}

# Error Handling Configuration
ERROR_CONFIG = {
    "max_retries": 3,
    "retry_delay_ms": 500,
    "fallback_message": "माफ करा, एक त्रुटी आली. कृपया पुन्हा प्रयत्न करा.",  # Konkani/Marathi fallback
    "timeout_seconds": 30,
}

# Memory/GPU Configuration
MEMORY_CONFIG = {
    "gpu_memory_threshold": 0.85,  # 85% VRAM usage triggers optimization
    "enable_fp16": True,
    "max_batch_size": 1,
    "cache_size_mb": 512,
}

# Logging
LOG_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    "sink": BASE_DIR / "logs" / "konkani_agent.log",
}
