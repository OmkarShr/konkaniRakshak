"""Production Configuration

Environment-specific configurations for deployment.
"""

import os
from pathlib import Path
from typing import Dict, Any


# Base paths
BASE_DIR = Path(__file__).parent.parent


class ProductionConfig:
    """
    Production environment configuration.
    For deployment on 2x RTX Ada 4000 (20GB each).
    """

    # GPU Configuration
    GPU_CONFIG = {
        "stt_device": "cuda:0",  # STT on GPU 0
        "tts_device": "cuda:0",  # TTS on GPU 0
        "llm_device": "cpu",  # LLM via API (cloud)
        "vad_device": "cpu",  # VAD on CPU
    }

    # Audio Configuration
    AUDIO_CONFIG = {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_duration_ms": 20,
        "frame_size": 320,
        "input_device": None,  # Default
        "output_device": None,  # Default
    }

    # Pipeline Configuration
    PIPELINE_CONFIG = {
        "max_sessions": 2,  # 2 concurrent sessions
        "session_timeout_seconds": 600,
        "allow_interruptions": True,
        "latency_target_ms": 1000,
        "enable_streaming": True,
        "enable_caching": True,
    }

    # STT Configuration
    STT_CONFIG = {
        "model_path": str(
            BASE_DIR / "models" / "indicconformer_stt_kok_hybrid_rnnt_large.nemo"
        ),
        "language": "kok",
        "device": "cuda",
        "fp16": True,
        "beam_width": 5,
        "use_lm": True,
    }

    # TTS Configuration (Enhanced)
    TTS_CONFIG = {
        "primary_model": "xtts",
        "fallback_model": "parler",
        "language": "mr",
        "device": "cuda",
        "fp16": True,
        "sample_rate": 24000,
        "enable_fallback_audio": True,
        "warmup": True,
    }

    # LLM Configuration
    LLM_CONFIG = {
        "provider": "gemini",  # or "ollama" for local
        "model": "gemini-1.5-flash",  # Faster for production
        "temperature": 0.5,  # Lower for consistency
        "max_tokens": 256,  # Shorter responses
        "timeout": 5.0,  # 5 second timeout
    }

    # VAD Configuration
    VAD_CONFIG = {
        "model": "silero_vad",
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 600,  # Slightly longer for production
        "sample_rate": 16000,
    }

    # Barge-In Configuration
    BARGE_IN_CONFIG = {
        "enabled": True,
        "vad_threshold": 0.5,
        "min_interrupt_duration_ms": 400,
        "debounce_ms": 100,
        "enable_echo_cancellation": True,
    }

    # GPU Memory Configuration
    MEMORY_CONFIG = {
        "monitoring_enabled": True,
        "check_interval": 1.0,
        "warning_threshold_mb": 10240,  # 10GB on 20GB GPU
        "critical_threshold_mb": 15360,  # 15GB
        "emergency_threshold_mb": 18432,  # 18GB
        "auto_optimize": True,
    }

    # Error Handling
    ERROR_CONFIG = {
        "max_retries": 3,
        "base_retry_delay": 1.0,
        "enable_fallback_audio": True,
        "stop_on_critical": True,
    }

    # Monitoring
    MONITORING_CONFIG = {
        "enabled": True,
        "log_level": "INFO",
        "metrics_interval": 60,
        "dashboard_enabled": True,
        "dashboard_port": 8080,
    }

    # Logging
    LOG_CONFIG = {
        "level": "INFO",
        "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        "sink": BASE_DIR / "logs" / "production.log",
        "rotation": "100 MB",
        "retention": "1 week",
    }

    # Security
    SECURITY_CONFIG = {
        "rate_limit": 60,  # requests per minute
        "session_timeout": 300,  # 5 minutes
        "max_utterance_length": 30,  # seconds
    }


class DevelopmentConfig:
    """
    Development environment configuration.
    For testing on RTX 4050 (8GB).
    """

    GPU_CONFIG = {
        "stt_device": "cuda:0",
        "tts_device": "cuda:0",
        "llm_device": "cpu",
        "vad_device": "cpu",
    }

    AUDIO_CONFIG = ProductionConfig.AUDIO_CONFIG.copy()

    PIPELINE_CONFIG = {
        "max_sessions": 1,
        "session_timeout_seconds": 300,
        "allow_interruptions": True,
        "latency_target_ms": 1500,  # More lenient for dev
        "enable_streaming": True,
        "enable_caching": True,
    }

    STT_CONFIG = ProductionConfig.STT_CONFIG.copy()

    TTS_CONFIG = {
        **ProductionConfig.TTS_CONFIG,
        "fp16": True,  # Force FP16 for 8GB GPU
    }

    LLM_CONFIG = {
        **ProductionConfig.LLM_CONFIG,
        "model": "gemini-2.0-flash-exp",
        "temperature": 0.7,
        "max_tokens": 512,
    }

    VAD_CONFIG = ProductionConfig.VAD_CONFIG.copy()

    BARGE_IN_CONFIG = ProductionConfig.BARGE_IN_CONFIG.copy()

    MEMORY_CONFIG = {
        "monitoring_enabled": True,
        "check_interval": 1.0,
        "warning_threshold_mb": 5120,  # 5GB on 8GB GPU
        "critical_threshold_mb": 6144,  # 6GB
        "emergency_threshold_mb": 7168,  # 7GB
        "auto_optimize": True,
    }

    ERROR_CONFIG = ProductionConfig.ERROR_CONFIG.copy()

    MONITORING_CONFIG = {
        **ProductionConfig.MONITORING_CONFIG,
        "log_level": "DEBUG",
    }

    LOG_CONFIG = {
        **ProductionConfig.LOG_CONFIG,
        "level": "DEBUG",
    }

    SECURITY_CONFIG = ProductionConfig.SECURITY_CONFIG.copy()


# Configuration selector
def get_config(environment: str = None) -> Dict[str, Any]:
    """Get configuration for specified environment."""
    env = environment or os.getenv("KONKANI_ENV", "development")

    if env.lower() == "production":
        config_class = ProductionConfig
    else:
        config_class = DevelopmentConfig

    return {
        "gpu": config_class.GPU_CONFIG,
        "audio": config_class.AUDIO_CONFIG,
        "pipeline": config_class.PIPELINE_CONFIG,
        "stt": config_class.STT_CONFIG,
        "tts": config_class.TTS_CONFIG,
        "llm": config_class.LLM_CONFIG,
        "vad": config_class.VAD_CONFIG,
        "barge_in": config_class.BARGE_IN_CONFIG,
        "memory": config_class.MEMORY_CONFIG,
        "error": config_class.ERROR_CONFIG,
        "monitoring": config_class.MONITORING_CONFIG,
        "log": config_class.LOG_CONFIG,
        "security": config_class.SECURITY_CONFIG,
        "environment": env,
    }


# Current configuration
CURRENT_CONFIG = get_config()

# Export specific configs for convenience
GPU_CONFIG = CURRENT_CONFIG["gpu"]
AUDIO_CONFIG = CURRENT_CONFIG["audio"]
PIPELINE_CONFIG = CURRENT_CONFIG["pipeline"]
STT_CONFIG = CURRENT_CONFIG["stt"]
TTS_CONFIG = CURRENT_CONFIG["tts"]
LLM_CONFIG = CURRENT_CONFIG["llm"]
VAD_CONFIG = CURRENT_CONFIG["vad"]
BARGE_IN_CONFIG = CURRENT_CONFIG["barge_in"]
MEMORY_CONFIG = CURRENT_CONFIG["memory"]
ERROR_CONFIG = CURRENT_CONFIG["error"]
MONITORING_CONFIG = CURRENT_CONFIG["monitoring"]
LOG_CONFIG = CURRENT_CONFIG["log"]
SECURITY_CONFIG = CURRENT_CONFIG["security"]
