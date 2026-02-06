#!/usr/bin/env python3
"""
NeMo STT HTTP Service

Standalone service that loads IndicConformer model and provides
HTTP API for speech-to-text transcription.

Uses AI4Bharat's NeMo fork for multilingual model support.
"""

import os
import sys
import tempfile
import wave
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
from loguru import logger
import torch

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    format="{time:HH:mm:ss} | {level: <8} | {message}",
    level="INFO",
    colorize=True,
)

# Configuration
MODEL_PATH = os.getenv(
    "STT_MODEL_PATH",
    "/app/models/indicconformer_stt_kok_hybrid_rnnt_large.nemo",
)
HOST = os.getenv("STT_HOST", "0.0.0.0")
PORT = int(os.getenv("STT_PORT", "50051"))
DEVICE = os.getenv("STT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
LANGUAGE = os.getenv("STT_LANGUAGE", "kok")  # Konkani

app = Flask(__name__)
model = None


def load_model():
    """Load IndicConformer model using AI4Bharat NeMo."""
    global model

    logger.info("=" * 60)
    logger.info("NeMo STT Service Starting (AI4Bharat)")
    logger.info("=" * 60)
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Language: {LANGUAGE}")
    logger.info(f"Port: {PORT}")
    logger.info("")

    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            sys.exit(1)

        logger.info("Loading IndicConformer model...")

        # Import AI4Bharat's NeMo ASR
        import nemo.collections.asr as nemo_asr

        # Load the multilingual hybrid model
        # AI4Bharat's fork supports tokenizer.type: multilingual
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            MODEL_PATH,
            map_location=DEVICE,
        )

        if DEVICE == "cuda" and torch.cuda.is_available():
            model = model.cuda()
            logger.info("✓ Model moved to CUDA")

        model.eval()

        # Set default language for transcription
        if hasattr(model, "cur_decoder"):
            model.cur_decoder = "rnnt"  # Use RNNT decoder

        logger.info("✓ Model loaded successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": DEVICE,
            "language": LANGUAGE,
        }
    )


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe audio data."""

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get audio data from request
        data = request.get_json()

        if "audio" not in data:
            return jsonify({"error": "No audio data provided"}), 400

        # Decode base64 audio
        import base64

        audio_bytes = base64.b64decode(data["audio"])
        sample_rate = data.get("sample_rate", 16000)
        lang = data.get("language", LANGUAGE)

        logger.info(
            f"Received audio: {len(audio_bytes)} bytes @ {sample_rate}Hz, lang={lang}"
        )

        # Convert to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0

        # Write to temporary WAV file (NeMo expects file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            with wave.open(f, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_float * 32768).astype(np.int16).tobytes())

        # Transcribe with language ID for multilingual model
        with torch.no_grad():
            try:
                # AI4Bharat model uses language_id parameter
                result = model.transcribe(
                    [wav_path],
                    batch_size=1,
                    language_id=lang,
                )
            except TypeError:
                # Fallback without language_id
                result = model.transcribe([wav_path], batch_size=1)

        # Clean up temp file
        os.unlink(wav_path)

        # Extract text from result
        text = ""
        if result:
            if isinstance(result, tuple) and len(result) > 0:
                # (hypotheses, ...) format
                text = result[0][0] if result[0] else ""
            elif isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], (list, tuple)):
                    text = result[0][0] if result[0] else ""
                else:
                    text = result[0]
            text = str(text).strip() if text else ""

        logger.info(
            f"Transcribed: {text[:50]}..." if len(text) > 50 else f"Transcribed: {text}"
        )

        return jsonify({"text": text, "language": lang, "success": True})

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/info", methods=["GET"])
def info():
    """Get model info."""
    return jsonify(
        {
            "model_path": MODEL_PATH,
            "device": DEVICE,
            "language": LANGUAGE,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    )


def main():
    """Start the service."""
    load_model()

    logger.info(f"Starting HTTP server on {HOST}:{PORT}")
    logger.info("Endpoints:")
    logger.info(f"  GET  http://{HOST}:{PORT}/health     - Health check")
    logger.info(f"  GET  http://{HOST}:{PORT}/info       - Model info")
    logger.info(f"  POST http://{HOST}:{PORT}/transcribe - Transcribe audio")
    logger.info("")
    logger.info("Press Ctrl+C to stop")

    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    main()
