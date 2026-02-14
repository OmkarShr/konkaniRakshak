#!/usr/bin/env python3
"""
Konkani Voice Agent - Web Backend
Flask server that handles audio from web UI and coordinates STT/LLM/TTS
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import requests
import os
import sys
import subprocess
import tempfile
import traceback

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = "/home/btech/Music/NagarRakshakKonkani/konkaniRakshak"
WEB_UI_DIR = os.path.join(BASE_DIR, "web_ui")

# Add src to path
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

app = Flask(__name__)
CORS(app)

# ── Load .env ──────────────────────────────────────────────────────
def load_env():
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

load_env()

# ── Gemini model (loaded once) ─────────────────────────────────────
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

SYSTEM_PROMPT = (
    "तूं एक सहाय्यक आसा जो फकत कोंकणी भाशेंत (देवनागरी लिपींत) उलयता. "
    "तूं गोंय पुलिसांखातीर एफआयआर दाखल करपाक मदत करता. "
    "सदांच कोंकणी भाशेंत जाप दी. जापो मटव्यो आनी स्पश्ट आसच्यो."
)

# Conversation history per session (simple in-memory for now)
conversation_history = []

# ── TTS model (loaded once) ────────────────────────────────────────
tts_model = None
tts_tokenizer = None
tts_available = False

def initialize_tts():
    """Initialize TTS model on startup."""
    global tts_model, tts_tokenizer, tts_available
    
    try:
        print("[TTS] Initializing Parler-TTS for Marathi/Konkani...")
        from transformers import AutoModel, AutoTokenizer, set_seed
        import torch
        
        set_seed(42)
        model_name = "ai4bharat/indic-parler-tts"
        
        print(f"[TTS] Loading tokenizer from {model_name}...")
        tts_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"[TTS] Loading model from {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = AutoModel.from_pretrained(model_name).to(device)
        tts_model.eval()
        
        tts_available = True
        print(f"[TTS] ✓ TTS initialized successfully on {device}")
        
    except Exception as e:
        print(f"[TTS] ✗ Failed to initialize TTS: {e}")
        print("[TTS] Continuing without TTS (text-only responses)")
        tts_available = False

def synthesize_speech(text: str) -> bytes:
    """Synthesize Konkani/Marathi text to speech."""
    global tts_model, tts_tokenizer, tts_available
    
    if not tts_available or tts_model is None:
        return b""
    
    try:
        import torch
        import numpy as np
        
        print(f"[TTS] Synthesizing: {text[:50]}...")
        
        device = next(tts_model.parameters()).device
        
        # Prepare input
        inputs = tts_tokenizer(text, return_tensors="pt", padding=True).to(device)
        
        # Generate speech
        with torch.no_grad():
            output = tts_model.generate(**inputs)
        
        # Extract audio
        if hasattr(output, "waveform"):
            audio = output.waveform.cpu().numpy().squeeze()
        elif isinstance(output, torch.Tensor):
            audio = output.cpu().numpy().squeeze()
        else:
            audio = np.array(output)
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Normalize to [-1, 1]
        if len(audio) > 0:
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        print(f"[TTS] Generated {len(audio)} samples ({len(audio)/24000:.2f}s)")
        return base64.b64encode(audio_bytes).decode("utf-8")
        
    except Exception as e:
        print(f"[TTS] Error: {e}")
        traceback.print_exc()
        return ""

# ── Audio conversion helper ────────────────────────────────────────
def convert_webm_to_wav16k(input_bytes: bytes) -> bytes:
    """Convert browser audio (WebM/Opus or any format) to 16kHz mono PCM WAV
    using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(input_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".webm", ".wav")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_in_path,
                "-ar", "16000",
                "-ac", "1",
                "-sample_fmt", "s16",
                "-f", "wav",
                tmp_out_path,
            ],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            print(f"ffmpeg stderr: {result.stderr.decode()}")
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")

        with open(tmp_out_path, "rb") as f:
            return f.read()
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the web UI"""
    return send_from_directory(WEB_UI_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files from web_ui/"""
    return send_from_directory(WEB_UI_DIR, filename)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check -- also pings STT service."""
    try:
        resp = requests.get("http://localhost:50051/health", timeout=5)
        if resp.json().get("status") == "healthy":
            return jsonify({"status": "healthy", "stt": "connected"})
    except Exception:
        pass
    return jsonify({"status": "unhealthy", "stt": "disconnected"}), 503


@app.route("/process", methods=["POST"])
def process_audio():
    """Process audio from web UI: Audio -> convert -> STT -> LLM -> response"""
    try:
        data = request.json
        audio_base64 = data.get("audio", "")

        if not audio_base64:
            return jsonify({"error": "No audio provided"}), 400

        # ── Step 0: Convert browser audio to 16 kHz WAV ──
        print("[1/3] Converting audio to 16kHz WAV ...")
        raw_bytes = base64.b64decode(audio_base64)
        wav_bytes = convert_webm_to_wav16k(raw_bytes)
        wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        print(f"      Converted: {len(raw_bytes)} bytes -> {len(wav_bytes)} bytes WAV")

        # ── Step 1: STT ──
        print("[2/3] Sending to STT ...")
        stt_response = requests.post(
            "http://localhost:50051/transcribe",
            json={"audio": wav_b64, "sample_rate": 16000},
            timeout=30,
        )

        stt_result = stt_response.json()
        transcription = stt_result.get("text", "").strip()

        if not transcription:
            return jsonify({
                "transcription": "",
                "response_text": "म्हाका आयकूंक ना. कृपया परत सांग.",
                "response_audio": "",
            })

        print(f"      STT result: {transcription}")

        # ── Step 2: LLM ──
        print("[3/3] Getting Gemini response ...")
        conversation_history.append(f"User: {transcription}")
        # Keep last 10 turns for context
        recent = conversation_history[-10:]

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            + "\n".join(recent)
            + "\n\nAssistant:"
        )

        llm_response = gemini_model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 300},
        )

        response_text = llm_response.text.strip()
        conversation_history.append(f"Assistant: {response_text}")
        print(f"      LLM response: {response_text}")

        # ── Step 3: TTS ──
        print("[4/4] Synthesizing speech ...")
        response_audio = synthesize_speech(response_text)
        print(f"      Audio generated: {len(response_audio)} bytes")

        return jsonify({
            "transcription": transcription,
            "response_text": response_text,
            "response_audio": response_audio,
        })

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset_conversation():
    """Clear conversation history."""
    conversation_history.clear()
    return jsonify({"status": "ok"})


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Konkani Voice Agent - Web Backend")
    print("=" * 70)
    print()
    print(f"  Web UI dir : {WEB_UI_DIR}")
    print(f"  Gemini key : {'set' if os.getenv('GEMINI_API_KEY') else 'MISSING'}")
    print()
    
    # Initialize TTS
    initialize_tts()
    print()
    
    print("  Endpoints:")
    print("    GET  /          - Web UI")
    print("    GET  /health    - Health check")
    print("    POST /process   - Process audio (STT -> LLM -> TTS)")
    print("    POST /reset     - Clear conversation history")
    print()
    print("  Open http://localhost:8080 in your browser")
    print("=" * 70)

    app.run(host="0.0.0.0", port=8080, debug=False)
