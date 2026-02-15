#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  Standalone English STT Test — IndicConformer Multilingual
═══════════════════════════════════════════════════════════════
  Tests the NeMo IndicConformer STT model directly (no Docker).
  Uses GPU 1 exclusively. Run with:

    CUDA_VISIBLE_DEVICES=1 python tests/test_stt_english.py

  Modes:
    1. File-based test  — Transcribes a WAV/raw file
    2. Mic recording     — Records from mic and transcribes
    3. Interactive REPL  — Continuous mic-to-text loop
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import wave
import struct
import tempfile
import argparse
import numpy as np

# ── Config ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "indicconformer_stt_multi_hybrid_rnnt_600m.nemo",
)
SAMPLE_RATE = 16000
RECORD_SECONDS = 5


# ── Helpers ───────────────────────────────────────────────────
def check_gpu():
    """Verify GPU is available and report which one we're using."""
    import torch

    if not torch.cuda.is_available():
        print("❌ CUDA not available. Make sure CUDA_VISIBLE_DEVICES=1 is set.")
        sys.exit(1)

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"✅ GPU: {name} ({mem:.1f} GB) — device index {device}")
    print(f"   CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    return device


def load_model():
    """Load the NeMo IndicConformer model."""
    print(f"\n📦 Loading model: {os.path.basename(MODEL_PATH)}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        sys.exit(1)

    import nemo.collections.asr as nemo_asr

    t0 = time.time()
    model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location="cuda:0")
    model.eval()
    elapsed = time.time() - t0
    print(f"✅ Model loaded in {elapsed:.1f}s")
    return model


def save_pcm_as_wav(pcm_data: np.ndarray, filepath: str):
    """Save float32 numpy array as 16-bit WAV file."""
    pcm_int16 = (pcm_data * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())


def record_from_mic(duration: float = RECORD_SECONDS) -> np.ndarray:
    """Record audio from microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    print(f"\n🎤 Recording for {duration}s ... (speak now!)")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("✅ Recording complete.")
    return audio.squeeze()


def transcribe(model, audio_path: str) -> str:
    """Transcribe a WAV file using the model."""
    t0 = time.time()
    result = model.transcribe([audio_path])
    elapsed = time.time() - t0

    # NeMo returns different formats depending on model type
    if isinstance(result, list):
        if len(result) > 0:
            if isinstance(result[0], str):
                text = result[0]
            elif hasattr(result[0], "text"):
                text = result[0].text
            else:
                text = str(result[0])
        else:
            text = ""
    else:
        text = str(result)

    return text.strip(), elapsed


# ── Test Modes ────────────────────────────────────────────────
def test_file(model, filepath: str):
    """Test STT on an existing audio file."""
    print(f"\n📄 Transcribing file: {filepath}")
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    text, elapsed = transcribe(model, filepath)
    print(f"\n{'='*50}")
    print(f"  📝 Transcription: \"{text}\"")
    print(f"  ⏱️  Time: {elapsed:.2f}s")
    print(f"{'='*50}")


def test_mic(model, duration: float = RECORD_SECONDS):
    """Record from mic and transcribe."""
    audio = record_from_mic(duration)

    # Save to temp WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    save_pcm_as_wav(audio, tmp.name)
    print(f"   Saved to: {tmp.name}")

    # Calculate audio stats
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    print(f"   Audio stats: RMS={rms:.4f}, Peak={peak:.4f}")

    if rms < 0.001:
        print("⚠️  Audio seems very quiet — check your mic input.")

    text, elapsed = transcribe(model, tmp.name)
    print(f"\n{'='*50}")
    print(f"  📝 Transcription: \"{text}\"")
    print(f"  ⏱️  Time: {elapsed:.2f}s")
    print(f"{'='*50}")

    # Cleanup
    os.unlink(tmp.name)
    return text


def test_interactive(model):
    """Interactive REPL: continuously record and transcribe."""
    print("\n" + "="*50)
    print("  🎙️  Interactive STT Mode")
    print("  Press Enter to record, 'q' to quit")
    print("  Type a number to change recording duration (default 5s)")
    print("="*50)

    duration = RECORD_SECONDS

    while True:
        try:
            user_input = input(f"\n[{duration}s] Press Enter to record (q=quit, number=set duration): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == "q":
            break

        if user_input.isdigit():
            duration = int(user_input)
            print(f"   Duration set to {duration}s")
            continue

        try:
            dur_float = float(user_input)
            duration = dur_float
            print(f"   Duration set to {duration}s")
            continue
        except ValueError:
            pass

        text = test_mic(model, duration)


def test_synthesis_roundtrip(model):
    """Generate a known tone and verify the model does not crash."""
    print("\n🔊 Synthesis roundtrip test (silence + tone) ...")

    # Test 1: Pure silence (should return empty or very short)
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    tmp1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    save_pcm_as_wav(silence, tmp1.name)
    text1, t1 = transcribe(model, tmp1.name)
    print(f"  Silence test:  \"{text1}\"  ({t1:.2f}s)")
    os.unlink(tmp1.name)

    # Test 2: 440Hz tone (should return something or nothing, but not crash)
    t = np.linspace(0, 2, SAMPLE_RATE * 2, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    tmp2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    save_pcm_as_wav(tone, tmp2.name)
    text2, t2 = transcribe(model, tmp2.name)
    print(f"  Tone test:     \"{text2}\"  ({t2:.2f}s)")
    os.unlink(tmp2.name)

    print("✅ Roundtrip test passed (model did not crash)")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Standalone English STT Test")
    parser.add_argument("--file", "-f", help="Path to a WAV file to transcribe")
    parser.add_argument("--mic", "-m", action="store_true", help="Record from mic and transcribe")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--duration", "-d", type=float, default=RECORD_SECONDS, help="Recording duration in seconds")
    parser.add_argument("--roundtrip", "-r", action="store_true", help="Run synthesis roundtrip test")
    parser.add_argument("--all", "-a", action="store_true", help="Run all tests")
    args = parser.parse_args()

    print("═" * 56)
    print("  Standalone English STT Test — IndicConformer 600M")
    print("═" * 56)

    # Check GPU
    check_gpu()

    # Load model
    model = load_model()

    # Default to --all if no mode specified
    if not any([args.file, args.mic, args.interactive, args.roundtrip, args.all]):
        args.all = True

    # Run tests
    if args.file:
        test_file(model, args.file)

    if args.roundtrip or args.all:
        test_synthesis_roundtrip(model)

    if args.mic or args.all:
        test_mic(model, args.duration)

    if args.interactive:
        test_interactive(model)

    print("\n✅ All STT tests complete.")


if __name__ == "__main__":
    main()
