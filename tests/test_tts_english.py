#!/usr/bin/env python3
"""
Standalone English TTS Test
============================
Tests the Indic Parler-TTS model for English synthesis — NO Docker,
NO WebSocket, NO services required.  Runs completely standalone.

Setup:
  bash tests/setup_tts_venv.sh        # one-time venv setup
  source tests/.venv/bin/activate     # Activate the venv

Run:
  python tests/test_tts_english.py
"""

import sys
import os
import time
import re
import struct
import wave
from pathlib import Path

# ── Emoji status ──────────────────────────────────────────────
PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
INFO = "ℹ️"

OUTPUT_DIR = Path(__file__).parent / "tts_output"

# ── TTS config (mirrors ws_pipeline_english.py) ──────────────
TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
TTS_SPEAKER_DESC = (
    "Sanjay speaks with a moderate pace and a clear, "
    "close-sounding recording with no background noise."
)
EXPECTED_SAMPLE_RATE = 44100


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries (same logic as ws_pipeline_english.py)."""
    parts = re.split(r'(?<=[।.?!\n])\s*', text)
    return [p for p in parts if p.strip()]


def save_wav(filepath: Path, pcm_s16, sample_rate: int):
    """Write raw int16 PCM to a .wav file."""
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_s16.tobytes())


# ==================================================================
#  STEP 1 — Check environment + imports
# ==================================================================
def test_environment():
    sep("STEP 1: Environment Check")

    checks = {
        "torch": False,
        "parler_tts": False,
        "transformers": False,
        "soundfile": False,
        "numpy": False,
    }

    for pkg in checks:
        try:
            __import__(pkg)
            checks[pkg] = True
            print(f"  {PASS} {pkg}")
        except ImportError:
            print(f"  {FAIL} {pkg} — not installed")

    if not all(checks.values()):
        print(f"\n  {FAIL} Missing packages! Run: bash tests/setup_tts_venv.sh")
        return False

    import torch
    cuda = torch.cuda.is_available()
    device = "cuda:0" if cuda else "cpu"
    print(f"\n  {INFO} PyTorch {torch.__version__}")
    print(f"  {INFO} CUDA available: {cuda}")
    if cuda:
        print(f"  {INFO} GPU: {torch.cuda.get_device_name(0)}")
        print(f"  {INFO} VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  {INFO} Using device: {device}")

    return True


# ==================================================================
#  STEP 2 — Load model + tokenizers
# ==================================================================
def test_model_load():
    sep("STEP 2: Model Loading")
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"  Loading model: {TTS_MODEL_ID} → {device} ...")
    t0 = time.time()

    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL_ID).to(device)
        tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
        desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    except Exception as e:
        print(f"  {FAIL} Failed to load model: {e}")
        return None

    elapsed = time.time() - t0
    print(f"  {PASS} Model loaded in {elapsed:.1f}s")

    # Quick sanity — check model architecture
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {INFO} Parameters: {n_params:.0f}M")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "desc_tokenizer": desc_tokenizer,
        "device": device,
    }


# ==================================================================
#  STEP 3 — Single sentence synthesis
# ==================================================================
def test_single_sentence(ctx: dict):
    sep("STEP 3: Single Sentence Synthesis")
    import numpy as np

    model = ctx["model"]
    tokenizer = ctx["tokenizer"]
    desc_tokenizer = ctx["desc_tokenizer"]
    device = ctx["device"]

    sentence = "Hello, welcome to the Goa Police assistance line."
    print(f"  Text: \"{sentence}\"")

    desc_ids = desc_tokenizer(TTS_SPEAKER_DESC, return_tensors="pt").to(device)
    prompt_ids = tokenizer(sentence, return_tensors="pt").to(device)

    print(f"  Generating ...")
    t0 = time.time()
    gen = model.generate(
        input_ids=desc_ids.input_ids,
        attention_mask=desc_ids.attention_mask,
        prompt_input_ids=prompt_ids.input_ids,
        prompt_attention_mask=prompt_ids.attention_mask,
    )
    elapsed = time.time() - t0
    wav = gen.cpu().numpy().squeeze()

    duration_s = len(wav) / EXPECTED_SAMPLE_RATE
    print(f"  {PASS} Generated {len(wav)} samples ({duration_s:.2f}s) in {elapsed:.2f}s")

    # Quality checks
    ok = True

    # Non-silent
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 0.001:
        print(f"  {FAIL} Audio appears to be silence (RMS={rms:.6f})")
        ok = False
    else:
        print(f"  {PASS} Audio is non-silent (RMS={rms:.4f})")

    # Reasonable duration (0.5s – 30s for this sentence)
    if duration_s < 0.5 or duration_s > 30.0:
        print(f"  {WARN} Duration {duration_s:.2f}s seems unusual for this sentence")
        ok = False
    else:
        print(f"  {PASS} Duration looks reasonable ({duration_s:.2f}s)")

    # Shape check
    if wav.ndim != 1:
        print(f"  {FAIL} Expected 1D array, got shape {wav.shape}")
        ok = False
    else:
        print(f"  {PASS} Output shape: {wav.shape} (1D mono)")

    ctx["single_wav"] = wav
    return ok


# ==================================================================
#  STEP 4 — Multi-sentence synthesis
# ==================================================================
def test_multi_sentence(ctx: dict):
    sep("STEP 4: Multi-Sentence Synthesis")
    import numpy as np

    model = ctx["model"]
    tokenizer = ctx["tokenizer"]
    desc_tokenizer = ctx["desc_tokenizer"]
    device = ctx["device"]

    text = (
        "Good morning. I would like to help you file a First Information Report. "
        "Could you please provide your full name? Also describe the incident briefly."
    )

    sentences = split_sentences(text)
    print(f"  Input text ({len(text)} chars) → {len(sentences)} sentence(s):")
    for i, s in enumerate(sentences):
        print(f"    [{i+1}] \"{s}\"")

    all_wavs = []
    total_time = 0.0
    ok = True

    for i, sentence in enumerate(sentences):
        desc_ids = desc_tokenizer(TTS_SPEAKER_DESC, return_tensors="pt").to(device)
        prompt_ids = tokenizer(sentence, return_tensors="pt").to(device)

        t0 = time.time()
        gen = model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        wav = gen.cpu().numpy().squeeze()
        dur = len(wav) / EXPECTED_SAMPLE_RATE
        all_wavs.append(wav)

        rms = np.sqrt(np.mean(wav ** 2))
        status = PASS if rms > 0.001 else FAIL
        if rms <= 0.001:
            ok = False
        print(f"  {status} Sentence {i+1}: {dur:.2f}s, {elapsed:.2f}s gen time, RMS={rms:.4f}")

    # Concatenate all
    combined = np.concatenate(all_wavs)
    total_dur = len(combined) / EXPECTED_SAMPLE_RATE
    print(f"\n  {INFO} Total audio: {total_dur:.2f}s")
    print(f"  {INFO} Total gen time: {total_time:.2f}s")
    print(f"  {INFO} Real-time factor: {total_time / total_dur:.2f}x")

    ctx["multi_wav"] = combined
    return ok


# ==================================================================
#  STEP 5 — Save WAV files + final report
# ==================================================================
def test_output_wav(ctx: dict):
    sep("STEP 5: WAV File Output")
    import numpy as np

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Save single sentence
    if "single_wav" in ctx:
        wav = ctx["single_wav"]
        pcm = (wav * 32767).clip(-32768, 32767).astype(np.int16)
        fpath = OUTPUT_DIR / "tts_single_sentence.wav"
        save_wav(fpath, pcm, EXPECTED_SAMPLE_RATE)
        size_kb = fpath.stat().st_size / 1024
        print(f"  {PASS} Saved: {fpath.name} ({size_kb:.0f} KB)")
        results["single"] = True
    else:
        print(f"  {WARN} No single-sentence audio to save")
        results["single"] = False

    # Save multi-sentence
    if "multi_wav" in ctx:
        wav = ctx["multi_wav"]
        pcm = (wav * 32767).clip(-32768, 32767).astype(np.int16)
        fpath = OUTPUT_DIR / "tts_multi_sentence.wav"
        save_wav(fpath, pcm, EXPECTED_SAMPLE_RATE)
        size_kb = fpath.stat().st_size / 1024
        print(f"  {PASS} Saved: {fpath.name} ({size_kb:.0f} KB)")
        results["multi"] = True
    else:
        print(f"  {WARN} No multi-sentence audio to save")
        results["multi"] = False

    print(f"\n  {INFO} Output directory: {OUTPUT_DIR}")
    print(f"  {INFO} Play with:  aplay tests/tts_output/tts_single_sentence.wav")

    return all(results.values())


# ==================================================================
#  Main
# ==================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STANDALONE ENGLISH TTS TEST")
    print("  Model: ai4bharat/indic-parler-tts")
    print("  No Docker / No WebSocket / No services needed")
    print("=" * 60)

    results = {}

    # Step 1: Environment
    results["Step 1: Environment"] = test_environment()
    if not results["Step 1: Environment"]:
        print(f"\n{FAIL} Fix environment issues first.")
        sys.exit(1)

    # Step 2: Model load
    ctx = test_model_load()
    results["Step 2: Model Load"] = ctx is not None
    if ctx is None:
        print(f"\n{FAIL} Cannot proceed without the model.")
        sys.exit(1)

    # Step 3: Single sentence
    results["Step 3: Single Sentence"] = test_single_sentence(ctx)

    # Step 4: Multi-sentence
    results["Step 4: Multi-Sentence"] = test_multi_sentence(ctx)

    # Step 5: WAV output
    results["Step 5: WAV Output"] = test_output_wav(ctx)

    # ── Summary ────────────────────────────────────────────────
    sep("SUMMARY")
    for step, ok in results.items():
        icon = PASS if ok else FAIL
        print(f"  {icon}  {step}")

    print()
    all_ok = all(results.values())
    if all_ok:
        print(f"  {PASS} All tests passed!")
        print(f"  Listen to outputs: ls tests/tts_output/")
    else:
        print(f"  {FAIL} Some tests failed — check the output above.")

    sys.exit(0 if all_ok else 1)
