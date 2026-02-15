#!/usr/bin/env python3
"""
Interactive English TTS — Type text, hear it spoken!
=====================================================
Uses ai4bharat/indic-parler-tts to read your text aloud.

Run:
  CUDA_VISIBLE_DEVICES=1 tests/.venv/bin/python tests/tts_interactive.py
"""

import sys
import os
import time
import re
import wave
import subprocess
import tempfile
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "tts_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
SAMPLE_RATE = 44100

# Mutable config so interactive_loop can update it
config = {
    "speaker_desc": (
        "Sanjay speaks with a moderate pace and a clear, "
        "close-sounding recording with no background noise."
    )
}

# ── Globals ───────────────────────────────────────────────────
model = None
tokenizer = None
desc_tokenizer = None
device = None


def load_model():
    global model, tokenizer, desc_tokenizer, device
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"\n🔧 Loading TTS model on {device} ({gpu_name}) ...")
    t0 = time.time()

    model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL_ID).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    # Warmup
    print("🔥 Warming up ...")
    _d = desc_tokenizer("Sanjay speaks clearly.", return_tensors="pt").to(device)
    _t = tokenizer("Hello", return_tensors="pt").to(device)
    _ = model.generate(
        input_ids=_d.input_ids, attention_mask=_d.attention_mask,
        prompt_input_ids=_t.input_ids, prompt_attention_mask=_t.attention_mask,
    )

    print(f"✅ Model ready in {time.time() - t0:.1f}s\n")


def synthesize(text: str):
    """Synthesize text → numpy float32 array."""
    import numpy as np

    desc_ids = desc_tokenizer(config["speaker_desc"], return_tensors="pt").to(device)
    prompt_ids = tokenizer(text, return_tensors="pt").to(device)

    gen = model.generate(
        input_ids=desc_ids.input_ids,
        attention_mask=desc_ids.attention_mask,
        prompt_input_ids=prompt_ids.input_ids,
        prompt_attention_mask=prompt_ids.attention_mask,
    )
    return gen.cpu().numpy().squeeze()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[।.?!\n])\s*', text)
    return [p for p in parts if p.strip()]


def save_and_play(wav_float, filepath: Path):
    """Convert to int16, save .wav, and play with aplay."""
    import numpy as np

    pcm = (wav_float * 32767).clip(-32768, 32767).astype(np.int16)

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    duration = len(pcm) / SAMPLE_RATE
    size_kb = filepath.stat().st_size / 1024
    print(f"   💾 Saved: {filepath.name} ({size_kb:.0f} KB, {duration:.1f}s)")

    # Play audio
    print(f"   🔊 Playing ...")
    try:
        subprocess.run(["aplay", str(filepath)], check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        # Try paplay (PulseAudio) as fallback
        try:
            subprocess.run(["paplay", str(filepath)], check=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("   ⚠️  No audio player found (tried aplay, paplay)")
            print(f"   📂 Play manually: aplay {filepath}")
    except subprocess.CalledProcessError:
        print(f"   ⚠️  Playback error — play manually: aplay {filepath}")


def interactive_loop():
    """Main REPL: type text → hear it spoken."""
    import numpy as np

    counter = 0

    print("=" * 60)
    print("  🎤 INTERACTIVE TTS — Type text, hear it spoken!")
    print("=" * 60)
    print()
    print("  Commands:")
    print("    Type any text    → synthesize and play")
    print("    'quit' or 'exit' → stop")
    print("    'voice <desc>'   → change speaker description")
    print()
    print(f"  Current voice: {config['speaker_desc']}")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print()



    while True:
        try:
            text = input("📝 Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

        # Voice change command
        if text.lower().startswith("voice "):
            config["speaker_desc"] = text[6:].strip()
            print(f"   🔄 Voice changed to: {config['speaker_desc']}")
            continue

        # Split into sentences for longer text
        sentences = split_sentences(text)
        print(f"   📖 {len(sentences)} sentence(s) to synthesize")

        all_wavs = []
        total_gen = 0.0

        for i, sentence in enumerate(sentences):
            print(f"   🔄 [{i+1}/{len(sentences)}] \"{sentence[:60]}{'...' if len(sentence)>60 else ''}\"")
            t0 = time.time()
            wav = synthesize(sentence)
            elapsed = time.time() - t0
            total_gen += elapsed
            dur = len(wav) / SAMPLE_RATE
            print(f"      Generated {dur:.1f}s audio in {elapsed:.1f}s")
            all_wavs.append(wav)

        # Concatenate all sentences
        combined = np.concatenate(all_wavs)
        total_dur = len(combined) / SAMPLE_RATE
        print(f"   ⏱️  Total: {total_dur:.1f}s audio, {total_gen:.1f}s generation")

        # Save and play
        counter += 1
        filepath = OUTPUT_DIR / f"spoken_{counter:03d}.wav"
        save_and_play(combined, filepath)
        print()


if __name__ == "__main__":
    load_model()
    interactive_loop()
