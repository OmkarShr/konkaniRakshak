#!/usr/bin/env python3
"""
Simple Konkani Voice Agent - Host Version
Uses your Bluetooth headphones directly
"""

import subprocess
import requests
import base64
import time
import os

def main():
    print("="*70)
    print("🎤 KONKANI VOICE AGENT - LIVE TEST")
    print("="*70)
    print()
    print("This captures audio from your Sony WH-1000XM4 headphones")
    print()
    
    # Record from Bluetooth headphones
    print("🎤 Recording... SPEAK NOW! (5 seconds)")
    subprocess.run([
        "arecord", "-D", "pulse", "-d", "5",
        "-f", "S16_LE", "-r", "16000", "-c", "1",
        "/tmp/live_recording.wav"
    ], check=True, capture_output=True)
    print("✅ Recording saved!")
    
    # Send to STT
    print("\n📝 Transcribing...")
    with open("/tmp/live_recording.wav", "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        "http://localhost:50051/transcribe",
        json={"audio": audio_b64, "sample_rate": 16000},
        timeout=30
    )
    
    result = response.json()
    text = result.get("text", "")
    
    if text:
        print(f"✅ You said: '{text}'")
        print("\n🎉 SUCCESS! Your voice was captured and transcribed!")
    else:
        print("⚠️  No speech detected. Try speaking louder.")
    
    print()
    print("="*70)

if __name__ == "__main__":
    main()
