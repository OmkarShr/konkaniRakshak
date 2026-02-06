#!/usr/bin/env python3
"""Quick test script for Konkani audio samples"""

import sys

sys.path.insert(
    0,
    "/home/omkar/ArchDrive/omk/Projects/GoaPolice/konkani/src",
)

import os
import base64
import requests

# Audio files in project directory
AUDIO_SAMPLES = {
    "konkani_song": "/home/omkar/ArchDrive/omk/Projects/GoaPolice/konkani/konkaniSong.mp3",
    "response": "/home/omkar/ArchDrive/omk/Projects/GoaPolice/konkani/response_1.wav",
}

STT_URL = "http://localhost:50051"


def test_stt_service():
    """Test STT service with available audio samples."""

    print("=" * 60)
    print("Konkani STT Test")
    print("=" * 60)

    # Check if STT service is running
    try:
        resp = requests.get(f"{STT_URL}/health", timeout=5)
        if resp.status_code == 200:
            print(f"✓ STT Service: {resp.json()}")
        else:
            print(f"✗ STT Service error: {resp.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to STT service: {e}")
        print("  Start it with: ./start_stt_service.sh")
        return False

    print("\nAvailable audio samples:")
    for name, path in AUDIO_SAMPLES.items():
        exists = "✓" if os.path.exists(path) else "✗"
        size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        print(f"  {exists} {name}: {path} ({size:.1f} KB)")

    # TODO: Convert and test transcription
    print("\n✓ Ready for testing!")
    print("  Use these samples to verify STT accuracy")

    return True


if __name__ == "__main__":
    success = test_stt_service()
    sys.exit(0 if success else 1)
