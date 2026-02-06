import requests
import base64
import numpy as np
import time
import sys


def generate_test_audio(duration=2.0, sr=16000, freq=440.0):
    """Generate a simple sine wave audio tone."""
    t = np.linspace(0, duration, int(sr * duration), False)
    # Generate sine wave
    tone = np.sin(freq * t * 2 * np.pi)
    # Normalize to 16-bit PCM range
    audio = (tone * 32767).astype(np.int16)
    return audio.tobytes()


def test_health():
    print("Checking Health...")
    try:
        response = requests.get("http://localhost:50051/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_transcribe():
    print("\nTesting Transcription...")
    audio_bytes = generate_test_audio()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {"audio": audio_b64, "sample_rate": 16000}

    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:50051/transcribe", json=payload, timeout=30
        )
        duration = time.time() - start_time

        print(f"Status Code: {response.status_code}")
        print(f"Time taken: {duration:.2f}s")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Transcription request failed: {e}")


if __name__ == "__main__":
    if test_health():
        test_transcribe()
    else:
        print("\nSkipping transcription test because service is unhealthy.")
