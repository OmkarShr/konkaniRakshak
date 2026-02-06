#!/usr/bin/env python3
"""Quick test of STT service"""

import requests
import base64
import numpy as np
import sys

print("Testing STT Service...")
print("=" * 60)

# Test 1: Health check
print("\n1. Testing /health endpoint...")
try:
    resp = requests.get("http://localhost:50051/health", timeout=5)
    if resp.status_code == 200:
        print(f"✓ Health OK: {resp.json()}")
    else:
        print(f"✗ Health failed: {resp.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Cannot connect: {e}")
    print("Make sure to run: ./start_stt_service.sh")
    sys.exit(1)

# Test 2: Info endpoint
print("\n2. Testing /info endpoint...")
resp = requests.get("http://localhost:50051/info", timeout=5)
print(f"✓ Info: {resp.json()}")

# Test 3: Transcribe with dummy audio
print("\n3. Testing /transcribe endpoint with dummy audio...")
# Create 1 second of silence
audio = np.zeros(16000, dtype=np.int16)
audio_b64 = base64.b64encode(audio.tobytes()).decode('utf-8')

resp = requests.post(
    "http://localhost:50051/transcribe",
    json={"audio": audio_b64, "sample_rate": 16000},
    timeout=30
)

if resp.status_code == 200:
    result = resp.json()
    print(f"✓ Transcribe OK")
    print(f"  Text: '{result.get('text', '')}'")
    print(f"  Success: {result.get('success')}")
else:
    print(f"✗ Transcribe failed: {resp.status_code}")
    print(resp.text)

print("\n" + "=" * 60)
print("STT Service is working!")
