#!/usr/bin/env python3
"""
Hindi Pipeline Debug Test Script
=================================
Tests each component of the Hindi pipeline in isolation:
  Step 1: WebSocket connectivity to port 8768
  Step 2: STT service (multilingual-stt:50052, language=hi)
  Step 3: LLM (Ollama gemma2:2b)
  Step 4: End-to-end WebSocket round-trip (send audio → get transcription)

Run from the HOST machine:
  python3 tests/test_hindi_pipeline.py
"""

import sys
import time
import json
import base64
import struct
import numpy as np

HINDI_WS_PORT = 8768
STT_URL = "http://localhost:50052"
OLLAMA_URL = "http://localhost:11435"  # host-mapped port (11435 → 11434)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────
#  Step 1: WebSocket connectivity
# ──────────────────────────────────────────────────────────────
def test_ws_connectivity():
    sep("STEP 1: WebSocket Connectivity (ws://localhost:8768)")

    try:
        import websocket
    except ImportError:
        print(f"{FAIL} 'websocket-client' not installed. Run: pip install websocket-client")
        return False

    ws_url = f"ws://localhost:{HINDI_WS_PORT}"
    print(f"  Connecting to {ws_url} ...")

    try:
        ws = websocket.create_connection(ws_url, timeout=10)
        print(f"  {PASS} WebSocket connected!")

        # Wait for the 'ready' message
        print(f"  Waiting for 'ready' message ...")
        msg = ws.recv()
        print(f"  Received: {msg}")

        data = json.loads(msg)
        if data.get("type") == "ready":
            print(f"  {PASS} Got 'ready' — Hindi pipeline WebSocket is working!")
            ws.close()
            return True
        else:
            print(f"  {WARN} Got unexpected message type: {data.get('type')}")
            ws.close()
            return True  # Connected but unexpected message

    except websocket.WebSocketTimeoutException:
        print(f"  {FAIL} Connection timed out after 10s")
        print(f"      Is 'pipeline-hindi' container running?")
        print(f"      Check: docker compose ps pipeline-hindi")
        return False
    except ConnectionRefusedError:
        print(f"  {FAIL} Connection refused on port {HINDI_WS_PORT}")
        print(f"      Is 'pipeline-hindi' container running?")
        print(f"      Check: docker compose ps pipeline-hindi")
        return False
    except Exception as e:
        print(f"  {FAIL} WebSocket error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
#  Step 2: STT Service (multilingual model, language=hi)
# ──────────────────────────────────────────────────────────────
def test_stt_service():
    sep("STEP 2: STT Service (http://localhost:50052, language=hi)")

    try:
        import requests
    except ImportError:
        print(f"{FAIL} 'requests' not installed. Run: pip install requests")
        return False

    # 2a: Health check
    print(f"  2a. Health check ...")
    try:
        resp = requests.get(f"{STT_URL}/health", timeout=5)
        print(f"      Status: {resp.status_code}")
        print(f"      Body:   {resp.json()}")
        if resp.status_code != 200:
            print(f"  {FAIL} Health check failed!")
            return False
        print(f"  {PASS} Health OK")
    except Exception as e:
        print(f"  {FAIL} Cannot reach STT service: {e}")
        return False

    # 2b: Transcribe with a 2-second 440Hz tone (won't produce real Hindi text, 
    #     but verifies the endpoint accepts requests with language=hi)
    print(f"\n  2b. Transcribe test (2s tone, language=hi) ...")
    duration = 2.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = np.sin(440 * t * 2 * np.pi)
    audio_s16 = (tone * 32767).astype(np.int16)
    audio_b64 = base64.b64encode(audio_s16.tobytes()).decode()

    payload = {
        "audio": audio_b64,
        "sample_rate": sr,
        "language": "hi"
    }
    print(f"      Payload size: {len(audio_b64)} chars (b64)")
    print(f"      Audio duration: {duration}s")

    try:
        t0 = time.time()
        resp = requests.post(f"{STT_URL}/transcribe", json=payload, timeout=30)
        elapsed = time.time() - t0
        print(f"      Status: {resp.status_code}")
        print(f"      Time:   {elapsed:.2f}s")
        print(f"      Body:   {resp.json()}")
        if resp.status_code == 200:
            print(f"  {PASS} STT service accepted Hindi transcription request!")
            return True
        else:
            print(f"  {FAIL} STT returned status {resp.status_code}")
            return False
    except Exception as e:
        print(f"  {FAIL} STT transcribe error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
#  Step 3: LLM (Ollama gemma2:2b)
# ──────────────────────────────────────────────────────────────
def test_llm():
    sep("STEP 3: LLM - Ollama gemma2:2b (http://localhost:11435)")

    try:
        import requests
    except ImportError:
        print(f"{FAIL} 'requests' not installed")
        return False

    # 3a: Check Ollama is reachable
    print(f"  3a. Ollama health check ...")
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        print(f"      Status: {resp.status_code}")
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"      Available models: {models}")
        if not any("gemma2" in m for m in models):
            print(f"  {WARN} gemma2:2b not found in model list!")
        else:
            print(f"  {PASS} gemma2 model found")
    except Exception as e:
        print(f"  {FAIL} Cannot reach Ollama: {e}")
        return False

    # 3b: Chat test
    print(f"\n  3b. Chat test (Hindi system prompt) ...")
    messages = [
        {"role": "system", "content": "Always respond in Hindi. Keep responses very short (1 sentence)."},
        {"role": "user", "content": "Hello, how are you?"}
    ]

    try:
        t0 = time.time()
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": "gemma2:2b", "messages": messages, "stream": False},
            timeout=60
        )
        elapsed = time.time() - t0
        print(f"      Status:  {resp.status_code}")
        print(f"      Time:    {elapsed:.2f}s")
        if resp.status_code == 200:
            data = resp.json()
            reply = data.get("message", {}).get("content", "")
            print(f"      Reply:   {reply[:200]}")
            print(f"  {PASS} LLM responded successfully!")
            return True
        else:
            print(f"      Body: {resp.text[:300]}")
            print(f"  {FAIL} LLM returned status {resp.status_code}")
            return False
    except Exception as e:
        print(f"  {FAIL} LLM chat error: {e}")
        return False


# ──────────────────────────────────────────────────────────────
#  Step 4: End-to-end WebSocket test
# ──────────────────────────────────────────────────────────────
def test_e2e():
    sep("STEP 4: End-to-End WebSocket Test")

    try:
        import websocket
    except ImportError:
        print(f"{FAIL} 'websocket-client' not installed")
        return False

    ws_url = f"ws://localhost:{HINDI_WS_PORT}"
    print(f"  Connecting to {ws_url} ...")

    try:
        ws = websocket.create_connection(ws_url, timeout=10)
        # Receive ready
        msg = ws.recv()
        print(f"  Got: {msg}")

        # Send 3 seconds of speech-like audio (mix of tones to trigger VAD)
        print(f"  Sending 3s of test audio ...")
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), False)
        # Create a more speech-like signal (multiple harmonics)
        audio = (
            0.5 * np.sin(200 * t * 2 * np.pi) +
            0.3 * np.sin(400 * t * 2 * np.pi) +
            0.2 * np.sin(800 * t * 2 * np.pi)
        )
        audio_s16 = (audio * 16000).astype(np.int16)

        # Send in 20ms chunks (320 samples = 640 bytes)
        chunk_size = 640  # bytes (320 samples * 2 bytes)
        audio_bytes = audio_s16.tobytes()
        chunks_sent = 0

        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            ws.send_binary(chunk)
            chunks_sent += 1
            time.sleep(0.015)  # ~20ms pacing

        print(f"  Sent {chunks_sent} chunks ({len(audio_bytes)} bytes)")

        # Now send 2s of silence to trigger speech end
        print(f"  Sending 2s of silence (to trigger speech end) ...")
        silence = np.zeros(sr * 2, dtype=np.int16).tobytes()
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i:i + chunk_size]
            ws.send_binary(chunk)
            time.sleep(0.015)
        print(f"  Silence sent")

        # Listen for responses for up to 60s
        print(f"  Waiting for pipeline responses (up to 60s) ...")
        ws.settimeout(60)
        received_types = []
        try:
            while True:
                msg = ws.recv()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    msg_type = data.get("type", "unknown")
                    received_types.append(msg_type)
                    print(f"    <- {data}")

                    if msg_type == "turn_done":
                        print(f"\n  {PASS} Full turn completed!")
                        break
                    elif msg_type == "error":
                        print(f"\n  {FAIL} Pipeline error: {data.get('message')}")
                        break
                elif isinstance(msg, bytes):
                    print(f"    <- [binary audio: {len(msg)} bytes]")
        except websocket.WebSocketTimeoutException:
            print(f"\n  {WARN} Timed out waiting for response")
            if received_types:
                print(f"      Messages received so far: {received_types}")
            else:
                print(f"      No messages received — VAD may not have detected speech")
                print(f"      (This is expected for synthetic tones — real speech needed)")

        print(f"\n  Message types received: {received_types}")
        ws.close()

        if "transcription" in received_types:
            print(f"  {PASS} STT transcription received!")
        if "response_text" in received_types:
            print(f"  {PASS} LLM response received!")
        if "tts_done" in received_types:
            print(f"  {PASS} TTS completed!")
        if "turn_done" in received_types:
            return True

        if "speech_too_short" in received_types:
            print(f"  {WARN} Speech was too short — VAD detected but audio was brief")
        if not received_types or received_types == []:
            print(f"  {WARN} No pipeline messages — VAD likely didn't detect speech from synthetic tone")
            print(f"      This does NOT mean the pipeline is broken")
            print(f"      Test with real speech via the browser UI")

        return len(received_types) > 0

    except Exception as e:
        print(f"  {FAIL} E2E test error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  HINDI PIPELINE DEBUG TEST")
    print("  Testing each component in isolation")
    print("=" * 60)

    results = {}

    results["Step 1: WebSocket"] = test_ws_connectivity()
    results["Step 2: STT"] = test_stt_service()
    results["Step 3: LLM"] = test_llm()

    # Only run E2E if all components pass
    if all(results.values()):
        results["Step 4: E2E"] = test_e2e()
    else:
        print(f"\n{WARN} Skipping E2E test — fix failing components first")

    sep("SUMMARY")
    for step, ok in results.items():
        icon = PASS if ok else FAIL
        print(f"  {icon}  {step}")

    print()
    all_ok = all(results.values())
    if all_ok:
        print(f"  All tests passed! Now test with real speech in the browser.")
        print(f"  Open: http://localhost:7777/realtime_multi.html")
        print(f"  Click 'Hindi' tab → click mic → speak Hindi")
        print(f"  Watch logs: docker compose logs pipeline-hindi -f")
    else:
        print(f"  Some tests failed — check the output above for details.")

    sys.exit(0 if all_ok else 1)
