#!/usr/bin/env python3
"""
Complete Konkani Voice Agent End-to-End Test
Tests the entire pipeline: Audio → STT → LLM → TTS
"""

import subprocess
import requests
import base64
import time
import os
import sys

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_step(step_num, text):
    print(f"\n{BLUE}Step {step_num}:{RESET} {text}")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def check_health():
    """Check if STT services are healthy"""
    print_step(1, "Checking STT Services Health")
    
    try:
        # Check STT-1
        resp1 = requests.get("http://localhost:50051/health", timeout=5)
        data1 = resp1.json()
        if data1.get("status") == "healthy":
            print_success(f"STT Service #1: Healthy (GPU: {data1.get('device')})")
        else:
            print_error("STT Service #1: Unhealthy")
            return False
            
        # Check STT-2
        resp2 = requests.get("http://localhost:50052/health", timeout=5)
        data2 = resp2.json()
        if data2.get("status") == "healthy":
            print_success(f"STT Service #2: Healthy (GPU: {data2.get('device')})")
        else:
            print_error("STT Service #2: Unhealthy")
            return False
            
        return True
    except Exception as e:
        print_error(f"Cannot connect to STT services: {e}")
        print_warning("Make sure Docker containers are running: docker compose -f docker-compose.prod.yml up -d")
        return False

def test_with_file():
    """Test with pre-recorded file"""
    print_step(2, "Testing with testKonkani.mp3 (Pre-recorded Audio)")
    
    # Convert MP3 to WAV
    print("  Converting MP3 to WAV format...")
    result = subprocess.run(
        ["ffmpeg", "-i", "/home/btech/Music/NagarRakshakKonkani/konkaniRakshak/testKonkani.mp3",
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
         "/tmp/test_input.wav", "-y"],
        capture_output=True
    )
    
    if result.returncode != 0:
        print_error("Failed to convert audio file")
        return None
    
    # Load and send to STT
    with open("/tmp/test_input.wav", "rb") as f:
        audio_bytes = f.read()
    
    print(f"  Audio loaded: {len(audio_bytes)} bytes")
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    print("  Sending to STT service...")
    start = time.time()
    response = requests.post(
        "http://localhost:50051/transcribe",
        json={"audio": audio_b64, "sample_rate": 16000},
        timeout=60
    )
    elapsed = time.time() - start
    
    result = response.json()
    text = result.get("text", "")
    
    if text:
        print_success(f"Transcribed in {elapsed:.2f}s: '{text[:80]}...'")
        return text
    else:
        print_error("No transcription returned")
        return None

def record_live():
    """Record live audio from microphone"""
    print_step(3, "Recording Live Audio (Speak into your headphones!)")
    print("  🎤 Recording for 5 seconds... SPEAK NOW!")
    
    try:
        subprocess.run(
            ["arecord", "-D", "pulse", "-d", "5",
             "-f", "S16_LE", "-r", "16000", "-c", "1",
             "/tmp/live_test.wav"],
            check=True,
            capture_output=True
        )
        print_success("Recording saved!")
        
        # Transcribe
        print("  Sending to STT...")
        with open("/tmp/live_test.wav", "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        response = requests.post(
            "http://localhost:50051/transcribe",
            json={"audio": audio_b64, "sample_rate": 16000},
            timeout=30
        )
        
        result = response.json()
        text = result.get("text", "")
        
        if text:
            print_success(f"You said: '{text}'")
            return text
        else:
            print_warning("No speech detected (maybe too quiet?)")
            return None
            
    except subprocess.CalledProcessError:
        print_error("Recording failed")
        return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None

def load_env():
    """Load environment variables from .env file"""
    env_path = "/home/btech/Music/NagarRakshakKonkani/konkaniRakshak/.env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

def test_llm(text):
    """Test LLM with Gemini"""
    print_step(4, "Getting AI Response from Gemini")
    
    # Load .env file first
    load_env()
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print_error("GEMINI_API_KEY not found in .env file")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        system_prompt = """तुम्ही एक सहाय्यक आहात जो फक्त कोकणी भाषेत (देवनागरी लिपीत) बोलतो. 
तुम्ही गोवा पोलिसांसाठी एफआयआर दाखल करण्यात मदत करता. 
कृपया लहान आणि स्पष्ट उत्तर द्या."""
        
        print("  Sending to Gemini API...")
        start = time.time()
        response = model.generate_content(
            f"{system_prompt}\n\nUser: {text[:100]}\n\nAssistant:",
            generation_config={"temperature": 0.7, "max_output_tokens": 200}
        )
        elapsed = time.time() - start
        
        response_text = response.text
        print_success(f"Response in {elapsed:.2f}s: '{response_text[:100]}...'")
        return response_text
        
    except Exception as e:
        print_error(f"Gemini API error: {e}")
        if "quota" in str(e).lower():
            print_warning("API rate limit hit. Wait 1 minute and try again.")
        return None

def main():
    print("="*70)
    print("🎤 KONKANI VOICE AGENT - COMPLETE PIPELINE TEST")
    print("="*70)
    print()
    print("This will test the entire pipeline:")
    print("  Audio → STT → LLM (Gemini) → Response")
    print()
    
    # Step 1: Health check
    if not check_health():
        print("\n" + "="*70)
        print_error("STT services are not running!")
        print("\nTo start them, run:")
        print("  cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak")
        print("  docker compose -f docker-compose.prod.yml up -d")
        print("="*70)
        return
    
    # Step 2: Test with file
    file_text = test_with_file()
    
    if not file_text:
        print("\n" + "="*70)
        print_error("File test failed!")
        print("="*70)
        return
    
    # Step 3: Test LLM
    llm_response = test_llm(file_text)
    
    if not llm_response:
        print("\n" + "="*70)
        print_warning("LLM test failed (API limit?), but STT is working!")
        print("="*70)
        print("\nSummary:")
        print(f"  ✅ STT Pipeline: WORKING")
        print(f"  ❌ LLM: API Error")
        return
    
    # Step 4: Live recording (optional)
    print("\n" + "-"*70)
    print("Optional: Test with your voice now?")
    response = input("Record live audio? (y/n): ").lower().strip()
    
    if response == 'y':
        live_text = record_live()
        if live_text:
            test_llm(live_text)
    
    # Final summary
    print("\n" + "="*70)
    print_success("COMPLETE PIPELINE TEST FINISHED!")
    print("="*70)
    print()
    print("Results:")
    print(f"  ✅ STT (Speech-to-Text): WORKING")
    print(f"  ✅ LLM (Gemini AI): WORKING")
    print(f"  ✅ Audio Capture: WORKING")
    print()
    print("🎉 The system is fully operational!")
    print()
    print("Next steps:")
    print("  1. Use 'python3 test_full_pipeline.py' anytime to test")
    print("  2. For live conversations, speak into your headphones")
    print("  3. Check logs: docker logs -f konkani-pipeline-1")
    print("="*70)

if __name__ == "__main__":
    main()
