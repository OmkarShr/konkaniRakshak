#!/usr/bin/env python3
"""
Konkani Conversational AI Agent with TTS
Full pipeline: Audio → STT → LLM → TTS → Audio Output
"""

import subprocess
import requests
import base64
import time
import os
import sys

# Load environment
def load_env():
    env_path = "/home/btech/Music/NagarRakshakKonkani/konkaniRakshak/.env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

def record_audio(duration=5):
    """Record audio from Bluetooth headphones"""
    print(f"\n🎤 Recording {duration} seconds...")
    print("   SPEAK NOW (in Konkani)!")
    
    try:
        subprocess.run(
            ["arecord", "-D", "pulse", "-d", str(duration),
             "-f", "S16_LE", "-r", "16000", "-c", "1",
             "/tmp/input.wav"],
            check=True, capture_output=True
        )
        return "/tmp/input.wav"
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None

def transcribe(audio_file):
    """Send to STT service"""
    print("\n📝 Transcribing...")
    
    with open(audio_file, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        "http://localhost:50051/transcribe",
        json={"audio": audio_b64, "sample_rate": 16000},
        timeout=30
    )
    
    result = response.json()
    return result.get("text", "")

def get_ai_response(text):
    """Get response from Gemini"""
    print("\n🤖 AI is thinking...")
    
    load_env()
    import google.generativeai as genai
    
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    system_prompt = """तुम्ही एक सहाय्यक आहात जो फक्त कोकणी भाषेत (देवनागरी लिपीत) बोलतो. 
तुम्ही गोवा पोलिसांसाठी एफआयआर दाखल करण्यात मदत करता.
कृपया नेहमी कोकणी भाषेत उत्तर द्या. उत्तरे लहान आणि स्पष्ट असावीत."""
    
    response = model.generate_content(
        f"{system_prompt}\n\nUser: {text}\n\nAssistant:",
        generation_config={"temperature": 0.7, "max_output_tokens": 200}
    )
    
    return response.text

def speak_text(text):
    """Generate TTS and play audio"""
    print("\n🔊 Generating speech...")
    
    try:
        # Use TTS to generate audio
        from TTS.api import TTS
        
        # Load TTS model (will download on first run)
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Generate audio
        tts.tts_to_file(
            text=text,
            file_path="/tmp/response.wav",
            speaker_wav=None,  # Use default voice
            language="mr"  # Marathi (closest to Konkani)
        )
        
        print("🔊 Playing response...")
        subprocess.run(["aplay", "/tmp/response.wav"], check=True)
        return True
        
    except Exception as e:
        print(f"⚠️  TTS error: {e}")
        print("   (TTS model needs to be downloaded first)")
        print(f"   Text response: {text}")
        return False

def main():
    print("="*70)
    print("🎤 KONKANI CONVERSATIONAL AI AGENT")
    print("="*70)
    print()
    print("Full pipeline: Audio → STT → LLM → TTS → Audio")
    print("Speak in Konkani, hear the AI respond!")
    print()
    print("Press Ctrl+C to exit")
    print("-"*70)
    
    # Check STT health
    try:
        resp = requests.get("http://localhost:50051/health", timeout=5)
        if resp.json().get("status") != "healthy":
            print("❌ STT service not healthy")
            return
        print("✅ STT Service ready")
    except Exception as e:
        print(f"❌ Cannot connect to STT: {e}")
        print("   Start services: docker compose -f docker-compose.prod.yml up -d")
        return
    
    # Main conversation loop
    turn = 1
    while True:
        try:
            print(f"\n{'='*70}")
            print(f"Turn {turn}")
            print("="*70)
            
            # 1. Record
            audio_file = record_audio(duration=5)
            if not audio_file:
                continue
            
            # 2. Transcribe
            text = transcribe(audio_file)
            if not text:
                print("⚠️  No speech detected, try again...")
                continue
            
            print(f"🗣️  You said: {text}")
            
            # 3. Get AI response
            response = get_ai_response(text)
            print(f"🤖 Agent: {response}")
            
            # 4. Speak response
            speak_text(response)
            
            turn += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()
