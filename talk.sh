#!/bin/bash
# Conversational AI Agent with TTS
# Uses Docker for TTS generation, host for audio playback

cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak

echo "======================================================================"
echo "🎤 KONKANI CONVERSATIONAL AI AGENT WITH TTS"
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Record your voice (host)"
echo "  2. Transcribe with STT (Docker)"
echo "  3. Get AI response from Gemini (host)"
echo "  4. Generate TTS audio (Docker)"
echo "  5. Play response through headphones (host)"
echo ""
echo "Press Enter to start (or Ctrl+C to exit)..."
read

# Check STT health
echo ""
echo "1️⃣  Checking STT service..."
if curl -s http://localhost:50051/health | grep -q "healthy"; then
    echo "   ✅ STT Service ready"
else
    echo "   ❌ STT Service not running!"
    echo "   Start with: docker compose -f docker-compose.prod.yml up -d"
    exit 1
fi

# Record audio
echo ""
echo "2️⃣  Recording audio..."
echo "   🎤 SPEAK NOW (in Konkani)!"
echo "   Recording for 5 seconds..."
arecord -D pulse -d 5 -f S16_LE -r 16000 -c 1 /tmp/input.wav
if [ $? -ne 0 ]; then
    echo "   ❌ Recording failed"
    exit 1
fi
echo "   ✅ Recording saved"

# Transcribe
echo ""
echo "3️⃣  Transcribing..."
python3 << 'EOF'
import requests
import base64

with open("/tmp/input.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:50051/transcribe",
    json={"audio": audio_b64, "sample_rate": 16000},
    timeout=30
)

result = response.json()
text = result.get("text", "")

if text:
    print(f"   🗣️  You said: {text}")
    with open("/tmp/transcribed.txt", "w") as f:
        f.write(text)
else:
    print("   ⚠️  No speech detected")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "   ❌ Transcription failed"
    exit 1
fi

# Get AI response
echo ""
echo "4️⃣  Getting AI response..."
python3 << 'EOF'
import os
import sys

# Load API key
with open("/home/btech/Music/NagarRakshakKonkani/konkaniRakshak/.env", "r") as f:
    for line in f:
        if line.startswith("GEMINI_API_KEY="):
            os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

import google.generativeai as genai

with open("/tmp/transcribed.txt", "r") as f:
    user_text = f.read()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

system_prompt = "तुम्ही एक सहाय्यक आहात जो फक्त कोकणी भाषेत (देवनागरी लिपीत) बोलतो. लहान उत्तर द्या."

response = model.generate_content(
    f"{system_prompt}\n\nUser: {user_text}\n\nAssistant:",
    generation_config={"temperature": 0.7, "max_output_tokens": 200}
)

response_text = response.text
print(f"   🤖 Agent: {response_text}")

with open("/tmp/response.txt", "w") as f:
    f.write(response_text)
EOF

if [ $? -ne 0 ]; then
    echo "   ❌ AI response failed"
    exit 1
fi

# Generate TTS in Docker
echo ""
echo "5️⃣  Generating speech (TTS)..."
echo "   (This uses Docker TTS model, first run may download ~2GB)"

RESPONSE_TEXT=$(cat /tmp/response.txt)

# Use Python in Docker to generate TTS
docker exec -i konkani-pipeline-1 python3 << EOF
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="""$RESPONSE_TEXT""",
    file_path="/tmp/response.wav",
    language="mr"
)
print("TTS generated")
EOF

if [ $? -ne 0 ]; then
    echo "   ⚠️  TTS generation failed (model may need downloading)"
    echo "   Text response: $RESPONSE_TEXT"
    exit 1
fi

# Copy audio from container
docker cp konkani-pipeline-1:/tmp/response.wav /tmp/response.wav

# Play audio
echo ""
echo "6️⃣  Playing response..."
echo "   🔊 Listen through your headphones!"
aplay /tmp/response.wav

echo ""
echo "======================================================================"
echo "🎉 CONVERSATION COMPLETE!"
echo "======================================================================"
echo ""
echo "You: $(cat /tmp/transcribed.txt)"
echo "Agent: $(cat /tmp/response.txt)"
echo ""
echo "✅ Full pipeline with TTS is working!"
echo "======================================================================"
