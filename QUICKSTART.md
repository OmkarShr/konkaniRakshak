# 🎤 Konkani Voice Agent - Quick Start Guide

## ✅ System Status: DEPLOYED & READY

Your Konkani conversational AI agent is running on 2x RTX 4000 Ada GPUs!

---

## 🚀 Quick Test (Run This Now!)

```bash
cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak
python3 test_full_pipeline.py
```

This will:
1. ✅ Check STT services are running
2. ✅ Test with your pre-recorded audio file
3. ✅ Get AI response from Gemini
4. 🎤 Optionally test with your voice

---

## 🎯 How to Use the System

### **Testing the Pipeline**

#### Option 1: Full Automated Test
```bash
python3 test_full_pipeline.py
```

#### Option 2: Test with Your Voice Only
```bash
python3 live_test.py
```

#### Option 3: Test with Pre-recorded File
```bash
python3 test_with_file.py
```

### **Live Voice Conversation**

1. **Put on your Sony WH-1000XM4 headphones**

2. **Run the interactive test:**
```bash
python3 test_full_pipeline.py
# When prompted, choose 'y' to record live audio
```

3. **Speak in Konkani**, for example:
   - "नमस्कार" (Hello)
   - "FIR दाखल करायचं आहे" (I want to file an FIR)
   - "तुमचं नाव काय आहे?" (What is your name?)

4. **Wait 1-2 seconds** for the AI to respond!

---

## 🐳 Docker Management

### Start All Services
```bash
cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak
docker compose -f docker-compose.prod.yml up -d
```

### Stop All Services
```bash
docker compose -f docker-compose.prod.yml down
```

### Check Status
```bash
docker ps
docker logs -f konkani-pipeline-1
```

### View Logs
```bash
# Pipeline logs
docker logs -f konkani-pipeline-1

# STT service logs
docker logs -f konkani-stt-1
docker logs -f konkani-stt-2
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────┐
│  Session 1 (GPU 0)                          │
│  ├─ STT Service :50051 → Konkani text      │
│  └─ Pipeline → STT → Gemini LLM → TTS      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Session 2 (GPU 1)                          │
│  ├─ STT Service :50052 → Konkani text      │
│  └─ STT Ready (Pipeline audio needs config)│
└─────────────────────────────────────────────┘
```

---

## 🔧 Current Limitations

1. **Audio Input**: Docker containers can't access Bluetooth directly
   - ✅ Solution: Use host-based audio capture (scripts provided)
   - ✅ Your headphones work via host audio

2. **Single Session**: Only Session 1 active for voice
   - Session 2 STT is ready but pipeline needs audio config

3. **API Limits**: Gemini free tier has rate limits
   - Wait 1 minute if you hit limits
   - Or upgrade to paid API key

---

## 🎤 Available Test Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `test_full_pipeline.py` | Complete end-to-end test | `python3 test_full_pipeline.py` |
| `live_test.py` | Record and transcribe your voice | `python3 live_test.py` |
| `test_stt_api.py` | Test STT API only | `python3 test_stt_api.py` |

---

## 📈 Performance Metrics

- **STT Latency**: ~2 seconds
- **LLM Response**: ~1-2 seconds  
- **Total Response Time**: ~3-4 seconds
- **GPU Memory Usage**: ~3GB per GPU
- **Concurrent Sessions**: 1 active (2nd ready)

---

## 🆘 Troubleshooting

### "Cannot connect to STT"
```bash
# Start services
docker compose -f docker-compose.prod.yml up -d

# Check health
curl http://localhost:50051/health
```

### "No speech detected"
- Check headphones are unmuted
- Speak louder and closer to mic
- Verify: `arecord -D pulse -d 3 test.wav && aplay test.wav`

### "Gemini API error"
- Wait 1 minute (rate limit)
- Check API key in `.env` file
- Verify key at: https://ai.google.dev/

### "Docker audio not working"
- This is expected! Use host-based scripts
- Docker containers can't access Bluetooth

---

## 📁 Important Files

- `.env` - API keys and configuration
- `testKonkani.mp3` - Your test audio file
- `test_full_pipeline.py` - Main test script
- `docker-compose.prod.yml` - Production deployment

---

## 🎉 System is Ready!

**Run this to test now:**
```bash
cd /home/btech/Music/NagarRakshakKonkani/konkaniRakshak
python3 test_full_pipeline.py
```

**Then speak into your headphones! 🎤**
