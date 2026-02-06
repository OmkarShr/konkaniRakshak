# Quick Reference Card

## 🎯 One-Line Commands

### Start Everything (Docker - RECOMMENDED)
```bash
./docker-start.sh
```

### Stop Everything
```bash
./docker-stop.sh
```

### View Logs
```bash
./docker-logs.sh        # All
./docker-logs.sh stt    # STT only
./docker-logs.sh pipeline  # Pipeline only
```

## 🔧 Manual Commands (No Docker)

### Terminal 1: STT Service
```bash
conda activate konkani-stt
export STT_MODEL_PATH=models/indicconformer_stt_kok_hybrid_rnnt_large.nemo
python services/stt_service.py
```

### Terminal 2: Pipeline
```bash
conda activate konkani-agent
export GEMINI_API_KEY=your_key_here
./start_pipeline.sh
```

## 🧪 Testing

### Test STT Service
```bash
curl http://localhost:50051/health
curl http://localhost:50051/info
```

### Test Audio I/O
```bash
conda activate konkani-agent
python tests/test_audio_io.py
```

### Test Full Pipeline
```bash
# With Docker
docker-compose up

# Or manual
./start_pipeline.sh
```

## 🐛 Quick Fixes

### Reset STT Environment
```bash
conda env remove -n konkani-stt
./setup_stt_env.sh
```

### Reset All
```bash
conda env remove -n konkani-agent
conda env remove -n konkani-stt
./setup.sh
./setup_stt_env.sh
```

### Clean Docker
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Resource Monitoring

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Container Stats
```bash
docker stats
```

### Disk Usage
```bash
docker system df
```

## 🚀 Production Deployment

### Build Production Images
```bash
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

### Scale Services
```bash
docker-compose up -d --scale pipeline=2
```

## 📞 Troubleshooting

| Problem | Solution |
|---------|----------|
| STT won't start | Check `docker-compose logs stt-service` |
| No audio | Check `pavucontrol` (PulseAudio) |
| GPU not found | Run `nvidia-smi` on host |
| Model not found | Run `python -m konkani_agent.utils.model_download` |
| API key error | Set `GEMINI_API_KEY` in `.env` |

## 📝 Key Files

- `main.py` - Pipeline entry point
- `services/stt_service.py` - STT HTTP service
- `processors/*.py` - Pipeline components
- `config/settings.py` - Configuration
- `.env` - API keys and secrets

## 🎓 Architecture Reminder

```
Mic → VAD → STT Client → LLM → TTS → Speakers
         ↑
         └─► STT Service (HTTP:50051)
```

- **STT Service**: NeMo + protobuf 5.x (separate container/process)
- **Pipeline**: Pipecat + protobuf 4.x (main container/process)
- **Communication**: HTTP JSON API on localhost:50051

## ✅ Success Checklist

- [ ] Model downloaded (499MB)
- [ ] STT service running (http://localhost:50051/health returns OK)
- [ ] Gemini API key set
- [ ] Audio devices working (test with `test_audio_io.py`)
- [ ] Pipeline starts without errors
- [ ] Can see VAD detecting speech
- [ ] Can see STT transcribing
- [ ] Can see LLM generating responses
- [ ] Can hear TTS output (or see text if TTS disabled)

**Ready to test? Start with:**
```bash
./docker-start.sh
```
