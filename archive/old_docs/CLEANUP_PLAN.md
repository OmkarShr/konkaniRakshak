# Files to Keep vs Remove

## ✅ **ACTIVE FILES** (Currently Used in Production)

### Core Pipeline Files
- `ws_pipeline.py` - Konkani pipeline (Port 8765)
- `ws_pipeline_english.py` - English pipeline (Port 8767)
- `ws_pipeline_hindi.py` - Hindi pipeline (Port 8768)
- `ws_pipeline_multi.py` - Multi-language pipeline (Port 8766) - **KEEP FOR NOW**

### Startup Scripts
- `start_pipeline.sh` - Konkani startup
- `start_pipeline_english.sh` - English startup
- `start_pipeline_hindi.sh` - Hindi startup
- `start_pipeline_multi.sh` - Multi startup - **KEEP FOR NOW**

### Docker Files
- `Dockerfile.pipeline` - Pipeline container
- `Dockerfile.stt` - STT service container
- `docker-compose.yml` - Main orchestration
- `.dockerignore` - Docker ignore rules
- `.env` - Environment variables

### Service Files
- `services/stt_service.py` - STT HTTP service

### Web UI
- `web_ui/` - Frontend files

### Documentation
- `START.md` - Quick start guide
- `PRODUCTION.md` - Production deployment
- `PIPELINE_COMPARISON.md` - Pipeline comparison
- `README.md` - Main documentation

---

## ❌ **UNUSED FILES** (Safe to Remove/Archive)

### Legacy Test Files
- `conversational_agent.py` - Old test file
- `live_test.py` - Local test
- `quick_test.py` - Quick test
- `test_full_pipeline.py` - Full pipeline test
- `test_host_audio.py` - Audio test
- `quick_start.py` - Old quickstart
- `quick_start_multi.py` - Old multi quickstart
- `quick_start_multi_2.py` - Old multi quickstart v2

### Setup Scripts (Already Dockerized)
- `setup.sh` - Manual setup
- `setup_venv.sh` - Venv setup
- `setup_stt_env.sh` - STT env setup
- `setup_stt_env_venv.sh` - STT venv setup
- `start.sh` - Generic start script
- `talk.sh` - CLI talk script

### Legacy Production Files
- `run_production.py` - Old production runner (replaced by Docker)
- `deploy-production.sh` - Old deploy script
- `build-and-run.sh` - Old build script
- `docker-compose.prod.yml` - Old prod compose (merged into main)

### Unused Docker Files
- `Dockerfile.router` - Language router (not in use)
- `Dockerfile.voxtral` - Voxtral STT (not in use)

### Helper Scripts (Replaced by Docker Compose)
- `docker-start.sh` - Use `docker compose up`
- `docker-stop.sh` - Use `docker compose stop`
- `docker-logs.sh` - Use `docker compose logs`
- `start_stt_service.sh` - Handled by compose
- `start_web.sh` - Handled by compose

### Other Files
- `language_aware_router.py` - Not in current architecture
- `web_backend.py` - Not used (WebSocket-only)
- `quick_download_model.py` - One-time use
- `testKonkani.mp3` - Test audio

### Redundant Documentation
- `DOCKER.md` - Covered in PRODUCTION.md
- `QUICKREF.md` - Covered in START.md
- `QUICKSTART.md` - Covered in START.md
- `RUN_ON_DOCKER.md` - Covered in PRODUCTION.md
- `SETUP_WITHOUT_CONDA.md` - Dockerized
- `IMPLEMENTATION_COMPLETE.md` - Historical
- `guide.md` - Covered elsewhere
- `COMMANDS.txt` - Covered in docs

---

## 🗑️ **RECOMMENDED ACTION**

### Create Archive Directory
```bash
mkdir -p archive/{tests,legacy_scripts,old_docs,unused_docker}
```

### Move Unused Files
```bash
# Test files
mv conversational_agent.py live_test.py quick_test.py test_*.py archive/tests/
mv quick_start*.py archive/tests/

# Legacy scripts
mv setup*.sh start.sh talk.sh build-and-run.sh deploy-production.sh archive/legacy_scripts/
mv docker-*.sh start_stt_service.sh start_web.sh archive/legacy_scripts/
mv run_production.py web_backend.py language_aware_router.py archive/legacy_scripts/

# Old docs
mv DOCKER.md QUICKREF.md QUICKSTART.md RUN_ON_DOCKER.md archive/old_docs/
mv SETUP_WITHOUT_CONDA.md IMPLEMENTATION_COMPLETE.md guide.md COMMANDS.txt archive/old_docs/

# Unused Docker
mv Dockerfile.router Dockerfile.voxtral docker-compose.prod.yml archive/unused_docker/

# Test audio
mv testKonkani.mp3 archive/tests/
mv quick_download_model.py archive/tests/
```

---

## 📊 **SUMMARY**

**Total Files**: 58  
**Active Files**: 20  
**Unused Files**: 38  
**Cleanup Savings**: ~65% reduction

**Result**: Clean, focused project with only production-ready files.
