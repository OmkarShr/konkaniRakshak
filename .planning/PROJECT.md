# Konkani Conversational AI Agent

## What This Is

A real-time Konkani conversational voice agent built with Pipecat framework for FIR filing assistance. Designed for desktop/kiosk deployment on self-hosted NVIDIA GPUs, this agent provides natural voice conversations in Konkani (Devanagari script) with streaming output and interruption handling.

Unlike the existing Gradio prototype (offline STT→LLM→TTS pipeline), this will be a true real-time conversational agent where users can speak naturally and interrupt mid-response.

## Why This Exists

Enable Goa Police to provide automated, conversational FIR filing assistance in the native Konkani language. This makes police services more accessible to local citizens who prefer speaking in their native language, reducing barriers to reporting incidents and accessing government services.

## Core Value

**Natural, low-latency voice conversation in Konkani with streaming output and interruption handling** — users should feel like they're talking to a real person, not a robot. The conversation must flow naturally with <1s time-to-first-audio and support barge-in when users want to interrupt.

If the agent feels robotic, slow, or doesn't handle interruptions gracefully, users will reject it. The experience must be conversational, not transactional.

## Success Looks Like

### User Experience
- User speaks in Konkani → agent responds in Konkani with <1s latency
- Streaming output: agent starts speaking immediately, doesn't wait for full response generation
- Barge-in support: user can interrupt mid-response, agent stops and listens
- Conversation feels natural and fluid, not stilted or slow
- Audio quality is clear and understandable

### Technical Validation
- Runs smoothly on RTX 4050 (8GB VRAM) during development
- Scales to production on 2x RTX 5000 (20GB VRAM each)
- Consistent <1s time-to-first-audio (acceptable up to 1.5s)
- Streaming TTS output (not batch)
- VAD (Voice Activity Detection) working reliably for turn-taking

## Current State

### Existing Prototype
Located at: `~/Music/NagarRakshak-Police-Portal-main/konkani/`

**What works:**
- **STT**: IndicConformer (AI4Bharat NeMo) - 499MB model, validated for Konkani
  - Model: `indicconformer_stt_kok_hybrid_rnnt_large.nemo`
  - Devanagari script output
  - Language code: "kok"
- **LLM**: Google Gemini API (gemini-3-flash-preview)
  - Cloud-hosted, validated for Konkani conversations
  - Maintains conversation history
  - Uses Konkani-specific system prompts
- **TTS**: Parler-TTS mini (English model)
  - Works but quality not ideal for Konkani
  - Falls back when Indic model unavailable
- **UI**: Gradio web interface
  - Voice tab: record → process → playback
  - Text tab: typing interface with chat history

**Limitations:**
- ❌ Offline pipeline: record full audio → STT → LLM → TTS → play result
- ❌ No streaming output (waits for full response)
- ❌ No barge-in/interruption handling
- ❌ Not real-time conversational
- ❌ TTS quality suboptimal for Konkani

### What We're Building
Migrate from Gradio prototype to Pipecat-based real-time agent with:
- ✅ Streaming audio input/output
- ✅ Real-time STT processing
- ✅ Streaming LLM responses
- ✅ Streaming TTS with immediate playback
- ✅ Barge-in/interruption handling
- ✅ Improved TTS (Marathi model for better Konkani support)

## Constraints

### Hard Requirements
1. **Language**: Konkani only (Devanagari script)
   - No English fallback in v1
   - No code-switching
   - No multi-language support

2. **Deployment**: Desktop/kiosk only
   - Not mobile apps
   - Not web browser (WebRTC)
   - Not phone system (telephony)
   - Physical hardware with microphone/speakers

3. **Hardware**:
   - Development: RTX 4050 (8GB VRAM) - current laptop
   - Production: 2x RTX 5000 (20GB VRAM each)
   - Must work within these VRAM constraints

4. **Performance**:
   - Target: <1s time-to-first-audio
   - Acceptable: up to 1.5s for typical 5-10 word replies
   - Streaming output required (start speaking ASAP)
   - Barge-in required (interrupt and listen)

### Technology Decisions
- **Framework**: Pipecat (purpose-built for real-time voice agents)
- **STT Model**: IndicConformer (already validated, must reuse)
- **LLM**: Gemini API (cloud, already validated)
- **TTS Strategy**: Marathi model (linguistically similar to Konkani)
- **Compute**: NVIDIA GPUs with CUDA

### Deferred to v2+
- FIR-specific workflow logic and prompts
- Backend database/storage integration
- Document generation (PDF/Word FIR forms)
- API integration with police systems
- Multi-language support
- Web or mobile deployment

## Requirements

### Validated

Capabilities from existing prototype:
- ✓ **STT-01**: Transcribe Konkani speech to Devanagari text — existing (IndicConformer)
- ✓ **LLM-01**: Generate Konkani conversational responses — existing (Gemini API)
- ✓ **TTS-01**: Synthesize Konkani speech from text — existing (Parler-TTS, quality needs improvement)
- ✓ **CTX-01**: Maintain conversation history within session — existing (Gemini chat history)

### Active

New capabilities for real-time voice agent:
- [ ] **PIPE-01**: Integrate Pipecat framework for real-time audio pipeline
- [ ] **STREAM-01**: Stream audio output as it's generated (don't wait for full response)
- [ ] **BARGE-01**: Detect user speech during agent output and interrupt gracefully
- [ ] **VAD-01**: Voice Activity Detection for turn-taking (know when user finishes speaking)
- [ ] **LATENCY-01**: Achieve <1s time-to-first-audio (acceptable up to 1.5s)
- [ ] **TTS-02**: Improve TTS quality using Marathi model for better Konkani support
- [ ] **AUDIO-01**: Desktop audio interface (microphone input, speaker output)
- [ ] **GPU-01**: Optimize STT/TTS to run within RTX 4050 8GB VRAM constraints
- [ ] **DEPLOY-01**: Production deployment configuration for 2x RTX 5000 GPUs
- [ ] **CTX-02**: Single-transaction context (each conversation is independent)

### Out of Scope

Explicitly not building in v1:
- **FIR workflow** — Defer domain-specific logic to v2 (focus on voice pipeline v1)
- **Backend storage** — No database, file storage, or persistence in v1
- **Document generation** — No PDF/Word FIR document creation in v1
- **System integration** — No API integration with police systems in v1
- **Multi-language** — Only Konkani in v1 (no English, Hindi, Marathi switching)
- **Web/mobile** — Desktop/kiosk only in v1
- **User authentication** — Anonymous usage in v1
- **Analytics/logging** — Basic console logging only in v1

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pipecat framework | Purpose-built for real-time voice agents with streaming and barge-in support. Handles audio pipeline complexity. | TBD |
| Keep Gemini API (cloud) | Already validated for Konkani generation. Cloud is acceptable for LLM while keeping STT/TTS local. | TBD |
| Marathi TTS model | Marathi is linguistically similar to Konkani. Better quality than English model for Konkani text. | TBD |
| Desktop/kiosk deployment | Matches police station use case. Simpler audio handling than web/mobile. | TBD |
| Defer FIR workflow | Get voice pipeline solid first. Domain logic can be added once real-time conversation works. | TBD |
| IndicConformer STT | Already validated, 499MB model works well for Konkani. No need to change. | TBD |
| Single transaction context | Each FIR filing is independent. No user accounts or cross-session history needed. | TBD |

## Stack

### Current Prototype Stack
```python
# STT
IndicConformerSTT → nemo.collections.asr.EncDecRNNTBPEModel
Model: indicconformer_stt_kok_hybrid_rnnt_large.nemo (499MB)
Language: "kok" (Konkani)

# LLM
GeminiModule → google.genai.Client
Model: gemini-3-flash-preview
API Key: from .env (Gemini_Api_Key)

# TTS
TTSModule → ParlerTTSForConditionalGeneration
Model: parler-tts/parler-tts-mini-v1 (English fallback)
Target: ai4bharat/indic-parler-tts (needs access)

# UI
Gradio → web interface with voice/text tabs
```

### Target Stack (Pipecat)
```python
# Framework
Pipecat → real-time voice agent orchestration

# Components
STT: IndicConformer (NeMo) - keep existing model
LLM: Gemini API (cloud) - keep existing
TTS: Marathi model TBD (IndicTTS/Coqui/VITS)
VAD: Silero VAD or WebRTC VAD
Transport: Local audio (desktop/kiosk)

# Deployment
CUDA + PyTorch
RTX 4050 (dev) / 2x RTX 5000 (prod)
```

## Team & Stakeholders

**Developer**: Omkar (solo developer)

**End Users**: Goa citizens filing FIRs at police stations/kiosks

**Stakeholders**: Goa Police (NagarRakshak project)

## Timeline & Scope

### v1 Scope
Build real-time voice pipeline with Pipecat:
- Streaming conversation works smoothly
- Latency meets targets (<1s to first audio)
- Barge-in handles interruptions
- TTS quality improved via Marathi model
- Runs on dev hardware (RTX 4050)

**Success metric**: Can have a natural back-and-forth conversation in Konkani with acceptable latency and quality.

### v2 Scope (Future)
Add FIR-specific functionality:
- Structured data collection (complainant details, incident info, etc.)
- Backend storage (database)
- Document generation (FIR PDF)
- API integration with police systems
- Production hardening

### Out of Scope (Not Planned)
- Multi-language support
- Web/mobile deployment
- Cloud deployment
- Advanced analytics
- User authentication

## Reference Materials

### Existing Code
- Prototype: `~/Music/NagarRakshak-Police-Portal-main/konkani/`
- STT module: `stt_indicconformer.py`
- LLM module: `gemini_module.py`
- TTS module: `tts_module.py`
- Main app: `app.py`
- Model: `models/indicconformer_stt_kok_hybrid_rnnt_large.nemo`

### Documentation
- Pipecat: https://github.com/pipecat-ai/pipecat
- IndicConformer: AI4Bharat NeMo models
- Gemini API: google.genai SDK

---
*Last updated: 2026-01-25 after initialization*
