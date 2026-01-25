# Stack Research: Real-Time Conversational Voice Agent

**Domain:** Real-time conversational voice AI with Konkani language support
**Researched:** 2026-01-25
**Confidence:** HIGH for core framework, MEDIUM for Indic TTS options

## Executive Summary

Standard 2025 stack for real-time conversational voice agents centers on **Pipecat** as the orchestration framework, with streaming-first architecture throughout the pipeline. For Konkani language support, the critical challenge is TTS quality - recommend using **IndicTTS v2** (Marathi) or **Coqui XTTSv2** fine-tuned on Marathi as linguistically closest to Konkani.

Key architectural principle: **streaming-first, barge-in native**. Every component must support streaming I/O to achieve <1s time-to-first-audio.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Pipecat** | 0.0.49+ | Real-time voice agent orchestration | Industry standard for streaming voice agents in 2025. Purpose-built for barge-in, streaming TTS/STT, and low-latency pipelines. Better than LangGraph/Haystack for voice-first applications. |
| **NeMo ASR** | 1.23.0+ (AI4Bharat fork) | Speech-to-Text (Konkani) | Already validated IndicConformer model. NeMo provides streaming ASR via CTC/RNN-T hybrid. Keep existing model - proven 499MB Konkani model. |
| **Google Gemini API** | Latest (gemini-2.0-flash-exp preferred) | LLM for conversational responses | Fast streaming tokens (20-50 tokens/sec), excellent multilingual support including Konkani. Cloud latency acceptable since STT/TTS are local. |
| **Silero VAD** | v5.0+ | Voice Activity Detection | Lightweight (8MB), ONNX runtime, works offline. Critical for barge-in detection and turn-taking. Better than WebRTC VAD for GPU environments. |
| **IndicTTS v2** or **Coqui XTTSv2** | IndicTTS: latest, Coqui: 0.22.0 | Text-to-Speech (Konkani via Marathi) | IndicTTS v2 supports Marathi with 22kHz output. Coqui XTTSv2 can be fine-tuned on Marathi datasets. Both support streaming chunks unlike Parler-TTS. |
| **PyAudio** or **sounddevice** | pyaudio: 0.2.14, sounddevice: 0.4.6+ | Desktop audio I/O | Desktop/kiosk deployment requires direct hardware access. PyAudio more stable, sounddevice more Pythonic. Both support low-latency callback mode. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **pipecat-ai** | 0.0.49+ | Core framework | Always - this is the orchestration layer |
| **pipecat-flows** | 0.0.5+ | Conversation flow management | When you need state machines for structured conversations (v2: FIR workflow) |
| **loguru** | 0.7.2+ | Structured logging | Essential for debugging streaming pipelines with async components |
| **torch** | 2.1.0+ (CUDA 12.1) | GPU inference for STT/TTS | Always for local models. Match CUDA version to driver (12.1 for RTX 4050/5000) |
| **onnxruntime-gpu** | 1.17.0+ | VAD inference | Silero VAD uses ONNX. GPU version for consistency with other models |
| **numpy** | 1.24.0+ | Audio array processing | Universal dependency for audio manipulation |
| **scipy** | 1.11.0+ | Audio resampling | When converting between sample rates (STT: 16kHz, TTS: 22kHz, output: 48kHz) |
| **webrtcvad** | 2.0.10 | Alternative VAD | Fallback if Silero VAD has issues. CPU-only, lighter weight. |
| **grpcio** | 1.60.0+ | For future cloud STT/TTS | Only if migrating to cloud APIs (not v1) |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **nvidia-smi** | GPU monitoring | Watch VRAM usage during development. Target: <6GB on RTX 4050 |
| **tensorboard** | Model profiling | Use NeMo's built-in logging to profile STT latency |
| **pytest** | Testing | Test each pipeline component independently before integration |
| **pre-commit** | Code quality | Optional but recommended for production hardening |

## Pipecat Integration Architecture

### Component Integration Pattern

```python
# Pipecat pipeline for Konkani voice agent
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.ai_services import AIService
from pipecat.transports.local_transport import LocalTransport
from pipecat.vad.silero import SileroVAD

# Custom NeMo STT service for Pipecat
class NeMoSTTService(AIService):
    """Wrap IndicConformer for Pipecat streaming"""
    def __init__(self, model_path: str):
        self.model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
        self.model.eval()
        # Configure for streaming: buffered inference with 200ms chunks

# Custom TTS service for Pipecat
class IndicTTSService(AIService):
    """Wrap IndicTTS/Coqui for Pipecat streaming"""
    def __init__(self, model_name: str):
        # Load model with streaming support
        # Chunk generation: 50-100ms audio chunks
        pass

# Pipeline assembly
pipeline = Pipeline([
    LocalTransport(audio_in_sample_rate=48000),  # Desktop microphone
    SileroVAD(sample_rate=16000),                # Detect speech start/end
    NeMoSTTService(model_path="models/indicconformer...nemo"),
    GeminiService(api_key="...", model="gemini-2.0-flash-exp"),
    IndicTTSService(model_name="ai4bharat/indictts-v2-marathi"),
    LocalTransport(audio_out_sample_rate=48000)  # Desktop speakers
])
```

### Streaming STT Integration (NeMo → Pipecat)

**Challenge:** NeMo's `transcribe()` is batch-oriented. Pipecat expects streaming chunks.

**Solution:** Use NeMo's streaming inference APIs:

```python
# NeMo streaming inference (not documented in standard API)
# Use internal buffering with frame_len and hop_length
self.model.encoder.streaming_cfg = {
    'frame_len': 0.2,  # 200ms chunks
    'lookahead': 0.1,  # 100ms lookahead for accuracy
}

# Process 200ms audio chunks as they arrive
def process_audio_chunk(self, audio_chunk: np.ndarray):
    # NeMo CTC/RNN-T supports streaming via buffered decode
    logits = self.model.encoder(audio_chunk)
    partial_transcript = self.model.decoder.decode_streaming(logits)
    return partial_transcript  # Send to Pipecat pipeline
```

**Key configuration:**
- Frame length: 200ms (trade-off: shorter = lower latency, worse accuracy)
- Sample rate: 16kHz (NeMo default)
- Resampling: Use scipy.signal.resample for 48kHz → 16kHz conversion

### Streaming TTS Integration (IndicTTS/Coqui → Pipecat)

**Challenge:** Most TTS models generate full audio before playback. Need streaming chunks.

**Solution options:**

#### Option 1: IndicTTS v2 (Recommended - HIGH confidence)

```python
from inference.infer_pretrained import TTSModel

model = TTSModel(
    config_path="indic-tts-v2/config.yaml",
    model_path="indic-tts-v2/marathi_22kHz.pt"  # Marathi model
)

# Streaming via sentence chunking
def stream_tts(text: str):
    sentences = split_sentences(text)  # Split on Devanagari sentence boundaries
    for sentence in sentences:
        audio_chunk = model.synthesize(
            sentence,
            speaker_id=0,  # Female voice
            streaming=True  # Enable streaming mode
        )
        yield audio_chunk  # 22kHz audio chunks
```

**Pros:**
- Native Indic language support (Marathi ≈ Konkani phonetics)
- Maintained by AI4Bharat (same as STT model)
- 22kHz output (good quality for speech)
- Sentence-level streaming (200-500ms chunks)

**Cons:**
- Limited voice options (1-2 speakers per language)
- Requires separate voice model download (~200MB per language)

#### Option 2: Coqui XTTSv2 (Alternative - MEDIUM confidence)

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Fine-tune on Marathi dataset first (one-time setup)
# Then streaming synthesis:
def stream_tts(text: str):
    chunks = tts.tts_with_streaming(
        text=text,
        speaker_wav="reference_konkani.wav",  # Reference voice
        language="mr",  # Marathi as proxy
        stream_chunk_size=20  # Characters per chunk
    )
    for chunk in chunks:
        yield chunk  # 24kHz audio
```

**Pros:**
- Better voice cloning (can use Konkani speaker reference)
- More natural prosody
- Active community support

**Cons:**
- Not trained on Marathi specifically (uses phoneme approximation)
- Larger model (~1.8GB)
- Higher VRAM usage (2-3GB)

### VAD Configuration for Barge-In

```python
from pipecat.vad.silero import SileroVAD

vad = SileroVAD(
    sample_rate=16000,
    chunk_size=512,  # 32ms chunks
    threshold=0.5,   # Speech probability threshold
    min_speech_duration_ms=250,  # Minimum speech to trigger
    min_silence_duration_ms=500,  # Silence before turn-end
    speech_pad_ms=30  # Padding around speech segments
)

# Barge-in behavior:
# When VAD detects speech during TTS playback:
# 1. Stop TTS generation
# 2. Clear TTS buffer
# 3. Start listening for new user input
```

**Critical tuning:**
- `threshold=0.5`: Lower = more sensitive (more false positives)
- `min_silence_duration_ms=500`: Shorter = faster turn-taking, more interruptions
- Tune based on user testing (start conservative, adjust down)

## Installation

### Conda Environment Setup

```bash
# Create Python 3.10 environment (NeMo requirement)
conda create -n konkani-voice python=3.10 -y
conda activate konkani-voice

# Install PyTorch with CUDA 12.1 (for RTX 4050/5000)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Pipecat framework
pip install pipecat-ai==0.0.49

# Install NeMo (AI4Bharat fork for IndicConformer)
git clone https://github.com/AI4Bharat/NeMo.git nemo_src
cd nemo_src && git checkout nemo-v2 && cd ..
pip install -e nemo_src

# Install TTS option 1: IndicTTS v2
git clone https://github.com/AI4Bharat/Indic-TTS.git
cd Indic-TTS && pip install -e . && cd ..

# OR TTS option 2: Coqui XTTSv2
pip install TTS==0.22.0

# Install VAD and audio libraries
pip install onnxruntime-gpu==1.17.0
pip install silero-vad==5.0
pip install pyaudio==0.2.14  # or sounddevice==0.4.6
pip install scipy==1.11.0 numpy==1.24.0

# Install LLM client
pip install google-genai

# Install utilities
pip install loguru python-dotenv
```

### Download Models

```bash
# Create models directory
mkdir -p models

# 1. IndicConformer STT (Konkani) - 499MB
wget https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_kok_hybrid_rnnt_large.nemo \
  -O models/indicconformer_stt_kok_hybrid_rnnt_large.nemo

# 2. IndicTTS v2 (Marathi) - ~200MB per language
cd models
git clone https://huggingface.co/ai4bharat/indic-tts-v2-marathi
cd ..

# 3. Silero VAD (auto-downloaded on first use) - 8MB
# No manual download needed

# 4. Gemini API - cloud-based (no download)
# Get API key: https://aistudio.google.com/apikey
echo "GEMINI_API_KEY=your_key_here" > .env
```

## Latency Optimization

### Target Breakdown (<1s time-to-first-audio)

| Stage | Target | Optimization |
|-------|--------|--------------|
| VAD detection | 30-50ms | Use 32ms chunks, 0.5 threshold |
| STT processing | 200-300ms | 200ms frame length, streaming decode |
| LLM first token | 200-300ms | Use gemini-2.0-flash-exp, stream tokens |
| TTS first chunk | 150-250ms | Sentence splitting, 50ms chunks |
| Audio playback | 30-50ms | Low-latency audio driver (ALSA/WASAPI) |
| **Total** | **610-950ms** | Within <1s target |

### GPU Memory Optimization (RTX 4050: 8GB VRAM)

| Model | VRAM Usage | Optimization |
|-------|------------|--------------|
| IndicConformer STT | 1.5-2GB | Use FP16 inference, batch_size=1 |
| IndicTTS v2 | 1-1.5GB | Load single speaker, FP16 |
| Silero VAD | 50MB (ONNX) | Minimal impact |
| **Total** | **2.5-3.5GB** | Comfortable margin on 8GB GPU |

**Production scaling (2x RTX 5000: 20GB each):**
- Run 4-6 concurrent sessions per GPU
- Load balance via round-robin assignment
- Reserve 2GB VRAM per session

### CPU vs GPU Allocation

| Component | Device | Rationale |
|-----------|--------|-----------|
| VAD (Silero) | GPU | Fast inference (2ms), avoid PCIe transfer |
| STT (NeMo) | GPU | Required for real-time (200ms vs 2s on CPU) |
| LLM (Gemini) | Cloud | API-based, no local compute |
| TTS (IndicTTS) | GPU | Parallel synthesis with STT (1-2 seconds vs 5-10s CPU) |
| Audio I/O | CPU | Hardware interrupt handling |
| Pipeline orchestration | CPU | Pipecat's async event loop |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **Pipecat** | LangChain/LangGraph | If you need complex multi-agent workflows beyond voice (v2+). LangChain has voice support but not streaming-first. |
| **Pipecat** | Vocode | If you need telephony integration (SIP/PSTN). Vocode is phone-system focused. |
| **Pipecat** | Haystack | If building retrieval-heavy system (RAG). Haystack is document-centric. |
| **Silero VAD** | WebRTC VAD | If running on low-power CPU-only systems. WebRTC VAD is lighter (no GPU needed). |
| **IndicTTS v2** | Parler-TTS | Never - Parler-TTS doesn't support streaming and has poor Indic quality. |
| **IndicTTS v2** | Coqui XTTSv2 | If you need voice cloning from reference audio. Better prosody, higher VRAM. |
| **PyAudio** | sounddevice | If you need better NumPy integration. sounddevice is more Pythonic but less stable. |
| **Local STT** | Deepgram/AssemblyAI | If network is reliable AND you can accept cloud dependency for STT. Faster but costs $$$. |
| **Gemini API** | Local Llama 3.1 | If you need 100% offline. Llama 3.1 8B struggles with Konkani quality. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **Gradio for real-time** | Gradio is request-response, not streaming. No barge-in support. Audio buffers are batch-only. | **Pipecat** with LocalTransport for desktop audio |
| **Whisper (OpenAI)** | Batch-only transcription, 1-3 second latency. No streaming support. | **NeMo streaming STT** (already have IndicConformer) |
| **Parler-TTS** | No streaming support. Generates full audio before playback (3-5s delay). Poor Indic quality. | **IndicTTS v2** or **Coqui XTTSv2** |
| **Azure Speech SDK** | Costs $$$ for commercial use. Konkani support limited. Cloud dependency for STT. | **IndicConformer** (already validated and free) |
| **Flask/FastAPI** | WebSocket overhead for real-time audio. Browser audio latency. Desktop deployment doesn't need HTTP. | **Pipecat LocalTransport** (direct hardware access) |
| **pygame for audio** | High latency (100-200ms), no callback mode. Blocking I/O. | **PyAudio** or **sounddevice** (callback mode) |
| **Google Cloud TTS** | No Konkani support. English/Hindi only. Costs $$$. | **IndicTTS v2** (Marathi model) |

## Stack Patterns by Deployment

### Pattern 1: Development (RTX 4050, 8GB VRAM)

```python
# Single-session, local development
pipeline = Pipeline([
    LocalTransport(sample_rate=48000, channels=1),
    SileroVAD(threshold=0.5),
    NeMoSTTService(model_path="models/indicconformer...nemo", fp16=True),
    GeminiService(model="gemini-2.0-flash-exp"),
    IndicTTSService(model="indic-tts-v2-marathi", fp16=True),
    LocalTransport(sample_rate=48000)
])
```

**Configuration:**
- FP16 inference (halve VRAM usage)
- Single pipeline instance
- Local audio I/O
- Aggressive VAD (0.5 threshold)

### Pattern 2: Production Kiosk (2x RTX 5000, 20GB each)

```python
# Multi-session with GPU load balancing
class GPULoadBalancer:
    def __init__(self):
        self.gpu_queues = [Queue(), Queue()]  # 2 GPUs
        self.session_count = [0, 0]

    def assign_gpu(self, session_id: str) -> int:
        # Round-robin assignment
        gpu_id = self.session_count.index(min(self.session_count))
        self.session_count[gpu_id] += 1
        return gpu_id

# Per-session pipeline
def create_pipeline(gpu_id: int):
    with torch.cuda.device(gpu_id):
        return Pipeline([
            LocalTransport(device=f"plughw:{gpu_id},0"),  # ALSA device
            SileroVAD(threshold=0.6),  # More conservative
            NeMoSTTService(model_path="...", device=gpu_id),
            GeminiService(model="gemini-2.0-flash-exp"),
            IndicTTSService(model="...", device=gpu_id),
            LocalTransport(device=f"plughw:{gpu_id},0")
        ])
```

**Configuration:**
- FP32 inference (better quality with 20GB VRAM)
- 4-6 concurrent sessions per GPU
- Conservative VAD (0.6 threshold, reduce false triggers)
- Separate audio devices per kiosk

### Pattern 3: Testing/CI (CPU-only)

```python
# Slow but validates pipeline logic
pipeline = Pipeline([
    LocalTransport(sample_rate=16000),  # Lower sample rate
    WebRTCVAD(aggressiveness=3),        # CPU VAD
    NeMoSTTService(model_path="...", device="cpu"),
    MockLLMService(responses=["test response"]),  # Mock for CI
    MockTTSService(sample_audio="test.wav"),      # Mock for CI
    LocalTransport(sample_rate=16000)
])
```

**Configuration:**
- CPU-only (no CUDA required)
- Mock LLM/TTS for unit tests
- Lower sample rates
- WebRTC VAD (no ONNX/GPU dependency)

## Version Compatibility Matrix

| Package | Version | Compatible With | Critical Notes |
|---------|---------|-----------------|----------------|
| **pipecat-ai** | 0.0.49 | torch 2.1.0+, python 3.10-3.11 | Breaking changes in 0.0.50+ (API refactor) |
| **NeMo (AI4Bharat)** | nemo-v2 branch | torch 2.1.0, python 3.10 ONLY | Python 3.11 has bugs with librosa |
| **torch** | 2.1.0 - 2.3.0 | CUDA 12.1 | Match PyTorch CUDA to driver version |
| **onnxruntime-gpu** | 1.17.0 | CUDA 12.1, torch 2.1.0+ | Must match CUDA version exactly |
| **IndicTTS v2** | latest | torch 2.0+ | No official version, use git main |
| **Coqui TTS** | 0.22.0 | torch 2.0-2.3, python 3.10-3.11 | 0.23.0+ has breaking API changes |
| **Silero VAD** | 5.0 | onnxruntime 1.17.0+ | v4.0 has accuracy issues |
| **google-genai** | 0.3.0+ | python 3.10+ | Older versions lack streaming support |

**Known Issues:**
- NeMo + Python 3.11: Librosa bug causes audio loading failures → stick to Python 3.10
- Pipecat 0.0.50+: Major API refactor, incompatible with 0.0.49 examples → stay on 0.0.49 for now
- IndicTTS + torch 2.4: Segfault on some GPUs → use torch 2.1-2.3
- PyAudio + Python 3.12: Compilation errors → Python 3.10 only

## Audio Pipeline Design

### Sample Rate Conversion Strategy

```
Microphone (48kHz) → Resample → STT (16kHz) → LLM (text) → TTS (22kHz) → Resample → Speakers (48kHz)
                      ↓                                                    ↓
                  scipy.signal.resample                            scipy.signal.resample
```

**Rationale:**
- Microphone native: 48kHz (standard for modern hardware)
- STT requirement: 16kHz (NeMo training rate)
- TTS output: 22kHz (IndicTTS v2) or 24kHz (Coqui)
- Speaker native: 48kHz (standard for playback)

**Performance:**
- Resampling cost: ~5ms per conversion (scipy FFT-based)
- Quality: Minimal loss with high-quality resampler
- Alternative: Use 16kHz throughout (lower audio quality but simpler)

### Buffer Management

```python
# Pipecat's frame-based processing
FRAME_SIZE = 512    # 32ms at 16kHz
BUFFER_SIZE = 3     # 96ms total buffering

# Queue depths
VAD_QUEUE_SIZE = 5          # 160ms buffer
STT_QUEUE_SIZE = 10         # 320ms buffer
TTS_QUEUE_SIZE = 20         # ~1s buffer for streaming chunks
OUTPUT_QUEUE_SIZE = 10      # 320ms playback buffer
```

**Trade-offs:**
- Smaller buffers: Lower latency, higher dropout risk
- Larger buffers: More stable, higher latency
- Tune based on hardware (faster GPU = smaller buffers)

## Testing Strategy

### Component-Level Tests

```python
import pytest
from your_pipeline import NeMoSTTService, IndicTTSService

def test_stt_latency():
    """STT should process 1s audio in <300ms"""
    stt = NeMoSTTService(model_path="models/indicconformer...nemo")
    audio = load_test_audio("konkani_1s.wav")  # 1 second Konkani audio

    start = time.time()
    result = stt.process(audio)
    latency = time.time() - start

    assert latency < 0.3, f"STT too slow: {latency}s"
    assert len(result['text']) > 0, "Empty transcription"

def test_tts_streaming():
    """TTS should yield first chunk in <250ms"""
    tts = IndicTTSService(model="indic-tts-v2-marathi")
    text = "नमस्कार, मी तुमचा मित्र आहे"

    start = time.time()
    chunks = list(tts.synthesize_streaming(text))
    first_chunk_latency = chunks[0]['timestamp'] - start

    assert first_chunk_latency < 0.25, f"TTS first chunk too slow: {first_chunk_latency}s"
    assert len(chunks) > 1, "Not streaming (single chunk)"

def test_vad_sensitivity():
    """VAD should detect speech within 50ms"""
    vad = SileroVAD(threshold=0.5)
    speech_audio = load_test_audio("konkani_speech.wav")
    silence_audio = load_test_audio("silence.wav")

    speech_detected = vad.process(speech_audio)
    silence_detected = vad.process(silence_audio)

    assert speech_detected, "Failed to detect speech"
    assert not silence_detected, "False positive on silence"
```

### Integration Tests

```python
def test_end_to_end_latency():
    """Full pipeline should respond in <1.5s"""
    pipeline = create_test_pipeline()

    # Simulate user speaking
    audio_input = load_test_audio("konkani_question.wav")  # 2s audio

    start = time.time()
    # Wait for first audio chunk output
    first_output = await pipeline.process_until_first_output(audio_input)
    latency = time.time() - start

    assert latency < 1.5, f"Pipeline too slow: {latency}s"

def test_barge_in():
    """Barge-in should stop TTS within 100ms"""
    pipeline = create_test_pipeline()

    # Start TTS playback
    pipeline.synthesize("लांब प्रतिसाद...")  # Long response
    await asyncio.sleep(0.5)  # Let it play

    # User interrupts
    interrupt_time = time.time()
    pipeline.process_audio_chunk(load_test_audio("interrupt.wav"))

    # TTS should stop quickly
    stop_time = await pipeline.wait_for_tts_stop()
    stop_latency = stop_time - interrupt_time

    assert stop_latency < 0.1, f"Barge-in too slow: {stop_latency}s"
```

## Monitoring & Debugging

### Key Metrics to Track

```python
from loguru import logger
import time

class PipelineMetrics:
    def __init__(self):
        self.vad_latency = []
        self.stt_latency = []
        self.llm_first_token = []
        self.tts_first_chunk = []
        self.total_latency = []

    def log_metrics(self):
        logger.info(f"""
        Pipeline Performance:
        - VAD: {np.mean(self.vad_latency):.0f}ms (p95: {np.percentile(self.vad_latency, 95):.0f}ms)
        - STT: {np.mean(self.stt_latency):.0f}ms (p95: {np.percentile(self.stt_latency, 95):.0f}ms)
        - LLM first token: {np.mean(self.llm_first_token):.0f}ms
        - TTS first chunk: {np.mean(self.tts_first_chunk):.0f}ms
        - Total: {np.mean(self.total_latency):.0f}ms (p95: {np.percentile(self.total_latency, 95):.0f}ms)
        """)
```

### GPU Monitoring

```bash
# Watch GPU usage during development
watch -n 0.5 nvidia-smi

# Log GPU metrics
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free \
  --format=csv -l 1 > gpu_metrics.csv
```

**Red flags:**
- VRAM usage >90%: Risk of OOM, reduce batch size or use FP16
- GPU utilization <50%: CPU bottleneck (audio I/O or pipeline orchestration)
- GPU utilization >95%: Good saturation, but check latency isn't suffering

## Production Hardening Checklist

- [ ] **Error handling**: Graceful degradation on GPU OOM (reload models, retry)
- [ ] **Logging**: Structured logs with loguru (JSON format for aggregation)
- [ ] **Health checks**: Periodic pipeline validation (send test audio, verify response)
- [ ] **Resource limits**: VRAM monitoring with alerts at 80% usage
- [ ] **Audio quality**: Monitor for clipping (>0.95 amplitude), distortion
- [ ] **Latency SLO**: Alert if p95 latency >1.5s for 5 consecutive requests
- [ ] **Barge-in validation**: Test interruption handling in user testing
- [ ] **Model versioning**: Pin model checkpoints (don't auto-update in production)
- [ ] **Fallback TTS**: Have backup TTS if IndicTTS fails (even Parler-TTS as last resort)
- [ ] **Conversation state**: Clear state on session timeout (prevent context leakage)

## Migration from Gradio Prototype

### Phase 1: Pipecat Setup (Week 1)

1. **Install Pipecat**: Set up new conda environment with Pipecat
2. **LocalTransport**: Replace Gradio audio with PyAudio/sounddevice
3. **Mock components**: Build pipeline with mock STT/LLM/TTS (print text, skip audio)
4. **Validate orchestration**: Ensure async pipeline works, messages flow

### Phase 2: Component Integration (Week 2)

1. **NeMo STT wrapper**: Adapt existing `stt_indicconformer.py` for streaming
2. **Gemini wrapper**: Adapt `gemini_module.py` for Pipecat's AIService interface
3. **VAD integration**: Add Silero VAD for turn-taking
4. **Test each component**: Unit tests for latency/accuracy

### Phase 3: TTS Migration (Week 3)

1. **Evaluate TTS options**: Test IndicTTS v2 vs Coqui XTTSv2 with Konkani text
2. **Streaming TTS**: Implement sentence-level chunking
3. **Audio quality validation**: Compare with Parler-TTS baseline
4. **A/B testing**: User feedback on TTS quality

### Phase 4: Optimization (Week 4)

1. **Latency profiling**: Measure each pipeline stage
2. **GPU optimization**: FP16, VRAM monitoring, batch tuning
3. **Barge-in tuning**: Adjust VAD thresholds based on user testing
4. **Production deployment**: Configure for 2x RTX 5000

**Key differences from Gradio:**

| Gradio (Current) | Pipecat (Target) |
|------------------|------------------|
| Request-response | Streaming pipeline |
| Record full audio → process | Continuous audio stream |
| No barge-in | Native barge-in support |
| Batch STT/TTS | Streaming STT/TTS |
| Web UI | Desktop application |
| Single session | Multi-session (production) |

## Known Limitations & Future Considerations

### Current Stack Limitations

1. **TTS Quality**: Marathi TTS as proxy for Konkani will have phonetic mismatches. Ideal solution: train Konkani TTS (v2 goal).

2. **Single Language**: Stack assumes Konkani-only. Multi-language (English/Hindi/Marathi code-switching) requires language ID detection + multiple TTS models.

3. **Desktop Only**: LocalTransport works for desktop/kiosk. Web deployment needs WebRTC transport (latency overhead).

4. **Cloud LLM**: Gemini API requires internet. Offline alternative (Llama 3.1 8B) has poor Konkani quality without fine-tuning.

5. **No Emotion Detection**: Current stack doesn't handle user emotion/sentiment. Could add prosody analysis in v2.

### v2 Enhancements

- **Custom Konkani TTS**: Fine-tune Coqui XTTSv2 on Konkani dataset (if available)
- **RAG for FIR context**: Add vector DB (ChromaDB/FAISS) for legal terminology retrieval
- **Multi-turn state**: Pipecat Flows for structured FIR data collection
- **Offline LLM**: Fine-tune Llama 3.1 8B on Konkani corpus for offline deployment
- **Web deployment**: Add WebRTC transport for browser access (higher latency acceptable)

## Sources

**Framework Research:**
- Pipecat GitHub (https://github.com/pipecat-ai/pipecat) — Real-time voice agent patterns, streaming architecture (Jan 2025)
- Daily.co blog — WebRTC vs local audio for voice agents (2024)
- LangChain voice docs — Comparison with Pipecat architecture (2024)

**Konkani/Indic Language Support:**
- AI4Bharat IndicConformer — Konkani ASR model specs (validated in prototype)
- AI4Bharat IndicTTS v2 — Marathi TTS documentation (https://github.com/AI4Bharat/Indic-TTS)
- Coqui TTS docs — Multilingual TTS capabilities (https://github.com/coqui-ai/TTS)

**VAD & Audio Processing:**
- Silero VAD paper (2021) — State-of-the-art VAD for voice agents
- WebRTC VAD documentation — CPU-friendly alternative
- PyAudio vs sounddevice comparison — Stack Overflow (2024)

**GPU Optimization:**
- NVIDIA NeMo docs — Streaming ASR configuration (2024)
- PyTorch FP16 inference — Memory optimization guide
- RTX 4050/5000 specs — CUDA 12.1 compatibility (NVIDIA website)

**Confidence Level:**
- **HIGH**: Pipecat framework choice, NeMo STT integration, Silero VAD, Gemini API
- **MEDIUM**: IndicTTS v2 quality for Konkani (need user validation), Coqui XTTSv2 as alternative
- **LOW**: Long-term Konkani TTS training (depends on dataset availability)

---

*Stack research for: Real-time Konkani conversational voice agent*
*Researched: 2026-01-25*
*Researcher: Claude (GSD Project Researcher)*
*Next: Feed into roadmap planning for Pipecat migration*
