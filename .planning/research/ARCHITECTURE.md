# Architecture Research: Real-time Voice AI Agents

## Research Question
How are real-time conversational voice agents typically structured? What are the major components and data flow patterns?

**Context**: Building a Konkani voice agent with Pipecat framework using IndicConformer STT, Gemini LLM, and Marathi/Indic TTS for desktop/kiosk deployment.

---

## Executive Summary

Real-time voice agents are structured as **event-driven streaming pipelines** where audio flows continuously through processing stages without waiting for complete utterances. The key architectural shift from traditional batch processing is:

**Batch (Current Prototype)**: `Record → Complete STT → Complete LLM → Complete TTS → Playback`

**Streaming (Target)**: `Audio Stream → Streaming STT → Streaming LLM → Streaming TTS → Audio Stream` (all happening simultaneously)

Pipecat implements this as a **frame-based pipeline** where components process small chunks of data (frames) and pass them downstream immediately, enabling:
- Sub-second latency (time-to-first-audio)
- Interruption handling (barge-in)
- Natural conversation flow
- Parallel processing of input and output

---

## Core Architecture Patterns

### 1. Frame-Based Processing Pipeline

The fundamental unit in Pipecat is a **Frame** - a discrete chunk of data flowing through the pipeline:

```
Frame Types:
├── Audio Frames (raw PCM audio chunks, ~20ms)
├── Text Frames (transcribed text, partial or complete)
├── LLM Response Frames (generated text tokens/words)
├── Control Frames (start/stop/interrupt signals)
└── Error Frames (exception handling)
```

Components are **Processors** that:
1. Receive frames from upstream
2. Transform/process the frame
3. Emit frames downstream
4. Can be chained in a pipeline

```
Pipeline Structure:
Transport → VAD → STT → LLM → TTS → Transport
   ↑                                      ↓
   └─────────── Audio Loop ───────────────┘
```

### 2. Major Components and Boundaries

#### **A. Transport Layer** (Audio I/O Interface)
**What it does**: Manages bidirectional audio streams between hardware and the pipeline.

**In Pipecat**:
- `LocalAudioTransport`: Desktop/kiosk audio via PyAudio/sounddevice
- Handles microphone input → frames
- Handles frames → speaker output
- Manages audio formats (sample rate, channels, bit depth)

**For this project**:
- Use `LocalAudioTransport` for desktop/kiosk
- Input: 16kHz mono PCM (typical for STT)
- Output: varies by TTS model (IndicConformer expects 16kHz)

**Boundaries**:
- **Input**: Transport receives raw audio from mic → emits AudioFrame
- **Output**: Transport receives AudioFrame → plays through speakers

#### **B. Voice Activity Detection (VAD)**
**What it does**: Detects when user is speaking vs. silence to determine turn boundaries.

**In Pipecat**:
- Wraps models like Silero VAD or WebRTC VAD
- Consumes audio frames
- Emits start-of-speech and end-of-speech control frames
- Used for both turn-taking and barge-in detection

**For this project**:
- Silero VAD (recommended, accurate, runs on CPU)
- Configured with appropriate thresholds for noisy police station environments
- Detects when user finishes speaking → triggers STT finalization
- Detects when user interrupts → triggers barge-in

**Boundaries**:
- **Input**: AudioFrame stream
- **Output**: AudioFrame (pass-through) + VAD events (speech_started, speech_ended)

#### **C. Speech-to-Text (STT) Processor**
**What it does**: Converts audio stream to text, incrementally.

**In Pipecat**:
- STT processors can be streaming (emit partial transcripts) or batch (emit only final)
- Most STT integrations buffer audio until VAD signals end-of-speech
- Then emit TextFrame with transcribed text

**For this project**:
- **IndicConformer (NeMo)** is batch-mode STT
- Need to wrap it as a Pipecat processor:
  ```python
  class IndicConformerSTTProcessor(FrameProcessor):
      - Buffer audio frames until VAD signals speech_ended
      - Process buffered audio through IndicConformer
      - Emit TextFrame with transcribed Konkani text
      - Language setting: "kok" for Konkani
  ```
- Not true streaming STT (no partial transcripts), but acceptable
- Latency: ~200-500ms after speech ends for transcription

**Boundaries**:
- **Input**: AudioFrame stream + VAD events
- **Output**: TextFrame with complete transcribed text

#### **D. Language Model (LLM) Processor**
**What it does**: Generates conversational responses, streaming tokens as they're produced.

**In Pipecat**:
- LLM processors support streaming token-by-token output
- Maintains conversation context/history
- Emits TextFrame for each token or small chunk

**For this project**:
- **Gemini API** with streaming enabled
- Gemini SDK supports streaming: `generate_content_stream()`
- Configuration:
  ```python
  GeminiLLMProcessor:
      - Receive TextFrame (user message)
      - Call Gemini API with streaming=True
      - Emit TextFrame for each token/word chunk
      - Maintain conversation history in context
  ```
- System prompt: Konkani conversational assistant
- Latency: ~100-300ms to first token

**Boundaries**:
- **Input**: TextFrame (user message)
- **Output**: Stream of TextFrame chunks (response tokens/words)

#### **E. Text-to-Speech (TTS) Processor**
**What it does**: Converts text to audio, ideally streaming.

**In Pipecat**:
- TTS processors can be streaming (emit audio chunks) or batch (emit full audio)
- Streaming TTS is critical for low latency (start speaking before full response generated)

**For this project**:
- **Challenge**: Most Indic TTS models are batch-mode (need full sentence)
- **Options**:
  1. Marathi TTS streaming if available (VITS, Coqui)
  2. Sentence-level streaming: buffer tokens until sentence boundary, then synthesize
  3. Word-level streaming: synthesize word-by-word (quality trade-off)

**Recommended approach**:
```python
SentenceBufferedTTSProcessor:
    - Buffer TextFrame tokens from LLM
    - Detect sentence boundaries (। or . or ? or !)
    - Synthesize complete sentences
    - Emit AudioFrame chunks
    - Continue while LLM is still generating
```

- Latency: ~300-800ms per sentence (depending on TTS model)
- First sentence starts playing while LLM generates remaining sentences

**Boundaries**:
- **Input**: Stream of TextFrame chunks (LLM response)
- **Output**: Stream of AudioFrame chunks (synthesized audio)

#### **F. Pipeline Orchestrator**
**What it does**: Manages the overall pipeline, coordinates components, handles control flow.

**In Pipecat**:
- `PipelineTask`: Main orchestration object
- Connects processors in sequence
- Manages async event loop
- Handles interruptions and state transitions

**For this project**:
```python
pipeline = Pipeline([
    transport,
    vad_processor,
    stt_processor,
    llm_processor,
    tts_processor,
    transport  # back to output
])

task = PipelineTask(pipeline, params={
    "language": "kok",
    "allow_interruptions": True,
    "audio_out_enabled": True
})
```

**Boundaries**:
- Manages lifecycle of all components
- Routes frames between processors
- Handles errors and recovery

---

## Data Flow Architecture

### End-to-End Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPECAT PIPELINE                          │
│                                                                   │
│  ┌──────────┐      ┌─────┐      ┌─────┐      ┌─────┐      ┌────┐│
│  │ Mic      │─────▶│ VAD │─────▶│ STT │─────▶│ LLM │─────▶│TTS ││
│  │ Input    │      └─────┘      └─────┘      └─────┘      └────┘│
│  └──────────┘         │            │             │           │   │
│       ▲               │            │             │           │   │
│       │          ┌────▼────────────▼─────────────▼───────────▼┐  │
│       │          │     AudioFrame / TextFrame Stream          │  │
│       │          └─────────────────────────────────────────────┘  │
│       │                                                       │   │
│  ┌────┴────┐                                            ┌────▼──┐│
│  │ Speaker │◀───────────────────────────────────────────│ Audio ││
│  │ Output  │             AudioFrame Stream              │ Out   ││
│  └─────────┘                                            └───────┘│
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Barge-in Detector (monitors VAD)               │ │
│  │  Detects user speech during TTS output → Cancel TTS        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### Detailed Frame Flow

**Phase 1: User Speaking**
```
1. Microphone captures audio → 20ms chunks
   AudioFrame(audio_data, sample_rate=16000)

2. VAD processes each chunk
   → AudioFrame (pass-through)
   → speech_started event (first time)

3. STT buffers audio frames
   → Waiting for speech_ended event

4. VAD detects silence
   → speech_ended event

5. STT processes buffered audio
   → TextFrame("नमस्कार, तुमचं नाव काय?")
```

**Phase 2: LLM Processing**
```
6. LLM receives TextFrame
   → Context: Previous conversation history
   → API call: Gemini streaming API

7. LLM emits tokens as generated
   → TextFrame("माझं")
   → TextFrame(" नाव")
   → TextFrame(" AI")
   → TextFrame(" आहे।")
   ... continues until response complete
```

**Phase 3: TTS & Playback**
```
8. TTS buffers tokens until sentence boundary
   → Detects "।" or "."
   → Synthesizes first sentence: "माझं नाव AI आहे।"
   → AudioFrame(chunk_1)
   → AudioFrame(chunk_2)
   ... (while LLM continues generating next sentence)

9. Transport plays audio frames to speaker
   → User hears response starting ~1s after speaking

10. Meanwhile, LLM generates next sentence
    → TTS synthesizes it
    → Playback continues seamlessly
```

**Phase 4: Barge-in (Interruption)**
```
11. User starts speaking mid-response
    → VAD detects speech_started

12. Barge-in controller activated
    → Cancels TTS generation
    → Clears TTS audio buffer
    → Stops speaker output
    → LLM generation cancelled

13. Pipeline transitions to listening mode
    → STT starts buffering new audio
    → Cycle restarts from Phase 1
```

---

## Streaming Architecture Details

### Audio Streaming (Transport ↔ Pipeline)

**Input Stream**:
```python
Microphone
  ↓ (continuous)
Audio frames: 20ms chunks @ 16kHz = 320 samples/frame
  ↓
Queue: Non-blocking async queue
  ↓
VAD + STT processors
```

**Output Stream**:
```python
TTS processor
  ↓
Audio frames: Variable size chunks (typically 20-50ms)
  ↓
Queue: Async queue with playback buffer
  ↓
Speaker (sounddevice/PyAudio)
```

**Key consideration**: Buffer sizes and queue depths affect latency
- Too small → audio glitches
- Too large → increased latency
- Sweet spot: 2-4 frame buffer (~40-80ms)

### Text Streaming (STT → LLM → TTS)

**STT to LLM**:
- IndicConformer emits complete utterances (not streaming)
- Single TextFrame per user turn

**LLM to TTS**:
- Gemini streams tokens/words
- Multiple TextFrame emissions
- TTS needs to handle partial input

**Streaming pattern**:
```python
async for chunk in gemini.generate_stream(user_text):
    text_frame = TextFrame(chunk)
    await pipeline.queue_frame(text_frame)
```

### Sentence Segmentation for TTS

Since most Indic TTS is batch-mode, implement sentence buffering:

```python
class SentenceBufferedTTS:
    buffer = []
    sentence_endings = ["।", ".", "?", "!", "\n"]

    async def process_frame(self, frame: TextFrame):
        self.buffer.append(frame.text)
        combined = "".join(self.buffer)

        # Check for sentence boundary
        for ending in sentence_endings:
            if ending in combined:
                sentences = combined.split(ending)
                # Synthesize complete sentences
                for sentence in sentences[:-1]:
                    audio = await self.tts.synthesize(sentence + ending)
                    await self.push_audio_frames(audio)
                # Keep partial sentence in buffer
                self.buffer = [sentences[-1]]
                break
```

This enables:
- Streaming experience (start speaking quickly)
- Quality TTS (full sentences, not word-by-word)
- Low latency (don't wait for complete response)

---

## Barge-in Mechanism

### How Interruption Detection Works

**1. VAD Monitors Continuously**
```python
vad_processor:
    while pipeline.running:
        audio_frame = await self.get_input()
        speech_detected = vad.detect(audio_frame)

        if speech_detected and tts_is_playing:
            await emit_interrupt_frame()
```

**2. Interrupt Frame Propagates**
```python
InterruptFrame emitted
  ↓
LLM processor: cancel current generation
  ↓
TTS processor: clear buffer, stop synthesis
  ↓
Transport: stop audio playback, flush buffer
  ↓
STT processor: start buffering new input
```

**3. State Transition**
```
State: SPEAKING (TTS playing)
  → Interrupt detected
State: INTERRUPTED
  → Clear all buffers
State: LISTENING (STT active)
```

### Implementation Pattern in Pipecat

```python
class BotInterruptionHandler(FrameProcessor):
    async def process_frame(self, frame: Frame):
        if isinstance(frame, UserStartedSpeakingFrame):
            # User interrupted the bot
            if self.pipeline.current_state == "speaking":
                # Cancel LLM
                await self.cancel_llm_task()
                # Stop TTS
                await self.tts_processor.cancel()
                # Clear audio output buffer
                await self.transport.clear_output()
                # Transition to listening
                self.pipeline.current_state = "listening"

        # Pass frame downstream
        await self.push_frame(frame)
```

**Critical considerations**:
- Interruption should be near-instant (<100ms)
- Audio cutoff should be clean (no pops/clicks)
- LLM API calls may need explicit cancellation
- State must be consistent across all components

---

## Component Integration Patterns

### IndicConformer (NeMo) Integration

**Challenge**: NeMo models are synchronous, Pipecat is async.

**Solution**: Wrap in async executor
```python
class IndicConformerProcessor(FrameProcessor):
    def __init__(self, model_path):
        self.model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
        self.model.eval()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.audio_buffer = []

    async def process_frame(self, frame: Frame):
        if isinstance(frame, AudioFrame):
            self.audio_buffer.append(frame.audio)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # VAD detected end of speech
            audio = np.concatenate(self.audio_buffer)

            # Run transcription in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                self.executor,
                self._transcribe,
                audio
            )

            # Emit transcription
            await self.push_frame(TextFrame(text))
            self.audio_buffer.clear()

    def _transcribe(self, audio):
        # Blocking NeMo transcription
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, 16000)
            result = self.model.transcribe([f.name], language_id="kok")
            return result[0][0]
```

### Gemini API Integration

**Gemini supports streaming** via `generate_content_stream()`:

```python
class GeminiLLMProcessor(FrameProcessor):
    def __init__(self, api_key, model="gemini-3-flash-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.history = []

    async def process_frame(self, frame: Frame):
        if isinstance(frame, TextFrame):
            user_message = frame.text

            # Build context
            messages = self.history + [{"role": "user", "parts": [{"text": user_message}]}]

            # Stream response
            response_stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=messages
            )

            full_response = []
            async for chunk in response_stream:
                if chunk.text:
                    # Emit each chunk immediately
                    await self.push_frame(TextFrame(chunk.text))
                    full_response.append(chunk.text)

            # Update history
            response_text = "".join(full_response)
            self.history.append({"role": "user", "parts": [{"text": user_message}]})
            self.history.append({"role": "model", "parts": [{"text": response_text}]})

            # Signal end of LLM response
            await self.push_frame(LLMFullResponseEndFrame())
```

### TTS Model Integration

**For Marathi/Indic TTS** (assuming batch synthesis):

```python
class SentenceStreamingTTSProcessor(FrameProcessor):
    def __init__(self, tts_model):
        self.tts = tts_model
        self.sentence_buffer = ""
        self.sentence_endings = ["।", ".", "?", "!", "\n"]

    async def process_frame(self, frame: Frame):
        if isinstance(frame, TextFrame):
            self.sentence_buffer += frame.text

            # Check for sentence completion
            for ending in self.sentence_endings:
                if ending in self.sentence_buffer:
                    # Split on ending
                    parts = self.sentence_buffer.split(ending, 1)
                    sentence = parts[0] + ending
                    remaining = parts[1] if len(parts) > 1 else ""

                    # Synthesize complete sentence
                    audio = await self._synthesize(sentence)

                    # Emit audio frames
                    await self._push_audio(audio)

                    # Update buffer
                    self.sentence_buffer = remaining
                    break

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Synthesize any remaining text
            if self.sentence_buffer.strip():
                audio = await self._synthesize(self.sentence_buffer)
                await self._push_audio(audio)
                self.sentence_buffer = ""

            # Pass end frame downstream
            await self.push_frame(frame)

    async def _synthesize(self, text):
        # Run TTS in executor (blocking operation)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.tts.synthesize, text)

    async def _push_audio(self, audio):
        # Chunk audio into frames
        chunk_size = 320  # 20ms at 16kHz
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            await self.push_frame(AudioFrame(chunk, sample_rate=16000))
```

---

## Latency Budget & Optimization

### Target: <1s Time-to-First-Audio

**Breakdown of pipeline latency**:

```
User stops speaking
  ↓ VAD detection: ~50-100ms
Speech end detected
  ↓ STT processing: ~200-500ms (IndicConformer)
Transcription complete
  ↓ LLM first token: ~100-300ms (Gemini API, network + inference)
First token received
  ↓ TTS synthesis: ~300-800ms (sentence buffer + synthesis)
First audio chunk ready
  ↓ Audio playback start: ~20-50ms (buffer)
User hears response

Total: 670-1750ms
Target: <1000ms
Acceptable: <1500ms
```

### Optimization Strategies

**1. Reduce STT Latency**
- Use GPU for IndicConformer inference
- Optimize VAD thresholds (reduce delay in silence detection)
- Pre-load model on GPU (done once at startup)

**2. Reduce LLM Latency**
- Use Gemini Flash (faster than Pro)
- Optimize prompt length (shorter context = faster)
- Consider regional API endpoints (closer server)

**3. Reduce TTS Latency**
- Use smallest viable TTS model (quality vs speed trade-off)
- Optimize sentence detection (don't wait too long for boundaries)
- Pre-generate common phrases (e.g., greetings)
- GPU acceleration for TTS if available

**4. Pipeline Optimization**
- Async processing (components work in parallel)
- Minimize serialization/deserialization overhead
- Efficient audio format conversions
- Proper async/await usage (no blocking calls in main loop)

**5. Hardware Optimization**
- **Development (RTX 4050, 8GB VRAM)**:
  - STT on GPU (~2GB)
  - TTS on GPU (~2GB)
  - Leave headroom for system

- **Production (2x RTX 5000, 20GB each)**:
  - Option A: Dedicate GPU 1 to STT, GPU 2 to TTS/LLM
  - Option B: Run multiple agents in parallel (multi-tenant)

---

## Build Order Recommendations

### Phase 1: Foundation (Week 1-2)
**Goal**: Get basic pipeline working without streaming

1. **Set up Pipecat environment**
   - Install Pipecat framework
   - Set up LocalAudioTransport for desktop audio
   - Test mic input and speaker output

2. **Integrate IndicConformer STT**
   - Wrap NeMo model as Pipecat processor
   - Test with recorded Konkani audio
   - Validate GPU usage and latency

3. **Integrate Gemini LLM**
   - Implement basic LLM processor (non-streaming first)
   - Test with Konkani text input
   - Validate conversation history

4. **Simple TTS integration**
   - Start with existing Parler-TTS (known to work)
   - Wrap as Pipecat processor
   - Test end-to-end pipeline (no streaming yet)

**Milestone**: Can have a complete conversation (batch mode, high latency)

### Phase 2: Streaming & VAD (Week 3-4)
**Goal**: Add streaming and real-time behavior

5. **Add VAD processor**
   - Integrate Silero VAD
   - Test turn-taking detection
   - Tune thresholds for environment

6. **Enable LLM streaming**
   - Implement Gemini streaming API
   - Test token-by-token emission
   - Validate with TTS processor

7. **Implement sentence-buffered TTS**
   - Buffer LLM tokens
   - Detect sentence boundaries
   - Stream audio output

**Milestone**: Streaming conversation with acceptable latency

### Phase 3: Barge-in & Polish (Week 5-6)
**Goal**: Handle interruptions and optimize performance

8. **Implement barge-in**
   - Add interruption detection
   - Test cancellation logic
   - Validate state transitions

9. **Optimize latency**
   - Profile pipeline bottlenecks
   - Optimize model loading and inference
   - Tune buffer sizes and thresholds

10. **Integrate better TTS**
    - Research/test Marathi TTS models
    - Integrate IndicTTS or equivalent
    - A/B test quality vs. Parler-TTS

**Milestone**: Production-ready voice agent

### Phase 4: Production Deployment (Week 7-8)
**Goal**: Deploy to target hardware

11. **Multi-GPU configuration**
    - Distribute models across 2x RTX 5000
    - Load balancing if needed
    - Performance testing

12. **Error handling & recovery**
    - Network failures (Gemini API)
    - Model errors
    - Audio device issues

13. **Production hardening**
    - Logging and monitoring
    - Resource management
    - Graceful degradation

**Milestone**: Deployed and stable

---

## Key Architectural Decisions

### Decision 1: Frame-based vs. Callback-based
**Choice**: Frame-based (Pipecat's model)
**Rationale**:
- Easier to reason about data flow
- Better composability (processors as building blocks)
- Built-in backpressure handling
- Async-first design

### Decision 2: Streaming vs. Batch TTS
**Choice**: Sentence-buffered streaming
**Rationale**:
- True word-level streaming sacrifices quality for Indic languages
- Full-response batching sacrifices latency
- Sentence buffering balances quality and latency
- Compatible with most TTS models

### Decision 3: VAD Placement
**Choice**: After Transport, before STT
**Rationale**:
- All audio goes through VAD (input and monitoring for barge-in)
- STT only processes speech segments (efficiency)
- Barge-in can monitor during output phase

### Decision 4: GPU Allocation
**Choice**: STT and TTS on GPU, LLM on cloud
**Rationale**:
- STT and TTS are latency-critical and benefit most from GPU
- LLM is already cloud (Gemini API)
- Allows scaling to multiple concurrent users (TTS/STT per user, shared LLM API)

### Decision 5: Context Management
**Choice**: In-memory conversation history, no persistence
**Rationale**:
- v1 scope: single transaction per conversation
- No cross-session context needed
- Simplified architecture
- Can add persistence in v2 if needed

---

## Risk Assessment & Mitigation

### Risk 1: IndicConformer Latency Too High
**Likelihood**: Medium
**Impact**: High (breaks <1s target)
**Mitigation**:
- Profile inference time on target hardware
- Consider model quantization (INT8) if needed
- Fallback: use smaller IndicConformer variant if exists
- Last resort: switch to different Konkani STT (e.g., Whisper fine-tuned)

### Risk 2: No Good Marathi Streaming TTS
**Likelihood**: High
**Impact**: Medium (can work around)
**Mitigation**:
- Sentence buffering provides acceptable latency
- Can split long sentences at commas/clauses
- Quality preserved vs. word-level streaming
- Consider fine-tuning Coqui TTS for Konkani if budget allows

### Risk 3: Gemini API Latency/Reliability
**Likelihood**: Medium
**Impact**: Medium (user experience degraded)
**Mitigation**:
- Use Gemini Flash (faster than Pro)
- Implement retry logic with exponential backoff
- Consider fallback to local LLM for v2 (e.g., Indic LLaMA)
- Monitor API latency and alert on degradation

### Risk 4: Barge-in Audio Glitches
**Likelihood**: Medium
**Impact**: Medium (disrupts conversation)
**Mitigation**:
- Smooth audio fade-out on interrupt (not hard stop)
- Proper buffer management (avoid underruns)
- Test extensively with real users
- Tune VAD sensitivity (avoid false positives)

### Risk 5: VRAM Overflow on RTX 4050
**Likelihood**: Medium
**Impact**: High (development blocked)
**Mitigation**:
- Profile VRAM usage per component
- Use model quantization if needed
- Consider CPU fallback for TTS during dev
- Production hardware (RTX 5000) has ample VRAM

---

## References & Further Reading

### Pipecat Framework
- **GitHub**: https://github.com/pipecat-ai/pipecat
- **Docs**: https://docs.pipecat.ai/
- **Key concepts**: Frame processors, pipeline tasks, transports

### Voice Agent Architectures
- **Streaming STT patterns**: Buffered vs. incremental
- **LLM streaming**: Token-by-token generation
- **TTS streaming**: Challenges with generative models
- **Barge-in mechanisms**: Interrupt handling patterns

### Component Technologies
- **IndicConformer**: AI4Bharat NeMo ASR models
- **NeMo**: NVIDIA's speech AI toolkit
- **Gemini API**: Google's generative AI with streaming
- **Silero VAD**: Open-source voice activity detection
- **Indic TTS**: AI4Bharat TTS models for Indian languages

### Performance Optimization
- **Async Python**: Event loops, executors, non-blocking I/O
- **Audio processing**: Sample rates, bit depths, buffer sizes
- **GPU optimization**: CUDA streams, batch processing, memory management

---

## Appendix: Component Comparison Matrix

### STT Options
| Model | Streaming | Konkani | Latency | VRAM | Quality |
|-------|-----------|---------|---------|------|---------|
| IndicConformer (NeMo) | No (batch) | Yes | ~300ms | ~2GB | High |
| Whisper (fine-tuned) | No (batch) | Possible | ~500ms | ~1GB | High |
| Google Cloud STT | Yes | Limited | ~200ms | N/A (cloud) | Medium |

**Choice**: IndicConformer (validated, Konkani-specific)

### LLM Options
| Model | Streaming | Konkani | Latency | Cost | Self-hosted |
|-------|-----------|---------|---------|------|-------------|
| Gemini Flash | Yes | Yes | ~200ms | Low | No (cloud) |
| GPT-4 | Yes | Yes | ~300ms | High | No (cloud) |
| Indic LLaMA | Yes | Possible | ~500ms | Free | Yes (local) |

**Choice**: Gemini Flash (validated, fast, cloud-acceptable for v1)

### TTS Options
| Model | Streaming | Marathi | Latency | VRAM | Quality |
|-------|-----------|---------|---------|------|---------|
| Parler-TTS | No (batch) | No (English) | ~400ms | ~2GB | Medium (for Konkani) |
| AI4Bharat IndicTTS | No (batch) | Yes | ~600ms | ~3GB | High (for Konkani) |
| Coqui TTS | Possible | Possible (ft) | ~500ms | ~2GB | Medium-High |

**Choice**: Start with Parler-TTS, migrate to IndicTTS or Coqui

### VAD Options
| Model | Latency | Accuracy | CPU Usage | GPU Needed |
|-------|---------|----------|-----------|------------|
| Silero VAD | ~10ms | High | Low | No |
| WebRTC VAD | ~5ms | Medium | Very Low | No |
| PyAudio VAD | ~20ms | Low | Low | No |

**Choice**: Silero VAD (best accuracy, acceptable latency)

---

## Glossary

- **Frame**: Discrete unit of data in Pipecat pipeline (audio chunk, text, control signal)
- **Processor**: Component that transforms frames (STT, LLM, TTS, etc.)
- **Transport**: I/O interface for audio (mic/speaker, WebRTC, telephony)
- **VAD**: Voice Activity Detection - determines when user is speaking
- **Barge-in**: User interrupting the agent mid-response
- **Time-to-first-audio**: Latency from user finishing speech to agent starting response
- **Streaming**: Processing data incrementally (vs. waiting for complete input)
- **Sentence buffering**: Collecting tokens until sentence boundary before processing
- **Pipeline task**: Orchestrator that manages frame flow through processors

---

*Research completed: 2026-01-25*
*Author: Claude (GSD Project Researcher)*
*Next step: Use this architecture to inform roadmap phase structure*
