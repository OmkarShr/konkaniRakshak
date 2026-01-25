# Research Summary: Real-Time Konkani Voice Agent

**Project**: Konkani Conversational AI Agent (v1: Real-Time Voice Pipeline)
**Research Completed**: 2026-01-25
**Synthesis Date**: 2026-01-25

---

## Executive Summary

This project aims to build a production-ready real-time conversational voice agent for Konkani language deployed on desktop/kiosk hardware. Based on comprehensive research across stack, features, architecture, and pitfalls, three critical insights emerge:

**1. Streaming-First Architecture is Non-Negotiable**
Real-time voice agents require fundamentally different architecture than batch STT→LLM→TTS pipelines. The Pipecat framework provides the orchestration layer needed for frame-based streaming, barge-in support, and <1s latency targets. This is the correct foundation.

**2. The TTS Quality Risk is Significant**
Using Marathi TTS as a proxy for Konkani (no native Konkani TTS exists) introduces substantial pronunciation risk. This is a P0 validation item that must be tested with native Goan Konkani speakers in Phase 4. If Marathi TTS quality is unacceptable, the project needs a pivot strategy (fine-tune Coqui XTTSv2 on Marathi, or accept Parler-TTS limitations).

**3. GPU Memory Constraints Require Discipline**
The 8GB VRAM constraint on RTX 4050 (development hardware) leaves minimal margin for error. IndicConformer (499MB) + Marathi TTS (~2GB estimate) + PyTorch overhead (~2GB) = 4.5-5GB baseline. Every component must be profiled, optimized (FP16 where possible), and monitored for memory leaks. Production hardware (2x RTX 5000, 20GB each) provides headroom but v1 must work on 8GB.

**Recommended Approach**: Build incrementally with validation checkpoints. Start with Pipecat + existing validated components (IndicConformer STT, Gemini LLM), integrate Parler-TTS initially, then migrate to Marathi TTS in Phase 4 with rigorous quality testing. Do NOT assume Marathi TTS "will work" - this assumption could derail the project.

---

## Key Findings by Research Area

### Stack Research (STACK.md)

**Core Stack - HIGH Confidence:**
- **Pipecat 0.0.49+**: Industry-standard orchestration framework for streaming voice agents. Purpose-built for barge-in, low-latency, and desktop audio.
- **IndicConformer (NeMo)**: Validated 499MB Konkani STT model. Batch-mode (not streaming), but proven to work with ~300ms latency.
- **Gemini 2.0 Flash API**: Fast streaming LLM (20-50 tokens/sec), excellent Konkani support. Cloud dependency acceptable for v1.
- **Silero VAD v5.0+**: Lightweight (8MB) voice activity detection for turn-taking and barge-in. Superior to WebRTC VAD for GPU environments.

**TTS Stack - MEDIUM Confidence:**
- **IndicTTS v2 (Marathi)**: Primary recommendation. 22kHz output, sentence-level streaming, maintained by AI4Bharat (same as STT). Risk: Marathi phonology ≠ Konkani phonology.
- **Coqui XTTSv2 (fine-tuned)**: Alternative. Better voice cloning, more natural prosody, but larger model (1.8GB) and higher VRAM (2-3GB).
- **Parler-TTS**: Current prototype. Known to work but English-focused. Use as fallback/MVP, not final solution.

**Critical Insight**: No native Konkani TTS exists. Marathi is linguistically closest (Indo-Aryan, Devanagari), but pronunciation drift is a major risk requiring early validation.

**Installation Requirements**:
- Python 3.10 (NeMo constraint - 3.11 has librosa bugs)
- PyTorch 2.1.0+ with CUDA 12.1 (match RTX GPU driver)
- Pipecat 0.0.49 (avoid 0.0.50+ API breaking changes)
- Audio I/O: PyAudio or sounddevice (callback mode for low latency)

**Alternatives Rejected**:
- Gradio (not streaming, no barge-in)
- Whisper (batch-only, 1-3s latency)
- Azure Speech SDK (costs money, limited Konkani)
- Flask/FastAPI (WebSocket overhead for desktop audio)

### Features Research (FEATURES.md)

**Table Stakes Features (Must Have for v1):**

1. **Streaming Audio Output**: Start speaking immediately as tokens generate. Target <1s time-to-first-audio.
2. **Barge-In/Interruption Handling**: Detect user speech during agent output, stop immediately. Critical for natural conversation.
3. **VAD for Turn-Taking**: Automatic end-of-speech detection without button press. Typical timeout: 500-700ms silence.
4. **Low Latency Pipeline**: <1s to first audio chunk. Budget: VAD 50ms + STT 300ms + LLM 300ms + TTS 250ms = 900ms.
5. **Echo Cancellation**: Prevent agent hearing itself in desktop/kiosk setup with mic + speakers proximity.
6. **Error Handling**: Graceful degradation for network failures, GPU OOM, API timeouts.

**Differentiating Features (Include if Low Cost):**

10. **Multi-Turn Context**: Already implemented via Gemini chat history. Low-hanging fruit, keep it.
12. **Visual Feedback**: Basic status text ("Listening...", "Thinking...", "Speaking...") for user awareness.

**Deferred to v2:**

7. Context-aware interruption (requires sophisticated LLM state tracking)
8. Emotion/tone detection (useful for FIR workflow, not core v1)
9. Filler words ("um", "let me check") - add if latency profiling shows gaps
13. Conversation summarization (critical for FIR confirmation, but not needed for v1 voice testing)

**Anti-Features (Explicitly Out of Scope):**

14. Speaker diarization (single-user kiosk)
15. Accent/dialect adaptation (IndicConformer trained on mixed Konkani)
16. Persistent user profiles (anonymous kiosk usage)
17. Multi-language code-switching (Konkani-only by design)
18. Real-time transcription display (voice-first, not reading)
19. Advanced prosody control (baseline TTS sufficient)
20. Analytics dashboard (defer to production v2)

**Success Metric for v1**: Natural back-and-forth Konkani conversation with <1.5s latency and graceful interruptions.

### Architecture Research (ARCHITECTURE.md)

**Core Pattern: Frame-Based Streaming Pipeline**

```
Transport → VAD → STT → LLM → TTS → Transport
   ↑                                      ↓
   └───────── Audio Loop (Barge-in) ──────┘
```

**Key Architectural Principles:**

1. **Event-Driven Processing**: Components process small frames (20ms audio chunks, token-by-token text) and emit immediately downstream.

2. **Concurrent Execution**: LLM generates next sentence while TTS synthesizes previous sentence. Audio input monitored during output for barge-in.

3. **Streaming Over Batch**: Start TTS playback as soon as first sentence completes, don't wait for entire LLM response.

4. **Sentence Buffering for TTS**: Since Marathi TTS is batch-mode, buffer LLM tokens until sentence boundary (।, ., ?, !), then synthesize complete sentences for quality.

**Component Integration Patterns:**

**IndicConformer (NeMo) Integration**:
- NeMo is synchronous/batch, Pipecat is async/streaming
- Wrap in async executor: `asyncio.to_thread()` or `run_in_executor()`
- Buffer audio frames until VAD signals speech_ended
- Process complete utterance through NeMo in thread pool
- Emit TextFrame with transcription

**Gemini API Integration**:
- Use `generate_content_stream()` for token-by-token streaming
- Emit TextFrame for each token/chunk immediately
- Maintain conversation history for multi-turn context
- Implement retry logic with exponential backoff for rate limits

**TTS Integration (Sentence Buffering)**:
- Buffer TextFrame tokens from LLM
- Detect sentence boundaries (Devanagari: । or Roman: . ? !)
- Synthesize complete sentences (batch TTS call per sentence)
- Emit AudioFrame chunks while LLM continues generating
- First sentence audio starts playing ~1s after user stops speaking

**Latency Budget Breakdown**:

| Stage | Target | Optimization Strategy |
|-------|--------|----------------------|
| VAD detection | 50-100ms | 32ms chunks, 0.5 threshold |
| STT processing | 200-300ms | GPU inference, FP16, streaming decode |
| LLM first token | 100-300ms | Gemini Flash, optimized prompts |
| TTS first chunk | 150-250ms | Sentence splitting, GPU inference |
| Audio playback | 20-50ms | Low-latency audio driver |
| **Total** | **520-1000ms** | Achievable with optimization |

**GPU Allocation (RTX 4050, 8GB VRAM)**:

| Component | VRAM | Device | Rationale |
|-----------|------|--------|-----------|
| IndicConformer STT | 1.5-2GB (FP16) | GPU | Required for real-time |
| Marathi TTS | 1-1.5GB (FP16) | GPU | Parallel with STT |
| Silero VAD | 50MB | GPU/CPU | Minimal impact |
| Gemini LLM | 0MB | Cloud | API-based |
| **Total** | **2.5-3.5GB** | - | Comfortable on 8GB GPU |

**Critical Risks Identified:**

1. **NeMo-Pipecat Integration**: No built-in support, requires custom adapter. Must prototype this FIRST.
2. **Barge-in Audio Cancellation**: TTS audio buffer must flush immediately (<100ms) when interruption detected.
3. **Memory Leaks**: PyTorch inference loop must use `torch.no_grad()`, explicit tensor cleanup, and periodic `empty_cache()`.
4. **Sample Rate Mismatches**: Standardize on 16kHz throughout (STT input, TTS output, audio interface) to avoid distortion.

### Pitfalls Research (PITFALLS.md)

**P0 Critical Risks (Must Mitigate Before v1):**

**1. NeMo-Pipecat Integration (Pitfall 5.1)**
- **Risk**: NeMo batch processing incompatible with Pipecat streaming architecture.
- **Mitigation**: Build custom FrameProcessor wrapper in Phase 2. Buffer audio frames until VAD signals end-of-speech, then run NeMo transcription in thread pool. Prototype this EARLY (Phase 1-2).
- **Roadmap Impact**: Phase 2 (NeMo Integration) is high-risk and may take longer than expected.

**2. Marathi TTS for Konkani Quality (Pitfall 8.2)**
- **Risk**: Marathi TTS trained on different phonology, may mispronounce Konkani words significantly.
- **Mitigation**: Test Marathi TTS with real Konkani text in Phase 4. Get native Goan Konkani speaker validation. Have backup plan ready (Coqui XTTSv2 or accept Parler-TTS limitations).
- **Roadmap Impact**: Phase 4 (TTS Integration) is validation gate. Cannot proceed to Phase 5 without confirming TTS quality is acceptable.

**3. GPU Memory Constraints (Pitfalls 4.1-4.4)**
- **Risk**: 8GB VRAM on RTX 4050 leaves <2GB margin. Memory leaks or fragmentation cause OOM crashes.
- **Mitigation**:
  - Use FP16 inference (saves 30-50% VRAM)
  - Wrap inference in `torch.no_grad()`
  - Profile VRAM usage continuously (nvidia-smi)
  - Test 100+ conversation endurance test to detect leaks
- **Roadmap Impact**: Phase 5 (GPU Memory Optimization) is mandatory. Add Phase 7 stress testing to validate no leaks.

**4. Latency Optimization Mismatch (Pitfall 1.1)**
- **Risk**: Optimizing total latency instead of time-to-first-audio. Users perceive delay even if total time is acceptable.
- **Mitigation**: Benchmark and optimize specifically for time-to-first-audio (<1s target). Stream TTS output immediately. Don't batch entire response before playback.
- **Roadmap Impact**: Phase 3 (Latency Optimization) must focus on perceived latency, not just total time.

**5. Field Testing Neglect (Pitfall 7.1)**
- **Risk**: Testing only in quiet office. Fails in noisy police station with real users.
- **Mitigation**: Schedule police station field test in Phase 7. Test with diverse speakers (age, gender, accent). Tune VAD for production noise levels.
- **Roadmap Impact**: Phase 7 (Testing) is not optional. Must include on-site validation.

**P1 Important Risks (Address in v1, Can Iterate):**

**6. Barge-In Tuning (Pitfalls 3.1-3.4)**
- Balance VAD sensitivity: too high = false positives (agent stops randomly), too low = user ignored
- Require 300-500ms sustained speech for interruption (avoid backchanneling triggering it)
- Audio buffer must flush <100ms after detection

**7. VAD End-of-Speech Detection (Pitfalls 6.2-6.3)**
- Too fast: cuts off user mid-sentence
- Too slow: awkward pauses, users repeat themselves
- Sweet spot: 600-800ms silence for FIR context (users pause to recall details)

**8. Sample Rate Mismatches (Pitfall 2.1)**
- IndicConformer: 16kHz input
- Marathi TTS: likely 22kHz output (verify this)
- Desktop audio: 48kHz native
- Solution: Explicit resampling with scipy or librosa

**P2 Monitor (Lower Risk):**

**9. Gemini API Rate Limits (Pitfall 5.4)**
- Free tier: ~60 requests/minute
- Kiosk usage: ~1 conversation per 5-10 minutes (low volume)
- Implement retry with exponential backoff anyway

**10. Stress Testing (Pitfall 7.2)**
- Run 100+ conversation test (simulate 8-hour day)
- Monitor GPU temperature (laptop thermal throttling)
- Verify no performance degradation over time

**Language-Specific Risks:**

**11. Devanagari Rendering (Pitfall 8.1)**
- Use UTF-8 throughout
- Test conjuncts (क्ष, त्र, ज्ञ)
- Verify STT outputs proper Unicode, not transliteration

**12. Code-Mixing (Pitfall 8.3)**
- Users may inject English words ("complaint file करायची आहे")
- Gemini may inject English despite prompts
- Configure strict Konkani-only system prompt

---

## Implications for Roadmap

Based on synthesized research, recommended phase structure:

### Phase 1: Foundation & Pipecat Setup (Week 1-2)
**Goal**: Get basic pipeline skeleton working

**What to Build**:
- Install Pipecat, PyTorch, dependencies
- Set up LocalAudioTransport (desktop audio I/O)
- Build mock pipeline: mic → VAD (placeholder) → print text → TTS (mock) → speaker
- Test audio flow end-to-end without ML models

**Success Criteria**: Audio captured from mic, played to speakers with minimal latency

**Risk Mitigation**: Validates Pipecat works on target hardware before investing in ML integration

**Deliverables**: Working Pipecat pipeline skeleton, documented dependencies

---

### Phase 2: NeMo STT Integration (Week 2-3)
**Goal**: Integrate IndicConformer STT as Pipecat processor

**What to Build**:
- Custom FrameProcessor wrapper for NeMo IndicConformer
- Async executor integration (handle NeMo's synchronous inference)
- Audio frame buffering until VAD signals speech_ended
- Test with recorded Konkani audio samples

**Success Criteria**: Spoken Konkani → accurate transcription with <500ms latency

**Risk Mitigation**: This is **P0 critical risk #1** (NeMo-Pipecat integration). If this doesn't work, entire architecture fails. Prototype EARLY.

**Research Needed**: Check if NeMo has streaming/online inference APIs (better than batch)

**Deliverables**: Working STT processor, latency benchmarks

---

### Phase 3: Gemini LLM Integration (Week 3-4)
**Goal**: Add conversational intelligence with streaming output

**What to Build**:
- GeminiLLMProcessor using `generate_content_stream()`
- Maintain conversation history for multi-turn context
- Configure Konkani-only system prompt (prevent English injection)
- Test token-by-token streaming

**Success Criteria**: Text input → Konkani conversational response (streaming)

**Risk Mitigation**: Gemini API tested early ensures cloud dependency is viable

**Research Needed**: Verify Gemini streaming API latency (p50/p95/p99)

**Deliverables**: LLM processor with streaming, prompt templates

---

### Phase 4: TTS Integration & Quality Validation (Week 4-5)
**Goal**: Synthesize speech and validate Marathi TTS quality for Konkani

**What to Build**:
- Start with Parler-TTS (known working, fallback)
- Integrate IndicTTS v2 (Marathi model) with sentence buffering
- Test pronunciation with native Goan Konkani speaker
- A/B test: Parler-TTS vs IndicTTS quality

**Success Criteria**: Acceptable pronunciation quality validated by native speaker

**Risk Mitigation**: This is **P0 critical risk #2** (TTS quality). If Marathi TTS fails validation, pivot to:
- Option A: Fine-tune Coqui XTTSv2 on Marathi/Konkani data
- Option B: Accept Parler-TTS limitations for v1
- Option C: Defer project until Konkani TTS becomes available

**CHECKPOINT**: Cannot proceed to Phase 5 without TTS quality sign-off

**Deliverables**: Working TTS processor, quality validation report, go/no-go decision

---

### Phase 5: GPU Memory Optimization (Week 5)
**Goal**: Ensure pipeline fits in 8GB VRAM without leaks

**What to Build**:
- Profile VRAM usage per component (nvidia-smi, PyTorch profiler)
- Implement FP16 inference for STT and TTS
- Add proper tensor cleanup (torch.no_grad(), explicit del, empty_cache())
- Run endurance test: 100 conversations, monitor memory

**Success Criteria**: Stable VRAM usage <6GB, no memory leaks over 2 hours

**Risk Mitigation**: This is **P0 critical risk #3** (GPU memory). RTX 4050 constraint requires discipline.

**Deliverables**: Optimized pipeline, memory profiling report

---

### Phase 6: Latency Optimization (Week 6)
**Goal**: Achieve <1s time-to-first-audio

**What to Build**:
- Benchmark each pipeline stage separately
- Optimize hotspots: VAD threshold, STT frame length, LLM prompt length, TTS sentence splitting
- Implement streaming output (don't wait for complete response)
- Test perceived latency vs total latency

**Success Criteria**: p95 time-to-first-audio <1.2s (target <1s)

**Risk Mitigation**: Addresses **P0 risk #4** (latency optimization focus)

**Deliverables**: Latency benchmarks, optimization report

---

### Phase 7: Barge-In & Turn-Taking (Week 7)
**Goal**: Enable natural interruptions and conversation flow

**What to Build**:
- Integrate Silero VAD for continuous monitoring
- Implement interruption detection during TTS playback
- Audio buffer cancellation (<100ms flush time)
- Tune VAD parameters: speech threshold, silence duration
- Test conversation flow: normal turns + interruptions

**Success Criteria**: Clean interruptions with <300ms detection-to-silence latency

**Risk Mitigation**: Addresses **P1 risks #6 and #7** (barge-in and VAD tuning)

**Deliverables**: Working barge-in, VAD configuration documentation

---

### Phase 8: Error Handling & Polish (Week 8)
**Goal**: Graceful degradation and production readiness

**What to Build**:
- Retry logic for Gemini API (rate limits, network failures)
- Error messages in Konkani (audio + text)
- Visual status feedback ("Listening", "Thinking", "Speaking")
- Logging and monitoring hooks
- Echo cancellation testing and tuning

**Success Criteria**: Pipeline recovers from common errors without crashing

**Deliverables**: Robust error handling, user feedback UI

---

### Phase 9: Field Testing & Validation (Week 9)
**Goal**: Test in realistic police station environment

**What to Build**:
- Deploy to police station kiosk
- Test with diverse users: age, gender, accent variation
- Test in noisy environment (phones, people, traffic)
- Tune VAD for production noise levels
- Collect user feedback on naturalness and accuracy

**Success Criteria**: Acceptable STT accuracy, TTS intelligibility, and conversation flow with real users

**Risk Mitigation**: Addresses **P0 risk #5** (field testing). This is mandatory validation.

**CHECKPOINT**: Cannot declare v1 complete without field validation

**Deliverables**: Field test report, identified improvements for iteration

---

### Phase 10: Production Deployment (Week 10)
**Goal**: Deploy to production hardware and harden

**What to Build**:
- Deploy to 2x RTX 5000 GPUs (production kiosk)
- Multi-GPU configuration (if needed for future multi-user)
- Production monitoring and logging
- Documentation: deployment guide, troubleshooting, maintenance

**Success Criteria**: Stable operation for 8+ hour days

**Deliverables**: Production deployment, operations documentation

---

## Research Flags: Which Phases Need Deeper Research?

**Phases Requiring Additional Research:**

1. **Phase 2 (NeMo Integration)**: Research NeMo streaming APIs, investigate if online inference is available. Check NeMo GitHub issues for Pipecat integration examples.

2. **Phase 4 (TTS Integration)**: Research alternative Marathi TTS models (VITS, Coqui, IndicTTS variants). Investigate Coqui XTTSv2 fine-tuning process as backup plan.

**Phases with Standard Patterns (Less Research Needed):**

3. **Phase 1 (Pipecat Setup)**: Well-documented, standard setup.
4. **Phase 3 (Gemini Integration)**: Google SDK has examples, straightforward API.
5. **Phase 5-10**: Standard optimization, testing, deployment patterns.

**Recommendation**: Use `/gsd:research-phase` tool for:
- Phase 2: NeMo streaming/online inference patterns
- Phase 4: Marathi TTS alternatives if IndicTTS v2 fails validation

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack (Core Framework)** | HIGH | Pipecat is proven for voice agents. IndicConformer validated. Gemini API tested. |
| **Stack (TTS)** | MEDIUM | IndicTTS v2 Marathi model exists, but Konkani pronunciation quality unknown. Must validate in Phase 4. |
| **Features** | HIGH | Clear distinction between table stakes, differentiators, and anti-features based on voice AI patterns. |
| **Architecture** | HIGH | Frame-based streaming pipeline is standard pattern. Component boundaries well-defined. |
| **Pitfalls (Technical)** | HIGH | GPU memory, latency, integration risks are well-understood and have mitigation strategies. |
| **Pitfalls (Language)** | MEDIUM | Marathi-Konkani pronunciation drift is unknown risk. Requires native speaker validation. |

**Overall Confidence**: MEDIUM-HIGH

**Primary Gap**: TTS quality for Konkani. This is the largest unknown and could require pivot if Marathi TTS is unacceptable.

---

## Gaps to Address During Planning

1. **TTS Model Selection**: Need to test IndicTTS v2 Marathi model with actual Konkani text samples. Get sample audio for evaluation before Phase 4.

2. **NeMo Streaming Support**: Investigate NeMo documentation for streaming/online inference APIs. Contact AI4Bharat if documentation is unclear.

3. **Hardware Availability**: Confirm access to RTX 5000 GPUs for production testing. If not available, adjust roadmap for RTX 4050-only deployment.

4. **Native Speaker Access**: Identify and schedule Goan Konkani speaker for TTS validation in Phase 4 and field testing in Phase 9. This is critical path.

5. **Police Station Access**: Coordinate with police department for field testing site access in Phase 9. Need approval and logistics planning.

6. **Sample Rate Verification**: Confirm IndicConformer input sample rate (likely 16kHz) and IndicTTS v2 output sample rate (documentation says 22kHz, verify this).

7. **Gemini API Quota**: Clarify production Gemini API tier and costs. Free tier may be sufficient for kiosk, but verify this doesn't block deployment.

---

## Sources (Aggregated)

### Framework & Architecture
- Pipecat GitHub (https://github.com/pipecat-ai/pipecat) - Real-time voice agent patterns, frame-based architecture
- Pipecat Documentation (https://docs.pipecat.ai/) - Pipeline design, component integration
- Daily.co blog - WebRTC vs local audio for voice agents (2024)
- Microsoft Research - Human conversational turn-taking patterns (200-600ms baseline)
- Google Voice AI - Acceptable AI latency standards (<1s simple, <2s complex)

### Konkani/Indic Language Support
- AI4Bharat IndicConformer - Konkani ASR model specifications (validated in prototype)
- AI4Bharat IndicTTS v2 GitHub (https://github.com/AI4Bharat/Indic-TTS) - Marathi TTS documentation
- Coqui TTS GitHub (https://github.com/coqui-ai/TTS) - Multilingual TTS capabilities, fine-tuning guides

### VAD & Audio Processing
- Silero VAD paper (2021) - State-of-the-art voice activity detection
- WebRTC VAD documentation - Lightweight CPU alternative
- PyAudio vs sounddevice - Stack Overflow comparisons (2024)

### GPU Optimization
- NVIDIA NeMo documentation - Streaming ASR configuration, FP16 inference
- PyTorch Mixed Precision Training - AMP and autocast usage
- RTX 4050/5000 specifications - CUDA 12.1 compatibility, tensor core performance

### Industry Patterns
- Google Duplex architecture - Streaming TTS, barge-in patterns
- Amazon Alexa voice AI - Low-latency best practices
- OpenAI Realtime API - Streaming audio architecture patterns

---

## Roadmapper Handoff

**This research provides**:
- Clear stack recommendations with confidence levels
- Feature prioritization (table stakes vs nice-to-have vs anti-features)
- Architecture patterns for component integration
- Risk assessment with mitigation strategies mapped to phases

**Roadmapper should**:
1. Use suggested phase structure (Phases 1-10) as starting point
2. Add validation checkpoints after Phase 4 (TTS quality) and Phase 9 (field testing)
3. Flag Phase 2 (NeMo integration) and Phase 4 (TTS validation) as high-risk, potentially longer
4. Include native speaker coordination in Phase 4 and 9 logistics
5. Budget time for potential TTS pivot if Marathi model fails validation

**Critical Success Factors**:
- Test TTS quality EARLY (don't wait until Phase 7)
- Native speaker validation is MANDATORY
- Field testing in police station is MANDATORY
- GPU memory discipline throughout (monitor constantly on RTX 4050)

---

**Research Status**: COMPLETE

**Next Step**: Requirements definition with specific acceptance criteria for each phase.

**Key Decision Points**:
1. Phase 4: TTS quality go/no-go (if no-go, activate pivot plan)
2. Phase 9: Field testing results (if major issues, iterate before production)

---

*Synthesis completed by Claude (GSD Research Synthesizer)*
*Ready for roadmap planning*
