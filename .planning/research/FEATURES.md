# Real-Time Voice Agent Features Research

**Research Date**: 2026-01-25
**Research Question**: What features do production real-time voice agents need? What's table stakes vs differentiating features for natural conversation?
**Context**: Building v1 Konkani voice agent with focus on real-time voice pipeline (FIR workflow deferred to v2)

---

## Executive Summary

Real-time voice agents require a fundamentally different architecture than offline STT→LLM→TTS pipelines. The difference between "table stakes" and "differentiating" features is dramatic:

**Table Stakes** = Users leave if missing (breaks the real-time illusion)
**Differentiators** = Users delight when present (enhances the experience)
**Anti-Features** = Complexity that doesn't serve v1's core goal

For v1 success, focus ruthlessly on **latency, streaming, and interruption handling**. Everything else is secondary or deferred.

---

## Table Stakes Features
> *Must have or users will reject the agent as "not conversational"*

### 1. Streaming Audio Output (CRITICAL)
**What**: Start speaking immediately as tokens are generated, don't wait for complete response

**Why Table Stakes**:
- Human conversations don't wait for full sentences before speaking
- Perceived latency is what matters (time-to-first-audio << total response time)
- Users expect immediate feedback that the agent "heard them"

**Complexity**: Medium
- Requires streaming TTS model (not all TTS supports streaming)
- Need audio chunking/buffering logic
- Pipecat handles this orchestration

**Dependencies**:
- Streaming TTS model (Parler-TTS supports streaming)
- Audio transport that supports chunk-by-chunk playback
- LLM streaming API (Gemini supports streaming)

**Implementation Notes**:
- Pipecat's `TTSService` abstraction handles streaming
- Need to verify Parler-TTS/Marathi model supports streaming mode
- Target: <1s time-to-first-audio (acceptable up to 1.5s)

**v1 Priority**: MUST HAVE

---

### 2. Barge-In / Interruption Handling (CRITICAL)
**What**: Detect when user starts speaking during agent output and immediately stop/listen

**Why Table Stakes**:
- Real conversations allow interruptions
- Without it, users must wait for agent to finish (feels robotic)
- Critical for error correction ("no wait, I meant...")

**Complexity**: Medium-High
- Need continuous audio monitoring during agent speech
- VAD must run concurrently with TTS output
- State management: cancel TTS, flush buffers, switch to listening

**Dependencies**:
- VAD (Voice Activity Detection) running continuously
- Audio pipeline that can cancel mid-stream
- Proper cleanup of TTS/audio buffers

**Implementation Notes**:
- Pipecat has built-in barge-in support via `allow_interruptions`
- Uses Silero VAD or WebRTC VAD for detection
- Must tune VAD sensitivity (avoid false positives from echo/noise)

**v1 Priority**: MUST HAVE

---

### 3. Voice Activity Detection (VAD) for Turn-Taking (CRITICAL)
**What**: Automatically detect when user finishes speaking (without explicit button press)

**Why Table Stakes**:
- Natural conversations have fluid turn-taking
- Button-press UX is transactional, not conversational
- Users expect agent to "know" when they're done

**Complexity**: Medium
- Need robust VAD model (Silero, WebRTC)
- Tune end-of-speech timeout (too short = cuts off user, too long = laggy)
- Handle pauses mid-sentence (don't trigger false end-of-turn)

**Dependencies**:
- VAD model (Silero recommended for accuracy)
- Audio preprocessing (noise reduction helps VAD accuracy)
- Configuration: `vad_timeout`, `vad_threshold`

**Implementation Notes**:
- Pipecat integrates Silero VAD natively
- Typical timeout: 0.5-1.0s of silence = end of turn
- May need tuning based on Konkani speech patterns

**v1 Priority**: MUST HAVE

---

### 4. Low Latency Pipeline (<1s Time-to-First-Audio) (CRITICAL)
**What**: User stops speaking → agent starts responding in <1 second

**Why Table Stakes**:
- Humans respond within 200-600ms in natural conversation
- >2s latency feels like the agent is "thinking too hard" (unnatural)
- Acceptable degradation: up to 1.5s for longer/complex responses

**Complexity**: High
- Requires optimization across entire stack:
  - STT latency: streaming transcription (not batch)
  - LLM latency: fast model + prompt optimization
  - TTS latency: fast synthesis + streaming output
  - Network latency: minimize API round-trips
- GPU optimization: model batching, quantization, VRAM management

**Dependencies**:
- Streaming STT (IndicConformer must support streaming or use chunked processing)
- Fast LLM (Gemini Flash is fast, but cloud adds network latency)
- Fast TTS with streaming (Marathi model TBD)
- Local GPU with sufficient VRAM (RTX 4050 = 8GB)

**Implementation Notes**:
- Measure each component's latency separately
- Pipecat provides latency metrics/logging
- Budget breakdown (target):
  - STT: 200-300ms (streaming)
  - LLM: 300-500ms (first token from Gemini)
  - TTS: 200-300ms (first audio chunk)
  - Total: 700-1100ms

**Optimization Strategies**:
- Use streaming wherever possible
- Reduce prompt token count (shorter system prompts)
- Pre-warm models (keep loaded in memory)
- Consider LLM caching for common phrases

**v1 Priority**: MUST HAVE

---

### 5. Echo Cancellation / Noise Handling (IMPORTANT)
**What**: Prevent agent's own output from being picked up as user input (echo loop)

**Why Table Stakes**:
- Desktop/kiosk has mic + speakers in close proximity
- Echo creates feedback loops (agent hears itself and responds)
- Background noise triggers false VAD activations

**Complexity**: Medium
- Acoustic echo cancellation (AEC) algorithms
- Noise suppression preprocessing
- Hardware considerations (directional mic, speaker placement)

**Dependencies**:
- AEC library (WebRTC has built-in AEC)
- Audio preprocessing pipeline
- Hardware configuration (mic placement, gain settings)

**Implementation Notes**:
- Pipecat integrates WebRTC transport which includes AEC
- May need additional noise gate/suppression (RNNoise, Krisp)
- Test in actual kiosk environment (acoustic conditions matter)

**v1 Priority**: IMPORTANT (test early, may need hardware adjustments)

---

### 6. Graceful Error Handling (IMPORTANT)
**What**: Handle failures (STT error, LLM timeout, TTS crash) without breaking conversation

**Why Table Stakes**:
- Production systems fail (network, GPU OOM, API limits)
- Silent failures are worse than error messages
- Users need feedback when something goes wrong

**Complexity**: Low-Medium
- Try-catch blocks around each component
- Fallback responses ("I didn't catch that, could you repeat?")
- Logging and monitoring

**Dependencies**:
- None (pure application logic)

**Implementation Notes**:
- Pipecat has error callback hooks
- Pre-record fallback audio for common errors (in Konkani)
- Distinguish transient errors (retry) vs permanent (reset)

**v1 Priority**: IMPORTANT

---

## Differentiating Features
> *Nice-to-have improvements that enhance quality but aren't deal-breakers*

### 7. Context-Aware Interruption (NICE-TO-HAVE)
**What**: Intelligently resume or discard interrupted responses based on context

**Why Differentiating**:
- Basic barge-in = just stop and listen (table stakes)
- Smart barge-in = understand whether to resume or start fresh
- Example: User interrupts to correct info → don't resume old response

**Complexity**: High
- Requires LLM context management
- Track interruption point and reason
- Decide: resume, rephrase, or start new turn

**Dependencies**:
- Barge-in (table stakes)
- LLM with conversation history
- State tracking

**Implementation Notes**:
- Defer to v2 (adds complexity without clear v1 value)
- v1: Simple barge-in (stop → listen → new response)

**v1 Priority**: DEFER TO v2

---

### 8. Emotion/Tone Detection (NICE-TO-HAVE)
**What**: Detect user emotion (angry, confused, happy) and adjust agent response tone

**Why Differentiating**:
- Enhances empathy and naturalness
- Particularly valuable for FIR filing (sensitive situations)
- Not essential for basic conversation flow

**Complexity**: High
- Requires emotion detection model (audio or text-based)
- TTS must support emotional prosody control
- Prompt engineering for tone-appropriate responses

**Dependencies**:
- Emotion classifier (audio or LLM-based)
- TTS with prosody control
- Prompt templates for different tones

**Implementation Notes**:
- Defer to v2 (FIR workflow may benefit)
- v1: Neutral, professional tone is acceptable

**v1 Priority**: DEFER TO v2

---

### 9. Filler Words / "Thinking" Sounds (NICE-TO-HAVE)
**What**: Play brief "um", "let me check", or ambient sound during LLM processing delays

**Why Differentiating**:
- Reduces perceived latency (user knows agent is working)
- More natural than silence during long LLM responses
- Common in high-polish voice assistants (Google, Alexa)

**Complexity**: Low-Medium
- Pre-record filler audio snippets
- Trigger when LLM latency > threshold (e.g., 1s)
- Don't overuse (becomes annoying)

**Dependencies**:
- Pre-recorded audio in Konkani ("एक मिनिट", "सांगतो", etc.)
- Latency detection logic
- Audio mixing (play filler while waiting for TTS)

**Implementation Notes**:
- Easy to add later
- Test if actually helpful vs distracting in Konkani context

**v1 Priority**: DEFER (add if latency testing shows need)

---

### 10. Multi-Turn Context Awareness (NICE-TO-HAVE)
**What**: Remember previous turns in conversation for follow-up questions

**Why Differentiating**:
- Basic single-turn works for simple Q&A
- Multi-turn needed for complex interactions ("What about the other option?")
- Current Gemini integration already has chat history

**Complexity**: Low (already implemented in prototype)
- Gemini API maintains chat history
- Just need to preserve across Pipecat pipeline

**Dependencies**:
- LLM with conversation memory
- Session management

**Implementation Notes**:
- Prototype already has this via Gemini chat
- Ensure Pipecat LLM wrapper preserves history
- Decide: How many turns to keep? (memory/latency tradeoff)

**v1 Priority**: INCLUDE (low-hanging fruit, already works)

---

### 11. Wake Word / Push-to-Talk (OPTIONAL)
**What**: Require "hey agent" wake word or button press to activate listening

**Why Differentiating**:
- Reduces false activations (VAD picking up background speech)
- Some users prefer explicit control
- Not standard for kiosk (conversation is opt-in by approaching)

**Complexity**: Medium
- Wake word: Requires wake word detection model (Porcupine, Snowboy)
- Push-to-talk: Simple button UI

**Dependencies**:
- Wake word model (if using that approach)
- UI for push-to-talk button

**Implementation Notes**:
- Kiosk context: User explicitly approaches to interact
- VAD-only is more natural (user just starts speaking)
- Push-to-talk = fallback if VAD too noisy

**v1 Priority**: DEFER (add only if VAD issues arise)

---

### 12. Visual Feedback (Audio Waveform, Status Indicators) (NICE-TO-HAVE)
**What**: Show visual cues: "listening", "thinking", "speaking" states + audio waveforms

**Why Differentiating**:
- Helps users understand agent state
- Useful in noisy environments (user sees agent is listening)
- Not essential since primary modality is voice

**Complexity**: Low
- Simple UI: State icons or text
- Waveform: Audio visualization library

**Dependencies**:
- UI framework (Gradio, tkinter, etc.)
- Audio level monitoring

**Implementation Notes**:
- Desktop/kiosk has screen available
- Simple status text is sufficient for v1
- Defer fancy waveforms to v2

**v1 Priority**: INCLUDE (basic status text), DEFER (waveforms)

---

### 13. Conversation Summarization (NICE-TO-HAVE)
**What**: Summarize conversation at end or periodically for user confirmation

**Why Differentiating**:
- Useful for FIR workflow (confirm details before submission)
- Not needed for general conversation testing
- Adds latency if done synchronously

**Complexity**: Medium
- LLM call for summarization
- UI to present summary
- User confirmation flow

**Dependencies**:
- LLM with summarization capability (Gemini can do this)
- UI for confirmation

**Implementation Notes**:
- Critical for v2 FIR workflow (confirm before filing)
- Not needed for v1 voice pipeline testing

**v1 Priority**: DEFER TO v2

---

## Anti-Features (Don't Build for v1)
> *Complexity that doesn't serve v1's core goal: get real-time voice working*

### 14. Speaker Diarization / Multi-User Support
**What**: Identify and track multiple speakers in conversation

**Why Anti-Feature**:
- v1 is single-user kiosk (one person files one FIR)
- Adds significant complexity (diarization models, user tracking)
- No use case in v1 scope

**v1 Priority**: EXPLICITLY OUT OF SCOPE

---

### 15. Accent/Dialect Adaptation
**What**: Adapt STT/TTS to user's specific Konkani dialect/accent

**Why Anti-Feature**:
- Konkani has regional variations (Goan, Karnataka, etc.)
- Complex: Requires dialect detection, multiple models, or fine-tuning
- IndicConformer is trained on mixed Konkani data (should handle common dialects)

**v1 Priority**: EXPLICITLY OUT OF SCOPE (defer if users report issues)

---

### 16. Persistent User Profiles / Preferences
**What**: Remember user preferences across sessions (speech rate, verbosity, etc.)

**Why Anti-Feature**:
- v1 is single-transaction (each FIR filing is independent)
- Requires user auth and storage (out of scope)
- Kiosk is anonymous usage

**v1 Priority**: EXPLICITLY OUT OF SCOPE

---

### 17. Multi-Language Code-Switching
**What**: Handle mid-conversation language switches (Konkani → English → Konkani)

**Why Anti-Feature**:
- v1 is Konkani-only (design constraint)
- Adds complexity: Language detection, model switching
- Real users may code-switch, but v1 doesn't support it

**v1 Priority**: EXPLICITLY OUT OF SCOPE (defer to v2 if needed)

---

### 18. Real-Time Translation / Transcription Display
**What**: Show live text transcription or translation while speaking

**Why Anti-Feature**:
- Primary modality is voice (not reading transcripts)
- Adds latency (transcription must be fast enough to display real-time)
- Useful for debugging, not user-facing in v1

**v1 Priority**: OUT OF SCOPE (keep for debugging, not user-facing)

---

### 19. Advanced Prosody Control (Emphasis, Pauses, Intonation)
**What**: Fine-grained control over TTS voice characteristics (SSML tags, etc.)

**Why Anti-Feature**:
- Requires TTS model with prosody support
- Complex: Manual annotation or LLM-generated SSML
- Baseline natural TTS is sufficient for v1

**v1 Priority**: OUT OF SCOPE (defer to v2 if TTS quality is poor)

---

### 20. Conversation Analytics / Monitoring Dashboard
**What**: Real-time dashboard showing metrics (latency, errors, user satisfaction)

**Why Anti-Feature**:
- Critical for production v2, but not for v1 prototype
- Focus v1 on building the pipeline, not monitoring infrastructure
- Basic console logging is sufficient for dev

**v1 Priority**: OUT OF SCOPE (defer to v2 for production)

---

## Feature Dependency Map

```
FOUNDATIONAL (no dependencies):
- VAD for Turn-Taking [3]
- Echo Cancellation [5]
- Error Handling [6]

REQUIRES FOUNDATIONAL:
- Streaming Audio Output [1] → requires VAD
- Barge-In [2] → requires VAD
- Low Latency [4] → requires all above working

BUILDS ON CORE:
- Multi-Turn Context [10] → requires basic pipeline working
- Visual Feedback [12] → requires pipeline state tracking

DEFERRED (v2+):
- Context-Aware Interruption [7] → requires barge-in + LLM sophistication
- Emotion Detection [8] → requires base conversation working
- Filler Words [9] → requires latency profiling
- Conversation Summarization [13] → requires FIR workflow
```

---

## Complexity vs Impact Matrix

```
HIGH IMPACT, LOW-MEDIUM COMPLEXITY (Do First):
- Streaming Audio Output [1] ✓
- Barge-In [2] ✓
- VAD [3] ✓
- Multi-Turn Context [10] ✓

HIGH IMPACT, HIGH COMPLEXITY (Tackle Early, Iterate):
- Low Latency [4] ✓ (optimization is iterative)
- Echo Cancellation [5] ✓ (may need hardware iteration)

MEDIUM IMPACT, LOW COMPLEXITY (Include if Time):
- Error Handling [6] ✓
- Visual Feedback [12] ✓ (basic status)

LOW IMPACT or V2 SCOPE (Defer):
- Everything else
```

---

## V1 vs V2 Feature Split

### V1: Real-Time Voice Pipeline
**Goal**: Natural, low-latency Konkani conversation

**Must Have**:
1. Streaming Audio Output ✓
2. Barge-In / Interruption ✓
3. VAD for Turn-Taking ✓
4. Low Latency (<1s) ✓
5. Echo Cancellation ✓
6. Error Handling ✓
10. Multi-Turn Context ✓ (already implemented)
12. Visual Feedback (basic) ✓

**Success Metric**: Can have a natural back-and-forth conversation in Konkani with <1.5s latency and graceful interruptions.

---

### V2: FIR Workflow + Production
**Goal**: Domain-specific functionality and deployment

**Add in V2**:
- FIR-specific prompts and data collection
- Conversation summarization for confirmation [13]
- Emotion detection for sensitive situations [8]
- Backend storage and document generation
- Production monitoring and analytics [20]
- Context-aware interruption [7] (if user feedback demands)
- Filler words [9] (if latency profiling shows gaps)

---

## Technology-Specific Considerations

### Pipecat Framework Features
Pipecat provides out-of-the-box:
- ✅ Streaming audio pipeline
- ✅ Barge-in support (`allow_interruptions`)
- ✅ VAD integration (Silero, WebRTC)
- ✅ Transport abstraction (local audio, WebRTC, telephony)
- ✅ LLM streaming (OpenAI, Anthropic, etc.)
- ✅ TTS streaming (ElevenLabs, Cartesia, etc.)

**Gaps** (need custom implementation):
- IndicConformer STT wrapper (not built-in)
- Gemini LLM wrapper (Pipecat has OpenAI/Anthropic, need custom)
- Marathi TTS integration (depends on model chosen)

### Hardware Constraints (RTX 4050, 8GB VRAM)
**Implications**:
- Can't load multiple large models simultaneously
- Need to optimize VRAM usage:
  - STT: IndicConformer (499MB) ✓ fits
  - TTS: Marathi model size TBD (target <2GB)
  - Total budget: ~6-7GB (leave buffer for PyTorch overhead)
- May need model quantization (FP16 or INT8)
- LLM on cloud (Gemini) = no local VRAM usage ✓

---

## Open Questions / Research Needed

1. **Marathi TTS Model Selection**:
   - Which model supports streaming?
   - VRAM requirements?
   - Quality for Konkani text?
   - Options: IndicTTS, Coqui TTS, VITS-based models

2. **IndicConformer Streaming**:
   - Does NeMo IndicConformer support streaming mode?
   - Or need chunked processing?
   - What's the latency impact?

3. **Pipecat Gemini Integration**:
   - Does Pipecat have Gemini LLM service?
   - Or need custom wrapper using `LLMService` base class?

4. **Echo Cancellation Testing**:
   - Is WebRTC AEC sufficient for kiosk environment?
   - Need hardware testing to validate

5. **VAD Tuning**:
   - What's optimal timeout for Konkani speech patterns?
   - False positive rate acceptable?

---

## Recommendations for V1 Development

### Phase 1: Core Pipeline (Weeks 1-2)
1. Integrate Pipecat framework
2. Implement custom IndicConformer STT service
3. Implement custom Gemini LLM service
4. Select and integrate Marathi TTS (streaming-capable)
5. Get basic pipeline working: speak → transcribe → respond → synthesize

**Success**: End-to-end conversation works (even if latency is high)

### Phase 2: Real-Time Features (Weeks 3-4)
1. Enable streaming output (LLM + TTS)
2. Integrate Silero VAD for turn-taking
3. Implement barge-in/interruption
4. Add basic error handling

**Success**: Conversation feels real-time (no button press, can interrupt)

### Phase 3: Optimization (Weeks 5-6)
1. Profile latency (STT, LLM, TTS components)
2. Optimize hotspots (model loading, prompt length, etc.)
3. Add echo cancellation and noise handling
4. Test on target hardware (RTX 4050)

**Success**: <1.5s latency consistently, no echo issues

### Phase 4: Polish (Week 7)
1. Add visual feedback (status text)
2. Improve error messages (in Konkani)
3. User testing and iteration
4. Documentation

**Success**: Ready for v2 FIR workflow integration

---

## References & Research Sources

### Real-Time Voice AI Patterns
- **Pipecat Documentation**: https://github.com/pipecat-ai/pipecat
  - Design patterns for streaming voice agents
  - Barge-in implementation examples
  - VAD integration best practices

- **Industry Standards** (Google Duplex, Amazon Alexa, OpenAI Realtime API):
  - <1s latency is table stakes for "natural" feel
  - Streaming output dramatically reduces perceived latency
  - Barge-in is expected (not optional) in 2024+

- **Voice AI Latency Research**:
  - Human conversational turn-taking: 200-600ms (Microsoft Research)
  - Acceptable AI latency: <1000ms for simple queries, <2000ms for complex (Google Voice AI)
  - Latency perception: First token matters more than total time

### Konkani Language Considerations
- **Konkani Dialects**: Goan (majority), Karnataka, Kerala variants
  - IndicConformer trained on mixed Konkani corpus (should generalize)
  - TTS: Marathi is linguistically close (Indo-Aryan, similar phonology)

- **Speech Patterns**:
  - Need real user testing to tune VAD timeouts
  - Code-switching with Marathi/English common (but out of scope v1)

### Hardware Optimization
- **VRAM Management**:
  - PyTorch model loading: Use `torch.cuda.empty_cache()` between loads
  - FP16 precision: 50% VRAM savings, minimal quality loss for TTS/STT
  - Model offloading: Keep only active models in VRAM

- **RTX 4050 (8GB) Benchmarks**:
  - Can run Whisper Medium (769MB) + LLM (cloud) + Parler-TTS (600MB)
  - Headroom for IndicConformer (499MB) + Marathi TTS (target <2GB)

---

## Document Metadata

**Author**: AI Research Agent
**Date**: 2026-01-25
**Version**: 1.0
**Status**: Draft for Review
**Next Steps**:
1. Review with developer (Omkar)
2. Validate technology choices (Marathi TTS model)
3. Answer open questions (IndicConformer streaming, Pipecat Gemini integration)
4. Create requirements document based on v1 feature set
