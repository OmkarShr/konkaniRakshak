# Real-Time Voice Agent Pitfalls

Critical mistakes commonly made in real-time voice agent projects and how to avoid them.

**Research Date**: 2026-01-25
**Scope**: Real-time voice agents (specifically Pipecat + NeMo + Konkani)
**Context**: Desktop deployment, GPU-constrained (8GB VRAM dev, 20GB VRAM prod)

---

## 1. Latency Pitfalls

### 1.1 Optimizing the Wrong Latency Metrics

**The Mistake**:
Projects focus on overall end-to-end latency instead of time-to-first-audio. A 3-second response that starts speaking in 800ms feels better than a 2-second response that starts in 1.5s.

**Warning Signs**:
- Measuring only total response time
- Batching all TTS before playback starts
- Waiting for complete LLM response before TTS begins
- User feedback mentions "lag" or "delay" despite acceptable total times

**Prevention Strategy**:
- Measure and optimize **time-to-first-audio** separately from total response time
- Stream TTS output as soon as first few words are generated
- Start playing audio chunks immediately (don't buffer entire response)
- Target: <1s to first audio chunk (user perceives responsiveness)
- Total response time is secondary to perceived immediacy

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Configure streaming output pipeline
- Phase 3 (Latency Optimization): Benchmark and optimize time-to-first-audio

**Project-Specific Notes**:
- With Gemini API (cloud LLM), network latency is unavoidable
- Focus optimization on local components: STT processing time, TTS model initialization
- Stream audio chunks of 0.5-1s each for immediate playback

---

### 1.2 Model Loading on Every Request

**The Mistake**:
Loading STT/TTS models fresh for each conversation or turn instead of keeping them warm in VRAM. This adds 2-10 seconds of latency per request.

**Warning Signs**:
- First response is always slower than subsequent ones
- VRAM usage drops to zero between conversations
- GPU utilization spikes at start of each turn
- Logs show model initialization on every request

**Prevention Strategy**:
- Load models once at application startup
- Keep models resident in VRAM throughout session
- Implement model pooling for multi-user scenarios
- Use torch.no_grad() for inference to prevent unnecessary memory allocation
- Batch size = 1 for real-time (no batching latency)

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Initialize models once in pipeline setup
- Phase 5 (GPU Memory Optimization): Ensure models stay loaded

**Project-Specific Notes**:
- IndicConformer: 499MB model, load once at startup
- TTS (Marathi model): Keep loaded, size TBD based on model selection
- On RTX 4050 (8GB VRAM): ~2-3GB for STT, ~2-3GB for TTS, ~2GB overhead = tight fit
- Monitor VRAM with nvidia-smi, ensure models aren't swapped to system RAM

---

### 1.3 Synchronous Pipeline Processing

**The Mistake**:
Processing pipeline stages sequentially instead of overlapping them. Waiting for STT to finish before starting LLM, waiting for LLM to finish before starting TTS.

**Warning Signs**:
- Pipeline stages are blocking (await each stage before next)
- No concurrent processing of audio chunks
- CPU/GPU utilization shows gaps (idle time between stages)
- Code uses sequential function calls instead of streaming generators

**Prevention Strategy**:
- Use Pipecat's streaming architecture (generators/async iterators)
- Start TTS as soon as first LLM tokens arrive (don't wait for complete response)
- Process audio input continuously while output is playing (enable barge-in)
- Overlap STT processing with LLM generation when possible

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Design pipeline for streaming from start
- Phase 2 (NeMo Integration): Ensure IndicConformer outputs streaming results

**Project-Specific Notes**:
- Pipecat handles this naturally if configured correctly
- Key: use Gemini's streaming API (generate_content_stream) not batch API
- STT should output words/phrases as detected, not wait for silence

---

### 1.4 Network Latency Underestimation

**The Mistake**:
Not accounting for cloud LLM API latency variance. Assuming consistent 200ms round-trip when real-world can be 200-1000ms depending on network conditions.

**Warning Signs**:
- Latency is fine in development, terrible in production
- Performance degrades at certain times of day
- Latency spikes correlate with network usage
- No timeout handling for API calls

**Prevention Strategy**:
- Measure p50, p95, p99 latencies for cloud API calls (not just average)
- Design for p95 latency, not p50 (prepare for worst case)
- Implement request timeouts (5-10s) with graceful fallback
- Consider caching common phrases/responses (if applicable)
- Log API latency separately from model processing time

**Which Phase Addresses This**:
- Phase 3 (Latency Optimization): Benchmark API latency under various conditions
- Phase 6 (Error Handling): Add timeout handling for cloud API

**Project-Specific Notes**:
- Gemini API is cloud-hosted: expect 100-500ms baseline latency
- India → Google Cloud: latency depends on routing, time of day
- Streaming helps: first tokens arrive faster than complete response
- Police station network may be slower than development environment

---

## 2. Audio Quality Pitfalls

### 2.1 Sample Rate Mismatches

**The Mistake**:
Mixing audio sample rates across pipeline components (e.g., 16kHz STT, 24kHz TTS, 48kHz audio interface) without proper resampling. Results in distorted, sped-up, or garbled audio.

**Warning Signs**:
- Audio sounds "chipmunk-like" (too fast) or "demonic" (too slow)
- TTS output is distorted or unclear
- STT transcription accuracy is poor despite good model
- ffmpeg or audio library errors about sample rate

**Prevention Strategy**:
- Standardize on single sample rate across entire pipeline (typically 16kHz or 24kHz)
- Document sample rate for each component: STT input, TTS output, audio interface
- Use explicit resampling when rates must differ (librosa, torchaudio, ffmpeg)
- Test audio quality early with real hardware (microphone + speakers)

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Configure consistent sample rates
- Phase 4 (TTS Integration): Ensure Marathi model matches STT sample rate

**Project-Specific Notes**:
- IndicConformer: likely 16kHz input (verify in NeMo config)
- Marathi TTS model: check output sample rate (may be 22.05kHz or 24kHz)
- Desktop audio interface: can usually handle any rate, but standardize
- Recommendation: use 16kHz throughout for simplicity

---

### 2.2 Streaming TTS Quality Degradation

**The Mistake**:
Assuming streaming TTS quality matches batch TTS. Many models produce degraded audio when generating small chunks due to lack of prosody context.

**Warning Signs**:
- Streaming output sounds robotic or choppy compared to batch
- Word boundaries have clicks or pops
- Prosody (intonation, rhythm) is unnatural in streaming mode
- Users complain audio sounds "wrong" despite correct words

**Prevention Strategy**:
- Test TTS model in streaming mode early (don't assume it works)
- Use models designed for streaming (some explicitly support it)
- Generate slightly larger chunks (1-2 seconds) for better prosody context
- Implement audio smoothing/crossfading between chunks if needed
- Compare streaming vs batch quality side-by-side

**Which Phase Addresses This**:
- Phase 4 (TTS Integration): Test streaming output quality vs batch
- Phase 7 (Testing): A/B test streaming vs batch with Konkani speakers

**Project-Specific Notes**:
- Marathi TTS model: unknown streaming quality (must test)
- Parler-TTS (current): designed for batch, may not stream well
- Alternative: use smaller TTS models that generate faster (full response <1s)
- If streaming quality is poor, consider fast batch TTS instead

---

### 2.3 Language Model vs TTS Model Mismatch

**The Mistake**:
Using a TTS model trained on different script/language than LLM output. English TTS with Devanagari text, or Marathi TTS with Konkani spelling variations.

**Warning Signs**:
- TTS pronunciation is incorrect for many words
- Non-ASCII characters cause errors or are skipped
- Model produces English phonemes for Indic text
- Audio sounds unnatural despite correct script

**Prevention Strategy**:
- Verify TTS model training data matches LLM output (script, language)
- Test with actual Konkani text from Gemini (not just Marathi samples)
- Check character set support (all Devanagari characters used in Konkani)
- Test edge cases: numbers, punctuation, code-mixed text
- Have native speaker validate pronunciation

**Which Phase Addresses This**:
- Phase 4 (TTS Integration): Test Marathi model with Konkani text
- Phase 7 (Testing): Native speaker validation

**Project-Specific Notes**:
- Konkani uses Devanagari script (same as Marathi, Hindi)
- Marathi TTS should handle script correctly
- BUT: Konkani has distinct phonology (pronunciation rules differ from Marathi)
- Marathi model may mispronounce Konkani-specific words
- Test extensively: this is a MAJOR risk for your use case
- Backup plan: use Parler-TTS with Devanagari if Marathi model fails

---

### 2.4 Acoustic Noise Sensitivity

**The Mistake**:
Not accounting for background noise in police station environment. STT and VAD tuned for quiet environments fail with noise.

**Warning Signs**:
- STT transcribes background conversations, phone rings, traffic
- VAD triggers on non-speech sounds (door slams, keyboard clicks)
- Barge-in activates from ambient noise
- System works in office, fails in deployment environment

**Prevention Strategy**:
- Test STT/VAD with realistic background noise early
- Add noise suppression preprocessing (RNNoise, WebRTC NS)
- Use directional microphone with noise rejection
- Tune VAD threshold for noisy environment (higher threshold = less sensitive)
- Implement speaker diarization if multiple voices present

**Which Phase Addresses This**:
- Phase 7 (Testing): Test in realistic police station environment
- Phase 8 (Deployment): Configure VAD for production noise levels

**Project-Specific Notes**:
- Police stations are NOISY: phones, people talking, footsteps, doors
- Desktop/kiosk setup: use close-talk microphone (headset or boom mic)
- VAD must distinguish user speech from ambient noise
- IndicConformer may handle noise better than some models (test it)

---

## 3. Barge-In/Interruption Pitfalls

### 3.1 Interruption Detection Too Sensitive

**The Mistake**:
VAD threshold too low, causing false positive interruptions. Agent stops mid-sentence when user coughs, says "mm-hmm", or ambient noise triggers VAD.

**Warning Signs**:
- Agent frequently stops mid-response for no reason
- Users report agent is "jumpy" or "cuts itself off"
- Logs show interruptions from very short audio bursts (<200ms)
- Barge-in triggers on non-speech sounds

**Prevention Strategy**:
- Require minimum speech duration for interruption (300-500ms)
- Implement debouncing: wait for sustained speech before interrupting
- Use speech probability threshold (VAD confidence score) not just presence/absence
- Test with realistic user behaviors: "uh-huh", "mm", coughs, etc.
- Add hysteresis: different thresholds for start vs continue

**Which Phase Addresses This**:
- Phase 2 (NeMo Integration): Configure VAD parameters
- Phase 7 (Testing): Test with various interruption patterns

**Project-Specific Notes**:
- Konkani speakers may use backchanneling ("ho", "thik", etc.)
- These short affirmations should NOT trigger interruption
- Recommendation: require 400ms+ continuous speech to interrupt
- Police station noise (phone rings, etc.) should not trigger barge-in

---

### 3.2 Interruption Detection Too Insensitive

**The Mistake**:
VAD threshold too high, causing user to "talk over" agent for several seconds before interruption is detected. Frustrating user experience.

**Warning Signs**:
- Users must shout or speak very loudly to interrupt
- Interruption requires 1-2 seconds of speech before agent stops
- Users report feeling "ignored" when trying to interrupt
- Logs show long delay between user speech start and interruption

**Prevention Strategy**:
- Tune VAD threshold to detect normal speech volume (not just loud speech)
- Target 200-400ms detection latency for clear interruptions
- Test with various user volumes, speaking styles, accents
- Use energy-based VAD + model-based VAD for redundancy
- Monitor false negative rate (missed interruptions) vs false positive rate

**Which Phase Addresses This**:
- Phase 2 (NeMo Integration): Tune VAD parameters
- Phase 7 (Testing): Test interruption responsiveness

**Project-Specific Notes**:
- Users expect immediate interruption (like human conversation)
- 200-400ms detection + 100-200ms agent stop = 300-600ms total
- Test with shy/quiet users (police station visitors may be nervous)

---

### 3.3 Audio Playback Not Cancelable

**The Mistake**:
TTS output is queued in audio buffer and continues playing even after interruption is detected. Agent appears unresponsive.

**Warning Signs**:
- Agent "hears" interruption (logs show detection) but keeps talking for 1-2+ seconds
- Audio playback finishes entire sentence after interruption
- Users must wait for agent to finish before being heard
- Code stops TTS generation but audio buffer still plays

**Prevention Strategy**:
- Use audio playback with immediate stop/flush capability
- Clear audio buffers immediately when interruption detected
- Implement low-latency audio output (small buffer sizes)
- Test interruption latency: user speaks → agent stops
- Target <300ms from detection to audio silence

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Configure audio output for low-latency, cancelable playback
- Phase 2 (NeMo Integration): Implement interruption handling

**Project-Specific Notes**:
- Pipecat should handle this, but verify it works correctly
- Test: interrupt agent mid-sentence, measure time to silence
- Audio buffer size: keep small (50-100ms) for quick cancellation

---

### 3.4 Poor Interruption Recovery

**The Mistake**:
After interruption, agent doesn't handle context correctly. Either repeats previous response, loses thread of conversation, or starts new topic.

**Warning Signs**:
- Agent says same thing twice after interruption
- Agent forgets what it was saying before interruption
- Conversation context is lost or reset after barge-in
- User must repeat question after interrupting

**Prevention Strategy**:
- Maintain conversation state through interruptions
- LLM context includes: what agent was saying + user interruption
- Don't simply discard interrupted response (it may be relevant)
- Test conversation flow: interrupt → user speaks → agent responds relevantly
- Implement "resume" vs "new topic" detection

**Which Phase Addresses This**:
- Phase 2 (NeMo Integration): Design interruption state management
- Phase 7 (Testing): Test conversation flow through interruptions

**Project-Specific Notes**:
- Gemini API maintains chat history: include interrupted exchanges
- When interrupted, log partial response for context
- User interruption becomes new message in conversation history
- Test: "How do I file a FIR?" → agent starts explaining → interrupt: "Wait, I meant for theft" → agent should pivot, not restart

---

## 4. GPU Memory Pitfalls

### 4.1 Memory Leaks in Inference Loop

**The Mistake**:
Not properly managing PyTorch tensors in inference loop, causing gradual VRAM accumulation and eventual OOM crash after N requests.

**Warning Signs**:
- VRAM usage creeps up over time (nvidia-smi shows gradual increase)
- System crashes after 10-100 conversations
- Memory usage doesn't return to baseline after conversation ends
- Garbage collection doesn't free VRAM

**Prevention Strategy**:
- Wrap all inference in torch.no_grad() context
- Explicitly delete tensors after use: del tensor; torch.cuda.empty_cache()
- Use tensor.detach() before passing between pipeline stages
- Monitor VRAM usage during development (nvidia-smi in loop)
- Test long-running sessions (simulate 100+ conversations)

**Which Phase Addresses This**:
- Phase 5 (GPU Memory Optimization): Profile and fix memory leaks
- Phase 7 (Testing): Long-running stress tests

**Project-Specific Notes**:
- RTX 4050 (8GB VRAM): no room for leaks, will OOM quickly
- Test continuous operation for 1+ hour, monitor VRAM
- IndicConformer + TTS model + overhead = ~6-7GB baseline
- Memory leak of 50MB/conversation = crash after 20 conversations

---

### 4.2 Unnecessary Model Copies in VRAM

**The Mistake**:
Loading multiple copies of same model, or keeping models in both VRAM and RAM unnecessarily. Common with multi-user scenarios or improper device management.

**Warning Signs**:
- VRAM usage is 2x+ expected model size
- nvidia-smi shows multiple processes holding same VRAM
- Code loads model multiple times (e.g., per-user instances)
- Model weights are duplicated across devices

**Prevention Strategy**:
- Load each model once, share across conversations/users
- Use single device (cuda:0) consistently, don't scatter across devices
- For multi-user: implement proper model serving (single instance, queued requests)
- Verify model.to('cuda') is called once, not per-request
- Use torch.cuda.list_gpu_processes() to check for duplicates

**Which Phase Addresses This**:
- Phase 5 (GPU Memory Optimization): Audit model loading
- Phase 8 (Deployment): Design multi-user serving if needed

**Project-Specific Notes**:
- v1 is single-user (kiosk): only one conversation at a time
- One instance of IndicConformer, one instance of TTS model
- If multi-user later (v2): queue requests, don't duplicate models

---

### 4.3 Incorrect Mixed Precision Usage

**The Mistake**:
Not using FP16/BF16 inference when possible, or using it incorrectly, causing either unnecessary VRAM usage or numerical instability.

**Warning Signs**:
- Models use more VRAM than expected (FP32 is 2x size of FP16)
- Inference is slower than benchmarks suggest (FP32 on tensor cores)
- Mixed precision causes NaN/Inf outputs (numerical instability)
- Model accuracy degrades with FP16

**Prevention Strategy**:
- Use FP16/BF16 for inference where supported (RTX 4050 supports both)
- Test model accuracy with mixed precision (some models are sensitive)
- Use autocast correctly: with torch.cuda.amp.autocast()
- Convert model weights to FP16 if accuracy is acceptable
- Profile memory and speed with FP32 vs FP16

**Which Phase Addresses This**:
- Phase 5 (GPU Memory Optimization): Test mixed precision
- Phase 3 (Latency Optimization): Compare FP16 vs FP32 speed

**Project-Specific Notes**:
- IndicConformer: test if FP16 degrades STT accuracy (likely okay)
- TTS model: FP16 should be fine (prosody may be slightly different)
- RTX 4050: Tensor Cores accelerate FP16 (faster + less VRAM)
- Target: 20-30% VRAM savings with FP16 (6-7GB → 4-5GB baseline)

---

### 4.4 Fragmented VRAM Allocation

**The Mistake**:
Creating and destroying tensors in unpredictable patterns, causing VRAM fragmentation. Available memory exists but can't be allocated as contiguous block.

**Warning Signs**:
- OOM errors despite nvidia-smi showing free VRAM
- Allocation errors for small tensors (<100MB)
- Frequent torch.cuda.empty_cache() calls in code
- Memory usage is spiky/erratic

**Prevention Strategy**:
- Pre-allocate buffers for common tensor sizes
- Reuse tensors instead of creating new ones per-request
- Use torch.cuda.memory_summary() to diagnose fragmentation
- Call torch.cuda.empty_cache() strategically (not constantly)
- Design pipeline with predictable memory patterns

**Which Phase Addresses This**:
- Phase 5 (GPU Memory Optimization): Profile memory allocation patterns
- Phase 7 (Testing): Stress test for fragmentation issues

**Project-Specific Notes**:
- Real-time audio: tensor sizes vary with utterance length
- Pre-allocate max-size buffers, slice as needed
- Test with various utterance lengths (1s, 5s, 10s, 30s)

---

## 5. Integration Pitfalls

### 5.1 Pipecat + NeMo Incompatibility

**The Mistake**:
Assuming NeMo models integrate seamlessly with Pipecat. They use different abstractions (NeMo expects file paths or audio arrays, Pipecat uses streaming generators).

**Warning Signs**:
- No built-in Pipecat service for NeMo models
- Code has awkward adapters between NeMo and Pipecat
- Audio must be written to temp files for NeMo processing
- Real-time streaming is broken by batch processing

**Prevention Strategy**:
- Implement custom Pipecat service/frame processor for NeMo
- Use NeMo's streaming/online inference if available (not batch file processing)
- Test integration early (don't assume it works)
- Prototype minimal example: Pipecat → NeMo STT → print output
- Check Pipecat docs for similar integrations (maybe community examples)

**Which Phase Addresses This**:
- Phase 2 (NeMo Integration): Build NeMo-Pipecat adapter
- This is a CRITICAL risk for your project

**Project-Specific Notes**:
- NeMo models are designed for batch processing (file → file)
- Pipecat expects streaming (audio frame → text frame → audio frame)
- You'll need to write adapter layer
- IndicConformer may support streaming via `transcribe_generator()` or similar
- Prototype this FIRST before building full pipeline

---

### 5.2 Audio Format Conversion Overhead

**The Mistake**:
Excessive audio format conversions between pipeline stages (PCM ↔ WAV ↔ numpy ↔ torch) adding latency and CPU overhead.

**Warning Signs**:
- CPU usage is high during audio processing
- Profiler shows significant time in format conversion functions
- Audio is encoded/decoded multiple times per chunk
- Different libraries expect different formats (conflicts)

**Prevention Strategy**:
- Standardize on single internal format (typically float32 PCM numpy array)
- Convert once at input, once at output (not between stages)
- Use zero-copy operations where possible (torch tensor from numpy without copy)
- Profile audio conversion overhead (should be <10ms per chunk)

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Design consistent audio format pipeline
- Phase 3 (Latency Optimization): Profile and eliminate unnecessary conversions

**Project-Specific Notes**:
- Pipecat uses PCM audio frames (bytes or numpy)
- NeMo expects numpy arrays or torch tensors
- TTS outputs numpy or torch tensors
- Desktop audio I/O: PyAudio or sounddevice (numpy arrays)
- Recommendation: numpy float32 arrays throughout

---

### 5.3 Synchronization Issues with Async Code

**The Mistake**:
Mixing sync and async code incorrectly, causing blocking calls in async context or vice versa. Common with ML models (sync) in async web frameworks.

**Warning Signs**:
- Pipeline freezes or hangs intermittently
- Error messages about "coroutine not awaited" or "blocking in async context"
- Inference blocks entire application (no concurrent processing)
- Code has awkward asyncio.run() calls or threading hacks

**Prevention Strategy**:
- Use asyncio.to_thread() to wrap sync ML inference in async context
- Or run ML models in separate processes (multiprocessing)
- Understand Pipecat's threading model (likely async event loop)
- Test concurrent requests if multi-user (even if v1 is single-user)
- Avoid asyncio.run() in library code (use await properly)

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Understand Pipecat's async/threading model
- Phase 2 (NeMo Integration): Properly integrate sync NeMo inference

**Project-Specific Notes**:
- NeMo models are likely synchronous (blocking inference)
- Pipecat may expect async processing
- Use asyncio.to_thread() or run_in_executor() for NeMo inference
- Test: ensure barge-in works while model is processing

---

### 5.4 Gemini API Rate Limiting

**The Mistake**:
Not handling Gemini API rate limits or quota exhaustion gracefully. Application crashes or hangs when limits are hit.

**Warning Signs**:
- Errors about rate limits or quota exceeded
- Requests fail during high usage periods
- No retry logic or exponential backoff
- Application crashes instead of degrading gracefully

**Prevention Strategy**:
- Implement retry logic with exponential backoff for rate limit errors
- Monitor API quota usage (requests per minute, tokens per day)
- Design for quota limits: fewer, longer messages vs many short messages
- Implement fallback responses for API failures
- Test behavior when quota is exhausted

**Which Phase Addresses This**:
- Phase 6 (Error Handling): Add retry logic and fallback responses
- Phase 7 (Testing): Test rate limit scenarios

**Project-Specific Notes**:
- Gemini API has free tier limits: ~60 requests/minute
- Police kiosk usage pattern: ~1 conversation every 5-10 minutes (low volume)
- Rate limits unlikely in normal usage BUT test it anyway
- Fallback: "I'm having trouble connecting. Please try again in a moment."

---

## 6. Voice Activity Detection (VAD) Pitfalls

### 6.1 Confusing VAD with STT

**The Mistake**:
Using STT to detect end-of-speech instead of dedicated VAD. STT is slower and less reliable for turn-taking detection.

**Warning Signs**:
- Turn-taking feels slow or unnatural
- Agent starts responding before user finishes sentence
- Or agent waits too long after user stops speaking
- No separate VAD component in pipeline

**Prevention Strategy**:
- Use dedicated VAD model (Silero VAD, WebRTC VAD, Picovoice VAD)
- VAD runs continuously, STT only processes when VAD detects speech
- VAD decision latency: <100ms (much faster than STT)
- STT processes complete utterances after VAD marks end-of-speech

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Choose and integrate VAD
- Phase 2 (NeMo Integration): Coordinate VAD + STT

**Project-Specific Notes**:
- Pipecat likely has built-in VAD support (check docs)
- Silero VAD is popular, lightweight, good accuracy
- VAD parameters: speech threshold, silence duration for end-of-speech (typically 500-700ms)

---

### 6.2 End-of-Speech Detection Too Fast

**The Mistake**:
VAD marks end-of-speech too quickly, cutting off user mid-sentence during natural pauses (thinking, breathing).

**Warning Signs**:
- User sentences are cut off mid-thought
- STT transcribes incomplete utterances
- Users report being "interrupted" by agent
- Agent responds before user finishes speaking

**Prevention Strategy**:
- Tune VAD silence duration threshold (500-800ms typical)
- Longer threshold for thoughtful responses (police report = user may pause to think)
- Test with real users: observe natural pause lengths
- Consider language-specific patterns (Konkani speaking rhythm)

**Which Phase Addresses This**:
- Phase 2 (NeMo Integration): Configure VAD parameters
- Phase 7 (Testing): Test with native speakers

**Project-Specific Notes**:
- Police FIR reporting: users may pause to recall details
- Err on side of longer silence threshold (700-1000ms)
- Better to wait 0.5s too long than cut off user mid-sentence

---

### 6.3 End-of-Speech Detection Too Slow

**The Mistake**:
VAD silence threshold too long, causing awkward delays after user finishes speaking. Conversation feels unnatural.

**Warning Signs**:
- Noticeable pause after user stops speaking before agent responds
- Users start repeating themselves (think agent didn't hear)
- Total latency is acceptable but "feel" is wrong
- Test users say conversation feels "sluggish"

**Prevention Strategy**:
- Balance silence threshold: long enough for pauses, short enough for responsiveness
- Typical sweet spot: 500-700ms for conversational AI
- Test with multiple users: some speak faster, some slower
- Adjust based on use case (FIR filing may need longer threshold)

**Which Phase Addresses This**:
- Phase 2 (NeMo Integration): Tune VAD parameters
- Phase 7 (Testing): Test conversation flow with users

**Project-Specific Notes**:
- Start with 600ms silence threshold, adjust based on testing
- Police station visitors may be nervous → speak hesitantly → need longer threshold
- Confident speakers: 500ms is fine
- Nervous speakers: 800-1000ms may be needed

---

### 6.4 Cross-Talk Handling

**The Mistake**:
Not handling simultaneous speech (agent and user talking at same time). VAD only detects speech presence, doesn't distinguish sources.

**Warning Signs**:
- Barge-in doesn't work reliably when agent is speaking
- STT transcribes mix of user + agent audio (if using same microphone)
- Echo cancellation issues (agent hears itself)
- Conversation breaks down when both speak simultaneously

**Prevention Strategy**:
- Implement acoustic echo cancellation (AEC) so agent doesn't hear itself
- Use separate audio input/output devices if possible
- Test barge-in extensively: user speaks while agent is talking
- Consider reference audio subtraction (subtract agent's output from input)

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Configure audio I/O for echo cancellation
- Phase 7 (Testing): Test simultaneous speech scenarios

**Project-Specific Notes**:
- Desktop/kiosk: use headset or separate mic/speakers for better isolation
- Echo cancellation is CRITICAL for barge-in to work
- Pipecat may have built-in AEC (check docs)
- Test: agent speaks → user interrupts → ensure agent hears user, not itself

---

## 7. Testing & Deployment Pitfalls

### 7.1 Testing Only in Quiet Office Environment

**The Mistake**:
All testing done in ideal conditions (quiet room, clear speech, good microphone). Fails in real deployment (noisy police station).

**Warning Signs**:
- Development demos are flawless, production is unusable
- Complaints about accuracy/reliability after deployment
- Background noise wasn't considered during development
- No field testing in actual police station

**Prevention Strategy**:
- Test in realistic environment EARLY (by Phase 7 at latest)
- Bring prototype to police station for field testing
- Simulate noise in development: play background audio (crowd, traffic, phones)
- Test with various microphones (not just high-end studio mic)
- Include real users in testing (not just developers)

**Which Phase Addresses This**:
- Phase 7 (Testing): Field testing in police station
- Phase 8 (Deployment): Final validation in production environment

**Project-Specific Notes**:
- Police stations are NOISY: phones, people, doors, traffic outside
- Field test is ESSENTIAL (don't skip this)
- Test at different times: morning (busy) vs evening (quieter)
- Test with shy/nervous users (not just confident test users)

---

### 7.2 No Performance Benchmarking Under Load

**The Mistake**:
Only testing single conversations, not continuous operation or rapid sequential requests. Performance degrades over time (memory leaks, thermal throttling).

**Warning Signs**:
- First few conversations are fast, later ones slow down
- System becomes unstable after 30+ minutes of operation
- GPU thermal throttling not considered
- No stress testing or endurance testing

**Prevention Strategy**:
- Run 100+ conversation stress test (simulate full day of usage)
- Monitor GPU temperature, CPU usage, VRAM over time
- Test rapid-fire conversations (back-to-back requests)
- Profile memory usage over extended period
- Ensure performance doesn't degrade over time

**Which Phase Addresses This**:
- Phase 7 (Testing): Stress testing and endurance testing
- Phase 5 (GPU Memory Optimization): Profile memory over time

**Project-Specific Notes**:
- Police kiosk may run 8+ hours per day
- RTX 4050 in laptop: thermal constraints (may throttle if hot)
- Test 100 conversations in 2 hours: simulates busy day
- Monitor: VRAM usage, GPU temp, latency over time

---

### 7.3 Single Speaker Testing Only

**The Mistake**:
Testing with only one or two voices (developers). Model fails with accents, age ranges, genders not in test set.

**Warning Signs**:
- STT accuracy varies wildly across users
- Some users report "it doesn't understand me"
- Model works for male voices but not female, or vice versa
- No accent variation in testing

**Prevention Strategy**:
- Test with diverse speakers: male/female, young/old, various accents
- Konkani has regional variations: test Goan Konkani speakers specifically
- Include shy/quiet speakers (police station visitors may be nervous)
- Test with non-native Konkani speakers (if applicable)
- Measure STT accuracy per demographic (identify weak spots)

**Which Phase Addresses This**:
- Phase 7 (Testing): Diverse user testing
- Phase 8 (Deployment): Beta testing with real police station visitors

**Project-Specific Notes**:
- Goan Konkani accent is specific (different from Karwar, Mangalore dialects)
- Test with actual Goa residents, not just any Konkani speakers
- Age range: police station visitors could be 18-80 years old
- Gender: ensure equal accuracy for male/female voices

---

### 7.4 Development vs Production GPU Mismatch

**The Mistake**:
Optimizing for RTX 4050 (8GB VRAM) but deploying on RTX 5000 (20GB VRAM) without testing. Or vice versa: works on 20GB GPU, fails on 8GB.

**Warning Signs**:
- Code has hardcoded GPU-specific values (batch sizes, model configs)
- No testing on production hardware before deployment
- VRAM usage is optimized for wrong GPU
- Performance characteristics differ between dev and prod

**Prevention Strategy**:
- Test on BOTH dev GPU (RTX 4050) and prod GPU (RTX 5000) before v1 release
- Design code to auto-detect VRAM and adjust parameters
- Optimize for smallest GPU (RTX 4050) to ensure it works everywhere
- Document GPU-specific configurations and differences

**Which Phase Addresses This**:
- Phase 5 (GPU Memory Optimization): Ensure 8GB VRAM works
- Phase 8 (Deployment): Test on production RTX 5000 GPUs

**Project-Specific Notes**:
- You're developing on RTX 4050 (8GB) - this is the CONSTRAINT
- Production RTX 5000 (20GB) has more headroom: opportunity for optimizations
- Ensure v1 works on RTX 4050 (worst case)
- In production (RTX 5000), can consider: larger TTS model, FP32 instead of FP16, etc.

---

## 8. Language-Specific Pitfalls (Konkani)

### 8.1 Devanagari Rendering Issues

**The Mistake**:
Assuming Devanagari text "just works" everywhere. Character encoding issues, font rendering problems, or text processing that breaks ligatures.

**Warning Signs**:
- Text displays as boxes, question marks, or garbled characters
- Conjuncts (combined characters) are split incorrectly
- Copy-paste breaks Devanagari characters
- Logs show Unicode errors or encoding issues

**Prevention Strategy**:
- Use UTF-8 encoding throughout (files, logs, APIs)
- Test Devanagari rendering in all UI components
- Verify STT output is proper Devanagari (not transliteration)
- Test LLM output maintains Devanagari (no English injection)
- Use Unicode normalization (NFC) consistently

**Which Phase Addresses This**:
- Phase 1-2 (Early): Verify Devanagari handling throughout pipeline
- Phase 7 (Testing): Test with complex Devanagari text

**Project-Specific Notes**:
- IndicConformer outputs Devanagari: verify it's correct Unicode
- Gemini API should handle Devanagari (test it)
- Logs: ensure console can display Devanagari (or log as Unicode escapes)
- Test conjuncts: क्ष, त्र, ज्ञ (common in Konkani)

---

### 8.2 Konkani vs Marathi Pronunciation Drift

**The Mistake**:
Assuming Marathi TTS pronunciation is "good enough" for Konkani. Significant pronunciation differences may confuse users.

**Warning Signs**:
- Users report "accent sounds wrong" or "pronunciation is off"
- Some words are unintelligible (even if script is correct)
- Native speakers struggle to understand TTS output
- Words are technically correct but phonetically wrong

**Prevention Strategy**:
- Test Marathi TTS with Konkani text EARLY (Phase 4)
- Get native Konkani speaker feedback on pronunciation
- Document known pronunciation issues (may need workarounds)
- Consider phoneme-level editing if model supports it
- Have backup plan: use different TTS model if Marathi fails

**Which Phase Addresses This**:
- Phase 4 (TTS Integration): Test Marathi TTS with Konkani
- Phase 7 (Testing): Native speaker validation (CRITICAL)

**Project-Specific Notes**:
- THIS IS A MAJOR RISK: Marathi ≠ Konkani phonology
- Marathi TTS may mispronounce Konkani-specific sounds
- Get native Goan Konkani speaker to evaluate (not just any Konkani speaker)
- If pronunciation is poor: consider training custom Konkani TTS (v2) or use Parler-TTS fallback

---

### 8.3 Code-Mixing and Language Detection

**The Mistake**:
Not handling code-mixing (Konkani + English/Hindi words). Gemini may inject English, users may say English words, TTS may choke on mixed text.

**Warning Signs**:
- Gemini responses include English words or phrases
- TTS fails or produces garbage on English words in Devanagari context
- Users naturally mix languages (e.g., "complaint file करायची आहे")
- Language detection ambiguity (is this Konkani, Marathi, or Hindi?)

**Prevention Strategy**:
- Configure Gemini to output ONLY Konkani (strict system prompt)
- Test prompt engineering: "Always respond in Konkani, never use English"
- Handle mixed input: transliterate English words to Devanagari for TTS
- Test common code-mixed phrases (police terminology may include English)

**Which Phase Addresses This**:
- Phase 1 (Pipecat Setup): Configure Gemini prompts for Konkani-only
- Phase 7 (Testing): Test with code-mixed input

**Project-Specific Notes**:
- Indian languages commonly mix English (especially technical terms)
- Police terminology: "FIR", "complaint", "station" may appear in English
- Gemini may inject English despite prompts (test and iterate)
- TTS: Marathi model may handle English words (test it), or skip them

---

### 8.4 Low-Resource Language Data Scarcity

**The Mistake**:
Assuming Konkani has same model availability and quality as Hindi/English. Konkani is low-resource: models are rare, quality varies.

**Warning Signs**:
- Can't find suitable Konkani TTS model
- Available models are poorly trained or outdated
- Documentation is sparse or non-existent
- Community support is minimal

**Prevention Strategy**:
- Research available models EARLY (Phase 0, before starting)
- Have backup plan: Marathi TTS (current plan), or English TTS + transliteration
- Accept quality tradeoffs: low-resource languages = lower quality
- Document model limitations for stakeholders
- Consider training custom models if no suitable options (v2+)

**Which Phase Addresses This**:
- Phase 0 (Pre-planning): Research model availability
- Phase 4 (TTS Integration): Validate chosen model quality

**Project-Specific Notes**:
- Konkani is VERY low-resource language
- IndicConformer STT: validated, working (good news)
- Gemini LLM: handles Konkani (good news)
- TTS: NO native Konkani model found → using Marathi (major risk)
- Stakeholder expectation management: quality may not be perfect

---

## Summary: Critical Path Risks

Based on this research, the highest-risk pitfalls for this specific project are:

### P0 (Must address before v1 release):
1. **NeMo-Pipecat Integration (5.1)**: Custom adapter required, may not be trivial
2. **Marathi TTS for Konkani (8.2)**: Pronunciation quality is unknown, MUST test early
3. **GPU Memory on RTX 4050 (4.1-4.4)**: 8GB is tight, must optimize carefully
4. **Latency Target (1.1-1.4)**: <1s to first audio is aggressive, needs focused optimization
5. **Field Testing in Police Station (7.1)**: Noise and real users are essential validation

### P1 (Should address in v1, can iterate):
6. **Barge-In Detection (3.1-3.4)**: Complex to tune, but Pipecat should help
7. **VAD Configuration (6.1-6.3)**: End-of-speech detection needs tuning for Konkani speakers
8. **Audio Quality (2.1-2.4)**: Sample rates, streaming quality, noise handling

### P2 (Monitor but lower risk):
9. **Gemini API Rate Limits (5.4)**: Low usage pattern, but needs fallback handling
10. **Stress Testing (7.2)**: Important for production, but single-user v1 reduces risk

### Key Success Factors:
- **Test early, test often**: Don't wait until Phase 7 to discover TTS doesn't work
- **Native speaker validation**: Essential for language quality (STT accuracy, TTS pronunciation, LLM fluency)
- **Field testing**: Office demos are not enough, must test in real police station
- **GPU memory discipline**: 8GB VRAM is constraining, monitor constantly

---

**Next Steps**:
- Use this document during roadmap planning to identify which phases address each risk
- Prioritize early validation of highest-risk items (NeMo integration, TTS quality)
- Build in testing checkpoints: don't proceed to next phase if critical risks remain
- Maintain risk register throughout development (update as risks materialize or resolve)
