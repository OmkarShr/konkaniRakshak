# Complete Pipeline Comparison: Konkani vs English vs Hindi

## Overview

This document contrasts all three language pipelines, showing their files, configurations, and fallback mechanisms.

---

## 📊 Quick Comparison Matrix

| Feature | **Konkani Pipeline** | **English Pipeline** | **Hindi Pipeline** |
|---------|---------------------|---------------------|-------------------|
| **Main File** | `ws_pipeline.py` | `ws_pipeline_english.py` | `ws_pipeline_hindi.py` |
| **Startup Script** | `start_pipeline.sh` | `start_pipeline_english.sh` | `start_pipeline_hindi.sh` |
| **WebSocket Port** | 8765 | 8767 | 8768 |
| **STT Service** | Konkani STT (50051) or Multilingual STT (50052) | Multilingual STT (50052) | Multilingual STT (50052) |
| **STT Model** | `indicconformer_stt_kok_hybrid_rnnt_large.nemo` | `indicconformer_stt_multi_hybrid_rnnt_600m.nemo` | `indicconformer_stt_multi_hybrid_rnnt_600m.nemo` |
| **LLM Model** | `gemma2:2b` (Ollama) | `gemma2:2b` (Ollama) | `gemma2:2b` (Ollama) |
| **LLM Port** | 11435 | 11435 | 11435 |
| **TTS Model** | Indic Parler-TTS | Indic Parler-TTS | Indic Parler-TTS |
| **TTS Language** | `kok` (Konkani) | `en` (English) | `hi` (Hindi) |
| **GPU Allocation** | GPU 0 | GPU 0 | GPU 0 |
| **Container Name** | `konkani-pipeline` | `konkani-pipeline-english` | `konkani-pipeline-hindi` |
| **Docker Image** | `konkani-pipeline:latest` | `konkani-pipeline-english:latest` | `konkani-pipeline-hindi:latest` |

---

## 🗂️ Files Used by Each Pipeline

### 1️⃣ Konkani Pipeline

#### Core Files
```
ws_pipeline.py                          # Main pipeline (542 lines)
start_pipeline.sh                       # Startup script with health checks
```

#### Configuration
- **Port**: 8765
- **System Prompt** (Konkani):
  ```
  तूं एक कोंकणी भाशेंतलो (देवनागरी लिपींत) सहाय्यक आसा.
  तूं गोंय पुलिसांखातीर एफआयआर दाखल करपाक मदत करता.
  सदांच कोंकणी भाशेंत जाप दी. जापो मटव्यो आनी स्पश्ट आसच्यो.
  इंग्लीश वापरूं नाका.
  ```
- **Fallback Response**: `"माफ करा, म्हाका तुमचें म्हणणें समजूंक ना. कृपया परत सांगात."` (Sorry, I didn't understand)

#### Dependencies
- **STT**: Can switch between Konkani STT (50051) or Multilingual STT (50052)
- **Models**:
  - Silero VAD
  - Indic Parler-TTS (`ai4bharat/indic-parler-tts`)
  - Ollama LLM (`gemma2:2b`, 1.6GB)

#### Special Features
- **Dual STT Mode**: Supports `konkani` and `multilingual` mode switching via WebSocket
- **Session Persistence**: Stores conversation history by client IP

---

### 2️⃣ English Pipeline

#### Core Files
```
ws_pipeline_english.py                  # English-only pipeline (540 lines)
start_pipeline_english.sh               # Simple startup script
```

#### Configuration
- **Port**: 8767
- **System Prompt** (English with TTS constraints):
  ```
  You are an assistant for the Goa Police to help file FIRs (First Information Reports).
  Always respond in English. Keep responses short and clear.
  Ask for necessary details: complainant name, incident description, date, time, location.
  
  IMPORTANT: Do not use apostrophes or special symbols in your responses.
  Write words out fully instead of using contractions.
  For example, use 'cannot' instead of 'can't', 'do not' instead of 'don't'.
  Avoid quotation marks, asterisks, and other symbols that may confuse text-to-speech.
  ```
- **Fallback Response**: `"Sorry, I am having trouble connecting to my brain right now. Please try again."`

#### Dependencies
- **STT**: Multilingual STT only (50052)
- **TTS Language Hint**: `en`
- **Models**: Same as Konkani (Silero VAD, Indic Parler-TTS, Ollama)

#### Special Features
- **TTS-Friendly Prompts**: Explicitly instructs LLM to avoid symbols that break TTS
- **No Language Switching**: Hard-coded to English
- **Dedicated Endpoint**: Returns cleaner responses for TTS

---

### 3️⃣ Hindi Pipeline

#### Core Files
```
ws_pipeline_hindi.py                    # Hindi-only pipeline (540 lines)
start_pipeline_hindi.sh                 # Simple startup script
```

#### Configuration
- **Port**: 8768
- **System Prompt** (Hindi with TTS constraints):
  ```
  आप गोवा पुलिस के लिए एफआईआर (प्रथम सूचना रिपोर्ट) दर्ज करने में सहायता करने वाले सहायक हैं।
  हमेशा हिंदी में जवाब दें। जवाब छोटे और स्पष्ट रखें।
  आवश्यक विवरण पूछें: शिकायतकर्ता का नाम, घटना का विवरण, तारीख, समय, स्थान।
  
  महत्वपूर्ण: अपने उत्तरों में विशेष चिह्नों या प्रतीकों का उपयोग न करें जो टेक्स्ट-टू-स्पीच को भ्रमित कर सकते हैं।
  उद्धरण चिह्न, तारक चिह्न और अन्य प्रतीकों से बचें। स्वाभाविक हिंदी में स्पष्ट रूप से लिखें।
  ```
- **Fallback Response**: `"Sorry, I am having trouble connecting to my brain right now. Please try again."`

#### Dependencies
- **STT**: Multilingual STT only (50052)
- **TTS Language Hint**: `hi`
- **Models**: Same as others (Silero VAD, Indic Parler-TTS, Ollama)

#### Special Features
- **TTS-Friendly Prompts**: Hindi instructions to avoid symbols
- **No Language Switching**: Hard-coded to Hindi
- **Dedicated Endpoint**: Optimized for Hindi TTS output

---

## 🔄 Fallback Mechanisms

### STT Fallbacks

| Pipeline | Primary STT | Fallback STT | Error Handling |
|----------|------------|--------------|----------------|
| **Konkani** | Konkani STT (50051) | Multilingual STT (50052) via mode switch | Returns empty string on HTTP error |
| **English** | Multilingual STT (50052) | None (single endpoint) | Returns empty string on HTTP error |
| **Hindi** | Multilingual STT (50052) | None (single endpoint) | Returns empty string on HTTP error |

**Code Example (All Pipelines)**:
```python
async def _run_stt(self, pcm_bytes: bytes) -> str:
    audio_b64 = base64.b64encode(pcm_bytes).decode()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{STT_URL}/transcribe",
                json={"audio": audio_b64, "sample_rate": AUDIO_IN_RATE},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("text", "").strip()
                else:
                    logger.error(f"STT error {resp.status}")
                    return ""  # FALLBACK: Empty string
    except Exception as e:
        logger.error(f"STT exception: {e}")
        return ""  # FALLBACK: Empty string
```

---

### LLM Fallbacks

| Pipeline | Primary LLM | Fallback Response | Retry Logic |
|----------|------------|-------------------|-------------|
| **Konkani** | Ollama gemma2:2b | Konkani error message | Try-except with fallback |
| **English** | Ollama gemma2:2b | English error message | Try-except with fallback |
| **Hindi** | Ollama gemma2:2b | English error message | Try-except with fallback |

**Konkani Fallback**:
```python
FALLBACK_RESPONSE = "माफ करा, म्हाका तुमचें म्हणणें समजूंक ना. कृपया परत सांगात."
# Translation: "Sorry, I didn't understand. Please say it again."
```

**English/Hindi Fallback**:
```python
FALLBACK_RESPONSE = "Sorry, I am having trouble connecting to my brain right now. Please try again."
```

**Code Example**:
```python
async def _run_llm(self, user_text: str) -> str:
    try:
        import ollama
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ollama.Client(host=os.getenv("OLLAMA_URL")).chat(
                model="gemma2:2b",
                messages=messages,
                stream=False
            )
        )
        text = response['message']['content'].strip()
        return text
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return FALLBACK_RESPONSE  # FALLBACK: Pre-defined message
```

---

### TTS Fallbacks

| Pipeline | Primary TTS | Error Handling | Sentence Splitting |
|----------|------------|----------------|-------------------|
| **Konkani** | Indic Parler-TTS | Skip failed sentences, continue | Devanagari punctuation (।) |
| **English** | Indic Parler-TTS | Skip failed sentences, continue | Latin punctuation (.) |
| **Hindi** | Indic Parler-TTS | Skip failed sentences, continue | Devanagari punctuation (।) |

**Code Example**:
```python
async def _run_tts_streaming(self, text: str):
    sentences = self._split_sentences(text)
    for idx, sentence in enumerate(sentences):
        try:
            wav = await loop.run_in_executor(tts_executor, _synthesize)
            # Stream audio to browser...
        except Exception as e:
            logger.error(f"TTS error: {e}")
            continue  # FALLBACK: Skip this sentence, continue with next
```

---

## 🆚 Key Differences

### 1. System Prompts

| Aspect | Konkani | English | Hindi |
|--------|---------|---------|-------|
| **Language** | Konkani (Devanagari) | English | Hindi (Devanagari) |
| **TTS Constraints** | ❌ None | ✅ Avoid apostrophes, contractions | ✅ Avoid special symbols |
| **Purpose** | FIR filing assistant | FIR filing assistant | FIR filing assistant |
| **Tone** | Natural Konkani | Formal English | Natural Hindi |

**Why TTS Constraints?**
- English and Hindi pipelines added explicit LLM instructions to avoid symbols like `'`, `*`, `"` that break TTS synthesis
- Konkani didn't need this initially as Devanagari script has fewer TTS-problematic symbols

---

### 2. STT Service Selection

```python
# Konkani: Dynamic switching
stt_url = STT_URL if self.language_mode == "konkani" else MULTILINGUAL_STT_URL

# English: Fixed endpoint
STT_URL = os.getenv("MULTILINGUAL_STT_URL", "http://multilingual-stt:50052")

# Hindi: Fixed endpoint
STT_URL = os.getenv("MULTILINGUAL_STT_URL", "http://multilingual-stt:50052")
```

---

### 3. Startup Scripts

**Konkani** (`start_pipeline.sh`):
- 55 lines with health checks
- Waits up to 30 seconds for STT service
- Validates Gemini API key
- Sets Python path dynamically

**English/Hindi** (`start_pipeline_*.sh`):
- 8 lines minimal script
- No health checks (relies on Docker depends_on)
- Simply exports port and runs Python

---

### 4. Session Management

**All Three Pipelines**:
- Use global session persistence by client IP
- Support conversation history (last 6 exchanges)
- Identical VAD and barge-in logic

```python
GLOBAL_SESSIONS = {}  # {client_ip: conversation_history_list}

async def handle_client(ws):
    client_ip = ws.remote_address[0] if ws.remote_address else "unknown"
    session = PipelineSession(ws)
    
    # Restore history
    if client_ip in GLOBAL_SESSIONS:
        session.conversation = GLOBAL_SESSIONS[client_ip]
    
    try:
        await session.run()
    finally:
        # Save history
        GLOBAL_SESSIONS[client_ip] = session.conversation
```

---

## 🔧 Shared Components

All three pipelines share:

### Models
1. **Silero VAD** - Voice Activity Detection
2. **Indic Parler-TTS** - Text-to-Speech (`ai4bharat/indic-parler-tts`)
3. **Ollama LLM** - Language Model (`gemma2:2b`, 1.6GB)

### Configuration
- Sample rates: 16kHz input, 44.1kHz output
- VAD threshold: 0.50
- Min speech: 200ms
- Min silence: 1200ms
- Max recording: 30 seconds
- Pre-buffer: 500ms

### Features
- Barge-in support (interrupt agent)
- Streaming TTS (sentence-by-sentence)
- WebSocket transport
- Session persistence
- Conversation history (6 messages)

---

## 📈 Resource Usage

| Pipeline | VRAM (GPU 0) | CPU | Network Ports |
|----------|--------------|-----|---------------|
| **Konkani** | ~4.5 GB (VAD + TTS) | Medium | 8765 |
| **English** | ~4.5 GB (VAD + TTS) | Medium | 8767 |
| **Hindi** | ~4.5 GB (VAD + TTS) | Medium | 8768 |
| **Ollama** | ~1.6 GB (GPU 1) | Low | 11435 |
| **STT Services** | ~8.5 GB total | High | 50051, 50052 |

**Total System**: ~15.2 GB VRAM / 41 GB (37% usage)

---

## 🎯 Production Deployment

All pipelines are deployed via Docker Compose:

```bash
# Start all pipelines
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f pipeline           # Konkani
docker compose logs -f pipeline-english   # English
docker compose logs -f pipeline-hindi     # Hindi
```

**Access Points**:
- Web UI: http://localhost:7777/realtime_multi.html
- Konkani: ws://localhost:8765
- English: ws://localhost:8767
- Hindi: ws://localhost:8768

---

## 🐛 Error Handling Summary

### Pipeline-Wide Error Handling

| Component | Error Type | Fallback Action |
|-----------|-----------|-----------------|
| **WebSocket** | Connection closed | Graceful shutdown, save session |
| **VAD** | Processing error | Skip frame, continue |
| **STT** | HTTP error | Return empty string, notify UI |
| **STT** | Timeout (30s) | Return empty string, notify UI |
| **LLM** | Ollama error | Return fallback message |
| **LLM** | Network error | Return fallback message |
| **TTS** | Synthesis error | Skip sentence, continue |
| **TTS** | Barge-in | Cancel current synthesis |

### User Notifications

All pipelines send JSON messages to the browser:

```javascript
{ type: "stt_empty" }           // STT returned nothing
{ type: "llm_error" }           // LLM failed
{ type: "error", message: "..." } // Generic error
{ type: "speech_too_short" }    // Audio < 0.3s
```

---

## 📝 Configuration Files

### Docker Compose Entries

**Konkani**:
```yaml
pipeline:
  image: konkani-pipeline:latest
  ports: ["8765:8765"]
  command: ["bash", "/app/start_pipeline.sh"]
  environment:
    - STT_SERVICE_URL=http://stt-service:50051
    - MULTILINGUAL_STT_URL=http://multilingual-stt:50052
```

**English**:
```yaml
pipeline-english:
  image: konkani-pipeline-english:latest
  ports: ["8767:8767"]
  command: ["bash", "/app/start_pipeline_english.sh"]
  environment:
    - MULTILINGUAL_STT_URL=http://multilingual-stt:50052
    - WS_PORT=8767
```

**Hindi**:
```yaml
pipeline-hindi:
  image: konkani-pipeline-hindi:latest
  ports: ["8768:8768"]
  command: ["bash", "/app/start_pipeline_hindi.sh"]
  environment:
    - MULTILINGUAL_STT_URL=http://multilingual-stt:50052
    - WS_PORT=8768
```

---

## 🎨 Web UI Integration

The `realtime_multi.html` UI connects to all three:

```javascript
const TABS = {
  kok: { label: 'Konkani', port: 8765 },
  en:  { label: 'English', port: 8767 },
  hi:  { label: 'Hindi',   port: 8768 },
};

function switchTab(lang) {
  // Each tab connects to its dedicated WebSocket endpoint
  const wsUrl = `ws://${location.hostname}:${TABS[lang].port}`;
  // ...
}
```

---

## ✅ Summary

### Files Per Pipeline

**Konkani**:
- `ws_pipeline.py` (542 lines)
- `start_pipeline.sh` (55 lines)
- Shared: Dockerfile.pipeline, docker-compose.yml

**English**:
- `ws_pipeline_english.py` (540 lines)
- `start_pipeline_english.sh` (8 lines)
- Shared: Dockerfile.pipeline, docker-compose.yml

**Hindi**:
- `ws_pipeline_hindi.py` (540 lines)
- `start_pipeline_hindi.sh` (8 lines)
- Shared: Dockerfile.pipeline, docker-compose.yml

### Unique Features

| Pipeline | Unique Feature |
|----------|---------------|
| **Konkani** | Dual STT mode (Konkani/Multilingual switching) |
| **English** | TTS-friendly prompts (avoid contractions) |
| **Hindi** | TTS-friendly prompts (avoid symbols) |

### Common Fallbacks

✅ **STT**: Return empty string on error  
✅ **LLM**: Return pre-defined error message  
✅ **TTS**: Skip failed sentences, continue  
✅ **Network**: 30-second timeout with retry  
✅ **Sessions**: Auto-save on disconnect
