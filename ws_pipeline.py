"""
Konkani Voice Agent -- Real-time WebSocket Pipeline Server
==========================================================

Runs the full Pipecat-style pipeline over WebSocket so that a browser
client can stream audio in and receive TTS audio back in real time.

Pipeline:
  Browser mic  -->  [WebSocket]  -->  SileroVAD  -->  STTClient(HTTP)
       -->  Gemini LLM  -->  Indic Parler-TTS  -->  [WebSocket]  -->  Browser speaker

This file runs INSIDE the Docker pipeline container.
"""

import asyncio
import json
import os
import sys
import time
import base64
import struct
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import aiohttp
import websockets
from loguru import logger

# ── Logging ────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="{time:HH:mm:ss} | {level:<8} | {message}",
    level="INFO",
)

# ── Configuration ──────────────────────────────────────────────────
WS_HOST = "0.0.0.0"
WS_PORT = int(os.getenv("WS_PORT", "8765"))
STT_URL = os.getenv("STT_SERVICE_URL", "http://konkani-stt-1:50051")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

AUDIO_IN_RATE = 16000       # from browser mic
AUDIO_OUT_RATE = 44100      # from Indic Parler-TTS
CHUNK_MS = 20               # 20 ms frames from browser
IN_CHUNK = AUDIO_IN_RATE * CHUNK_MS // 1000   # 320 samples
OUT_CHUNK = AUDIO_OUT_RATE * CHUNK_MS // 1000  # 882 samples
VAD_CHUNK = 512             # Silero VAD minimum: 512 samples at 16kHz (32ms)

VAD_THRESHOLD = 0.30
MIN_SPEECH_MS = 200         # reduced from 300 to avoid missing initial words
MIN_SILENCE_MS = 600
MAX_RECORD_S = 20

PRE_BUFFER_MS = 500         # keep last 500ms of audio before speech confirmation
PRE_BUFFER_BYTES = int(AUDIO_IN_RATE * 2 * PRE_BUFFER_MS / 1000)  # 16000 bytes

TTS_SPEAKER_DESC = "Sanjay speaks with a moderate pace and a clear, close-sounding recording with no background noise."

SYSTEM_PROMPT = (
    "तूं एक कोंकणी भाशेंतलो (देवनागरी लिपींत) सहाय्यक आसा. "
    "तूं गोंय पुलिसांखातीर एफआयआर दाखल करपाक मदत करता. "
    "सदांच कोंकणी भाशेंत जाप दी. जापो मटव्यो आनी स्पश्ट आसच्यो. "
    "इंग्लीश वापरूं नाका."
)


# ═══════════════════════════════════════════════════════════════════
#  Global model holders (loaded once at startup)
# ═══════════════════════════════════════════════════════════════════
vad_model = None
tts_model = None
tts_tokenizer = None
tts_desc_tokenizer = None
gemini_model = None
tts_executor = ThreadPoolExecutor(max_workers=1)


def load_models():
    """Load all heavy models once at process start."""
    global vad_model, tts_model, tts_tokenizer, tts_desc_tokenizer, gemini_model

    # ── 1. Silero VAD ──
    import torch
    logger.info("Loading Silero VAD ...")
    vad_model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False
    )
    logger.info("  VAD loaded")

    # ── 2. Indic Parler-TTS ──
    logger.info("Loading Indic Parler-TTS (GPU) ...")
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts"
    ).to("cuda:0")
    tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    tts_desc_tokenizer = AutoTokenizer.from_pretrained(
        tts_model.config.text_encoder._name_or_path
    )
    # Warmup
    logger.info("  Warming up TTS ...")
    _warmup_desc = tts_desc_tokenizer(TTS_SPEAKER_DESC, return_tensors="pt").to("cuda:0")
    _warmup_text = tts_tokenizer("नमस्कार", return_tensors="pt").to("cuda:0")
    _ = tts_model.generate(
        input_ids=_warmup_desc.input_ids,
        attention_mask=_warmup_desc.attention_mask,
        prompt_input_ids=_warmup_text.input_ids,
        prompt_attention_mask=_warmup_text.attention_mask,
    )
    logger.info("  TTS loaded + warmed up")

    # ── 3. Gemini ──
    logger.info("Loading Gemini ...")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    logger.info("  Gemini ready")


# ═══════════════════════════════════════════════════════════════════
#  Per-session pipeline
# ═══════════════════════════════════════════════════════════════════

class PipelineSession:
    """One WebSocket client = one conversation session."""

    def __init__(self, ws):
        self.ws = ws
        self.conversation: list[dict] = []
        self.audio_buffer = bytearray()        # raw PCM s16le accumulated during speech
        self.vad_buffer = bytearray()           # buffer for accumulating VAD-sized chunks
        self.pre_buffer = bytearray()           # rolling buffer of last ~500ms before speech confirmed
        self.is_speaking = False
        self.speech_start: Optional[float] = None
        self.silence_start: Optional[float] = None
        self.agent_speaking = False             # True while TTS audio is being sent
        self._cancel_tts = False
        self._tts_task: Optional[asyncio.Task] = None

    # ── Send helpers ───────────────────────────────────────────────
    async def send_json(self, msg: dict):
        try:
            await self.ws.send(json.dumps(msg))
        except Exception:
            pass

    async def send_audio(self, pcm_bytes: bytes):
        """Send raw PCM s16le audio bytes to browser."""
        try:
            await self.ws.send(pcm_bytes)
        except Exception:
            pass

    # ── VAD ────────────────────────────────────────────────────────
    def run_vad(self, pcm_s16: bytes) -> float:
        """Return speech probability for a chunk."""
        import torch
        audio = np.frombuffer(pcm_s16, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)
        with torch.no_grad():
            prob = vad_model(tensor, AUDIO_IN_RATE).item()
        return prob

    # ── Main receive loop ──────────────────────────────────────────
    async def run(self):
        logger.info(f"Session started: {self.ws.remote_address}")
        # Reset VAD model state for this session
        vad_model.reset_states()
        await self.send_json({"type": "ready"})

        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    await self._handle_audio(message)
                elif isinstance(message, str):
                    data = json.loads(message)
                    await self._handle_control(data)
        except websockets.ConnectionClosed:
            pass
        finally:
            logger.info(f"Session ended: {self.ws.remote_address}")

    async def _handle_control(self, data: dict):
        cmd = data.get("type", "")
        if cmd == "reset":
            self.conversation.clear()
            await self.send_json({"type": "reset_ok"})
        elif cmd == "ping":
            await self.send_json({"type": "pong"})

    async def _handle_audio(self, pcm_data: bytes):
        """Process incoming 16kHz s16le PCM audio from browser."""
        # Accumulate into VAD buffer, process in VAD_CHUNK-sized frames
        self.vad_buffer.extend(pcm_data)
        vad_frame_bytes = VAD_CHUNK * 2  # 512 samples * 2 bytes = 1024 bytes

        while len(self.vad_buffer) >= vad_frame_bytes:
            frame = bytes(self.vad_buffer[:vad_frame_bytes])
            self.vad_buffer = self.vad_buffer[vad_frame_bytes:]
            await self._process_vad_frame(frame)

    async def _process_vad_frame(self, pcm_frame: bytes):
        """Run VAD on a single frame and manage speech state."""
        now = time.time()
        prob = self.run_vad(pcm_frame)

        # Always maintain rolling pre-buffer (last ~500ms of audio)
        self.pre_buffer.extend(pcm_frame)
        if len(self.pre_buffer) > PRE_BUFFER_BYTES:
            self.pre_buffer = self.pre_buffer[-PRE_BUFFER_BYTES:]

        # ── Barge-in: user speaks while agent is talking ──
        if self.agent_speaking and prob > VAD_THRESHOLD:
            if self.speech_start is None:
                self.speech_start = now
            elif (now - self.speech_start) * 1000 >= MIN_SPEECH_MS * 0.6:
                # Interrupt!
                logger.info("BARGE-IN detected -- stopping TTS")
                self._cancel_tts = True
                self.agent_speaking = False
                if self._tts_task and not self._tts_task.done():
                    self._tts_task.cancel()
                await self.send_json({"type": "interrupted"})
                # Reset state so we start recording the new utterance
                self.audio_buffer = bytearray(pcm_frame)
                self.is_speaking = True
                self.speech_start = now
                self.silence_start = None
                return

        # ── Normal VAD: detect speech start/stop ──
        if prob > VAD_THRESHOLD:
            if not self.is_speaking:
                if self.speech_start is None:
                    self.speech_start = now

                if (now - self.speech_start) * 1000 >= MIN_SPEECH_MS:
                    self.is_speaking = True
                    self.silence_start = None
                    # Prepend pre-buffer so we don't lose the initial words
                    self.audio_buffer = bytearray(self.pre_buffer)
                    await self.send_json({"type": "speech_start"})
                    logger.info("Speech started")

            if self.is_speaking:
                self.audio_buffer.extend(pcm_frame)
                # Safety: don't record forever
                if len(self.audio_buffer) > AUDIO_IN_RATE * 2 * MAX_RECORD_S:
                    await self._on_speech_end()

            self.silence_start = None

        else:
            # Silence
            self.speech_start = None if not self.is_speaking else self.speech_start

            if self.is_speaking:
                # Still accumulate during short silences
                self.audio_buffer.extend(pcm_frame)

                if self.silence_start is None:
                    self.silence_start = now
                elif (now - self.silence_start) * 1000 >= MIN_SILENCE_MS:
                    await self._on_speech_end()
            else:
                self.speech_start = None

    async def _on_speech_end(self):
        """Speech ended -- run STT -> LLM -> TTS."""
        self.is_speaking = False
        self.speech_start = None
        self.silence_start = None

        audio_bytes = bytes(self.audio_buffer)
        self.audio_buffer = bytearray()

        if len(audio_bytes) < AUDIO_IN_RATE * 2 * 0.3:  # less than 0.3s
            logger.info("Speech too short, ignoring")
            await self.send_json({"type": "speech_too_short"})
            return

        await self.send_json({"type": "processing"})
        logger.info(f"Speech ended: {len(audio_bytes)} bytes ({len(audio_bytes)/(AUDIO_IN_RATE*2):.1f}s)")

        # Run the pipeline
        t_start = time.time()
        try:
            # ── STT ──
            transcription = await self._run_stt(audio_bytes)
            t_stt = time.time()
            if not transcription:
                await self.send_json({"type": "stt_empty"})
                return

            await self.send_json({"type": "transcription", "text": transcription})
            logger.info(f"STT ({t_stt - t_start:.1f}s): {transcription}")

            # ── LLM ──
            logger.info("Calling LLM...")
            response_text = await self._run_llm(transcription)
            t_llm = time.time()
            logger.info(f"LLM returned: {bool(response_text)}")
            if not response_text:
                await self.send_json({"type": "llm_error"})
                return

            await self.send_json({"type": "response_text", "text": response_text})
            logger.info(f"LLM ({t_llm - t_stt:.1f}s): {response_text[:80]}...")

            # ── TTS (stream audio back) ──
            self._cancel_tts = False
            self.agent_speaking = True
            await self.send_json({"type": "tts_start"})

            self._tts_task = asyncio.create_task(self._run_tts_streaming(response_text))
            await self._tts_task

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled (barge-in)")
        except Exception as e:
            logger.error(f"Pipeline error: {e}\n{traceback.format_exc()}")
            await self.send_json({"type": "error", "message": str(e)})
        finally:
            self.agent_speaking = False
            await self.send_json({"type": "turn_done"})

    # ── STT via HTTP ───────────────────────────────────────────────
    async def _run_stt(self, pcm_bytes: bytes) -> str:
        """Send PCM audio to STT service, return text."""
        audio_b64 = base64.b64encode(pcm_bytes).decode()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{STT_URL}/transcribe",
                json={"audio": audio_b64, "sample_rate": AUDIO_IN_RATE},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return (data.get("text") or "").strip()
                else:
                    logger.error(f"STT error {resp.status}")
                    return ""

    # ── LLM via Gemini ─────────────────────────────────────────────
    async def _run_llm(self, user_text: str) -> str:
        self.conversation.append({"role": "user", "content": user_text})

        messages = []
        messages.append({"role": "user", "parts": [SYSTEM_PROMPT]})
        messages.append({"role": "model", "parts": ["व्हय, हांव समजलों. हांव फकत कोंकणी भाशेंत उलयतलों."]})

        for msg in self.conversation[-10:]:
            role = "user" if msg["role"] == "user" else "model"
            messages.append({"role": role, "parts": [msg["content"]]})

        loop = asyncio.get_event_loop()

        # Retry with exponential backoff for 429 rate-limit errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: gemini_model.generate_content(
                            messages,
                            generation_config={"temperature": 0.7, "max_output_tokens": 300},
                        ),
                    ),
                    timeout=30.0,
                )

                text = response.text.strip()
                self.conversation.append({"role": "assistant", "content": text})
                return text

            except asyncio.TimeoutError:
                logger.error("LLM call timed out after 30s")
                return ""
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = 2 ** attempt + 1
                    logger.warning(f"LLM rate-limited (429), retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"LLM error: {e}\n{traceback.format_exc()}")
                return ""

        logger.error("LLM: all retries exhausted (429)")
        return ""

    # ── TTS (streaming chunks) ─────────────────────────────────────
    async def _run_tts_streaming(self, text: str):
        """Synthesize text and stream PCM chunks to browser."""
        loop = asyncio.get_event_loop()

        # Split into sentences for lower latency (send first sentence ASAP)
        sentences = self._split_sentences(text)
        logger.info(f"TTS: synthesizing {len(sentences)} sentence(s)")

        for idx, sentence in enumerate(sentences):
            if self._cancel_tts:
                break
            if not sentence.strip():
                continue

            logger.info(f"TTS: generating sentence {idx+1}/{len(sentences)}: {sentence[:50]}...")

            # Synthesize on executor (blocking call)
            def _synthesize(s=sentence):
                desc_ids = tts_desc_tokenizer(
                    TTS_SPEAKER_DESC, return_tensors="pt"
                ).to("cuda:0")
                prompt_ids = tts_tokenizer(s, return_tensors="pt").to("cuda:0")
                gen = tts_model.generate(
                    input_ids=desc_ids.input_ids,
                    attention_mask=desc_ids.attention_mask,
                    prompt_input_ids=prompt_ids.input_ids,
                    prompt_attention_mask=prompt_ids.attention_mask,
                )
                return gen.cpu().numpy().squeeze()

            try:
                wav = await loop.run_in_executor(tts_executor, _synthesize)
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}\n{traceback.format_exc()}")
                continue

            if self._cancel_tts:
                break

            if wav is None or len(wav) == 0:
                logger.warning("TTS: empty audio returned")
                continue

            audio = np.array(wav)
            # Convert float32 [-1,1] to int16
            pcm_s16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            logger.info(f"TTS: sentence {idx+1} done, {len(pcm_s16)} samples ({len(pcm_s16)/AUDIO_OUT_RATE:.1f}s)")

            # Stream in chunks
            chunk_samples = OUT_CHUNK * 4   # ~80ms chunks for network efficiency
            for i in range(0, len(pcm_s16), chunk_samples):
                if self._cancel_tts:
                    break
                chunk = pcm_s16[i : i + chunk_samples].tobytes()
                await self.send_audio(chunk)
                # Pace sending so browser can play smoothly
                await asyncio.sleep(len(chunk) / (2 * AUDIO_OUT_RATE) * 0.8)

        self.agent_speaking = False
        await self.send_json({"type": "tts_done"})
        logger.info("TTS: all done")

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split Devanagari text at sentence boundaries."""
        import re
        parts = re.split(r'(?<=[।.?!\n])\s*', text)
        return [p for p in parts if p.strip()]


# ═══════════════════════════════════════════════════════════════════
#  WebSocket server
# ═══════════════════════════════════════════════════════════════════

async def handle_client(ws):
    session = PipelineSession(ws)
    await session.run()


async def main():
    logger.info("=" * 60)
    logger.info("  Konkani Voice Agent -- Real-time Pipeline Server")
    logger.info("=" * 60)

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set!")
        sys.exit(1)

    # Load all models
    load_models()

    logger.info("")
    logger.info(f"  WebSocket server: ws://{WS_HOST}:{WS_PORT}")
    logger.info(f"  STT service:      {STT_URL}")
    logger.info(f"  Gemini model:     {GEMINI_MODEL}")
    logger.info(f"  TTS voice:        Indic Parler-TTS (Sanjay)")
    logger.info("")
    logger.info("  Waiting for browser connections ...")
    logger.info("=" * 60)

    async with websockets.serve(
        handle_client,
        WS_HOST,
        WS_PORT,
        max_size=10 * 1024 * 1024,  # 10MB max message
        ping_interval=30,
        ping_timeout=10,
    ):
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    asyncio.run(main())
