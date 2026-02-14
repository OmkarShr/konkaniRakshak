"""
Konkani Voice Agent -- English-Only Pipeline Server
====================================================

Runs a WebSocket pipeline for English using:
  - Voxtral STT (via vLLM Realtime WebSocket API on GPU 1)
  - Ollama LLM (gemma2:2b with TTS-friendly prompts)
  - Indic Parler-TTS (shared model on GPU 0)

This file runs INSIDE a Docker pipeline container on GPU 0.
Dedicated English-only pipeline on port 8767.
"""

import asyncio
import json
import os
import sys
import io
import time
import base64
import struct
import tempfile
import traceback
import wave
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import aiohttp
import websockets
from loguru import logger

# -- Logging --
logger.remove()
logger.add(
    sys.stderr,
    format="{time:HH:mm:ss} | {level:<8} | {message}",
    level="INFO",
)

# -- Configuration --
WS_HOST = "0.0.0.0"
WS_PORT = int(os.getenv("WS_PORT", "8767"))
STT_URL = os.getenv("MULTILINGUAL_STT_URL", "http://multilingual-stt:50052")


AUDIO_IN_RATE = 16000       # from browser mic
AUDIO_OUT_RATE = 44100      # from Indic Parler-TTS
CHUNK_MS = 20               # 20 ms frames from browser
IN_CHUNK = AUDIO_IN_RATE * CHUNK_MS // 1000   # 320 samples
OUT_CHUNK = AUDIO_OUT_RATE * CHUNK_MS // 1000  # 882 samples
VAD_CHUNK = 512             # Silero VAD minimum: 512 samples at 16kHz

VAD_THRESHOLD = 0.50
MIN_SPEECH_MS = 200
MIN_SILENCE_MS = 1200       # Increased to 1.2s to prevent premature cutoff
MAX_RECORD_S = 30           # Increased to 30s

PRE_BUFFER_MS = 500
PRE_BUFFER_BYTES = int(AUDIO_IN_RATE * 2 * PRE_BUFFER_MS / 1000)

FALLBACK_RESPONSE = "Sorry, I am having trouble connecting to my brain right now. Please try again."

# English-only configuration with TTS-friendly prompt
SYSTEM_PROMPT = (
    "You are an assistant for the Goa Police to help file FIRs (First Information Reports). "
    "Always respond in English. Keep responses short and clear. "
    "Ask for necessary details: complainant name, incident description, date, time, location. "
    "\n\nIMPORTANT: Do not use apostrophes or special symbols in your responses. "
    "Write words out fully instead of using contractions. For example, use 'cannot' instead of 'can't', "
    "'do not' instead of 'don't'. Avoid quotation marks, asterisks, and other symbols that may confuse text-to-speech."
)

TTS_SPEAKER_DESC = "Sanjay speaks with a moderate pace and a clear, close-sounding recording with no background noise."
TTS_LANG_HINT = "en"

# -- Global model holders (loaded once at startup) --
vad_model = None
tts_model = None
tts_tokenizer = None
tts_desc_tokenizer = None

tts_executor = ThreadPoolExecutor(max_workers=1)


def load_models():
    """Load all heavy models once at process start."""
    global vad_model, tts_model, tts_tokenizer, tts_desc_tokenizer

    # -- 1. Silero VAD --
    import torch
    logger.info("Loading Silero VAD ...")
    vad_model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False
    )
    logger.info("  VAD loaded")

    # -- 2. Indic Parler-TTS --
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
    _warmup_desc = tts_desc_tokenizer("Sanjay speaks clearly.", return_tensors="pt").to("cuda:0")
    _warmup_text = tts_tokenizer("Hello", return_tensors="pt").to("cuda:0")
    _ = tts_model.generate(
        input_ids=_warmup_desc.input_ids,
        attention_mask=_warmup_desc.attention_mask,
        prompt_input_ids=_warmup_text.input_ids,
        prompt_attention_mask=_warmup_text.attention_mask,
    )
    logger.info("  TTS loaded + warmed up")

    # -- 3. LLM (Ollama) --
    logger.info("LLM backend: Ollama (gemma2:2b)")
    # Warmup to force model load
    try:
        import ollama
        logger.info("  Warming up Ollama (forcing model load)...")
        ollama.Client(host=os.getenv("OLLAMA_URL")).chat(model="gemma2:2b", messages=[{"role": "user", "content": "hi"}])
        logger.info("  Ollama warmed up!")
    except Exception as e:
        logger.warning(f"  Ollama warmup failed: {e}")


# ==================================================================
#  Per-session pipeline
# ==================================================================

class PipelineSession:
    """One WebSocket client = one conversation session."""

    def __init__(self, ws):
        self.ws = ws
        self.conversation: list[dict] = []
        self.audio_buffer = bytearray()
        self.vad_buffer = bytearray()
        self.pre_buffer = bytearray()
        self.is_speaking = False
        self.speech_start: Optional[float] = None
        self.silence_start: Optional[float] = None
        self.agent_speaking = False
        self._cancel_tts = False
        self._tts_task: Optional[asyncio.Task] = None

    # -- Send helpers --
    async def send_json(self, msg: dict):
        try:
            await self.ws.send(json.dumps(msg))
        except Exception:
            pass

    async def send_audio(self, pcm_bytes: bytes):
        try:
            await self.ws.send(pcm_bytes)
        except Exception:
            pass

    # -- VAD --
    def run_vad(self, pcm_s16: bytes) -> float:
        import torch
        audio = np.frombuffer(pcm_s16, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)
        with torch.no_grad():
            prob = vad_model(tensor, AUDIO_IN_RATE).item()
        return prob

    # -- Main receive loop --
    async def run(self):
        logger.info(f"[English] Session started: {self.ws.remote_address}")
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
            logger.info(f"[English] Session ended: {self.ws.remote_address}")

    async def _handle_control(self, data: dict):
        cmd = data.get("type", "")
        if cmd == "reset":
            self.conversation.clear()
            await self.send_json({"type": "reset_ok"})
        elif cmd == "ping":
            await self.send_json({"type": "pong"})

    async def _handle_audio(self, pcm_data: bytes):
        self.vad_buffer.extend(pcm_data)
        vad_frame_bytes = VAD_CHUNK * 2

        while len(self.vad_buffer) >= vad_frame_bytes:
            frame = bytes(self.vad_buffer[:vad_frame_bytes])
            self.vad_buffer = self.vad_buffer[vad_frame_bytes:]
            await self._process_vad_frame(frame)

    async def _process_vad_frame(self, pcm_frame: bytes):
        now = time.time()
        prob = self.run_vad(pcm_frame)

        # Rolling pre-buffer
        self.pre_buffer.extend(pcm_frame)
        if len(self.pre_buffer) > PRE_BUFFER_BYTES:
            self.pre_buffer = self.pre_buffer[-PRE_BUFFER_BYTES:]

        # -- Barge-in --
        if self.agent_speaking and prob > VAD_THRESHOLD:
            if self.speech_start is None:
                self.speech_start = now
            elif (now - self.speech_start) * 1000 >= MIN_SPEECH_MS * 0.6:
                logger.info("[English] BARGE-IN detected")
                self._cancel_tts = True
                self.agent_speaking = False
                if self._tts_task and not self._tts_task.done():
                    self._tts_task.cancel()
                await self.send_json({"type": "interrupted"})
                self.audio_buffer = bytearray(pcm_frame)
                self.is_speaking = True
                self.speech_start = now
                self.silence_start = None
                return

        # -- Normal VAD --
        if prob > VAD_THRESHOLD:
            if not self.is_speaking:
                if self.speech_start is None:
                    self.speech_start = now
                if (now - self.speech_start) * 1000 >= MIN_SPEECH_MS:
                    self.is_speaking = True
                    self.silence_start = None
                    self.audio_buffer = bytearray(self.pre_buffer)
                    await self.send_json({"type": "speech_start"})
                    logger.info("[English] Speech started")

            if self.is_speaking:
                self.audio_buffer.extend(pcm_frame)
                if len(self.audio_buffer) > AUDIO_IN_RATE * 2 * MAX_RECORD_S:
                    await self._on_speech_end()

            self.silence_start = None
        else:
            self.speech_start = None if not self.is_speaking else self.speech_start
            if self.is_speaking:
                self.audio_buffer.extend(pcm_frame)
                if self.silence_start is None:
                    self.silence_start = now
                elif (now - self.silence_start) * 1000 >= MIN_SILENCE_MS:
                    await self._on_speech_end()
            else:
                self.speech_start = None

    async def _on_speech_end(self):
        self.is_speaking = False
        self.speech_start = None
        self.silence_start = None

        audio_bytes = bytes(self.audio_buffer)
        self.audio_buffer = bytearray()

        if len(audio_bytes) < AUDIO_IN_RATE * 2 * 0.3:
            logger.info("[English] Speech too short, ignoring")
            await self.send_json({"type": "speech_too_short"})
            return

        await self.send_json({"type": "processing"})
        logger.info(f"[English] Speech ended: {len(audio_bytes)} bytes ({len(audio_bytes)/(AUDIO_IN_RATE*2):.1f}s)")

        t_start = time.time()
        try:
            # -- STT via Voxtral (vLLM transcription API) --
            transcription = await self._run_stt_voxtral(audio_bytes)
            t_stt = time.time()
            if not transcription:
                await self.send_json({"type": "stt_empty"})
                return

            await self.send_json({"type": "transcription", "text": transcription})
            logger.info(f"[English] STT ({t_stt - t_start:.1f}s): {transcription}")

            # -- LLM --
            logger.info("[English] Calling LLM...")
            response_text = await self._run_llm(transcription)
            t_llm = time.time()
            if not response_text:
                await self.send_json({"type": "llm_error"})
                return

            await self.send_json({"type": "response_text", "text": response_text})
            logger.info(f"[English] LLM ({t_llm - t_stt:.1f}s): {response_text[:80]}...")

            # -- TTS --
            self._cancel_tts = False
            self.agent_speaking = True
            await self.send_json({"type": "tts_start"})

            self._tts_task = asyncio.create_task(self._run_tts_streaming(response_text))
            await self._tts_task

        except asyncio.CancelledError:
            logger.info("[English] Pipeline cancelled (barge-in)")
        except Exception as e:
            logger.error(f"[English] Pipeline error: {e}\n{traceback.format_exc()}")
            await self.send_json({"type": "error", "message": str(e)})
        finally:
            self.agent_speaking = False
            await self.send_json({"type": "turn_done"})

    # -- STT via IndicConformer Multilingual (HTTP API) --
    async def _run_stt_voxtral(self, pcm_bytes: bytes) -> str:
        """Send audio to IndicConformer multilingual STT service."""
        audio_b64 = base64.b64encode(pcm_bytes).decode()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{STT_URL}/transcribe",
                    json={"audio": audio_b64, "sample_rate": AUDIO_IN_RATE, "language": "en"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return (data.get("text") or "").strip()
                    else:
                        body = await resp.text()
                        logger.error(f"[English] STT error {resp.status}: {body[:200]}")
                        return ""
        except Exception as e:
            logger.error(f"[English] STT exception: {e}")
            return ""

    # ── LLM via Ollama ─────────────────────────────────────────────
    async def _run_llm(self, user_text: str) -> str:
        self.conversation.append({"role": "user", "content": user_text})

        # Convert conversation history to Ollama format
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        for msg in self.conversation[-6:]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

        try:
            import ollama
            
            # Use synchronous client in executor to avoid blocking asyncio loop
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
            self.conversation.append({"role": "assistant", "content": text})
            return text

        except Exception as e:
            logger.error(f"[English] LLM error: {e}\n{traceback.format_exc()}")
            return FALLBACK_RESPONSE
            
        if not text:
             return FALLBACK_RESPONSE
        return text

    # -- TTS (streaming chunks) --
    async def _run_tts_streaming(self, text: str):
        loop = asyncio.get_event_loop()

        sentences = self._split_sentences(text)
        logger.info(f"[English] TTS: synthesizing {len(sentences)} sentence(s)")

        for idx, sentence in enumerate(sentences):
            if self._cancel_tts:
                break
            if not sentence.strip():
                continue

            logger.info(f"[English] TTS: sentence {idx+1}/{len(sentences)}: {sentence[:50]}...")

            def _synthesize(s=sentence, desc=TTS_SPEAKER_DESC):
                desc_ids = tts_desc_tokenizer(desc, return_tensors="pt").to("cuda:0")
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
                logger.error(f"[English] TTS error: {e}\n{traceback.format_exc()}")
                continue

            if self._cancel_tts:
                break

            if wav is None or len(wav) == 0:
                logger.warning("[English] TTS: empty audio")
                continue

            audio = np.array(wav)
            pcm_s16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            logger.info(f"[English] TTS: sentence {idx+1} done, {len(pcm_s16)} samples ({len(pcm_s16)/AUDIO_OUT_RATE:.1f}s)")

            chunk_samples = OUT_CHUNK * 4
            for i in range(0, len(pcm_s16), chunk_samples):
                if self._cancel_tts:
                    break
                chunk = pcm_s16[i : i + chunk_samples].tobytes()
                await self.send_audio(chunk)
                await asyncio.sleep(len(chunk) / (2 * AUDIO_OUT_RATE) * 0.8)

        self.agent_speaking = False
        await self.send_json({"type": "tts_done"})
        logger.info("[English] TTS: all done")

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text at sentence boundaries (supports both Devanagari and Latin punctuation)."""
        import re
        parts = re.split(r'(?<=[।.?!\n])\s*', text)
        return [p for p in parts if p.strip()]


# ==================================================================
#  WebSocket server
# ==================================================================

# -- Global Session Persistence --
GLOBAL_SESSIONS = {}  # {client_ip: conversation_history_list}

async def handle_client(ws):
    # Use remote address (IP) as reliability key
    client_ip = ws.remote_address[0] if ws.remote_address else "unknown"
    logger.info(f"[English] New connection from {client_ip}")

    session = PipelineSession(ws)
    
    # Restore history if exists
    if client_ip in GLOBAL_SESSIONS:
        logger.info(f"[English] Restoring session for {client_ip} ({len(GLOBAL_SESSIONS[client_ip])} msgs)")
        session.conversation = GLOBAL_SESSIONS[client_ip]
    
    try:
        await session.run()
    finally:
        # Save history on disconnect
        GLOBAL_SESSIONS[client_ip] = session.conversation
        logger.info(f"[English] Saved session for {client_ip}")


async def main():
    logger.info("=" * 60)
    logger.info("  English-Only Pipeline Server")
    logger.info("=" * 60)

    load_models()

    logger.info("")
    logger.info(f"  WebSocket server: ws://{WS_HOST}:{WS_PORT}")
    logger.info(f"  STT service:      {STT_URL}")
    logger.info(f"  LLM model:        gemma2:2b (TTS-friendly)")
    logger.info(f"  TTS voice:        Indic Parler-TTS")
    logger.info(f"  Language:         English only")
    logger.info("")
    logger.info("  Waiting for browser connections ...")
    logger.info("=" * 60)

    async with websockets.serve(
        handle_client,
        WS_HOST,
        WS_PORT,
        max_size=10 * 1024 * 1024,
        ping_interval=None,
        ping_timeout=None,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
