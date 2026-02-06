"""Enhanced TTS Processor with better Marathi support and error recovery

Text-to-Speech using multiple model options:
1. XTTSv2 (Marathi) - primary
2. Parler-TTS (Indic) - fallback
3. IndicTTS (if available) - secondary

Includes:
- Sentence buffering for better prosody
- Auto-fallback on errors
- Pre-recorded fallback audio for critical phrases
- Voice warm-up for faster first-time synthesis
"""

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import numpy as np
from loguru import logger
from pathlib import Path

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class EnhancedTTSProcessor(FrameProcessor):
    """
    Enhanced TTS with multiple backends, error recovery, and optimization.
    """

    def __init__(
        self,
        primary_model: str = "xtts",  # "xtts", "parler", "indic"
        fallback_model: Optional[str] = "parler",
        language: str = "mr",  # Marathi
        device: str = "cuda",
        fp16: bool = True,
        sample_rate: int = 24000,
        warmup: bool = True,
        enable_fallback_audio: bool = True,
    ):
        super().__init__()
        self.primary_model_name = primary_model
        self.fallback_model_name = fallback_model
        self.language = language
        self.device = device
        self.fp16 = fp16
        self.sample_rate = sample_rate
        self.warmup = warmup
        self.enable_fallback_audio = enable_fallback_audio

        # Models
        self.primary_model = None
        self.fallback_model = None
        self.executor = ThreadPoolExecutor(max_workers=1)

        # State
        self.text_buffer = ""
        self.tts_available = False
        self.current_model = "primary"
        self.fallback_phrases = {}  # Pre-recorded audio

        # Devanagari sentence endings
        self.sentence_endings = ["।", ".", "?", "!", "\n"]

        # Buffer for latency optimization
        self.audio_cache = {}
        self.max_cache_size = 100

        logger.info(f"EnhancedTTSProcessor initialized")
        logger.info(f"  Primary: {primary_model}")
        logger.info(f"  Fallback: {fallback_model}")
        logger.info(f"  Device: {device} (FP16: {fp16})")
        logger.info(f"  Sample rate: {sample_rate}Hz")

    async def start(self, frame):
        """Load TTS models."""
        await super().start(frame)

        # Load primary model
        await self._load_primary_model()

        # Load fallback if specified
        if self.fallback_model_name:
            await self._load_fallback_model()

        # Warmup synthesis
        if self.warmup and self.tts_available:
            await self._warmup_model()

        # Load fallback phrases
        if self.enable_fallback_audio:
            await self._load_fallback_phrases()

    async def _load_primary_model(self):
        """Load primary XTTSv2 model."""
        try:
            from TTS.api import TTS

            logger.info("Loading XTTSv2 model...")

            loop = asyncio.get_event_loop()
            self.primary_model = await loop.run_in_executor(
                self.executor, TTS, "tts_models/multilingual/multi-dataset/xtts_v2"
            )

            self.tts_available = True
            logger.info("✓ XTTSv2 loaded (primary)")

        except Exception as e:
            logger.error(f"✗ Failed to load XTTSv2: {e}")
            self.primary_model = None

    async def _load_fallback_model(self):
        """Load fallback Parler-TTS model."""
        try:
            from transformers import AutoTokenizer, AutoModel

            logger.info("Loading Parler-TTS (Indic) as fallback...")

            model_name = "ai4bharat/indic-parler-tts"

            loop = asyncio.get_event_loop()
            self.fallback_model = await loop.run_in_executor(
                self.executor,
                lambda: {
                    "tokenizer": AutoTokenizer.from_pretrained(model_name),
                    "model": AutoModel.from_pretrained(model_name),
                },
            )

            logger.info("✓ Parler-TTS loaded (fallback)")

        except Exception as e:
            logger.warning(f"⚠ Fallback model load failed: {e}")
            self.fallback_model = None

    async def _warmup_model(self):
        """Warmup synthesis to reduce first-time latency."""
        try:
            logger.info("Warming up TTS model...")

            warmup_text = "नमस्कार"
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                self.executor, self._synthesize_xtts_sync, warmup_text
            )

            logger.info("✓ TTS warmup complete")

        except Exception as e:
            logger.warning(f"⚠ TTS warmup failed: {e}")

    async def _load_fallback_phrases(self):
        """Load or generate pre-recorded fallback phrases."""
        fallback_dir = Path(__file__).parent.parent / "data" / "fallback_audio"
        fallback_dir.mkdir(parents=True, exist_ok=True)

        critical_phrases = {
            "error": "क्षमस्व, मला समजलं नाही. कृपया पुन्हा सांगा.",
            "loading": "एक मिनिट",
            "confirm": "ठीक आहे",
            "greeting": "नमस्कार, मी आपली मदत कशी करू शकतो?",
            "wait": "कृपया थांबा",
        }

        for key, phrase in critical_phrases.items():
            audio_path = fallback_dir / f"{key}.npy"

            if audio_path.exists():
                # Load existing
                self.fallback_phrases[key] = np.load(audio_path)
                logger.debug(f"Loaded fallback: {key}")
            elif self.tts_available:
                # Generate new
                try:
                    loop = asyncio.get_event_loop()
                    audio = await loop.run_in_executor(
                        self.executor, self._synthesize_xtts_sync, phrase
                    )

                    if audio is not None:
                        np.save(audio_path, audio)
                        self.fallback_phrases[key] = audio
                        logger.info(f"✓ Generated fallback: {key}")

                except Exception as e:
                    logger.warning(f"⚠ Failed to generate fallback {key}: {e}")

    async def process_frame(self, frame, direction):
        """Process text frames and synthesize audio."""

        if isinstance(frame, TextFrame):
            if self.tts_available:
                await self._process_text(frame.text)
            else:
                logger.warning(f"TTS unavailable - text: {frame.text[:50]}...")

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Synthesize any remaining text
            if self.tts_available and self.text_buffer.strip():
                await self._synthesize_and_emit(self.text_buffer, use_cache=True)
                self.text_buffer = ""

        # Pass frame downstream
        await self.push_frame(frame, direction)

    async def _process_text(self, text: str):
        """Buffer text and synthesize at sentence boundaries."""
        self.text_buffer += text

        # Check for sentence boundaries
        for ending in self.sentence_endings:
            if ending in self.text_buffer:
                parts = self.text_buffer.split(ending, 1)
                if len(parts) == 2:
                    sentence = parts[0] + ending
                    self.text_buffer = parts[1]

                    if sentence.strip():
                        await self._synthesize_and_emit(sentence, use_cache=True)
                break

    async def _synthesize_and_emit(self, text: str, use_cache: bool = True):
        """Synthesize text with error recovery and fallback."""
        if not self.tts_available:
            return

        # Check cache first
        cache_key = text.strip()
        if use_cache and cache_key in self.audio_cache:
            audio = self.audio_cache[cache_key]
            logger.debug(f"TTS cache hit: {text[:30]}...")
        else:
            # Synthesize
            audio = await self._synthesize_with_fallback(text)

            if audio is not None and use_cache:
                # Cache result
                if len(self.audio_cache) >= self.max_cache_size:
                    # Remove oldest entry
                    oldest = next(iter(self.audio_cache))
                    del self.audio_cache[oldest]
                self.audio_cache[cache_key] = audio

        if audio is not None:
            await self._emit_audio_frames(audio)
        else:
            # Try fallback phrase
            await self._emit_fallback_phrase("error")

    async def _synthesize_with_fallback(self, text: str) -> Optional[np.ndarray]:
        """Try primary model, then fallback on error."""
        # Try primary
        try:
            loop = asyncio.get_event_loop()
            audio = await asyncio.wait_for(
                loop.run_in_executor(self.executor, self._synthesize_xtts_sync, text),
                timeout=10.0,  # Max 10s synthesis time
            )

            if audio is not None and len(audio) > 0:
                return audio

        except asyncio.TimeoutError:
            logger.warning(f"TTS timeout for: {text[:50]}...")
        except Exception as e:
            logger.error(f"TTS error: {e}")

        # Try fallback if available
        if self.fallback_model is not None:
            logger.info(f"Trying fallback TTS for: {text[:50]}...")
            try:
                audio = await self._synthesize_parler_sync(text)
                if audio is not None:
                    return audio
            except Exception as e:
                logger.error(f"Fallback TTS failed: {e}")

        return None

    def _synthesize_xtts_sync(self, text: str) -> Optional[np.ndarray]:
        """Synchronous XTTSv2 synthesis."""
        try:
            if self.primary_model is None:
                return None

            wav = self.primary_model.tts(
                text=text,
                speaker_wav=None,
                language=self.language,
            )

            return np.array(wav)

        except Exception as e:
            logger.error(f"XTTSv2 synthesis error: {e}")
            return None

    async def _synthesize_parler_sync(self, text: str) -> Optional[np.ndarray]:
        """Synchronous Parler-TTS synthesis."""
        try:
            if self.fallback_model is None:
                return None

            # TODO: Implement Parler-TTS synthesis
            # For now, return None to trigger error fallback
            return None

        except Exception as e:
            logger.error(f"Parler-TTS synthesis error: {e}")
            return None

    async def _emit_audio_frames(self, audio: np.ndarray):
        """Chunk audio and emit frames."""
        try:
            chunk_size = int(self.sample_rate * 0.02)  # 20ms chunks

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]

                # Pad last chunk
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                # Convert to int16 bytes
                audio_bytes = (chunk * 32767).astype(np.int16).tobytes()

                # Create and emit frame
                audio_frame = AudioRawFrame(
                    audio=audio_bytes, sample_rate=self.sample_rate, num_channels=1
                )
                await self.push_frame(audio_frame)

            logger.debug(f"Emitted {len(audio)} samples")

        except Exception as e:
            logger.error(f"Audio emission error: {e}")

    async def _emit_fallback_phrase(self, phrase_key: str):
        """Emit pre-recorded fallback audio."""
        if phrase_key in self.fallback_phrases:
            audio = self.fallback_phrases[phrase_key]
            await self._emit_audio_frames(audio)
            logger.info(f"Emitted fallback phrase: {phrase_key}")

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()

        # Clear cache
        self.audio_cache.clear()
        self.fallback_phrases.clear()

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)

        # Clean up models
        if self.primary_model is not None:
            import torch

            del self.primary_model
            if self.device == "cuda":
                torch.cuda.empty_cache()

        if self.fallback_model is not None:
            import torch

            del self.fallback_model
            if self.device == "cuda":
                torch.cuda.empty_cache()

        logger.info("EnhancedTTSProcessor cleaned up")

    def get_memory_usage(self) -> dict:
        """Get current memory usage."""
        import torch

        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                "reserved": torch.cuda.memory_reserved() / 1024**2,
                "cached": torch.cuda.memory_cached() / 1024**2
                if hasattr(torch.cuda, "memory_cached")
                else 0,
            }
        return {"allocated": 0, "reserved": 0, "cached": 0}
