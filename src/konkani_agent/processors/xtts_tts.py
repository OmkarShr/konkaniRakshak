"""XTTSv2 TTS Processor for Pipecat

Text-to-Speech using Coqui XTTSv2 with Marathi language support.
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class XTTSv2Processor(FrameProcessor):
    """
    XTTSv2 TTS processor with sentence buffering.

    Since XTTSv2 is batch-based, we buffer text until sentence boundaries
    and then synthesize complete sentences for better quality.
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        language: str = "mr",  # Marathi
        device: str = "cuda",
        fp16: bool = True,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.model_name = model_name
        self.language = language
        self.device = device
        self.fp16 = fp16
        self.sample_rate = sample_rate

        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.text_buffer = ""
        self.tts_available = False

        # Devanagari sentence endings
        self.sentence_endings = ["।", ".", "?", "!", "\n"]

        logger.info(f"XTTSv2Processor initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Language: {language}")
        logger.info(f"  Device: {device}")
        logger.info(f"  FP16: {fp16}")

    async def start(self, frame):
        """Load TTS model."""
        await super().start(frame)
        await self._load_model()

    async def _load_model(self):
        """Load XTTSv2 model."""
        try:
            from TTS.api import TTS

            logger.info("Loading XTTSv2 model...")

            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(self.executor, TTS, self.model_name)

            self.tts_available = True
            logger.info(f"✓ XTTSv2 loaded")

        except Exception as e:
            logger.error(f"Failed to load XTTSv2: {e}")
            logger.warning("TTS will be disabled - text responses only")
            self.tts_available = False

    async def process_frame(self, frame, direction):
        """Process text frames and synthesize audio."""

        if isinstance(frame, TextFrame):
            if self.tts_available:
                await self._process_text(frame.text)
            else:
                # TTS not available, just log the text
                logger.info(f"LLM Response (no TTS): {frame.text[:50]}...")

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Synthesize any remaining text
            if self.tts_available and self.text_buffer.strip():
                await self._synthesize_and_emit(self.text_buffer)
                self.text_buffer = ""

        # Pass frame downstream
        await self.push_frame(frame, direction)

    async def _process_text(self, text: str):
        """Buffer text and synthesize at sentence boundaries."""
        self.text_buffer += text

        # Check for sentence boundaries
        for ending in self.sentence_endings:
            if ending in self.text_buffer:
                # Split on ending
                parts = self.text_buffer.split(ending, 1)
                if len(parts) == 2:
                    sentence = parts[0] + ending
                    self.text_buffer = parts[1]

                    # Synthesize complete sentence
                    if sentence.strip():
                        await self._synthesize_and_emit(sentence)
                break

    async def _synthesize_and_emit(self, text: str):
        """Synthesize text and emit audio frames."""
        if not self.tts_available:
            return

        try:
            logger.info(f"TTS synthesizing: {text[:50]}...")

            loop = asyncio.get_event_loop()

            # Run TTS in thread pool
            audio = await loop.run_in_executor(
                self.executor, self._synthesize_sync, text
            )

            if audio is not None:
                # Chunk audio into frames (20ms chunks)
                chunk_size = int(self.sample_rate * 0.02)  # 20ms

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i : i + chunk_size]

                    # Pad last chunk if needed
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                    # Convert to int16 bytes
                    audio_bytes = (chunk * 32767).astype(np.int16).tobytes()

                    # Create and emit frame
                    audio_frame = AudioRawFrame(
                        audio=audio_bytes, sample_rate=self.sample_rate, num_channels=1
                    )
                    await self.push_frame(audio_frame)

                logger.info(f"TTS emitted {len(audio)} samples")

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous TTS (runs in thread pool)."""
        try:
            # Generate speech
            wav = self.model.tts(
                text=text,
                speaker_wav=None,  # Use default speaker
                language=self.language,
            )

            return np.array(wav)

        except Exception as e:
            logger.error(f"TTS sync error: {e}")
            return None

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()

        if self.executor:
            self.executor.shutdown(wait=True)

        if self.model is not None:
            import torch

            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()

        logger.info("XTTSv2Processor cleaned up")
