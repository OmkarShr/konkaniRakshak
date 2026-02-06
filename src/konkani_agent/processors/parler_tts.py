"""AI4Bharat Indic-Parler-TTS Processor for Pipecat

Text-to-Speech using AI4Bharat's Indic-Parler-TTS model.
Optimized for Indic languages including Konkani (via Marathi).
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class ParlerTTSProcessor(FrameProcessor):
    """
    AI4Bharat Indic-Parler-TTS processor with sentence buffering.

    Uses Hugging Face Transformers for cleaner dependency management.
    Supports both male and female voices via configuration.
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/indic-parler-tts",
        voice: str = "female",  # "female" or "male"
        device: str = "cuda",
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.model_name = model_name
        self.voice = voice
        self.device = device
        self.sample_rate = sample_rate

        self.model = None
        self.tokenizer = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.text_buffer = ""
        self.tts_available = False

        # Devanagari sentence endings
        self.sentence_endings = ["।", ".", "?", "!", "\n"]

        logger.info(f"ParlerTTSProcessor initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Voice: {voice}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Sample rate: {sample_rate}Hz")

    async def start(self, frame):
        """Load TTS model."""
        await super().start(frame)
        await self._load_model()

    async def _load_model(self):
        """Load AI4Bharat Parler-TTS model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info("Loading AI4Bharat Parler-TTS model...")
            logger.info(f"  Using Hugging Face model: {self.model_name}")

            # Use HF_TOKEN if available
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                from huggingface_hub import login

                login(token=hf_token)
                logger.info("✓ Hugging Face token authenticated")

            loop = asyncio.get_event_loop()

            # Load model and tokenizer in thread pool
            self.tokenizer = await loop.run_in_executor(
                self.executor, lambda: AutoTokenizer.from_pretrained(self.model_name)
            )

            self.model = await loop.run_in_executor(
                self.executor,
                lambda: AutoModel.from_pretrained(self.model_name).to(self.device),
            )

            self.model.eval()
            self.tts_available = True

            logger.info(f"✓ Parler-TTS loaded successfully!")
            logger.info(f"  Model size: ~600MB")
            logger.info(f"  Ready for {self.voice} voice synthesis")

        except Exception as e:
            logger.error(f"Failed to load Parler-TTS: {e}")
            logger.warning("TTS will be disabled - text responses only")
            self.tts_available = False

    async def process_frame(self, frame, direction):
        """Process text frames and synthesize audio."""

        if isinstance(frame, TextFrame):
            if self.tts_available:
                await self._process_text(frame.text)
            else:
                # TTS not available, just log the text
                logger.info(f"LLM Response: {frame.text[:80]}...")

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
        if not self.tts_available or not self.model:
            return

        try:
            logger.info(f"TTS synthesizing: {text[:50]}...")

            loop = asyncio.get_event_loop()

            # Run TTS in thread pool
            audio = await loop.run_in_executor(
                self.executor, self._synthesize_sync, text
            )

            if audio is not None and len(audio) > 0:
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

                logger.info(
                    f"TTS emitted {len(audio)} samples ({len(audio) / self.sample_rate:.2f}s)"
                )

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous TTS (runs in thread pool)."""
        try:
            import torch

            # Prepare text with voice description
            voice_desc = f"A {self.voice} speaker speaks in Marathi"

            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(
                self.device
            )

            # Generate speech
            with torch.no_grad():
                output = self.model.generate(**inputs)

            # Extract audio (model specific - adjust as needed)
            if hasattr(output, "waveform"):
                audio = output.waveform.cpu().numpy().squeeze()
            elif isinstance(output, torch.Tensor):
                audio = output.cpu().numpy().squeeze()
            else:
                # Fallback - try to extract from output
                audio = np.array(output)

            # Normalize to [-1, 1]
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.abs(audio).max()

            return audio

        except Exception as e:
            logger.error(f"TTS sync error: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()

        if self.executor:
            self.executor.shutdown(wait=True)

        if self.model is not None:
            import torch

            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()

        logger.info("ParlerTTSProcessor cleaned up")
