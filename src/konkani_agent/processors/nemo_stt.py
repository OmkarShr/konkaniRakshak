"""NeMo STT Processor for Pipecat

Custom FrameProcessor that wraps IndicConformer (NeMo) for Pipecat pipeline.
Handles the batch-to-streaming conversion.
"""

import asyncio
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    InputAudioRawFrame,
    TextFrame,
    UserStoppedSpeakingFrame,
    StartFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class NeMoSTTProcessor(FrameProcessor):
    """
    Custom processor for IndicConformer (NeMo) STT.

    NeMo is synchronous/batch, Pipecat is async/streaming.
    This wrapper handles the conversion.
    """

    def __init__(
        self,
        model_path: str,
        language: str = "kok",
        device: str = "cuda",
        fp16: bool = True,
    ):
        super().__init__()
        self.model_path = model_path
        self.language = language
        self.device = device
        self.fp16 = fp16

        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.audio_buffer = []
        self.is_processing = False

        logger.info(f"NeMoSTTProcessor initialized")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Language: {language}")
        logger.info(f"  Device: {device}")
        logger.info(f"  FP16: {fp16}")

    async def start(self, frame: StartFrame):
        """Load model when pipeline starts."""
        await super().start(frame)
        await self._load_model()

    async def _load_model(self):
        """Load IndicConformer model."""
        try:
            import nemo.collections.asr as nemo_asr

            logger.info("Loading IndicConformer model...")

            # Run model loading in thread pool (it's blocking)
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                nemo_asr.models.EncDecRNNTBPEModel.restore_from,
                self.model_path,
            )

            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()

            # Set to evaluation mode
            self.model.eval()

            logger.info(f"✓ Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def process_frame(self, frame, direction):
        """Process incoming frames."""

        # Accumulate audio frames
        if isinstance(frame, (AudioRawFrame, InputAudioRawFrame)):
            self.audio_buffer.append(frame.audio)

        # When VAD signals end of speech, process the buffer
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self.audio_buffer and not self.is_processing:
                await self._process_audio_buffer()

        # Pass through non-audio frames
        await self.push_frame(frame, direction)

    async def _process_audio_buffer(self):
        """Process accumulated audio through NeMo."""
        if not self.audio_buffer or self.model is None:
            return

        self.is_processing = True

        try:
            # Concatenate all audio chunks
            audio_data = b"".join(self.audio_buffer)
            self.audio_buffer = []  # Clear buffer

            logger.info(f"Processing {len(audio_data)} bytes of audio...")

            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # Convert to float32 and normalize
            audio_float = audio_np.astype(np.float32) / 32768.0

            # Run transcription in thread pool
            loop = asyncio.get_event_loop()

            text = await loop.run_in_executor(
                self.executor, self._transcribe_sync, audio_float
            )

            if text:
                logger.info(f"STT: {text}")
                # Create text frame and push downstream
                text_frame = TextFrame(text=text)
                await self.push_frame(text_frame)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback

            logger.error(traceback.format_exc())

        finally:
            self.is_processing = False

    def _transcribe_sync(self, audio: np.ndarray) -> Optional[str]:
        """Synchronous transcription (runs in thread pool)."""

        try:
            # Write audio to temporary WAV file (NeMo expects file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

                with wave.open(f, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes((audio * 32768).astype(np.int16).tobytes())

            # Transcribe with torch.no_grad() for inference
            import torch

            with torch.no_grad():
                if self.fp16 and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        result = self.model.transcribe(
                            [wav_path], batch_size=1, language_id=self.language
                        )
                else:
                    result = self.model.transcribe(
                        [wav_path], batch_size=1, language_id=self.language
                    )

            # Clean up temp file
            import os

            os.unlink(wav_path)

            # Extract text from result
            if result and len(result) > 0:
                text = (
                    result[0][0] if isinstance(result[0], (list, tuple)) else result[0]
                )
                return text.strip() if text else None

            return None

        except Exception as e:
            logger.error(f"Transcription sync error: {e}")
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
            if self.device == "cuda":
                torch.cuda.empty_cache()

        logger.info("NeMoSTTProcessor cleaned up")
