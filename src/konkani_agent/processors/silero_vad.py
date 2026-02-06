"""Silero VAD Processor for Pipecat

Voice Activity Detection using Silero VAD v5.0
"""

import asyncio
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class SileroVADProcessor(FrameProcessor):
    """
    Silero VAD for voice activity detection.

    Detects when user starts and stops speaking.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate

        self.model = None
        self.utils = None
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None

        logger.info(f"SileroVADProcessor initialized")
        logger.info(f"  Threshold: {threshold}")
        logger.info(f"  Min speech: {min_speech_duration_ms}ms")
        logger.info(f"  Min silence: {min_silence_duration_ms}ms")

    async def start(self, frame):
        """Load VAD model."""
        await super().start(frame)
        await self._load_model()

    async def _load_model(self):
        """Load Silero VAD model."""
        try:
            import torch

            logger.info("Loading Silero VAD model...")

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            self.model = model
            self.utils = utils

            logger.info("✓ Silero VAD loaded")

        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            # Continue without VAD (fallback)
            self.model = None

    async def process_frame(self, frame, direction):
        """Process audio frames for voice activity."""

        if not isinstance(frame, (AudioRawFrame, InputAudioRawFrame)):
            await self.push_frame(frame, direction)
            return

        # Pass through audio
        await self.push_frame(frame, direction)

        # If VAD not loaded, skip detection
        if self.model is None:
            return

        try:
            # Convert audio to tensor
            import torch

            audio_data = np.frombuffer(frame.audio, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float)

            # Get speech probability
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()

            current_time = asyncio.get_event_loop().time()

            # Detect speech start
            if speech_prob > self.threshold and not self.is_speaking:
                if self.speech_start_time is None:
                    self.speech_start_time = current_time

                # Check if speech duration exceeds minimum
                speech_duration = (current_time - self.speech_start_time) * 1000
                if speech_duration >= self.min_speech_duration_ms:
                    self.is_speaking = True
                    self.silence_start_time = None
                    logger.info("VAD: Speech started")

                    # Emit speech started frame
                    await self.push_frame(UserStartedSpeakingFrame())

            # Detect speech end
            elif speech_prob <= self.threshold and self.is_speaking:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time

                # Check if silence duration exceeds minimum
                silence_duration = (current_time - self.silence_start_time) * 1000
                if silence_duration >= self.min_silence_duration_ms:
                    self.is_speaking = False
                    self.speech_start_time = None
                    logger.info("VAD: Speech stopped")

                    # Emit speech stopped frame
                    await self.push_frame(UserStoppedSpeakingFrame())

            # Reset if not speaking and low probability
            elif speech_prob <= self.threshold and not self.is_speaking:
                self.speech_start_time = None

        except Exception as e:
            logger.error(f"VAD processing error: {e}")
