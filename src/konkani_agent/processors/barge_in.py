"""Barge-In and Interruption Handler

Handles user interruptions during agent speech with:
- Voice activity detection during agent output
- Audio cancellation and state management
- Conversation context preservation through interruptions
- Echo cancellation support
"""

import asyncio
import numpy as np
from typing import Optional, Callable
from enum import Enum
from dataclasses import dataclass
from loguru import logger
import time

from pipecat.frames.frames import (
    AudioRawFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class InterruptionState(Enum):
    """Current interruption state."""

    LISTENING = "listening"  # Waiting for user
    SPEAKING = "speaking"  # Agent is speaking
    INTERRUPTING = "interrupting"  # User interrupted, stopping
    PROCESSING = "processing"  # Processing interruption


@dataclass
class InterruptionContext:
    """Context preserved during interruption."""

    interrupted_text: str  # What agent was saying
    user_interruption: str  # What user said
    timestamp: float
    should_resume: bool = False  # Whether to resume interrupted speech


class BargeInHandler(FrameProcessor):
    """
    Handles barge-in (user interruption) during agent speech.

    Monitors input audio while output is playing and detects
    when user starts speaking.
    """

    def __init__(
        self,
        vad_threshold: float = 0.5,
        min_interrupt_duration_ms: int = 400,  # Require 400ms speech to interrupt
        debounce_ms: int = 100,
        enable_echo_cancellation: bool = True,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.vad_threshold = vad_threshold
        self.min_interrupt_duration_ms = min_interrupt_duration_ms
        self.debounce_ms = debounce_ms
        self.enable_echo_cancellation = enable_echo_cancellation
        self.sample_rate = sample_rate

        # State
        self.state = InterruptionState.LISTENING
        self.interruption_context: Optional[InterruptionContext] = None

        # VAD during speech
        self.vad_model = None
        self.is_user_speaking = False
        self.speech_start_time: Optional[float] = None
        self.last_interrupt_time: Optional[float] = None

        # Audio buffers for echo cancellation
        self.output_buffer = []  # Recent output audio
        self.input_buffer = []  # Recent input audio
        self.buffer_duration_ms = 100  # Keep 100ms of audio

        # Callbacks
        self._on_interruption: Optional[Callable] = None
        self._on_resume: Optional[Callable] = None

        # Stats
        self.interruption_count = 0
        self.total_agent_speech_time = 0.0

        logger.info(f"BargeInHandler initialized")
        logger.info(f"  VAD threshold: {vad_threshold}")
        logger.info(f"  Min interrupt: {min_interrupt_duration_ms}ms")
        logger.info(f"  Echo cancellation: {enable_echo_cancellation}")

    async def start(self, frame):
        """Initialize VAD model."""
        await super().start(frame)
        await self._load_vad_model()

    async def _load_vad_model(self):
        """Load Silero VAD model."""
        try:
            import torch

            logger.info("Loading Silero VAD for barge-in detection...")

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            self.vad_model = model
            logger.info("✓ Barge-in VAD loaded")

        except Exception as e:
            logger.error(f"Failed to load barge-in VAD: {e}")
            self.vad_model = None

    def set_interruption_callback(self, callback: Callable):
        """Set callback for when interruption is detected."""
        self._on_interruption = callback

    def set_resume_callback(self, callback: Callable):
        """Set callback for when agent should resume after interruption."""
        self._on_resume = callback

    async def process_frame(self, frame, direction):
        """Process frames for interruption detection."""

        # Handle input audio (user speech)
        if isinstance(frame, (AudioRawFrame, InputAudioRawFrame)):
            await self._handle_input_audio(frame)

        # Handle output audio (agent speech)
        elif isinstance(frame, OutputAudioRawFrame):
            await self._handle_output_audio(frame)

            # If agent is speaking, monitor for interruption
            if self.state == InterruptionState.SPEAKING:
                await self._monitor_for_interruption()

        # Detect state changes from other processors
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.state = InterruptionState.LISTENING
            logger.debug("Agent finished speaking - back to listening")

        # Pass through all frames
        await self.push_frame(frame, direction)

    async def _handle_input_audio(self, frame):
        """Process input audio from user."""
        # Store in buffer for echo cancellation
        audio_data = np.frombuffer(frame.audio, dtype=np.int16)
        self.input_buffer.append(audio_data)

        # Keep buffer size limited
        samples_to_keep = int(self.sample_rate * self.buffer_duration_ms / 1000)
        if len(self.input_buffer) > samples_to_keep:
            self.input_buffer.pop(0)

    async def _handle_output_audio(self, frame):
        """Process output audio from agent."""
        if self.state == InterruptionState.LISTENING:
            # Agent just started speaking
            self.state = InterruptionState.SPEAKING
            self.speech_start_time = time.time()
            logger.info("Agent started speaking - monitoring for barge-in")

        # Store in buffer for echo cancellation
        audio_data = np.frombuffer(frame.audio, dtype=np.int16)
        self.output_buffer.append(audio_data)

        # Keep buffer size limited
        samples_to_keep = int(self.sample_rate * self.buffer_duration_ms / 1000)
        if len(self.output_buffer) > samples_to_keep:
            self.output_buffer.pop(0)

    async def _monitor_for_interruption(self):
        """Monitor for user interruption during agent speech."""
        if self.vad_model is None:
            return

        if not self.input_buffer:
            return

        try:
            import torch

            # Get recent input audio
            audio_data = np.concatenate(self.input_buffer)

            # Echo cancellation: subtract output from input
            if self.enable_echo_cancellation and self.output_buffer:
                output_data = np.concatenate(self.output_buffer)
                min_len = min(len(audio_data), len(output_data))
                audio_data = audio_data[-min_len:] - output_data[-min_len:]

            # Normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float)

            # Run VAD
            with torch.no_grad():
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

            current_time = time.time()

            # Check for speech
            if speech_prob > self.vad_threshold:
                if not self.is_user_speaking:
                    # Speech started
                    self.is_user_speaking = True
                    self.speech_start_time = current_time
                    logger.debug("Barge-in: User speech detected")

                # Check if speech duration exceeds minimum for interruption
                speech_duration = (current_time - self.speech_start_time) * 1000

                # Debounce check
                if self.last_interrupt_time:
                    time_since_last = (current_time - self.last_interrupt_time) * 1000
                    if time_since_last < self.debounce_ms:
                        return

                if speech_duration >= self.min_interrupt_duration_ms:
                    # Trigger interruption
                    await self._trigger_interruption()

            else:
                # No speech detected
                if self.is_user_speaking:
                    self.is_user_speaking = False
                    logger.debug("Barge-in: User speech ended")

        except Exception as e:
            logger.error(f"Barge-in detection error: {e}")

    async def _trigger_interruption(self):
        """Handle interruption detection."""
        current_time = time.time()
        self.last_interrupt_time = current_time
        self.interruption_count += 1

        # Calculate how much agent speech was interrupted
        if self.speech_start_time:
            interrupted_duration = current_time - self.speech_start_time
            self.total_agent_speech_time += interrupted_duration

        logger.info(f"🛑 INTERRUPTION #{self.interruption_count} detected!")

        # Change state
        self.state = InterruptionState.INTERRUPTING

        # Clear audio buffers
        self.output_buffer.clear()

        # Create interruption context
        self.interruption_context = InterruptionContext(
            interrupted_text="",  # Will be updated by LLM processor
            user_interruption="",
            timestamp=current_time,
            should_resume=False,
        )

        # Emit interruption frame
        await self.push_frame(UserStartedSpeakingFrame())

        # Call registered callback
        if self._on_interruption:
            try:
                if asyncio.iscoroutinefunction(self._on_interruption):
                    await self._on_interruption(self.interruption_context)
                else:
                    self._on_interruption(self.interruption_context)
            except Exception as e:
                logger.error(f"Interruption callback error: {e}")

    def update_interrupted_text(self, text: str):
        """Update what the agent was saying when interrupted."""
        if self.interruption_context:
            self.interruption_context.interrupted_text = text

    def update_user_interruption(self, text: str):
        """Update what the user said during interruption."""
        if self.interruption_context:
            self.interruption_context.user_interruption = text
            self.state = InterruptionState.PROCESSING

    def should_resume_interrupted(self) -> bool:
        """Determine if agent should resume interrupted speech."""
        if not self.interruption_context:
            return False

        # Simple heuristic: Don't resume if user asked a new question
        user_text = self.interruption_context.user_interruption.lower()
        resume_phrases = ["continue", "go on", "what were you saying"]

        for phrase in resume_phrases:
            if phrase in user_text:
                return True

        return False

    async def resume_speaking(self, text: str):
        """Resume speaking after interruption."""
        logger.info(f"Resuming interrupted speech: {text[:50]}...")

        # Create text frame to resume
        text_frame = TextFrame(text=text)
        await self.push_frame(text_frame)

        if self._on_resume:
            try:
                if asyncio.iscoroutinefunction(self._on_resume):
                    await self._on_resume(text)
                else:
                    self._on_resume(text)
            except Exception as e:
                logger.error(f"Resume callback error: {e}")

    def get_stats(self) -> dict:
        """Get interruption statistics."""
        return {
            "interruption_count": self.interruption_count,
            "total_interrupted_speech_time": self.total_agent_speech_time,
            "current_state": self.state.value,
        }

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()

        # Clear buffers
        self.input_buffer.clear()
        self.output_buffer.clear()

        # Clean up model
        if self.vad_model is not None:
            import torch

            del self.vad_model
            torch.cuda.empty_cache()

        logger.info(f"BargeInHandler cleaned up. Stats: {self.get_stats()}")
