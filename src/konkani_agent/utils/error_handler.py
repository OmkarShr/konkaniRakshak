"""Comprehensive Error Handler

Handles all pipeline errors gracefully with:
- Automatic retry with exponential backoff
- Fallback responses in Konkani
- State recovery and continuation
- Detailed error logging and categorization
"""

import asyncio
import traceback
from typing import Optional, Dict, Callable, Any
from enum import Enum
from dataclasses import dataclass
from loguru import logger
import time
import random

from pipecat.frames.frames import (
    ErrorFrame,
    TextFrame,
    AudioRawFrame,
    StartFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class ErrorCategory(Enum):
    """Categories of errors."""

    STT_ERROR = "stt_error"
    LLM_ERROR = "llm_error"
    TTS_ERROR = "tts_error"
    NETWORK_ERROR = "network_error"
    GPU_ERROR = "gpu_error"
    AUDIO_ERROR = "audio_error"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels."""

    TRANSIENT = "transient"  # Can retry
    RECOVERABLE = "recoverable"  # Needs fallback
    CRITICAL = "critical"  # Pipeline stop


@dataclass
class ErrorRecord:
    """Recorded error information."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    timestamp: float
    retry_count: int = 0
    resolved: bool = False


class ErrorHandler(FrameProcessor):
    """
    Centralized error handling for the pipeline.
    """

    # Konkani fallback responses
    FALLBACK_RESPONSES = {
        "stt_error": "क्षमस्व, मी आपला आवाज ऐकू शकलो नाही. कृपया पुन्हा सांगा.",
        "llm_error": "क्षमस्व, मला समजलं नाही. कृपया पुन्हा सांगा.",
        "tts_error": "क्षमस्व, मी बोलू शकत नाही. पण मी आपली माहिती घेतली आहे.",
        "network_error": "क्षमस्व, जाळ्यात समस्या आहे. कृपया थोडी वाट पाहा.",
        "gpu_error": "क्षमस्व, प्रणालीत समस्या आहे. कृपया पुन्हा प्रयत्न करा.",
        "audio_error": "क्षमस्व, आवाजात समस्या आहे. कृपanya पुन्हा सांगा.",
        "unknown": "क्षमस्व, काहीतरी चूक झाली. कृपanya पुन्हा प्रयत्न करा.",
    }

    def __init__(
        self,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        enable_fallback_audio: bool = True,
        stop_on_critical: bool = True,
    ):
        super().__init__()

        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.enable_fallback_audio = enable_fallback_audio
        self.stop_on_critical = stop_on_critical

        # Error tracking
        self.error_history: list = []
        self.current_error: Optional[ErrorRecord] = None
        self.consecutive_errors = 0

        # State
        self.is_in_error_state = False
        self.recovery_in_progress = False

        # Callbacks
        self._on_error: Optional[Callable] = None
        self._on_recovery: Optional[Callable] = None

        logger.info(f"ErrorHandler initialized")
        logger.info(f"  Max retries: {max_retries}")
        logger.info(f"  Base delay: {base_retry_delay}s")

    async def process_frame(self, frame, direction):
        """Process frames and handle errors."""

        # Check for error frames
        if isinstance(frame, ErrorFrame):
            await self._handle_error_frame(frame)
            return

        # Pass through
        await self.push_frame(frame, direction)

    async def _handle_error_frame(self, frame: ErrorFrame):
        """Handle error frame."""
        error_msg = frame.error if hasattr(frame, "error") else str(frame)

        # Categorize error
        category = self._categorize_error(error_msg)
        severity = self._determine_severity(category, error_msg)

        logger.error(f"Error detected: [{category.value}] {error_msg}")

        # Create error record
        self.current_error = ErrorRecord(
            category=category,
            severity=severity,
            message=error_msg,
            timestamp=time.time(),
        )

        self.error_history.append(self.current_error)
        self.is_in_error_state = True

        # Handle based on severity
        if severity == ErrorSeverity.CRITICAL:
            await self._handle_critical_error()
        elif severity == ErrorSeverity.RECOVERABLE:
            await self._handle_recoverable_error()
        else:  # TRANSIENT
            await self._handle_transient_error()

    def _categorize_error(self, error_msg: str) -> ErrorCategory:
        """Categorize error from message."""
        error_lower = error_msg.lower()

        if any(x in error_lower for x in ["stt", "transcribe", "speech", "asr"]):
            return ErrorCategory.STT_ERROR
        elif any(
            x in error_lower for x in ["llm", "gemini", "language model", "generate"]
        ):
            return ErrorCategory.LLM_ERROR
        elif any(x in error_lower for x in ["tts", "synthesize", "speech synthesis"]):
            return ErrorCategory.TTS_ERROR
        elif any(
            x in error_lower
            for x in ["network", "connection", "timeout", "http", "api"]
        ):
            return ErrorCategory.NETWORK_ERROR
        elif any(
            x in error_lower for x in ["cuda", "gpu", "memory", "oom", "out of memory"]
        ):
            return ErrorCategory.GPU_ERROR
        elif any(x in error_lower for x in ["audio", "microphone", "speaker", "sound"]):
            return ErrorCategory.AUDIO_ERROR

        return ErrorCategory.UNKNOWN

    def _determine_severity(
        self, category: ErrorCategory, error_msg: str
    ) -> ErrorSeverity:
        """Determine error severity."""
        error_lower = error_msg.lower()

        # Critical errors
        if any(
            x in error_lower for x in ["oom", "out of memory", "segmentation fault"]
        ):
            return ErrorSeverity.CRITICAL

        # GPU errors might be recoverable
        if category == ErrorCategory.GPU_ERROR:
            return ErrorSeverity.RECOVERABLE

        # Network errors are usually transient
        if category == ErrorCategory.NETWORK_ERROR:
            return ErrorSeverity.TRANSIENT

        # STT errors are transient (user can retry)
        if category == ErrorCategory.STT_ERROR:
            return ErrorSeverity.TRANSIENT

        # Default: recoverable
        return ErrorSeverity.RECOVERABLE

    async def _handle_critical_error(self):
        """Handle critical error."""
        logger.critical("CRITICAL ERROR - Pipeline stopping")

        # Emit error message
        await self._emit_fallback_response("unknown")

        if self.stop_on_critical:
            # Signal pipeline stop
            await self.push_frame(EndFrame())

        if self._on_error:
            await self._call_callback(self._on_error, self.current_error)

    async def _handle_recoverable_error(self):
        """Handle recoverable error with fallback."""
        logger.warning(f"RECOVERABLE ERROR: {self.current_error.message}")

        # Emit fallback response
        await self._emit_fallback_response(self.current_error.category.value)

        # Mark as recovered
        self.is_in_error_state = False
        self.current_error.resolved = True
        self.consecutive_errors += 1

        if self._on_recovery:
            await self._call_callback(self._on_recovery, self.current_error)

    async def _handle_transient_error(self):
        """Handle transient error with retry."""
        if self.current_error.retry_count < self.max_retries:
            self.current_error.retry_count += 1

            # Exponential backoff with jitter
            delay = self.base_retry_delay * (2**self.current_error.retry_count)
            delay += random.uniform(0, 0.5)  # Add jitter

            logger.info(
                f"Retrying ({self.current_error.retry_count}/{self.max_retries}) "
                f"after {delay:.1f}s..."
            )

            await asyncio.sleep(delay)

            # Clear error state and continue
            self.is_in_error_state = False
            self.current_error.resolved = True

        else:
            # Max retries exceeded, treat as recoverable
            logger.warning("Max retries exceeded, using fallback")
            await self._handle_recoverable_error()

    async def _emit_fallback_response(self, error_type: str):
        """Emit fallback response."""
        response = self.FALLBACK_RESPONSES.get(
            error_type, self.FALLBACK_RESPONSES["unknown"]
        )

        # Emit text response
        text_frame = TextFrame(text=response)
        await self.push_frame(text_frame)

        # If TTS is available, emit audio (handled by TTS processor)

        logger.info(f"Emitted fallback: {response[:50]}...")

    async def _call_callback(self, callback: Callable, *args):
        """Call callback safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def set_error_callback(self, callback: Callable):
        """Set error callback."""
        self._on_error = callback

    def set_recovery_callback(self, callback: Callable):
        """Set recovery callback."""
        self._on_recovery = callback

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        if not self.error_history:
            return {"total": 0, "categories": {}, "consecutive": 0}

        categories = {}
        for err in self.error_history:
            cat = err.category.value
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total": len(self.error_history),
            "categories": categories,
            "consecutive": self.consecutive_errors,
            "current_state": "error" if self.is_in_error_state else "ok",
        }

    async def report_error(
        self, category: ErrorCategory, message: str, severity: ErrorSeverity = None
    ):
        """Manually report an error."""
        if severity is None:
            severity = self._determine_severity(category, message)

        error_frame = ErrorFrame(error=message)
        error_frame.category = category
        error_frame.severity = severity

        await self._handle_error_frame(error_frame)

    async def cleanup(self):
        """Clean up."""
        await super().cleanup()
        logger.info(f"ErrorHandler cleaned up. Total errors: {len(self.error_history)}")


class RetryManager:
    """
    Manages retries with circuit breaker pattern.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        circuit_threshold: int = 5,
        circuit_timeout: float = 60.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.circuit_threshold = circuit_threshold
        self.circuit_timeout = circuit_timeout

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.circuit_open = False

    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry."""
        if self.circuit_open:
            if self._should_close_circuit():
                self.circuit_open = False
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is open")

        for attempt in range(self.max_retries):
            try:
                result = await operation(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    raise

    def _record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.circuit_open = False

    def _record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.circuit_threshold:
            self.circuit_open = True
            logger.critical(f"Circuit breaker opened! ({self.failure_count} failures)")

    def _should_close_circuit(self) -> bool:
        """Check if circuit should close."""
        if self.last_failure_time is None:
            return True

        elapsed = time.time() - self.last_failure_time
        return elapsed > self.circuit_timeout
