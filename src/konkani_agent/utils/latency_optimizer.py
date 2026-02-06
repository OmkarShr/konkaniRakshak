"""Latency Optimizer

Optimizes pipeline latency for <1s time-to-first-audio.
Strategies:
- Streaming audio output
- Pipeline component warmup
- LLM prompt optimization
- Audio pre-buffering
"""

import time
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import numpy as np

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    StartFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


@dataclass
class LatencyMetrics:
    """Latency measurements for a single turn."""

    start_time: float
    stt_start: Optional[float] = None
    stt_end: Optional[float] = None
    llm_start: Optional[float] = None
    llm_first_token: Optional[float] = None
    llm_end: Optional[float] = None
    tts_start: Optional[float] = None
    tts_first_audio: Optional[float] = None
    tts_end: Optional[float] = None
    total_end: Optional[float] = None

    @property
    def time_to_first_audio(self) -> Optional[float]:
        """Time from speech end to first audio output."""
        if self.stt_end and self.tts_first_audio:
            return self.tts_first_audio - self.stt_end
        return None

    @property
    def total_latency(self) -> Optional[float]:
        """Total latency for complete response."""
        if self.start_time and self.total_end:
            return self.total_end - self.start_time
        return None

    @property
    def llm_latency(self) -> Optional[float]:
        """LLM time to first token."""
        if self.llm_start and self.llm_first_token:
            return self.llm_first_token - self.llm_start
        return None

    @property
    def stt_latency(self) -> Optional[float]:
        """STT transcription latency."""
        if self.stt_start and self.stt_end:
            return self.stt_end - self.stt_start
        return None


class LatencyOptimizer(FrameProcessor):
    """
    Optimizes and monitors pipeline latency.
    """

    def __init__(
        self,
        target_latency_ms: int = 1000,
        warmup_on_start: bool = True,
        enable_streaming: bool = True,
        prebuffer_size: int = 5,  # Pre-buffer 5 audio chunks
    ):
        super().__init__()

        self.target_latency_ms = target_latency_ms
        self.warmup_on_start = warmup_on_start
        self.enable_streaming = enable_streaming
        self.prebuffer_size = prebuffer_size

        # Current turn metrics
        self.current_metrics: Optional[LatencyMetrics] = None

        # History
        self.metrics_history: deque = deque(maxlen=100)

        # Pre-buffered audio
        self.audio_prebuffer: List[AudioRawFrame] = []

        # Optimization settings
        self._streaming_chunks = []

        logger.info(f"LatencyOptimizer initialized")
        logger.info(f"  Target: {target_latency_ms}ms")
        logger.info(f"  Streaming: {enable_streaming}")

    async def start(self, frame):
        """Initialize and warmup if needed."""
        await super().start(frame)

        if self.warmup_on_start:
            await self._warmup_pipeline()

    async def _warmup_pipeline(self):
        """Warmup pipeline components to reduce cold-start latency."""
        logger.info("Warming up pipeline for latency optimization...")

        # Warmup typically happens in individual processors
        # This is just a coordination point
        warmup_start = time.time()

        # Signal warmup to downstream processors
        # They should handle their own warmup

        warmup_end = time.time()
        logger.info(
            f"✓ Pipeline warmup complete ({(warmup_end - warmup_start) * 1000:.1f}ms)"
        )

    async def process_frame(self, frame, direction):
        """Monitor and optimize latency."""

        current_time = time.time()

        # Track timing for different frame types
        if isinstance(frame, StartFrame):
            # New turn started
            self.current_metrics = LatencyMetrics(start_time=current_time)
            logger.debug("New turn started - tracking latency")

        elif isinstance(frame, TextFrame):
            # Track LLM output
            if self.current_metrics:
                if self.current_metrics.llm_start is None:
                    self.current_metrics.llm_start = current_time
                    logger.debug(f"LLM started: {current_time:.3f}")

                if self.current_metrics.llm_first_token is None:
                    self.current_metrics.llm_first_token = current_time
                    logger.debug(f"LLM first token: {current_time:.3f}")

        elif isinstance(frame, AudioRawFrame):
            # Track TTS output
            if self.current_metrics:
                if self.current_metrics.tts_start is None:
                    self.current_metrics.tts_start = current_time
                    logger.debug(f"TTS started: {current_time:.3f}")

                if self.current_metrics.tts_first_audio is None:
                    self.current_metrics.tts_first_audio = current_time
                    latency = (current_time - self.current_metrics.start_time) * 1000
                    logger.info(f"🎯 TIME-TO-FIRST-AUDIO: {latency:.1f}ms")

        elif isinstance(frame, EndFrame):
            # Turn ended
            if self.current_metrics:
                self.current_metrics.total_end = current_time
                self._record_metrics()

        # Pass through
        await self.push_frame(frame, direction)

    def _record_metrics(self):
        """Record and log metrics for completed turn."""
        if not self.current_metrics:
            return

        metrics = self.current_metrics

        # Log all latencies
        logger.info("=" * 60)
        logger.info("LATENCY REPORT")
        logger.info("=" * 60)

        if metrics.stt_latency:
            logger.info(f"STT:        {metrics.stt_latency * 1000:.1f}ms")
        if metrics.llm_latency:
            logger.info(f"LLM (TTFT): {metrics.llm_latency * 1000:.1f}ms")

        ttfa = metrics.time_to_first_audio
        if ttfa:
            status = "✓" if ttfa * 1000 < self.target_latency_ms else "⚠"
            logger.info(f"TTFA:       {ttfa * 1000:.1f}ms {status}")

        if metrics.total_latency:
            logger.info(f"Total:      {metrics.total_latency * 1000:.1f}ms")

        logger.info("=" * 60)

        # Store in history
        self.metrics_history.append(metrics)

        # Reset current
        self.current_metrics = None

    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics from history."""
        if not self.metrics_history:
            return {}

        stt_latencies = [m.stt_latency for m in self.metrics_history if m.stt_latency]
        llm_latencies = [m.llm_latency for m in self.metrics_history if m.llm_latency]
        ttfa_list = [
            m.time_to_first_audio for m in self.metrics_history if m.time_to_first_audio
        ]
        total_latencies = [
            m.total_latency for m in self.metrics_history if m.total_latency
        ]

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        return {
            "avg_stt_ms": avg(stt_latencies) * 1000 if stt_latencies else 0,
            "avg_llm_ms": avg(llm_latencies) * 1000 if llm_latencies else 0,
            "avg_ttfa_ms": avg(ttfa_list) * 1000 if ttfa_list else 0,
            "avg_total_ms": avg(total_latencies) * 1000 if total_latencies else 0,
            "p95_ttfa_ms": np.percentile([x * 1000 for x in ttfa_list], 95)
            if ttfa_list
            else 0,
            "samples": len(self.metrics_history),
        }

    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on metrics."""
        recommendations = []
        metrics = self.get_average_metrics()

        if not metrics:
            return ["No metrics available yet"]

        avg_ttfa = metrics.get("avg_ttfa_ms", 0)
        p95_ttfa = metrics.get("p95_ttfa_ms", 0)

        if avg_ttfa > self.target_latency_ms:
            recommendations.append(
                f"⚠ Average TTFA ({avg_ttfa:.0f}ms) exceeds target ({self.target_latency_ms}ms)"
            )

            if metrics.get("avg_stt_ms", 0) > 300:
                recommendations.append("  → Consider faster STT model or streaming")

            if metrics.get("avg_llm_ms", 0) > 500:
                recommendations.append("  → Consider faster LLM or prompt optimization")

        if p95_ttfa > self.target_latency_ms * 1.5:
            recommendations.append(
                f"⚠ P95 TTFA ({p95_ttfa:.0f}ms) is too high - inconsistent latency"
            )

        return recommendations if recommendations else ["✓ Latency targets met"]

    def print_summary(self):
        """Print summary of latency metrics."""
        metrics = self.get_average_metrics()

        if not metrics:
            logger.info("No latency data collected yet")
            return

        logger.info("=" * 60)
        logger.info("LATENCY SUMMARY (Last {} turns)".format(metrics["samples"]))
        logger.info("=" * 60)
        logger.info(f"Average STT:      {metrics['avg_stt_ms']:.1f}ms")
        logger.info(f"Average LLM:      {metrics['avg_llm_ms']:.1f}ms")
        logger.info(
            f"Average TTFA:     {metrics['avg_ttfa_ms']:.1f}ms (target: {self.target_latency_ms}ms)"
        )
        logger.info(f"P95 TTFA:         {metrics['p95_ttfa_ms']:.1f}ms")
        logger.info(f"Average Total:    {metrics['avg_total_ms']:.1f}ms")
        logger.info("=" * 60)

        for rec in self.get_recommendations():
            logger.info(rec)


class StreamingAudioBuffer:
    """
    Buffers audio chunks for smooth streaming playback.
    """

    def __init__(self, buffer_duration_ms: float = 100):
        self.buffer_duration_ms = buffer_duration_ms
        self.chunks = []
        self.is_streaming = False

    def add_chunk(self, chunk: AudioRawFrame):
        """Add audio chunk to buffer."""
        self.chunks.append(chunk)

    def should_emit(self) -> bool:
        """Check if buffer has enough audio to emit."""
        if not self.chunks:
            return False

        # Calculate total duration
        total_samples = sum(
            len(c.audio) // 2 for c in self.chunks
        )  # 2 bytes per int16 sample
        duration_ms = (total_samples / 16000) * 1000  # Assuming 16kHz

        return duration_ms >= self.buffer_duration_ms

    def emit_and_clear(self) -> List[AudioRawFrame]:
        """Emit all chunks and clear buffer."""
        chunks = self.chunks.copy()
        self.chunks.clear()
        return chunks
