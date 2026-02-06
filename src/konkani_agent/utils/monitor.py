#!/usr/bin/env python3
"""Pipeline Monitor and Metrics

Tracks performance metrics, errors, and health status.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List
from collections import deque
from loguru import logger


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""

    # Timing metrics (in milliseconds)
    vad_latency_ms: float = 0.0
    stt_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Error tracking
    errors: List[Dict] = field(default_factory=list)

    # Audio metrics
    audio_duration_sec: float = 0.0
    transcription_length: int = 0
    response_length: int = 0


class PipelineMonitor:
    """Monitors pipeline health and performance."""

    def __init__(self, history_size: int = 100):
        """Initialize monitor.

        Args:
            history_size: Number of requests to keep in history
        """
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_session_start = time.time()

        logger.info("PipelineMonitor initialized")

    def record_request_start(self) -> str:
        """Record start of new request.

        Returns:
            Request ID for tracking
        """
        request_id = f"req_{int(time.time() * 1000)}"
        self.metrics_history.append(
            {"request_id": request_id, "start_time": time.time(), "status": "started"}
        )
        return request_id

    def record_metric(self, request_id: str, stage: str, latency_ms: float) -> None:
        """Record metric for a processing stage.

        Args:
            request_id: Request identifier
            stage: Processing stage (vad, stt, llm, tts)
            latency_ms: Latency in milliseconds
        """
        for metric in self.metrics_history:
            if metric.get("request_id") == request_id:
                metric[f"{stage}_latency_ms"] = latency_ms
                break

        logger.debug(f"{stage.upper()} latency: {latency_ms:.1f}ms")

    def record_completion(
        self, request_id: str, success: bool, error: str = None
    ) -> None:
        """Record request completion.

        Args:
            request_id: Request identifier
            success: Whether request succeeded
            error: Error message if failed
        """
        for metric in self.metrics_history:
            if metric.get("request_id") == request_id:
                metric["status"] = "completed" if success else "failed"
                metric["end_time"] = time.time()
                metric["duration_ms"] = (
                    metric["end_time"] - metric["start_time"]
                ) * 1000

                if error:
                    metric["error"] = error
                    logger.error(f"Request {request_id} failed: {error}")
                else:
                    logger.info(
                        f"Request {request_id} completed in {metric['duration_ms']:.1f}ms"
                    )
                break

    def get_statistics(self) -> Dict:
        """Get pipeline statistics.

        Returns:
            Dictionary with performance statistics
        """
        if not self.metrics_history:
            return {"status": "no_data"}

        completed = [m for m in self.metrics_history if m.get("status") == "completed"]
        failed = [m for m in self.metrics_history if m.get("status") == "failed"]

        if not completed:
            return {"status": "no_completed_requests"}

        stats = {
            "total_requests": len(self.metrics_history),
            "successful": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(self.metrics_history) * 100,
            # Average latencies
            "avg_total_latency_ms": sum(m.get("duration_ms", 0) for m in completed)
            / len(completed),
            "avg_vad_latency_ms": sum(
                m.get("vad_latency_ms", 0) for m in completed if "vad_latency_ms" in m
            )
            / len([m for m in completed if "vad_latency_ms" in m])
            if any("vad_latency_ms" in m for m in completed)
            else 0,
            "avg_stt_latency_ms": sum(
                m.get("stt_latency_ms", 0) for m in completed if "stt_latency_ms" in m
            )
            / len([m for m in completed if "stt_latency_ms" in m])
            if any("stt_latency_ms" in m for m in completed)
            else 0,
            "avg_llm_latency_ms": sum(
                m.get("llm_latency_ms", 0) for m in completed if "llm_latency_ms" in m
            )
            / len([m for m in completed if "llm_latency_ms" in m])
            if any("llm_latency_ms" in m for m in completed)
            else 0,
            "avg_tts_latency_ms": sum(
                m.get("tts_latency_ms", 0) for m in completed if "tts_latency_ms" in m
            )
            / len([m for m in completed if "tts_latency_ms" in m])
            if any("tts_latency_ms" in m for m in completed)
            else 0,
            # Session info
            "session_duration_sec": time.time() - self.current_session_start,
            "history_size": len(self.metrics_history),
        }

        return stats

    def print_report(self) -> None:
        """Print performance report to console."""
        stats = self.get_statistics()

        if stats.get("status") == "no_data":
            logger.info("No metrics collected yet")
            return

        logger.info("=" * 60)
        logger.info("Pipeline Performance Report")
        logger.info("=" * 60)
        logger.info(f"Total Requests: {stats['total_requests']}")
        logger.info(f"Success Rate: {stats['success_rate']:.1f}%")
        logger.info(f"Session Duration: {stats['session_duration_sec']:.0f}s")
        logger.info("")
        logger.info("Average Latencies:")
        logger.info(f"  VAD:   {stats['avg_vad_latency_ms']:>6.1f}ms")
        logger.info(f"  STT:   {stats['avg_stt_latency_ms']:>6.1f}ms")
        logger.info(f"  LLM:   {stats['avg_llm_latency_ms']:>6.1f}ms")
        logger.info(f"  TTS:   {stats['avg_tts_latency_ms']:>6.1f}ms")
        logger.info(f"  Total: {stats['avg_total_latency_ms']:>6.1f}ms")
        logger.info("=" * 60)


# Global monitor instance
monitor = PipelineMonitor()
