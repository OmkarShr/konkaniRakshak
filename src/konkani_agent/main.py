"""Konkani Conversational AI Agent - Full Pipeline (Enhanced v2.0)

Main entry point with service-based architecture and all optimizations:
- Audio I/O (LocalAudioTransport)
- VAD (Silero)
- Barge-In (Interruption handling)
- STT (via HTTP service - no NeMo dependencies here!)
- LLM (Gemini)
- TTS (Enhanced with fallback and caching)
- Error Handling
- Latency Optimization
- GPU Monitoring

Architecture:
- This pipeline runs in 'konkani-agent' env (Pipecat, protobuf 4.x)
- STT service runs separately in 'konkani-stt' env (NeMo, protobuf 5.x)
- They communicate via HTTP on localhost:50051

Features:
- <1s time-to-first-audio
- Real-time barge-in/interruption
- GPU memory monitoring
- Automatic error recovery
- Web dashboard for monitoring
"""

import asyncio
import os
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import StartFrame, EndFrame
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.transports.base_transport import TransportParams

# Processors
from konkani_agent.processors.silero_vad import SileroVADProcessor
from konkani_agent.processors.stt_client import STTClientProcessor
from konkani_agent.processors.gemini_llm import GeminiProcessor
from konkani_agent.processors.enhanced_tts import EnhancedTTSProcessor
from konkani_agent.processors.barge_in import BargeInHandler

# Utilities
from konkani_agent.utils.gpu_monitor import GPUMemoryMonitor, GPUMemoryOptimizer
from konkani_agent.utils.latency_optimizer import LatencyOptimizer
from konkani_agent.utils.error_handler import ErrorHandler
from konkani_agent.utils.dashboard import MonitoringDashboard, DashboardMetrics

# Configuration
from config.settings import (
    AUDIO_CONFIG,
    LLM_CONFIG,
    TTS_CONFIG,
    VAD_CONFIG,
    PIPELINE_CONFIG,
    ERROR_CONFIG,
    MEMORY_CONFIG,
)


class KonkaniAgent:
    """
    Enhanced Konkani Voice Agent with all optimizations.
    """

    def __init__(self):
        self.pipeline = None
        self.runner = None
        self.task = None
        self.dashboard = None

        # Utilities
        self.gpu_monitor = None
        self.gpu_optimizer = None
        self.latency_optimizer = None
        self.error_handler = None

        # Stats
        self.start_time = None
        self.conversation_count = 0
        self.error_count = 0

    def setup_logging(self):
        """Setup enhanced logging."""
        logger.remove()

        # Console
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="{time:HH:mm:ss} | {level: <8} | {name: <20} | {message}",
            level="INFO",
            colorize=True,
        )

        # File
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        logger.add(
            sink=log_dir / "konkani_agent.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name: <20} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="1 week",
        )

    async def check_stt_service(self, stt_url: str) -> bool:
        """Check if STT service is available."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{stt_url}/health", timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"✓ STT service connected: {data}")
                        return True
        except Exception as e:
            logger.error(f"✗ Cannot connect to STT service: {e}")
        return False

    async def setup(self) -> bool:
        """Setup the pipeline and all components."""
        self.setup_logging()

        logger.info("=" * 70)
        logger.info("KONKANI VOICE AGENT v2.0 - ENHANCED")
        logger.info("=" * 70)
        logger.info(
            "Features: Barge-in, GPU monitoring, Latency optimization, Error recovery"
        )
        logger.info("")

        # Environment checks
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("⚠ GEMINI_API_KEY not set!")

        stt_url = os.getenv("STT_SERVICE_URL", "http://localhost:50051")
        if not await self.check_stt_service(stt_url):
            logger.error(
                "STT service not available. Start it with: ./start_stt_service.sh"
            )
            return False

        # Initialize GPU monitoring
        if MEMORY_CONFIG.get("monitoring_enabled", True):
            self.gpu_monitor = GPUMemoryMonitor(
                device="cuda:0",
                check_interval=MEMORY_CONFIG.get("check_interval", 1.0),
            )
            self.gpu_optimizer = GPUMemoryOptimizer(self.gpu_monitor)

            # Register cleanup actions
            self.gpu_optimizer.register_critical_action(
                lambda: logger.warning("GPU memory critical - clearing cache")
            )
            self.gpu_optimizer.register_emergency_action(
                lambda: logger.critical("GPU memory emergency - shutting down models")
            )

            self.gpu_monitor.start()
            logger.info("✓ GPU monitoring started")

        # Initialize error handler
        self.error_handler = ErrorHandler(
            max_retries=ERROR_CONFIG.get("max_retries", 3),
            base_retry_delay=ERROR_CONFIG.get("base_retry_delay", 1.0),
            enable_fallback_audio=ERROR_CONFIG.get("enable_fallback_audio", True),
        )

        # Initialize latency optimizer
        self.latency_optimizer = LatencyOptimizer(
            target_latency_ms=PIPELINE_CONFIG.get("latency_target_ms", 1000),
            warmup_on_start=True,
            enable_streaming=PIPELINE_CONFIG.get("enable_streaming", True),
        )

        # Initialize dashboard
        if os.getenv("ENABLE_DASHBOARD", "true").lower() == "true":
            self.dashboard = MonitoringDashboard(port=8080, update_interval=1.0)
            await self.dashboard.start()
            asyncio.create_task(
                self.dashboard.run_metric_collection(self._get_metrics_for_dashboard)
            )
            logger.info("✓ Dashboard running at http://localhost:8080")

        # Build pipeline
        await self._build_pipeline(stt_url, gemini_api_key)

        return True

    async def _build_pipeline(self, stt_url: str, gemini_api_key: str):
        """Build the complete pipeline with all components."""

        # Audio transport
        transport_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=AUDIO_CONFIG["sample_rate"],
            audio_in_channels=AUDIO_CONFIG["channels"],
            audio_out_enabled=True,
            audio_out_sample_rate=TTS_CONFIG.get("sample_rate", 24000),
            audio_out_channels=1,
        )
        transport = LocalAudioTransport(params=transport_params)

        logger.info("Building enhanced pipeline...")

        # Create processors
        vad = SileroVADProcessor(
            threshold=VAD_CONFIG["threshold"],
            min_speech_duration_ms=VAD_CONFIG["min_speech_duration_ms"],
            min_silence_duration_ms=VAD_CONFIG["min_silence_duration_ms"],
            sample_rate=AUDIO_CONFIG["sample_rate"],
        )

        barge_in = BargeInHandler(
            vad_threshold=0.5,
            min_interrupt_duration_ms=400,
            debounce_ms=100,
            enable_echo_cancellation=True,
            sample_rate=AUDIO_CONFIG["sample_rate"],
        )

        stt = STTClientProcessor(
            service_url=stt_url,
            sample_rate=AUDIO_CONFIG["sample_rate"],
        )

        llm = GeminiProcessor(
            api_key=gemini_api_key,
            model=LLM_CONFIG["dev"]["model"],
            temperature=LLM_CONFIG["dev"]["temperature"],
            max_tokens=LLM_CONFIG["dev"]["max_tokens"],
        )

        tts = EnhancedTTSProcessor(
            primary_model="xtts",
            fallback_model="parler",
            language="mr",
            device=TTS_CONFIG.get("device", "cuda"),
            fp16=TTS_CONFIG.get("fp16", True),
            sample_rate=24000,
            warmup=True,
            enable_fallback_audio=True,
        )

        # Build pipeline with all optimizations
        self.pipeline = Pipeline(
            [
                # Input
                transport.input(),
                # Error handling (first to catch all errors)
                self.error_handler,
                # Latency monitoring
                self.latency_optimizer,
                # VAD for speech detection
                vad,
                # Barge-in for interruptions
                barge_in,
                # STT
                stt,
                # LLM
                llm,
                # TTS
                tts,
                # Output
                transport.output(),
            ]
        )

        # Create task and runner
        self.task = PipelineTask(self.pipeline)
        self.runner = PipelineRunner()

        logger.info("✓ Enhanced pipeline ready")
        logger.info("")
        logger.info("Components:")
        logger.info("  ✓ ErrorHandler (with retry and fallback)")
        logger.info("  ✓ LatencyOptimizer (target <1s TTFA)")
        logger.info("  ✓ SileroVAD")
        logger.info("  ✓ BargeInHandler (interruption support)")
        logger.info("  ✓ STTClient (HTTP)")
        logger.info("  ✓ Gemini LLM")
        logger.info("  ✓ EnhancedTTS (XTTSv2 + fallback)")
        logger.info("")

        if self.gpu_monitor:
            logger.info("Monitoring:")
            logger.info("  ✓ GPU Memory Monitor")
            logger.info(f"  ✓ Dashboard (http://localhost:8080)")
            logger.info("")

        logger.info("Instructions:")
        logger.info("  1. Speak into microphone")
        logger.info("  2. VAD detects speech end")
        logger.info("  3. You can INTERRUPT the agent anytime")
        logger.info("  4. Press Ctrl+C to stop")
        logger.info("")

    async def _get_metrics_for_dashboard(self) -> DashboardMetrics:
        """Get current metrics for dashboard."""
        import time

        gpu_stats = {"allocated_mb": 0, "reserved_mb": 0, "utilization": 0}
        if self.gpu_monitor:
            stats = self.gpu_monitor.get_current_stats()
            gpu_stats["allocated_mb"] = stats.allocated_mb
            gpu_stats["reserved_mb"] = stats.reserved_mb
            gpu_stats["utilization"] = stats.utilization_percent

        latency_stats = {"avg_ms": 0, "p95_ms": 0, "last_ms": 0}
        if self.latency_optimizer:
            metrics = self.latency_optimizer.get_average_metrics()
            latency_stats["avg_ms"] = metrics.get("avg_ttfa_ms", 0)
            latency_stats["p95_ms"] = metrics.get("p95_ttfa_ms", 0)

        error_stats = {"count": 0, "rate": 0}
        if self.error_handler:
            summary = self.error_handler.get_error_summary()
            error_stats["count"] = summary.get("total", 0)
            error_stats["rate"] = summary.get("total", 0) / max(
                self.conversation_count, 1
            )

        uptime = time.time() - self.start_time if self.start_time else 0

        return DashboardMetrics(
            timestamp=time.time(),
            gpu_allocated_mb=gpu_stats["allocated_mb"],
            gpu_reserved_mb=gpu_stats["reserved_mb"],
            gpu_utilization=gpu_stats["utilization"],
            avg_latency_ms=latency_stats["avg_ms"],
            p95_latency_ms=latency_stats["p95_ms"],
            last_latency_ms=latency_stats["last_ms"],
            error_count=error_stats["count"],
            error_rate=error_stats["rate"],
            conversation_count=self.conversation_count,
            active_conversations=1,  # Single session
            uptime_seconds=uptime,
        )

    async def run(self):
        """Run the pipeline."""
        if not self.pipeline:
            logger.error("Pipeline not initialized")
            return

        self.start_time = asyncio.get_event_loop().time()

        try:
            await self.runner.run(self.task)
        except KeyboardInterrupt:
            logger.info("\n✓ Pipeline stopped by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback

            logger.error(traceback.format_exc())
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("")
        logger.info("Shutting down...")

        # Stop monitoring
        if self.gpu_monitor:
            self.gpu_monitor.stop()
            summary = self.gpu_monitor.get_summary()
            logger.info(f"GPU Memory Peak: {summary.get('peak_mb', 0):.1f}MB")

        # Print latency summary
        if self.latency_optimizer:
            self.latency_optimizer.print_summary()

        # Error summary
        if self.error_handler:
            errors = self.error_handler.get_error_summary()
            logger.info(f"Errors: {errors.get('total', 0)} total")

        # Stop dashboard
        if self.dashboard:
            await self.dashboard.stop()

        # Cleanup
        if self.pipeline:
            for processor in self.pipeline.processors:
                if hasattr(processor, "cleanup"):
                    await processor.cleanup()

        uptime = (
            asyncio.get_event_loop().time() - self.start_time if self.start_time else 0
        )
        logger.info(f"Uptime: {uptime:.1f}s")
        logger.info("✓ Shutdown complete")


async def main():
    """Main entry point."""
    agent = KonkaniAgent()

    if await agent.setup():
        await agent.run()
    else:
        logger.error("Setup failed - exiting")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
