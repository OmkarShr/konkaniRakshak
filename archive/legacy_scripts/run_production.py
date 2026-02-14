"""Production Deployment Runner

Runs the complete pipeline with production configuration.
Includes monitoring, error handling, and all optimizations.
"""

import asyncio
import os
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

# Production configuration
from config.production import (
    CURRENT_CONFIG,
    GPU_CONFIG,
    AUDIO_CONFIG,
    PIPELINE_CONFIG,
    STT_CONFIG,
    TTS_CONFIG,
    LLM_CONFIG,
    VAD_CONFIG,
    BARGE_IN_CONFIG,
    MEMORY_CONFIG,
    ERROR_CONFIG,
    LOG_CONFIG,
)

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import StartFrame, EndFrame
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.transports.base_transport import TransportParams

# Our custom processors
from konkani_agent.processors.silero_vad import SileroVADProcessor
from konkani_agent.processors.stt_client import STTClientProcessor
from konkani_agent.processors.gemini_llm import GeminiProcessor
from konkani_agent.processors.enhanced_tts import EnhancedTTSProcessor
from konkani_agent.processors.barge_in import BargeInHandler

# Utilities
from konkani_agent.utils.gpu_monitor import GPUMemoryMonitor, GPUMemoryOptimizer
from konkani_agent.utils.latency_optimizer import LatencyOptimizer
from konkani_agent.utils.error_handler import ErrorHandler, RetryManager
from konkani_agent.utils.field_testing import AutomatedTester, PerformanceLogger


def setup_production_logging():
    """Setup production-grade logging."""
    from loguru import logger

    # Remove default
    logger.remove()

    # Console handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level=LOG_CONFIG["level"],
        colorize=True,
    )

    # File handler
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        sink=log_dir / "production.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name: <20} | {message}",
        level=LOG_CONFIG["level"],
        rotation=LOG_CONFIG["rotation"],
        retention=LOG_CONFIG["retention"],
    )

    # Error-only file
    logger.add(
        sink=log_dir / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}\n{exception}",
        level="ERROR",
        rotation="10 MB",
        retention="1 week",
    )

    return logger


class ProductionPipeline:
    """
    Production-ready pipeline with all optimizations.
    """

    def __init__(self):
        self.pipeline = None
        self.runner = None
        self.task = None

        # Monitoring
        self.gpu_monitor = None
        self.gpu_optimizer = None
        self.latency_optimizer = None
        self.error_handler = None
        self.performance_logger = None

        # Stats
        self.start_time = None
        self.conversation_count = 0

    async def setup(self):
        """Setup the production pipeline."""
        logger.info("=" * 70)
        logger.info("KONKANI VOICE AGENT - PRODUCTION MODE")
        logger.info("=" * 70)
        logger.info(f"Environment: {CURRENT_CONFIG['environment']}")
        logger.info(f"Target Latency: {PIPELINE_CONFIG['latency_target_ms']}ms")
        logger.info("")

        # Check environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not set! Exiting.")
            return False

        # Check STT service
        stt_url = os.getenv("STT_SERVICE_URL", "http://localhost:50051")
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{stt_url}/health", timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"✓ STT service ready at {stt_url}")
                    else:
                        logger.error(
                            f"✗ STT service unavailable (status {resp.status})"
                        )
                        return False
        except Exception as e:
            logger.error(f"✗ Cannot connect to STT service: {e}")
            return False

        # Initialize GPU monitoring
        if MEMORY_CONFIG["monitoring_enabled"]:
            self.gpu_monitor = GPUMemoryMonitor(
                device=GPU_CONFIG["stt_device"],
                check_interval=MEMORY_CONFIG["check_interval"],
            )
            self.gpu_optimizer = GPUMemoryOptimizer(self.gpu_monitor)
            self.gpu_monitor.start()
            logger.info("✓ GPU monitoring started")

        # Initialize latency optimizer
        self.latency_optimizer = LatencyOptimizer(
            target_latency_ms=PIPELINE_CONFIG["latency_target_ms"],
            warmup_on_start=True,
            enable_streaming=PIPELINE_CONFIG["enable_streaming"],
        )

        # Initialize error handler
        self.error_handler = ErrorHandler(
            max_retries=ERROR_CONFIG["max_retries"],
            base_retry_delay=ERROR_CONFIG["base_retry_delay"],
            enable_fallback_audio=ERROR_CONFIG["enable_fallback_audio"],
        )

        # Initialize performance logger
        self.performance_logger = PerformanceLogger()

        # Build pipeline
        await self._build_pipeline(stt_url, gemini_api_key)

        return True

    async def _build_pipeline(self, stt_url: str, gemini_api_key: str):
        """Build the complete pipeline."""

        # Configure audio transport
        transport_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=AUDIO_CONFIG["sample_rate"],
            audio_in_channels=AUDIO_CONFIG["channels"],
            audio_out_enabled=True,
            audio_out_sample_rate=TTS_CONFIG["sample_rate"],
            audio_out_channels=1,
        )

        transport = LocalAudioTransport(params=transport_params)

        # Build processors
        logger.info("Building pipeline processors...")

        # 1. VAD Processor
        vad_processor = SileroVADProcessor(
            threshold=VAD_CONFIG["threshold"],
            min_speech_duration_ms=VAD_CONFIG["min_speech_duration_ms"],
            min_silence_duration_ms=VAD_CONFIG["min_silence_duration_ms"],
            sample_rate=AUDIO_CONFIG["sample_rate"],
        )

        # 2. Barge-In Handler
        barge_in = BargeInHandler(
            vad_threshold=BARGE_IN_CONFIG["vad_threshold"],
            min_interrupt_duration_ms=BARGE_IN_CONFIG["min_interrupt_duration_ms"],
            debounce_ms=BARGE_IN_CONFIG["debounce_ms"],
            enable_echo_cancellation=BARGE_IN_CONFIG["enable_echo_cancellation"],
            sample_rate=AUDIO_CONFIG["sample_rate"],
        )

        # 3. STT Client
        stt_client = STTClientProcessor(
            service_url=stt_url,
            sample_rate=AUDIO_CONFIG["sample_rate"],
        )

        # 4. LLM Processor
        llm_processor = GeminiProcessor(
            api_key=gemini_api_key,
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
        )

        # 5. Enhanced TTS Processor
        tts_processor = EnhancedTTSProcessor(
            primary_model=TTS_CONFIG["primary_model"],
            fallback_model=TTS_CONFIG.get("fallback_model"),
            language=TTS_CONFIG["language"],
            device=TTS_CONFIG["device"],
            fp16=TTS_CONFIG["fp16"],
            sample_rate=TTS_CONFIG["sample_rate"],
            warmup=TTS_CONFIG["warmup"],
            enable_fallback_audio=TTS_CONFIG["enable_fallback_audio"],
        )

        # Build pipeline
        self.pipeline = Pipeline(
            [
                # Input
                transport.input(),
                # Error handling
                self.error_handler,
                # Latency monitoring
                self.latency_optimizer,
                # VAD + Barge-in
                vad_processor,
                barge_in,
                # STT
                stt_client,
                # LLM
                llm_processor,
                # TTS
                tts_processor,
                # Output
                transport.output(),
            ]
        )

        # Create task and runner
        self.task = PipelineTask(self.pipeline)
        self.runner = PipelineRunner()

        logger.info("✓ Pipeline built successfully")
        logger.info("")
        logger.info("Pipeline components:")
        logger.info("  1. LocalAudioTransport (Input)")
        logger.info("  2. ErrorHandler")
        logger.info("  3. LatencyOptimizer")
        logger.info("  4. SileroVAD")
        logger.info("  5. BargeInHandler")
        logger.info("  6. STTClient (HTTP)")
        logger.info("  7. Gemini LLM")
        logger.info("  8. EnhancedTTS (XTTSv2)")
        logger.info("  9. LocalAudioTransport (Output)")
        logger.info("")

    async def run(self):
        """Run the pipeline."""
        if not self.pipeline:
            logger.error("Pipeline not initialized")
            return

        self.start_time = asyncio.get_event_loop().time()

        logger.info("=" * 70)
        logger.info("PIPELINE STARTED")
        logger.info("=" * 70)
        logger.info("Ready for conversations")
        logger.info("Speak into your microphone")
        logger.info("Press Ctrl+C to stop")
        logger.info("")

        try:
            await self.runner.run(self.task)
        except KeyboardInterrupt:
            logger.info("\n✓ Pipeline stopped by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("")
        logger.info("Shutting down...")

        # Stop monitoring
        if self.gpu_monitor:
            self.gpu_monitor.stop()
            logger.info("✓ GPU monitoring stopped")

        # Save performance metrics
        if self.performance_logger:
            self.performance_logger.save_metrics()

        # Print summary
        if self.latency_optimizer:
            self.latency_optimizer.print_summary()

        if self.gpu_monitor:
            summary = self.gpu_monitor.get_summary()
            logger.info(f"GPU Memory - Peak: {summary.get('peak_mb', 0):.1f}MB")

        if self.error_handler:
            error_summary = self.error_handler.get_error_summary()
            logger.info(f"Errors - Total: {error_summary.get('total', 0)}")

        # Cleanup processors
        if self.pipeline:
            for processor in self.pipeline.processors:
                if hasattr(processor, "cleanup"):
                    await processor.cleanup()

        logger.info("✓ Shutdown complete")

    def print_final_stats(self):
        """Print final statistics."""
        if self.start_time:
            uptime = asyncio.get_event_loop().time() - self.start_time
            logger.info("")
            logger.info("=" * 70)
            logger.info("FINAL STATISTICS")
            logger.info("=" * 70)
            logger.info(f"Uptime: {uptime:.1f}s")
            logger.info(f"Conversations: {self.conversation_count}")
            logger.info("")


async def main():
    """Main entry point."""
    setup_production_logging()

    pipeline = ProductionPipeline()

    if await pipeline.setup():
        await pipeline.run()
        pipeline.print_final_stats()
    else:
        logger.error("Setup failed - exiting")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
