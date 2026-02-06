"""Test Audio I/O - Phase 1 Deliverable

This script tests that audio input and output are working correctly
with the Pipecat LocalAudioTransport.
"""

import asyncio
import numpy as np
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import AudioRawFrame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.transports.base_transport import TransportParams


class AudioEchoProcessor(FrameProcessor):
    """Simple echo processor - plays back received audio."""

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.frames_received = 0
        logger.info("AudioEchoProcessor initialized")

    async def process_frame(self, frame, direction):
        if isinstance(frame, (AudioRawFrame, InputAudioRawFrame)):
            self.frames_received += 1

            # Log every 50 frames (~1 second)
            if self.frames_received % 50 == 0:
                audio_data = np.frombuffer(frame.audio, dtype=np.int16)
                amplitude = np.abs(audio_data).mean()
                logger.info(
                    f"Received {self.frames_received} frames | "
                    f"Amplitude: {amplitude:.0f} | "
                    f"Sample rate: {frame.sample_rate}Hz"
                )

            # Echo back the audio
            await self.push_frame(frame, direction)


async def test_audio_io():
    """Test audio input/output."""
    logger.info("=" * 60)
    logger.info("Audio I/O Test - Phase 1")
    logger.info("=" * 60)
    logger.info("Speak into your microphone - you should hear your voice echoed.")
    logger.info("Press Ctrl+C to stop.")
    logger.info("")

    try:
        # Configure transport parameters
        transport_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_channels=1,
        )

        # Set up transport
        transport = LocalAudioTransport(params=transport_params)

        # Simple pipeline: mic -> echo -> speaker
        pipeline = Pipeline(
            [
                transport.input(),
                AudioEchoProcessor(sample_rate=16000),
                transport.output(),
            ]
        )

        # Create task and runner
        task = PipelineTask(pipeline)
        runner = PipelineRunner()

        logger.info("Starting audio test...")
        logger.info("")

        # Run for 30 seconds or until interrupted
        await asyncio.wait_for(runner.run(task), timeout=30.0)

    except asyncio.TimeoutError:
        logger.info("\n✓ Audio test completed successfully (30s timeout)")
        return True
    except KeyboardInterrupt:
        logger.info("\n✓ Audio test stopped by user")
        return True
    except Exception as e:
        logger.error(f"\n✗ Audio test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = asyncio.run(test_audio_io())
    exit(0 if success else 1)
