"""STT Client Processor for Pipecat

Calls the standalone NeMo STT HTTP service.
This runs in the Pipecat pipeline and has NO NeMo dependencies.
"""

import asyncio
import base64
from typing import Optional
import aiohttp
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    InputAudioRawFrame,
    TextFrame,
    UserStoppedSpeakingFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class STTClientProcessor(FrameProcessor):
    """
    Client for the standalone NeMo STT HTTP service.
    
    This processor has NO NeMo dependencies - it just makes HTTP calls
to the separate STT service process.
    """
    
    def __init__(
        self,
        service_url: str = "http://localhost:50051",
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.service_url = service_url.rstrip("/")
        self.sample_rate = sample_rate
        self.audio_buffer = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_processing = False
        
        logger.info(f"STTClientProcessor initialized")
        logger.info(f"  Service URL: {service_url}")
        logger.info(f"  Sample rate: {sample_rate}Hz")
    
    async def start(self, frame: StartFrame):
        """Initialize HTTP session."""
        await super().start(frame)
        self.session = aiohttp.ClientSession()
        
        # Health check
        try:
            async with self.session.get(f"{self.service_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✓ STT service connected: {data}")
                else:
                    logger.warning(f"⚠ STT service health check failed: {resp.status}")
        except Exception as e:
            logger.error(f"✗ Cannot connect to STT service: {e}")
            logger.info("Make sure to run: python services/stt_service.py")
    
    async def process_frame(self, frame, direction):
        """Process frames - accumulate audio until speech ends."""
        
        # Accumulate audio frames
        if isinstance(frame, (AudioRawFrame, InputAudioRawFrame)):
            self.audio_buffer.append(frame.audio)
        
        # When speech ends, send to STT service
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self.audio_buffer and not self.is_processing:
                await self._transcribe_buffer()
        
        # Pass through all frames
        await self.push_frame(frame, direction)
    
    async def _transcribe_buffer(self):
        """Send audio buffer to STT service and get transcription."""
        if not self.audio_buffer or self.session is None:
            return
        
        self.is_processing = True
        
        try:
            # Concatenate all audio
            audio_data = b"".join(self.audio_buffer)
            self.audio_buffer = []
            
            logger.info(f"Sending {len(audio_data)} bytes to STT service...")
            
            # Encode as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Make HTTP request
            payload = {
                "audio": audio_b64,
                "sample_rate": self.sample_rate
            }
            
            async with self.session.post(
                f"{self.service_url}/transcribe",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    
                    if result.get("success") and result.get("text"):
                        text = result["text"]
                        logger.info(f"STT: {text[:50]}..." if len(text) > 50 else f"STT: {text}")
                        
                        # Create and push text frame
                        text_frame = TextFrame(text=text)
                        await self.push_frame(text_frame)
                    else:
                        logger.warning(f"STT returned empty or failed: {result}")
                else:
                    error_text = await resp.text()
                    logger.error(f"STT service error {resp.status}: {error_text}")
        
        except asyncio.TimeoutError:
            logger.error("STT service timeout (>30s)")
        
        except Exception as e:
            logger.error(f"STT client error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            self.is_processing = False
    
    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        
        if self.session:
            await self.session.close()
        
        logger.info("STTClientProcessor cleaned up")
