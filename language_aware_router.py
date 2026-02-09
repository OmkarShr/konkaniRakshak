#!/usr/bin/env python3
"""
Language-Aware Speech Router for Goa Police Voice Portal
=========================================================

A production-ready skeleton that intelligently routes incoming audio streams
to one of two AI4Bharat ASR models based on real-time language detection.

Architecture:
    Audio Stream → AudioBufferManager → LanguageDiscriminator → Queue A/B → ModelWorker

Models:
    - Konkani Specialist: indicconformer_stt_kok_hybrid_rnnt_large.nemo
    - General Multilingual: indic-conformer-600m-multilingual

Author: Goa Police Voice Portal Team
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any
from collections import deque

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
)
logger = logging.getLogger("SpeechRouter")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RouterConfig:
    """Immutable configuration for the speech router."""
    
    # Audio settings
    sample_rate: int = 16000
    chunk_duration_ms: int = 50  # Each incoming chunk
    buffer_window_ms: int = 500  # Rolling buffer for language detection
    
    # Language detection thresholds
    konkani_confidence_threshold: float = 0.8
    
    # Sticky routing settings (prevents jitter)
    sticky_duration_seconds: float = 10.0
    
    # Model paths (AI4Bharat NeMo models)
    konkani_model_path: str = "models/indicconformer_stt_kok_hybrid_rnnt_large.nemo"
    multilingual_model_path: str = "models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"
    
    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)
    
    @property
    def buffer_samples(self) -> int:
        return int(self.sample_rate * self.buffer_window_ms / 1000)


class RoutingDecision(Enum):
    """Enum for routing decisions."""
    KONKANI_SPECIALIST = "konkani"
    GENERAL_MULTILINGUAL = "general"
    PENDING = "pending"


# =============================================================================
# AUDIO BUFFER MANAGER
# =============================================================================

class AudioBufferManager:
    """
    Manages a rolling audio buffer for language detection.
    
    Ingests raw audio chunks (simulating WebSocket stream) and maintains
    a fixed-size rolling window for analysis.
    
    Attributes:
        config: Router configuration
        buffer: Rolling numpy buffer
        write_pos: Current write position in circular buffer
        samples_received: Total samples received (for statistics)
    """
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self._buffer = np.zeros(config.buffer_samples, dtype=np.float32)
        self._write_pos = 0
        self._samples_received = 0
        self._is_buffer_full = False
        self._lock = asyncio.Lock()
        
        logger.info(
            f"AudioBufferManager initialized: "
            f"buffer_size={config.buffer_samples} samples "
            f"({config.buffer_window_ms}ms)"
        )
    
    async def ingest_chunk(self, chunk: np.ndarray) -> None:
        """
        Ingest an audio chunk into the rolling buffer.
        
        Args:
            chunk: Audio samples as float32 numpy array (normalized -1 to 1)
        
        Raises:
            ValueError: If chunk has wrong dtype or dimensions
        """
        if chunk.ndim != 1:
            raise ValueError(f"Expected 1D audio chunk, got shape {chunk.shape}")
        
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        
        async with self._lock:
            chunk_len = len(chunk)
            buffer_len = len(self._buffer)
            
            # Handle chunks larger than buffer (truncate to most recent)
            if chunk_len >= buffer_len:
                self._buffer[:] = chunk[-buffer_len:]
                self._write_pos = 0
                self._is_buffer_full = True
            else:
                # Circular buffer write
                end_pos = self._write_pos + chunk_len
                
                if end_pos <= buffer_len:
                    self._buffer[self._write_pos:end_pos] = chunk
                else:
                    # Wrap around
                    first_part = buffer_len - self._write_pos
                    self._buffer[self._write_pos:] = chunk[:first_part]
                    self._buffer[:chunk_len - first_part] = chunk[first_part:]
                
                self._write_pos = end_pos % buffer_len
            
            self._samples_received += chunk_len
            
            if self._samples_received >= buffer_len:
                self._is_buffer_full = True
    
    async def get_buffer_snapshot(self) -> Optional[np.ndarray]:
        """
        Get a copy of the current buffer in chronological order.
        
        Returns:
            Numpy array of audio samples, or None if buffer not yet full
        """
        async with self._lock:
            if not self._is_buffer_full:
                return None
            
            # Reconstruct chronological order from circular buffer
            if self._write_pos == 0:
                return self._buffer.copy()
            else:
                return np.concatenate([
                    self._buffer[self._write_pos:],
                    self._buffer[:self._write_pos]
                ])
    
    async def clear(self) -> None:
        """Reset the buffer to empty state."""
        async with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._samples_received = 0
            self._is_buffer_full = False
    
    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough data for language detection."""
        return self._is_buffer_full
    
    @property
    def samples_received(self) -> int:
        return self._samples_received


# =============================================================================
# LANGUAGE DISCRIMINATOR (ROUTER)
# =============================================================================

@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    detected_language: str
    confidence: float
    audio_duration_ms: float
    routing_decision: RoutingDecision


class LanguageDiscriminator:
    """
    Lightweight language detection and routing logic.
    
    Analyzes audio buffer to determine if content is Konkani or other language,
    then routes to appropriate ASR model queue.
    
    Integration Point:
        In production, replace `_detect_language_impl` with actual
        language identification model (e.g., AI4Bharat LID or SpeechBrain).
    """
    
    def __init__(
        self,
        config: RouterConfig,
        konkani_queue: asyncio.Queue,
        general_queue: asyncio.Queue
    ):
        self.config = config
        self.konkani_queue = konkani_queue
        self.general_queue = general_queue
        self._detection_count = 0
        
        logger.info(
            f"LanguageDiscriminator initialized: "
            f"konkani_threshold={config.konkani_confidence_threshold}"
        )
    
    async def detect_language(self, audio_buffer: np.ndarray) -> LanguageDetectionResult:
        """
        Detect language from audio buffer.
        
        Args:
            audio_buffer: Audio samples as numpy array
            
        Returns:
            LanguageDetectionResult with detected language and confidence
        
        Note:
            This is an async function to support future integration with
            GPU-based language ID models that may use async inference.
        """
        self._detection_count += 1
        
        # Calculate audio duration
        duration_ms = len(audio_buffer) / self.config.sample_rate * 1000
        
        # =====================================================================
        # INTEGRATION POINT: Language Identification Model
        # =====================================================================
        # In production, replace this mock with actual LID model call:
        #
        # from nemo.collections.asr.models import EncDecSpeakerLabelModel
        # lid_model = EncDecSpeakerLabelModel.from_pretrained("langid_model")
        # with torch.no_grad():
        #     logits = lid_model.forward(audio_buffer)
        #     probs = torch.softmax(logits, dim=-1)
        #     language, confidence = decode_lid_output(probs)
        # =====================================================================
        
        language, confidence = await self._detect_language_impl(audio_buffer)
        
        # Make routing decision
        if language == "konkani" and confidence >= self.config.konkani_confidence_threshold:
            decision = RoutingDecision.KONKANI_SPECIALIST
        else:
            decision = RoutingDecision.GENERAL_MULTILINGUAL
        
        return LanguageDetectionResult(
            detected_language=language,
            confidence=confidence,
            audio_duration_ms=duration_ms,
            routing_decision=decision
        )
    
    async def _detect_language_impl(
        self, 
        audio_buffer: np.ndarray
    ) -> tuple[str, float]:
        """
        Mock language detection implementation.
        
        Simulates language detection with random results weighted toward
        Konkani for testing the routing logic.
        
        Args:
            audio_buffer: Audio samples
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        # Simulate async inference delay (real model would be ~50-100ms)
        await asyncio.sleep(0.02)
        
        # Mock: Use audio energy as a pseudo-feature
        energy = np.sqrt(np.mean(audio_buffer ** 2))
        
        # Simulate detection with weighted randomness
        # In reality, this would be actual model inference
        import random
        
        if random.random() < 0.6:  # 60% Konkani for testing
            language = "konkani"
            confidence = 0.75 + random.random() * 0.25  # 0.75-1.0
        else:
            languages = ["hindi", "marathi", "english", "kannada"]
            language = random.choice(languages)
            confidence = 0.5 + random.random() * 0.4  # 0.5-0.9
        
        return language, confidence
    
    async def route_audio(
        self,
        session_id: str,
        audio_buffer: np.ndarray,
        forced_decision: Optional[RoutingDecision] = None
    ) -> LanguageDetectionResult:
        """
        Detect language and route audio to appropriate queue.
        
        Args:
            session_id: Unique session identifier
            audio_buffer: Audio to route
            forced_decision: If set, skip detection and use this decision (for sticky routing)
            
        Returns:
            LanguageDetectionResult with routing details
        """
        if forced_decision:
            # Sticky routing - skip detection
            result = LanguageDetectionResult(
                detected_language="sticky",
                confidence=1.0,
                audio_duration_ms=len(audio_buffer) / self.config.sample_rate * 1000,
                routing_decision=forced_decision
            )
        else:
            result = await self.detect_language(audio_buffer)
        
        # Create work item
        work_item = {
            "session_id": session_id,
            "audio": audio_buffer,
            "timestamp": time.time(),
            "detection_result": result
        }
        
        # Route to appropriate queue
        if result.routing_decision == RoutingDecision.KONKANI_SPECIALIST:
            await self.konkani_queue.put(work_item)
            logger.debug(f"Session {session_id[:8]}: Routed to Konkani queue")
        else:
            await self.general_queue.put(work_item)
            logger.debug(f"Session {session_id[:8]}: Routed to General queue")
        
        return result


# =============================================================================
# MODEL WORKERS (ASR ENGINES)
# =============================================================================

class ModelNotLoadedError(Exception):
    """Raised when attempting inference on an unloaded model."""
    pass


@dataclass
class TranscriptionResult:
    """Result of ASR transcription."""
    text: str
    language: str
    confidence: float
    duration_ms: float
    model_name: str
    latency_ms: float


class ASRWorkerBase(ABC):
    """
    Abstract base class for ASR workers.
    
    Subclasses implement specific model loading and inference logic
    for different AI4Bharat models.
    
    Integration Points:
        - `_load_model_impl`: Load NeMo model from path
        - `_transcribe_impl`: Run inference on audio
    """
    
    def __init__(
        self,
        name: str,
        model_path: str,
        work_queue: asyncio.Queue,
        config: RouterConfig
    ):
        self.name = name
        self.model_path = model_path
        self.work_queue = work_queue
        self.config = config
        
        self._model: Any = None
        self._is_loaded = False
        self._is_running = False
        self._transcription_count = 0
        self._total_latency_ms = 0.0
        
        logger.info(f"ASRWorker '{name}' created (model: {model_path})")
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def average_latency_ms(self) -> float:
        if self._transcription_count == 0:
            return 0.0
        return self._total_latency_ms / self._transcription_count
    
    async def load_model(self) -> None:
        """Load the ASR model into memory."""
        if self._is_loaded:
            logger.warning(f"Model '{self.name}' already loaded")
            return
        
        logger.info(f"Loading model '{self.name}'...")
        
        try:
            self._model = await self._load_model_impl()
            self._is_loaded = True
            logger.info(f"Model '{self.name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model '{self.name}': {e}")
            raise
    
    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            # =========================================================
            # INTEGRATION POINT: Cleanup GPU memory
            # =========================================================
            # del self._model
            # torch.cuda.empty_cache()
            # =========================================================
            self._model = None
            self._is_loaded = False
            logger.info(f"Model '{self.name}' unloaded")
    
    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe audio using the loaded model.
        
        Args:
            audio: Audio samples as numpy array
            
        Returns:
            TranscriptionResult with transcribed text
            
        Raises:
            ModelNotLoadedError: If model is not loaded
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(
                f"Model '{self.name}' is not loaded. Call load_model() first."
            )
        
        start_time = time.perf_counter()
        
        try:
            text, confidence = await self._transcribe_impl(audio)
        except Exception as e:
            logger.error(f"Transcription failed for '{self.name}': {e}")
            raise
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        duration_ms = len(audio) / self.config.sample_rate * 1000
        
        self._transcription_count += 1
        self._total_latency_ms += latency_ms
        
        return TranscriptionResult(
            text=text,
            language=self._get_language(),
            confidence=confidence,
            duration_ms=duration_ms,
            model_name=self.name,
            latency_ms=latency_ms
        )
    
    async def run(self, result_callback: Callable[[str, TranscriptionResult], None]) -> None:
        """
        Run the worker loop, processing items from the queue.
        
        Args:
            result_callback: Called with (session_id, result) for each transcription
        """
        self._is_running = True
        logger.info(f"Worker '{self.name}' started")
        
        while self._is_running:
            try:
                # Wait for work with timeout to allow graceful shutdown
                try:
                    work_item = await asyncio.wait_for(
                        self.work_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                session_id = work_item["session_id"]
                audio = work_item["audio"]
                
                try:
                    result = await self.transcribe(audio)
                    result_callback(session_id, result)
                except ModelNotLoadedError:
                    logger.error(f"Worker '{self.name}': Model not loaded, skipping")
                except Exception as e:
                    logger.error(f"Worker '{self.name}': Transcription error: {e}")
                finally:
                    self.work_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker '{self.name}': Unexpected error: {e}")
        
        logger.info(f"Worker '{self.name}' stopped")
    
    async def stop(self) -> None:
        """Stop the worker loop."""
        self._is_running = False
    
    @abstractmethod
    async def _load_model_impl(self) -> Any:
        """Implementation-specific model loading."""
        pass
    
    @abstractmethod
    async def _transcribe_impl(self, audio: np.ndarray) -> tuple[str, float]:
        """Implementation-specific transcription."""
        pass
    
    @abstractmethod
    def _get_language(self) -> str:
        """Return the primary language this model handles."""
        pass


class KonkaniSpecialistWorker(ASRWorkerBase):
    """
    ASR Worker for Konkani language using indicconformer_stt_kok_hybrid_rnnt_large.
    
    This model is fine-tuned specifically for Konkani and provides
    superior accuracy for Konkani speech compared to general models.
    
    NeMo Integration:
        Model class: nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        Decoder: RNNT (preferred) or CTC
    """
    
    def __init__(self, work_queue: asyncio.Queue, config: RouterConfig):
        super().__init__(
            name="KonkaniSpecialist",
            model_path=config.konkani_model_path,
            work_queue=work_queue,
            config=config
        )
    
    async def _load_model_impl(self) -> Any:
        """
        Load the Konkani IndicConformer model.
        
        Production code:
            import nemo.collections.asr as nemo_asr
            model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
                self.model_path,
                map_location='cuda'
            )
            model.eval()
            model.cur_decoder = 'rnnt'
            return model
        """
        # Mock: Simulate model loading delay
        await asyncio.sleep(0.5)
        logger.info("  → [MOCK] Konkani model loaded (placeholder)")
        return {"type": "konkani_mock", "loaded": True}
    
    async def _transcribe_impl(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe Konkani audio.
        
        Production code:
            with torch.no_grad():
                result = self._model.transcribe([audio], batch_size=1)
            return result[0], 0.95
        """
        # Mock: Simulate inference delay (real would be ~100-300ms)
        await asyncio.sleep(0.1)
        
        # Return mock Konkani transcription
        mock_texts = [
            "हांव गोंयचो आसां",  # I am from Goa
            "पुलिसांक खबर दी",     # Inform the police
            "मजत करात",          # Please help
            "कितें जालें?",        # What happened?
        ]
        import random
        text = random.choice(mock_texts)
        confidence = 0.90 + random.random() * 0.1
        
        return text, confidence
    
    def _get_language(self) -> str:
        return "konkani"


class GeneralMultilingualWorker(ASRWorkerBase):
    """
    ASR Worker for general Indian languages using indic-conformer-600m-multilingual.
    
    This model handles multiple Indian languages including Hindi, Marathi,
    English, Kannada, and others. Used as fallback when Konkani is not detected.
    
    NeMo Integration:
        Model class: nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        Supports: 22+ Indian languages
    """
    
    def __init__(self, work_queue: asyncio.Queue, config: RouterConfig):
        super().__init__(
            name="GeneralMultilingual",
            model_path=config.multilingual_model_path,
            work_queue=work_queue,
            config=config
        )
    
    async def _load_model_impl(self) -> Any:
        """
        Load the multilingual IndicConformer model.
        
        Production code:
            import nemo.collections.asr as nemo_asr
            model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
                self.model_path,
                map_location='cuda'
            )
            model.eval()
            return model
        """
        await asyncio.sleep(0.5)
        logger.info("  → [MOCK] Multilingual model loaded (placeholder)")
        return {"type": "multilingual_mock", "loaded": True}
    
    async def _transcribe_impl(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe multilingual audio.
        
        Production code:
            with torch.no_grad():
                # Optionally specify language_id for better accuracy
                result = self._model.transcribe(
                    [audio],
                    batch_size=1,
                    language_id='hi'  # or auto-detect
                )
            return result[0], 0.90
        """
        await asyncio.sleep(0.1)
        
        # Return mock Hindi/English transcription
        mock_texts = [
            "मुझे मदद चाहिए",           # I need help (Hindi)
            "Please help me",            # English
            "पोलिस स्टेशन कहाँ है?",     # Where is police station (Hindi)
            "There has been an accident", # English
            "आपत्कालीन सेवा",           # Emergency service (Hindi)
        ]
        import random
        text = random.choice(mock_texts)
        confidence = 0.85 + random.random() * 0.1
        
        return text, confidence
    
    def _get_language(self) -> str:
        return "multilingual"


# =============================================================================
# SESSION MANAGER (STICKY ROUTING)
# =============================================================================

@dataclass
class SessionState:
    """State for a single user session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Sticky routing
    locked_model: Optional[RoutingDecision] = None
    lock_expires_at: Optional[float] = None
    
    # Statistics
    chunks_processed: int = 0
    konkani_routes: int = 0
    general_routes: int = 0


class SessionManager:
    """
    Manages user session state and implements sticky routing logic.
    
    Sticky Logic:
        Once a user is routed to the Konkani Specialist model, they are
        locked to that model for `sticky_duration_seconds` to prevent
        rapid switching (jitter) caused by noisy language detection.
    
    Benefits:
        - Reduces model switching overhead
        - Improves user experience with consistent transcription
        - Handles code-switching (Konkani-Hindi mix) gracefully
    """
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        
        logger.info(
            f"SessionManager initialized: "
            f"sticky_duration={config.sticky_duration_seconds}s"
        )
    
    async def get_or_create_session(self, session_id: str) -> SessionState:
        """Get existing session or create new one."""
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
                logger.info(f"New session created: {session_id[:8]}...")
            return self._sessions[session_id]
    
    async def update_session(
        self,
        session_id: str,
        routing_decision: RoutingDecision
    ) -> Optional[RoutingDecision]:
        """
        Update session with routing decision and check for sticky override.
        
        Args:
            session_id: Session to update
            routing_decision: The new routing decision from language detection
            
        Returns:
            If sticky routing is active, returns the locked decision.
            Otherwise returns None (use the provided decision).
        """
        session = await self.get_or_create_session(session_id)
        current_time = time.time()
        
        async with self._lock:
            session.last_activity = current_time
            session.chunks_processed += 1
            
            # Check if sticky lock is still active
            if session.locked_model and session.lock_expires_at:
                if current_time < session.lock_expires_at:
                    # Sticky routing active - use locked model
                    logger.debug(
                        f"Session {session_id[:8]}: Sticky routing to "
                        f"{session.locked_model.value} "
                        f"(expires in {session.lock_expires_at - current_time:.1f}s)"
                    )
                    return session.locked_model
                else:
                    # Lock expired
                    logger.info(f"Session {session_id[:8]}: Sticky lock expired")
                    session.locked_model = None
                    session.lock_expires_at = None
            
            # Apply new routing and potentially set sticky lock
            if routing_decision == RoutingDecision.KONKANI_SPECIALIST:
                session.konkani_routes += 1
                
                # Set sticky lock for Konkani
                session.locked_model = RoutingDecision.KONKANI_SPECIALIST
                session.lock_expires_at = current_time + self.config.sticky_duration_seconds
                
                logger.info(
                    f"Session {session_id[:8]}: Locked to Konkani for "
                    f"{self.config.sticky_duration_seconds}s"
                )
            else:
                session.general_routes += 1
            
            return None  # Use the provided decision
    
    async def get_session_stats(self, session_id: str) -> dict:
        """Get statistics for a session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return {}
            
            return {
                "session_id": session.session_id,
                "chunks_processed": session.chunks_processed,
                "konkani_routes": session.konkani_routes,
                "general_routes": session.general_routes,
                "is_locked": session.locked_model is not None,
                "locked_to": session.locked_model.value if session.locked_model else None,
                "session_duration_s": time.time() - session.created_at
            }
    
    async def cleanup_stale_sessions(self, max_idle_seconds: float = 300) -> int:
        """Remove sessions that have been idle too long."""
        current_time = time.time()
        removed = 0
        
        async with self._lock:
            stale_ids = [
                sid for sid, session in self._sessions.items()
                if current_time - session.last_activity > max_idle_seconds
            ]
            
            for sid in stale_ids:
                del self._sessions[sid]
                removed += 1
        
        if removed:
            logger.info(f"Cleaned up {removed} stale sessions")
        
        return removed


# =============================================================================
# MAIN SPEECH ROUTER ORCHESTRATOR
# =============================================================================

class LanguageAwareSpeechRouter:
    """
    Main orchestrator that ties all components together.
    
    Usage:
        router = LanguageAwareSpeechRouter()
        await router.start()
        
        # Process audio from WebSocket
        async for chunk in websocket:
            await router.process_audio(session_id, chunk)
        
        await router.stop()
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        
        # Create queues
        self.konkani_queue: asyncio.Queue = asyncio.Queue()
        self.general_queue: asyncio.Queue = asyncio.Queue()
        
        # Create components
        self.session_manager = SessionManager(self.config)
        self.discriminator = LanguageDiscriminator(
            self.config,
            self.konkani_queue,
            self.general_queue
        )
        
        # Create workers
        self.konkani_worker = KonkaniSpecialistWorker(
            self.konkani_queue, 
            self.config
        )
        self.general_worker = GeneralMultilingualWorker(
            self.general_queue,
            self.config
        )
        
        # Per-session audio buffers
        self._buffers: dict[str, AudioBufferManager] = {}
        self._worker_tasks: list[asyncio.Task] = []
        self._results: list[tuple[str, TranscriptionResult]] = []
        
        logger.info("LanguageAwareSpeechRouter initialized")
    
    def _on_transcription_result(
        self, 
        session_id: str, 
        result: TranscriptionResult
    ) -> None:
        """Callback for transcription results."""
        self._results.append((session_id, result))
        logger.info(
            f"[{result.model_name}] Session {session_id[:8]}: "
            f"\"{result.text}\" "
            f"(conf={result.confidence:.2f}, latency={result.latency_ms:.0f}ms)"
        )
    
    async def start(self) -> None:
        """Start the router and all workers."""
        logger.info("Starting LanguageAwareSpeechRouter...")
        
        # Load models
        await self.konkani_worker.load_model()
        await self.general_worker.load_model()
        
        # Start worker tasks
        self._worker_tasks = [
            asyncio.create_task(
                self.konkani_worker.run(self._on_transcription_result)
            ),
            asyncio.create_task(
                self.general_worker.run(self._on_transcription_result)
            )
        ]
        
        logger.info("LanguageAwareSpeechRouter started")
    
    async def stop(self) -> None:
        """Stop the router and all workers."""
        logger.info("Stopping LanguageAwareSpeechRouter...")
        
        # Stop workers
        await self.konkani_worker.stop()
        await self.general_worker.stop()
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Unload models
        await self.konkani_worker.unload_model()
        await self.general_worker.unload_model()
        
        logger.info("LanguageAwareSpeechRouter stopped")
    
    async def process_audio(
        self, 
        session_id: str, 
        audio_chunk: np.ndarray
    ) -> Optional[LanguageDetectionResult]:
        """
        Process an incoming audio chunk for a session.
        
        Args:
            session_id: Unique identifier for the user session
            audio_chunk: Raw audio samples
            
        Returns:
            LanguageDetectionResult if routing occurred, None if buffering
        """
        # Get or create buffer for this session
        if session_id not in self._buffers:
            self._buffers[session_id] = AudioBufferManager(self.config)
        
        buffer = self._buffers[session_id]
        await buffer.ingest_chunk(audio_chunk)
        
        # Check if buffer is ready for analysis
        if not buffer.is_ready:
            return None
        
        # Get buffer snapshot
        audio_snapshot = await buffer.get_buffer_snapshot()
        if audio_snapshot is None:
            return None
        
        # Check for sticky routing
        session = await self.session_manager.get_or_create_session(session_id)
        forced_decision = None
        
        if session.locked_model and session.lock_expires_at:
            if time.time() < session.lock_expires_at:
                forced_decision = session.locked_model
        
        # Route audio
        result = await self.discriminator.route_audio(
            session_id,
            audio_snapshot,
            forced_decision=forced_decision
        )
        
        # Update session state
        await self.session_manager.update_session(session_id, result.routing_decision)
        
        # Clear buffer after routing
        await buffer.clear()
        
        return result
    
    def get_results(self) -> list[tuple[str, TranscriptionResult]]:
        """Get all transcription results."""
        return self._results.copy()


# =============================================================================
# DEMONSTRATION / TESTING
# =============================================================================

async def simulate_audio_stream(
    router: LanguageAwareSpeechRouter,
    session_id: str,
    duration_seconds: float = 5.0,
    chunk_interval_ms: int = 50
) -> None:
    """
    Simulate an incoming audio stream for testing.
    
    Generates synthetic audio chunks and feeds them to the router.
    """
    config = router.config
    chunks_to_send = int(duration_seconds * 1000 / chunk_interval_ms)
    
    logger.info(
        f"Simulating {duration_seconds}s audio stream "
        f"({chunks_to_send} chunks) for session {session_id[:8]}..."
    )
    
    for i in range(chunks_to_send):
        # Generate synthetic audio chunk (white noise with varying amplitude)
        chunk = np.random.randn(config.chunk_samples).astype(np.float32) * 0.1
        
        # Add some variation to simulate speech patterns
        if i % 20 < 10:  # Simulate speech activity
            chunk *= 3.0
        
        result = await router.process_audio(session_id, chunk)
        
        if result:
            logger.info(
                f"  Chunk {i}: Routed to {result.routing_decision.value} "
                f"(lang={result.detected_language}, conf={result.confidence:.2f})"
            )
        
        await asyncio.sleep(chunk_interval_ms / 1000)


async def main():
    """Main demonstration function."""
    print("=" * 70)
    print("LANGUAGE-AWARE SPEECH ROUTER - Goa Police Voice Portal")
    print("=" * 70)
    print()
    
    # Create router with default config
    config = RouterConfig(
        sticky_duration_seconds=10.0,
        konkani_confidence_threshold=0.8
    )
    router = LanguageAwareSpeechRouter(config)
    
    try:
        # Start the router
        await router.start()
        print()
        
        # Simulate multiple concurrent sessions
        session1 = str(uuid.uuid4())
        session2 = str(uuid.uuid4())
        
        print(f"Session 1: {session1[:8]}...")
        print(f"Session 2: {session2[:8]}...")
        print()
        
        # Run simulations concurrently
        await asyncio.gather(
            simulate_audio_stream(router, session1, duration_seconds=3.0),
            simulate_audio_stream(router, session2, duration_seconds=3.0),
        )
        
        # Allow workers to process remaining items
        await asyncio.sleep(1.0)
        
        # Print session statistics
        print()
        print("=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)
        
        for sid in [session1, session2]:
            stats = await router.session_manager.get_session_stats(sid)
            print(f"\nSession {sid[:8]}:")
            print(f"  Chunks processed: {stats.get('chunks_processed', 0)}")
            print(f"  Konkani routes:   {stats.get('konkani_routes', 0)}")
            print(f"  General routes:   {stats.get('general_routes', 0)}")
            print(f"  Currently locked: {stats.get('locked_to', 'None')}")
        
        # Print all transcription results
        print()
        print("=" * 70)
        print("TRANSCRIPTION RESULTS")
        print("=" * 70)
        
        results = router.get_results()
        for session_id, result in results:
            print(
                f"\n[{result.model_name}] {session_id[:8]}: "
                f"\"{result.text}\""
            )
            print(
                f"  Language: {result.language}, "
                f"Confidence: {result.confidence:.2f}, "
                f"Latency: {result.latency_ms:.0f}ms"
            )
        
        # Print worker statistics
        print()
        print("=" * 70)
        print("WORKER STATISTICS")
        print("=" * 70)
        print(f"\nKonkani Worker:")
        print(f"  Transcriptions: {router.konkani_worker._transcription_count}")
        print(f"  Avg Latency:    {router.konkani_worker.average_latency_ms:.1f}ms")
        print(f"\nGeneral Worker:")
        print(f"  Transcriptions: {router.general_worker._transcription_count}")
        print(f"  Avg Latency:    {router.general_worker.average_latency_ms:.1f}ms")
        
    finally:
        # Clean shutdown
        await router.stop()
    
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


# =============================================================================
# PRODUCTION HTTP/WEBSOCKET SERVER (for Docker deployment)
# =============================================================================

import os
import json

# Global router instance for HTTP handlers
_router: Optional[LanguageAwareSpeechRouter] = None


async def run_production_server():
    """
    Run production HTTP/WebSocket server for Docker deployment.
    
    Endpoints:
        GET  /health  - Health check for Docker
        GET  /stats   - Router statistics
        WS   /ws      - WebSocket endpoint for audio streaming
    """
    global _router
    
    # Import aiohttp here to keep it optional for demo mode
    try:
        from aiohttp import web
    except ImportError:
        logger.error("aiohttp not installed. Run: pip install aiohttp")
        logger.info("Falling back to demo mode...")
        await main()
        return
    
    # Load config from environment
    config = RouterConfig(
        sticky_duration_seconds=float(os.environ.get("STICKY_DURATION", "10.0")),
        konkani_confidence_threshold=float(os.environ.get("KONKANI_THRESHOLD", "0.8")),
        konkani_model_path=os.environ.get(
            "KONKANI_MODEL_PATH",
            "/app/models/indicconformer_stt_kok_hybrid_rnnt_large.nemo"
        ),
        multilingual_model_path=os.environ.get(
            "MULTILINGUAL_MODEL_PATH",
            "/app/models/indic-conformer-600m-multilingual.nemo"
        )
    )
    
    _router = LanguageAwareSpeechRouter(config)
    
    # HTTP Handlers
    async def health_handler(request: web.Request) -> web.Response:
        """Health check endpoint for Docker."""
        return web.json_response({
            "status": "healthy",
            "service": "language-router",
            "models": {
                "konkani": _router.konkani_worker.is_loaded,
                "multilingual": _router.general_worker.is_loaded
            },
            "workers": {
                "konkani_running": _router.konkani_worker._is_running,
                "multilingual_running": _router.general_worker._is_running
            }
        })
    
    async def stats_handler(request: web.Request) -> web.Response:
        """Statistics endpoint."""
        return web.json_response({
            "konkani_worker": {
                "transcriptions": _router.konkani_worker._transcription_count,
                "avg_latency_ms": _router.konkani_worker.average_latency_ms,
                "queue_size": _router.konkani_queue.qsize()
            },
            "general_worker": {
                "transcriptions": _router.general_worker._transcription_count,
                "avg_latency_ms": _router.general_worker.average_latency_ms,
                "queue_size": _router.general_queue.qsize()
            },
            "active_sessions": len(_router._buffers)
        })
    
    async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
        """
        WebSocket endpoint for audio streaming.
        
        Protocol:
            1. Client connects and sends: {"session_id": "uuid"}
            2. Client sends binary audio frames (float32, 16kHz)
            3. Server sends back: {"transcript": "...", "model": "...", "confidence": 0.95}
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        session_id = None
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Parse JSON messages
                    try:
                        data = json.loads(msg.data)
                        if "session_id" in data:
                            session_id = data["session_id"]
                            logger.info(f"WebSocket session started: {session_id[:8]}...")
                            await ws.send_json({"status": "connected", "session_id": session_id})
                    except json.JSONDecodeError:
                        await ws.send_json({"error": "Invalid JSON"})
                
                elif msg.type == web.WSMsgType.BINARY:
                    # Process audio chunk
                    if session_id is None:
                        await ws.send_json({"error": "Send session_id first"})
                        continue
                    
                    # Convert bytes to numpy array
                    audio_chunk = np.frombuffer(msg.data, dtype=np.float32)
                    
                    # Process through router
                    result = await _router.process_audio(session_id, audio_chunk)
                    
                    if result:
                        # Send routing result
                        await ws.send_json({
                            "type": "routing",
                            "language": result.detected_language,
                            "confidence": result.confidence,
                            "routed_to": result.routing_decision.value
                        })
                
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
        
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        
        finally:
            if session_id:
                logger.info(f"WebSocket session ended: {session_id[:8]}...")
        
        return ws
    
    # Create and configure app
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stats", stats_handler)
    app.router.add_get("/ws", websocket_handler)
    
    # Startup/shutdown handlers
    async def on_startup(app: web.Application):
        logger.info("Starting Language-Aware Speech Router (Production Mode)...")
        await _router.start()
    
    async def on_shutdown(app: web.Application):
        logger.info("Shutting down...")
        await _router.stop()
    
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    
    # Run server
    host = os.environ.get("ROUTER_HOST", "0.0.0.0")
    port = int(os.environ.get("ROUTER_PORT", "50052"))
    
    logger.info(f"Starting server on {host}:{port}")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    logger.info(f"Server running at http://{host}:{port}")
    logger.info(f"  Health: http://{host}:{port}/health")
    logger.info(f"  Stats:  http://{host}:{port}/stats")
    logger.info(f"  WS:     ws://{host}:{port}/ws")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    import sys
    
    if "--demo" in sys.argv:
        # Run demo mode
        asyncio.run(main())
    else:
        # Run production server (default in Docker)
        asyncio.run(run_production_server())

