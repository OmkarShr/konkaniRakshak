"""Gemini LLM Processor for Pipecat

Language Model using Google Gemini API with streaming support.
"""

import os
from typing import List, Dict, Any
from loguru import logger

from pipecat.frames.frames import (
    TextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class GeminiProcessor(FrameProcessor):
    """
    Gemini LLM processor with streaming output.

    Uses Google Gemini API for Konkani conversational responses.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str = None,
    ):
        super().__init__()

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Default system prompt for Konkani
        self.system_prompt = system_prompt or (
            "तुम्ही एक सहाय्यक आहात जो फक्त कोकणी भाषेत (देवनागरी लिपीत) बोलतो. "
            "तुम्ही गोवा पोलिसांसाठी एफआयआर दाखल करण्यात मदत करता. "
            "कृपया नेहमी कोकणी भाषेत उत्तर द्या. इंग्रजी किंवा इतर भाषा वापरू नका."
        )

        self.client = None
        self.conversation_history: List[Dict[str, Any]] = []

        logger.info(f"GeminiProcessor initialized")
        logger.info(f"  Model: {model}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Max tokens: {max_tokens}")

    async def start(self, frame):
        """Initialize Gemini client."""
        await super().start(frame)
        self._init_client()

    def _init_client(self):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai

            logger.info("Initializing Gemini client...")

            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)

            logger.info("✓ Gemini client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    async def process_frame(self, frame, direction):
        """Process text and generate response."""

        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return

        user_message = frame.text
        logger.info(f"LLM input: {user_message[:50]}...")

        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_message})

            # Build context for Gemini
            messages = []

            # Add system prompt
            messages.append({"role": "user", "parts": [self.system_prompt]})
            messages.append(
                {"role": "model", "parts": ["मी समजलो. मी फक्त कोकणी भाषेत बोलेल."]}
            )

            # Add conversation history
            for msg in self.conversation_history:
                role = "user" if msg["role"] == "user" else "model"
                messages.append({"role": role, "parts": [msg["content"]]})

            # Generate response with streaming
            response = self.client.generate_content(
                messages,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
                stream=True,
            )

            # Collect full response
            full_response = ""

            # Stream tokens
            for chunk in response:
                if chunk.text:
                    text_chunk = chunk.text
                    full_response += text_chunk

                    # Emit text frame for each chunk
                    text_frame = TextFrame(text=text_chunk)
                    await self.push_frame(text_frame)

            # Add assistant response to history
            self.conversation_history.append(
                {"role": "assistant", "content": full_response}
            )

            logger.info(f"LLM output: {full_response[:50]}...")

            # Signal end of response
            await self.push_frame(LLMFullResponseEndFrame())

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            import traceback

            logger.error(traceback.format_exc())

            # Emit error message in Konkani
            error_frame = TextFrame(text="क्षमस्व, मला समजलं नाही. कृपया पुन्हा सांगा.")
            await self.push_frame(error_frame)
            await self.push_frame(LLMFullResponseEndFrame())

        # Pass original frame downstream
        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        self.conversation_history = []
        logger.info("GeminiProcessor cleaned up")
