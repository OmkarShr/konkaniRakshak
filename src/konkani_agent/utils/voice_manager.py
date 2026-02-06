#!/usr/bin/env python3
"""Voice Configuration Manager

Manages TTS voice settings, speaker profiles, and voice switching.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict
from loguru import logger


@dataclass
class VoiceProfile:
    """Voice profile configuration."""

    name: str
    description: str
    speaker_id: str
    speed: float = 1.0
    pitch: float = 1.0
    language: str = "mr"  # Marathi for Konkani


class VoiceManager:
    """Manages voice profiles and TTS configuration."""

    # Predefined voice profiles for AI4Bharat Parler-TTS
    VOICES: Dict[str, VoiceProfile] = {
        "female_clear": VoiceProfile(
            name="female_clear",
            description="Clear female voice for official communication",
            speaker_id="female_1",
            speed=1.0,
            pitch=1.0,
        ),
        "female_warm": VoiceProfile(
            name="female_warm",
            description="Warm, friendly female voice",
            speaker_id="female_2",
            speed=0.95,
            pitch=1.02,
        ),
        "male_clear": VoiceProfile(
            name="male_clear",
            description="Clear male voice for official communication",
            speaker_id="male_1",
            speed=1.0,
            pitch=0.98,
        ),
        "male_authoritative": VoiceProfile(
            name="male_authoritative",
            description="Authoritative male voice for police station",
            speaker_id="male_2",
            speed=0.9,
            pitch=0.95,
        ),
    }

    def __init__(self, default_voice: str = "female_clear"):
        """Initialize voice manager.

        Args:
            default_voice: Default voice profile name
        """
        self.current_voice = default_voice
        self._validate_voice(default_voice)
        logger.info(f"VoiceManager initialized with voice: {default_voice}")

    def _validate_voice(self, voice_name: str) -> None:
        """Validate voice name exists."""
        if voice_name not in self.VOICES:
            available = ", ".join(self.VOICES.keys())
            raise ValueError(f"Unknown voice '{voice_name}'. Available: {available}")

    def get_voice(self, voice_name: Optional[str] = None) -> VoiceProfile:
        """Get voice profile by name.

        Args:
            voice_name: Voice profile name (uses current if None)

        Returns:
            VoiceProfile configuration
        """
        name = voice_name or self.current_voice
        self._validate_voice(name)
        return self.VOICES[name]

    def set_voice(self, voice_name: str) -> None:
        """Set current voice profile.

        Args:
            voice_name: Voice profile name to use
        """
        self._validate_voice(voice_name)
        self.current_voice = voice_name
        logger.info(f"Voice switched to: {voice_name}")

    def list_voices(self) -> Dict[str, str]:
        """List all available voices with descriptions.

        Returns:
            Dictionary of voice names and descriptions
        """
        return {name: profile.description for name, profile in self.VOICES.items()}

    def get_voice_prompt(self, voice_name: Optional[str] = None) -> str:
        """Get voice description prompt for TTS.

        Args:
            voice_name: Voice profile name

        Returns:
            Voice description string for model
        """
        profile = self.get_voice(voice_name)
        return f"A {profile.name.replace('_', ' ')} speaker speaks in Marathi"


# Global voice manager instance
voice_manager = VoiceManager(default_voice=os.getenv("TTS_VOICE", "female_clear"))
