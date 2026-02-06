"""
Konkani Conversational AI Agent

A real-time voice agent for FIR filing assistance in Konkani language.
Built with Pipecat framework.

Usage:
    python -m konkani_agent

For development testing, run:
    python tests/test_audio_io.py
"""

__version__ = "0.1.0"
__author__ = "Omkar - NagarRakshak Project"

from .main import main

__all__ = ["main"]
