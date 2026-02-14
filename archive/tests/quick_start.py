#!/usr/bin/env python3
"""
Quick Start Script for Konkani Voice Agent

Run this directly without module import issues.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Check environment
print("=" * 60)
print("Konkani Voice Agent - Quick Start")
print("=" * 60)

# Check model
model_path = Path("models/indicconformer_stt_kok_hybrid_rnnt_large.nemo")
if not model_path.exists():
    print("\n❌ STT model not found!")
    print("   Download with:")
    print("   python quick_download_model.py")
    sys.exit(1)
else:
    print(f"✓ STT model found ({model_path.stat().st_size / (1024**2):.1f} MB)")

# Check API key
if not os.getenv("GEMINI_API_KEY"):
    from dotenv import load_dotenv

    load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("\n❌ GEMINI_API_KEY not set!")
    print("   Set with: export GEMINI_API_KEY=your_key")
    print("   Or add to .env file")
    sys.exit(1)
else:
    print("✓ GEMINI_API_KEY configured")

# Check STT service
print("\n⚠️  Before running, start the STT service:")
print("   Terminal 1: ./start_stt_service.sh")
print("   Then run this script in Terminal 2")
print()

response = input("Is STT service running? (y/n): ")
if response.lower() != "y":
    print("\nStart STT service first, then run this script again.")
    sys.exit(0)

print("\n" + "=" * 60)
print("Starting Pipeline...")
print("=" * 60)
print()

# Import and run
try:
    from konkani_agent.main import main
    import asyncio

    asyncio.run(main())
except KeyboardInterrupt:
    print("\n\n✓ Stopped by user")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback

    traceback.print_exc()
