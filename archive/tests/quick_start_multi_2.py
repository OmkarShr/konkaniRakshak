#!/usr/bin/env python3
"""
Quick Start Script for Multilingual Voice Agent (Hindi/English)

Run this directly without module import issues.
"""

import os
import sys
from pathlib import Path
import asyncio

# Add src to path so we can find the agent modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Check environment
print("=" * 60)
print("Multilingual Voice Agent (Hindi/English) - Quick Start")
print("=" * 60)

# 1. Check for the 600M Model
# Note: Ensure this matches the exact filename you downloaded
model_filename = "indicconformer_stt_multi_hybrid_rnnt_600m.nemo"
model_path = Path(f"models/{model_filename}")

if not model_path.exists():
    # Fallback check for the HF Hub name just in case
    fallback_path = Path("models/indic-conformer-600m-multilingual.nemo")
    if fallback_path.exists():
        model_path = fallback_path
    else:
        print(f"\n❌ Multilingual STT model not found: {model_filename}")
        print("   Please run the download script first.")
        sys.exit(1)

print(f"✓ Multilingual STT model found ({model_path.stat().st_size / (1024**2):.1f} MB)")

# 2. Check API Key
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

# 3. Service Check
print("\n⚠️  Before running, ensure the Multilingual STT service is active.")
print("   (If you are using a unified server, make sure it is running)")
print()

response = input("Is the STT service running? (y/n): ")
if response.lower() != "y":
    print("\nStart the STT service first, then run this script again.")
    sys.exit(0)

print("\n" + "=" * 60)
print("Starting Multilingual Pipeline...")
print("=" * 60)
print()

# 4. Import and Run
try:
    # Assuming your folder structure mirrors the Konkani one:
    # src/multilingual_agent/main.py
    from multilingual_agent.main import main
    
    asyncio.run(main())

except ModuleNotFoundError as e:
    print(f"\n❌ Import Error: {e}")
    print("   Make sure you have created the folder 'src/multilingual_agent/'")
    print("   and it contains an __init__.py and main.py")
except KeyboardInterrupt:
    print("\n\n✓ Stopped by user")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()