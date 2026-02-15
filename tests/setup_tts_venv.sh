#!/usr/bin/env bash
# ============================================================
#  Setup a standalone .venv for Testing English TTS
#  Run from the repo root:  bash tests/setup_tts_venv.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "============================================"
echo "  English TTS Test — Virtual Environment Setup"
echo "============================================"

# 1. Create the venv
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  .venv already exists at $VENV_DIR — skipping creation"
else
    echo "📦 Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Activate
source "$VENV_DIR/bin/activate"
echo "✅ Activated: $(which python3)"

# 3. Upgrade pip
pip install --upgrade pip -q

# 4. Install PyTorch with CUDA 12.1
echo ""
echo "🔧 Installing PyTorch (CUDA 12.1) ..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# 5. Install TTS dependencies
echo ""
echo "🔧 Installing parler-tts + transformers ..."
pip install parler-tts transformers accelerate soundfile scipy numpy -q

echo ""
echo "============================================"
echo "  ✅ Setup complete!"
echo ""
echo "  To activate:  source tests/.venv/bin/activate"
echo "  To run test:   python tests/test_tts_english.py"
echo "============================================"
