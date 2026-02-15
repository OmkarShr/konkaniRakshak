#!/usr/bin/env bash
# ============================================================
#  Setup a standalone .venv for Testing English STT (NeMo)
#  Run from repo root:  bash tests/setup_stt_venv.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.stt_venv"

echo "============================================"
echo "  English STT Test — Virtual Environment Setup"
echo "============================================"

# 1. Create the venv
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  .stt_venv already exists at $VENV_DIR — skipping creation"
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

# 5. Install NeMo toolkit (ASR)
echo ""
echo "🔧 Installing NeMo toolkit ..."
pip install Cython -q
pip install nemo_toolkit[asr] -q

# 6. Install audio recording dependencies
echo ""
echo "🔧 Installing audio utilities ..."
pip install sounddevice soundfile numpy scipy -q

echo ""
echo "============================================"
echo "  ✅ Setup complete!"
echo ""
echo "  To activate:  source tests/.stt_venv/bin/activate"
echo "  To run test:   CUDA_VISIBLE_DEVICES=1 python tests/test_stt_english.py"
echo "============================================"
