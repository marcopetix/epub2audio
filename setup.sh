#!/bin/bash
set -e

echo "=== epub2audio setup ==="

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 3. GPU acceleration (uncomment if you have NVIDIA GPU)
echo "Installing onnxruntime-gpu for CUDA acceleration..."
pip install onnxruntime-gpu>=1.17.0

# 4. Check ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "WARNING: ffmpeg not found. Install it:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  Mac: brew install ffmpeg"
fi

# 5. Download Kokoro model files from GitHub releases
echo "Downloading Kokoro TTS model files..."
KOKORO_BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
wget -nc "${KOKORO_BASE}/kokoro-v1.0.onnx" || true
wget -nc "${KOKORO_BASE}/voices-v1.0.bin" || true

echo ""
echo "Setup complete! Usage:"
echo "  source .venv/bin/activate"
echo "  python epub2audio.py <file.epub> --output ./audiobook"
