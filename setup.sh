#!/bin/bash
set -e

echo "=== epub2audio v2 setup ==="

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 3. GPU acceleration
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

# 5. Check Ollama
if command -v ollama &> /dev/null; then
    echo "Ollama found: $(ollama --version 2>&1)"
    echo "Pulling Qwen3-8B model for LLM enrichment..."
    ollama pull qwen3:8b
else
    echo "WARNING: Ollama not found. LLM enrichment will be skipped."
    echo "  Install from: https://ollama.com/download"
    echo "  Then run: ollama pull qwen3:8b"
fi

# 6. Download Kokoro model files
echo "Downloading Kokoro TTS model files..."
KOKORO_BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
wget -nc "${KOKORO_BASE}/kokoro-v1.0.onnx" || true
wget -nc "${KOKORO_BASE}/voices-v1.0.bin" || true

# 7. Optional: Quality check with Whisper
echo ""
echo "Optional: Install faster-whisper for quality check:"
echo "  pip install faster-whisper>=1.0.0"

# 8. Smoke test
echo ""
echo "Running smoke test..."
python3 -c "from kokoro_onnx import Kokoro; print('Kokoro OK')" 2>/dev/null && echo "Kokoro loaded successfully" || echo "WARNING: Kokoro failed to load"

echo ""
echo "=== Setup complete! ==="
echo "Usage:"
echo "  source .venv/bin/activate"
echo "  python epub2audio.py <file.epub> --output ./audiobook"
echo "  python epub2audio.py <file.epub> --chapters 1 --dry-run"
