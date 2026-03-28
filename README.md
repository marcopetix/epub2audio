# epub2audio v2

Convert technical EPUB books into M4B audiobooks with companion PDFs/HTML.

Everything runs locally on your GPU: **Kokoro TTS** for speech, **Qwen3-8B via Ollama** for LLM enrichment, **Whisper** for quality check. Zero cloud costs.

## Features

- **M4B audiobook** with chapter markers (navigable in any audiobook player)
- **Per-chapter MP3s** as alternative output
- **Companion PDF** with syntax-highlighted code, figures, math formulas, tables, and audio timestamps
- **Companion HTML** with responsive design and zoomable images
- **LLM enrichment** via Ollama: chapter intros, code annotations, figure descriptions, table narrations
- **Parallel TTS** synthesis on GPU (4 workers default)
- **Whisper quality check** to detect TTS pronunciation errors
- **Pronunciation dictionary** for technical terms
- **Idempotent**: caches LLM results and skips existing outputs

## Quick Start

```bash
# Setup
./setup.sh

# Basic usage
source .venv/bin/activate
python epub2audio.py book.epub

# Preview without generating audio
python epub2audio.py book.epub --dry-run

# First 3 chapters, no LLM, MP3 format
python epub2audio.py book.epub --chapters 1 2 3 --no-llm --format mp3

# Full pipeline with quality check
python epub2audio.py book.epub --qc

# Upload to Google Drive
python epub2audio.py book.epub --upload
```

## Requirements

- Python 3.11+
- ffmpeg
- NVIDIA GPU (recommended, CPU fallback supported)
- [Ollama](https://ollama.com) + `qwen3:8b` (optional, for LLM enrichment)

## Pipeline

```
EPUB -> Extract -> [LLM Enrich] -> Clean -> Chunk -> Synthesize -> Assemble -> Companion
                                                                      |            |
                                                                   M4B/MP3    PDF + HTML
```

## Output

```
audiobook/
  BookTitle.m4b              # Audiobook with chapter markers
  audio/                     # Per-chapter MP3s (--format both)
  companions/                # PDF + HTML per chapter
  cover.png
  manifest.json
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--format m4b\|mp3\|both` | Audio output format (default: m4b) |
| `--companion pdf\|html\|both` | Companion format (default: both) |
| `--no-llm` | Skip LLM enrichment |
| `--qc` | Run Whisper quality check |
| `--chapters 1 2 3` | Process specific chapters |
| `--workers N` | Parallel TTS workers (default: 4) |
| `--voice ID` | Kokoro voice (default: af_nicole) |
| `--speed N` | TTS speed (default: 1.1) |
| `--dry-run` | Show stats without generating |
| `--force` | Regenerate existing files |
| `--upload` | Upload to Google Drive |

## GPU VRAM Management

The pipeline runs three GPU workloads **sequentially** to stay within 16GB VRAM:

1. **LLM phase**: Qwen3-8B (~5-6 GB) -> unload
2. **TTS phase**: Kokoro (~1-2 GB, parallel workers) -> unload
3. **QC phase**: Whisper medium (~2 GB) -> unload
