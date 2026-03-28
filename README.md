# epub2audio

Convert technical EPUB books into audiobooks with local AI models — zero cloud costs.

Orchestrates three AI models on a single consumer GPU (RTX 5080, 16 GB VRAM): Kokoro 82M (ONNX)
for speech synthesis, Qwen3-8B via Ollama for LLM enrichment of code blocks, figures, and dense
passages, and Whisper for optional quality-check transcription. Produces per-chapter MP3s with ID3
metadata, an M4B with chapter markers, and companion PDFs with syntax-highlighted code, figures,
and MathML-rendered formulas in DOM order. Total cloud cost: $0.

## Features

- **Multi-model VRAM orchestration** — sequential GPU scheduling across LLM (~6 GB), TTS (~2 GB),
  and STT (~2 GB) phases; each model loads, runs, and unloads before the next starts
- **LLM-enriched audio** — Ollama/Qwen3-8B generates chapter intros, spoken code annotations,
  figure descriptions, table narrations, and rewrites dense passages (Flesch < 40) before TTS
- **Parallel TTS synthesis** — ThreadPoolExecutor workers calling Kokoro with a shared ONNX lock
  (thread-safe GPU inference); SHA-based chunk cache for idempotent reruns
- **Companion PDF + HTML** — per-chapter documents with figures, Pygments syntax-highlighted code,
  MathML→LaTeX→PNG math, tables, and approximate audio timestamps — all in source DOM order
- **Pronunciation dictionary** — JSON overrides for acronyms and domain terms (RLHF, RAG, LLaMA,
  LoRA, PEFT, etc.) that Kokoro mispronounces by default
- **Quality check via Whisper STT** (optional) — transcribes generated audio, diffs against source
  text with SequenceMatcher WER, reports worst 10 chunks per chapter as JSON

## Quick start

```bash
git clone https://github.com/marcopetix/epub2audio && cd epub2audio
bash setup.sh                                         # venv, deps, models (~4 GB download)
python epub2audio.py book.epub --chapters 1           # test on chapter 1
python epub2audio.py book.epub --format mp3           # full run
```

## How it works

```
EPUB
 └─ Extract ──► Clean + LLM Enrich ──► Chunk ──► Parallel TTS (Kokoro, GPU)
                                                        │
                                          ┌─────────────┼─────────────┐
                                          ▼             ▼             ▼
                                     Assemble MP3   Companion     QC Whisper
                                      + M4B + ID3   PDF + HTML     (opt.)
```

| Phase | What happens |
|-------|-------------|
| Extract | Parse EPUB DOM: chapters, figures, code blocks, MathML, tables, cover art |
| LLM Enrich | Qwen3-8B generates spoken prose for each non-text element and rewrites dense passages |
| Chunk | Split enriched text at paragraph/sentence boundaries; hard cap at 2500 chars (Kokoro 512-token limit) |
| Parallel TTS | Kokoro ONNX synthesizes chunks in parallel; results written as per-chunk WAVs |
| Assemble | Concatenate WAVs → MP3 (ID3 tags) and/or M4B (chapter markers at H2 boundaries) |
| Companion | FPDF2 + Pygments + matplotlib produce PDF; same content in responsive HTML |
| QC | faster-whisper transcribes audio; WER diff against source flags synthesis errors |

## Performance

Tested on *Generative AI Design Patterns* (Lakshmanan & Hapke, O'Reilly 2025) —
10 chapters, 256 code blocks, 110 figures, ~800K characters.

| Phase | Time | Notes |
|-------|------|-------|
| LLM enrichment | ~15 min | Qwen3-8B via Ollama, cached on rerun |
| TTS synthesis | ~23 min | 4 workers, RTX 5080, 256 chunks total |
| Assembly | ~3.5 min | WAV concat + ffmpeg MP3/M4B encoding |
| Companion gen | ~6 s | 10 PDF + 10 HTML files |
| **Total** | **~42 min** | First run; ~25 min with cached LLM |

**Output:** 10 MP3 chapters, 1.04 GB total, ~18.5 hours of audio at 128 kbps.

**Cost comparison** for 800K characters of TTS:

| Service | Cost |
|---------|------|
| This project | $0 |
| Google Cloud TTS (WaveNet) | ~$12 |
| OpenAI tts-1 | ~$12 |
| ElevenLabs Creator | $22/mo (100K char limit) |

## Roadmap

- [x] EPUB extraction with structured elements (figures, code, math, tables in DOM order)
- [x] LLM enrichment via local Ollama (Qwen3-8B)
- [x] Parallel Kokoro TTS synthesis
- [x] Companion PDF + HTML with syntax highlighting
- [x] Whisper quality check
- [ ] M4B with hierarchical chapter markers (chapter + section level)
- [ ] Anti-distraction features: micro-recaps, retrieval quizzes, Anki export
- [ ] RSS feed for self-hosted podcast distribution
- [ ] Responsive companion HTML (mobile-first alternative to PDF)
- [ ] Multi-format input (PDF, MOBI)

## Tech stack

| Component | Technology | License |
|-----------|-----------|---------|
| TTS | Kokoro 82M (ONNX) | Apache 2.0 |
| LLM | Qwen3-8B via Ollama | Apache 2.0 |
| STT | faster-whisper | MIT |
| Audio | ffmpeg + pydub + mutagen | LGPL / MIT |
| PDF | fpdf2 + Pygments + matplotlib | LGPL / BSD / PSF |
| Parsing | BeautifulSoup4 + lxml | MIT / BSD |

## Blog series (coming soon)

1. "Orchestrating 3 AI Models on a Single Consumer GPU"
2. "From $15/book to $0: The Economics of Local vs Cloud TTS"
3. "5 Prompts, 5 Tasks: Systematic Prompt Engineering in a Real Pipeline"
4. "Building an Audio Quality Pipeline with Whisper"
5. "M4B with Chapter Markers: Audio Engineering for AI-Generated Audiobooks"

## License

MIT
