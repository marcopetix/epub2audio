# epub2audio

Convert technical EPUB books into audiobooks with multimodal AI enrichment — local by default, cloud only when it pays off.

Orchestrates a multi-model pipeline on a single consumer GPU (RTX 5080, 16 GB VRAM): Kokoro 82M (ONNX)
for speech synthesis, Qwen3-8B via Ollama for text-based enrichment of code, formulas, figures, and
tables, and (optionally) Gemini 2.5 Flash via OpenRouter for *vision-based* descriptions of figures and
math formulas. Produces an M4B audiobook with chapter markers, per-chapter MP3s with ID3 metadata, and
companion PDFs/HTMLs with syntax-highlighted code, MathML-rendered formulas, and figures in DOM order.

LLM orchestration is built on **[datapizza-ai](https://github.com/datapizza-labs/datapizza-ai)**: one client
abstraction across providers, native caching via the `@cacheable` decorator, automatic OpenTelemetry
tracing on every model call.

## Features

- **Single-framework multi-provider LLM** — `OpenAILikeClient` from datapizza-ai talks to local Ollama
  for text and to OpenRouter for vision, sharing the same cache and tracing infrastructure
- **Vision enrichment for figures and formulas** (opt-in via `--vision`) — Gemini 2.5 Flash describes
  what each figure actually shows instead of paraphrasing alt text and captions
- **Persistent LLM cache** — `FileCache(Cache)` implements datapizza-ai's abstract cache contract on
  disk via pickle; cache hits short-circuit before the model call and don't appear in token tracing
- **Native observability** — `--tracing` wraps the pipeline in `ContextTracing()` and prints a console
  summary with token usage per model. Cache hits vs real calls are visible in the trace
- **Multi-model VRAM orchestration** — sequential GPU scheduling: LLM (~6 GB) loads, runs, unloads
  before TTS (~2 GB) starts. Kokoro inference uses `ThreadPoolExecutor` with a shared ONNX lock
- **Companion PDF + HTML** — per-chapter documents with figures, Pygments syntax-highlighted code,
  MathML→LaTeX→PNG math (pre-rendered during extraction so vision can describe them), tables, and
  approximate audio timestamps — all in source DOM order
- **Pronunciation dictionary** — JSON overrides for acronyms and domain terms (RLHF, RAG, LLaMA,
  LoRA, etc.) that Kokoro mispronounces by default

## Quick start

```bash
git clone https://github.com/marcopetix/epub2audio && cd epub2audio
bash setup.sh                                          # venv, deps, models (~4 GB download)

# Text-only enrichment (default, $0 cloud cost):
python epub2audio.py book.epub --chapters 1            # test on chapter 1
python epub2audio.py book.epub                         # full book, M4B output

# With vision enrichment (cents per book on OpenRouter):
export OPENROUTER_API_KEY=sk-or-...
python epub2audio.py book.epub --vision --tracing      # text + vision + console trace
```

## CLI flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--chapters N [N...]` | all | Process only the listed chapter numbers |
| `--no-llm` | off | Skip LLM enrichment entirely (no Ollama required) |
| `--vision` | off | Enable vision-based description of figures and formulas (requires `OPENROUTER_API_KEY`) |
| `--vision-model NAME` | `google/gemini-2.5-flash` | Override the OpenRouter model |
| `--tracing` | off | Print OpenTelemetry trace summary at the end of the run |
| `--langfuse` | off | Reserved — Langfuse exporter not yet wired (see Roadmap) |
| `--format {m4b,mp3,both}` | `m4b` | Audio output format |
| `--companion {pdf,html,both}` | `both` | Companion output format |
| `--dry-run` | off | Run extraction + enrichment + chunking, then exit (no TTS) |

## How it works

```
EPUB
 └─ Extract ──► LLM Enrich ──► Clean + Chunk ──► Parallel TTS (Kokoro, GPU)
       │           │                                       │
       │           │                                ┌──────┴──────┐
       │           │                                ▼             ▼
       │           │                          Assemble M4B   Companion
       │           │                          + per-ch MP3   PDF + HTML
       │           │
       │           └── Ollama (qwen3:8b) for text tasks
       │               OpenRouter (gemini-2.5-flash) for vision (opt-in)
       │
       └── pre-renders math formulas to PNG so vision can describe them
```

| Phase | What happens |
|-------|-------------|
| Extract | Parse EPUB DOM into a `list[NarratedElement]` (figures, code, math, tables, all sharing `number`, `context`, `dom_position`, `narration`, `rendered_path`). Figure images and math formulas are written to disk during extraction. |
| Enrich | `Enricher` (datapizza-ai based) generates a chapter intro, code annotations, table narrations, and figure/formula descriptions. Vision adapter dispatches polymorphically over `rendered_path` when `--vision` is on. |
| Clean + Chunk | HTML → TTS-friendly text, replacing each visual element with its narration (or a generic placeholder if enrichment failed). Hard cap at 2500 chars per chunk to stay below Kokoro's 512-token internal limit. |
| Parallel TTS | Kokoro ONNX synthesizes chunks across 4 workers; a shared lock serializes calls to the (non-thread-safe) ONNX session. |
| Assemble | Concatenate WAVs → M4B (AAC + chapter markers from H1/H2 boundaries) and/or per-chapter MP3 with ID3 tags. |
| Companion | FPDF2 + Pygments + matplotlib produce PDF; same content in responsive HTML. Figures and pre-rendered math PNGs are read from disk. |

## Performance

Tested on *Generative AI Design Patterns* (Lakshmanan & Hapke, O'Reilly 2025) —
10 chapters, 256 code blocks, 110 figures, 11 math formulas, ~800K characters.

### Single-chapter (Chapter 1: Introduction, 31 chunks, 7 figures, 21 code, 2 formulas)

| Phase | Cold cache | Warm cache |
|-------|-----------|------------|
| LLM enrichment (text-only) | ~2 min | <1 s |
| LLM enrichment (`--vision`) | ~2 min + ~10 s vision | <1 s |
| TTS synthesis | ~2.5 min | (idempotent on chunk WAVs) |
| Assemble + companion | ~30 s | ~30 s |

Cache hits short-circuit before the model is called; the OpenTelemetry trace correctly distinguishes
real calls (with token usage) from cache hits (no span at all).

### Full book (10 chapters, text-only)

| Phase | Time | Notes |
|-------|------|-------|
| LLM enrichment | ~15 min | Qwen3-8B via Ollama, FileCache persistent across runs |
| TTS synthesis | ~23 min | 4 workers, RTX 5080 |
| Assembly | ~3.5 min | WAV concat + ffmpeg M4B encoding |
| Companion gen | ~6 s | 10 PDF + 10 HTML files |
| **Total** | **~42 min** | First run; ~25 min with warm LLM cache |

**Output:** ~18.5 hours of audio at 128 kbps, ~1 GB total.

### Cost

| Service | Cost (800K chars TTS + 110 figure descriptions) |
|---------|------|
| This project, text-only | $0 |
| This project, `--vision` | <$0.05 (Gemini Flash via OpenRouter) |
| Google Cloud TTS (WaveNet) | ~$12 |
| OpenAI tts-1 | ~$12 |
| ElevenLabs Creator | $22/mo (100K char limit) |

## Architecture notes

### `NarratedElement` polymorphism

Figures, code blocks, math formulas, and tables share a base `NarratedElement` dataclass with
`number`, `context`, `dom_position`, `narration`, `rendered_path`. The vision adapter is a single
method that takes any `NarratedElement` whose `rendered_path` exists and produces a description —
no per-type branching. This is what allows `--vision` to enrich both figures and formulas through
the same code path.

### Why pre-render math during extraction

Math formulas are MathML in the source. To describe them via vision we need a PNG. The earlier
version rendered MathML→LaTeX→PNG inside the companion generator; the refactor moves this to
extraction so the rendered PNG path is a first-class field on `MathFormula`. Same `rendered_path`
contract as figures, same vision adapter, no special-casing.

### Why pickle for the cache

datapizza-ai's `@cacheable` decorator stores `ClientResponse` objects. These are dataclasses with
nested `TextBlock`/`MediaBlock` instances — JSON-friendly serialization would lose information on
re-hydration. `FileCache` uses pickle, which round-trips the full object graph and lets cache hits
return values indistinguishable from a real call.

### Why the Kokoro chunk limit is 2500 chars

Kokoro's internal phoneme model has a hard ~512-token limit. Dense LLM-enriched technical text can
produce ~1 token per 4-5 chars. At 3500 chars some chunks crash with `index 510 is out of bounds for
axis 0 with size 510` inside Kokoro's duration predictor. 2500 chars keeps peak token counts below
512 even on the densest passages. This regression is covered by tests in `tests/test_chunker.py`.

## Roadmap

- [x] EPUB extraction with structured `NarratedElement` (figures, code, math, tables in DOM order)
- [x] Local LLM enrichment via Ollama (Qwen3-8B) for intros, code, tables, and text-only figure/formula descriptions
- [x] Vision enrichment for figures and math formulas via OpenRouter (Gemini 2.5 Flash)
- [x] datapizza-ai `OpenAILikeClient` for both providers, with shared `FileCache` and OpenTelemetry tracing
- [x] Parallel Kokoro TTS synthesis with thread-safe ONNX lock
- [x] M4B with chapter markers + per-chapter MP3 with ID3 tags
- [x] Companion PDF + HTML with syntax highlighting and pre-rendered math
- [ ] **Langfuse exporter wiring** — `--langfuse` flag is reserved but the `OTLPSpanExporter` setup is not yet implemented; once wired, traces will export automatically without changes elsewhere
- [ ] **MathML edge cases** — `\inchunk`, `\sumP`, and a handful of other tokens are not handled by the MathML→LaTeX visitor in `pipeline/math_renderer.py`. Affected formulas fall back to the alttext-only path
- [ ] **End-to-end re-validation** with vision on full book — current measurements are on chapter 1; a full 10-chapter run with `--vision` is pending
- [ ] M4B with hierarchical chapter markers (chapter + section level)
- [ ] Anti-distraction features: micro-recaps, retrieval quizzes, Anki export
- [ ] RSS feed for self-hosted podcast distribution
- [ ] Multi-format input (PDF, MOBI)

## Tech stack

| Component | Technology | License |
|-----------|-----------|---------|
| LLM orchestration | [datapizza-ai](https://github.com/datapizza-labs/datapizza-ai) | Apache 2.0 |
| Text LLM | Qwen3-8B via Ollama | Apache 2.0 |
| Vision LLM (opt-in) | Gemini 2.5 Flash via OpenRouter | proprietary, paid |
| TTS | Kokoro 82M (ONNX) | Apache 2.0 |
| STT (optional, not wired) | faster-whisper | MIT |
| Audio | ffmpeg + pydub + mutagen | LGPL / MIT |
| PDF | fpdf2 + Pygments + matplotlib | LGPL / BSD / PSF |
| Parsing | BeautifulSoup4 + lxml | MIT / BSD |
| Tracing | OpenTelemetry (via datapizza-ai) | Apache 2.0 |

## Known limitations

- **MathML→LaTeX visitor** does not cover all OpenMath operators; some formulas degrade to
  alttext-only rendering and are described from text alone (no vision)
- **Vision cache + `Path` objects**: passing a `pathlib.Path` directly into `Media.source` crashes
  the `@cacheable` decorator (`PosixPath has no encode`). The enricher always converts via
  `str(path)`. Worth filing upstream
- **`MediaBlock.__hash__` for cache keys** depends on path strings, not image bytes. Two different
  images at the same path would collide. Not an issue for this pipeline (paths are deterministic
  per-EPUB) but worth noting if reused

## License

MIT