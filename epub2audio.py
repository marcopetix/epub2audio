#!/usr/bin/env python3
"""epub2audio v2 — Convert EPUB books to M4B audiobooks with companion PDFs/HTML.

Usage:
    python epub2audio.py book.epub # Process entire book with defaults
    python epub2audio.py book.epub --chapters 1 2 3 --no-llm # Only process chapters 1-3, skip LLM enrichment
    python epub2audio.py book.epub --format both --qc --upload # Generate both M4B and MP3, run quality check, and upload to Google Drive
    python epub2audio.py book.epub --dry-run # Show stats and what would be done without generating audio
"""

import argparse
import json
import logging
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from config import Config
from pipeline.extractor import Chapter, extract_chapters, extract_cover, extract_metadata
from pipeline.cleaner import clean_chapter
from pipeline.chunker import chunk_text
from pipeline.synthesizer import Synthesizer, WavResult
from pipeline.assembler import assemble_chapter, assemble_m4b
from pipeline.companion import generate_companion

logger = logging.getLogger("epub2audio")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Convert EPUB to audiobook (M4B/MP3 + companion PDFs/HTML)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("epub_path", type=Path, help="Path to .epub file")
    parser.add_argument("-o", "--output", type=Path, default=Path("./output/audiobook"),
                        help="Output directory (default: ./output/audiobook)")
    parser.add_argument("--voice", default="af_heart",
                        help="Kokoro voice ID (default: af_heart)")
    parser.add_argument("--speed", type=float, default=1.1,
                        help="TTS speed multiplier (default: 1.1)")
    parser.add_argument("--chapters", type=int, nargs="+",
                        help="Only process these chapter numbers")
    parser.add_argument("--format", choices=["m4b", "mp3", "both"], default="m4b",
                        help="Audio output format (default: m4b)")
    parser.add_argument("--companion", choices=["pdf", "html", "both"], default="both",
                        help="Companion output format (default: both)")
    parser.add_argument("--bitrate", default="128k",
                        help="Audio bitrate (default: 128k)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if output files exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without synthesizing")

    # LLM
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM enrichment (no Ollama required)")
    parser.add_argument("--llm-model", default="qwen3:8b",
                        help="Ollama model for LLM enrichment (default: qwen3:8b)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama API URL (default: http://localhost:11434)")

    # TTS
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel TTS workers (default: 4)")
    parser.add_argument("--model", default="kokoro-v1.0.onnx",
                        help="Path to Kokoro ONNX model")
    parser.add_argument("--voices", default="voices-v1.0.bin",
                        help="Path to Kokoro voices file")
    parser.add_argument("--max-chars", type=int, default=2500,
                        help="Max chars per TTS chunk (default: 2500)")

    # Pronunciation
    parser.add_argument("--pronunciation", default="pronunciation.json",
                        help="Path to pronunciation dictionary JSON")
    
    parser.add_argument("--vision", action="store_true",
                        help="Enable vision enrichment for figures (requires OPENROUTER_API_KEY)")
    parser.add_argument("--vision-model", default="google/gemini-2.5-flash",
                        help="Vision model (default: google/gemini-2.5-flash)")
    parser.add_argument("--tracing", action="store_true",
                        help="Enable OpenTelemetry tracing with console summary")
    parser.add_argument("--langfuse", action="store_true",
                        help="Export traces to Langfuse (requires LANGFUSE_*_KEY env vars)")

    args = parser.parse_args()

    if not args.epub_path.exists():
        parser.error(f"EPUB file not found: {args.epub_path}")

    return Config(
        epub_path=args.epub_path,
        output_dir=args.output,
        voice=args.voice,
        speed=args.speed,
        lang="en-us",
        kokoro_model=args.model,
        kokoro_voices=args.voices,
        num_tts_workers=args.workers,
        output_format=args.format,
        companion_format=args.companion,
        mp3_bitrate=args.bitrate,
        aac_bitrate=args.bitrate,
        max_chunk_chars=args.max_chars,
        chapters=args.chapters,
        force=args.force,
        dry_run=args.dry_run,
        enable_llm=not args.no_llm,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
        pronunciation_file=args.pronunciation,
        enable_vision=args.vision,
        vision_model=args.vision_model,
        enable_tracing=args.tracing,
        enable_langfuse=args.langfuse, 
    )


def check_dependencies(config: Config):
    """Check that required tools and files are available."""
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg not found. Install it: sudo apt install ffmpeg")
        sys.exit(1)

    model_path = Path(config.kokoro_model)
    voices_path = Path(config.kokoro_voices)
    if not model_path.exists():
        logger.error("Kokoro model not found: %s", model_path)
        logger.error("Run setup.sh to download model files")
        sys.exit(1)
    if not voices_path.exists():
        logger.error("Kokoro voices not found: %s", voices_path)
        logger.error("Run setup.sh to download model files")
        sys.exit(1)


def _safe_title(title: str) -> str:
    return "_".join(
        "".join(c if c.isalnum() or c in " -" else "_" for c in title).split()
    )


class Pipeline:
    """Orchestrates the 6-phase EPUB → audiobook conversion.

    State (chapters, wav_results, manifest_chapters, etc.) is held on the
    instance so that phase methods can communicate without parameter chains.
    """

    def __init__(self, config: Config):
        self.config = config
        self.phase_times: dict[str, float] = {}
        self.start_time: float = 0.0
        self.all_chapters: list[Chapter] = []
        self.chapters: list[Chapter] = []
        self.total_chapters: int = 0
        self.cover_art: bytes | None = None
        self.chapter_data: list[tuple] = []
        self.wav_results: dict[int, list[WavResult]] = {}
        self.manifest_chapters: list[dict] = []
        self.manifest_path: Path | None = None

    @contextmanager
    def _timed(self, phase_name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self.phase_times[phase_name] = time.time() - t0

    def run(self) -> dict:
        """Execute the full pipeline. Returns a summary dict for printing."""
        self.start_time = time.time()

        if self.config.enable_tracing:
            from datapizza.tracing import ContextTracing
            trace_name = f"epub2audio_{self.config.epub_path.stem}"
            with ContextTracing().trace(trace_name):
                self._run_phases()
        else:
            self._run_phases()

        return self._build_summary()

    def _run_phases(self):
        """Internal: execute pipeline phases in order. Wrapped by tracing in run()."""
        self.phase1_extract()
        self.phase2_enrich()
        self.phase3_clean_and_chunk()
        if self.config.dry_run:
            self._print_dry_run_stats()
            return
        self.phase4_synthesize()
        self.phase5_assemble()
        self.phase6_companion()
        self._cleanup()
        self._write_manifest()

    # ----- Phase methods (filled step-by-step) -----
    def phase1_extract(self):
        logger.info("[Phase 1/6] Extracting chapters from %s", self.config.epub_path.name)
        with self._timed("extract"):
            self.all_chapters = extract_chapters(self.config.epub_path, self.config.assets_dir)
            self.cover_art = extract_cover(self.config.epub_path)

            # Auto-detect book metadata from EPUB if not overridden
            if not self.config.book_title or not self.config.book_author:
                meta = extract_metadata(self.config.epub_path)
                if not self.config.book_title:
                    self.config.book_title = meta.get("title", self.config.epub_path.stem)
                if not self.config.book_author:
                    self.config.book_author = meta.get("author", "Unknown")
                if not self.config.book_year:
                    self.config.book_year = meta.get("year", "")

            if self.cover_art:
                cover_path = self.config.output_dir / "cover.png"
                cover_path.write_bytes(self.cover_art)
                logger.info("Cover image saved")

            # Filter chapters
            self.chapters = self.all_chapters
            if self.config.chapters:
                self.chapters = [ch for ch in self.all_chapters if ch.number in self.config.chapters]
                logger.info(
                    "[Phase 1/6] Processing %d/%d chapters: %s",
                    len(self.chapters), len(self.all_chapters), self.config.chapters,
                )

            self.total_chapters = len(self.all_chapters)
        logger.info(
            "[Phase 1/6] Extracted %d chapters (%d selected)",
            len(self.all_chapters), len(self.chapters),
        )

    def phase2_enrich(self):
        if not self.config.enable_llm:
            logger.info("[Phase 2/6] LLM enrichment skipped (--no-llm)")
            return

        vision_status = "with vision" if self.config.enable_vision else "text-only"
        logger.info("[Phase 2/6] LLM enrichment (%s)...", vision_status)

        with self._timed("llm"):
            try:
                import os
                from pipeline.enricher import Enricher
                from pipeline.file_cache import FileCache

                vision_api_key = os.getenv("OPENROUTER_API_KEY") if self.config.enable_vision else None
                if self.config.enable_vision and not vision_api_key:
                    logger.warning(
                        "--vision requested but OPENROUTER_API_KEY env var not set; "
                        "falling back to text-only enrichment"
                    )

                enricher = Enricher(
                    ollama_url=self.config.ollama_url,
                    ollama_model=self.config.llm_model,
                    cache=FileCache(self.config.llm_cache_dir),
                    vision_enabled=self.config.enable_vision and bool(vision_api_key),
                    vision_api_key=vision_api_key,
                    vision_base_url=self.config.vision_base_url,
                    vision_model=self.config.vision_model,
                )
                if enricher.available:
                    for chapter in self.chapters:
                        enricher.enrich_chapter(chapter)
                    enricher.unload()
                    logger.info("LLM enrichment complete, model unloaded")
                else:
                    logger.warning("Ollama not available, skipping LLM enrichment")
            except ImportError as e:
                logger.warning("Enricher unavailable (%s), skipping LLM enrichment", e)
            except Exception as e:
                logger.warning("LLM enrichment failed: %s, continuing without it", e)

    def phase3_clean_and_chunk(self):
        logger.info("[Phase 3/6] Cleaning and chunking text...")
        with self._timed("clean_chunk"):
            self.chapter_data = []  # (chapter, clean_text, chunks, section_markers)
            for chapter in self.chapters:
                clean_text, section_markers = clean_chapter(
                    chapter.raw_html,
                    pronunciation_file=self.config.pronunciation_file,
                    chapter=chapter,
                )
                chunks = chunk_text(clean_text, self.config.max_chunk_chars, section_markers)
                self.chapter_data.append((chapter, clean_text, chunks, section_markers))
                logger.info(
                    "[Phase 3/6]  Ch %d: %d chars -> %d chunks",
                    chapter.number,
                    len(clean_text),
                    len(chunks),
                )

    def phase4_synthesize(self):
        logger.info("[Phase 4/6] Synthesizing audio with Kokoro TTS...")
        with self._timed("synthesize"):
            synth = Synthesizer(
                model_path=self.config.kokoro_model,
                voices_path=self.config.kokoro_voices,
                voice=self.config.voice,
                speed=self.config.speed,
                num_workers=self.config.num_tts_workers,
            )

            self.wav_results = {}
            for chapter, clean_text, chunks, section_markers in self.chapter_data:
                chapter_temp = self.config.temp_dir / f"ch{chapter.number:02d}"
                chapter_temp.mkdir(parents=True, exist_ok=True)

                results = synth.synthesize_chapter(
                    chunks=chunks,
                    chapter_num=chapter.number,
                    temp_dir=chapter_temp,
                    lang=self.config.lang,
                    force=self.config.force,
                )

                if results:
                    self.wav_results[chapter.number] = results
                else:
                    logger.error("No audio for chapter %d, skipping", chapter.number)

            synth.unload()
            logger.info("TTS complete, Kokoro unloaded")

    def phase5_assemble(self):
        logger.info("[Phase 5/6] Assembling audio...")
        with self._timed("assemble"):
            self.manifest_chapters = []

            # Per-chapter MP3s (if format is "mp3" or "both")
            if self.config.output_format in ("mp3", "both"):
                for chapter, clean_text, chunks, _ in self.chapter_data:
                    if chapter.number not in self.wav_results:
                        continue
                    wav_paths = [r.path for r in self.wav_results[chapter.number]]
                    mp3_path = assemble_chapter(
                        wav_paths=wav_paths,
                        chapter_num=chapter.number,
                        chapter_title=chapter.title,
                        output_dir=self.config.audio_dir,
                        bitrate=self.config.mp3_bitrate,
                        book_title=self.config.book_title,
                        book_author=self.config.book_author,
                        book_year=self.config.book_year,
                        total_chapters=self.total_chapters,
                        cover_art=self.cover_art,
                    )
                    if mp3_path:
                        logger.info("MP3 written: %s", mp3_path)

            # M4B (if format is "m4b" or "both")
            if self.config.output_format in ("m4b", "both"):
                m4b_path = assemble_m4b(
                    wav_results=self.wav_results,
                    chapters=[ch for ch, _, _, _ in self.chapter_data if ch.number in self.wav_results],
                    output_dir=self.config.output_dir,
                    config=self.config,
                    cover_art=self.cover_art,
                )
                if m4b_path:
                    logger.info("M4B written: %s", m4b_path)

            # Build manifest entries
            for chapter, clean_text, chunks, _ in self.chapter_data:
                safe = _safe_title(chapter.title)
                entry = {
                    "index": chapter.number,
                    "title": chapter.title,
                    "chars": len(clean_text),
                    "chunks": len(chunks),
                    "figures": len(chapter.figures),
                    "code_blocks": len(chapter.code_blocks),
                    "math_formulas": len(chapter.math_formulas),
                    "tables": len(chapter.tables),
                }
                if self.config.output_format in ("mp3", "both"):
                    entry["audio_file"] = f"audio/{chapter.number:02d}_{safe}.mp3"
                if self.config.companion_format in ("pdf", "both"):
                    entry["companion_pdf"] = f"companions/{chapter.number:02d}_{safe}_companion.pdf"
                if self.config.companion_format in ("html", "both"):
                    entry["companion_html"] = f"companions/{chapter.number:02d}_{safe}_companion.html"
                self.manifest_chapters.append(entry)

    def phase6_companion(self):
        logger.info("[Phase 6/6] Generating companion documents...")
        with self._timed("companion"):
            for chapter, _, _, _ in self.chapter_data:
                # Images and math PNGs are written to assets_dir during extraction;
                # companion just reads them via element.rendered_path.

                audio_timestamps = None
                if chapter.number in self.wav_results:
                    audio_timestamps = self.wav_results[chapter.number]

                generate_companion(
                    chapter=chapter,
                    output_dir=self.config.companions_dir,
                    book_title=self.config.book_title,
                    companion_format=self.config.companion_format,
                    audio_timestamps=audio_timestamps,
                )

    # ----- Helpers (filled step-by-step) -----
    def _print_dry_run_stats(self):
        total_chars = sum(len(ct) for _, ct, _, _ in self.chapter_data)
        total_chunks = sum(len(ch) for _, _, ch, _ in self.chapter_data)
        est_minutes = total_chars / 1000  # rough: ~1000 chars/min audio
        print(f"\n{'='*60}")
        print(f"DRY RUN — {self.config.epub_path.name}")
        print(f"  Chapters:     {len(self.chapters)}/{len(self.all_chapters)}")
        print(f"  Total chars:  {total_chars:,}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Est. audio:   ~{est_minutes:.0f} min")
        print(f"  Output:       {self.config.output_dir}")
        print(f"  Format:       {self.config.output_format}")
        print(f"  LLM:          {'enabled' if self.config.enable_llm else 'disabled'}")
        print(f"  Vision:       {'enabled' if self.config.enable_vision else 'disabled'}")
        print(f"  Tracing:      {'enabled' if self.config.enable_tracing else 'disabled'}")
        print(f"{'='*60}")
        for ch, ct, chs, _ in self.chapter_data:
            print(f"  Ch {ch.number:2d}: {ch.title[:50]:50s} "
                  f"{len(ct):6,} chars  {len(chs):3d} chunks  "
                  f"{len(ch.figures):2d} fig  {len(ch.code_blocks):3d} code  "
                  f"{len(ch.math_formulas):2d} math  {len(ch.tables):2d} tables")

    def _cleanup(self):
        shutil.rmtree(self.config.temp_dir, ignore_errors=True)

    def _write_manifest(self):
        manifest = {
            "title": self.config.book_title,
            "author": self.config.book_author,
            "generated": datetime.now().isoformat(),
            "voice": self.config.voice,
            "speed": self.config.speed,
            "format": self.config.output_format,
            "chapters": sorted(self.manifest_chapters, key=lambda c: c["index"]),
        }
        self.manifest_path = self.config.output_dir / "manifest.json"
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        logger.info("Manifest written to %s", self.manifest_path)

    def _build_summary(self) -> dict:
        elapsed = time.time() - self.start_time
        audio_files = list(self.config.audio_dir.glob("*.mp3")) + list(self.config.output_dir.glob("*.m4b"))
        companion_files = (
            list(self.config.companions_dir.glob("*.pdf"))
            + list(self.config.companions_dir.glob("*.html"))
        )
        total_size = sum(f.stat().st_size for f in audio_files + companion_files)
        return {
            "elapsed_seconds": elapsed,
            "audio_files": audio_files,
            "companion_files": companion_files,
            "total_size": total_size,
            "manifest_path": self.manifest_path,
            "phase_times": self.phase_times,
            "audio_dir": self.config.audio_dir,
            "output_dir": self.config.output_dir,
            "companions_dir": self.config.companions_dir,
            "dry_run": self.config.dry_run,
        }


def setup_logging():
    """Configure root logger and silence noisy dependencies."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)


def _print_summary(summary: dict):
    """Pretty-print pipeline summary at the end."""
    if summary.get("dry_run"):
        return
    elapsed = summary["elapsed_seconds"]
    audio_files = summary["audio_files"]
    companion_files = summary["companion_files"]
    total_size = summary["total_size"]
    manifest_path = summary["manifest_path"]
    output_dir = summary["output_dir"]
    companions_dir = summary["companions_dir"]
    phase_times = summary["phase_times"]

    print(f"\n{'='*60}")
    print(f"DONE in {elapsed / 60:.1f} minutes")
    print(f"  Audio:       {len(audio_files)} files in {output_dir}")
    print(f"  Companions:  {len(companion_files)} files in {companions_dir}")
    print(f"  Total size:  {total_size / 1024 / 1024:.1f} MB")
    print(f"  Manifest:    {manifest_path}")
    print("  Phase times:")
    for phase, t in phase_times.items():
        print(f"    {phase:15s} {t:.1f}s")
    print(f"{'='*60}")


def main():
    setup_logging()
    config = parse_args()
    check_dependencies(config)
    config.ensure_dirs()

    pipeline = Pipeline(config)
    summary = pipeline.run()

    _print_summary(summary)


if __name__ == "__main__":
    main()
