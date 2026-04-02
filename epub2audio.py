#!/usr/bin/env python3
"""epub2audio v2 — Convert EPUB books to M4B audiobooks with companion PDFs/HTML.

Usage:
    python epub2audio.py book.epub
    python epub2audio.py book.epub --chapters 1 2 3 --no-llm
    python epub2audio.py book.epub --format both --qc --upload
    python epub2audio.py book.epub --dry-run
"""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from config import Config
from pipeline.extractor import extract_chapters, extract_cover, extract_metadata
from pipeline.cleaner import clean_chapter
from pipeline.chunker import chunk_text
from pipeline.synthesizer import Synthesizer
from pipeline.assembler import assemble_chapter, assemble_m4b
from pipeline.companion import generate_companion

logger = logging.getLogger("epub2audio")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Convert EPUB to audiobook (M4B/MP3 + companion PDFs/HTML)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("epub_path", type=Path, help="Path to .epub file")
    parser.add_argument("-o", "--output", type=Path, default=Path("./audiobook"),
                        help="Output directory (default: ./audiobook)")
    parser.add_argument("--voice", default="af_nicole",
                        help="Kokoro voice ID (default: af_nicole)")
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
    parser.add_argument("--max-chars", type=int, default=3500,
                        help="Max chars per TTS chunk (default: 3500)")

    # Quality check
    parser.add_argument("--qc", action="store_true",
                        help="Run Whisper quality check after synthesis")
    parser.add_argument("--whisper-model", default="medium",
                        help="Whisper model size for QC (default: medium)")

    # Pronunciation
    parser.add_argument("--pronunciation", default="pronunciation.json",
                        help="Path to pronunciation dictionary JSON")

    # Upload
    parser.add_argument("--upload", action="store_true",
                        help="Upload to Google Drive after generation")

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
        enable_qc=args.qc,
        whisper_model=args.whisper_model,
        pronunciation_file=args.pronunciation,
        upload_gdrive=args.upload,
    )


def check_dependencies(config: Config):
    """Check that required tools and files are available."""
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg not found. Install it: sudo apt install ffmpeg")
        sys.exit(1)

    model_path = Path(config.kokoro_model)
    voices_path = Path(config.kokoro_voices)
    if not model_path.exists():
        logger.error(f"Kokoro model not found: {model_path}")
        logger.error("Run setup.sh to download model files")
        sys.exit(1)
    if not voices_path.exists():
        logger.error(f"Kokoro voices not found: {voices_path}")
        logger.error("Run setup.sh to download model files")
        sys.exit(1)


def _safe_title(title: str) -> str:
    return "_".join(
        "".join(c if c.isalnum() or c in " -" else "_" for c in title).split()
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)

    config = parse_args()
    check_dependencies(config)
    config.ensure_dirs()

    phase_times = {}
    start_time = time.time()

    # ================================================================
    # Phase 1: Extract chapters
    # ================================================================
    logger.info(f"[Phase 1/7] Extracting chapters from {config.epub_path.name}...")
    t0 = time.time()

    all_chapters = extract_chapters(config.epub_path)
    cover_art = extract_cover(config.epub_path)

    # Auto-detect book metadata from EPUB if not overridden
    if not config.book_title or not config.book_author:
        meta = extract_metadata(config.epub_path)
        if not config.book_title:
            config.book_title = meta.get("title", config.epub_path.stem)
        if not config.book_author:
            config.book_author = meta.get("author", "Unknown")
        if not config.book_year:
            config.book_year = meta.get("year", "")

    if cover_art:
        cover_path = config.output_dir / "cover.png"
        cover_path.write_bytes(cover_art)
        logger.info("Cover image saved")

    # Filter chapters
    chapters = all_chapters
    if config.chapters:
        chapters = [ch for ch in all_chapters if ch.number in config.chapters]
        logger.info(f"Processing {len(chapters)}/{len(all_chapters)} chapters: {config.chapters}")

    total_chapters = len(all_chapters)
    phase_times["extract"] = time.time() - t0
    logger.info(f"Extracted {len(all_chapters)} chapters ({len(chapters)} selected)")

    # ================================================================
    # Phase 2: LLM Enrichment (optional)
    # ================================================================
    if config.enable_llm:
        logger.info("[Phase 2/7] LLM enrichment with Ollama...")
        t0 = time.time()
        try:
            from pipeline.llm_enricher import LLMEnricher
            enricher = LLMEnricher(
                model=config.llm_model,
                ollama_url=config.ollama_url,
                cache_dir=config.llm_cache_dir,
            )
            if enricher.available:
                for chapter in chapters:
                    enricher.enrich_chapter(chapter)
                enricher.unload()
                logger.info("LLM enrichment complete, model unloaded")
            else:
                logger.warning("Ollama not available, skipping LLM enrichment")
        except ImportError:
            logger.warning("llm_enricher not available, skipping LLM enrichment")
        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}, continuing without it")
        phase_times["llm"] = time.time() - t0
    else:
        logger.info("[Phase 2/7] LLM enrichment skipped (--no-llm)")

    # ================================================================
    # Phase 3: Clean + Chunk all chapters
    # ================================================================
    logger.info("[Phase 3/7] Cleaning and chunking text...")
    t0 = time.time()

    chapter_data = []  # (chapter, clean_text, chunks, section_markers)
    for chapter in chapters:
        clean_text, section_markers = clean_chapter(
            chapter.raw_html,
            pronunciation_file=config.pronunciation_file,
            chapter=chapter,
        )
        chunks = chunk_text(clean_text, config.max_chunk_chars, section_markers)
        chapter_data.append((chapter, clean_text, chunks, section_markers))
        logger.info(
            f"  Ch {chapter.number}: {len(clean_text)} chars -> {len(chunks)} chunks"
        )

    phase_times["clean_chunk"] = time.time() - t0

    # ================================================================
    # Dry run: print stats and exit
    # ================================================================
    if config.dry_run:
        total_chars = sum(len(ct) for _, ct, _, _ in chapter_data)
        total_chunks = sum(len(ch) for _, _, ch, _ in chapter_data)
        est_minutes = total_chars / 1000  # rough: ~1000 chars/min audio
        print(f"\n{'='*60}")
        print(f"DRY RUN — {config.epub_path.name}")
        print(f"  Chapters:     {len(chapters)}/{len(all_chapters)}")
        print(f"  Total chars:  {total_chars:,}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Est. audio:   ~{est_minutes:.0f} min")
        print(f"  Output:       {config.output_dir}")
        print(f"  Format:       {config.output_format}")
        print(f"  LLM:          {'enabled' if config.enable_llm else 'disabled'}")
        print(f"  QC:           {'enabled' if config.enable_qc else 'disabled'}")
        print(f"{'='*60}")
        for ch, ct, chs, _ in chapter_data:
            print(f"  Ch {ch.number:2d}: {ch.title[:50]:50s} "
                  f"{len(ct):6,} chars  {len(chs):3d} chunks  "
                  f"{len(ch.figures):2d} fig  {len(ch.code_blocks):3d} code  "
                  f"{len(ch.math_formulas):2d} math  {len(ch.tables):2d} tables")
        return

    # ================================================================
    # Phase 4: Synthesize all chapters
    # ================================================================
    logger.info("[Phase 4/7] Synthesizing audio with Kokoro TTS...")
    t0 = time.time()

    synth = Synthesizer(
        model_path=config.kokoro_model,
        voices_path=config.kokoro_voices,
        voice=config.voice,
        speed=config.speed,
        num_workers=config.num_tts_workers,
    )

    # wav_results[chapter_num] = list of WavResult
    wav_results = {}
    for chapter, clean_text, chunks, section_markers in chapter_data:
        chapter_temp = config.temp_dir / f"ch{chapter.number:02d}"
        chapter_temp.mkdir(parents=True, exist_ok=True)

        results = synth.synthesize_chapter(
            chunks=chunks,
            chapter_num=chapter.number,
            temp_dir=chapter_temp,
            lang=config.lang,
            force=config.force,
        )

        if results:
            wav_results[chapter.number] = results
        else:
            logger.error(f"No audio for chapter {chapter.number}, skipping")

    synth.unload()
    logger.info("TTS complete, Kokoro unloaded")
    phase_times["synthesize"] = time.time() - t0

    # ================================================================
    # Phase 5: Assemble audio (M4B and/or MP3)
    # ================================================================
    logger.info("[Phase 5/7] Assembling audio...")
    t0 = time.time()

    manifest_chapters = []

    # Per-chapter MP3s (if format is "mp3" or "both")
    if config.output_format in ("mp3", "both"):
        for chapter, clean_text, chunks, _ in chapter_data:
            if chapter.number not in wav_results:
                continue
            wav_paths = [r.path for r in wav_results[chapter.number]]
            mp3_path = assemble_chapter(
                wav_paths=wav_paths,
                chapter_num=chapter.number,
                chapter_title=chapter.title,
                output_dir=config.audio_dir,
                bitrate=config.mp3_bitrate,
                book_title=config.book_title,
                book_author=config.book_author,
                book_year=config.book_year,
                total_chapters=total_chapters,
                cover_art=cover_art,
            )

    # M4B (if format is "m4b" or "both")
    if config.output_format in ("m4b", "both"):
        m4b_path = assemble_m4b(
            wav_results=wav_results,
            chapters=[ch for ch, _, _, _ in chapter_data if ch.number in wav_results],
            output_dir=config.output_dir,
            config=config,
            cover_art=cover_art,
        )
        if m4b_path:
            logger.info(f"M4B written: {m4b_path}")

    # Build manifest entries
    for chapter, clean_text, chunks, _ in chapter_data:
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
        if config.output_format in ("mp3", "both"):
            entry["audio_file"] = f"audio/{chapter.number:02d}_{safe}.mp3"
        if config.companion_format in ("pdf", "both"):
            entry["companion_pdf"] = f"companions/{chapter.number:02d}_{safe}_companion.pdf"
        if config.companion_format in ("html", "both"):
            entry["companion_html"] = f"companions/{chapter.number:02d}_{safe}_companion.html"
        manifest_chapters.append(entry)

    phase_times["assemble"] = time.time() - t0

    # ================================================================
    # Phase 6: Generate companions (PDF and/or HTML)
    # ================================================================
    logger.info("[Phase 6/7] Generating companion documents...")
    t0 = time.time()

    for chapter, _, _, _ in chapter_data:
        # Save chapter images to assets dir
        for src, img_data in chapter.images.items():
            img_path = config.assets_dir / Path(src).name
            img_path.write_bytes(img_data)

        # Get audio timestamps from wav_results if available
        audio_timestamps = None
        if chapter.number in wav_results:
            audio_timestamps = wav_results[chapter.number]

        generate_companion(
            chapter=chapter,
            output_dir=config.companions_dir,
            book_title=config.book_title,
            companion_format=config.companion_format,
            audio_timestamps=audio_timestamps,
        )

    phase_times["companion"] = time.time() - t0

    # ================================================================
    # Phase 7: Quality Check (optional)
    # ================================================================
    if config.enable_qc:
        logger.info("[Phase 7/7] Running Whisper quality check...")
        t0 = time.time()
        try:
            from pipeline.quality_check import QualityChecker
            checker = QualityChecker(model_size=config.whisper_model)
            qc_reports = []
            for chapter, clean_text, chunks, _ in chapter_data:
                if chapter.number not in wav_results:
                    continue
                report = checker.check_chapter(
                    wav_results=wav_results[chapter.number],
                    chunks=chunks,
                )
                qc_reports.append(report)
                logger.info(
                    f"  Ch {chapter.number}: similarity {report.similarity_ratio:.2%}"
                )
            checker.unload()

            # Save QC report
            qc_path = config.output_dir / "qc_report.json"
            qc_path.write_text(json.dumps(
                [r.to_dict() for r in qc_reports], indent=2, ensure_ascii=False
            ))
            logger.info(f"QC report saved to {qc_path}")
            phase_times["qc"] = time.time() - t0
        except ImportError:
            logger.warning("faster-whisper not installed, skipping QC")
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
    else:
        logger.info("[Phase 7/7] Quality check skipped (use --qc to enable)")

    # ================================================================
    # Cleanup temp WAVs
    # ================================================================
    shutil.rmtree(config.temp_dir, ignore_errors=True)

    # ================================================================
    # Write manifest
    # ================================================================
    manifest = {
        "title": config.book_title,
        "author": config.book_author,
        "generated": datetime.now().isoformat(),
        "voice": config.voice,
        "speed": config.speed,
        "format": config.output_format,
        "chapters": sorted(manifest_chapters, key=lambda c: c["index"]),
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Manifest written to {manifest_path}")

    # ================================================================
    # Optional upload
    # ================================================================
    if config.upload_gdrive:
        try:
            from pipeline.uploader import upload_to_drive
            upload_to_drive(config)
        except ImportError:
            logger.error(
                "Google Drive upload requires: "
                "pip install google-api-python-client google-auth-oauthlib"
            )
        except Exception as e:
            logger.error(f"Upload failed: {e}")

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - start_time
    audio_files = list(config.audio_dir.glob("*.mp3")) + list(config.output_dir.glob("*.m4b"))
    companion_files = list(config.companions_dir.glob("*.pdf")) + list(config.companions_dir.glob("*.html"))
    total_size = sum(f.stat().st_size for f in audio_files + companion_files)

    print(f"\n{'='*60}")
    print(f"DONE in {elapsed / 60:.1f} minutes")
    print(f"  Audio:       {len(audio_files)} files in {config.output_dir}")
    print(f"  Companions:  {len(companion_files)} files in {config.companions_dir}")
    print(f"  Total size:  {total_size / 1024 / 1024:.1f} MB")
    print(f"  Manifest:    {manifest_path}")
    print(f"  Phase times:")
    for phase, t in phase_times.items():
        print(f"    {phase:15s} {t:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
