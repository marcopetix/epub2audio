#!/usr/bin/env python3
"""epub2audio — Convert EPUB books to audiobook MP3s with companion PDFs.

Usage:
    python epub2audio.py book.epub --output ./audiobook --voice af_nicole --speed 1.1
    python epub2audio.py book.epub --chapters 1 2 3 --output ./test
    python epub2audio.py book.epub --output ./audiobook --upload
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
from pipeline.extractor import extract_chapters, extract_cover
from pipeline.cleaner import clean_chapter
from pipeline.chunker import chunk_text
from pipeline.synthesizer import Synthesizer
from pipeline.assembler import assemble_chapter
from pipeline.companion import generate_companion

logger = logging.getLogger("epub2audio")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Convert EPUB to audiobook (MP3 + companion PDFs)",
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
                        help="Only process these chapter numbers (e.g., --chapters 1 2 3)")
    parser.add_argument("--bitrate", default="128k",
                        help="MP3 bitrate (default: 128k)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if output files exist")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to Google Drive after generation")
    parser.add_argument("--model", default="kokoro-v1.0.onnx",
                        help="Path to Kokoro ONNX model")
    parser.add_argument("--voices", default="voices-v1.0.bin",
                        help="Path to Kokoro voices file")
    parser.add_argument("--max-chars", type=int, default=3500,
                        help="Max chars per TTS chunk (default: 3500)")

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
        mp3_bitrate=args.bitrate,
        max_chunk_chars=args.max_chars,
        chapters=args.chapters,
        force=args.force,
        upload_gdrive=args.upload,
    )


def check_dependencies(config: Config):
    """Check that required tools and files are available."""
    # Check ffmpeg
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg not found. Install it: sudo apt install ffmpeg")
        sys.exit(1)

    # Check Kokoro model files
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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy library logs
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)

    config = parse_args()
    check_dependencies(config)
    config.ensure_dirs()

    start_time = time.time()

    # --- Step 1: Extract chapters ---
    logger.info(f"Extracting chapters from {config.epub_path.name}...")
    all_chapters = extract_chapters(config.epub_path)
    cover_art = extract_cover(config.epub_path)

    # Save cover
    if cover_art:
        cover_path = config.output_dir / "cover.png"
        cover_path.write_bytes(cover_art)
        logger.info("Cover image saved")

    # Filter chapters if specified
    chapters = all_chapters
    if config.chapters:
        chapters = [ch for ch in all_chapters if ch.number in config.chapters]
        logger.info(f"Processing {len(chapters)}/{len(all_chapters)} chapters: {config.chapters}")

    total_chapters = len(all_chapters)

    # --- Step 2: Initialize TTS ---
    logger.info("Initializing Kokoro TTS...")
    synth = Synthesizer(
        model_path=config.kokoro_model,
        voices_path=config.kokoro_voices,
        voice=config.voice,
        speed=config.speed,
    )

    # --- Step 3: Process each chapter ---
    manifest_chapters = []

    for chapter in chapters:
        ch_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Chapter {chapter.number}: {chapter.title}")
        logger.info(f"{'='*60}")

        # Check idempotency
        safe_title = "_".join(
            "".join(c if c.isalnum() or c in " -" else "_" for c in chapter.title).split()
        )
        mp3_filename = f"{chapter.number:02d}_{safe_title}.mp3"
        mp3_path = config.audio_dir / mp3_filename
        pdf_filename = f"{chapter.number:02d}_{safe_title}_companion.pdf"
        pdf_path = config.companions_dir / pdf_filename

        if mp3_path.exists() and pdf_path.exists() and not config.force:
            logger.info(f"Skipping chapter {chapter.number} (outputs exist, use --force to regenerate)")

            manifest_chapters.append({
                "index": chapter.number,
                "title": chapter.title,
                "audio_file": f"audio/{mp3_path.name}",
                "companion_file": f"companions/{pdf_path.name}",
                "figures": len(chapter.figures),
                "code_blocks": len(chapter.code_blocks),
                "math_formulas": len(chapter.math_formulas),
            })
            continue

        # a. Clean HTML → TTS text
        logger.info("Cleaning text for TTS...")
        clean_text = clean_chapter(chapter.raw_html)
        char_count = len(clean_text)
        logger.info(f"Cleaned text: {char_count} chars")

        # b. Chunk text
        chunks = chunk_text(clean_text, config.max_chunk_chars)
        logger.info(f"Split into {len(chunks)} chunks")

        # c. Synthesize → WAV
        chapter_temp = config.temp_dir / f"ch{chapter.number:02d}"
        chapter_temp.mkdir(parents=True, exist_ok=True)

        wav_paths = synth.synthesize_chapter(
            chunks=chunks,
            chapter_num=chapter.number,
            temp_dir=chapter_temp,
            lang=config.lang,
            force=config.force,
        )

        if not wav_paths:
            logger.error(f"No audio generated for chapter {chapter.number}, skipping")
            continue

        # d. Assemble → MP3
        logger.info("Assembling MP3...")
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

        # e. Generate companion PDF
        logger.info("Generating companion PDF...")
        # Save chapter images to assets dir for PDF generation
        for src, img_data in chapter.images.items():
            img_path = config.assets_dir / Path(src).name
            img_path.write_bytes(img_data)

        pdf_path = generate_companion(
            chapter=chapter,
            output_dir=config.companions_dir,
            book_title=config.book_title,
        )

        # f. Cleanup temp WAVs
        shutil.rmtree(chapter_temp, ignore_errors=True)

        ch_elapsed = time.time() - ch_start
        logger.info(f"Chapter {chapter.number} done in {ch_elapsed / 60:.1f} min")

        manifest_chapters.append({
            "index": chapter.number,
            "title": chapter.title,
            "audio_file": f"audio/{mp3_path.name}",
            "companion_file": f"companions/{pdf_path.name}",
            "chars": char_count,
            "chunks": len(chunks),
            "figures": len(chapter.figures),
            "code_blocks": len(chapter.code_blocks),
            "math_formulas": len(chapter.math_formulas),
        })

    # --- Step 4: Write manifest ---
    manifest = {
        "title": config.book_title,
        "author": config.book_author,
        "generated": datetime.now().isoformat(),
        "voice": config.voice,
        "speed": config.speed,
        "chapters": sorted(manifest_chapters, key=lambda c: c["index"]),
    }

    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Manifest written to {manifest_path}")

    # --- Step 5: Optional upload ---
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

    # --- Summary ---
    elapsed = time.time() - start_time
    mp3_files = list(config.audio_dir.glob("*.mp3"))
    pdf_files = list(config.companions_dir.glob("*.pdf"))
    total_size = sum(f.stat().st_size for f in mp3_files + pdf_files)

    print(f"\n{'='*60}")
    print(f"DONE in {elapsed / 60:.1f} minutes")
    print(f"  Audio files:     {len(mp3_files)} MP3s in {config.audio_dir}")
    print(f"  Companion PDFs:  {len(pdf_files)} PDFs in {config.companions_dir}")
    print(f"  Total size:      {total_size / 1024 / 1024:.1f} MB")
    print(f"  Manifest:        {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
