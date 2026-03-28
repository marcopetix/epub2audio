"""WAV chunks → MP3 per chapter and/or single M4B audiobook with chapter markers."""

import logging
import subprocess
import tempfile
from pathlib import Path

from pydub import AudioSegment
from mutagen.id3 import (
    APIC,
    ID3,
    TALB,
    TCON,
    TDRC,
    TIT2,
    TPE1,
    TRCK,
)
from mutagen.mp4 import MP4, MP4Cover

from pipeline.extractor import Chapter
from pipeline.synthesizer import WavResult

logger = logging.getLogger(__name__)


def _sanitize_title(title: str) -> str:
    """Make title safe for filenames."""
    safe = "".join(c if c.isalnum() or c in " -" else "_" for c in title)
    safe = "_".join(safe.split())
    return safe


def assemble_chapter(
    wav_paths: list[Path],
    chapter_num: int,
    chapter_title: str,
    output_dir: Path,
    *,
    bitrate: str = "128k",
    book_title: str = "",
    book_author: str = "",
    book_year: str = "",
    total_chapters: int = 10,
    cover_art: bytes | None = None,
) -> Path:
    """Concatenate WAV files into a single MP3 with ID3 tags."""
    if not wav_paths:
        raise ValueError(f"No WAV files for chapter {chapter_num}")

    logger.info(f"Assembling chapter {chapter_num}: {len(wav_paths)} chunks -> MP3")
    combined = AudioSegment.empty()
    for wav_path in wav_paths:
        segment = AudioSegment.from_wav(str(wav_path))
        combined += segment

    safe_title = _sanitize_title(chapter_title)
    mp3_filename = f"{chapter_num:02d}_{safe_title}.mp3"
    mp3_path = output_dir / mp3_filename

    combined.export(str(mp3_path), format="mp3", bitrate=bitrate)

    try:
        audio = ID3(str(mp3_path))
    except Exception:
        audio = ID3()

    audio.add(TIT2(encoding=3, text=f"Chapter {chapter_num} - {chapter_title}"))
    audio.add(TPE1(encoding=3, text=book_author))
    audio.add(TALB(encoding=3, text=book_title))
    audio.add(TRCK(encoding=3, text=f"{chapter_num}/{total_chapters}"))
    audio.add(TCON(encoding=3, text="Audiobook"))
    audio.add(TDRC(encoding=3, text=book_year))

    if cover_art:
        audio.add(APIC(
            encoding=3,
            mime="image/png",
            type=3,
            desc="Cover",
            data=cover_art,
        ))

    audio.save(str(mp3_path))

    duration_sec = len(combined) / 1000.0
    logger.info(
        f"Chapter {chapter_num}: {mp3_path.name} — "
        f"{duration_sec / 60:.1f} min, {mp3_path.stat().st_size / 1024 / 1024:.1f} MB"
    )

    return mp3_path


def assemble_m4b(
    wav_results: dict[int, list[WavResult]],
    chapters: list[Chapter],
    output_dir: Path,
    config,
    cover_art: bytes | None = None,
) -> Path | None:
    """Assemble all chapters into a single M4B audiobook with chapter markers.

    Steps:
    1. Concatenate all chapter WAVs into a single WAV
    2. Calculate chapter timestamps
    3. Write ffmpeg metadata with chapter markers
    4. Encode to M4B (AAC in MP4 container) with metadata
    5. Add cover art and MP4 tags
    """
    if not wav_results:
        logger.error("No WAV results to assemble")
        return None

    # Sort chapters by number
    sorted_chapters = sorted(chapters, key=lambda c: c.number)

    # Step 1: Concatenate all WAVs
    logger.info("Concatenating all chapter WAVs...")
    combined = AudioSegment.empty()
    chapter_markers = []  # (title, start_ms, end_ms)
    current_ms = 0

    for chapter in sorted_chapters:
        if chapter.number not in wav_results:
            continue

        chapter_start_ms = current_ms
        results = wav_results[chapter.number]

        # Add section markers within the chapter
        sub_markers = []
        chunk_offset_ms = current_ms

        for wav_result in results:
            # Check for section markers in this chunk
            for marker in wav_result.chunk.section_markers:
                if marker.level <= 2:
                    sub_markers.append((marker.title, chunk_offset_ms))

            segment = AudioSegment.from_wav(str(wav_result.path))
            combined += segment
            chunk_offset_ms += len(segment)

        chapter_end_ms = chunk_offset_ms

        # Main chapter marker
        chapter_markers.append((
            f"Chapter {chapter.number}: {chapter.title}",
            chapter_start_ms,
            chapter_end_ms,
        ))

        # Sub-chapter markers (h2 headings within chapters)
        for title, start_ms in sub_markers:
            # Skip if it's the same as the chapter title
            if title != chapter.title:
                chapter_markers.append((title, start_ms, chapter_end_ms))

        current_ms = chapter_end_ms

    if not combined:
        logger.error("No audio content to assemble")
        return None

    total_duration_ms = len(combined)
    logger.info(f"Total audio: {total_duration_ms / 1000 / 60:.1f} minutes")

    # Sort markers by start time and fix end times
    chapter_markers.sort(key=lambda m: m[1])
    # Fix end times: each marker ends where the next one starts
    fixed_markers = []
    for i, (title, start_ms, end_ms) in enumerate(chapter_markers):
        if i + 1 < len(chapter_markers):
            end_ms = chapter_markers[i + 1][1]
        else:
            end_ms = total_duration_ms
        fixed_markers.append((title, start_ms, end_ms))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 2: Export combined WAV
        combined_wav = tmpdir / "combined.wav"
        combined.export(str(combined_wav), format="wav")

        # Step 3: Write ffmpeg metadata file
        metadata_file = tmpdir / "metadata.txt"
        metadata_lines = [";FFMETADATA1", f"title={config.book_title}", f"artist={config.book_author}", ""]
        for title, start_ms, end_ms in fixed_markers:
            metadata_lines.extend([
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={title}",
                "",
            ])
        metadata_file.write_text("\n".join(metadata_lines))

        # Step 4: Encode to M4B
        safe_book = _sanitize_title(config.book_title)
        m4b_path = output_dir / f"{safe_book}.m4b"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(combined_wav),
            "-i", str(metadata_file),
            "-map_metadata", "1",
            "-c:a", "aac",
            "-b:a", config.aac_bitrate,
            "-movflags", "+faststart",
            str(m4b_path),
        ]

        logger.info("Encoding M4B with ffmpeg...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr[-500:]}")
            return None

    # Step 5: Add MP4 tags and cover art
    try:
        audio = MP4(str(m4b_path))
        audio["\xa9nam"] = [config.book_title]
        audio["\xa9ART"] = [config.book_author]
        audio["\xa9alb"] = [config.book_title]
        audio["\xa9day"] = [config.book_year]
        audio["\xa9gen"] = ["Audiobook"]

        if cover_art:
            audio["covr"] = [MP4Cover(cover_art, imageformat=MP4Cover.FORMAT_PNG)]

        audio.save()
    except Exception as e:
        logger.warning(f"Failed to add MP4 tags: {e}")

    size_mb = m4b_path.stat().st_size / 1024 / 1024
    logger.info(
        f"M4B: {m4b_path.name} — "
        f"{total_duration_ms / 1000 / 60:.1f} min, {size_mb:.1f} MB, "
        f"{len(fixed_markers)} chapter markers"
    )

    return m4b_path
