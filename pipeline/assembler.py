"""WAV chunks → single MP3 per chapter with ID3 metadata."""

import logging
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

logger = logging.getLogger(__name__)


def _sanitize_title(title: str) -> str:
    """Make title safe for filenames."""
    # Keep alphanumeric, spaces, hyphens
    safe = "".join(c if c.isalnum() or c in " -" else "_" for c in title)
    # Collapse multiple underscores/spaces
    safe = "_".join(safe.split())
    return safe


def assemble_chapter(
    wav_paths: list[Path],
    chapter_num: int,
    chapter_title: str,
    output_dir: Path,
    *,
    bitrate: str = "128k",
    book_title: str = "Generative AI Design Patterns",
    book_author: str = "Lakshmanan, Hapke",
    book_year: str = "2025",
    total_chapters: int = 10,
    cover_art: bytes | None = None,
) -> Path:
    """Concatenate WAV files into a single MP3 with ID3 tags.

    Returns path to the output MP3 file.
    """
    if not wav_paths:
        raise ValueError(f"No WAV files for chapter {chapter_num}")

    # Concatenate WAVs
    logger.info(f"Assembling chapter {chapter_num}: {len(wav_paths)} chunks → MP3")
    combined = AudioSegment.empty()
    for wav_path in wav_paths:
        segment = AudioSegment.from_wav(str(wav_path))
        combined += segment

    # Export MP3
    safe_title = _sanitize_title(chapter_title)
    mp3_filename = f"{chapter_num:02d}_{safe_title}.mp3"
    mp3_path = output_dir / mp3_filename

    combined.export(
        str(mp3_path),
        format="mp3",
        bitrate=bitrate,
    )

    # Add ID3 tags
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

    # Embed cover art if available
    if cover_art:
        audio.add(APIC(
            encoding=3,
            mime="image/png",
            type=3,  # Cover (front)
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
