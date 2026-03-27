"""Split TTS-ready text into chunks respecting sentence boundaries.

Returns Chunk dataclasses with char offsets and section marker assignments,
enabling M4B chapter marker timestamp calculation.
"""

import re
from dataclasses import dataclass, field

from pipeline.cleaner import SectionMarker

# Kokoro's internal phoneme model has a hard ~512-token limit.
# Technical/LLM-enriched text can produce ~1 token per 4-5 chars, so
# 3500 chars can yield 700+ tokens and crash with an off-by-one in
# Kokoro's duration predictor ("index 510 is out of bounds for axis 0
# with size 510").  2500 chars keeps peak token counts safely below 512
# even for dense technical prose.
DEFAULT_MAX_CHARS = 2500

# Sentence boundary: period/question/exclamation followed by whitespace
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


@dataclass
class Chunk:
    text: str
    index: int                                    # 0-based within chapter
    char_start: int = 0                           # Offset in original cleaned text
    char_end: int = 0
    section_markers: list[SectionMarker] = field(default_factory=list)


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    section_markers: list[SectionMarker] | None = None,
) -> list[Chunk]:
    """Split text into Chunk objects of at most max_chars.

    Splitting priority:
    1. Paragraph boundaries (double newlines)
    2. Sentence boundaries (. ! ? followed by whitespace)
    3. Comma/semicolon boundaries (last resort)

    Section markers are assigned to chunks based on char_offset ranges.
    """
    raw_chunks = _split_text(text, max_chars)

    # Build Chunk objects with char offsets
    chunks = []
    search_start = 0
    for i, chunk_text_str in enumerate(raw_chunks):
        # Find this chunk's position in the original text
        idx = text.find(chunk_text_str, search_start)
        if idx < 0:
            # Fallback: use cumulative offset
            idx = search_start
        char_start = idx
        char_end = idx + len(chunk_text_str)
        search_start = char_end

        chunks.append(Chunk(
            text=chunk_text_str,
            index=i,
            char_start=char_start,
            char_end=char_end,
        ))

    # Assign section markers to chunks
    if section_markers:
        for marker in section_markers:
            for chunk in chunks:
                if chunk.char_start <= marker.char_offset < chunk.char_end:
                    chunk.section_markers.append(marker)
                    break

    return chunks


def _split_text(text: str, max_chars: int) -> list[str]:
    """Core splitting algorithm. Returns list of text strings."""
    paragraphs = re.split(r'\n\n+', text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            current = ""

        if len(para) <= max_chars:
            current = para
            continue

        # Paragraph too long: split on sentences
        sentences = SENTENCE_SPLIT.split(para)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = f"{current} {sentence}" if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(sentence) > max_chars:
                    sub_parts = re.split(r'(?<=[,;])\s+', sentence)
                    current = ""
                    for part in sub_parts:
                        candidate = f"{current} {part}" if current else part
                        if len(candidate) <= max_chars:
                            current = candidate
                        else:
                            if current:
                                chunks.append(current.strip())
                            # Hard-split last resort: slice at max_chars boundaries
                            if len(part) > max_chars:
                                for i in range(0, len(part), max_chars):
                                    chunks.append(part[i:i + max_chars].strip())
                                current = ""
                            else:
                                current = part
                else:
                    current = sentence

    if current and current.strip():
        chunks.append(current.strip())

    return chunks
