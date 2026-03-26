"""Split TTS-ready text into chunks respecting sentence boundaries."""

import re

# Kokoro handles up to ~4000 chars well; leave margin
DEFAULT_MAX_CHARS = 3500

# Sentence boundary: period/question/exclamation followed by whitespace
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split text into chunks of at most max_chars.

    Splitting priority:
    1. Paragraph boundaries (double newlines)
    2. Sentence boundaries (. ! ? followed by whitespace)
    3. Comma/semicolon boundaries (last resort)

    Never splits mid-word.
    """
    paragraphs = re.split(r'\n\n+', text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph fits in current chunk, append
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue

        # Flush current chunk if non-empty
        if current:
            chunks.append(current.strip())
            current = ""

        # If paragraph itself fits in a single chunk
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
                # If single sentence exceeds max, split on comma/semicolon
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
                            current = part
                else:
                    current = sentence

    # Flush remaining
    if current and current.strip():
        chunks.append(current.strip())

    return chunks
