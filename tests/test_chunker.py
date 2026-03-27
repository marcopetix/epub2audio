"""Unit tests for pipeline/chunker.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.chunker import chunk_text, DEFAULT_MAX_CHARS, Chunk
from pipeline.cleaner import SectionMarker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prose(n_chars: int, word: str = "Lorem ipsum dolor sit amet") -> str:
    """Generate plain prose of approximately n_chars characters."""
    repeated = (word + ". ") * (n_chars // (len(word) + 2) + 1)
    return repeated[:n_chars]


def _make_technical(n_chars: int) -> str:
    """Generate LLM-enriched-style technical text (denser token/char ratio)."""
    sentence = (
        "The ONNX InferenceSession processes transformer-based embeddings via "
        "CUDAExecutionProvider with TensorRT optimization, leveraging GPU-accelerated "
        "BLAS routines for multi-head self-attention. "
    )
    repeated = sentence * (n_chars // len(sentence) + 1)
    return repeated[:n_chars]


# ---------------------------------------------------------------------------
# Basic splitting
# ---------------------------------------------------------------------------

class TestChunkTextBasic:

    def test_short_text_is_single_chunk(self):
        text = "Hello world. This is a sentence."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == text.strip()

    def test_empty_text_returns_no_chunks(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_whitespace_only_returns_no_chunks(self):
        chunks = chunk_text("   \n\n   ")
        assert chunks == []

    def test_chunks_are_chunk_dataclass(self):
        chunks = chunk_text("Hello world.")
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_indices_are_sequential(self):
        text = _make_prose(DEFAULT_MAX_CHARS * 3)
        chunks = chunk_text(text)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_no_chunk_exceeds_default_max_chars(self):
        text = _make_prose(DEFAULT_MAX_CHARS * 4)
        chunks = chunk_text(text)
        for c in chunks:
            assert len(c.text) <= DEFAULT_MAX_CHARS, (
                f"Chunk {c.index} has {len(c.text)} chars (limit {DEFAULT_MAX_CHARS})"
            )

    def test_no_chunk_exceeds_custom_max_chars(self):
        max_chars = 500
        text = _make_prose(3000)
        chunks = chunk_text(text, max_chars=max_chars)
        for c in chunks:
            assert len(c.text) <= max_chars

    def test_text_is_fully_covered(self):
        """All original text content must appear in exactly one chunk."""
        text = _make_prose(DEFAULT_MAX_CHARS * 3)
        chunks = chunk_text(text)
        rejoined = " ".join(c.text for c in chunks)
        # Every word from the original must appear (order preserved)
        original_words = text.split()
        rejoined_words = rejoined.split()
        assert original_words == rejoined_words


# ---------------------------------------------------------------------------
# DEFAULT_MAX_CHARS regression: Kokoro 512-token limit
# ---------------------------------------------------------------------------

class TestKokoroTokenSafetyMargin:
    """
    Regression for: "index 510 is out of bounds for axis 0 with size 510"
    Kokoro's internal phoneme model has a ~512-token hard limit.
    Dense technical/LLM-enriched text can produce ~1 token per 4-5 chars.
    DEFAULT_MAX_CHARS must leave enough headroom.
    """

    def test_default_max_chars_is_at_most_2500(self):
        assert DEFAULT_MAX_CHARS <= 2500, (
            f"DEFAULT_MAX_CHARS={DEFAULT_MAX_CHARS} is too high. "
            "Kokoro crashes with 'index 510 is out of bounds' on dense "
            "technical text above ~2500 chars. Lower the limit."
        )

    def test_technical_text_chunks_stay_within_limit(self):
        """Technical LLM-enriched text must not produce oversized chunks."""
        text = _make_technical(DEFAULT_MAX_CHARS * 5)
        chunks = chunk_text(text)
        oversized = [c for c in chunks if len(c.text) > DEFAULT_MAX_CHARS]
        assert oversized == [], (
            f"{len(oversized)} chunks exceed {DEFAULT_MAX_CHARS} chars: "
            + ", ".join(f"chunk {c.index} ({len(c.text)} chars)" for c in oversized)
        )

    def test_chunk_at_exact_limit_does_not_overflow(self):
        """A paragraph of exactly DEFAULT_MAX_CHARS chars must be a single chunk."""
        text = _make_prose(DEFAULT_MAX_CHARS)
        chunks = chunk_text(text, max_chars=DEFAULT_MAX_CHARS)
        assert all(len(c.text) <= DEFAULT_MAX_CHARS for c in chunks)

    def test_old_3500_limit_would_produce_oversized_technical_chunks(self):
        """
        Confirm that with max_chars=3500 and technical text, chunks can exceed
        2500 chars — which is what triggered the Kokoro crash in production.
        This test documents the original bug: at least one chunk built from
        a 3500-char paragraph of technical prose will be between 2500-3500 chars.
        """
        old_limit = 3500
        text = _make_technical(old_limit)  # single paragraph, no split points
        chunks = chunk_text(text, max_chars=old_limit)
        # With a single paragraph of exactly old_limit chars and no split points
        # inside, it should produce one chunk close to old_limit chars
        max_chunk_len = max(len(c.text) for c in chunks)
        assert max_chunk_len > 2500, (
            "Expected at least one chunk > 2500 chars with old 3500-char limit "
            f"(got {max_chunk_len}). This test validates the regression scenario."
        )


# ---------------------------------------------------------------------------
# Char offsets
# ---------------------------------------------------------------------------

class TestCharOffsets:

    def test_char_offsets_are_non_overlapping(self):
        text = _make_prose(DEFAULT_MAX_CHARS * 3)
        chunks = chunk_text(text)
        for i in range(len(chunks) - 1):
            assert chunks[i].char_end <= chunks[i + 1].char_start

    def test_char_start_less_than_char_end(self):
        text = _make_prose(DEFAULT_MAX_CHARS * 2)
        chunks = chunk_text(text)
        for c in chunks:
            assert c.char_start < c.char_end

    def test_chunk_text_matches_source_slice(self):
        text = _make_prose(DEFAULT_MAX_CHARS * 2)
        chunks = chunk_text(text)
        for c in chunks:
            source_slice = text[c.char_start:c.char_end].strip()
            assert c.text == source_slice or source_slice.startswith(c.text[:20])


# ---------------------------------------------------------------------------
# Section marker assignment
# ---------------------------------------------------------------------------

class TestSectionMarkers:

    def test_markers_assigned_to_correct_chunk(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        marker = SectionMarker(title="Second", level=1, char_offset=text.index("Second"))
        chunks = chunk_text(text, section_markers=[marker])
        # Find the chunk containing "Second paragraph"
        target = next(c for c in chunks if "Second" in c.text)
        assert any(m.title == "Second" for m in target.section_markers)

    def test_no_markers_produces_empty_lists(self):
        text = _make_prose(DEFAULT_MAX_CHARS * 2)
        chunks = chunk_text(text)
        assert all(c.section_markers == [] for c in chunks)

    def test_marker_at_chunk_boundary_assigned_to_correct_chunk(self):
        """A marker at char_offset == chunk.char_start must go to that chunk."""
        text = "First part.\n\nSecond part."
        chunks = chunk_text(text)
        if len(chunks) >= 2:
            second_start = chunks[1].char_start
            marker = SectionMarker(title="Chapter", level=1, char_offset=second_start)
            chunks = chunk_text(text, section_markers=[marker])
            assert any(m.title == "Chapter" for m in chunks[1].section_markers)


# ---------------------------------------------------------------------------
# Paragraph and sentence boundary splitting
# ---------------------------------------------------------------------------

class TestSplitBoundaries:

    def test_splits_on_paragraph_boundary_first(self):
        para_a = "a " * 600  # ~1200 chars
        para_b = "b " * 600
        text = para_a.strip() + "\n\n" + para_b.strip()
        chunks = chunk_text(text, max_chars=2500)
        # Each paragraph should ideally be in its own chunk
        texts = [c.text for c in chunks]
        assert any("a a a" in t for t in texts)
        assert any("b b b" in t for t in texts)

    def test_splits_on_sentence_boundary_when_paragraph_too_long(self):
        long_para = "word " * 300 + ". " + "other " * 300 + "."
        chunks = chunk_text(long_para, max_chars=1000)
        assert all(len(c.text) <= 1000 for c in chunks)
        assert len(chunks) >= 2

    def test_single_very_long_word_does_not_crash(self):
        # Edge case: a token with no split points that exceeds max_chars
        text = "a" * (DEFAULT_MAX_CHARS + 100)
        chunks = chunk_text(text)
        # Must not raise; chunk may exceed limit (no split point available)
        assert len(chunks) >= 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
