"""Whisper-based quality check: transcribe audio and compare to source text.

Uses faster-whisper (CTranslate2) for GPU-accelerated transcription.
Compares transcription against original chunk text to detect TTS errors.
"""

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.chunker import Chunk
from pipeline.synthesizer import WavResult

logger = logging.getLogger(__name__)


@dataclass
class ChunkQC:
    chunk_index: int
    similarity: float
    original_words: int
    transcribed_words: int
    mismatches: list[str] = field(default_factory=list)


@dataclass
class QCReport:
    chapter_num: int
    similarity_ratio: float
    chunk_reports: list[ChunkQC] = field(default_factory=list)
    total_words_original: int = 0
    total_words_transcribed: int = 0

    def to_dict(self) -> dict:
        return {
            "chapter": self.chapter_num,
            "similarity": round(self.similarity_ratio, 4),
            "total_words_original": self.total_words_original,
            "total_words_transcribed": self.total_words_transcribed,
            "worst_chunks": [
                {
                    "chunk": c.chunk_index,
                    "similarity": round(c.similarity, 4),
                    "mismatches": c.mismatches[:5],
                }
                for c in sorted(self.chunk_reports, key=lambda x: x.similarity)[:10]
            ],
        }


def _word_similarity(original: str, transcribed: str) -> float:
    """Compute word-level similarity ratio using difflib."""
    import difflib
    orig_words = original.lower().split()
    trans_words = transcribed.lower().split()
    if not orig_words:
        return 1.0
    matcher = difflib.SequenceMatcher(None, orig_words, trans_words)
    return matcher.ratio()


def _find_mismatches(original: str, transcribed: str, threshold: int = 3) -> list[str]:
    """Find word sequences that differ between original and transcription."""
    import difflib
    orig_words = original.lower().split()
    trans_words = transcribed.lower().split()
    matcher = difflib.SequenceMatcher(None, orig_words, trans_words)

    mismatches = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete") and (i2 - i1) > threshold:
            orig_segment = " ".join(orig_words[i1:i2])
            trans_segment = " ".join(trans_words[j1:j2]) if tag == "replace" else ""
            mismatches.append(f"'{orig_segment}' -> '{trans_segment}'")

    return mismatches


class QualityChecker:
    """Whisper-based audio quality checker."""

    def __init__(self, model_size: str = "medium"):
        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper {model_size} model...")
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        logger.info("Whisper model loaded")

    def check_chapter(
        self,
        wav_results: list[WavResult],
        chunks: list[Chunk],
    ) -> QCReport:
        """Transcribe audio and compare with original text."""
        chunk_reports = []
        total_orig_words = 0
        total_trans_words = 0
        weighted_similarity = 0.0

        for wav_result in wav_results:
            chunk = wav_result.chunk

            # Transcribe
            segments, info = self.model.transcribe(
                str(wav_result.path), language="en"
            )
            transcribed = " ".join(s.text.strip() for s in segments)

            # Compare
            similarity = _word_similarity(chunk.text, transcribed)
            mismatches = _find_mismatches(chunk.text, transcribed)

            orig_words = len(chunk.text.split())
            trans_words = len(transcribed.split())
            total_orig_words += orig_words
            total_trans_words += trans_words
            weighted_similarity += similarity * orig_words

            chunk_reports.append(ChunkQC(
                chunk_index=chunk.index,
                similarity=similarity,
                original_words=orig_words,
                transcribed_words=trans_words,
                mismatches=mismatches,
            ))

        overall_sim = weighted_similarity / total_orig_words if total_orig_words else 0.0

        # Log warnings for poor chunks
        for cr in chunk_reports:
            if cr.similarity < 0.85:
                logger.warning(
                    f"  Chunk {cr.chunk_index}: low similarity {cr.similarity:.2%}"
                )

        return QCReport(
            chapter_num=wav_results[0].chunk.index if wav_results else 0,
            similarity_ratio=overall_sim,
            chunk_reports=chunk_reports,
            total_words_original=total_orig_words,
            total_words_transcribed=total_trans_words,
        )

    def unload(self):
        """Release Whisper model and free VRAM."""
        del self.model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Whisper model unloaded from VRAM")
