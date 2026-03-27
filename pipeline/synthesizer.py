"""Text chunks → WAV audio files using Kokoro TTS with parallel synthesis."""

import gc
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class WavResult:
    path: Path
    duration: float   # seconds
    chunk: Chunk


class Synthesizer:
    """Kokoro TTS wrapper with GPU support, parallel synthesis, and VRAM management."""

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        voice: str,
        speed: float,
        num_workers: int = 4,
    ):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        if "ONNX_PROVIDER" not in os.environ:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
                logger.info("CUDA GPU detected, enabling GPU acceleration")

        ort.set_default_logger_severity(3)

        logger.info("Loading Kokoro TTS model...")
        self.kokoro = Kokoro(model_path, voices_path)
        self.voice = voice
        self.speed = speed
        self.num_workers = num_workers
        self._kokoro_lock = threading.Lock()
        self._model_path = model_path
        self._voices_path = voices_path

        provider = self.kokoro.sess.get_providers()[0]
        logger.info(
            f"Kokoro loaded — voice={voice}, speed={speed}, "
            f"workers={num_workers}, provider={provider}"
        )

    def synthesize_chunk(
        self,
        text: str,
        output_path: Path,
        lang: str = "en-us",
    ) -> tuple[Path, float] | None:
        """Synthesize a single text chunk to WAV.

        Returns (output_path, duration_seconds) on success, None on failure.
        """
        try:
            with self._kokoro_lock:
                samples, sample_rate = self.kokoro.create(
                    text=text,
                    voice=self.voice,
                    speed=self.speed,
                    lang=lang,
                )
            sf.write(str(output_path), samples, sample_rate)
            duration = len(samples) / sample_rate
            return output_path, duration
        except Exception as e:
            logger.error(f"Synthesis failed for {output_path.name}: {e}")
            return None

    def synthesize_chapter(
        self,
        chunks: list[Chunk],
        chapter_num: int,
        temp_dir: Path,
        lang: str = "en-us",
        force: bool = False,
    ) -> list[WavResult]:
        """Synthesize all chunks for a chapter.

        Uses parallel workers if num_workers > 1.
        Returns list of WavResult in chunk order (skips failed chunks).
        """
        if self.num_workers > 1:
            return self._synthesize_parallel(chunks, chapter_num, temp_dir, lang, force)
        return self._synthesize_serial(chunks, chapter_num, temp_dir, lang, force)

    def _synthesize_serial(
        self,
        chunks: list[Chunk],
        chapter_num: int,
        temp_dir: Path,
        lang: str,
        force: bool,
    ) -> list[WavResult]:
        results = []
        desc = f"Chapter {chapter_num} TTS"

        for chunk in tqdm(chunks, desc=desc, unit="chunk"):
            chunk_id = f"ch{chapter_num:02d}_chunk{chunk.index:04d}"
            wav_path = temp_dir / f"{chunk_id}.wav"

            if wav_path.exists() and not force:
                info = sf.info(str(wav_path))
                results.append(WavResult(
                    path=wav_path,
                    duration=info.duration,
                    chunk=chunk,
                ))
                continue

            result = self.synthesize_chunk(chunk.text, wav_path, lang=lang)
            if result:
                path, duration = result
                results.append(WavResult(path=path, duration=duration, chunk=chunk))

        logger.info(
            f"Chapter {chapter_num}: synthesized {len(results)}/{len(chunks)} chunks"
        )
        return results

    def _synthesize_parallel(
        self,
        chunks: list[Chunk],
        chapter_num: int,
        temp_dir: Path,
        lang: str,
        force: bool,
    ) -> list[WavResult]:
        results: dict[int, WavResult] = {}
        to_synthesize = []
        desc = f"Chapter {chapter_num} TTS"

        # Check idempotency first
        for chunk in chunks:
            chunk_id = f"ch{chapter_num:02d}_chunk{chunk.index:04d}"
            wav_path = temp_dir / f"{chunk_id}.wav"

            if wav_path.exists() and not force:
                info = sf.info(str(wav_path))
                results[chunk.index] = WavResult(
                    path=wav_path,
                    duration=info.duration,
                    chunk=chunk,
                )
            else:
                to_synthesize.append((chunk, wav_path))

        if to_synthesize:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}
                for chunk, wav_path in to_synthesize:
                    future = executor.submit(
                        self.synthesize_chunk, chunk.text, wav_path, lang
                    )
                    futures[future] = chunk

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc=desc, unit="chunk"
                ):
                    chunk = futures[future]
                    try:
                        result = future.result()
                        if result:
                            path, duration = result
                            results[chunk.index] = WavResult(
                                path=path, duration=duration, chunk=chunk
                            )
                    except Exception as e:
                        logger.error(f"Worker failed for chunk {chunk.index}: {e}")

        # Return in chunk order
        ordered = [results[i] for i in sorted(results.keys())]
        logger.info(
            f"Chapter {chapter_num}: synthesized {len(ordered)}/{len(chunks)} chunks"
        )
        return ordered

    def unload(self):
        """Release Kokoro model and free VRAM."""
        if hasattr(self, 'kokoro') and self.kokoro is not None:
            del self.kokoro
            self.kokoro = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Kokoro TTS unloaded from memory")
