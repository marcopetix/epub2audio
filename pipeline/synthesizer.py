"""Text chunks → WAV audio files using Kokoro TTS."""

import logging
import os
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Synthesizer:
    """Kokoro TTS wrapper with GPU support and error resilience."""

    def __init__(self, model_path: str, voices_path: str, voice: str, speed: float):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        # Enable GPU if available (kokoro-onnx checks ONNX_PROVIDER env var)
        if "ONNX_PROVIDER" not in os.environ:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
                logger.info("CUDA GPU detected, enabling GPU acceleration")

        # Suppress C++ level ONNX Runtime warnings (memcpy nodes, provider assignments)
        ort.set_default_logger_severity(3)  # 3 = ERROR only

        logger.info("Loading Kokoro TTS model...")
        self.kokoro = Kokoro(model_path, voices_path)
        self.voice = voice
        self.speed = speed

        # Report which provider is active
        provider = self.kokoro.sess.get_providers()[0]
        logger.info(f"Kokoro loaded — voice={voice}, speed={speed}, provider={provider}")

    def synthesize_chunk(
        self,
        text: str,
        output_path: Path,
        lang: str = "en-us",
    ) -> Path | None:
        """Synthesize a single text chunk to WAV.

        Returns the output path on success, None on failure.
        """
        try:
            samples, sample_rate = self.kokoro.create(
                text=text,
                voice=self.voice,
                speed=self.speed,
                lang=lang,
            )
            sf.write(str(output_path), samples, sample_rate)
            return output_path
        except Exception as e:
            logger.error(f"Synthesis failed for {output_path.name}: {e}")
            return None

    def synthesize_chapter(
        self,
        chunks: list[str],
        chapter_num: int,
        temp_dir: Path,
        lang: str = "en-us",
        force: bool = False,
    ) -> list[Path]:
        """Synthesize all chunks for a chapter.

        Returns list of WAV file paths (skips failed chunks).
        """
        wav_paths = []
        desc = f"Chapter {chapter_num} TTS"

        for i, chunk in enumerate(tqdm(chunks, desc=desc, unit="chunk")):
            chunk_id = f"ch{chapter_num:02d}_chunk{i:04d}"
            wav_path = temp_dir / f"{chunk_id}.wav"

            # Idempotency: skip if WAV exists
            if wav_path.exists() and not force:
                wav_paths.append(wav_path)
                continue

            result = self.synthesize_chunk(chunk, wav_path, lang=lang)
            if result:
                wav_paths.append(result)

        logger.info(
            f"Chapter {chapter_num}: synthesized {len(wav_paths)}/{len(chunks)} chunks"
        )
        return wav_paths
