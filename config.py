from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Input
    epub_path: Path
    output_dir: Path = Path("./audiobook")

    # TTS
    voice: str = "af_nicole"
    speed: float = 1.1
    lang: str = "en-us"
    kokoro_model: str = "kokoro-v1.0.onnx"
    kokoro_voices: str = "voices-v1.0.bin"
    num_tts_workers: int = 4

    # Audio output
    output_format: str = "m4b"       # "m4b", "mp3", "both"
    mp3_bitrate: str = "128k"
    aac_bitrate: str = "128k"
    sample_rate: int = 24000

    # Book metadata (auto-detected from EPUB, overridable)
    book_title: str = ""
    book_author: str = ""
    book_year: str = ""

    # LLM enrichment
    enable_llm: bool = True
    llm_model: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"

    # Companion output
    companion_format: str = "both"   # "pdf", "html", "both"

    # Quality check
    enable_qc: bool = False
    whisper_model: str = "medium"

    # Processing
    max_chunk_chars: int = 3500
    chapters: list[int] | None = None  # None = all
    force: bool = False
    dry_run: bool = False

    # Pronunciation
    pronunciation_file: str = "pronunciation.json"

    # Upload
    upload_gdrive: bool = False
    gdrive_folder: str = "Audiobooks"

    # Derived paths
    @property
    def audio_dir(self) -> Path:
        return self.output_dir / "audio"

    @property
    def companions_dir(self) -> Path:
        return self.output_dir / "companions"

    @property
    def assets_dir(self) -> Path:
        return self.output_dir / "assets"

    @property
    def temp_dir(self) -> Path:
        return self.output_dir / ".temp"

    @property
    def llm_cache_dir(self) -> Path:
        return self.output_dir / ".llm_cache"

    def ensure_dirs(self) -> None:
        dirs = [self.audio_dir, self.companions_dir, self.assets_dir, self.temp_dir]
        if self.enable_llm:
            dirs.append(self.llm_cache_dir)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
