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

    # Audio
    mp3_bitrate: str = "128k"
    sample_rate: int = 24000

    # Book metadata (ID3 tags)
    book_title: str = "Generative AI Design Patterns"
    book_author: str = "Lakshmanan, Hapke"
    book_year: str = "2025"

    # Processing
    max_chunk_chars: int = 3500
    chapters: list[int] | None = None  # None = all
    force: bool = False

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

    def ensure_dirs(self) -> None:
        for d in [self.audio_dir, self.companions_dir, self.assets_dir, self.temp_dir]:
            d.mkdir(parents=True, exist_ok=True)
