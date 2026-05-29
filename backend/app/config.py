from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path("data")
    model_id: str = "OpenMuQ/MuQ-MuLan-large"
    sample_rate: int = 24_000

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def art_dir(self) -> Path:
        return self.data_dir / "art"

    @property
    def stem_dir(self) -> Path:
        return self.data_dir / "stems"

    @property
    def separator_model_dir(self) -> Path:
        return self.data_dir / "models" / "audio-separator"

    @property
    def cover_model_dir(self) -> Path:
        return self.data_dir / "models" / "discogs-vinet"

    @property
    def cover_alignment_model_dir(self) -> Path:
        return self.data_dir / "models" / "clews"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "app.sqlite"

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_data_dirs(settings: Settings) -> None:
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    settings.art_dir.mkdir(parents=True, exist_ok=True)
    settings.stem_dir.mkdir(parents=True, exist_ok=True)
    settings.separator_model_dir.mkdir(parents=True, exist_ok=True)
    settings.cover_model_dir.mkdir(parents=True, exist_ok=True)
    settings.cover_alignment_model_dir.mkdir(parents=True, exist_ok=True)
    settings.lancedb_dir.mkdir(parents=True, exist_ok=True)
