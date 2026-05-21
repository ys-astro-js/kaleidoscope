import asyncio
from pathlib import Path
import subprocess

from app.audio import (
    cached_art_thumbnail,
    create_placeholder,
    normalize_audio_for_model,
    save_upload,
    title_from_filename,
)

JAPANESE_FILENAME = "\u6771\u4eac\u4e8b\u5909 - \u7fa4\u9752\u65e5\u548c.mp3"
JAPANESE_TITLE = "\u6771\u4eac\u4e8b\u5909 - \u7fa4\u9752\u65e5\u548c"
KOREAN_FILENAME = "\ubc24\uc591\uac31.flac"
KOREAN_TITLE = "\ubc24\uc591\uac31"


def test_title_from_filename() -> None:
    assert title_from_filename("example.track.mp3") == "example.track"
    assert title_from_filename("") == "Untitled"
    assert title_from_filename(JAPANESE_FILENAME) == JAPANESE_TITLE
    assert title_from_filename(KOREAN_FILENAME) == KOREAN_TITLE


def test_save_upload_handles_multibyte_filename(tmp_path: Path) -> None:
    class Upload:
        filename = JAPANESE_FILENAME

        def __init__(self) -> None:
            self._chunks = [b"audio", b""]

        async def read(self, size: int) -> bytes:
            return self._chunks.pop(0)

    track_id, path = asyncio.run(save_upload(Upload(), tmp_path))

    assert path.exists()
    assert path.read_bytes() == b"audio"
    assert path.name == f"{track_id}.mp3"


def test_placeholder_is_written(tmp_path: Path) -> None:
    path = tmp_path / "placeholder.png"
    create_placeholder(path, "abcdef")
    assert path.exists()
    assert path.stat().st_size > 0


def test_cached_art_thumbnail_creates_smaller_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "cover.png"
    create_placeholder(path, "abcdef", size=512)

    thumbnail = cached_art_thumbnail(path, size=128)

    assert thumbnail.exists()
    assert thumbnail.suffix == ".jpg"
    assert thumbnail.parent.name == "thumbs"


def test_normalize_audio_for_model_uses_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_ffmpeg() -> str:
        return "/tmp/ffmpeg"

    def fake_run(command, capture_output, text, check):
        calls.append(command)
        Path(command[-1]).write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("app.audio.imageio_ffmpeg.get_ffmpeg_exe", fake_ffmpeg)
    monkeypatch.setattr("app.audio.subprocess.run", fake_run)

    input_path = tmp_path / "song.m4a"
    input_path.write_bytes(b"m4a")
    output_path = normalize_audio_for_model(input_path, tmp_path / "song.wav", 24_000)

    assert output_path.exists()
    assert "-ac" in calls[0]
    assert "1" in calls[0]
    assert "-ar" in calls[0]
    assert "24000" in calls[0]
