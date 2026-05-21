from pathlib import Path
import subprocess

from app.audio import create_placeholder, normalize_audio_for_model, title_from_filename


def test_title_from_filename() -> None:
    assert title_from_filename("example.track.mp3") == "example.track"
    assert title_from_filename("") == "Untitled"


def test_placeholder_is_written(tmp_path: Path) -> None:
    path = tmp_path / "placeholder.png"
    create_placeholder(path, "abcdef")
    assert path.exists()
    assert path.stat().st_size > 0


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
