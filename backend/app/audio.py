from pathlib import Path
import subprocess
from uuid import uuid4

from fastapi import UploadFile
import imageio_ffmpeg
from mutagen import File as MutagenFile
from PIL import Image, ImageDraw


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


async def save_upload(upload: UploadFile, audio_dir: Path) -> tuple[str, Path]:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        suffix = ".audio"
    track_id = uuid4().hex
    target = audio_dir / f"{track_id}{suffix}"

    with target.open("wb") as out:
        while chunk := await upload.read(1024 * 1024):
            out.write(chunk)

    return track_id, target


def title_from_filename(filename: str | None) -> str:
    stem = Path(filename or "Untitled").stem.strip()
    return stem or "Untitled"


def normalize_audio_for_model(input_path: Path, target_path: Path, sample_rate: int) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(target_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffmpeg could not decode the uploaded audio"
        raise ValueError(f"Audio normalization failed: {detail}")
    return target_path


def extract_metadata(audio_path: Path) -> tuple[str | None, str | None, bytes | None]:
    audio = MutagenFile(audio_path)
    if audio is None or audio.tags is None:
        return None, None, None

    artist = None
    album = None
    art = None

    for key, value in audio.tags.items():
        normalized = str(key).lower()
        if artist is None and normalized in {"artist", "\xa9art", "tpe1"}:
            artist = _first_text(value)
        if album is None and normalized in {"album", "\xa9alb", "talb"}:
            album = _first_text(value)
        if art is None and (normalized.startswith("apic") or normalized in {"covr", "metadata_block_picture"}):
            art = _extract_art_bytes(value)

    return artist, album, art


def write_art_or_placeholder(track_id: str, art_bytes: bytes | None, art_dir: Path) -> Path:
    if art_bytes:
        path = art_dir / f"{track_id}{_image_suffix(art_bytes)}"
        path.write_bytes(art_bytes)
        return path

    path = art_dir / f"{track_id}.png"
    create_placeholder(path, track_id)
    return path


def cached_art_thumbnail(art_path: Path, *, size: int = 256) -> Path:
    thumb_dir = art_path.parent / "thumbs"
    thumb_path = thumb_dir / f"{art_path.stem}.jpg"
    if (
        thumb_path.exists()
        and art_path.exists()
        and thumb_path.stat().st_mtime >= art_path.stat().st_mtime
    ):
        return thumb_path

    try:
        thumb_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(art_path) as image:
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            image.convert("RGB").save(thumb_path, format="JPEG", quality=82, optimize=True)
        return thumb_path
    except Exception:
        return art_path


def create_placeholder(path: Path, seed: str, size: int = 512) -> None:
    image = Image.new("RGB", (size, size), "#111111")
    draw = ImageDraw.Draw(image)
    band_count = 8
    base = int(seed[:2], 16) if seed[:2] else 0

    for idx in range(band_count):
        shade = 28 + ((base + idx * 19) % 130)
        x0 = int(idx * size / band_count)
        x1 = int((idx + 1) * size / band_count)
        draw.rectangle((x0, 0, x1, size), fill=(shade, shade, shade))

    inset = size // 5
    draw.ellipse((inset, inset, size - inset, size - inset), outline="#eeeeee", width=6)
    image.save(path)


def _first_text(value: object) -> str | None:
    if isinstance(value, list) and value:
        return str(value[0])
    text = str(value).strip()
    return text or None


def _extract_art_bytes(value: object) -> bytes | None:
    if hasattr(value, "data"):
        data = getattr(value, "data")
        return data if isinstance(data, bytes) else None
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, bytes):
            return first
        if hasattr(first, "data"):
            data = getattr(first, "data")
            return data if isinstance(data, bytes) else None
    if isinstance(value, bytes):
        return value
    return None


def _image_suffix(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"
