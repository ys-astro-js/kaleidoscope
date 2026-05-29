from pathlib import Path
import subprocess
import tempfile
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


def normalize_audio_for_playback(input_path: Path, target_path: Path, sample_rate: int) -> Path:
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
        "2",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(target_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffmpeg could not decode the uploaded audio"
        raise ValueError(f"Audio playback normalization failed: {detail}")
    return target_path


def align_audio_for_playback(
    reference_path: Path,
    input_path: Path,
    target_path: Path,
    sample_rate: int,
) -> Path:
    import librosa
    import numpy as np
    import soundfile as sf

    target_path.parent.mkdir(parents=True, exist_ok=True)
    reference, _ = librosa.load(reference_path, sr=sample_rate, mono=True)
    candidate, _ = librosa.load(input_path, sr=sample_rate, mono=True)
    if len(reference) == 0 or len(candidate) == 0:
        raise ValueError("Audio playback alignment failed: empty audio")

    offset_samples = _estimate_alignment_offset(reference, candidate, sample_rate)
    waveform, _ = librosa.load(input_path, sr=sample_rate, mono=False)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    adjusted = _shift_audio_channels(waveform, offset_samples, len(reference))

    with tempfile.NamedTemporaryFile(
        suffix=target_path.suffix,
        dir=target_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        sf.write(temp_path, adjusted.T, sample_rate)
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
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


def _estimate_alignment_offset(reference, candidate, sample_rate: int) -> int:
    import librosa
    import numpy as np

    hop_length = 512
    analysis_samples = min(len(reference), len(candidate), sample_rate * 90)
    reference_envelope = _normalized_envelope(
        librosa.feature.rms(
            y=reference[:analysis_samples],
            frame_length=2048,
            hop_length=hop_length,
        )[0]
    )
    candidate_envelope = _normalized_envelope(
        librosa.feature.rms(
            y=candidate[:analysis_samples],
            frame_length=2048,
            hop_length=hop_length,
        )[0]
    )
    if len(reference_envelope) < 4 or len(candidate_envelope) < 4:
        return 0

    max_lag = min(
        int(2.0 * sample_rate / hop_length),
        len(reference_envelope) - 1,
        len(candidate_envelope) - 1,
    )
    best_lag = 0
    best_score = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            ref = reference_envelope[: len(reference_envelope) - lag]
            cand = candidate_envelope[lag : lag + len(ref)]
        else:
            cand = candidate_envelope[: len(candidate_envelope) + lag]
            ref = reference_envelope[-lag : -lag + len(cand)]
        if len(ref) < 8 or len(cand) < 8:
            continue
        denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
        if denominator <= 1e-8:
            continue
        score = float(np.dot(ref, cand) / denominator)
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_score < 0.15:
        return 0
    return best_lag * hop_length


def _normalized_envelope(envelope):
    import numpy as np

    centered = envelope.astype(np.float32) - float(np.mean(envelope))
    norm = float(np.linalg.norm(centered))
    if norm <= 1e-8:
        return centered
    return centered / norm


def _shift_audio_channels(waveform, offset_samples: int, target_samples: int):
    import numpy as np

    if offset_samples > 0:
        shifted = waveform[:, offset_samples:]
    elif offset_samples < 0:
        shifted = np.pad(waveform, ((0, 0), (-offset_samples, 0)))
    else:
        shifted = waveform

    if shifted.shape[1] < target_samples:
        shifted = np.pad(shifted, ((0, 0), (0, target_samples - shifted.shape[1])))
    return shifted[:, :target_samples]


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
