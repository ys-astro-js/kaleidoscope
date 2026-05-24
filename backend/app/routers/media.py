from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse

from app import database
from app.audio import cached_art_thumbnail
from app.dependencies import AppContextDep
from app.models import AudioStem

router = APIRouter(prefix="/api/tracks", tags=["media"])


@router.get("/{track_id}/art")
def get_art(track_id: str, context: AppContextDep) -> FileResponse:
    row = database.get_track(context.conn, track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Track not found")
    art_path = cached_art_thumbnail(Path(row["art_path"]))
    return FileResponse(
        art_path,
        media_type="image/jpeg" if art_path.suffix.lower() == ".jpg" else None,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/{track_id}/audio")
def get_audio(
    track_id: str,
    context: AppContextDep,
    stem: Annotated[AudioStem, Query()] = "original",
    range_header: Annotated[str | None, Header(alias="Range")] = None,
) -> Response:
    row = database.get_track(context.conn, track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Track not found")
    path = _audio_path(row, stem)
    if range_header is None:
        return FileResponse(path, media_type=_audio_media_type(path))
    return range_response(path, range_header, media_type=_audio_media_type(path))


def range_response(path: Path, range_header: str, *, media_type: str = "audio/mpeg") -> StreamingResponse:
    file_size = path.stat().st_size
    start_text, _, end_text = range_header.replace("bytes=", "").partition("-")
    start = int(start_text or 0)
    end = int(end_text) if end_text else file_size - 1
    end = min(end, file_size - 1)
    chunk_size = end - start + 1

    def iter_file():
        with path.open("rb") as file:
            file.seek(start)
            remaining = chunk_size
            while remaining > 0:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iter_file(),
        status_code=206,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        },
        media_type=media_type,
    )


def _audio_path(row, stem: AudioStem) -> Path:
    if stem == "original":
        path = Path(row["audio_path"])
    elif stem == "vocals":
        path = Path(row["vocals_path"]) if row["vocals_path"] else None
    else:
        path = Path(row["instrumental_path"]) if row["instrumental_path"] else None

    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"{stem} audio is not available")
    return path


def _audio_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".flac":
        return "audio/flac"
    if suffix == ".ogg":
        return "audio/ogg"
    if suffix == ".m4a":
        return "audio/mp4"
    return "audio/mpeg"
