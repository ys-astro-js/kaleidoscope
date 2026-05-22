from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from app import database
from app.audio import cached_art_thumbnail
from app.dependencies import AppContextDep

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
    range_header: Annotated[str | None, Header(alias="Range")] = None,
) -> Response:
    row = database.get_track(context.conn, track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Track not found")
    path = Path(row["audio_path"])
    if range_header is None:
        return FileResponse(path)
    return range_response(path, range_header)


def range_response(path: Path, range_header: str) -> StreamingResponse:
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
        media_type="audio/mpeg",
    )
