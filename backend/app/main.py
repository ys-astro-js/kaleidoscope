import asyncio
from pathlib import Path
from typing import cast

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from app import database
from app.audio import cached_art_thumbnail, save_upload, title_from_filename, write_art_or_placeholder
from app.config import ensure_data_dirs, get_settings
from app.models import FeedbackLabel, FeedbackRequest, SimilarSegment, Track
from app.service import TrackService
from app.vector_store import VectorStore

settings = get_settings()
ensure_data_dirs(settings)
conn = database.connect(settings.sqlite_path)
database.init_db(conn)
vectors = VectorStore(settings.lancedb_dir)
service = TrackService(settings, conn, vectors)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def resume_pending_tracks() -> None:
    service.recompute_layout()
    for row in database.list_tracks(conn):
        if row["status"] in {"queued", "processing"}:
            asyncio.create_task(asyncio.to_thread(service.process_track, row["id"]))


@app.post("/api/tracks", response_model=Track)
async def upload_track(upload: UploadFile, background_tasks: BackgroundTasks) -> Track:
    track_id, audio_path = await save_upload(upload, settings.audio_dir)
    art_path = write_art_or_placeholder(track_id, None, settings.art_dir)
    database.insert_track(
        conn,
        track_id=track_id,
        filename=upload.filename or audio_path.name,
        title=title_from_filename(upload.filename),
        audio_path=audio_path,
        art_path=art_path,
    )
    background_tasks.add_task(service.process_track, track_id)
    row = database.get_track(conn, track_id)
    if row is None:
        raise HTTPException(status_code=500, detail="Track was not saved")
    return Track(
        id=row["id"],
        filename=row["filename"],
        title=row["title"],
        artist=row["artist"],
        status=row["status"],
        error=row["error"],
        x=row["x"],
        y=row["y"],
        z=row["z"],
        cluster=row["cluster"],
        segment_count=0,
    )


@app.get("/api/tracks", response_model=list[Track])
def list_tracks() -> list[Track]:
    return service.list_tracks()


@app.get("/api/tracks/{track_id}/segments/{segment_index}/similar", response_model=list[SimilarSegment])
def similar_segments(track_id: str, segment_index: int, limit: int = 5) -> list[SimilarSegment]:
    row = database.get_track(conn, track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Track not found")
    if row["status"] != "ready":
        raise HTTPException(status_code=400, detail="Track is not ready")
    if segment_index < 0:
        raise HTTPException(status_code=400, detail="Segment index must be non-negative")

    return [
        SimilarSegment(
            id=str(item["id"]),
            score=float(item["score"]),
            segment_index=int(item["segment_index"]),
            start_seconds=float(item["start_seconds"]),
        )
        for item in vectors.similar_segments(track_id, segment_index, limit=limit)
    ]


@app.post("/api/feedback", status_code=204)
def submit_feedback(feedback: FeedbackRequest) -> Response:
    if feedback.query_track_id == feedback.candidate_track_id:
        raise HTTPException(status_code=400, detail="Feedback requires two different tracks")
    if feedback.label not in {"similar", "not_similar"}:
        raise HTTPException(status_code=400, detail="Feedback label must be similar or not_similar")

    query = database.get_track(conn, feedback.query_track_id)
    candidate = database.get_track(conn, feedback.candidate_track_id)
    if query is None or candidate is None:
        raise HTTPException(status_code=404, detail="Track not found")
    if query["status"] != "ready" or candidate["status"] != "ready":
        raise HTTPException(status_code=400, detail="Feedback requires ready tracks")

    service.record_feedback(
        query_track_id=feedback.query_track_id,
        candidate_track_id=feedback.candidate_track_id,
        label=cast(FeedbackLabel, feedback.label),
    )
    return Response(status_code=204)


@app.get("/api/tracks/{track_id}/art")
def get_art(track_id: str) -> FileResponse:
    row = database.get_track(conn, track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Track not found")
    art_path = cached_art_thumbnail(Path(row["art_path"]))
    return FileResponse(
        art_path,
        media_type="image/jpeg" if art_path.suffix.lower() == ".jpg" else None,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/tracks/{track_id}/audio")
def get_audio(track_id: str, range_header: str | None = Header(None, alias="Range")) -> Response:
    row = database.get_track(conn, track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Track not found")
    path = Path(row["audio_path"])
    if range_header is None:
        return FileResponse(path)
    return _range_response(path, range_header)


@app.delete("/api/tracks", status_code=204)
def delete_all_tracks(confirm: bool = False) -> Response:
    if not confirm:
        raise HTTPException(status_code=400, detail="Track deletion requires confirm=true")
    rows = list(database.list_tracks(conn))
    for row in rows:
        _delete_track_assets(row)
    service.retrain_feedback_weights()
    service.recompute_layout()
    return Response(status_code=204)


@app.delete("/api/tracks/{track_id}", status_code=204)
def delete_track(track_id: str) -> Response:
    row = database.get_track(conn, track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Track not found")
    _delete_track_assets(row)
    service.retrain_feedback_weights()
    service.recompute_layout()
    return Response(status_code=204)


def _delete_track_assets(row) -> None:
    track_id = row["id"]
    database.delete_track(conn, track_id)
    vectors.delete(track_id)
    for path_key in ("audio_path", "art_path"):
        path = Path(row[path_key])
        if path.exists():
            path.unlink()
    model_audio_path = settings.audio_dir / f"{track_id}.model.wav"
    if model_audio_path.exists():
        model_audio_path.unlink()


def _range_response(path: Path, range_header: str) -> StreamingResponse:
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
