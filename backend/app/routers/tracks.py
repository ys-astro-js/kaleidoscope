from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app import database
from app.audio import save_upload, title_from_filename, write_art_or_placeholder
from app.dependencies import AppContextDep
from app.models import SimilarSegment, Track

router = APIRouter(prefix="/api/tracks", tags=["tracks"])


@router.post("", response_model=Track)
async def upload_track(
    upload: UploadFile,
    background_tasks: BackgroundTasks,
    context: AppContextDep,
) -> Track:
    track_id, audio_path = await save_upload(upload, context.settings.audio_dir)
    art_path = write_art_or_placeholder(track_id, None, context.settings.art_dir)
    database.insert_track(
        context.conn,
        track_id=track_id,
        filename=upload.filename or audio_path.name,
        title=title_from_filename(upload.filename),
        audio_path=audio_path,
        art_path=art_path,
    )
    background_tasks.add_task(context.service.process_track, track_id)
    row = database.get_track(context.conn, track_id)
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


@router.get("", response_model=list[Track])
def list_tracks(context: AppContextDep) -> list[Track]:
    return context.service.list_tracks()


@router.get("/{track_id}/segments/{segment_index}/similar", response_model=list[SimilarSegment])
def similar_segments(
    track_id: str,
    segment_index: int,
    context: AppContextDep,
    limit: Annotated[int, Query()] = 5,
) -> list[SimilarSegment]:
    row = database.get_track(context.conn, track_id)
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
        for item in context.vectors.similar_segments(track_id, segment_index, limit=limit)
    ]


@router.delete("", status_code=204)
def delete_all_tracks(
    context: AppContextDep,
    confirm: Annotated[bool, Query()] = False,
) -> Response:
    if not confirm:
        raise HTTPException(status_code=400, detail="Track deletion requires confirm=true")
    context.service.delete_all_tracks()
    return Response(status_code=204)


@router.delete("/{track_id}", status_code=204)
def delete_track(track_id: str, context: AppContextDep) -> Response:
    if not context.service.delete_track(track_id):
        raise HTTPException(status_code=404, detail="Track not found")
    return Response(status_code=204)
