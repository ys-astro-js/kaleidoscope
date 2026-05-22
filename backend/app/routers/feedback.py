from typing import cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app import database
from app.dependencies import AppContextDep
from app.models import FeedbackLabel, FeedbackRequest

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("", status_code=204)
def submit_feedback(feedback: FeedbackRequest, context: AppContextDep) -> Response:
    if feedback.query_track_id == feedback.candidate_track_id:
        raise HTTPException(status_code=400, detail="Feedback requires two different tracks")
    if feedback.label not in {"similar", "not_similar"}:
        raise HTTPException(status_code=400, detail="Feedback label must be similar or not_similar")

    query = database.get_track(context.conn, feedback.query_track_id)
    candidate = database.get_track(context.conn, feedback.candidate_track_id)
    if query is None or candidate is None:
        raise HTTPException(status_code=404, detail="Track not found")
    if query["status"] != "ready" or candidate["status"] != "ready":
        raise HTTPException(status_code=400, detail="Feedback requires ready tracks")

    context.service.record_feedback(
        query_track_id=feedback.query_track_id,
        candidate_track_id=feedback.candidate_track_id,
        label=cast(FeedbackLabel, feedback.label),
    )
    return Response(status_code=204)
