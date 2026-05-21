from typing import Literal

from pydantic import BaseModel, Field

TrackStatus = Literal["queued", "processing", "ready", "error"]
FeedbackLabel = Literal["similar", "not_similar"]


class FeedbackRequest(BaseModel):
    query_track_id: str
    candidate_track_id: str
    label: str


class SimilarTrack(BaseModel):
    id: str
    score: float


class SimilarSegment(BaseModel):
    id: str
    score: float
    segment_index: int
    start_seconds: float


class Track(BaseModel):
    id: str
    filename: str
    title: str
    artist: str | None = None
    status: TrackStatus
    error: str | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    cluster: int | None = None
    segment_count: int = 0
    similar: list[SimilarTrack] = Field(default_factory=list)
