from typing import Literal

from pydantic import BaseModel

TrackStatus = Literal["queued", "processing", "ready", "error"]


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
    similar: list[str] = []

