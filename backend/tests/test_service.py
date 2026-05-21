import threading
import time
from pathlib import Path

from app import database
from app.config import Settings
from app.service import TrackService


class SlowVectors:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.limits = []

    def similar_by_track(self, *, limit, weights):
        self.calls += 1
        self.limits.append(limit)
        self.started.set()
        self.release.wait(timeout=2)
        return {"query": [{"id": "candidate", "score": 0.9}], "candidate": []}


def insert_ready_track(conn, tmp_path: Path, track_id: str) -> None:
    audio_path = tmp_path / f"{track_id}.mp3"
    art_path = tmp_path / f"{track_id}.png"
    audio_path.write_bytes(b"audio")
    art_path.write_bytes(b"art")
    database.insert_track(
        conn,
        track_id=track_id,
        filename=audio_path.name,
        title=track_id,
        audio_path=audio_path,
        art_path=art_path,
    )
    database.update_track(conn, track_id, status="ready")


def test_list_tracks_returns_before_similarity_cache_is_ready(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query")
    insert_ready_track(conn, tmp_path, "candidate")
    vectors = SlowVectors()
    service = TrackService(Settings(data_dir=tmp_path), conn, vectors)

    tracks = service.list_tracks()

    assert {track.id for track in tracks} == {"query", "candidate"}
    assert next(track for track in tracks if track.id == "query").similar == []
    assert vectors.started.wait(timeout=1)

    vectors.release.set()
    for _ in range(20):
        tracks = service.list_tracks()
        query = next(track for track in tracks if track.id == "query")
        if query.similar:
            break
        time.sleep(0.05)

    assert query.similar[0].id == "candidate"
    assert query.similar[0].score == 0.9
    assert vectors.calls == 1
    assert vectors.limits == [5]
