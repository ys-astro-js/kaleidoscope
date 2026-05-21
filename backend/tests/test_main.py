from pathlib import Path

from fastapi.testclient import TestClient

from app import database
from app import main as app_main


class FakeService:
    def __init__(self) -> None:
        self.calls = []

    def recompute_layout(self) -> None:
        pass

    def record_feedback(self, *, query_track_id: str, candidate_track_id: str, label: str) -> None:
        self.calls.append(
            {
                "query_track_id": query_track_id,
                "candidate_track_id": candidate_track_id,
                "label": label,
            }
        )


class FakeVectors:
    def similar_segments(self, track_id: str, segment_index: int, *, limit: int):
        return [
            {
                "id": "candidate",
                "score": 0.92,
                "segment_index": 3,
                "start_seconds": 45.0,
            }
        ]


def insert_ready_track(conn, tmp_path: Path, track_id: str) -> None:
    database.insert_track(
        conn,
        track_id=track_id,
        filename=f"{track_id}.mp3",
        title=track_id,
        audio_path=tmp_path / f"{track_id}.mp3",
        art_path=tmp_path / f"{track_id}.png",
    )
    database.update_track(conn, track_id, status="ready")


def test_submit_feedback_records_valid_event(monkeypatch, tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query")
    insert_ready_track(conn, tmp_path, "candidate")
    service = FakeService()
    monkeypatch.setattr(app_main, "conn", conn)
    monkeypatch.setattr(app_main, "service", service)

    response = TestClient(app_main.app).post(
        "/api/feedback",
        json={
            "query_track_id": "query",
            "candidate_track_id": "candidate",
            "label": "similar",
        },
    )

    assert response.status_code == 204
    assert service.calls == [
        {
            "query_track_id": "query",
            "candidate_track_id": "candidate",
            "label": "similar",
        }
    ]


def test_similar_segments_returns_segment_matches(monkeypatch, tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query")
    monkeypatch.setattr(app_main, "conn", conn)
    monkeypatch.setattr(app_main, "vectors", FakeVectors())

    response = TestClient(app_main.app).get("/api/tracks/query/segments/2/similar")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "candidate",
            "score": 0.92,
            "segment_index": 3,
            "start_seconds": 45.0,
        }
    ]


def test_submit_feedback_rejects_bad_label(monkeypatch, tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    monkeypatch.setattr(app_main, "conn", conn)

    response = TestClient(app_main.app).post(
        "/api/feedback",
        json={
            "query_track_id": "query",
            "candidate_track_id": "candidate",
            "label": "maybe",
        },
    )

    assert response.status_code == 400


def test_submit_feedback_rejects_missing_track(monkeypatch, tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    monkeypatch.setattr(app_main, "conn", conn)

    response = TestClient(app_main.app).post(
        "/api/feedback",
        json={
            "query_track_id": "query",
            "candidate_track_id": "candidate",
            "label": "similar",
        },
    )

    assert response.status_code == 404
