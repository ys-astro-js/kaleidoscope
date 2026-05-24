from pathlib import Path

from fastapi.testclient import TestClient

from app import database
from app import main as app_main
from app.config import Settings
from app.context import AppContext
from app.models import SimilarityMix


class FakeService:
    def __init__(self, *, delete_result: bool = True) -> None:
        self.calls = []
        self.deleted_track_ids = []
        self.delete_all_calls = 0
        self.delete_result = delete_result
        self.mix = SimilarityMix(whole=0.5, vocals=0.25, instrumental=0.25, style=0.65, cover=0.35)
        self.mix_requests = []

    def recompute_layout(self) -> None:
        pass

    def backfill_identity_features(self) -> None:
        pass

    def backfill_cover_identity_features(self) -> None:
        pass

    def record_feedback(self, *, query_track_id: str, candidate_track_id: str, label: str) -> None:
        self.calls.append(
            {
                "query_track_id": query_track_id,
                "candidate_track_id": candidate_track_id,
                "label": label,
            }
        )

    def delete_track(self, track_id: str) -> bool:
        self.deleted_track_ids.append(track_id)
        return self.delete_result

    def delete_all_tracks(self) -> None:
        self.delete_all_calls += 1

    def similarity_mix(self) -> SimilarityMix:
        return self.mix

    def set_similarity_mix(
        self,
        *,
        vocals: float,
        instrumental: float,
        style: float | None = None,
        cover: float | None = None,
    ) -> SimilarityMix:
        self.mix_requests.append(
            {"vocals": vocals, "instrumental": instrumental, "style": style, "cover": cover}
        )
        total = vocals + instrumental
        if total <= 0.0:
            self.mix = SimilarityMix(
                whole=0.5,
                vocals=0.25,
                instrumental=0.25,
                style=0.65,
                cover=0.35,
            )
        else:
            style_value = self.mix.style if style is None else style
            cover_value = self.mix.cover if cover is None else cover
            style_total = style_value + cover_value
            self.mix = SimilarityMix(
                whole=0.5,
                vocals=0.5 * vocals / total,
                instrumental=0.5 * instrumental / total,
                style=style_value / style_total,
                cover=cover_value / style_total,
            )
        return self.mix

    def similar_segments(self, track_id: str, segment_index: int, *, limit: int):
        return [
            {
                "id": "candidate",
                "score": 0.92,
                "segment_index": 3,
                "start_seconds": 45.0,
            }
        ][:limit]


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


def make_client(
    conn,
    tmp_path: Path,
    *,
    service: FakeService | None = None,
    vectors: FakeVectors | None = None,
) -> TestClient:
    settings = Settings(data_dir=tmp_path)
    context = AppContext(
        settings=settings,
        conn=conn,
        vectors=vectors or FakeVectors(),
        service=service or FakeService(),
    )
    return TestClient(app_main.create_app(context))


def insert_ready_track(conn, tmp_path: Path, track_id: str, audio_bytes: bytes = b"audio") -> None:
    audio_path = tmp_path / f"{track_id}.mp3"
    art_path = tmp_path / f"{track_id}.png"
    audio_path.write_bytes(audio_bytes)
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


def add_stems(conn, tmp_path: Path, track_id: str) -> tuple[Path, Path]:
    vocals_path = tmp_path / f"{track_id}.vocals.wav"
    instrumental_path = tmp_path / f"{track_id}.instrumental.wav"
    vocals_path.write_bytes(b"vocals")
    instrumental_path.write_bytes(b"instrumental")
    database.update_track(
        conn,
        track_id,
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
    )
    return vocals_path, instrumental_path


def test_submit_feedback_records_valid_event(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query")
    insert_ready_track(conn, tmp_path, "candidate")
    service = FakeService()
    client = make_client(conn, tmp_path, service=service)

    response = client.post(
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


def test_similar_segments_returns_segment_matches(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query")
    client = make_client(conn, tmp_path, service=FakeService())

    response = client.get("/api/tracks/query/segments/2/similar")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "candidate",
            "score": 0.92,
            "segment_index": 3,
            "start_seconds": 45.0,
        }
    ]


def test_submit_feedback_rejects_bad_label(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    client = make_client(conn, tmp_path)

    response = client.post(
        "/api/feedback",
        json={
            "query_track_id": "query",
            "candidate_track_id": "candidate",
            "label": "maybe",
        },
    )

    assert response.status_code == 400


def test_submit_feedback_rejects_missing_track(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    client = make_client(conn, tmp_path)

    response = client.post(
        "/api/feedback",
        json={
            "query_track_id": "query",
            "candidate_track_id": "candidate",
            "label": "similar",
        },
    )

    assert response.status_code == 404


def test_delete_track_calls_service(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = FakeService()
    client = make_client(conn, tmp_path, service=service)

    response = client.delete("/api/tracks/query")

    assert response.status_code == 204
    assert service.deleted_track_ids == ["query"]


def test_delete_track_returns_404_when_service_cannot_delete(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = FakeService(delete_result=False)
    client = make_client(conn, tmp_path, service=service)

    response = client.delete("/api/tracks/missing")

    assert response.status_code == 404
    assert service.deleted_track_ids == ["missing"]


def test_delete_all_tracks_requires_confirm(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = FakeService()
    client = make_client(conn, tmp_path, service=service)

    response = client.delete("/api/tracks")

    assert response.status_code == 400
    assert service.delete_all_calls == 0


def test_delete_all_tracks_calls_service_when_confirmed(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = FakeService()
    client = make_client(conn, tmp_path, service=service)

    response = client.delete("/api/tracks?confirm=true")

    assert response.status_code == 204
    assert service.delete_all_calls == 1


def test_get_similarity_mix_returns_current_mix(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = FakeService()
    client = make_client(conn, tmp_path, service=service)

    response = client.get("/api/similarity/mix")

    assert response.status_code == 200
    assert response.json() == {
        "whole": 0.5,
        "vocals": 0.25,
        "instrumental": 0.25,
        "style": 0.65,
        "cover": 0.35,
    }


def test_update_similarity_mix_normalizes_stem_split(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = FakeService()
    client = make_client(conn, tmp_path, service=service)

    response = client.put(
        "/api/similarity/mix",
        json={"vocals": 3.0, "instrumental": 1.0, "style": 4.0, "cover": 1.0},
    )

    assert response.status_code == 200
    assert response.json() == {
        "whole": 0.5,
        "vocals": 0.375,
        "instrumental": 0.125,
        "style": 0.8,
        "cover": 0.2,
    }
    assert service.mix_requests == [
        {"vocals": 3.0, "instrumental": 1.0, "style": 4.0, "cover": 1.0}
    ]


def test_audio_range_response_returns_partial_content(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query", audio_bytes=b"audio")
    client = make_client(conn, tmp_path)

    response = client.get("/api/tracks/query/audio", headers={"Range": "bytes=1-3"})

    assert response.status_code == 206
    assert response.content == b"udi"
    assert response.headers["content-range"] == "bytes 1-3/5"


def test_audio_endpoint_serves_requested_stem(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query", audio_bytes=b"audio")
    add_stems(conn, tmp_path, "query")
    client = make_client(conn, tmp_path)

    response = client.get("/api/tracks/query/audio?stem=vocals")

    assert response.status_code == 200
    assert response.content == b"vocals"


def test_audio_endpoint_returns_404_for_missing_stem(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query", audio_bytes=b"audio")
    client = make_client(conn, tmp_path)

    response = client.get("/api/tracks/query/audio?stem=vocals")

    assert response.status_code == 404
