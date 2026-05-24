import threading
import time
from pathlib import Path

from app import database
from app.config import Settings
from app.embedding import MIN_SEGMENT_ERROR, TrackEmbeddings
from app.separation import StemSeparationResult
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


class CaptureVectors:
    def __init__(self) -> None:
        self.upserts = []

    def upsert(self, *args) -> None:
        self.upserts.append(args)

    def delete(self, track_id: str) -> None:
        pass


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


def test_process_track_stores_stems_and_upserts_whole_and_stem_embeddings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    database.insert_track(
        conn,
        track_id="track-id",
        filename="track-id.mp3",
        title="track-id",
        audio_path=tmp_path / "track-id.mp3",
        art_path=tmp_path / "track-id.png",
    )
    (tmp_path / "track-id.mp3").write_bytes(b"audio")

    vocals_path = tmp_path / "stems" / "track-id.vocals.wav"
    instrumental_path = tmp_path / "stems" / "track-id.instrumental.wav"
    vocals_path.parent.mkdir()
    vocals_path.write_bytes(b"vocals")
    instrumental_path.write_bytes(b"instrumental")

    def fake_separate(input_path, *, track_id, output_dir, model_dir):
        return StemSeparationResult(vocals_path=vocals_path, instrumental_path=instrumental_path)

    class FakeEmbedder:
        def embed_file(self, path: str) -> TrackEmbeddings:
            if "vocals" in path:
                return TrackEmbeddings([0.0, 1.0], [[0.0, 1.0]])
            if "instrumental" in path:
                return TrackEmbeddings([1.0, 1.0], [[1.0, 1.0]])
            return TrackEmbeddings([1.0, 0.0], [[1.0, 0.0]])

    def fake_normalize(input_path, target_path, sample_rate):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(b"wav")
        return Path(target_path)

    monkeypatch.setattr("app.service.extract_metadata", lambda path: (None, None, None))
    monkeypatch.setattr("app.service.write_art_or_placeholder", lambda *args: tmp_path / "art.png")
    monkeypatch.setattr("app.service.separate_vocals_and_instrumental", fake_separate)
    monkeypatch.setattr("app.service.normalize_audio_for_model", fake_normalize)
    monkeypatch.setattr("app.service.get_embedder", lambda model_id, sample_rate: FakeEmbedder())

    vectors = CaptureVectors()
    service = TrackService(Settings(data_dir=tmp_path), conn, vectors)
    service.recompute_layout = lambda: None

    service.process_track("track-id")

    row = database.get_track(conn, "track-id")
    assert row is not None
    assert row["status"] == "ready"
    assert row["vocals_path"] == str(vocals_path)
    assert row["instrumental_path"] == str(instrumental_path)
    assert vectors.upserts == [
        (
            "track-id",
            [1.0, 0.0],
            [[1.0, 0.0]],
            [0.0, 1.0],
            [[0.0, 1.0]],
            [1.0, 1.0],
            [[1.0, 1.0]],
        )
    ]


def test_process_track_skips_too_short_stem_embeddings(monkeypatch, tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    database.insert_track(
        conn,
        track_id="track-id",
        filename="track-id.mp3",
        title="track-id",
        audio_path=tmp_path / "track-id.mp3",
        art_path=tmp_path / "track-id.png",
    )
    (tmp_path / "track-id.mp3").write_bytes(b"audio")

    vocals_path = tmp_path / "track-id.vocals.wav"
    instrumental_path = tmp_path / "track-id.instrumental.wav"
    vocals_path.write_bytes(b"vocals")
    instrumental_path.write_bytes(b"instrumental")
    monkeypatch.setattr(
        "app.service.separate_vocals_and_instrumental",
        lambda input_path, *, track_id, output_dir, model_dir: StemSeparationResult(
            vocals_path=vocals_path,
            instrumental_path=instrumental_path,
        ),
    )

    class FakeEmbedder:
        def embed_file(self, path: str) -> TrackEmbeddings:
            if "vocals" in path:
                raise ValueError(MIN_SEGMENT_ERROR)
            if "instrumental" in path:
                return TrackEmbeddings([0.0, 1.0], [[0.0, 1.0]])
            return TrackEmbeddings([1.0, 0.0], [[1.0, 0.0]])

    monkeypatch.setattr("app.service.extract_metadata", lambda path: (None, None, None))
    monkeypatch.setattr("app.service.write_art_or_placeholder", lambda *args: tmp_path / "art.png")
    monkeypatch.setattr("app.service.normalize_audio_for_model", lambda input_path, target_path, sample_rate: Path(target_path))
    monkeypatch.setattr("app.service.get_embedder", lambda model_id, sample_rate: FakeEmbedder())

    vectors = CaptureVectors()
    service = TrackService(Settings(data_dir=tmp_path), conn, vectors)
    service.recompute_layout = lambda: None

    service.process_track("track-id")

    row = database.get_track(conn, "track-id")
    assert row is not None
    assert row["status"] == "ready"
    assert vectors.upserts[0][3] is None
    assert vectors.upserts[0][4] is None
    assert vectors.upserts[0][5] == [0.0, 1.0]
