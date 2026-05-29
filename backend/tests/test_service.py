import threading
import time
from pathlib import Path

from app import database
from app.config import Settings
from app.cover_alignment import CoverAlignmentFeature
from app.cover_identity import CoverIdentityFeature
from app.embedding import MIN_SEGMENT_ERROR, TrackEmbeddings
from app.separation import StemSeparationResult
from app.service import TrackService, _similar_by_id_from_matrix


class SlowVectors:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.limits = []

    def segment_counts(self):
        return {}

    def all_embeddings(self):
        return {"query": [], "candidate": []}

    def similarity_matrix(self, *, embeddings, weights):
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=2)
        return {
            "query": {"query": 1.0, "candidate": 0.9},
            "candidate": {"candidate": 1.0, "query": 0.9},
        }


class CountingVectors:
    def __init__(self) -> None:
        self.calls = 0

    def segment_counts(self):
        return {}

    def all_embeddings(self):
        return {"query": [], "candidate": []}

    def similarity_matrix(self, *, embeddings, weights):
        self.calls += 1
        return {
            "query": {"query": 1.0, "candidate": 0.9},
            "candidate": {"candidate": 1.0, "query": 0.9},
        }


class CaptureVectors:
    def __init__(self) -> None:
        self.upserts = []
        self.similarity_matrix_calls = []

    def upsert(self, *args, **kwargs) -> None:
        self.upserts.append({"args": args, "kwargs": kwargs})

    def delete(self, track_id: str) -> None:
        pass

    def all_embeddings(self):
        return {}

    def similarity_matrix(self, *, embeddings, weights):
        self.similarity_matrix_calls.append({"embeddings": embeddings, "weights": weights})
        return {}

    def similar_segments(self, track_id: str, segment_index: int, *, limit: int):
        return []


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


def _cover_feature(vector: list[float]) -> CoverIdentityFeature:
    return CoverIdentityFeature(
        global_embedding=vector,
        chunk_embeddings=[vector],
        chunk_start_seconds=[0.0],
    )


def _alignment_feature(vector: list[float]) -> CoverAlignmentFeature:
    return CoverAlignmentFeature(
        model_key="clews",
        global_embedding=vector,
        segment_embeddings=[vector],
        segment_start_seconds=[0.0],
    )


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
    assert query.similar[0].score > 0.0
    assert vectors.calls == 1


def test_similarity_cache_refreshes_when_mix_changes_without_weight_change(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query")
    insert_ready_track(conn, tmp_path, "candidate")
    vectors = CountingVectors()
    service = TrackService(Settings(data_dir=tmp_path), conn, vectors)

    service._refresh_similarity_cache(service.reranker_coefficients())
    assert vectors.calls == 1

    database.set_similarity_mix(
        conn,
        vocals_weight=0.0,
        instrumental_weight=0.5,
        style_weight=0.8,
        cover_weight=0.2,
    )
    service.cached_similar_by_id()

    for _ in range(20):
        if vectors.calls == 2:
            break
        time.sleep(0.05)

    assert vectors.calls == 2


def test_similarity_mix_defaults_to_whole_instrumental_and_cover_split(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = TrackService(Settings(data_dir=tmp_path), conn, CaptureVectors())

    mix = service.similarity_mix()

    assert mix.whole == 0.5
    assert mix.vocals == 0.0
    assert mix.instrumental == 0.5
    assert mix.style == 0.85
    assert mix.cover == 0.15


def test_feedback_weights_mute_vocal_bucket_without_changing_whole_bucket(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    database.set_feedback_weights(
        conn,
        global_weight=0.1,
        segment_weight=0.4,
        vocals_global_weight=0.2,
        vocals_segment_weight=0.3,
        instrumental_global_weight=0.25,
        instrumental_segment_weight=0.25,
        event_count=2,
    )
    database.set_similarity_mix(
        conn,
        vocals_weight=3.0,
        instrumental_weight=0.5,
        style_weight=0.85,
        cover_weight=0.15,
    )
    service = TrackService(Settings(data_dir=tmp_path), conn, CaptureVectors())

    weights = service.feedback_weights()

    assert weights.global_semantic == 0.1
    assert weights.segment_semantic == 0.4
    assert weights.vocals_global_semantic == 0.0
    assert weights.vocals_segment_semantic == 0.0
    assert weights.instrumental_global_semantic == 0.25
    assert weights.instrumental_segment_semantic == 0.25


def test_set_similarity_mix_persists_whole_instrumental_bucket_and_recomputes_layout(
    tmp_path: Path,
) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = TrackService(Settings(data_dir=tmp_path), conn, CaptureVectors())
    recompute_calls = 0

    def fake_recompute_layout() -> None:
        nonlocal recompute_calls
        recompute_calls += 1

    service.recompute_layout = fake_recompute_layout

    mix = service.set_similarity_mix(whole=3.0, instrumental=1.0)
    row = database.get_similarity_mix(conn)

    assert mix.whole == 0.75
    assert mix.vocals == 0.0
    assert mix.instrumental == 0.25
    assert mix.style == 0.85
    assert mix.cover == 0.15
    assert row is not None
    assert row["vocals_weight"] == 0.0
    assert row["instrumental_weight"] == 0.25
    assert row["style_weight"] == 0.85
    assert row["cover_weight"] == 0.15
    assert recompute_calls == 1
    assert len(service.vectors.similarity_matrix_calls) == 1


def test_set_similarity_mix_normalizes_style_and_cover_bucket(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = TrackService(Settings(data_dir=tmp_path), conn, CaptureVectors())
    service.recompute_layout = lambda: None

    mix = service.set_similarity_mix(style=3.0, cover=1.0)
    row = database.get_similarity_mix(conn)

    assert mix.whole == 0.5
    assert mix.vocals == 0.0
    assert mix.instrumental == 0.5
    assert mix.style == 0.75
    assert mix.cover == 0.25
    assert row is not None
    assert row["style_weight"] == 0.75
    assert row["cover_weight"] == 0.25


def test_set_similarity_mix_normalizes_whole_and_instrumental_bucket(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    service = TrackService(Settings(data_dir=tmp_path), conn, CaptureVectors())
    service.recompute_layout = lambda: None

    mix = service.set_similarity_mix(whole=1.0, instrumental=3.0)
    row = database.get_similarity_mix(conn)

    assert mix.whole == 0.25
    assert mix.vocals == 0.0
    assert mix.instrumental == 0.75
    assert row is not None
    assert row["vocals_weight"] == 0.0
    assert row["instrumental_weight"] == 0.75


def test_backfill_cover_identity_features_stores_missing_ready_tracks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "track-id")
    cover_feature = _cover_feature([1.0, 0.0, 0.0])
    monkeypatch.setattr(
        "app.service.extract_cover_identity_feature",
        lambda path, model_dir: cover_feature,
    )
    service = TrackService(Settings(data_dir=tmp_path), conn, CaptureVectors())

    service.backfill_cover_identity_features()

    assert database.get_track_cover_identity_feature(conn, "track-id") == cover_feature


def test_backfill_cover_alignment_features_stores_missing_ready_tracks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "track-id")
    model_audio_path = tmp_path / "audio" / "track-id.model.wav"
    model_audio_path.parent.mkdir()
    model_audio_path.write_bytes(b"model wav")
    alignment_feature = _alignment_feature([1.0, 0.0, 0.0])

    extracted_paths = []

    def fake_extract(path, model_dir):
        extracted_paths.append(path)
        return alignment_feature

    monkeypatch.setattr(
        "app.service.extract_cover_alignment_feature",
        fake_extract,
    )
    service = TrackService(Settings(data_dir=tmp_path), conn, CaptureVectors())
    service.recompute_layout = lambda: None

    service.backfill_cover_alignment_features()

    assert extracted_paths == [model_audio_path]
    assert database.get_track_cover_alignment_feature(conn, "track-id") == alignment_feature


def test_cover_identity_score_can_rerank_style_similarity(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    for track_id in ("query", "remix", "same_artist"):
        insert_ready_track(conn, tmp_path, track_id)
    database.set_track_cover_identity_feature(
        conn,
        "query",
        _cover_feature([1.0, 0.0, 0.0]),
    )
    database.set_track_cover_identity_feature(
        conn,
        "remix",
        _cover_feature([1.0, 0.0, 0.0]),
    )
    database.set_track_cover_identity_feature(
        conn,
        "same_artist",
        _cover_feature([0.0, 1.0, 0.0]),
    )

    class StyleVectors(CaptureVectors):
        def all_embeddings(self):
            return {track_id: [] for track_id in ("query", "remix", "same_artist")}

        def similarity_matrix(self, *, embeddings, weights):
            return {
                "query": {"query": 1.0, "remix": 0.6, "same_artist": 0.65},
                "remix": {"remix": 1.0, "query": 0.6, "same_artist": 0.2},
                "same_artist": {"same_artist": 1.0, "query": 0.65, "remix": 0.2},
            }

    service = TrackService(Settings(data_dir=tmp_path), conn, StyleVectors())

    matrix = service._combined_similarity_matrix(service.reranker_coefficients())

    assert matrix["query"]["remix"] > matrix["query"]["same_artist"]


def test_cover_alignment_score_takes_priority_over_discogs_identity(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    for track_id in ("query", "remix", "same_artist"):
        insert_ready_track(conn, tmp_path, track_id)
    database.set_track_cover_identity_feature(conn, "query", _cover_feature([1.0, 0.0, 0.0]))
    database.set_track_cover_identity_feature(conn, "remix", _cover_feature([0.0, 1.0, 0.0]))
    database.set_track_cover_identity_feature(conn, "same_artist", _cover_feature([1.0, 0.0, 0.0]))
    database.set_track_cover_alignment_feature(conn, "query", _alignment_feature([1.0, 0.0, 0.0]))
    database.set_track_cover_alignment_feature(conn, "remix", _alignment_feature([1.0, 0.0, 0.0]))
    database.set_track_cover_alignment_feature(
        conn,
        "same_artist",
        _alignment_feature([0.0, 1.0, 0.0]),
    )

    class StyleVectors(CaptureVectors):
        def all_embeddings(self):
            return {track_id: [] for track_id in ("query", "remix", "same_artist")}

        def similarity_matrix(self, *, embeddings, weights):
            return {
                "query": {"query": 1.0, "remix": 0.6, "same_artist": 0.65},
                "remix": {"remix": 1.0, "query": 0.6, "same_artist": 0.2},
                "same_artist": {"same_artist": 1.0, "query": 0.65, "remix": 0.2},
            }

    service = TrackService(Settings(data_dir=tmp_path), conn, StyleVectors())

    matrix = service._combined_similarity_matrix(service.reranker_coefficients())

    assert matrix["query"]["remix"] > matrix["query"]["same_artist"]


def test_similar_by_id_uses_matrix_scores_directly() -> None:
    matrix = {
        "facade": {
            "facade": 1.0,
            "facade_remix": 0.97,
            "mesmerizer": 0.91,
            "obsolete_meat": 0.90,
        },
        "facade_remix": {
            "facade_remix": 1.0,
            "facade": 0.97,
            "mesmerizer": 0.20,
            "obsolete_meat": 0.20,
        },
        "mesmerizer": {
            "mesmerizer": 1.0,
            "facade": 0.91,
            "obsolete_meat": 0.89,
            "facade_remix": 0.20,
        },
        "obsolete_meat": {
            "obsolete_meat": 1.0,
            "facade": 0.90,
            "mesmerizer": 0.89,
            "facade_remix": 0.20,
        },
    }
    similar = _similar_by_id_from_matrix(
        matrix,
        limit=2,
    )

    assert similar["facade"][0]["id"] == "facade_remix"
    assert similar["facade_remix"][0]["id"] == "facade"
    assert similar["mesmerizer"][0]["id"] == "facade"
    assert similar["obsolete_meat"][0]["id"] == "facade"
    for matches in similar.values():
        scores = [float(match["score"]) for match in matches]
        assert scores == sorted(scores, reverse=True)


def test_similar_segments_can_lift_cover_match_with_track_rerank(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    for track_id in ("query", "remix", "same_artist"):
        insert_ready_track(conn, tmp_path, track_id)
    database.set_track_cover_identity_feature(
        conn,
        "query",
        _cover_feature([1.0, 0.0, 0.0]),
    )
    database.set_track_cover_identity_feature(
        conn,
        "remix",
        _cover_feature([1.0, 0.0, 0.0]),
    )
    database.set_track_cover_identity_feature(
        conn,
        "same_artist",
        _cover_feature([0.0, 1.0, 0.0]),
    )

    class SegmentVectors(CaptureVectors):
        def similar_segments(self, track_id: str, segment_index: int, *, limit: int):
            return [
                {"id": "same_artist", "score": 0.8, "segment_index": 0, "start_seconds": 0.0},
                {"id": "remix", "score": 0.75, "segment_index": 2, "start_seconds": 30.0},
            ]

    service = TrackService(Settings(data_dir=tmp_path), conn, SegmentVectors())

    matches = service.similar_segments("query", 1, limit=2)

    assert matches[0]["id"] == "remix"
    assert matches[0]["segment_index"] == 2


def test_similar_segments_caches_same_source_request(tmp_path: Path) -> None:
    conn = database.connect(tmp_path / "app.sqlite")
    database.init_db(conn)
    insert_ready_track(conn, tmp_path, "query")
    insert_ready_track(conn, tmp_path, "candidate")

    class SegmentVectors(CaptureVectors):
        def __init__(self) -> None:
            super().__init__()
            self.embedding_calls = 0
            self.segment_calls = 0

        def all_embeddings(self):
            self.embedding_calls += 1
            return {"query": [], "candidate": []}

        def similar_segments(self, track_id: str, segment_index: int, *, limit: int):
            self.segment_calls += 1
            return [
                {"id": "candidate", "score": 0.8, "segment_index": 0, "start_seconds": 0.0},
            ]

    vectors = SegmentVectors()
    service = TrackService(Settings(data_dir=tmp_path), conn, vectors)

    first = service.similar_segments("query", 1, limit=1)
    second = service.similar_segments("query", 1, limit=1)

    assert first == second
    assert vectors.embedding_calls == 1
    assert vectors.segment_calls == 1


def test_process_track_stores_instrumental_and_upserts_whole_and_instrumental_embeddings(
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

    instrumental_path = tmp_path / "stems" / "track-id.instrumental.wav"
    instrumental_path.parent.mkdir()
    instrumental_path.write_bytes(b"instrumental")

    def fake_separate(input_path, *, track_id, output_dir, model_dir):
        return StemSeparationResult(instrumental_path=instrumental_path)

    class FakeEmbedder:
        def embed_file(self, path: str) -> TrackEmbeddings:
            if "instrumental" in path:
                return TrackEmbeddings([1.0, 1.0], [[1.0, 1.0]])
            return TrackEmbeddings([1.0, 0.0], [[1.0, 0.0]])

    def fake_normalize(input_path, target_path, sample_rate):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(b"wav")
        return Path(target_path)

    monkeypatch.setattr("app.service.extract_metadata", lambda path: (None, None, None))
    monkeypatch.setattr("app.service.write_art_or_placeholder", lambda *args: tmp_path / "art.png")
    monkeypatch.setattr("app.service.separate_instrumental", fake_separate)
    monkeypatch.setattr("app.service.normalize_audio_for_model", fake_normalize)
    monkeypatch.setattr("app.service.get_embedder", lambda model_id, sample_rate: FakeEmbedder())
    cover_feature = _cover_feature([1.0, 0.0, 0.0])
    monkeypatch.setattr(
        "app.service.extract_cover_identity_feature",
        lambda path, model_dir: cover_feature,
    )
    alignment_feature = _alignment_feature([0.0, 1.0, 0.0])
    extracted_alignment_paths = []

    def fake_extract_alignment(path, model_dir):
        extracted_alignment_paths.append(path)
        return alignment_feature

    monkeypatch.setattr("app.service.extract_cover_alignment_feature", fake_extract_alignment)

    vectors = CaptureVectors()
    service = TrackService(Settings(data_dir=tmp_path), conn, vectors)
    service.recompute_layout = lambda: None

    service.process_track("track-id")

    row = database.get_track(conn, "track-id")
    assert row is not None
    assert row["status"] == "ready"
    assert row["vocals_path"] is None
    assert row["instrumental_path"] == str(instrumental_path)
    assert database.get_track_cover_identity_feature(conn, "track-id") == cover_feature
    assert database.get_track_cover_alignment_feature(conn, "track-id") == alignment_feature
    assert extracted_alignment_paths == [tmp_path / "audio" / "track-id.model.wav"]
    assert vectors.upserts == [
        {
            "args": ("track-id", [1.0, 0.0], [[1.0, 0.0]]),
            "kwargs": {
                "instrumental_global_semantic_vector": [1.0, 1.0],
                "instrumental_segment_semantic_vectors": [[1.0, 1.0]],
            },
        }
    ]


def test_process_track_skips_too_short_instrumental_embeddings(monkeypatch, tmp_path: Path) -> None:
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

    instrumental_path = tmp_path / "track-id.instrumental.wav"
    instrumental_path.write_bytes(b"instrumental")
    monkeypatch.setattr(
        "app.service.separate_instrumental",
        lambda input_path, *, track_id, output_dir, model_dir: StemSeparationResult(
            instrumental_path=instrumental_path
        ),
    )

    class FakeEmbedder:
        def embed_file(self, path: str) -> TrackEmbeddings:
            if "instrumental" in path:
                raise ValueError(MIN_SEGMENT_ERROR)
            return TrackEmbeddings([1.0, 0.0], [[1.0, 0.0]])

    monkeypatch.setattr("app.service.extract_metadata", lambda path: (None, None, None))
    monkeypatch.setattr("app.service.write_art_or_placeholder", lambda *args: tmp_path / "art.png")
    monkeypatch.setattr(
        "app.service.normalize_audio_for_model",
        lambda input_path, target_path, sample_rate: Path(target_path),
    )
    monkeypatch.setattr("app.service.get_embedder", lambda model_id, sample_rate: FakeEmbedder())
    monkeypatch.setattr(
        "app.service.extract_cover_identity_feature",
        lambda path, model_dir: None,
    )

    def fail_cover_alignment(path, model_dir):
        raise RuntimeError("CLEWS is not installed")

    monkeypatch.setattr("app.service.extract_cover_alignment_feature", fail_cover_alignment)

    vectors = CaptureVectors()
    service = TrackService(Settings(data_dir=tmp_path), conn, vectors)
    service.recompute_layout = lambda: None

    service.process_track("track-id")

    row = database.get_track(conn, "track-id")
    assert row is not None
    assert row["status"] == "ready"
    assert vectors.upserts[0]["kwargs"]["instrumental_global_semantic_vector"] is None
    assert vectors.upserts[0]["kwargs"]["instrumental_segment_semantic_vectors"] is None
