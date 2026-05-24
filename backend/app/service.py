import sqlite3
import threading
from pathlib import Path

from app import database
from app.audio import extract_metadata, normalize_audio_for_model, write_art_or_placeholder
from app.config import Settings
from app.embedding import MIN_SEGMENT_ERROR, TrackEmbeddings, get_embedder
from app.feedback import DEFAULT_WEIGHTS, learn_feedback_weights
from app.layout import compute_similarity_layout
from app.models import AudioStem, FeedbackLabel, SimilarTrack, Track
from app.separation import separate_vocals_and_instrumental
from app.vector_store import SimilarityWeights, VectorStore

SimilarById = dict[str, list[dict[str, float | str]]]
SIMILAR_TRACK_LIMIT = 5


class TrackService:
    def __init__(self, settings: Settings, conn: sqlite3.Connection, vectors: VectorStore) -> None:
        self.settings = settings
        self.conn = conn
        self.vectors = vectors
        self._similar_lock = threading.Lock()
        self._similar_cache: SimilarById = {}
        self._similar_cache_weights: SimilarityWeights | None = None
        self._similar_refreshing = False
        self._processing_lock = threading.Lock()

    def list_tracks(self) -> list[Track]:
        similar_by_id = self.cached_similar_by_id()
        embedded_ids = set(similar_by_id)
        segment_counts = (
            self.vectors.segment_counts()
            if hasattr(self.vectors, "segment_counts")
            else {}
        )
        rows = database.rows_to_dicts(database.list_tracks(self.conn))
        row_ids = {row["id"] for row in rows}
        tracks: list[Track] = []
        for row in rows:
            similar = [
                item
                for item in similar_by_id.get(row["id"], [])
                if item["id"] in row_ids
            ]
            if row["status"] != "ready" or row["id"] not in embedded_ids:
                similar = []
            tracks.append(
                Track(
                    id=row["id"],
                    filename=row["filename"],
                    title=row["title"],
                    artist=row["artist"],
                    album=row["album"],
                    status=row["status"],
                    error=row["error"],
                    x=row["x"],
                    y=row["y"],
                    z=row["z"],
                    cluster=row["cluster"],
                    segment_count=segment_counts.get(row["id"], 0),
                    available_stems=_available_stems(row),
                    similar=[SimilarTrack(id=str(item["id"]), score=float(item["score"])) for item in similar],
                )
            )
        return tracks

    def cached_similar_by_id(self) -> SimilarById:
        weights = self.feedback_weights()
        with self._similar_lock:
            cached = self._similar_cache
            cache_is_current = self._similar_cache_weights == weights
            should_refresh = not cache_is_current and not self._similar_refreshing
            if should_refresh:
                self._similar_refreshing = True

        if should_refresh:
            thread = threading.Thread(
                target=self._refresh_similarity_cache,
                args=(weights,),
                daemon=True,
            )
            thread.start()

        return cached

    def invalidate_similarity_cache(self) -> None:
        with self._similar_lock:
            self._similar_cache_weights = None

    def _refresh_similarity_cache(self, weights: SimilarityWeights) -> None:
        try:
            similar_by_id = self.vectors.similar_by_track(limit=SIMILAR_TRACK_LIMIT, weights=weights)
            with self._similar_lock:
                self._similar_cache = similar_by_id
                self._similar_cache_weights = weights
        finally:
            with self._similar_lock:
                self._similar_refreshing = False

    def process_track(self, track_id: str) -> None:
        row = database.get_track(self.conn, track_id)
        if row is None:
            return

        try:
            database.update_track(self.conn, track_id, status="processing")
            artist, album, art_bytes = extract_metadata(Path(row["audio_path"]))
            art_path = write_art_or_placeholder(track_id, art_bytes, self.settings.art_dir)

            with self._processing_lock:
                stem_paths = separate_vocals_and_instrumental(
                    Path(row["audio_path"]),
                    track_id=track_id,
                    output_dir=self.settings.stem_dir,
                    model_dir=self.settings.separator_model_dir,
                )
                embedder = get_embedder(self.settings.model_id, self.settings.sample_rate)
                embeddings = self._embed_for_model(
                    embedder,
                    Path(row["audio_path"]),
                    self.settings.audio_dir / f"{track_id}.model.wav",
                )
                vocals_embeddings = self._embed_optional_stem(
                    embedder,
                    stem_paths.vocals_path,
                    self.settings.audio_dir / f"{track_id}.vocals.model.wav",
                )
                instrumental_embeddings = self._embed_optional_stem(
                    embedder,
                    stem_paths.instrumental_path,
                    self.settings.audio_dir / f"{track_id}.instrumental.model.wav",
                )

            self.vectors.upsert(
                track_id,
                embeddings.global_semantic,
                embeddings.segment_semantic,
                vocals_embeddings.global_semantic if vocals_embeddings else None,
                vocals_embeddings.segment_semantic if vocals_embeddings else None,
                instrumental_embeddings.global_semantic if instrumental_embeddings else None,
                instrumental_embeddings.segment_semantic if instrumental_embeddings else None,
            )
            self.invalidate_similarity_cache()
            database.update_track(
                self.conn,
                track_id,
                status="ready",
                artist=artist,
                album=album,
                art_path=art_path,
                vocals_path=stem_paths.vocals_path,
                instrumental_path=stem_paths.instrumental_path,
            )
            self.recompute_layout()
        except Exception as exc:
            database.update_track(self.conn, track_id, status="error", error=str(exc))

    def _embed_for_model(self, embedder, input_path: Path, model_audio_path: Path) -> TrackEmbeddings:
        normalized_path = normalize_audio_for_model(
            input_path,
            model_audio_path,
            self.settings.sample_rate,
        )
        return embedder.embed_file(str(normalized_path))

    def _embed_optional_stem(
        self,
        embedder,
        input_path: Path,
        model_audio_path: Path,
    ) -> TrackEmbeddings | None:
        try:
            return self._embed_for_model(embedder, input_path, model_audio_path)
        except ValueError as exc:
            if str(exc) == MIN_SEGMENT_ERROR:
                return None
            raise

    def recompute_layout(self) -> None:
        ready_ids = {row["id"] for row in database.ready_tracks(self.conn)}
        embeddings = {
            track_id: records
            for track_id, records in self.vectors.all_embeddings().items()
            if track_id in ready_ids
        }
        similarities = self.vectors.similarity_matrix(
            embeddings=embeddings,
            weights=self.feedback_weights(),
        )
        database.set_track_coords(self.conn, compute_similarity_layout(similarities))

    def record_feedback(
        self,
        *,
        query_track_id: str,
        candidate_track_id: str,
        label: FeedbackLabel,
    ) -> None:
        database.insert_feedback_event(
            self.conn,
            query_track_id=query_track_id,
            candidate_track_id=candidate_track_id,
            label=label,
        )
        self.retrain_feedback_weights()
        self.recompute_layout()

    def retrain_feedback_weights(self) -> SimilarityWeights:
        result = learn_feedback_weights(self.vectors, database.list_feedback_events(self.conn))
        database.set_feedback_weights(
            self.conn,
            global_weight=result.weights.global_semantic,
            segment_weight=result.weights.segment_semantic,
            vocals_global_weight=result.weights.vocals_global_semantic,
            vocals_segment_weight=result.weights.vocals_segment_semantic,
            instrumental_global_weight=result.weights.instrumental_global_semantic,
            instrumental_segment_weight=result.weights.instrumental_segment_semantic,
            event_count=result.event_count,
        )
        self.invalidate_similarity_cache()
        return result.weights

    def feedback_weights(self) -> SimilarityWeights:
        row = database.get_feedback_weights(self.conn)
        if row is None:
            return DEFAULT_WEIGHTS
        return SimilarityWeights(
            global_semantic=float(row["global_weight"]),
            segment_semantic=float(row["segment_weight"]),
            vocals_global_semantic=float(row["vocals_global_weight"]),
            vocals_segment_semantic=float(row["vocals_segment_weight"]),
            instrumental_global_semantic=float(row["instrumental_global_weight"]),
            instrumental_segment_semantic=float(row["instrumental_segment_weight"]),
        )

    def delete_all_tracks(self) -> None:
        rows = list(database.list_tracks(self.conn))
        for row in rows:
            self._delete_track_assets(row)
        self.retrain_feedback_weights()
        self.recompute_layout()

    def delete_track(self, track_id: str) -> bool:
        row = database.get_track(self.conn, track_id)
        if row is None:
            return False
        self._delete_track_assets(row)
        self.retrain_feedback_weights()
        self.recompute_layout()
        return True

    def _delete_track_assets(self, row) -> None:
        track_id = row["id"]
        database.delete_track(self.conn, track_id)
        self.vectors.delete(track_id)
        for path_key in ("audio_path", "art_path", "vocals_path", "instrumental_path"):
            if not row[path_key]:
                continue
            path = Path(row[path_key])
            if path.exists():
                path.unlink()
        for filename in (
            f"{track_id}.model.wav",
            f"{track_id}.vocals.model.wav",
            f"{track_id}.instrumental.model.wav",
        ):
            model_audio_path = self.settings.audio_dir / filename
            if model_audio_path.exists():
                model_audio_path.unlink()


def _available_stems(row: dict) -> list[AudioStem]:
    stems: list[AudioStem] = ["original"]
    if row.get("vocals_path"):
        stems.append("vocals")
    if row.get("instrumental_path"):
        stems.append("instrumental")
    return stems
