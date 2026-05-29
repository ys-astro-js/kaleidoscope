import sqlite3
import threading
import json
from dataclasses import replace
from pathlib import Path

from app import database
from app.audio import (
    align_audio_for_playback,
    extract_metadata,
    normalize_audio_for_model,
    normalize_audio_for_playback,
    write_art_or_placeholder,
)
from app.config import Settings
from app.cover_alignment import (
    CoverAlignmentFeature,
    cover_alignment_scores,
    cover_alignment_segment_score,
    extract_cover_alignment_feature,
    release_cover_alignment_resources,
)
from app.cover_identity import (
    CoverIdentityFeature,
    cover_identity_scores,
    extract_cover_identity_feature,
)
from app.embedding import MIN_SEGMENT_ERROR, TrackEmbeddings, get_embedder
from app.feedback import DEFAULT_WEIGHTS, learn_feedback_reranker
from app.layout import compute_similarity_layout
from app.models import AudioStem, FeedbackLabel, SimilarityMix, SimilarTrack, Track
from app.reranker import (
    DEFAULT_RERANKER_COEFFICIENTS,
    PairEvidence,
    RerankerCoefficients,
    coefficients_from_mapping,
    coefficients_to_dict,
    rerank_score,
)
from app.separation import SEPARATOR_SAMPLE_RATE, separate_instrumental
from app.vector_store import SEGMENT_HOP_SECONDS, SimilarityWeights, VectorStore

SimilarById = dict[str, list[dict[str, float | str]]]
SegmentMatches = list[dict[str, float | int | str]]
SimilarityCacheKey = tuple[RerankerCoefficients, tuple[float, float, float, float, float]]
SegmentCacheKey = tuple[str, int, int, SimilarityCacheKey]
SIMILAR_TRACK_LIMIT = 5


class TrackService:
    def __init__(self, settings: Settings, conn: sqlite3.Connection, vectors: VectorStore) -> None:
        self.settings = settings
        self.conn = conn
        self.vectors = vectors
        self._similar_lock = threading.Lock()
        self._similar_cache: SimilarById = {}
        self._similar_cache_key: SimilarityCacheKey | None = None
        self._segment_cache: dict[SegmentCacheKey, SegmentMatches] = {}
        self._similar_refreshing = False
        self._embedding_lock = threading.Lock()
        self._ready_ids_cache: set[str] | None = None
        self._ready_embeddings_cache: dict | None = None
        self._normalized_embeddings_cache: dict | None = None
        self._processing_lock = threading.Lock()

    def list_tracks(self) -> list[Track]:
        similar_by_id = self.cached_similar_by_id()
        embedded_ids = set(similar_by_id)
        _, ready_embeddings = self._ready_embedding_snapshot()
        segment_counts = (
            self._segment_counts(ready_embeddings)
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
        coefficients = self.reranker_coefficients()
        mix = self.similarity_mix()
        cache_key = _similarity_cache_key(coefficients, mix)
        with self._similar_lock:
            cached = self._similar_cache
            cache_is_current = self._similar_cache_key == cache_key
            should_refresh = not cache_is_current and not self._similar_refreshing
            if should_refresh:
                self._similar_refreshing = True

        if should_refresh:
            thread = threading.Thread(
                target=self._refresh_similarity_cache,
                args=(coefficients, mix, cache_key),
                daemon=True,
            )
            thread.start()

        return cached

    def invalidate_similarity_cache(self) -> None:
        with self._similar_lock:
            self._similar_cache_key = None
            self._segment_cache = {}

    def invalidate_embedding_cache(self) -> None:
        with self._embedding_lock:
            self._ready_ids_cache = None
            self._ready_embeddings_cache = None
            self._normalized_embeddings_cache = None
        self.invalidate_similarity_cache()

    def _refresh_similarity_cache(
        self,
        coefficients: RerankerCoefficients,
        mix: SimilarityMix | None = None,
        cache_key: SimilarityCacheKey | None = None,
    ) -> None:
        mix = mix or self.similarity_mix()
        cache_key = cache_key or _similarity_cache_key(coefficients, mix)
        try:
            embeddings = self._ready_embeddings()
            matrix = self._combined_similarity_matrix(
                coefficients,
                embeddings=embeddings,
                mix=mix,
            )
            similar_by_id = _similar_by_id_from_matrix(
                matrix,
                limit=SIMILAR_TRACK_LIMIT,
            )
            current_cache_key = _similarity_cache_key(
                self.reranker_coefficients(),
                self.similarity_mix(),
            )
            if current_cache_key != cache_key:
                return
            with self._similar_lock:
                self._similar_cache = similar_by_id
                self._similar_cache_key = cache_key
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
                stem_paths = separate_instrumental(
                    Path(row["audio_path"]),
                    track_id=track_id,
                    output_dir=self.settings.stem_dir,
                    model_dir=self.settings.separator_model_dir,
                )
                playback_path = self.settings.audio_dir / f"{track_id}.playback.wav"
                try:
                    normalize_audio_for_playback(
                        Path(row["audio_path"]),
                        playback_path,
                        SEPARATOR_SAMPLE_RATE,
                    )
                    instrumental_playback_path = (
                        self.settings.audio_dir / f"{track_id}.instrumental.playback.wav"
                    )
                    align_audio_for_playback(
                        playback_path,
                        stem_paths.instrumental_path,
                        instrumental_playback_path,
                        SEPARATOR_SAMPLE_RATE,
                    )
                except ValueError:
                    pass
                embedder = get_embedder(self.settings.model_id, self.settings.sample_rate)
                model_audio_path = self.settings.audio_dir / f"{track_id}.model.wav"
                embeddings = self._embed_for_model(
                    embedder,
                    Path(row["audio_path"]),
                    model_audio_path,
                )
                instrumental_embeddings = self._embed_optional_stem(
                    embedder,
                    stem_paths.instrumental_path,
                    self.settings.audio_dir / f"{track_id}.instrumental.model.wav",
                )
                cover_identity_feature = self._extract_optional_cover_identity(
                    Path(row["audio_path"])
                )
                try:
                    cover_alignment_feature = self._extract_optional_cover_alignment(
                        model_audio_path
                    )
                finally:
                    release_cover_alignment_resources()

            self.vectors.upsert(
                track_id,
                embeddings.global_semantic,
                embeddings.segment_semantic,
                instrumental_global_semantic_vector=(
                    instrumental_embeddings.global_semantic
                    if instrumental_embeddings
                    else None
                ),
                instrumental_segment_semantic_vectors=(
                    instrumental_embeddings.segment_semantic
                    if instrumental_embeddings
                    else None
                ),
            )
            if cover_identity_feature is not None:
                database.set_track_cover_identity_feature(
                    self.conn,
                    track_id,
                    cover_identity_feature,
                )
            if cover_alignment_feature is not None:
                database.set_track_cover_alignment_feature(
                    self.conn,
                    track_id,
                    cover_alignment_feature,
                )
            self.invalidate_similarity_cache()
            database.update_track(
                self.conn,
                track_id,
                status="ready",
                artist=artist,
                album=album,
                art_path=art_path,
                instrumental_path=stem_paths.instrumental_path,
            )
            self.invalidate_embedding_cache()
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

    def _extract_optional_cover_identity(self, input_path: Path) -> CoverIdentityFeature | None:
        try:
            return extract_cover_identity_feature(input_path, self.settings.cover_model_dir)
        except Exception:
            return None

    def _extract_optional_cover_alignment(self, input_path: Path) -> CoverAlignmentFeature | None:
        try:
            return extract_cover_alignment_feature(
                input_path,
                self.settings.cover_alignment_model_dir,
            )
        except Exception:
            return None

    def _cover_alignment_audio_path(self, row) -> Path:
        model_audio_path = self.settings.audio_dir / f"{row['id']}.model.wav"
        if model_audio_path.exists():
            return model_audio_path
        return Path(row["audio_path"])

    def recompute_layout(self) -> None:
        similarities = self._combined_similarity_matrix(self.reranker_coefficients())
        database.set_track_coords(self.conn, compute_similarity_layout(similarities))

    def backfill_cover_identity_features(self) -> None:
        existing_ids = set(database.list_track_cover_identity_features(self.conn))
        changed = False
        for row in database.ready_tracks(self.conn):
            if row["id"] in existing_ids:
                continue
            with self._processing_lock:
                feature = self._extract_optional_cover_identity(Path(row["audio_path"]))
            if feature is None:
                continue
            database.set_track_cover_identity_feature(self.conn, row["id"], feature)
            changed = True
        if changed:
            self.invalidate_similarity_cache()
        if changed or self._ready_layout_is_missing():
            self.recompute_layout()

    def backfill_cover_alignment_features(self) -> None:
        existing_ids = set(database.list_track_cover_alignment_features(self.conn))
        changed = False
        try:
            for row in database.ready_tracks(self.conn):
                if row["id"] in existing_ids:
                    continue
                with self._processing_lock:
                    feature = self._extract_optional_cover_alignment(
                        self._cover_alignment_audio_path(row),
                    )
                if feature is None:
                    continue
                database.set_track_cover_alignment_feature(self.conn, row["id"], feature)
                changed = True
            if changed:
                self.invalidate_similarity_cache()
            if changed or self._ready_layout_is_missing():
                self.recompute_layout()
        finally:
            release_cover_alignment_resources()

    def similar_segments(
        self,
        track_id: str,
        segment_index: int,
        *,
        limit: int = 5,
    ) -> SegmentMatches:
        mix = self.similarity_mix()
        coefficients = self.reranker_coefficients()
        cache_key = (
            track_id,
            segment_index,
            limit,
            _similarity_cache_key(coefficients, mix),
        )
        with self._similar_lock:
            cached = self._segment_cache.get(cache_key)
            if cached is not None:
                return list(cached)

        ready_ids, embeddings = self._ready_embedding_snapshot()
        normalized_embeddings = self._ready_normalized_embeddings(embeddings)
        feature_track_ids = set(embeddings) | ready_ids | {track_id}
        cover_alignment_features = self._cover_alignment_features(feature_track_ids)
        cover_identity_features = self._cover_identity_features(feature_track_ids)
        try:
            raw_style_matches = self.vectors.similar_segments(
                track_id,
                segment_index,
                limit=max(limit, 50),
                embeddings=embeddings,
                normalized_embeddings=normalized_embeddings,
            )
        except TypeError:
            try:
                raw_style_matches = self.vectors.similar_segments(
                    track_id,
                    segment_index,
                    limit=max(limit, 50),
                    embeddings=embeddings,
                )
            except TypeError:
                raw_style_matches = self.vectors.similar_segments(
                    track_id,
                    segment_index,
                    limit=max(limit, 50),
                )
        style_matches = {
            str(item["id"]): item
            for item in raw_style_matches
        }
        scored: list[tuple[str, float, int]] = []
        for candidate_id, style_match in style_matches.items():
            if candidate_id == track_id:
                continue
            if candidate_id not in ready_ids:
                continue
            track_evidence = self._pair_evidence(
                track_id,
                candidate_id,
                embeddings=embeddings,
                normalized_embeddings=normalized_embeddings,
                cover_alignment_features=cover_alignment_features,
                cover_identity_features=cover_identity_features,
                fallback_style_score=float(style_match["score"]),
            )
            track_score = (
                rerank_score(track_evidence, mix, coefficients)
                if track_evidence is not None
                else None
            )
            local_score = float(style_match.get("local_score", style_match["score"]))
            context_score = float(style_match.get("context_score", style_match["score"]))
            cover_segment_score = self._cover_segment_score(
                track_id,
                candidate_id,
                segment_index,
                int(style_match["segment_index"]),
                cover_alignment_features,
            )
            score = _segment_rerank_score(
                local_score=local_score,
                context_score=context_score,
                track_score=track_score,
                cover_segment_score=cover_segment_score,
            )
            scored.append((candidate_id, score, int(style_match["segment_index"])))

        scored.sort(key=lambda item: item[1], reverse=True)
        matches = [
            {
                "id": candidate_id,
                "score": score,
                "segment_index": candidate_segment_index,
                "start_seconds": candidate_segment_index * SEGMENT_HOP_SECONDS,
            }
            for candidate_id, score, candidate_segment_index in scored[:limit]
        ]
        with self._similar_lock:
            self._segment_cache[cache_key] = matches
        return list(matches)

    def _combined_similarity_matrix(
        self,
        coefficients: RerankerCoefficients,
        *,
        embeddings: dict | None = None,
        mix: SimilarityMix | None = None,
    ) -> dict[str, dict[str, float]]:
        if embeddings is None:
            embeddings = self._ready_embeddings()
            normalized_embeddings = self._ready_normalized_embeddings(embeddings)
        else:
            normalized_embeddings = (
                self.vectors.normalized_embeddings(embeddings)
                if hasattr(self.vectors, "normalized_embeddings")
                else None
            )
        mix = mix or self.similarity_mix()
        track_ids = list(embeddings)
        matrix: dict[str, dict[str, float]] = {
            track_id: {track_id: 1.0}
            for track_id in track_ids
        }
        cover_alignment_features = self._cover_alignment_features(set(track_ids))
        cover_identity_features = self._cover_identity_features(set(track_ids))
        fallback_style_matrix = None
        if not hasattr(self.vectors, "pair_evidence"):
            fallback_style_matrix = self.vectors.similarity_matrix(
                embeddings=embeddings,
                weights=self.feedback_weights(),
            )

        track_ids = list(matrix)
        for first_index, first_id in enumerate(track_ids):
            for second_id in track_ids[first_index + 1 :]:
                evidence = self._pair_evidence(
                    first_id,
                    second_id,
                    embeddings=embeddings,
                    normalized_embeddings=normalized_embeddings,
                    cover_alignment_features=cover_alignment_features,
                    cover_identity_features=cover_identity_features,
                    fallback_style_score=(
                        fallback_style_matrix.get(first_id, {}).get(second_id)
                        if fallback_style_matrix is not None
                        else None
                    ),
                )
                score = rerank_score(evidence, mix, coefficients) if evidence is not None else None
                if score is None:
                    continue
                matrix[first_id][second_id] = score
                matrix[second_id][first_id] = score
        return matrix

    def _ready_embeddings(self) -> dict:
        return self._ready_embedding_snapshot()[1]

    def _ready_embedding_snapshot(self) -> tuple[set[str], dict]:
        with self._embedding_lock:
            if self._ready_ids_cache is not None and self._ready_embeddings_cache is not None:
                return self._ready_ids_cache, self._ready_embeddings_cache

        ready_ids = {row["id"] for row in database.ready_tracks(self.conn)}
        embeddings = {
            track_id: records
            for track_id, records in self.vectors.all_embeddings().items()
            if track_id in ready_ids
        }
        with self._embedding_lock:
            self._ready_ids_cache = ready_ids
            self._ready_embeddings_cache = embeddings
            self._normalized_embeddings_cache = None
        return ready_ids, embeddings

    def _ready_normalized_embeddings(self, embeddings: dict) -> dict | None:
        if not hasattr(self.vectors, "normalized_embeddings"):
            return None
        with self._embedding_lock:
            if self._normalized_embeddings_cache is not None:
                return self._normalized_embeddings_cache

        normalized = self.vectors.normalized_embeddings(embeddings)
        with self._embedding_lock:
            self._normalized_embeddings_cache = normalized
        return normalized

    def _segment_counts(self, embeddings: dict) -> dict[str, int]:
        try:
            return self.vectors.segment_counts(embeddings)
        except TypeError:
            return self.vectors.segment_counts()

    def _ready_layout_is_missing(self) -> bool:
        return any(
            row["x"] is None or row["y"] is None or row["z"] is None
            for row in database.ready_tracks(self.conn)
        )

    def _cover_identity_features(self, track_ids: set[str]) -> dict[str, CoverIdentityFeature]:
        if not track_ids:
            return {}
        return {
            track_id: feature
            for track_id, feature in database.list_track_cover_identity_features(
                self.conn
            ).items()
            if track_id in track_ids
        }

    def _cover_alignment_features(self, track_ids: set[str]) -> dict[str, CoverAlignmentFeature]:
        if not track_ids:
            return {}
        return {
            track_id: feature
            for track_id, feature in database.list_track_cover_alignment_features(
                self.conn
            ).items()
            if track_id in track_ids
        }

    def _pair_evidence(
        self,
        query_track_id: str,
        candidate_track_id: str,
        *,
        embeddings: dict,
        normalized_embeddings: dict | None = None,
        cover_alignment_features: dict[str, CoverAlignmentFeature],
        cover_identity_features: dict[str, CoverIdentityFeature],
        fallback_style_score: float | None = None,
    ) -> PairEvidence | None:
        if hasattr(self.vectors, "pair_evidence"):
            if normalized_embeddings is None:
                evidence = self.vectors.pair_evidence(
                    query_track_id,
                    candidate_track_id,
                    embeddings=embeddings,
                )
            else:
                evidence = self.vectors.pair_evidence(
                    query_track_id,
                    candidate_track_id,
                    embeddings=embeddings,
                    normalized_embeddings=normalized_embeddings,
                )
        elif fallback_style_score is not None:
            evidence = PairEvidence(semantic_global=fallback_style_score)
        else:
            evidence = None
        if evidence is None:
            return None

        cover_scores = self._cover_scores(
            query_track_id,
            candidate_track_id,
            cover_alignment_features,
            cover_identity_features,
        )
        if cover_scores is None:
            return evidence
        return replace(
            evidence,
            cover_global=cover_scores.global_score,
            cover_best_segment=cover_scores.best_segment_score,
            cover_alignment_consistency=cover_scores.alignment_consistency,
            cover_available=1.0,
        )

    def _cover_scores(
        self,
        query_track_id: str,
        candidate_track_id: str,
        cover_alignment_features: dict[str, CoverAlignmentFeature],
        cover_identity_features: dict[str, CoverIdentityFeature],
    ):
        query_alignment = cover_alignment_features.get(query_track_id)
        candidate_alignment = cover_alignment_features.get(candidate_track_id)
        if query_alignment is not None and candidate_alignment is not None:
            return cover_alignment_scores(query_alignment, candidate_alignment)

        query_identity = cover_identity_features.get(query_track_id)
        candidate_identity = cover_identity_features.get(candidate_track_id)
        if query_identity is None or candidate_identity is None:
            return None
        return cover_identity_scores(query_identity, candidate_identity)

    def _cover_segment_score(
        self,
        query_track_id: str,
        candidate_track_id: str,
        query_segment_index: int,
        candidate_segment_index: int,
        cover_alignment_features: dict[str, CoverAlignmentFeature],
    ) -> float | None:
        query = cover_alignment_features.get(query_track_id)
        candidate = cover_alignment_features.get(candidate_track_id)
        if query is None or candidate is None:
            return None
        return cover_alignment_segment_score(
            query,
            candidate,
            query_segment_index=query_segment_index,
            candidate_segment_index=candidate_segment_index,
            query_start_seconds=query_segment_index * SEGMENT_HOP_SECONDS,
            candidate_start_seconds=candidate_segment_index * SEGMENT_HOP_SECONDS,
        )

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
        self.retrain_feedback_reranker()
        self.recompute_layout()

    def retrain_feedback_reranker(self) -> RerankerCoefficients:
        result = learn_feedback_reranker(
            self._feedback_reranker_examples(),
            default=DEFAULT_RERANKER_COEFFICIENTS,
        )
        database.set_feedback_reranker(
            self.conn,
            coefficients=coefficients_to_dict(result.coefficients),
            event_count=result.event_count,
        )
        self.invalidate_similarity_cache()
        return self.reranker_coefficients()

    def retrain_feedback_weights(self) -> RerankerCoefficients:
        return self.retrain_feedback_reranker()

    def reranker_coefficients(self) -> RerankerCoefficients:
        row = database.get_feedback_reranker(self.conn)
        if row is None:
            return DEFAULT_RERANKER_COEFFICIENTS
        return coefficients_from_mapping(json.loads(row["coefficients_json"]))

    def _feedback_reranker_examples(self) -> list[tuple[PairEvidence, int]]:
        embeddings = self._ready_embeddings()
        cover_alignment_features = self._cover_alignment_features(set(embeddings))
        cover_identity_features = self._cover_identity_features(set(embeddings))
        examples: list[tuple[PairEvidence, int]] = []
        for event in database.list_feedback_events(self.conn):
            evidence = self._pair_evidence(
                str(event["query_track_id"]),
                str(event["candidate_track_id"]),
                embeddings=embeddings,
                cover_alignment_features=cover_alignment_features,
                cover_identity_features=cover_identity_features,
            )
            if evidence is None:
                continue
            label = 1 if event["label"] == "similar" else 0
            examples.append((evidence, label))
        return examples

    def feedback_weights(self) -> SimilarityWeights:
        row = database.get_feedback_weights(self.conn)
        if row is None:
            return _apply_similarity_mix(DEFAULT_WEIGHTS, self.similarity_mix())
        weights = SimilarityWeights(
            global_semantic=float(row["global_weight"]),
            segment_semantic=float(row["segment_weight"]),
            vocals_global_semantic=float(row["vocals_global_weight"]),
            vocals_segment_semantic=float(row["vocals_segment_weight"]),
            instrumental_global_semantic=float(row["instrumental_global_weight"]),
            instrumental_segment_semantic=float(row["instrumental_segment_weight"]),
        )
        return _apply_similarity_mix(weights, self.similarity_mix())

    def similarity_mix(self) -> SimilarityMix:
        row = database.get_similarity_mix(self.conn)
        if row is None:
            return _with_style_cover_mix(
                _stem_mix(
                    database.DEFAULT_WHOLE_SIMILARITY_WEIGHT,
                    database.DEFAULT_INSTRUMENTAL_SIMILARITY_WEIGHT,
                ),
                database.DEFAULT_STYLE_SIMILARITY_WEIGHT,
                database.DEFAULT_COVER_SIMILARITY_WEIGHT,
            )
        return _with_style_cover_mix(
            _stem_mix(
                1.0 - float(row["instrumental_weight"]),
                float(row["instrumental_weight"]),
            ),
            float(row["style_weight"]),
            float(row["cover_weight"]),
        )

    def set_similarity_mix(
        self,
        *,
        whole: float | None = None,
        instrumental: float | None = None,
        style: float | None = None,
        cover: float | None = None,
    ) -> SimilarityMix:
        current_mix = self.similarity_mix()
        stem_mix = _stem_mix(
            current_mix.whole if whole is None else whole,
            current_mix.instrumental if instrumental is None else instrumental,
        )
        mix = _with_style_cover_mix(
            stem_mix,
            current_mix.style if style is None else style,
            current_mix.cover if cover is None else cover,
        )
        database.set_similarity_mix(
            self.conn,
            vocals_weight=0.0,
            instrumental_weight=mix.instrumental,
            style_weight=mix.style,
            cover_weight=mix.cover,
        )
        self.invalidate_similarity_cache()
        self.recompute_layout()
        self._refresh_similarity_cache(self.reranker_coefficients())
        return mix

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
        self.invalidate_embedding_cache()
        for path_key in ("audio_path", "art_path", "vocals_path", "instrumental_path"):
            if not row[path_key]:
                continue
            path = Path(row[path_key])
            if path.exists():
                path.unlink()
        for filename in (
            f"{track_id}.model.wav",
            f"{track_id}.playback.wav",
            f"{track_id}.instrumental.playback.wav",
            f"{track_id}.vocals.model.wav",
            f"{track_id}.instrumental.model.wav",
        ):
            model_audio_path = self.settings.audio_dir / filename
            if model_audio_path.exists():
                model_audio_path.unlink()


def _available_stems(row: dict) -> list[AudioStem]:
    stems: list[AudioStem] = ["original"]
    if row.get("instrumental_path"):
        stems.append("instrumental")
    return stems


def _similar_by_id_from_matrix(
    matrix: dict[str, dict[str, float]],
    *,
    limit: int,
) -> SimilarById:
    similar_by_id: SimilarById = {}
    for track_id, scores in matrix.items():
        ranked = sorted(
            (
                {"id": candidate_id, "score": score}
                for candidate_id, score in scores.items()
                if candidate_id != track_id
            ),
            key=lambda item: float(scores[str(item["id"])]),
            reverse=True,
        )
        similar_by_id[track_id] = ranked[:limit]
    return similar_by_id


def _stem_mix(whole: float, instrumental: float) -> SimilarityMix:
    total = whole + instrumental
    if total <= 0.0:
        whole = database.DEFAULT_WHOLE_SIMILARITY_WEIGHT
        instrumental = database.DEFAULT_INSTRUMENTAL_SIMILARITY_WEIGHT
        total = whole + instrumental
    return SimilarityMix(
        whole=whole / total,
        vocals=0.0,
        instrumental=instrumental / total,
        style=database.DEFAULT_STYLE_SIMILARITY_WEIGHT,
        cover=database.DEFAULT_COVER_SIMILARITY_WEIGHT,
    )


def _similarity_cache_key(
    coefficients: RerankerCoefficients,
    mix: SimilarityMix,
) -> SimilarityCacheKey:
    return (
        coefficients,
        (
            mix.whole,
            mix.vocals,
            mix.instrumental,
            mix.style,
            mix.cover,
        ),
    )


def _segment_rerank_score(
    *,
    local_score: float,
    context_score: float,
    track_score: float | None,
    cover_segment_score: float | None,
) -> float:
    weighted_score = local_score * 0.45 + context_score * 0.20
    weight_sum = 0.65
    if track_score is not None:
        weighted_score += track_score * 0.25
        weight_sum += 0.25
    if cover_segment_score is not None:
        weighted_score += cover_segment_score * 0.10
        weight_sum += 0.10
    return max(0.0, min(1.0, weighted_score / weight_sum))


def _with_style_cover_mix(stem_mix: SimilarityMix, style: float, cover: float) -> SimilarityMix:
    total = style + cover
    if total <= 0.0:
        style = database.DEFAULT_STYLE_SIMILARITY_WEIGHT
        cover = database.DEFAULT_COVER_SIMILARITY_WEIGHT
        total = style + cover
    return SimilarityMix(
        whole=stem_mix.whole,
        vocals=stem_mix.vocals,
        instrumental=stem_mix.instrumental,
        style=style / total,
        cover=cover / total,
    )


def _apply_similarity_mix(weights: SimilarityWeights, mix: SimilarityMix) -> SimilarityWeights:
    global_semantic, segment_semantic = _scale_feature_pair(
        weights.global_semantic,
        weights.segment_semantic,
        DEFAULT_WEIGHTS.global_semantic,
        DEFAULT_WEIGHTS.segment_semantic,
        mix.whole,
    )
    vocals_global_semantic, vocals_segment_semantic = _scale_feature_pair(
        weights.vocals_global_semantic,
        weights.vocals_segment_semantic,
        DEFAULT_WEIGHTS.vocals_global_semantic,
        DEFAULT_WEIGHTS.vocals_segment_semantic,
        mix.vocals,
    )
    instrumental_global_semantic, instrumental_segment_semantic = _scale_feature_pair(
        weights.instrumental_global_semantic,
        weights.instrumental_segment_semantic,
        DEFAULT_WEIGHTS.instrumental_global_semantic,
        DEFAULT_WEIGHTS.instrumental_segment_semantic,
        mix.instrumental,
    )
    return SimilarityWeights(
        global_semantic=global_semantic,
        segment_semantic=segment_semantic,
        vocals_global_semantic=vocals_global_semantic,
        vocals_segment_semantic=vocals_segment_semantic,
        instrumental_global_semantic=instrumental_global_semantic,
        instrumental_segment_semantic=instrumental_segment_semantic,
    )


def _scale_feature_pair(
    global_weight: float,
    segment_weight: float,
    default_global_weight: float,
    default_segment_weight: float,
    group_weight: float,
) -> tuple[float, float]:
    if group_weight <= 0.0:
        return (0.0, 0.0)
    pair_total = global_weight + segment_weight
    if pair_total <= 0.0:
        pair_total = default_global_weight + default_segment_weight
        global_weight = default_global_weight
        segment_weight = default_segment_weight
    if pair_total <= 0.0:
        return (0.0, 0.0)
    return (
        group_weight * global_weight / pair_total,
        group_weight * segment_weight / pair_total,
    )
