import sqlite3
import threading
from pathlib import Path

from app import database
from app.audio import extract_metadata, normalize_audio_for_model, write_art_or_placeholder
from app.config import Settings
from app.cover_identity import (
    CoverIdentityFeature,
    combine_cover_similarity_scores,
    cover_identity_similarity,
    extract_cover_identity_feature,
)
from app.embedding import MIN_SEGMENT_ERROR, TrackEmbeddings, get_embedder
from app.feedback import DEFAULT_WEIGHTS, learn_feedback_weights
from app.layout import compute_similarity_layout
from app.models import AudioStem, FeedbackLabel, SimilarityMix, SimilarTrack, Track
from app.separation import separate_vocals_and_instrumental
from app.vector_store import SEGMENT_HOP_SECONDS, SimilarityWeights, VectorStore

SimilarById = dict[str, list[dict[str, float | str]]]
SegmentMatches = list[dict[str, float | int | str]]
SIMILAR_TRACK_LIMIT = 5
RANKING_RAW_WEIGHT = 0.50
RANKING_REVERSE_RANK_WEIGHT = 0.20
RANKING_SEGMENT_COVERAGE_WEIGHT = 0.20
RANKING_HUBNESS_WEIGHT = 0.10


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
            embeddings = self._ready_embeddings()
            matrix = self._combined_similarity_matrix(weights, embeddings=embeddings)
            transition_matrix = (
                self.vectors.segment_coverage_matrix(embeddings=embeddings)
                if hasattr(self.vectors, "segment_coverage_matrix")
                else {}
            )
            similar_by_id = _similar_by_id_from_matrix(
                matrix,
                transition_matrix=transition_matrix,
                limit=SIMILAR_TRACK_LIMIT,
            )
            if self.feedback_weights() != weights:
                return
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
                cover_identity_feature = self._extract_optional_cover_identity(
                    Path(row["audio_path"])
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
            if cover_identity_feature is not None:
                database.set_track_cover_identity_feature(
                    self.conn,
                    track_id,
                    cover_identity_feature,
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

    def _extract_optional_cover_identity(self, input_path: Path) -> CoverIdentityFeature | None:
        try:
            return extract_cover_identity_feature(input_path, self.settings.cover_model_dir)
        except Exception:
            return None

    def recompute_layout(self) -> None:
        similarities = self._combined_similarity_matrix(self.feedback_weights())
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
        self.recompute_layout()

    def backfill_identity_features(self) -> None:
        self.backfill_cover_identity_features()

    def similar_segments(
        self,
        track_id: str,
        segment_index: int,
        *,
        limit: int = 5,
    ) -> SegmentMatches:
        style_matches = {
            str(item["id"]): item
            for item in self.vectors.similar_segments(track_id, segment_index, limit=max(limit, 50))
        }
        mix = self.similarity_mix()
        ready_ids = {row["id"] for row in database.ready_tracks(self.conn)}
        cover_features = (
            self._cover_identity_features(ready_ids)
            if mix.cover > 0.0
            else {}
        )
        query_cover_feature = cover_features.get(track_id)

        scored: list[tuple[str, float, int]] = []
        for candidate_id, style_match in style_matches.items():
            if candidate_id == track_id:
                continue
            if candidate_id not in ready_ids:
                continue
            cover_score = None
            candidate_cover_feature = cover_features.get(candidate_id)
            if query_cover_feature is not None and candidate_cover_feature is not None:
                cover_score = cover_identity_similarity(
                    query_cover_feature,
                    candidate_cover_feature,
                )

            score = combine_cover_similarity_scores(
                float(style_match["score"]),
                cover_score,
                cover_weight=mix.cover,
            )
            if score is None:
                continue
            scored.append((candidate_id, score, int(style_match["segment_index"])))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            {
                "id": candidate_id,
                "score": score,
                "segment_index": candidate_segment_index,
                "start_seconds": candidate_segment_index * SEGMENT_HOP_SECONDS,
            }
            for candidate_id, score, candidate_segment_index in scored[:limit]
        ]

    def _combined_similarity_matrix(
        self,
        weights: SimilarityWeights,
        *,
        embeddings: dict | None = None,
    ) -> dict[str, dict[str, float]]:
        embeddings = embeddings if embeddings is not None else self._ready_embeddings()
        style_matrix = self.vectors.similarity_matrix(embeddings=embeddings, weights=weights)
        mix = self.similarity_mix()
        cover_features = self._cover_identity_features(set(style_matrix))
        if not cover_features or mix.cover <= 0.0:
            return style_matrix

        matrix = {track_id: dict(scores) for track_id, scores in style_matrix.items()}
        track_ids = list(matrix)
        for first_index, first_id in enumerate(track_ids):
            first_feature = cover_features.get(first_id)
            for second_id in track_ids[first_index + 1 :]:
                second_feature = cover_features.get(second_id)
                cover_score = (
                    cover_identity_similarity(first_feature, second_feature)
                    if first_feature is not None and second_feature is not None
                    else None
                )
                style_score = matrix.get(first_id, {}).get(second_id)
                score = combine_cover_similarity_scores(
                    style_score,
                    cover_score,
                    cover_weight=mix.cover,
                )
                if score is None:
                    continue
                matrix[first_id][second_id] = score
                matrix[second_id][first_id] = score
        return matrix

    def _ready_embeddings(self) -> dict:
        ready_ids = {row["id"] for row in database.ready_tracks(self.conn)}
        return {
            track_id: records
            for track_id, records in self.vectors.all_embeddings().items()
            if track_id in ready_ids
        }

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
        return self.feedback_weights()

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
            stems = _normalize_stem_mix(
                database.DEFAULT_VOCALS_SIMILARITY_WEIGHT,
                database.DEFAULT_INSTRUMENTAL_SIMILARITY_WEIGHT,
            )
            return _with_style_cover_mix(
                stems,
                database.DEFAULT_STYLE_SIMILARITY_WEIGHT,
                database.DEFAULT_COVER_SIMILARITY_WEIGHT,
            )
        stems = _normalize_stem_mix(
            float(row["vocals_weight"]),
            float(row["instrumental_weight"]),
        )
        return _with_style_cover_mix(
            stems,
            float(row["style_weight"]),
            float(row["cover_weight"]),
        )

    def set_similarity_mix(
        self,
        *,
        vocals: float,
        instrumental: float,
        style: float | None = None,
        cover: float | None = None,
    ) -> SimilarityMix:
        current_mix = self.similarity_mix()
        stem_mix = _normalize_stem_mix(vocals, instrumental)
        mix = _with_style_cover_mix(
            stem_mix,
            current_mix.style if style is None else style,
            current_mix.cover if cover is None else cover,
        )
        database.set_similarity_mix(
            self.conn,
            vocals_weight=mix.vocals,
            instrumental_weight=mix.instrumental,
            style_weight=mix.style,
            cover_weight=mix.cover,
        )
        self.invalidate_similarity_cache()
        self.recompute_layout()
        self._refresh_similarity_cache(self.feedback_weights())
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


def _similar_by_id_from_matrix(
    matrix: dict[str, dict[str, float]],
    *,
    transition_matrix: dict[str, dict[str, float]] | None = None,
    limit: int,
) -> SimilarById:
    ranking_matrix = _ranking_matrix(matrix, transition_matrix or {})
    similar_by_id: SimilarById = {}
    for track_id, scores in ranking_matrix.items():
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


def _ranking_matrix(
    matrix: dict[str, dict[str, float]],
    transition_matrix: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    ranks = _rank_by_id(matrix)
    hubness_scores = _top_hit_hubness_scores(ranks)
    ranking_matrix: dict[str, dict[str, float]] = {}
    for track_id, scores in matrix.items():
        ranking_matrix[track_id] = {}
        for candidate_id, raw_score in scores.items():
            if candidate_id == track_id:
                ranking_matrix[track_id][candidate_id] = raw_score
                continue
            weighted_score = RANKING_RAW_WEIGHT * raw_score
            weight_sum = RANKING_RAW_WEIGHT

            reverse_rank = ranks.get(candidate_id, {}).get(track_id)
            if reverse_rank is not None:
                weighted_score += RANKING_REVERSE_RANK_WEIGHT / reverse_rank
                weight_sum += RANKING_REVERSE_RANK_WEIGHT

            transition_score = transition_matrix.get(track_id, {}).get(candidate_id)
            if transition_score is not None:
                weighted_score += RANKING_SEGMENT_COVERAGE_WEIGHT * transition_score
                weight_sum += RANKING_SEGMENT_COVERAGE_WEIGHT

            weighted_score += RANKING_HUBNESS_WEIGHT * hubness_scores.get(candidate_id, 1.0)
            weight_sum += RANKING_HUBNESS_WEIGHT
            ranking_matrix[track_id][candidate_id] = weighted_score / weight_sum
    return ranking_matrix


def _rank_by_id(matrix: dict[str, dict[str, float]]) -> dict[str, dict[str, int]]:
    ranks: dict[str, dict[str, int]] = {}
    for track_id, scores in matrix.items():
        ranked_ids = [
            candidate_id
            for candidate_id, _score in sorted(
                scores.items(),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            if candidate_id != track_id
        ]
        ranks[track_id] = {
            candidate_id: rank
            for rank, candidate_id in enumerate(ranked_ids, start=1)
        }
    return ranks


def _top_hit_hubness_scores(ranks: dict[str, dict[str, int]]) -> dict[str, float]:
    top_hit_counts: dict[str, int] = {}
    for track_ranks in ranks.values():
        for candidate_id, rank in track_ranks.items():
            if rank == 1:
                top_hit_counts[candidate_id] = top_hit_counts.get(candidate_id, 0) + 1
                break
    return {
        candidate_id: 1.0 / (count**0.5)
        for candidate_id, count in top_hit_counts.items()
    }


def _normalize_stem_mix(vocals: float, instrumental: float) -> SimilarityMix:
    stem_total = vocals + instrumental
    if stem_total <= 0.0:
        return SimilarityMix(
            whole=database.DEFAULT_WHOLE_SIMILARITY_WEIGHT,
            vocals=database.DEFAULT_VOCALS_SIMILARITY_WEIGHT,
            instrumental=database.DEFAULT_INSTRUMENTAL_SIMILARITY_WEIGHT,
            style=database.DEFAULT_STYLE_SIMILARITY_WEIGHT,
            cover=database.DEFAULT_COVER_SIMILARITY_WEIGHT,
        )

    stem_bucket = 1.0 - database.DEFAULT_WHOLE_SIMILARITY_WEIGHT
    return SimilarityMix(
        whole=database.DEFAULT_WHOLE_SIMILARITY_WEIGHT,
        vocals=stem_bucket * vocals / stem_total,
        instrumental=stem_bucket * instrumental / stem_total,
        style=database.DEFAULT_STYLE_SIMILARITY_WEIGHT,
        cover=database.DEFAULT_COVER_SIMILARITY_WEIGHT,
    )


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
    pair_total = global_weight + segment_weight
    if pair_total <= 0.0:
        pair_total = default_global_weight + default_segment_weight
        global_weight = default_global_weight
        segment_weight = default_segment_weight
    return (
        group_weight * global_weight / pair_total,
        group_weight * segment_weight / pair_total,
    )
