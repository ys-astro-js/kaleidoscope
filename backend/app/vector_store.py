from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEGMENT_HOP_SECONDS = 15.0
TABLE_NAME = "track_embeddings"
GLOBAL_SEMANTIC_KIND = "global_semantic"
SEGMENT_SEMANTIC_KIND = "segment_semantic"
VOCALS_GLOBAL_SEMANTIC_KIND = "vocals_global_semantic"
VOCALS_SEGMENT_SEMANTIC_KIND = "vocals_segment_semantic"
INSTRUMENTAL_GLOBAL_SEMANTIC_KIND = "instrumental_global_semantic"
INSTRUMENTAL_SEGMENT_SEMANTIC_KIND = "instrumental_segment_semantic"
AVERAGE_KIND = "average"
SEGMENT_KIND = "segment"
DEFAULT_GLOBAL_WEIGHT = 0.34375
DEFAULT_SEGMENT_WEIGHT = 0.15625
DEFAULT_VOCALS_GLOBAL_WEIGHT = 0.171875
DEFAULT_VOCALS_SEGMENT_WEIGHT = 0.078125
DEFAULT_INSTRUMENTAL_GLOBAL_WEIGHT = 0.171875
DEFAULT_INSTRUMENTAL_SEGMENT_WEIGHT = 0.078125
TOP_SEGMENT_MATCHES = 3
SEGMENT_COVERAGE_MATCHES = 6
WEIGHT_FIELDS = (
    "global_semantic",
    "segment_semantic",
    "vocals_global_semantic",
    "vocals_segment_semantic",
    "instrumental_global_semantic",
    "instrumental_segment_semantic",
)


@dataclass(frozen=True)
class SimilarityWeights:
    global_semantic: float = DEFAULT_GLOBAL_WEIGHT
    segment_semantic: float = DEFAULT_SEGMENT_WEIGHT
    vocals_global_semantic: float = DEFAULT_VOCALS_GLOBAL_WEIGHT
    vocals_segment_semantic: float = DEFAULT_VOCALS_SEGMENT_WEIGHT
    instrumental_global_semantic: float = DEFAULT_INSTRUMENTAL_GLOBAL_WEIGHT
    instrumental_segment_semantic: float = DEFAULT_INSTRUMENTAL_SEGMENT_WEIGHT


@dataclass(frozen=True)
class SimilarityFeatureScores:
    global_semantic: float | None = None
    segment_semantic: float | None = None
    vocals_global_semantic: float | None = None
    vocals_segment_semantic: float | None = None
    instrumental_global_semantic: float | None = None
    instrumental_segment_semantic: float | None = None


@dataclass(frozen=True)
class EmbeddingRecord:
    kind: str
    segment_index: int
    vector: list[float]


@dataclass
class NormalizedEmbeddings:
    global_semantic: np.ndarray | None = None
    segment_semantic: list[np.ndarray] = field(default_factory=list)
    vocals_global_semantic: np.ndarray | None = None
    vocals_segment_semantic: list[np.ndarray] = field(default_factory=list)
    instrumental_global_semantic: np.ndarray | None = None
    instrumental_segment_semantic: list[np.ndarray] = field(default_factory=list)


class VectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._tables = {}

    def upsert(
        self,
        track_id: str,
        global_semantic_vector: list[float],
        segment_semantic_vectors: list[list[float]],
        vocals_global_semantic_vector: list[float] | None = None,
        vocals_segment_semantic_vectors: list[list[float]] | None = None,
        instrumental_global_semantic_vector: list[float] | None = None,
        instrumental_segment_semantic_vectors: list[list[float]] | None = None,
    ) -> None:
        semantic_rows = _embedding_rows(
            track_id,
            GLOBAL_SEMANTIC_KIND,
            SEGMENT_SEMANTIC_KIND,
            global_semantic_vector,
            segment_semantic_vectors,
        )
        if vocals_global_semantic_vector is not None:
            semantic_rows.extend(
                _embedding_rows(
                    track_id,
                    VOCALS_GLOBAL_SEMANTIC_KIND,
                    VOCALS_SEGMENT_SEMANTIC_KIND,
                    vocals_global_semantic_vector,
                    vocals_segment_semantic_vectors or [],
                )
            )
        if instrumental_global_semantic_vector is not None:
            semantic_rows.extend(
                _embedding_rows(
                    track_id,
                    INSTRUMENTAL_GLOBAL_SEMANTIC_KIND,
                    INSTRUMENTAL_SEGMENT_SEMANTIC_KIND,
                    instrumental_global_semantic_vector,
                    instrumental_segment_semantic_vectors or [],
                )
            )

        self.delete(track_id)
        self._add_rows(TABLE_NAME, semantic_rows)

    def delete(self, track_id: str) -> None:
        table = self._open_table(TABLE_NAME)
        if table is None:
            return
        try:
            table.delete(f"track_id = '{track_id}'")
        except Exception:
            return

    def all_vectors(self) -> dict[str, list[float]]:
        return {
            row["track_id"]: row["vector"]
            for row in self._table_rows(TABLE_NAME)
            if _canonical_kind(row.get("kind")) == GLOBAL_SEMANTIC_KIND
        }

    def all_embeddings(self) -> dict[str, list[EmbeddingRecord]]:
        embeddings: dict[str, list[EmbeddingRecord]] = {}
        for row in self._table_rows(TABLE_NAME):
            embeddings.setdefault(row["track_id"], []).append(
                EmbeddingRecord(
                    kind=_canonical_kind(row["kind"]),
                    segment_index=row.get("segment_index", -1),
                    vector=row["vector"],
                )
            )
        return embeddings

    def segment_counts(
        self,
        embeddings: dict[str, list[EmbeddingRecord]] | None = None,
    ) -> dict[str, int]:
        embeddings = embeddings if embeddings is not None else self.all_embeddings()
        return {
            track_id: len(
                {
                    record.segment_index
                    for record in records
                    if _canonical_kind(record.kind) == SEGMENT_SEMANTIC_KIND
                    and record.segment_index >= 0
                }
            )
            for track_id, records in embeddings.items()
        }

    def similar(
        self,
        track_id: str,
        *,
        limit: int = 3,
        embeddings: dict[str, list[EmbeddingRecord]] | None = None,
        weights: SimilarityWeights | None = None,
    ) -> list[dict[str, float | str]]:
        embeddings = embeddings if embeddings is not None else self.all_embeddings()
        normalized = _normalize_embeddings(embeddings)
        return _similar_from_normalized(
            normalized,
            track_id,
            limit=limit,
            weights=weights or SimilarityWeights(),
        )

    def similar_by_track(
        self,
        *,
        limit: int = 3,
        weights: SimilarityWeights | None = None,
    ) -> dict[str, list[dict[str, float | str]]]:
        normalized = _normalize_embeddings(self.all_embeddings())
        resolved_weights = weights or SimilarityWeights()
        return {
            track_id: _similar_from_normalized(
                normalized,
                track_id,
                limit=limit,
                weights=resolved_weights,
            )
            for track_id in normalized
        }

    def similarity_matrix(
        self,
        *,
        embeddings: dict[str, list[EmbeddingRecord]] | None = None,
        weights: SimilarityWeights | None = None,
    ) -> dict[str, dict[str, float]]:
        embeddings = embeddings if embeddings is not None else self.all_embeddings()
        normalized = _normalize_embeddings(embeddings)
        resolved_weights = weights or SimilarityWeights()
        matrix: dict[str, dict[str, float]] = {
            track_id: {track_id: 1.0}
            for track_id in normalized
        }
        ids = list(normalized)

        for first_index, first_id in enumerate(ids):
            for second_id in ids[first_index + 1 :]:
                score = _combined_score(
                    normalized[first_id],
                    normalized[second_id],
                    resolved_weights,
                )
                if score is None:
                    continue
                matrix[first_id][second_id] = score
                matrix[second_id][first_id] = score

        return matrix

    def feature_scores(
        self,
        query_track_id: str,
        candidate_track_id: str,
        *,
        embeddings: dict[str, list[EmbeddingRecord]] | None = None,
    ) -> SimilarityFeatureScores | None:
        embeddings = embeddings if embeddings is not None else self.all_embeddings()
        normalized = _normalize_embeddings(embeddings)
        query = normalized.get(query_track_id)
        candidate = normalized.get(candidate_track_id)
        if query is None or candidate is None:
            return None
        return _feature_scores(query, candidate)

    def similar_segments(
        self,
        track_id: str,
        segment_index: int,
        *,
        limit: int = 5,
        embeddings: dict[str, list[EmbeddingRecord]] | None = None,
    ) -> list[dict[str, float | int | str]]:
        embeddings = embeddings if embeddings is not None else self.all_embeddings()
        normalized = _normalize_embeddings(embeddings)
        query_track = normalized.get(track_id)
        if query_track is None:
            return []
        query_segment = _segment_at(query_track.segment_semantic, segment_index)
        if query_segment is None:
            return []

        scored: list[tuple[str, float, int]] = []
        for candidate_id, candidate in normalized.items():
            if candidate_id == track_id or not candidate.segment_semantic:
                continue
            candidate_scores = [
                (index, _cosine_score(query_segment, candidate_segment))
                for index, candidate_segment in enumerate(candidate.segment_semantic)
            ]
            candidate_segment_index, score = max(candidate_scores, key=lambda item: item[1])
            scored.append((candidate_id, score, candidate_segment_index))

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

    def segment_coverage_matrix(
        self,
        *,
        embeddings: dict[str, list[EmbeddingRecord]] | None = None,
    ) -> dict[str, dict[str, float]]:
        embeddings = embeddings if embeddings is not None else self.all_embeddings()
        normalized = _normalize_embeddings(embeddings)
        matrix: dict[str, dict[str, float]] = {track_id: {track_id: 1.0} for track_id in normalized}
        ids = list(normalized)

        for first_index, first_id in enumerate(ids):
            for second_id in ids[first_index + 1 :]:
                score = _segment_coverage_score(
                    normalized[first_id].segment_semantic,
                    normalized[second_id].segment_semantic,
                )
                if score is None:
                    continue
                matrix[first_id][second_id] = score
                matrix[second_id][first_id] = score

        return matrix

    def _connect(self):
        import lancedb

        self.path.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(self.path))

    def _add_rows(self, table_name: str, rows: list[dict]) -> None:
        if not rows:
            return

        table = self._open_table(table_name)
        if table is None:
            self._tables[table_name] = self._connect().create_table(table_name, data=rows)
            return

        table.add(rows)

    def _table_rows(self, table_name: str) -> list[dict]:
        table = self._open_table(table_name)
        if table is None:
            return []
        return table.to_arrow().to_pylist()

    def _open_table(self, table_name: str):
        if table_name in self._tables:
            return self._tables[table_name]

        db = self._connect()
        if table_name in db.table_names():
            self._tables[table_name] = db.open_table(table_name)
            return self._tables[table_name]

        return None


def _embedding_rows(
    track_id: str,
    global_kind: str,
    segment_kind: str,
    global_vector: list[float],
    segment_vectors: list[list[float]],
) -> list[dict]:
    return [
        {
            "id": f"{track_id}:{global_kind}",
            "track_id": track_id,
            "kind": global_kind,
            "segment_index": -1,
            "vector": global_vector,
        },
        *[
            {
                "id": f"{track_id}:{segment_kind}:{idx}",
                "track_id": track_id,
                "kind": segment_kind,
                "segment_index": idx,
                "vector": vector,
            }
            for idx, vector in enumerate(segment_vectors)
        ],
    ]


def _as_normalized_array(vector: list[float]) -> np.ndarray | None:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return None
    return array / norm


def _normalize_embeddings(
    embeddings: dict[str, list[EmbeddingRecord]],
) -> dict[str, NormalizedEmbeddings]:
    normalized: dict[str, NormalizedEmbeddings] = {}
    for track_id, records in embeddings.items():
        track = NormalizedEmbeddings()
        for record in sorted(records, key=lambda item: item.segment_index):
            kind = _canonical_kind(record.kind)
            if kind == GLOBAL_SEMANTIC_KIND:
                array = _as_normalized_array(record.vector)
                if array is None:
                    continue
                track.global_semantic = array
            elif kind == SEGMENT_SEMANTIC_KIND:
                array = _as_normalized_array(record.vector)
                if array is None:
                    continue
                track.segment_semantic.append(array)
            elif kind == VOCALS_GLOBAL_SEMANTIC_KIND:
                array = _as_normalized_array(record.vector)
                if array is None:
                    continue
                track.vocals_global_semantic = array
            elif kind == VOCALS_SEGMENT_SEMANTIC_KIND:
                array = _as_normalized_array(record.vector)
                if array is None:
                    continue
                track.vocals_segment_semantic.append(array)
            elif kind == INSTRUMENTAL_GLOBAL_SEMANTIC_KIND:
                array = _as_normalized_array(record.vector)
                if array is None:
                    continue
                track.instrumental_global_semantic = array
            elif kind == INSTRUMENTAL_SEGMENT_SEMANTIC_KIND:
                array = _as_normalized_array(record.vector)
                if array is None:
                    continue
                track.instrumental_segment_semantic.append(array)
        if _has_any_embedding(track):
            normalized[track_id] = track
    return normalized


def _has_any_embedding(track: NormalizedEmbeddings) -> bool:
    return (
        track.global_semantic is not None
        or bool(track.segment_semantic)
        or track.vocals_global_semantic is not None
        or bool(track.vocals_segment_semantic)
        or track.instrumental_global_semantic is not None
        or bool(track.instrumental_segment_semantic)
    )


def _similar_from_normalized(
    normalized: dict[str, NormalizedEmbeddings],
    track_id: str,
    *,
    limit: int,
    weights: SimilarityWeights,
) -> list[dict[str, float | str]]:
    query = normalized.get(track_id)
    if query is None:
        return []

    scored: list[tuple[str, float]] = []
    for candidate_id, candidate in normalized.items():
        if candidate_id == track_id:
            continue

        score = _combined_score(query, candidate, weights)
        if score is not None:
            scored.append((candidate_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [{"id": candidate_id, "score": score} for candidate_id, score in scored[:limit]]


def _combined_score(
    query: NormalizedEmbeddings,
    candidate: NormalizedEmbeddings,
    weights: SimilarityWeights,
) -> float | None:
    return score_from_features(_feature_scores(query, candidate), weights)


def score_from_features(
    features: SimilarityFeatureScores,
    weights: SimilarityWeights,
) -> float | None:
    weighted_score = 0.0
    weight_sum = 0.0

    for field_name in WEIGHT_FIELDS:
        score = getattr(features, field_name)
        weight = getattr(weights, field_name)
        if score is None or weight <= 0.0:
            continue
        weighted_score += weight * score
        weight_sum += weight

    if weight_sum == 0.0:
        return None
    return _clamp_score(weighted_score / weight_sum)


def _feature_scores(
    query: NormalizedEmbeddings,
    candidate: NormalizedEmbeddings,
) -> SimilarityFeatureScores:
    global_score = None
    if query.global_semantic is not None and candidate.global_semantic is not None:
        global_score = _cosine_score(query.global_semantic, candidate.global_semantic)

    return SimilarityFeatureScores(
        global_semantic=global_score,
        segment_semantic=_top_segment_score(query.segment_semantic, candidate.segment_semantic),
        vocals_global_semantic=_optional_cosine_score(
            query.vocals_global_semantic,
            candidate.vocals_global_semantic,
        ),
        vocals_segment_semantic=_top_segment_score(
            query.vocals_segment_semantic,
            candidate.vocals_segment_semantic,
        ),
        instrumental_global_semantic=_optional_cosine_score(
            query.instrumental_global_semantic,
            candidate.instrumental_global_semantic,
        ),
        instrumental_segment_semantic=_top_segment_score(
            query.instrumental_segment_semantic,
            candidate.instrumental_segment_semantic,
        ),
    )


def _top_segment_score(
    query_arrays: list[np.ndarray],
    candidate_arrays: list[np.ndarray],
    *,
    top_k: int = TOP_SEGMENT_MATCHES,
) -> float | None:
    if not query_arrays or not candidate_arrays:
        return None

    scores = sorted(
        (
            _cosine_score(query, candidate)
            for query in query_arrays
            for candidate in candidate_arrays
        ),
        reverse=True,
    )
    return _clamp_score(float(np.mean(scores[:top_k])))


def _segment_coverage_score(
    query_arrays: list[np.ndarray],
    candidate_arrays: list[np.ndarray],
) -> float | None:
    if not query_arrays or not candidate_arrays:
        return None

    target_matches = min(len(query_arrays), len(candidate_arrays), SEGMENT_COVERAGE_MATCHES)
    coverage_target = min(max(len(query_arrays), len(candidate_arrays)), SEGMENT_COVERAGE_MATCHES)
    scored_pairs = sorted(
        (
            (_cosine_score(query, candidate), query_index, candidate_index)
            for query_index, query in enumerate(query_arrays)
            for candidate_index, candidate in enumerate(candidate_arrays)
        ),
        reverse=True,
    )
    used_queries = set()
    used_candidates = set()
    selected_scores: list[float] = []
    for score, query_index, candidate_index in scored_pairs:
        if query_index in used_queries or candidate_index in used_candidates:
            continue
        selected_scores.append(score)
        used_queries.add(query_index)
        used_candidates.add(candidate_index)
        if len(selected_scores) >= target_matches:
            break

    if not selected_scores:
        return None
    coverage = len(selected_scores) / coverage_target
    return _clamp_score(float(np.mean(selected_scores)) * coverage)


def _optional_cosine_score(
    query: np.ndarray | None,
    candidate: np.ndarray | None,
) -> float | None:
    if query is None or candidate is None:
        return None
    return _cosine_score(query, candidate)


def _segment_at(segments: list[np.ndarray], segment_index: int) -> np.ndarray | None:
    if segment_index < 0 or segment_index >= len(segments):
        return None
    return segments[segment_index]


def _cosine_score(query: np.ndarray, candidate: np.ndarray) -> float:
    if len(query) != len(candidate):
        return 0.0
    return _clamp_score(float(np.dot(query, candidate)))


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def _canonical_kind(kind: str | None) -> str:
    if kind == AVERAGE_KIND:
        return GLOBAL_SEMANTIC_KIND
    if kind == SEGMENT_KIND:
        return SEGMENT_SEMANTIC_KIND
    return kind or ""
