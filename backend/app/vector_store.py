from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TABLE_NAME = "track_embeddings"
CHROMA_TABLE_NAME = "track_chroma_embeddings"
GLOBAL_SEMANTIC_KIND = "global_semantic"
SEGMENT_SEMANTIC_KIND = "segment_semantic"
COVER_CHROMA_KIND = "cover_chroma"
AVERAGE_KIND = "average"
SEGMENT_KIND = "segment"
DEFAULT_GLOBAL_WEIGHT = 0.55
DEFAULT_SEGMENT_WEIGHT = 0.25
DEFAULT_CHROMA_WEIGHT = 0.20
TOP_SEGMENT_MATCHES = 3


@dataclass(frozen=True)
class SimilarityWeights:
    global_semantic: float = DEFAULT_GLOBAL_WEIGHT
    segment_semantic: float = DEFAULT_SEGMENT_WEIGHT
    cover_chroma: float = DEFAULT_CHROMA_WEIGHT


@dataclass(frozen=True)
class SimilarityFeatureScores:
    global_semantic: float | None
    segment_semantic: float | None
    cover_chroma: float | None


@dataclass(frozen=True)
class EmbeddingRecord:
    kind: str
    segment_index: int
    vector: list[float]


@dataclass
class NormalizedEmbeddings:
    global_semantic: np.ndarray | None = None
    segment_semantic: list[np.ndarray] = field(default_factory=list)
    cover_chroma: np.ndarray | None = None


class VectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._tables = {}

    def upsert(
        self,
        track_id: str,
        global_semantic_vector: list[float],
        segment_semantic_vectors: list[list[float]],
        cover_chroma_vector: list[float] | None = None,
    ) -> None:
        semantic_rows = [
            {
                "id": f"{track_id}:global_semantic",
                "track_id": track_id,
                "kind": GLOBAL_SEMANTIC_KIND,
                "segment_index": -1,
                "vector": global_semantic_vector,
            },
            *[
                {
                    "id": f"{track_id}:segment_semantic:{idx}",
                    "track_id": track_id,
                    "kind": SEGMENT_SEMANTIC_KIND,
                    "segment_index": idx,
                    "vector": vector,
                }
                for idx, vector in enumerate(segment_semantic_vectors)
            ],
        ]
        chroma_rows = []
        if cover_chroma_vector:
            chroma_rows.append(
                {
                    "id": f"{track_id}:cover_chroma",
                    "track_id": track_id,
                    "kind": COVER_CHROMA_KIND,
                    "segment_index": -1,
                    "vector": cover_chroma_vector,
                }
            )

        self.delete(track_id)
        self._add_rows(TABLE_NAME, semantic_rows)
        self._add_rows(CHROMA_TABLE_NAME, chroma_rows)

    def delete(self, track_id: str) -> None:
        for table_name in (TABLE_NAME, CHROMA_TABLE_NAME):
            table = self._open_table(table_name)
            if table is None:
                continue
            try:
                table.delete(f"track_id = '{track_id}'")
            except Exception:
                continue

    def all_vectors(self) -> dict[str, list[float]]:
        return {
            row["track_id"]: row["vector"]
            for row in self._table_rows(TABLE_NAME)
            if _canonical_kind(row.get("kind")) == GLOBAL_SEMANTIC_KIND
        }

    def all_embeddings(self) -> dict[str, list[EmbeddingRecord]]:
        embeddings: dict[str, list[EmbeddingRecord]] = {}
        for row in [*self._table_rows(TABLE_NAME), *self._table_rows(CHROMA_TABLE_NAME)]:
            embeddings.setdefault(row["track_id"], []).append(
                EmbeddingRecord(
                    kind=_canonical_kind(row["kind"]),
                    segment_index=row.get("segment_index", -1),
                    vector=row["vector"],
                )
            )
        return embeddings

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
        for record in records:
            array = _as_normalized_array(record.vector)
            if array is None:
                continue
            kind = _canonical_kind(record.kind)
            if kind == GLOBAL_SEMANTIC_KIND:
                track.global_semantic = array
            elif kind == SEGMENT_SEMANTIC_KIND:
                track.segment_semantic.append(array)
            elif kind == COVER_CHROMA_KIND:
                track.cover_chroma = array
        if track.global_semantic is not None or track.segment_semantic or track.cover_chroma is not None:
            normalized[track_id] = track
    return normalized


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

    for score, weight in (
        (features.global_semantic, weights.global_semantic),
        (features.segment_semantic, weights.segment_semantic),
        (features.cover_chroma, weights.cover_chroma),
    ):
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
        cover_chroma=_chroma_cover_score(query.cover_chroma, candidate.cover_chroma),
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


def _chroma_cover_score(
    query: np.ndarray | None,
    candidate: np.ndarray | None,
) -> float | None:
    if query is None or candidate is None:
        return None
    if len(query) < 12 or len(candidate) < 12:
        return _cosine_score(query, candidate)

    query_profile = _as_normalized_array(query[:12].tolist())
    candidate_profile = _as_normalized_array(candidate[:12].tolist())
    if query_profile is None or candidate_profile is None:
        return None

    return max(
        _cosine_score(query_profile, np.roll(candidate_profile, shift))
        for shift in range(12)
    )


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
