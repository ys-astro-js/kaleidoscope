from pathlib import Path

from app.feedback import DEFAULT_WEIGHTS, MIN_TRAINING_EVENTS, learn_feedback_weights
from app.vector_store import (
    COVER_CHROMA_KIND,
    GLOBAL_SEMANTIC_KIND,
    SEGMENT_SEMANTIC_KIND,
    EmbeddingRecord,
    VectorStore,
)


def record(kind: str, vector: list[float], segment_index: int = -1) -> EmbeddingRecord:
    return EmbeddingRecord(kind=kind, segment_index=segment_index, vector=vector)


def test_feedback_learning_keeps_default_weights_with_too_little_data(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "query": [record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0])],
            "candidate": [record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0])],
        },
    )

    result = learn_feedback_weights(
        store,
        [
            {
                "query_track_id": "query",
                "candidate_track_id": "candidate",
                "label": "similar",
            }
        ],
    )

    assert result.event_count == 1
    assert result.weights == DEFAULT_WEIGHTS
    assert MIN_TRAINING_EVENTS > result.event_count


def test_feedback_learning_increases_predictive_chroma_weight(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    query_chroma = [1.0, 1.0, *([0.0] * 10)]
    matching_chroma = [0.0, 0.0, 1.0, 1.0, *([0.0] * 8)]
    different_chroma = [1.0, 0.0, 1.0, *([0.0] * 9)]
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "query": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [1.0, 0.0], 0),
                record(COVER_CHROMA_KIND, query_chroma),
            ],
            "cover_a": [
                record(GLOBAL_SEMANTIC_KIND, [0.0, 1.0]),
                record(SEGMENT_SEMANTIC_KIND, [0.0, 1.0], 0),
                record(COVER_CHROMA_KIND, matching_chroma),
            ],
            "cover_b": [
                record(GLOBAL_SEMANTIC_KIND, [0.0, 1.0]),
                record(SEGMENT_SEMANTIC_KIND, [0.0, 1.0], 0),
                record(COVER_CHROMA_KIND, matching_chroma),
            ],
            "genre_a": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [1.0, 0.0], 0),
                record(COVER_CHROMA_KIND, different_chroma),
            ],
            "genre_b": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [1.0, 0.0], 0),
                record(COVER_CHROMA_KIND, different_chroma),
            ],
        },
    )

    result = learn_feedback_weights(
        store,
        [
            {"query_track_id": "query", "candidate_track_id": "cover_a", "label": "similar"},
            {"query_track_id": "query", "candidate_track_id": "cover_b", "label": "similar"},
            {"query_track_id": "query", "candidate_track_id": "genre_a", "label": "not_similar"},
            {"query_track_id": "query", "candidate_track_id": "genre_b", "label": "not_similar"},
        ],
    )

    assert result.event_count == 4
    assert result.weights.cover_chroma > DEFAULT_WEIGHTS.cover_chroma
