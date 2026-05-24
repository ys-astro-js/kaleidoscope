from pathlib import Path

from app.feedback import DEFAULT_WEIGHTS, MIN_TRAINING_EVENTS, learn_feedback_weights
from app.vector_store import (
    GLOBAL_SEMANTIC_KIND,
    SEGMENT_SEMANTIC_KIND,
    EmbeddingRecord,
    VectorStore,
    WEIGHT_FIELDS,
)


def record(kind: str, vector: list[float], segment_index: int = -1) -> EmbeddingRecord:
    return EmbeddingRecord(kind=kind, segment_index=segment_index, vector=vector)


def test_feedback_learning_uses_single_similar_event(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "query": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [0.0, 1.0], 0),
            ],
            "candidate": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [1.0, 0.0], 0),
            ],
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
    assert MIN_TRAINING_EVENTS == 1
    assert result.weights.global_semantic > DEFAULT_WEIGHTS.global_semantic


def test_feedback_learning_uses_single_not_similar_event(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "query": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [0.0, 1.0], 0),
            ],
            "candidate": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [1.0, 0.0], 0),
            ],
        },
    )

    result = learn_feedback_weights(
        store,
        [
            {
                "query_track_id": "query",
                "candidate_track_id": "candidate",
                "label": "not_similar",
            }
        ],
    )

    assert result.event_count == 1
    assert result.weights.global_semantic < DEFAULT_WEIGHTS.global_semantic


def test_feedback_learning_uses_only_global_and_segment_features(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "query": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [1.0, 0.0], 0),
            ],
            "cover_a": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [1.0, 0.0], 0),
            ],
            "cover_b": [
                record(GLOBAL_SEMANTIC_KIND, [1.0, 0.0]),
                record(SEGMENT_SEMANTIC_KIND, [1.0, 0.0], 0),
            ],
        },
    )

    result = learn_feedback_weights(
        store,
        [
            {"query_track_id": "query", "candidate_track_id": "cover_a", "label": "similar"},
            {"query_track_id": "query", "candidate_track_id": "cover_b", "label": "similar"},
        ],
    )

    assert result.event_count == 2
    assert sum(getattr(result.weights, field_name) for field_name in WEIGHT_FIELDS) == 1.0
