from pathlib import Path

from app.vector_store import (
    GLOBAL_SEMANTIC_KIND,
    INSTRUMENTAL_GLOBAL_SEMANTIC_KIND,
    INSTRUMENTAL_SEGMENT_SEMANTIC_KIND,
    SEGMENT_SEMANTIC_KIND,
    VOCALS_GLOBAL_SEMANTIC_KIND,
    VOCALS_SEGMENT_SEMANTIC_KIND,
    EmbeddingRecord,
    SimilarityWeights,
    VectorStore,
)


def global_semantic(vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=GLOBAL_SEMANTIC_KIND, segment_index=-1, vector=vector)


def segment_semantic(index: int, vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=SEGMENT_SEMANTIC_KIND, segment_index=index, vector=vector)


def vocals_global(vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=VOCALS_GLOBAL_SEMANTIC_KIND, segment_index=-1, vector=vector)


def vocals_segment(index: int, vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=VOCALS_SEGMENT_SEMANTIC_KIND, segment_index=index, vector=vector)


def instrumental_global(vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=INSTRUMENTAL_GLOBAL_SEMANTIC_KIND, segment_index=-1, vector=vector)


def instrumental_segment(index: int, vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(
        kind=INSTRUMENTAL_SEGMENT_SEMANTIC_KIND,
        segment_index=index,
        vector=vector,
    )


def weights(**overrides: float) -> SimilarityWeights:
    values = {
        "global_semantic": 0.0,
        "segment_semantic": 0.0,
        "vocals_global_semantic": 0.0,
        "vocals_segment_semantic": 0.0,
        "instrumental_global_semantic": 0.0,
        "instrumental_segment_semantic": 0.0,
    }
    values.update(overrides)
    return SimilarityWeights(**values)


def test_similar_uses_global_and_segment_scores(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                global_semantic([1.0, 0.0]),
                segment_semantic(0, [1.0, 0.0]),
            ],
            "close": [
                global_semantic([0.95, 0.05]),
                segment_semantic(0, [0.9, 0.1]),
            ],
            "far": [
                global_semantic([0.0, 1.0]),
                segment_semantic(0, [0.0, 1.0]),
            ],
        },
    )

    similar = store.similar("selected", limit=2)

    assert similar[0]["id"] == "close"
    assert similar[0]["score"] > similar[1]["score"]
    assert similar[1]["id"] == "far"


def test_similar_uses_top3_segment_mean_instead_of_single_best_hit(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [segment_semantic(0, [1.0, 0.0])],
            "one_hit": [
                segment_semantic(0, [1.0, 0.0]),
                segment_semantic(1, [0.0, 1.0]),
                segment_semantic(2, [0.0, 1.0]),
            ],
            "consistent": [segment_semantic(0, [0.8, 0.6])],
        },
    )

    similar = store.similar("selected", limit=2)

    assert [track["id"] for track in similar] == ["consistent", "one_hit"]


def test_custom_weights_change_similarity_ranking(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                global_semantic([1.0, 0.0]),
                segment_semantic(0, [1.0, 0.0]),
            ],
            "semantic": [global_semantic([1.0, 0.0])],
            "segment": [
                global_semantic([0.0, 1.0]),
                segment_semantic(0, [1.0, 0.0]),
            ],
        },
    )

    semantic_first = store.similar(
        "selected",
        limit=2,
        weights=weights(global_semantic=1.0),
    )
    segment_first = store.similar(
        "selected",
        limit=2,
        weights=weights(segment_semantic=1.0),
    )

    assert semantic_first[0]["id"] == "semantic"
    assert segment_first[0]["id"] == "segment"


def test_similarity_matrix_uses_same_weighted_scores_as_similar(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                global_semantic([1.0, 0.0]),
                segment_semantic(0, [1.0, 0.0]),
            ],
            "semantic": [
                global_semantic([1.0, 0.0]),
                segment_semantic(0, [0.0, 1.0]),
            ],
            "segment": [
                global_semantic([0.0, 1.0]),
                segment_semantic(0, [1.0, 0.0]),
            ],
        },
    )

    semantic_matrix = store.similarity_matrix(
        weights=weights(global_semantic=1.0),
    )
    segment_matrix = store.similarity_matrix(
        weights=weights(segment_semantic=1.0),
    )

    assert semantic_matrix["selected"]["selected"] == 1.0
    assert semantic_matrix["selected"]["semantic"] == semantic_matrix["semantic"]["selected"]
    assert semantic_matrix["selected"]["semantic"] > semantic_matrix["selected"]["segment"]
    assert segment_matrix["selected"]["segment"] > segment_matrix["selected"]["semantic"]


def test_similar_segments_returns_best_segment_per_track(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                segment_semantic(0, [1.0, 0.0]),
                segment_semantic(1, [0.0, 1.0]),
            ],
            "early_match": [
                segment_semantic(0, [0.1, 0.9]),
                segment_semantic(1, [1.0, 0.0]),
            ],
            "late_match": [
                segment_semantic(0, [0.0, 1.0]),
                segment_semantic(1, [0.8, 0.6]),
            ],
        },
    )

    similar = store.similar_segments("selected", 1, limit=2)

    assert similar[0]["id"] == "late_match"
    assert similar[0]["segment_index"] == 0
    assert similar[0]["start_seconds"] == 0.0
    assert similar[1]["id"] == "early_match"
    assert similar[1]["segment_index"] == 0


def test_segment_counts_counts_stored_segment_embeddings(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                global_semantic([1.0, 0.0]),
                segment_semantic(0, [1.0, 0.0]),
                segment_semantic(1, [0.0, 1.0]),
            ],
            "empty": [global_semantic([0.0, 1.0])],
        },
    )

    assert store.segment_counts() == {"selected": 2, "empty": 0}


def test_stem_features_can_change_similarity_ranking(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                global_semantic([1.0, 0.0]),
                vocals_global([1.0, 0.0]),
                vocals_segment(0, [1.0, 0.0]),
                instrumental_global([0.0, 1.0]),
                instrumental_segment(0, [0.0, 1.0]),
            ],
            "vocal_cover": [
                global_semantic([0.0, 1.0]),
                vocals_global([1.0, 0.0]),
                vocals_segment(0, [1.0, 0.0]),
                instrumental_global([1.0, 0.0]),
                instrumental_segment(0, [1.0, 0.0]),
            ],
            "instrumental_cover": [
                global_semantic([0.0, 1.0]),
                vocals_global([0.0, 1.0]),
                vocals_segment(0, [0.0, 1.0]),
                instrumental_global([0.0, 1.0]),
                instrumental_segment(0, [0.0, 1.0]),
            ],
        },
    )

    vocal_first = store.similar(
        "selected",
        limit=2,
        weights=weights(vocals_global_semantic=1.0, vocals_segment_semantic=1.0),
    )
    instrumental_first = store.similar(
        "selected",
        limit=2,
        weights=weights(
            instrumental_global_semantic=1.0,
            instrumental_segment_semantic=1.0,
        ),
    )

    assert vocal_first[0]["id"] == "vocal_cover"
    assert instrumental_first[0]["id"] == "instrumental_cover"


def test_similar_segments_ignores_stem_segments(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                segment_semantic(0, [1.0, 0.0]),
                vocals_segment(0, [0.0, 1.0]),
            ],
            "whole_match": [segment_semantic(0, [1.0, 0.0])],
            "stem_only_match": [
                segment_semantic(0, [0.0, 1.0]),
                vocals_segment(0, [0.0, 1.0]),
            ],
        },
    )

    similar = store.similar_segments("selected", 0, limit=2)

    assert similar[0]["id"] == "whole_match"
