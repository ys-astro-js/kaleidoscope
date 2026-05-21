from pathlib import Path

from app.vector_store import (
    COVER_CHROMA_KIND,
    GLOBAL_SEMANTIC_KIND,
    SEGMENT_SEMANTIC_KIND,
    EmbeddingRecord,
    SimilarityWeights,
    VectorStore,
)


def global_semantic(vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=GLOBAL_SEMANTIC_KIND, segment_index=-1, vector=vector)


def segment_semantic(index: int, vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=SEGMENT_SEMANTIC_KIND, segment_index=index, vector=vector)


def cover_chroma(vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=COVER_CHROMA_KIND, segment_index=-1, vector=vector)


def test_similar_combines_global_segment_and_chroma_scores(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    chroma = [1.0, 1.0, *([0.0] * 10)]
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                global_semantic([1.0, 0.0]),
                segment_semantic(0, [1.0, 0.0]),
                cover_chroma(chroma),
            ],
            "close": [
                global_semantic([0.95, 0.05]),
                segment_semantic(0, [0.9, 0.1]),
                cover_chroma([0.0, 0.0, 1.0, 1.0, *([0.0] * 8)]),
            ],
            "far": [
                global_semantic([0.0, 1.0]),
                segment_semantic(0, [0.0, 1.0]),
                cover_chroma([1.0] * 12),
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


def test_chroma_cover_score_can_lift_cover_candidate(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    selected_chroma = [1.0, 1.0, *([0.0] * 10)]
    shifted_cover_chroma = [0.0, 0.0, 0.0, 1.0, 1.0, *([0.0] * 7)]
    different_chroma = [1.0, 0.0, 1.0, *([0.0] * 9)]
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                global_semantic([1.0, 0.0]),
                segment_semantic(0, [1.0, 0.0]),
                cover_chroma(selected_chroma),
            ],
            "cover": [
                global_semantic([0.8, 0.6]),
                segment_semantic(0, [0.8, 0.6]),
                cover_chroma(shifted_cover_chroma),
            ],
            "same_genre": [
                global_semantic([0.9, 0.435]),
                segment_semantic(0, [0.9, 0.435]),
                cover_chroma(different_chroma),
            ],
        },
    )

    similar = store.similar("selected", limit=2)

    assert [track["id"] for track in similar] == ["cover", "same_genre"]


def test_custom_weights_change_similarity_ranking(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    selected_chroma = [1.0, 1.0, *([0.0] * 10)]
    matching_chroma = [0.0, 0.0, 1.0, 1.0, *([0.0] * 8)]
    different_chroma = [1.0, 0.0, 1.0, *([0.0] * 9)]
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                global_semantic([1.0, 0.0]),
                cover_chroma(selected_chroma),
            ],
            "semantic": [
                global_semantic([1.0, 0.0]),
                cover_chroma(different_chroma),
            ],
            "cover": [
                global_semantic([0.0, 1.0]),
                cover_chroma(matching_chroma),
            ],
        },
    )

    semantic_first = store.similar(
        "selected",
        limit=2,
        weights=SimilarityWeights(global_semantic=1.0, segment_semantic=0.0, cover_chroma=0.0),
    )
    chroma_first = store.similar(
        "selected",
        limit=2,
        weights=SimilarityWeights(global_semantic=0.0, segment_semantic=0.0, cover_chroma=1.0),
    )

    assert semantic_first[0]["id"] == "semantic"
    assert chroma_first[0]["id"] == "cover"


def test_similarity_matrix_uses_same_weighted_scores_as_similar(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    selected_chroma = [1.0, 1.0, *([0.0] * 10)]
    matching_chroma = [0.0, 0.0, 1.0, 1.0, *([0.0] * 8)]
    different_chroma = [1.0, 0.0, 1.0, *([0.0] * 9)]
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                global_semantic([1.0, 0.0]),
                cover_chroma(selected_chroma),
            ],
            "semantic": [
                global_semantic([1.0, 0.0]),
                cover_chroma(different_chroma),
            ],
            "cover": [
                global_semantic([0.0, 1.0]),
                cover_chroma(matching_chroma),
            ],
        },
    )

    semantic_matrix = store.similarity_matrix(
        weights=SimilarityWeights(global_semantic=1.0, segment_semantic=0.0, cover_chroma=0.0),
    )
    chroma_matrix = store.similarity_matrix(
        weights=SimilarityWeights(global_semantic=0.0, segment_semantic=0.0, cover_chroma=1.0),
    )

    assert semantic_matrix["selected"]["selected"] == 1.0
    assert semantic_matrix["selected"]["semantic"] == semantic_matrix["semantic"]["selected"]
    assert semantic_matrix["selected"]["semantic"] > semantic_matrix["selected"]["cover"]
    assert chroma_matrix["selected"]["cover"] > chroma_matrix["selected"]["semantic"]


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
