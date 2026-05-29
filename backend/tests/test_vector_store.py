from pathlib import Path

from app.vector_store import (
    GLOBAL_SEMANTIC_KIND,
    SEGMENT_SEMANTIC_KIND,
    VOCALS_SEGMENT_SEMANTIC_KIND,
    EmbeddingRecord,
    VectorStore,
)


def global_semantic(vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=GLOBAL_SEMANTIC_KIND, segment_index=-1, vector=vector)


def segment_semantic(index: int, vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=SEGMENT_SEMANTIC_KIND, segment_index=index, vector=vector)


def vocals_segment(index: int, vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(kind=VOCALS_SEGMENT_SEMANTIC_KIND, segment_index=index, vector=vector)


def test_upsert_handles_vector_dimension_changes_with_lancedb(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "lancedb")

    store.upsert("old-model", [1.0, 0.0, 0.0], [[1.0, 0.0, 0.0]])
    store.upsert("new-model", [1.0, 0.0], [[1.0, 0.0]])
    store.upsert("old-model", [0.0, 1.0], [[0.0, 1.0]])

    embeddings = store.all_embeddings()

    assert set(embeddings) == {"old-model", "new-model"}
    assert {
        len(record.vector)
        for record in embeddings["old-model"]
    } == {2}
    assert {
        len(record.vector)
        for record in embeddings["new-model"]
    } == {2}


def test_pair_evidence_exposes_segment_coverage_instead_of_single_peak(monkeypatch) -> None:
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
            "one_hit": [
                global_semantic([1.0, 0.0]),
                segment_semantic(0, [1.0, 0.0]),
                segment_semantic(1, [1.0, 0.0]),
            ],
            "covered": [
                global_semantic([1.0, 0.0]),
                segment_semantic(0, [1.0, 0.0]),
                segment_semantic(1, [0.0, 1.0]),
            ],
        },
    )

    one_hit = store.pair_evidence("selected", "one_hit")
    covered = store.pair_evidence("selected", "covered")

    assert one_hit is not None
    assert covered is not None
    assert covered.semantic_segment_coverage == 1.0
    assert covered.semantic_segment_coverage > one_hit.semantic_segment_coverage


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


def test_similar_segments_prefers_context_consistency_over_single_peak(monkeypatch) -> None:
    store = VectorStore(Path("unused"))
    monkeypatch.setattr(
        store,
        "all_embeddings",
        lambda: {
            "selected": [
                segment_semantic(0, [1.0, 0.0]),
                segment_semantic(1, [0.0, 1.0]),
                segment_semantic(2, [1.0, 0.0]),
            ],
            "single_peak": [
                segment_semantic(0, [1.0, 0.0]),
                segment_semantic(1, [0.0, 1.0]),
                segment_semantic(2, [0.0, 1.0]),
            ],
            "consistent": [
                segment_semantic(0, [0.9, 0.1]),
                segment_semantic(1, [0.0, 0.95]),
                segment_semantic(2, [0.9, 0.1]),
            ],
        },
    )

    similar = store.similar_segments("selected", 1, limit=2)

    assert similar[0]["id"] == "consistent"
    assert similar[0]["context_score"] > similar[1]["context_score"]


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
