import numpy as np

from app.cover_identity import (
    CoverIdentityFeature,
    _chunk_cqt,
    combine_cover_similarity_scores,
    cover_identity_similarity,
)


def test_cover_identity_similarity_uses_global_and_chunk_embeddings() -> None:
    query = CoverIdentityFeature(
        global_embedding=[1.0, 0.0, 0.0],
        chunk_embeddings=[[0.0, 1.0, 0.0]],
        chunk_start_seconds=[0.0],
    )
    candidate = CoverIdentityFeature(
        global_embedding=[0.0, 1.0, 0.0],
        chunk_embeddings=[[0.0, 1.0, 0.0]],
        chunk_start_seconds=[0.0],
    )

    score = cover_identity_similarity(query, candidate)

    assert score == 0.35


def test_cover_score_blend_can_lift_cover_identity_match() -> None:
    score = combine_cover_similarity_scores(0.6, 1.0, cover_weight=0.35)

    assert score == 0.74


def test_chunk_cqt_uses_official_context_window() -> None:
    cqt = np.ones((84, 760), dtype=np.float32)

    chunks, starts = _chunk_cqt(cqt)

    assert chunks.shape == (3, 84, 380)
    assert starts[0] == 0.0
    assert starts[-1] > starts[0]
