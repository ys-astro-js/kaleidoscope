import numpy as np

from app.cover_identity import (
    CoverIdentityFeature,
    _chunk_cqt,
    cover_identity_scores,
)


def test_cover_identity_scores_expose_global_segment_and_consistency() -> None:
    query = CoverIdentityFeature(
        global_embedding=[1.0, 0.0, 0.0],
        chunk_embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        chunk_start_seconds=[0.0, 15.0],
    )
    candidate = CoverIdentityFeature(
        global_embedding=[1.0, 0.0, 0.0],
        chunk_embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        chunk_start_seconds=[0.0, 15.0],
    )

    scores = cover_identity_scores(query, candidate)

    assert scores.global_score == 1.0
    assert scores.best_segment_score == 1.0
    assert scores.alignment_consistency == 1.0


def test_chunk_cqt_uses_official_context_window() -> None:
    cqt = np.ones((84, 760), dtype=np.float32)

    chunks, starts = _chunk_cqt(cqt)

    assert chunks.shape == (3, 84, 380)
    assert starts[0] == 0.0
    assert starts[-1] > starts[0]
