import numpy as np

from app.identity import (
    IdentityFeature,
    align_identity_features,
    combine_similarity_scores,
    identity_feature_from_chroma,
    identity_frame_to_segment_index,
    identity_similarity,
    segment_index_to_identity_frame,
)


def chroma_sequence(pitches: list[int]) -> list[list[float]]:
    rows = []
    for pitch in pitches:
        vector = [0.0] * 12
        vector[pitch] = 1.0
        rows.append(vector)
    return rows


def test_identity_feature_from_chroma_downsamples_active_frames() -> None:
    chroma = np.zeros((12, 8), dtype=np.float32)
    chroma[0, :2] = 1.0
    chroma[4, 2:4] = 1.0
    chroma[7, 4:6] = 1.0
    chroma[11, 6:8] = 1.0
    rms = np.ones(8, dtype=np.float32)

    feature = identity_feature_from_chroma(
        chroma,
        rms,
        chroma_frame_seconds=1.0,
        identity_frame_seconds=2.0,
    )

    assert feature is not None
    assert feature.hop_seconds == 2.0
    assert np.argmax(feature.chroma, axis=1).tolist() == [0, 4, 7, 11]


def test_identity_similarity_is_key_shift_invariant() -> None:
    query = IdentityFeature(chroma=chroma_sequence([0, 4, 7, 2, 5]), hop_seconds=2.0)
    shifted = IdentityFeature(chroma=chroma_sequence([3, 7, 10, 5, 8]), hop_seconds=2.0)

    score = identity_similarity(query, shifted)

    assert score is not None
    assert score > 0.99


def test_combining_similarity_can_disable_identity_score() -> None:
    score = combine_similarity_scores(0.42, 0.99, identity_weight=0.0)

    assert score == 0.42


def test_alignment_maps_query_frame_to_candidate_frame() -> None:
    query = IdentityFeature(chroma=chroma_sequence([0, 4, 7, 2]), hop_seconds=2.0)
    candidate = IdentityFeature(chroma=chroma_sequence([9, 0, 4, 7, 2, 11]), hop_seconds=2.0)

    alignment = align_identity_features(query, candidate, query_frame_index=2)

    assert alignment is not None
    assert alignment.score > 0.99
    assert alignment.candidate_frame_index == 3


def test_identity_frame_and_segment_index_conversion_uses_segment_center() -> None:
    frame_index = segment_index_to_identity_frame(
        2,
        identity_hop_seconds=2.0,
        segment_hop_seconds=15.0,
    )
    segment_index = identity_frame_to_segment_index(
        frame_index,
        identity_hop_seconds=2.0,
        segment_hop_seconds=15.0,
    )

    assert frame_index == 19
    assert segment_index == 2
