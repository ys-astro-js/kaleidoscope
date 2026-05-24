from dataclasses import dataclass
from pathlib import Path

import numpy as np

IDENTITY_SAMPLE_RATE = 22_050
CHROMA_HOP_LENGTH = 2048
IDENTITY_FRAME_SECONDS = 2.0
MAX_IDENTITY_FRAMES = 240
MIN_ACTIVE_RMS_RATIO = 0.05
MIN_IDENTITY_FRAMES = 3
IDENTITY_SCORE_WEIGHT = 0.0
SEGMENT_IDENTITY_SCORE_WEIGHT = 0.0


@dataclass(frozen=True)
class IdentityFeature:
    chroma: list[list[float]]
    hop_seconds: float = IDENTITY_FRAME_SECONDS


@dataclass(frozen=True)
class IdentityAlignment:
    score: float
    candidate_frame_index: int | None = None


def extract_identity_feature(audio_path: Path) -> IdentityFeature | None:
    import librosa

    wav, sample_rate = librosa.load(
        audio_path,
        sr=IDENTITY_SAMPLE_RATE,
        mono=True,
    )
    if len(wav) < sample_rate:
        return None

    try:
        chroma = librosa.feature.chroma_cqt(
            y=wav,
            sr=sample_rate,
            hop_length=CHROMA_HOP_LENGTH,
            n_chroma=12,
        )
    except Exception:
        chroma = librosa.feature.chroma_stft(
            y=wav,
            sr=sample_rate,
            hop_length=CHROMA_HOP_LENGTH,
            n_chroma=12,
        )
    rms = librosa.feature.rms(
        y=wav,
        frame_length=CHROMA_HOP_LENGTH,
        hop_length=CHROMA_HOP_LENGTH,
    )[0]
    return identity_feature_from_chroma(
        chroma,
        rms,
        chroma_frame_seconds=CHROMA_HOP_LENGTH / sample_rate,
    )


def identity_feature_from_chroma(
    chroma: np.ndarray,
    rms: np.ndarray,
    *,
    chroma_frame_seconds: float,
    identity_frame_seconds: float = IDENTITY_FRAME_SECONDS,
) -> IdentityFeature | None:
    if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] == 0:
        return None

    frame_count = min(chroma.shape[1], len(rms))
    if frame_count == 0:
        return None

    frames_per_identity = max(1, int(round(identity_frame_seconds / chroma_frame_seconds)))
    active_threshold = float(np.max(rms[:frame_count])) * MIN_ACTIVE_RMS_RATIO
    vectors: list[np.ndarray] = []
    for start in range(0, frame_count, frames_per_identity):
        end = min(frame_count, start + frames_per_identity)
        if end <= start:
            continue
        if active_threshold > 0.0 and float(np.mean(rms[start:end])) < active_threshold:
            continue
        vector = np.mean(chroma[:, start:end], axis=1)
        normalized = _normalize_chroma_vector(vector)
        if normalized is not None:
            vectors.append(normalized)

    if len(vectors) > MAX_IDENTITY_FRAMES:
        indices = np.linspace(0, len(vectors) - 1, MAX_IDENTITY_FRAMES).round().astype(int)
        vectors = [vectors[index] for index in indices]

    if len(vectors) < MIN_IDENTITY_FRAMES:
        return None
    return IdentityFeature(
        chroma=[vector.astype(float).tolist() for vector in vectors],
        hop_seconds=identity_frame_seconds,
    )


def identity_similarity(
    query: IdentityFeature,
    candidate: IdentityFeature,
) -> float | None:
    forward = align_identity_features(query, candidate)
    reverse = align_identity_features(candidate, query)
    scores = [
        alignment.score
        for alignment in (forward, reverse)
        if alignment is not None
    ]
    if not scores:
        return None
    return max(scores)


def align_identity_features(
    query: IdentityFeature,
    candidate: IdentityFeature,
    *,
    query_frame_index: int | None = None,
) -> IdentityAlignment | None:
    query_array = _feature_array(query)
    candidate_array = _feature_array(candidate)
    if query_array is None or candidate_array is None:
        return None

    if query_frame_index is not None:
        query_frame_index = max(0, min(query_array.shape[0] - 1, query_frame_index))

    best_alignment: IdentityAlignment | None = None
    for shift in range(12):
        shifted_candidate = np.roll(candidate_array, shift=shift, axis=1)
        alignment = _dtw_alignment(
            query_array,
            shifted_candidate,
            query_frame_index=query_frame_index,
        )
        if alignment is None:
            continue
        if best_alignment is None or alignment.score > best_alignment.score:
            best_alignment = alignment
    return best_alignment


def combine_similarity_scores(
    style_score: float | None,
    identity_score: float | None,
    *,
    identity_weight: float = IDENTITY_SCORE_WEIGHT,
) -> float | None:
    weight = max(0.0, min(1.0, identity_weight))
    if weight == 0.0:
        return style_score
    if style_score is None:
        return identity_score
    if identity_score is None:
        return style_score
    return _clamp_score(identity_score * weight + style_score * (1.0 - weight))


def segment_index_to_identity_frame(
    segment_index: int,
    *,
    identity_hop_seconds: float,
    segment_hop_seconds: float,
) -> int:
    segment_center_seconds = segment_index * segment_hop_seconds + segment_hop_seconds / 2.0
    return max(0, int(round(segment_center_seconds / identity_hop_seconds)))


def identity_frame_to_segment_index(
    frame_index: int,
    *,
    identity_hop_seconds: float,
    segment_hop_seconds: float,
) -> int:
    frame_seconds = frame_index * identity_hop_seconds
    segment_start_seconds = max(0.0, frame_seconds - segment_hop_seconds / 2.0)
    return max(0, int(round(segment_start_seconds / segment_hop_seconds)))


def _dtw_alignment(
    query: np.ndarray,
    candidate: np.ndarray,
    *,
    query_frame_index: int | None,
) -> IdentityAlignment | None:
    import librosa

    try:
        accumulated_cost, path = librosa.sequence.dtw(
            X=query.T,
            Y=candidate.T,
            metric="cosine",
            subseq=True,
            backtrack=True,
        )
    except Exception:
        return None

    if len(path) == 0:
        return None
    best_end = int(np.argmin(accumulated_cost[-1]))
    best_cost = float(accumulated_cost[-1, best_end])
    if not np.isfinite(best_cost):
        return None

    mean_cost = best_cost / max(1, len(path))
    score = _clamp_score(1.0 - mean_cost)
    candidate_frame_index = None
    if query_frame_index is not None:
        candidate_frame_index = _candidate_frame_for_query(path, query_frame_index)
    return IdentityAlignment(score=score, candidate_frame_index=candidate_frame_index)


def _candidate_frame_for_query(path: np.ndarray, query_frame_index: int) -> int | None:
    pairs = [(int(query_index), int(candidate_index)) for query_index, candidate_index in path]
    if not pairs:
        return None
    query_index, candidate_index = min(
        pairs,
        key=lambda pair: abs(pair[0] - query_frame_index),
    )
    if abs(query_index - query_frame_index) > 1:
        return None
    return candidate_index


def _feature_array(feature: IdentityFeature) -> np.ndarray | None:
    array = np.asarray(feature.chroma, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 12 or array.shape[0] < MIN_IDENTITY_FRAMES:
        return None
    normalized = [_normalize_chroma_vector(row) for row in array]
    vectors = [vector for vector in normalized if vector is not None]
    if len(vectors) < MIN_IDENTITY_FRAMES:
        return None
    return np.asarray(vectors, dtype=np.float32)


def _normalize_chroma_vector(vector: np.ndarray) -> np.ndarray | None:
    vector = np.maximum(np.asarray(vector, dtype=np.float32), 0.0)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return None
    return vector / norm


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))
