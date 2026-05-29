import json
import logging
import gc
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CLEWS_MODEL_KEY = "clews"
CLEWS_SEGMENT_HOP_SECONDS = 5.0
PRECOMPUTED_FEATURE_DIR = "features"
SEGMENT_CONSISTENCY_MATCHES = 6
LOGGER = logging.getLogger(__name__)
_MISSING_ASSETS_WARNING_EMITTED = False
_LOAD_WARNING_EMITTED = False


@dataclass(frozen=True)
class CoverAlignmentFeature:
    model_key: str
    global_embedding: list[float]
    segment_embeddings: list[list[float]]
    segment_start_seconds: list[float]


@dataclass(frozen=True)
class CoverAlignmentScores:
    global_score: float | None = None
    best_segment_score: float | None = None
    alignment_consistency: float | None = None


def extract_cover_alignment_feature(
    audio_path: Path,
    model_dir: Path,
) -> CoverAlignmentFeature | None:
    precomputed = _load_precomputed_feature(audio_path, model_dir)
    if precomputed is not None:
        return precomputed

    from app.clews_inference import find_clews_assets

    assets = find_clews_assets(model_dir)
    if assets is None:
        _warn_missing_assets(model_dir)
        return None

    embedder = _optional_clews_embedder(str(assets.checkpoint_path.resolve()))
    if embedder is None:
        return None
    try:
        return embedder.embed_file(audio_path)
    except Exception as exc:
        _warn_load_failed(exc)
        return None


def cover_alignment_scores(
    query: CoverAlignmentFeature,
    candidate: CoverAlignmentFeature,
) -> CoverAlignmentScores:
    return CoverAlignmentScores(
        global_score=_cosine_score(query.global_embedding, candidate.global_embedding),
        best_segment_score=_top_segment_score(
            query.segment_embeddings,
            candidate.segment_embeddings,
        ),
        alignment_consistency=_segment_consistency_score(
            query.segment_embeddings,
            candidate.segment_embeddings,
        ),
    )


def cover_alignment_segment_score(
    query: CoverAlignmentFeature,
    candidate: CoverAlignmentFeature,
    *,
    query_segment_index: int,
    candidate_segment_index: int,
    query_start_seconds: float | None = None,
    candidate_start_seconds: float | None = None,
) -> float | None:
    if query_start_seconds is not None:
        query_segment_index = _nearest_segment_index(query.segment_start_seconds, query_start_seconds)
    if candidate_start_seconds is not None:
        candidate_segment_index = _nearest_segment_index(
            candidate.segment_start_seconds,
            candidate_start_seconds,
        )
    query_segment = _segment_at(query.segment_embeddings, query_segment_index)
    candidate_segment = _segment_at(candidate.segment_embeddings, candidate_segment_index)
    if query_segment is None or candidate_segment is None:
        return None

    local_score = _cosine_score(query_segment, candidate_segment)
    if local_score is None:
        return None

    context_scores = [local_score]
    for offset in (-1, 1):
        query_context = _segment_at(query.segment_embeddings, query_segment_index + offset)
        candidate_context = _segment_at(
            candidate.segment_embeddings,
            candidate_segment_index + offset,
        )
        if query_context is None or candidate_context is None:
            continue
        score = _cosine_score(query_context, candidate_context)
        if score is not None:
            context_scores.append(score)
    return _clamp_score(float(np.mean(context_scores)))


def _load_precomputed_feature(
    audio_path: Path,
    model_dir: Path,
) -> CoverAlignmentFeature | None:
    candidates = (
        model_dir / PRECOMPUTED_FEATURE_DIR / f"{audio_path.stem}.json",
        model_dir / f"{audio_path.stem}.clews.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return CoverAlignmentFeature(
            model_key=str(payload.get("model_key") or CLEWS_MODEL_KEY),
            global_embedding=[float(value) for value in payload["global_embedding"]],
            segment_embeddings=[
                [float(value) for value in embedding]
                for embedding in payload["segment_embeddings"]
            ],
            segment_start_seconds=[
                float(value)
                for value in payload.get(
                    "segment_start_seconds",
                    [
                        index * CLEWS_SEGMENT_HOP_SECONDS
                        for index in range(len(payload["segment_embeddings"]))
                    ],
                )
            ],
        )
    return None


@lru_cache(maxsize=2)
def _optional_clews_embedder(checkpoint_path: str):
    try:
        from app.clews_inference import ClewsEmbedder
    except Exception as exc:
        _warn_load_failed(exc)
        return None
    return ClewsEmbedder(Path(checkpoint_path))


def release_cover_alignment_resources() -> None:
    _optional_clews_embedder.cache_clear()
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _warn_missing_assets(model_dir: Path) -> None:
    global _MISSING_ASSETS_WARNING_EMITTED
    if _MISSING_ASSETS_WARNING_EMITTED:
        return
    _MISSING_ASSETS_WARNING_EMITTED = True
    LOGGER.warning(
        "CLEWS assets are not ready under %s; cover alignment will fall back to Discogs-VINet "
        "when available. Run scripts/prepare_clews_assets.py to install checkpoint and source.",
        model_dir,
    )


def _warn_load_failed(exc: Exception) -> None:
    global _LOAD_WARNING_EMITTED
    if _LOAD_WARNING_EMITTED:
        return
    _LOAD_WARNING_EMITTED = True
    LOGGER.warning("CLEWS inference is unavailable: %s", exc)


def _top_segment_score(
    query_embeddings: list[list[float]],
    candidate_embeddings: list[list[float]],
) -> float | None:
    best_score = None
    for scores, _, _ in _segment_score_blocks(query_embeddings, candidate_embeddings):
        block_best = float(np.max(scores))
        best_score = block_best if best_score is None else max(best_score, block_best)
    if best_score is None:
        return None
    return best_score


def _segment_consistency_score(
    query_embeddings: list[list[float]],
    candidate_embeddings: list[list[float]],
) -> float | None:
    if not query_embeddings or not candidate_embeddings:
        return None

    target_matches = min(
        len(query_embeddings),
        len(candidate_embeddings),
        SEGMENT_CONSISTENCY_MATCHES,
    )
    coverage_target = min(
        max(len(query_embeddings), len(candidate_embeddings)),
        SEGMENT_CONSISTENCY_MATCHES,
    )
    scored_pairs: list[tuple[float, int, int]] = []
    for scores, query_indices, candidate_indices in _segment_score_blocks(
        query_embeddings,
        candidate_embeddings,
    ):
        for flat_index in np.argsort(scores, axis=None)[::-1]:
            query_pos, candidate_pos = np.unravel_index(flat_index, scores.shape)
            scored_pairs.append(
                (
                    float(scores[query_pos, candidate_pos]),
                    query_indices[query_pos],
                    candidate_indices[candidate_pos],
                )
            )
    scored_pairs.sort(reverse=True)
    used_queries = set()
    used_candidates = set()
    selected_scores: list[float] = []
    for score, query_index, candidate_index in scored_pairs:
        if query_index in used_queries or candidate_index in used_candidates:
            continue
        selected_scores.append(score)
        used_queries.add(query_index)
        used_candidates.add(candidate_index)
        if len(selected_scores) >= target_matches:
            break

    if not selected_scores:
        return None
    coverage = len(selected_scores) / coverage_target
    return _clamp_score(float(np.mean(selected_scores)) * coverage)


def _segment_score_blocks(
    query_embeddings: list[list[float]],
    candidate_embeddings: list[list[float]],
):
    query_groups = _normalized_embedding_groups(query_embeddings)
    candidate_groups = _normalized_embedding_groups(candidate_embeddings)
    for dimension, (query_matrix, query_indices) in query_groups.items():
        candidate_group = candidate_groups.get(dimension)
        if candidate_group is None:
            continue
        candidate_matrix, candidate_indices = candidate_group
        scores = np.clip(query_matrix @ candidate_matrix.T, 0.0, 1.0)
        yield scores, query_indices, candidate_indices


def _normalized_embedding_groups(
    embeddings: list[list[float]],
) -> dict[int, tuple[np.ndarray, list[int]]]:
    grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
    for index, embedding in enumerate(embeddings):
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            continue
        norm = float(np.linalg.norm(array))
        if norm <= 0.0:
            continue
        grouped.setdefault(array.shape[0], []).append((index, array / norm))

    return {
        dimension: (
            np.stack([array for _, array in values]),
            [index for index, _ in values],
        )
        for dimension, values in grouped.items()
    }


def _segment_at(segments: list[list[float]], segment_index: int) -> list[float] | None:
    if segment_index < 0 or segment_index >= len(segments):
        return None
    return segments[segment_index]


def _nearest_segment_index(segment_start_seconds: list[float], start_seconds: float) -> int:
    if not segment_start_seconds:
        return -1
    return min(
        range(len(segment_start_seconds)),
        key=lambda index: abs(segment_start_seconds[index] - start_seconds),
    )


def _cosine_score(query: list[float], candidate: list[float]) -> float | None:
    query_array = _as_normalized_array(query)
    candidate_array = _as_normalized_array(candidate)
    if query_array is None or candidate_array is None:
        return None
    if len(query_array) != len(candidate_array):
        return None
    return _clamp_score(float(np.dot(query_array, candidate_array)))


def _as_normalized_array(vector: list[float]) -> np.ndarray | None:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return None
    return array / norm


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))
