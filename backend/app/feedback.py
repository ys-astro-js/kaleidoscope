from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import log
from typing import Any

from app.vector_store import (
    DEFAULT_CHROMA_WEIGHT,
    DEFAULT_GLOBAL_WEIGHT,
    DEFAULT_SEGMENT_WEIGHT,
    SimilarityFeatureScores,
    SimilarityWeights,
    VectorStore,
    score_from_features,
)

MIN_TRAINING_EVENTS = 4
GRID_UNITS = 10
PRIOR_EVENT_STRENGTH = 6.0
EPSILON = 1e-6
DEFAULT_WEIGHTS = SimilarityWeights(
    global_semantic=DEFAULT_GLOBAL_WEIGHT,
    segment_semantic=DEFAULT_SEGMENT_WEIGHT,
    cover_chroma=DEFAULT_CHROMA_WEIGHT,
)


@dataclass(frozen=True)
class FeedbackTrainingResult:
    weights: SimilarityWeights
    event_count: int


def learn_feedback_weights(
    vectors: VectorStore,
    events: Iterable[Mapping[str, Any]],
) -> FeedbackTrainingResult:
    embeddings = vectors.all_embeddings()
    examples: list[tuple[SimilarityFeatureScores, int]] = []

    for event in events:
        features = vectors.feature_scores(
            str(event["query_track_id"]),
            str(event["candidate_track_id"]),
            embeddings=embeddings,
        )
        if features is None or score_from_features(features, DEFAULT_WEIGHTS) is None:
            continue
        label = 1 if event["label"] == "similar" else 0
        examples.append((features, label))

    if len(examples) < MIN_TRAINING_EVENTS:
        return FeedbackTrainingResult(weights=DEFAULT_WEIGHTS, event_count=len(examples))

    best_weights = min(
        _candidate_weights(),
        key=lambda weights: _binary_cross_entropy(examples, weights),
    )
    return FeedbackTrainingResult(
        weights=_blend_with_default(best_weights, len(examples)),
        event_count=len(examples),
    )


def _candidate_weights() -> list[SimilarityWeights]:
    candidates = []
    for global_units in range(GRID_UNITS + 1):
        remaining = GRID_UNITS - global_units
        for segment_units in range(remaining + 1):
            chroma_units = remaining - segment_units
            candidates.append(
                SimilarityWeights(
                    global_semantic=global_units / GRID_UNITS,
                    segment_semantic=segment_units / GRID_UNITS,
                    cover_chroma=chroma_units / GRID_UNITS,
                )
            )
    return candidates


def _binary_cross_entropy(
    examples: list[tuple[SimilarityFeatureScores, int]],
    weights: SimilarityWeights,
) -> float:
    total = 0.0
    for features, label in examples:
        probability = score_from_features(features, weights)
        if probability is None:
            probability = 0.5
        probability = min(1.0 - EPSILON, max(EPSILON, probability))
        total += -(label * log(probability) + (1 - label) * log(1.0 - probability))
    return total / len(examples)


def _blend_with_default(
    weights: SimilarityWeights,
    event_count: int,
) -> SimilarityWeights:
    learned_strength = float(event_count)
    total_strength = learned_strength + PRIOR_EVENT_STRENGTH
    return _normalize_weights(
        SimilarityWeights(
            global_semantic=(
                weights.global_semantic * learned_strength
                + DEFAULT_WEIGHTS.global_semantic * PRIOR_EVENT_STRENGTH
            )
            / total_strength,
            segment_semantic=(
                weights.segment_semantic * learned_strength
                + DEFAULT_WEIGHTS.segment_semantic * PRIOR_EVENT_STRENGTH
            )
            / total_strength,
            cover_chroma=(
                weights.cover_chroma * learned_strength
                + DEFAULT_WEIGHTS.cover_chroma * PRIOR_EVENT_STRENGTH
            )
            / total_strength,
        )
    )


def _normalize_weights(weights: SimilarityWeights) -> SimilarityWeights:
    total = weights.global_semantic + weights.segment_semantic + weights.cover_chroma
    if total <= 0.0:
        return DEFAULT_WEIGHTS
    return SimilarityWeights(
        global_semantic=weights.global_semantic / total,
        segment_semantic=weights.segment_semantic / total,
        cover_chroma=weights.cover_chroma / total,
    )
