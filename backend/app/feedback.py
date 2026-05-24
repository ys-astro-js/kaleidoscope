from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import log
from typing import Any

from app.vector_store import (
    DEFAULT_INSTRUMENTAL_GLOBAL_WEIGHT,
    DEFAULT_INSTRUMENTAL_SEGMENT_WEIGHT,
    DEFAULT_GLOBAL_WEIGHT,
    DEFAULT_SEGMENT_WEIGHT,
    DEFAULT_VOCALS_GLOBAL_WEIGHT,
    DEFAULT_VOCALS_SEGMENT_WEIGHT,
    SimilarityFeatureScores,
    SimilarityWeights,
    VectorStore,
    WEIGHT_FIELDS,
    score_from_features,
)

MIN_TRAINING_EVENTS = 1
GRID_UNITS = 20
PRIOR_EVENT_STRENGTH = 2.0
EPSILON = 1e-6
DEFAULT_WEIGHTS = SimilarityWeights(
    global_semantic=DEFAULT_GLOBAL_WEIGHT,
    segment_semantic=DEFAULT_SEGMENT_WEIGHT,
    vocals_global_semantic=DEFAULT_VOCALS_GLOBAL_WEIGHT,
    vocals_segment_semantic=DEFAULT_VOCALS_SEGMENT_WEIGHT,
    instrumental_global_semantic=DEFAULT_INSTRUMENTAL_GLOBAL_WEIGHT,
    instrumental_segment_semantic=DEFAULT_INSTRUMENTAL_SEGMENT_WEIGHT,
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
        _candidate_weights(examples),
        key=lambda weights: _binary_cross_entropy(examples, weights),
    )
    return FeedbackTrainingResult(
        weights=_blend_with_default(best_weights, len(examples)),
        event_count=len(examples),
    )


def _candidate_weights(
    examples: list[tuple[SimilarityFeatureScores, int]],
) -> list[SimilarityWeights]:
    available = _available_features(examples)
    candidates = []
    seen: set[tuple[float, ...]] = set()
    available_fields = [
        field_name
        for field_name in WEIGHT_FIELDS
        if getattr(available, field_name) > 0.0
    ]
    if not available_fields:
        return [DEFAULT_WEIGHTS]

    for units_by_field in _weight_unit_allocations(available_fields):
        weights = _normalize_weights(
            SimilarityWeights(
                **{
                    field_name: units_by_field.get(field_name, 0) / GRID_UNITS
                    for field_name in WEIGHT_FIELDS
                }
            )
        )
        key = tuple(getattr(weights, field_name) for field_name in WEIGHT_FIELDS)
        if key in seen:
            continue
        candidates.append(weights)
        seen.add(key)
    return candidates


def _available_features(
    examples: list[tuple[SimilarityFeatureScores, int]],
) -> SimilarityWeights:
    return SimilarityWeights(
        **{
            field_name: (
                1.0
                if any(getattr(features, field_name) is not None for features, _ in examples)
                else 0.0
            )
            for field_name in WEIGHT_FIELDS
        }
    )


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
            **{
                field_name: (
                    getattr(weights, field_name) * learned_strength
                    + getattr(DEFAULT_WEIGHTS, field_name) * PRIOR_EVENT_STRENGTH
                )
                / total_strength
                for field_name in WEIGHT_FIELDS
            }
        )
    )


def _normalize_weights(weights: SimilarityWeights) -> SimilarityWeights:
    total = sum(getattr(weights, field_name) for field_name in WEIGHT_FIELDS)
    if total <= 0.0:
        return DEFAULT_WEIGHTS
    return SimilarityWeights(
        **{
            field_name: getattr(weights, field_name) / total
            for field_name in WEIGHT_FIELDS
        }
    )


def _weight_unit_allocations(fields: list[str]) -> list[dict[str, int]]:
    allocations: list[dict[str, int]] = []

    def visit(index: int, remaining: int, current: dict[str, int]) -> None:
        if index == len(fields) - 1:
            allocations.append({**current, fields[index]: remaining})
            return
        field_name = fields[index]
        for units in range(remaining + 1):
            visit(index + 1, remaining - units, {**current, field_name: units})

    visit(0, GRID_UNITS, {})
    return allocations
