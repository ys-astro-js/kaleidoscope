from dataclasses import asdict, dataclass
from math import exp
from typing import Any

import numpy as np

from app.models import SimilarityMix

EVIDENCE_FIELDS = (
    "semantic_global",
    "semantic_segment_coverage",
    "instrumental_global",
    "instrumental_segment_coverage",
    "cover_global",
    "cover_best_segment",
    "cover_alignment_consistency",
    "cover_available",
)
MIN_RERANKER_TRAINING_EVENTS = 1
RERANKER_PRIOR_EVENT_STRENGTH = 6.0
RERANKER_L2 = 0.08
RERANKER_LEARNING_RATE = 0.25
RERANKER_STEPS = 500


@dataclass(frozen=True)
class PairEvidence:
    semantic_global: float | None = None
    semantic_segment_coverage: float | None = None
    instrumental_global: float | None = None
    instrumental_segment_coverage: float | None = None
    cover_global: float | None = None
    cover_best_segment: float | None = None
    cover_alignment_consistency: float | None = None
    cover_available: float = 0.0


@dataclass(frozen=True)
class RerankerCoefficients:
    bias: float = -1.0
    semantic_global: float = 3.0
    semantic_segment_coverage: float = 2.0
    instrumental_global: float = 1.8
    instrumental_segment_coverage: float = 1.2
    cover_global: float = 4.5
    cover_best_segment: float = 6.0
    cover_alignment_consistency: float = 6.0
    cover_available: float = -0.1


@dataclass(frozen=True)
class RerankerTrainingResult:
    coefficients: RerankerCoefficients
    event_count: int


DEFAULT_RERANKER_COEFFICIENTS = RerankerCoefficients()


def rerank_score(
    evidence: PairEvidence,
    mix: SimilarityMix,
    coefficients: RerankerCoefficients = DEFAULT_RERANKER_COEFFICIENTS,
) -> float | None:
    if not _has_score_evidence(evidence):
        return None

    features = _feature_vector(evidence, mix)
    values = coefficients_to_array(coefficients)
    logit = float(coefficients.bias + np.dot(features, values))
    return _sigmoid(logit)


def learn_reranker_coefficients(
    examples: list[tuple[PairEvidence, int]],
    *,
    default: RerankerCoefficients = DEFAULT_RERANKER_COEFFICIENTS,
) -> RerankerTrainingResult:
    if len(examples) < MIN_RERANKER_TRAINING_EVENTS:
        return RerankerTrainingResult(coefficients=default, event_count=len(examples))

    default_weights = coefficients_to_array(default)
    weights = default_weights.astype(np.float64, copy=True)
    bias = float(default.bias)
    x = np.asarray(
        [_feature_vector(evidence, _unit_mix()) for evidence, _ in examples],
        dtype=np.float64,
    )
    y = np.asarray([label for _, label in examples], dtype=np.float64)

    for _ in range(RERANKER_STEPS):
        logits = bias + x @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        errors = probabilities - y
        grad_weights = (x.T @ errors) / len(examples)
        grad_weights += RERANKER_L2 * (weights - default_weights)
        grad_bias = float(errors.mean() + RERANKER_L2 * (bias - default.bias))
        weights -= RERANKER_LEARNING_RATE * grad_weights
        bias -= RERANKER_LEARNING_RATE * grad_bias

    learned = coefficients_from_array(weights, bias=bias)
    return RerankerTrainingResult(
        coefficients=blend_coefficients(default, learned, len(examples)),
        event_count=len(examples),
    )


def coefficients_to_dict(coefficients: RerankerCoefficients) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(coefficients).items()}


def coefficients_from_mapping(mapping: dict[str, Any] | None) -> RerankerCoefficients:
    if not mapping:
        return DEFAULT_RERANKER_COEFFICIENTS
    defaults = coefficients_to_dict(DEFAULT_RERANKER_COEFFICIENTS)
    return RerankerCoefficients(
        **{
            key: float(mapping.get(key, default_value))
            for key, default_value in defaults.items()
        }
    )


def coefficients_to_array(coefficients: RerankerCoefficients) -> np.ndarray:
    return np.asarray(
        [getattr(coefficients, field_name) for field_name in EVIDENCE_FIELDS],
        dtype=np.float64,
    )


def coefficients_from_array(values: np.ndarray, *, bias: float) -> RerankerCoefficients:
    return RerankerCoefficients(
        bias=float(bias),
        **{
            field_name: float(values[index])
            for index, field_name in enumerate(EVIDENCE_FIELDS)
        },
    )


def blend_coefficients(
    default: RerankerCoefficients,
    learned: RerankerCoefficients,
    event_count: int,
) -> RerankerCoefficients:
    learned_strength = float(event_count)
    total_strength = learned_strength + RERANKER_PRIOR_EVENT_STRENGTH
    values = {
        key: (
            getattr(learned, key) * learned_strength
            + getattr(default, key) * RERANKER_PRIOR_EVENT_STRENGTH
        )
        / total_strength
        for key in coefficients_to_dict(default)
    }
    return RerankerCoefficients(**values)


def _feature_vector(evidence: PairEvidence, mix: SimilarityMix) -> np.ndarray:
    semantic_prior = max(0.0, mix.style)
    cover_prior = max(0.0, mix.cover)
    whole_prior = max(0.0, mix.whole)
    instrumental_prior = max(0.0, mix.instrumental)
    values = {
        "semantic_global": _value(evidence.semantic_global) * semantic_prior * whole_prior,
        "semantic_segment_coverage": (
            _value(evidence.semantic_segment_coverage) * semantic_prior * whole_prior
        ),
        "instrumental_global": (
            _value(evidence.instrumental_global) * semantic_prior * instrumental_prior
        ),
        "instrumental_segment_coverage": (
            _value(evidence.instrumental_segment_coverage) * semantic_prior * instrumental_prior
        ),
        "cover_global": _value(evidence.cover_global) * cover_prior,
        "cover_best_segment": _value(evidence.cover_best_segment) * cover_prior,
        "cover_alignment_consistency": (
            _value(evidence.cover_alignment_consistency) * cover_prior
        ),
        "cover_available": max(0.0, min(1.0, float(evidence.cover_available))),
    }
    return np.asarray([values[field_name] for field_name in EVIDENCE_FIELDS], dtype=np.float64)


def _has_score_evidence(evidence: PairEvidence) -> bool:
    return any(
        getattr(evidence, field_name) is not None
        for field_name in EVIDENCE_FIELDS
        if field_name != "cover_available"
    )


def _value(score: float | None) -> float:
    if score is None:
        return 0.0
    return max(0.0, min(1.0, float(score)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _unit_mix() -> SimilarityMix:
    return SimilarityMix(
        whole=1.0,
        vocals=0.0,
        instrumental=1.0,
        style=1.0,
        cover=1.0,
    )
