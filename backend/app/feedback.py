from collections.abc import Iterable

from app.reranker import (
    DEFAULT_RERANKER_COEFFICIENTS,
    PairEvidence,
    RerankerCoefficients,
    RerankerTrainingResult,
    learn_reranker_coefficients,
)
from app.vector_store import (
    DEFAULT_INSTRUMENTAL_GLOBAL_WEIGHT,
    DEFAULT_INSTRUMENTAL_SEGMENT_WEIGHT,
    DEFAULT_GLOBAL_WEIGHT,
    DEFAULT_SEGMENT_WEIGHT,
    DEFAULT_VOCALS_GLOBAL_WEIGHT,
    DEFAULT_VOCALS_SEGMENT_WEIGHT,
    SimilarityWeights,
)

DEFAULT_WEIGHTS = SimilarityWeights(
    global_semantic=DEFAULT_GLOBAL_WEIGHT,
    segment_semantic=DEFAULT_SEGMENT_WEIGHT,
    vocals_global_semantic=DEFAULT_VOCALS_GLOBAL_WEIGHT,
    vocals_segment_semantic=DEFAULT_VOCALS_SEGMENT_WEIGHT,
    instrumental_global_semantic=DEFAULT_INSTRUMENTAL_GLOBAL_WEIGHT,
    instrumental_segment_semantic=DEFAULT_INSTRUMENTAL_SEGMENT_WEIGHT,
)


def learn_feedback_reranker(
    examples: Iterable[tuple[PairEvidence, int]],
    *,
    default: RerankerCoefficients = DEFAULT_RERANKER_COEFFICIENTS,
) -> RerankerTrainingResult:
    return learn_reranker_coefficients(list(examples), default=default)
