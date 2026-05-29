from app.feedback import learn_feedback_reranker
from app.models import SimilarityMix
from app.reranker import DEFAULT_RERANKER_COEFFICIENTS, PairEvidence, rerank_score


def test_feedback_reranker_learns_from_pair_evidence() -> None:
    positive = PairEvidence(semantic_global=0.2, cover_alignment_consistency=1.0)
    negative = PairEvidence(semantic_global=1.0, cover_alignment_consistency=0.0)

    result = learn_feedback_reranker(
        [
            (positive, 1),
            (negative, 0),
        ],
    )
    mix = SimilarityMix(whole=1.0, vocals=0.0, instrumental=0.0, style=1.0, cover=1.0)

    assert result.event_count == 2
    assert rerank_score(positive, mix, result.coefficients) > rerank_score(
        negative,
        mix,
        DEFAULT_RERANKER_COEFFICIENTS,
    )
