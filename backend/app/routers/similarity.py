from fastapi import APIRouter

from app.dependencies import AppContextDep
from app.models import SimilarityMix, SimilarityMixRequest

router = APIRouter(prefix="/api/similarity", tags=["similarity"])


@router.get("/mix", response_model=SimilarityMix)
def get_similarity_mix(context: AppContextDep) -> SimilarityMix:
    return context.service.similarity_mix()


@router.put("/mix", response_model=SimilarityMix)
def update_similarity_mix(mix: SimilarityMixRequest, context: AppContextDep) -> SimilarityMix:
    return context.service.set_similarity_mix(
        whole=mix.whole,
        instrumental=mix.instrumental,
        style=mix.style,
        cover=mix.cover,
    )
