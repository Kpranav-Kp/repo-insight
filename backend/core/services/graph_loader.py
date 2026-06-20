import logging
from collections import OrderedDict

from django.conf import settings

from ..models import Repository
from .recommender import RecommendationEngine

_cache: OrderedDict[int, RecommendationEngine] = OrderedDict()
_MAX_CACHE = 20

logger = logging.getLogger(__name__)


def load_engine_for_repo(repo_id: int) -> RecommendationEngine:
    global _cache

    if repo_id in _cache:
        _cache.move_to_end(repo_id)
        return _cache[repo_id]

    repo = Repository.objects.get(id=repo_id)

    if repo.status != "completed" or not repo.index_path:
        raise RuntimeError(
            f"Repository {repo_id} is not ready. status={repo.status}, index_path={repo.index_path}"
        )

    engine = RecommendationEngine()
    engine.graph.skills._service.model_name = getattr(
        settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
    )
    engine.graph.issues._service.model_name = getattr(
        settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
    )
    engine.graph.prs._service.model_name = getattr(
        settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
    )

    engine.load_index(repo.index_path)

    _cache[repo_id] = engine
    _cache.move_to_end(repo_id)
    if len(_cache) > _MAX_CACHE:
        _cache.popitem(last=False)

    logger.info("Loaded engine for repo %d (cache size %d)", repo_id, len(_cache))

    return engine


def clear_cache(repo_id: int = 0) -> None:
    global _cache
    if repo_id:
        _cache.pop(repo_id, None)
    else:
        _cache.clear()
