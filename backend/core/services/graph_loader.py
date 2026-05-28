# backend/core/services/graph_loader.py

import logging

from django.conf import settings

from ..models import Repository
from .recommender import RecommendationEngine

_engine_cache = {}

logger = logging.getLogger(__name__)


def load_engine_for_repo(repo_id: int) -> RecommendationEngine:
    global _engine_cache

    if repo_id in _engine_cache:
        return _engine_cache[repo_id]

    repo = Repository.objects.get(id=repo_id)

    if repo.status != "completed" or not repo.index_path:
        raise RuntimeError(
            f"Repository {repo_id} is not ready. status={repo.status}, index_path={repo.index_path}"
        )

    model_path = getattr(settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    engine = RecommendationEngine()
    engine.graph.skills._service.model_name = model_path
    engine.graph.issues._service.model_name = model_path
    engine.graph.prs._service.model_name = model_path
    engine.load_index(repo.index_path)

    _engine_cache[repo_id] = engine
    logger.info(f"Loaded engine for repo {repo_id}")

    return engine


def clear_cache(repo_id: int = 0) -> None:
    """Call this if repo is re-analyzed"""
    global _engine_cache
    if repo_id:
        _engine_cache.pop(repo_id, None)
    else:
        _engine_cache.clear()
