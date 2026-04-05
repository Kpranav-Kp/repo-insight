# core/services/graph_loader.py
from django.conf import settings

from ..models import Repository
from .recommender import RecommendationEngine


def load_engine_for_repo(repo_id: int) -> RecommendationEngine:
    repo = Repository.objects.get(id=repo_id)

    if repo.status != "completed" or not repo.index_path:
        raise RuntimeError(f"Repository {repo_id} is not ready.")

    model_path = getattr(settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    engine = RecommendationEngine()
    engine.graph.skills._service.model_name = model_path
    engine.graph.issues._service.model_name = model_path
    engine.graph.prs._service.model_name = model_path
    engine.load_index(repo.index_path)
    return engine
