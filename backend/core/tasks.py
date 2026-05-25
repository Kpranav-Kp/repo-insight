import os

from celery import shared_task
from django.conf import settings

from .models import ConversationSession, Repository
from .services.agents.graph import build_graph
from .services.graph_loader import clear_cache
from .services.recommender import RecommendationEngine


@shared_task
def analyze_repository_task(repo_id: int):
    try:
        repo = Repository.objects.get(id=repo_id)
    except Repository.DoesNotExist as e:
        raise ValueError("Repository not found") from e
    try:
        engine = RecommendationEngine(github_token=settings.GITHUB_TOKEN)
        result = engine.build_from_repository(repo.url)

        index_dir = os.path.join(settings.BASE_DIR, "indexes", str(repo_id))
        os.makedirs(index_dir, exist_ok=True)
        engine.save_index(index_dir)

        repo.status = "completed"
        repo.index_path = index_dir
        repo.issues_indexed = result["issues_indexed"]
        repo.prs_indexed = result["prs_indexed"]
        repo.skills_found = result["skills_found"]
        repo.save()
        clear_cache(repo_id)
        return result
    except Exception as e:
        repo.status = "failed"
        repo.error_message = str(e)
        repo.save()
        raise e


@shared_task
def run_chat_task(session_id, current_state):
    session = ConversationSession.objects.get(id=session_id)
    current_state["code_assist_count"] = session.code_assist_count
    current_state["stuck_counter"] = session.stuck_counter

    current_state["user_id"] = session.user.pk
    current_state["session_id"] = session.pk
    graph = build_graph()
    result = graph.invoke(current_state)

    session.state = result
    session.phase = result.get("conversation_phase", "onboarding")

    session.code_assist_count = result.get(
        "code_assist_count", session.code_assist_count
    )
    session.stuck_counter = result.get("stuck_counter", session.stuck_counter)
    session.save()

    return result
