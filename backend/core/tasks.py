from celery import shared_task
from .models import Repository
from .services.recommender import RecommendationEngine
from django.conf import settings

@shared_task
def analyze_repository_task(repo_id : int):
    try:
        repo = Repository.objects.get(id=repo_id)
    except Repository.DoesNotExist:
        raise ValueError("Repository not found")
    try:
        engine = RecommendationEngine(github_token=settings.GITHUB_TOKEN)
        result = engine.build_from_repository(repo.url)
        repo.status = 'completed'
        repo.issues_indexed = result['issues_indexed']
        repo.prs_indexed = result['prs_indexed']
        repo.skills_found = result['skills_found']
        repo.save()
        return result
    except Exception as e:
        repo.status = 'failed'
        repo.error_message = str(e)
        repo.save()
        raise e
