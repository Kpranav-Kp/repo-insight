# backend/core/services/agents/views.py
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ConversationSession, LearnerProfile, Recommendation, Repository
from .serializers import (
    ChatMessageSerializer,
    ConversationSessionSerializer,
    LearnerProfileSerializer,
    RecommendationSerializer,
    RepositorySerializer,
)
from .tasks import analyze_repository_task, run_chat_task

# Create your views here.


class RepositoryAnalyzeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        url = request.data.get("url")
        if not url:
            return Response(
                {"error": "URL is required."}, status=status.HTTP_400_BAD_REQUEST
            )

        repo, created = Repository.objects.get_or_create(url=url)
        if repo.status == "completed":
            return Response(RepositorySerializer(repo).data)

        repo.status = "processing"
        repo.error_message = ""
        repo.save()

        async_result = analyze_repository_task.delay(repo.pk)
        repo.task_id = async_result.id
        repo.save()
        return Response(
            {
                "message": "Repository analysis started.",
                "repository_id": repo.pk,
                "task_id": async_result.id,
            }
        )


class RepositoryStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, repo_id):
        repo = get_object_or_404(Repository, id=repo_id)
        return Response(RepositorySerializer(repo).data)


class ChatMessageView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = ChatMessageSerializer(
            data=request.data, context={"request": request}
        )
        serializer.is_valid(raise_exception=True)

        session_id = serializer.validated_data["session_id"]  # type: ignore
        user_message = serializer.validated_data["message"]  # type: ignore

        session = get_object_or_404(
            ConversationSession, id=session_id, user=request.user
        )

        current_state = session.state or {}
        current_state.setdefault("messages", [])
        current_state.setdefault("conversation_phase", "onboarding")
        current_state.setdefault("repo_id", session.repository.pk)
        current_state.setdefault("repo_url", session.repository.url)
        current_state["messages"].append({"role": "user", "content": user_message})

        # ← Send to Celery, don't wait
        task = run_chat_task.delay(session_id, current_state)
        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "session_id": session_id,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class ChatSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        repo_id = request.data.get("repository_id")
        if not repo_id:
            return Response(
                {"error": "repository_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        repo = get_object_or_404(Repository, id=repo_id, status="completed")

        session, created = ConversationSession.objects.get_or_create(
            user=request.user,
            repository=repo,
            defaults={
                "state": {
                    "repo_id": repo.pk,
                    "repo_url": repo.url,
                    "user_skills": [],
                    "selected_issue": None,
                    "conversation_phase": "onboarding",
                    "messages": [],
                },
                "phase": "onboarding",
            },
        )

        return Response(
            ConversationSessionSerializer(session).data,
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


class RecommendationFeedbackView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, rec_id):
        rec = get_object_or_404(Recommendation, id=rec_id, user=request.user)
        feedback = request.data.get("feedback")
        if feedback is None:
            return Response(
                {"error": "Feedback must be provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        rec.feedback = feedback
        rec.save()
        return Response(RecommendationSerializer(rec).data)


class ChatResultView(APIView):
    def get(self, request, task_id):
        from celery.result import AsyncResult

        result = AsyncResult(task_id)

        if result.ready():
            state = result.get()
            if state is not None:
                agent_message = ""
                for m in reversed(state["messages"]):
                    if m["role"] == "assistant":
                        agent_message = m["content"]
                        break
                return Response(
                    {
                        "status": "done",
                        "message": agent_message,
                        "phase": state.get("conversation_phase"),
                    }
                )

        return Response({"status": "processing"})


class UpdateSessionSkillsView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request, session_id):
        session = get_object_or_404(
            ConversationSession, id=session_id, user=request.user
        )
        skills = request.data.get("skills")
        if not isinstance(skills, list):
            return Response(
                {"error": "skills must be a list"}, status=status.HTTP_400_BAD_REQUEST
            )
        current_state = session.state or {}
        current_state["user_skills"] = skills
        session.state = current_state
        session.save(update_fields=["state"])
        return Response({"status": "skills updated"})


class LearnerProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        profile, _ = LearnerProfile.objects.get_or_create(user=request.user)
        return Response(LearnerProfileSerializer(profile).data)

    def patch(self, request):
        profile, _ = LearnerProfile.objects.get_or_create(user=request.user)
        serializer = LearnerProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)
