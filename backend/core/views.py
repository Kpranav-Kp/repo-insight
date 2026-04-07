from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ConversationSession, Recommendation, Repository
from .serializers import (
    ChatMessageSerializer,
    ConversationSessionSerializer,
    RecommendationSerializer,
    RepositorySerializer,
)
from .services.agents.graph import build_graph
from .tasks import analyze_repository_task

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
    def post(self, request):
        # existing serializer validation stays the same
        serializer = ChatMessageSerializer(
            data=request.data, context={"request": request}
        )
        serializer.is_valid(raise_exception=True)

        session_id = serializer.validated_data["session_id"]
        user_message = serializer.validated_data["message"]

        session = get_object_or_404(
            ConversationSession, id=session_id, user=request.user
        )

        # load current state from DB
        current_state = session.state or {}
        current_state.setdefault("messages", [])
        current_state.setdefault("conversation_phase", "onboarding")
        current_state.setdefault("repo_id", session.repository.id)
        current_state.setdefault("repo_url", session.repository.url)

        # add user message
        current_state["messages"].append({"role": "user", "content": user_message})

        # run through LangGraph
        graph = build_graph()
        result = graph.invoke(current_state)

        # save updated state back to DB
        session.state = result
        session.phase = result.get("conversation_phase", "onboarding")
        session.save()

        # get last agent message
        agent_message = ""
        for m in reversed(result["messages"]):
            if m["role"] == "assistant":
                agent_message = m["content"]
                break

        return Response(
            {
                "message": agent_message,
                "phase": session.phase,
                "session_id": session.pk,
            }
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
                    "intent": "",
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


"""
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
                    "intent": "",
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
"""


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
