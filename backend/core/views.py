from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import Repository, Recommendation, ConversationSession
from .serializers import RepositorySerializer, RecommendationSerializer, ConversationSessionSerializer, ChatMessageSerializer

# Create your views here.

class RepositoryAnalyzeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        url = request.data.get("url")
        if not url:
            return Response({"error": "URL is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        repo, created = Repository.objects.get_or_create(url=url)
        if repo.status == "completed":
            return Response(RepositorySerializer(repo).data)
        
        repo.status = "processing"
        repo.save()

        return Response({"message": "Repository analysis started."}, status=status.HTTP_202_ACCEPTED)

class  RepositoryStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, repo_id):
        repo = get_object_or_404(Repository, id=repo_id)
        return Response(RepositorySerializer(repo).data)

class ChatSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        repo_id = request.data.get("repository_id")

        if not repo_id:
            return Response({"error": "repository_id is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        repo = get_object_or_404(Repository, id=repo_id, status="ready")

        session, created = ConversationSession.objects.get_or_create(
            user=request.user,
            repository=repo,
            defaults={"state":{}, "phase": "onboarding"}
        )

        return Response(ConversationSessionSerializer(session).data, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)

class ChatMessageView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = ChatMessageSerializer(
            data=request.data,
            context={"request": request}
        )
        if not serializer.is_valid():
            return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)
        session_id = serializer.validated_data["session_id"]
        session: ConversationSession = get_object_or_404(
            ConversationSession,
            id=session_id,
            user=request.user
        )

        return Response({
            "message": "Agent response goes here",
            "phase": session.phase,
            "session_id": session.pk,
        })

class RecommendationFeedbackView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, rec_id):
        rec = get_object_or_404(Recommendation, id=rec_id, user=request.user)
        feedback = request.data.get("feedback")
        if feedback is None:
            return Response({"error": "Feedback must be provided."}, status=status.HTTP_400_BAD_REQUEST)
        
        rec.feedback = feedback
        rec.save()
        return Response(RecommendationSerializer(rec).data)
