import logging
import re

import requests
from django.conf import settings
from django.contrib.auth.models import User
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import (
    ConversationSession,
    LearnerProfile,
    Recommendation,
    Repository,
    UserProfile,
)
from .serializers import (
    ChatMessageSerializer,
    ConversationSessionSerializer,
    LearnerProfileSerializer,
    RecommendationSerializer,
    RepositorySerializer,
    StructuredSkillsSerializer,
)
from .tasks import analyze_repository_task, run_chat_task

# Create your views here.
logger = logging.getLogger(__name__)


def is_valid_github_url(url: str) -> bool:
    pattern = r"^https?://github\.com/[\w.-]+/[\w.-]+/?$"
    return bool(re.match(pattern, url.rstrip("/")))


class RepositoryAnalyzeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        url = request.data.get("url")
        if not url:
            logger.warning("RepositoryAnalyzeView: URL is required")
            return Response(
                {"error": "URL is required."}, status=status.HTTP_400_BAD_REQUEST
            )

        url = url.strip()
        if not is_valid_github_url(url):
            logger.warning(f"RepositoryAnalyzeView: Invalid GitHub URL: {url}")
            return Response(
                {"error": "Invalid GitHub repository URL."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            with transaction.atomic():
                repo, created = Repository.objects.select_for_update().get_or_create(
                    url=url
                )

                if repo.status == "completed" and repo.index_path:
                    logger.info(
                        f"RepositoryAnalyzeView: Repository already completed: {repo.pk}"
                    )
                    return Response(RepositorySerializer(repo).data)

                if repo.status == "processing":
                    logger.info(
                        f"RepositoryAnalyzeView: Repository already processing: {repo.pk}"
                    )
                    return Response(
                        {
                            "message": "Repository analysis already in progress.",
                            "repository_id": repo.pk,
                            "task_id": repo.task_id,
                        },
                        status=status.HTTP_202_ACCEPTED,
                    )

                repo.status = "processing"
                repo.error_message = ""
                repo.save()

            async_result = analyze_repository_task.delay(repo.pk)

            repo.task_id = async_result.id
            repo.save(update_fields=["task_id"])

            logger.info(
                f"RepositoryAnalyzeView: Started analysis for repo {repo.pk}, task {async_result.id}"
            )
            return Response(
                {
                    "message": "Repository analysis started.",
                    "repository_id": repo.pk,
                    "task_id": async_result.id,
                }
            )
        except Exception as e:
            logger.exception(f"RepositoryAnalyzeView: Failed to start analysis: {e}")
            return Response(
                {"error": "Failed to start repository analysis"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
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


class UpdateUsernameView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request):
        new_username = request.data.get("username", "").strip()
        if not new_username:
            return Response({"error": "Username is required"}, status=400)

        if (
            User.objects.filter(username=new_username)
            .exclude(pk=request.user.pk)
            .exists()
        ):
            return Response({"error": "Username already taken"}, status=400)

        request.user.username = new_username
        request.user.save()
        return Response({"username": new_username})


class ResendVerificationView(APIView):
    permission_classes = []

    def post(self, request):
        email = request.data.get("email", "").strip()
        if not email:
            return Response({"error": "Email is required"}, status=400)

        try:
            resp = requests.post(
                f"{settings.SUPABASE_URL}/auth/v1/resend",
                headers={
                    "apikey": settings.SUPABASE_PUBLISHABLE_KEY,
                    "Content-Type": "application/json",
                },
                json={"type": "signup", "email": email},
                timeout=10,
            )
            if not resp.ok:
                return Response(
                    {"error": "Failed to resend verification email"},
                    status=resp.status_code,
                )
        except requests.RequestException as e:
            return Response({"error": f"Unable to send email: {e}"}, status=502)

        return Response({"message": "Verification email sent"})


class SignupView(APIView):
    permission_classes = []

    def post(self, request):
        username = request.data.get("username")
        email = request.data.get("email")
        password = request.data.get("password")
        password2 = request.data.get("password2")

        if not username or not email or not password or not password2:
            return Response({"error": "All fields required"}, status=400)

        if password != password2:
            return Response({"error": "Passwords do not match"}, status=400)

        if User.objects.filter(username=username).exists():
            return Response({"error": "Username taken"}, status=400)

        if User.objects.filter(email=email).exists():
            return Response({"error": "Email already exists"}, status=400)

        _ = User.objects.create_user(username=username, email=email, password=password)

        return Response({"message": "User created"})


class LoginView(APIView):
    permission_classes = []

    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")

        if not email or not password:
            return Response({"error": "Email and password required"}, status=400)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"error": "Invalid credentials"}, status=401)

        if not user.check_password(password):
            return Response({"error": "Invalid credentials"}, status=401)

        refresh = RefreshToken.for_user(user)
        access_token = str(refresh.access_token)
        refresh_token = str(refresh)

        response = JsonResponse(
            {
                "username": user.username,
            }
        )
        response.set_cookie(
            "access_token",
            access_token,
            httponly=True,
            secure=settings.SIMPLE_JWT["AUTH_COOKIE_SECURE"],
            samesite=settings.SIMPLE_JWT["AUTH_COOKIE_SAMESITE"],
        )
        response.set_cookie(
            "refresh_token",
            refresh_token,
            httponly=True,
            secure=settings.SIMPLE_JWT["AUTH_COOKIE_SECURE"],
            samesite=settings.SIMPLE_JWT["AUTH_COOKIE_SAMESITE"],
        )
        return response


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        response = Response({"message": "Logged out"})
        response.delete_cookie("access_token")
        response.delete_cookie("refresh_token")
        return response


class SupabaseLoginView(APIView):
    permission_classes = []

    def post(self, request):
        access_token = request.data.get("access_token")
        if not access_token:
            logger.warning("SupabaseLoginView: Missing access_token")
            return Response({"error": "access_token required"}, status=400)

        # Verify token and get user info via Supabase REST API
        try:
            user_resp = requests.get(
                f"{settings.SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "apikey": settings.SUPABASE_PUBLISHABLE_KEY,
                },
                timeout=10,
            )
        except requests.RequestException as e:
            logger.exception(
                f"SupabaseLoginView: Failed to verify token with Supabase: {e}"
            )
            return Response({"error": f"Unable to verify token: {e}"}, status=502)

        if not user_resp.ok:
            logger.warning(
                f"SupabaseLoginView: Supabase token validation failed: {user_resp.status_code}"
            )
            return Response(
                {"error": "Invalid or expired token"},
                status=401,
            )

        user_data = user_resp.json()
        email = user_data.get("email", "")
        supabase_uid = user_data.get("id", "")
        email_confirmed_at = user_data.get("email_confirmed_at")

        if not email or not supabase_uid:
            logger.warning(
                "SupabaseLoginView: Invalid token payload - missing email or supabase_uid"
            )
            return Response({"error": "Invalid token payload"}, status=400)

        provider = user_data.get("app_metadata", {}).get("provider", "")
        is_oauth = provider != "email" and provider != ""
        email_verified = user_data.get("user_metadata", {}).get("email_verified", False)

        if not email_confirmed_at and not is_oauth and not email_verified:
            logger.info(f"SupabaseLoginView: Email not verified for {email}")
            return Response(
                {
                    "error": "Please verify your email before signing in. Check your inbox for the verification link."
                },
                status=403,
            )

        # Use atomic transaction with get_or_create to prevent race conditions
        try:
            with transaction.atomic():
                # First try to find existing profile by supabase_uid
                profile = (
                    UserProfile.objects.select_for_update()
                    .filter(supabase_uid=supabase_uid)
                    .first()
                )
                if profile:
                    user = profile.user
                    logger.info(
                        f"SupabaseLoginView: Found existing profile for supabase_uid {supabase_uid}"
                    )
                else:
                    # Try to find user by email
                    user = User.objects.select_for_update().filter(email=email).first()
                    if user:
                        profile, created = UserProfile.objects.get_or_create(user=user)
                        if created or profile.supabase_uid != supabase_uid:
                            profile.supabase_uid = supabase_uid
                            profile.email_verified = True
                            profile.save()
                        logger.info(
                            f"SupabaseLoginView: Linked existing user {user.username} to supabase_uid {supabase_uid}"
                        )
                    else:
                        # Create new user with unique username
                        username = email.split("@")[0].replace(".", "_")
                        base_username = username
                        counter = 1
                        while User.objects.filter(username=username).exists():
                            username = f"{base_username}_{counter}"
                            counter += 1

                        user = User.objects.create_user(username=username, email=email)
                        user.set_unusable_password()
                        user.save()

                        profile = UserProfile.objects.create(
                            user=user,
                            supabase_uid=supabase_uid,
                            email_verified=True,
                        )
        except Exception as e:
            return Response({"error": f"Failed to create user: {str(e)}"}, status=500)

        refresh = RefreshToken.for_user(user)
        access = str(refresh.access_token)
        refresh_token = str(refresh)

        response = JsonResponse({"username": user.username, "email": user.email})
        response.set_cookie(
            "access_token",
            access,
            httponly=True,
            secure=settings.SIMPLE_JWT["AUTH_COOKIE_SECURE"],
            samesite=settings.SIMPLE_JWT["AUTH_COOKIE_SAMESITE"],
        )
        response.set_cookie(
            "refresh_token",
            refresh_token,
            httponly=True,
            secure=settings.SIMPLE_JWT["AUTH_COOKIE_SECURE"],
            samesite=settings.SIMPLE_JWT["AUTH_COOKIE_SAMESITE"],
        )
        return response


class StructuredSkillsView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request, session_id):
        session = get_object_or_404(
            ConversationSession, id=session_id, user=request.user
        )
        skills_data = request.data.get("skills")
        if not isinstance(skills_data, list):
            return Response(
                {"error": "skills must be a list of objects with 'skill' and 'band'"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        validated_skills = []
        for item in skills_data:
            serializer = StructuredSkillsSerializer(data=item)
            if not serializer.is_valid():
                return Response(
                    {"error": f"Invalid skill entry: {serializer.errors}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            validated_skills.append(serializer.validated_data)

        current_state = session.state or {}
        current_state["user_skills"] = validated_skills
        session.state = current_state
        session.save(update_fields=["state"])

        return Response(
            {
                "status": "skills updated",
                "user_skills": validated_skills,
            },
            status=status.HTTP_200_OK,
        )


class RecommendationView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        """Get recommended issues based on user's selected skills."""
        import os

        from .services.learning_path import generate_learning_path
        from .services.recommender import RecommendationEngine

        session = get_object_or_404(
            ConversationSession, id=session_id, user=request.user
        )
        repo = session.repository

        # Get user skills from session state
        user_skills = session.state.get("user_skills", [])
        if (
            isinstance(user_skills, list)
            and user_skills
            and isinstance(user_skills[0], dict)
        ):
            # Extract just the skill names from structured skills
            user_skill_names = [s["skill"] for s in user_skills]
        else:
            user_skill_names = user_skills if isinstance(user_skills, list) else []

        try:
            # Load the recommendation engine with repo index
            index_path = repo.index_path
            if not index_path or not os.path.exists(index_path):
                return Response(
                    {
                        "error": "Repository index not found. Please re-analyze the repository."
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            engine = RecommendationEngine()
            engine.load_index(index_path)

            # If no user skills selected, use default skills from repo
            if not user_skill_names:
                user_skill_names = repo.skills_found[:3] if repo.skills_found else []

            # Get recommendations
            recommendations = (
                engine.recommend(user_skill_names, top_k=5) if user_skill_names else []
            )

            if not recommendations:
                # No matching issues - generate learning path
                all_issues = engine.graph.issues.meta
                learning_path = generate_learning_path(all_issues)
                return Response(
                    {
                        "status": "no_match",
                        "recommendations": [],
                        "learning_path": learning_path,
                        "message": learning_path,
                    }
                )

            # Format recommendations for frontend
            formatted_recs = []
            for rec in recommendations:
                formatted_recs.append(
                    {
                        "id": rec.get("id"),
                        "title": rec.get("title", ""),
                        "difficulty": rec.get("difficulty", "intermediate"),
                        "skills": rec.get("skills", []),
                        "labels": rec.get("labels", []),
                        "combined_score": rec.get("combined_score", 0),
                        "match_score": rec.get("match_score", 0),
                        "summary": rec.get("summary", ""),
                    }
                )

            return Response(
                {
                    "status": "success",
                    "recommendations": formatted_recs,
                    "count": len(formatted_recs),
                }
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return Response(
                {"error": f"Failed to fetch recommendations: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SelectIssueView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request, session_id):
        """Store selected issue in session state."""
        session = get_object_or_404(
            ConversationSession, id=session_id, user=request.user
        )

        issue_data = request.data.get("issue")
        if not issue_data:
            return Response(
                {"error": "issue data is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        current_state = session.state or {}
        current_state["selected_issue"] = issue_data
        current_state["conversation_phase"] = "guidance"
        session.state = current_state
        session.phase = "guidance"
        session.save(update_fields=["state", "phase"])

        return Response(
            {
                "status": "issue_selected",
                "issue": issue_data,
                "phase": session.phase,
            },
            status=status.HTTP_200_OK,
        )
