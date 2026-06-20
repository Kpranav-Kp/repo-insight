import logging
import os
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
    IssueClaim,
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


REPO_CACHE_TTL_HOURS = getattr(settings, "REPO_CACHE_TTL_HOURS", 6)


class RepositoryAnalyzeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from datetime import timedelta

        from django.utils import timezone

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

                cache_valid = (
                    repo.status == "completed"
                    and repo.index_path
                    and repo.analyzed_at is not None
                    and timezone.now() - repo.analyzed_at
                    < timedelta(hours=REPO_CACHE_TTL_HOURS)
                )
                if cache_valid:
                    logger.info(
                        f"RepositoryAnalyzeView: Repository cache still valid: {repo.pk}"
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
                repo.task_id = ""
                repo.save()

            async_result = analyze_repository_task.delay(repo.pk)
            Repository.objects.filter(id=repo.pk).update(task_id=async_result.id)

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
                        )
        except Exception:
            logger.exception("SupabaseLoginView: Failed to create user")
            return Response({"error": "Failed to create user"}, status=500)

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
        from .services.learning_path import generate_learning_path
        from .services.recommender import RecommendationEngine

        session = get_object_or_404(
            ConversationSession, id=session_id, user=request.user
        )
        repo = session.repository

        user_skills = session.state.get("user_skills", [])
        if (
            isinstance(user_skills, list)
            and user_skills
            and isinstance(user_skills[0], dict)
        ):
            user_skill_names = [s["skill"] for s in user_skills]
        else:
            user_skill_names = user_skills if isinstance(user_skills, list) else []

        claimed = IssueClaim.objects.filter(repository=repo).values_list(
            "issue_number", flat=True
        )
        exclude_ids = {str(num) for num in claimed}

        try:
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

            if not user_skill_names:
                sk = repo.skills_found or []
                sk_sorted = sorted(sk, key=lambda s: (-len(s), s))
                user_skill_names = sk_sorted[:3]
                user_skills = user_skill_names

            recommendations = (
                engine.recommend(user_skills, top_k=5, exclude_issue_ids=exclude_ids)
                if user_skill_names
                else []
            )

            if not recommendations:
                has_history = IssueClaim.objects.filter(
                    repository=repo, user=request.user
                ).exists()
                learning_path = generate_learning_path(engine.graph, user_skill_names)
                return Response(
                    {
                        "status": "no_match",
                        "recommendations": [],
                        "learning_path": learning_path,
                        "message": learning_path,
                        "has_contribution_history": has_history,
                        "all_skills_in_repo": repo.skills_found or [],
                    }
                )

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
        session = get_object_or_404(
            ConversationSession, id=session_id, user=request.user
        )

        issue_data = request.data.get("issue")
        if not issue_data:
            return Response(
                {"error": "issue data is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        issue_number = issue_data.get("id")
        if issue_number:
            IssueClaim.objects.get_or_create(
                issue_number=int(issue_number),
                repository=session.repository,
                defaults={"user": request.user},
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


class ClaimIssueView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, repo_id):
        issue_number = request.data.get("issue_number")
        if not issue_number:
            return Response(
                {"error": "issue_number is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        repo = get_object_or_404(Repository, id=repo_id)
        _, created = IssueClaim.objects.get_or_create(
            issue_number=int(issue_number),
            repository=repo,
            defaults={"user": request.user},
        )

        if not created:
            return Response(
                {"error": "Issue already claimed by another user"},
                status=status.HTTP_409_CONFLICT,
            )

        return Response({"status": "claimed", "issue_number": issue_number})


class ReleaseIssueView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, repo_id, issue_number):
        repo = get_object_or_404(Repository, id=repo_id)
        IssueClaim.objects.filter(
            issue_number=issue_number,
            repository=repo,
            user=request.user,
        ).delete()
        return Response({"status": "released", "issue_number": issue_number})


class NoSkillsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        session = get_object_or_404(
            ConversationSession, id=session_id, user=request.user
        )
        repo: Repository = session.repository

        skills = repo.skills_found or []

        if not skills:
            return Response(
                {
                    "status": "no_skills",
                    "roadmap": "No skills were identified for this repository.",
                }
            )

        lines = [
            f"To contribute to **{repo.url}**, you'll need familiarity with these skills:",
            "",
        ]
        for i, skill in enumerate(skills[:8], 1):
            lines.append(f"{i}. **{skill}**")
        if len(skills) > 8:
            lines.append(f"\n... and {len(skills) - 8} more.")
        lines.extend(
            [
                "",
                "Start with the skills that interest you most. "
                "Pick one and practice with beginner-friendly issues, "
                "then come back when you feel ready!",
            ]
        )

        return Response(
            {
                "status": "no_skills",
                "roadmap": "\n".join(lines),
                "skills_found": skills[:8],
            }
        )
