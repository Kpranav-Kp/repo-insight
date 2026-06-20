# backend/core/urls.py
from django.urls import path

from .views import (
    ChatMessageView,
    ChatResultView,
    ChatSessionView,
    ClaimIssueView,
    LearnerProfileView,
    NoSkillsView,
    RecommendationFeedbackView,
    RecommendationView,
    ReleaseIssueView,
    RepositoryAnalyzeView,
    RepositoryStatusView,
    SelectIssueView,
    StructuredSkillsView,
    SupabaseLoginView,
    UpdateSessionSkillsView,
)

urlpatterns = [
    path(
        "repositories/<int:repo_id>/status/",
        RepositoryStatusView.as_view(),
        name="repository-status",
    ),
    path(
        "repositories/analyze/",
        RepositoryAnalyzeView.as_view(),
        name="repository-analyze",
    ),
    path("chat/session/", ChatSessionView.as_view(), name="chat-session"),
    path("chat/message/", ChatMessageView.as_view(), name="chat-message"),
    path("chat/result/<str:task_id>/", ChatResultView.as_view(), name="chat-result"),
    path(
        "chat/session/<int:session_id>/skills/",
        UpdateSessionSkillsView.as_view(),
        name="update-session-skills",
    ),
    path(
        "learner-profile/",
        LearnerProfileView.as_view(),
        name="learner-profile",
    ),
    path(
        "recommendations/<int:recommendation_id>/feedback/",
        RecommendationFeedbackView.as_view(),
        name="recommendation-feedback",
    ),
    path("auth/supabase/", SupabaseLoginView.as_view()),
    path(
        "chat/session/<int:session_id>/skills/structured/",
        StructuredSkillsView.as_view(),
    ),
    path(
        "chat/session/<int:session_id>/recommendations/",
        RecommendationView.as_view(),
        name="recommendations",
    ),
    path(
        "chat/session/<int:session_id>/select-issue/",
        SelectIssueView.as_view(),
        name="select-issue",
    ),
    path(
        "repositories/<int:repo_id>/claim/",
        ClaimIssueView.as_view(),
        name="claim-issue",
    ),
    path(
        "repositories/<int:repo_id>/release/<int:issue_number>/",
        ReleaseIssueView.as_view(),
        name="release-issue",
    ),
    path(
        "chat/session/<int:session_id>/no-skills/",
        NoSkillsView.as_view(),
        name="no-skills",
    ),
]
