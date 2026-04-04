from django.urls import path
from .views import RepositoryStatusView, RepositoryAnalyzeView, ChatSessionView, ChatMessageView, RecommendationFeedbackView

urlpatterns = [
    path('repositories/<int:repo_id>/status/', RepositoryStatusView.as_view(), name='repository-status'),
    path('repositories/analyze/', RepositoryAnalyzeView.as_view(), name='repository-analyze'),
    path('chat/session/', ChatSessionView.as_view(), name='chat-session'),
    path('chat/message/', ChatMessageView.as_view(), name='chat-message'),
    path('recommendations/<int:recommendation_id>/feedback/',
     RecommendationFeedbackView.as_view()),
]