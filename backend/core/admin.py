from django.contrib import admin

from .models import (
    ConversationSession,
    IssueClaim,
    LearnerProfile,
    Recommendation,
    Repository,
    SkillFeedbackSummary,
    UserProfile,
)

admin.site.register(Repository)
admin.site.register(UserProfile)
admin.site.register(Recommendation)
admin.site.register(ConversationSession)
admin.site.register(LearnerProfile)
admin.site.register(IssueClaim)
admin.site.register(SkillFeedbackSummary)
