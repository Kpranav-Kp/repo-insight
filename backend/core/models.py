from django.contrib.auth.models import User
from django.db import models


class Repository(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    url = models.URLField(unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    task_id = models.CharField(max_length=255, blank=True)
    issues_indexed = models.IntegerField(default=0)
    prs_indexed = models.IntegerField(default=0)
    skills_found = models.JSONField(default=list)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    index_path = models.CharField(max_length=500, blank=True)

    def __str__(self):
        return self.url


class UserProfile(models.Model):
    BAND_CHOICES = [
        ("beginner", "Beginner"),
        ("intermediate", "Intermediate"),
        ("advanced", "Advanced"),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    skills = models.JSONField(default=list)
    target_repo = models.ForeignKey(
        Repository, null=True, blank=True, on_delete=models.SET_NULL
    )

    def __str__(self):
        return f"{self.user.username}'s profile"


class Recommendation(models.Model):
    repository = models.ForeignKey(Repository, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    issue_id = models.CharField(max_length=255)
    title = models.TextField()
    summary = models.TextField(blank=True)
    labels = models.JSONField(default=list)
    skills = models.JSONField(default=list)
    skills_matched = models.JSONField(default=list)
    match_score = models.FloatField()
    novelty_score = models.FloatField()
    combined_score = models.FloatField()
    feedback = models.BooleanField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    pr_created_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-combined_score"]

    def __str__(self):
        return f"Recommendation for {self.user.username} on {self.repository.url} - Issue {self.issue_id}"


class ConversationSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    repository = models.ForeignKey(Repository, on_delete=models.CASCADE)
    state = models.JSONField(default=dict)
    phase = models.CharField(max_length=50, default="onboarding")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    code_assist_count = models.IntegerField(default=0)
    stuck_counter = models.IntegerField(default=0)

    class Meta:
        unique_together = ["user", "repository"]

    def __str__(self):
        return f"Session for {self.user.username} on {self.repository.url} - Phase: {self.phase}"


class LearnerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    completed_issues = models.JSONField(default=list)
    code_assist_used = models.IntegerField(default=0)
    last_active = models.DateTimeField(auto_now=True)
    mastered_skills = models.JSONField(default=list)

    def __str__(self):
        return f"{self.user.username}'s Learner Profile"
