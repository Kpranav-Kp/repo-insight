from django.db import models
from django.contrib.auth.models import User

class Repository(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    url = models.URLField(unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    task_id = models.CharField(max_length=255, blank=True)
    issues_indexed = models.IntegerField(default=0)
    prs_indexed = models.IntegerField(default=0)
    skills_found = models.JSONField(default=list)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.url

class UserProfile(models.Model):
    BAND_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    skills = models.JSONField(default=list)
    target_repo = models.ForeignKey(Repository, null=True, blank=True, on_delete=models.SET_NULL)
    intent = models.TextField(blank=True)

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

    class Meta:
        ordering = ["-combined_score"]

class ConversationSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    repository = models.ForeignKey(Repository, on_delete=models.CASCADE)
    state = models.JSONField(default=dict)
    phase = models.CharField(max_length=50, default="onboarding")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['user', 'repository']