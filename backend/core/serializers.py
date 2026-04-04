from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Repository, UserProfile, Recommendation, ConversationSession

class RepositorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Repository
        fields = ["id", "url", "status", "task_id",
                  "issues_indexed", "prs_indexed", "skills_found",
                  "error_message", "created_at"]
        read_only_fields = ["status", "task_id", "issues_indexed",
                            "prs_indexed", "skills_found", "error_message"]
        
class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ["skills", "target_repo", "intent"]
        
    def validate_skills(self, value):
        valid_bands = {"beginner", "intermediate", "advanced"}

        if not isinstance(value, list):
            raise serializers.ValidationError("Skills must be a list.")
        
        for item in value:
            if not isinstance(item, dict):
                raise serializers.ValidationError("Each skill must be a dictionary with skill and band.")
            if "skill" not in item or "band" not in item:
                raise serializers.ValidationError("Each skill must have 'skill' and 'band' keys.")
            if item["band"] not in valid_bands:
                raise serializers.ValidationError(f"Band must be one of {valid_bands}.")
            if not isinstance(item["skill"], str) or not item["skill"].strip():
                raise serializers.ValidationError("Skill must be a non-empty string.")
        return value
    
class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = "__all__"
        read_only_fields = ["user", "repository", "match_score", 
                            "novelty_score", "combined_score"]
        
class ChatMessageSerializer(serializers.Serializer):
    
    session_id = serializers.UUIDField()
    message = serializers.CharField(max_length=2000)
    
    def validate_session_id(self, value):
        request = self.context.get("request")
        try:
            session = ConversationSession.objects.get(id=value)
            if session.user != request.user:
                raise serializers.ValidationError(
                    "This session does not belong to you."
                )
        except ConversationSession.DoesNotExist:
            raise serializers.ValidationError("Invalid session ID.")
        
        return value
    
class ConversationSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConversationSession
        fields = ["id", "repository", "phase", "created_at", "updated_at"]
        read_only_fields = ["phase"]