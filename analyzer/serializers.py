from rest_framework import serializers
from .models import WorkoutSession, UserProfile

class WorkoutSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkoutSession
        fields = ['id', 'exercise_type', 'duration_seconds', 'reps', 'date', 'accuracy_score']
        read_only_fields = ['id', 'date']

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['id', 'display_name', 'email', 'user_level', 'avatar_seed', 'updated_at']
        read_only_fields = ['id', 'updated_at']


