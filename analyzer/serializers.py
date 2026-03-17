from rest_framework import serializers
from .models import WorkoutSession

class WorkoutSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkoutSession
        fields = ['id', 'exercise_type', 'duration_seconds', 'reps', 'date', 'accuracy_score']
        read_only_fields = ['id', 'date']
