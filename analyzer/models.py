from django.db import models

class WorkoutSession(models.Model):
    exercise_type = models.CharField(max_length=50)
    duration_seconds = models.FloatField(default=0.0)
    reps = models.IntegerField(default=0)
    date = models.DateTimeField(auto_now_add=True)
    accuracy_score = models.FloatField(default=100.0, help_text="Percentage accuracy based on form")

    def __str__(self):
        return f"{self.exercise_type} on {self.date.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        ordering = ['-date']

class UserProfile(models.Model):
    LEVEL_CHOICES = [
        ('Beginner', 'Beginner'),
        ('Intermediate', 'Intermediate'),
        ('Pro', 'Pro'),
    ]
    
    display_name = models.CharField(max_length=100, default="Athlete_01")
    email = models.EmailField(max_length=255, default="athlete@fitsense.com")
    user_level = models.CharField(max_length=20, choices=LEVEL_CHOICES, default='Intermediate')
    avatar_seed = models.CharField(max_length=50, default="Gym")
    updated_at = models.DateTimeField(auto_now=True)


    def __str__(self):
        return self.display_name

