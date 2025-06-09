from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze/live_pushup/', views.live_pushup, name='live_pushup'),
    path('analyze/live_plank/', views.live_plank, name='live_plank'),
    path('analyze/live_situp/', views.live_situp, name='live_situp'),
    path('analyze/live_squat/', views.live_squat, name='live_squat'),
    path('analyze/analyze_pushup/', views.upload_and_analyze_pushup, name='analyze_pushup'),
    path('analyze/analyze_plank/', views.upload_and_analyze_plank, name='analyze_plank'),
    path('analyze/analyze_situp/', views.upload_and_analyze_situp, name='analyze_situp'),
    path('analyze/analyze_squat/', views.upload_and_analyze_squat, name='analyze_squat'),
]
