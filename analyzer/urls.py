from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.home, name='home'),
    # New browser-based analysis endpoints
    path('analyze/live_frame/', views.analyze_live_frame, name='live_frame'),
    path('analyze/reset_state/', views.reset_analysis_state, name='reset_state'),
    path('analyze/get_live_status/', views.get_live_status, name='get_live_status'),
    # Legacy live feed endpoints (use server camera - keep for local testing)
    path('analyze/live_pushup/', views.live_pushup, name='live_pushup'),
    path('analyze/live_plank/', views.live_plank, name='live_plank'),
    path('analyze/live_situp/', views.live_situp, name='live_situp'),
    path('analyze/live_squat/', views.live_squat, name='live_squat'),
    path('analyze/live_jumping_jacks/', views.live_jumping_jacks, name='live_jumping_jacks'),
    path('analyze/live_reverse_plank/', views.live_reverse_plank, name='live_reverse_plank'),
    path('analyze/live_side_plank/', views.live_side_plank, name='live_side_plank'),
    path('analyze/live_bicep_curl/', views.live_bicep_curl, name='live_bicep_curl'),
    # Video upload endpoints
    path('analyze/analyze_pushup/', views.upload_and_analyze_pushup, name='analyze_pushup'),
    path('analyze/analyze_plank/', views.upload_and_analyze_plank, name='analyze_plank'),
    path('analyze/analyze_situp/', views.upload_and_analyze_situp, name='analyze_situp'),
    path('analyze/analyze_squat/', views.upload_and_analyze_squat, name='analyze_squat'),
    path('analyze/analyze_jumping_jacks/', views.upload_and_analyze_jumping_jacks, name='upload_jumping_jacks'),
    path('analyze/analyze_reverse_plank/', views.upload_and_analyze_reverse_plank, name='upload_reverse_plank'),
    path('analyze/analyze_side_plank/', views.upload_and_analyze_side_plank, name='upload_side_plank'),
    path('analyze/analyze_bicep_curl/', views.upload_and_analyze_bicep_curl, name='analyze_bicep_curl'),
]
