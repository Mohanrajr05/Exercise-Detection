from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload_pushup/', views.upload_and_analyze_pushup, name='upload_pushup'),
    path('live_pushup/', views.live_pushup, name='live_pushup'),
    path('upload_situp/', views.upload_and_analyze_situp, name='upload_situp'),
    path('live_situp/', views.live_situp, name='live_situp'),
    path('upload_plank/', views.upload_and_analyze_plank, name='upload_plank'),
    path('live_plank/', views.live_plank, name='live_plank'),
]