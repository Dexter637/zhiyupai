from django.urls import path
from . import views

app_name = 'user_app'

urlpatterns = [
    # 用户注册和认证
    path('register/', views.register_user, name='register'),
    path('login/', views.login_user, name='login'),
    path('profile/', views.get_user_profile, name='user_profile'),
    path('profile/<str:user_id>/', views.get_user_profile, name='user_profile_detail'),
    path('update-profile/', views.update_user_profile, name='update_profile'),
    
    # 用户健康数据概览
    path('health-overview/', views.user_health_overview, name='health_overview'),
]