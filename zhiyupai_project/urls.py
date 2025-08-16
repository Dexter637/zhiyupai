"""
URL configuration for zhiyupai_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView

from django.views.generic import TemplateView

urlpatterns = [
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('sleep_analysis/', TemplateView.as_view(template_name='sleep_analysis.html'), name='sleep_analysis'),
    path('admin/', admin.site.urls),
    
    # JWT认证路由
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    
    # 应用API路由
    path('api/health/', include('health_app.urls')),
    path('api/users/', include('user_app.urls')),
    
    # REST Framework浏览器API
    path('api-auth/', include('rest_framework.urls')),
    
    # 临时测试API，绕过认证
    path('test/sleep_data/', lambda request: JsonResponse({
        "status": "success",
        "count": 5,
        "data": [
            {"_id": "1", "user_id": "admin", "data_type": "sleep", "timestamp": "2025-08-15T23:00:00", "values": {"sleep_start_time": "2025-08-15T23:30:00", "sleep_end_time": "2025-08-16T06:45:00", "total_sleep_mins": 435, "deep_sleep_mins": 135, "sleep_quality": 85}},
            {"_id": "2", "user_id": "admin", "data_type": "sleep", "timestamp": "2025-08-14T23:00:00", "values": {"sleep_start_time": "2025-08-14T22:15:00", "sleep_end_time": "2025-08-15T06:30:00", "total_sleep_mins": 495, "deep_sleep_mins": 165, "sleep_quality": 90}},
            {"_id": "3", "user_id": "admin", "data_type": "sleep", "timestamp": "2025-08-13T23:00:00", "values": {"sleep_start_time": "2025-08-13T00:15:00", "sleep_end_time": "2025-08-13T06:00:00", "total_sleep_mins": 345, "deep_sleep_mins": 90, "sleep_quality": 70, "is_night_owl": True}},
            {"_id": "4", "user_id": "admin", "data_type": "sleep", "timestamp": "2025-08-12T23:00:00", "values": {"sleep_start_time": "2025-08-12T22:45:00", "sleep_end_time": "2025-08-13T06:15:00", "total_sleep_mins": 450, "deep_sleep_mins": 150, "sleep_quality": 88}},
            {"_id": "5", "user_id": "admin", "data_type": "sleep", "timestamp": "2025-08-11T23:00:00", "values": {"sleep_start_time": "2025-08-12T00:30:00", "sleep_end_time": "2025-08-12T06:45:00", "total_sleep_mins": 375, "deep_sleep_mins": 105, "sleep_quality": 75, "is_night_owl": True}}
        ]
    }), name='test_sleep_data')
]

# 在开发环境中提供媒体文件服务
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
