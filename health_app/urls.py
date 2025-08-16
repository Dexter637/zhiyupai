from django.urls import path
from . import views, sleep_views

app_name = 'health_app'

urlpatterns = [
    # 心率数据相关API
    path('heart-rate/', views.heart_rate_list, name='heart_rate_list'),
    path('heart-rate/<str:user_id>/', views.heart_rate_detail, name='heart_rate_detail'),
    path('heart-rate/alerts/<str:user_id>/', views.heart_rate_alerts, name='heart_rate_alerts'),
    
    # 睡眠数据相关API
    path('sleep/', views.sleep_data_list, name='sleep_data_list'),
    path('sleep/<str:user_id>/', views.sleep_data_detail, name='sleep_data_detail'),
    
    # 活动推荐相关API
    path('recommendations/<str:user_id>/', views.activity_recommendations, name='activity_recommendations'),
    path('recommendations/complete/<str:recommendation_id>/', views.complete_recommendation, name='complete_recommendation'),
    
    # 可穿戴设备数据上传API
    path('wearable-data/upload/', views.wearable_data_upload, name='wearable_data_upload'),
    path('wearable-data/<str:user_id>/', views.get_user_wearable_data, name='get_user_wearable_data'),
    path('wearable-data/summary/<str:user_id>/<str:data_type>/', views.get_data_summary, name='get_data_summary'),
    
    # 睡眠分析页面和API
    path('sleep-analysis/', sleep_views.sleep_analysis_view, name='sleep_analysis'),
    path('sleep-data/', sleep_views.get_sleep_data_api, name='get_sleep_data_api'),
    path('api/sleep-data/', sleep_views.get_sleep_data_api, name='get_sleep_data_api_alt'),
    path('api/sleep_data/', sleep_views.get_sleep_data_api, name='get_sleep_data_api_alt_underscore'),
]