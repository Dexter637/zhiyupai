from django.shortcuts import render, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.contrib.auth import authenticate
from django.db import IntegrityError
from django.conf import settings

from rest_framework import permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.http import JsonResponse
import json
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

import json
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MongoDB模型
from models import UserProfile, WearableData, HeartRateAlert, ActivityRecommendation

# 用户注册
@api_view(['POST'])
@permission_classes([AllowAny])
def register_user(request):
    """注册新用户"""
    try:
        import json
        data = json.loads(request.body)
        
        # 验证必填字段
        required_fields = ['username', 'password']
        for field in required_fields:
            if field not in data:
                return JsonResponse(
                    {"error": f"缺少必填字段: {field}"},
                    status=400
                )
        
        # 验证用户名格式（仅允许英文与数字）
        import re
        if not re.match(r'^[a-zA-Z0-9]+$', data['username']):
            return JsonResponse(
                {"error": "用户名只能包含英文和数字"},
                status=400
            )

        # 验证密码格式（不允许中文，只允许字母、数字和.符号）
        if re.search(r'[一-龥]', data['password']):
            return JsonResponse(
                {"error": "密码不允许包含中文"},
                status=400
            )
        if not re.match(r'^[a-zA-Z0-9.]+$', data['password']):
            return JsonResponse(
                {"error": "密码只能包含字母、数字和.符号"},
                status=400
            )
        
        # 创建Django用户
        try:
            user = User.objects.create(
                username=data['username'],
                password=make_password(data['password'])
            )
        except IntegrityError:
            return JsonResponse(
                {"error": "用户名已存在"},
                status=400
            )
        
        # 创建MongoDB用户档案
        profile_data = {
            "user_id": str(user.id),
            "username": data['username'],
            "email": None,
            "date_joined": datetime.now(),
            "is_active": True
        }
        
        # 添加可选字段
        optional_fields = ['age', 'gender', 'height', 'weight', 'health_conditions', 'fitness_level']
        for field in optional_fields:
            if field in data:
                profile_data[field] = data[field]
        
        # 保存用户档案到MongoDB
        profile_id = UserProfile.create(profile_data)
        
        # 生成JWT令牌
        refresh = RefreshToken.for_user(user)
        
        return JsonResponse({
            "status": "success",
            "message": "用户注册成功",
            "user_id": user.id,
            "profile_id": str(profile_id),
            "tokens": {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return JsonResponse(
                {"error": f"注册用户时出错: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# 用户登录
@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    """用户登录"""
    try:
        data = json.loads(request.body)
        
        # 验证必填字段
        required_fields = ['username', 'password']
        for field in required_fields:
            if field not in data:
                return JsonResponse(
                {"error": f"缺少必填字段: {field}"},
                status=400
            )
        
        # 验证用户
        user = authenticate(username=data['username'], password=data['password'])
        
        if user is None:
            return JsonResponse(
                {"error": "用户名或密码错误"},
                status=401
            )
        
        # 生成JWT令牌
        refresh = RefreshToken.for_user(user)
        
        # 获取用户档案
        profile = UserProfile.find_by_user_id(str(user.id))
        
        return JsonResponse({
            "status": "success",
            "message": "用户登录成功",
            "user_id": user.id,
            "profile_id": str(profile['_id']) if profile else None,
            "tokens": {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }
        }, status=200)
        
    except Exception as e:
        return JsonResponse(
            {"error": f"登录用户时出错: {str(e)}"},
            status=500
        )

# 用户登录（使用JWT令牌，由rest_framework_simplejwt提供的视图处理）

# 用户档案
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_profile(request, user_id=None):
    """获取用户档案"""
    try:
        # 如果未指定user_id，则使用当前登录用户的ID
        if user_id is None:
            user_id = request.user.id
        
        # 检查权限（只能查看自己的档案，除非是管理员）
        if str(request.user.id) != str(user_id) and not request.user.is_staff:
            return Response(
                {"error": "您没有权限查看此用户档案"},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # 从MongoDB获取用户档案
        profile = UserProfile.find_by_user_id(str(user_id))
        
        if not profile:
            return Response(
                {"error": "未找到用户档案"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        return Response({
            "status": "success",
            "profile": profile
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取用户档案时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_user_profile(request):
    """更新用户档案"""
    try:
        user_id = request.user.id
        data = request.data
        
        # 从MongoDB获取用户档案
        profile = UserProfile.find_by_user_id(str(user_id))
        
        if not profile:
            return Response(
                {"error": "未找到用户档案"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # 可更新字段
        updatable_fields = [
            'first_name', 'last_name', 'email', 'age', 'gender',
            'height', 'weight', 'health_conditions', 'fitness_level',
            'notification_preferences', 'device_settings'
        ]
        
        # 准备更新数据
        update_data = {}
        for field in updatable_fields:
            if field in data:
                update_data[field] = data[field]
        
        # 如果有更新数据
        if update_data:
            # 更新MongoDB中的用户档案
            result = UserProfile.update(profile['_id'], update_data)
            
            # 同时更新Django用户模型中的相关字段
            django_user = User.objects.get(id=user_id)
            if 'first_name' in update_data:
                django_user.first_name = update_data['first_name']
            if 'last_name' in update_data:
                django_user.last_name = update_data['last_name']
            if 'email' in update_data:
                django_user.email = update_data['email']
            django_user.save()
            
            # 获取更新后的档案
            updated_profile = UserProfile.find_by_user_id(str(user_id))
            
            return Response({
                "status": "success",
                "message": "用户档案已更新",
                "profile": updated_profile
            })
        else:
            return Response({
                "status": "success",
                "message": "没有提供需要更新的字段",
                "profile": profile
            })
        
    except Exception as e:
        return Response(
            {"error": f"更新用户档案时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# 用户健康数据概览
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_health_overview(request):
    """获取用户健康数据概览"""
    try:
        user_id = str(request.user.id)
        
        # 获取最近一周的数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        # 获取心率数据
        heart_rate_data = WearableData.find_by_user(
            user_id=user_id,
            data_type="heart_rate",
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # 获取睡眠数据
        sleep_data = WearableData.find_by_user(
            user_id=user_id,
            data_type="sleep",
            start_time=start_time,
            end_time=end_time,
            limit=7  # 一周7天的睡眠数据
        )
        
        # 获取活动数据
        activity_data = WearableData.find_by_user(
            user_id=user_id,
            data_type="activity",
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # 获取未读警报
        alerts = HeartRateAlert.get_user_alerts(user_id, unread_only=True, limit=10)
        
        # 获取活动推荐
        recommendations = ActivityRecommendation.get_user_recommendations(user_id, recent_only=True, limit=5)
        
        # 处理心率数据统计
        heart_rate_stats = calculate_heart_rate_stats(heart_rate_data) if heart_rate_data else None
        
        # 处理睡眠数据统计
        sleep_stats = calculate_sleep_stats(sleep_data) if sleep_data else None
        
        # 处理活动数据统计
        activity_stats = calculate_activity_stats(activity_data) if activity_data else None
        
        return Response({
            "status": "success",
            "heart_rate": heart_rate_stats,
            "sleep": sleep_stats,
            "activity": activity_stats,
            "alerts": {
                "count": len(alerts),
                "items": alerts[:5]  # 只返回前5个警报
            },
            "recommendations": {
                "count": len(recommendations),
                "items": recommendations
            }
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取健康数据概览时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# 辅助函数
def calculate_heart_rate_stats(heart_rate_data):
    """计算心率数据统计"""
    if not heart_rate_data:
        return None
    
    # 提取心率值
    heart_rates = []
    for item in heart_rate_data:
        if isinstance(item.get('values'), (int, float)):
            heart_rates.append(item['values'])
        elif isinstance(item.get('values'), list):
            heart_rates.extend([hr for hr in item['values'] if isinstance(hr, (int, float))])
    
    if not heart_rates:
        return None
    
    # 计算统计值
    return {
        "count": len(heart_rates),
        "average": sum(heart_rates) / len(heart_rates),
        "min": min(heart_rates),
        "max": max(heart_rates),
        "latest": heart_rates[0] if heart_rate_data else None
    }

def calculate_sleep_stats(sleep_data):
    """计算睡眠数据统计"""
    if not sleep_data:
        return None
    
    total_sleep_mins = []
    deep_sleep_mins = []
    sleep_quality_scores = []
    
    for item in sleep_data:
        if isinstance(item.get('values'), dict):
            if 'total_sleep_mins' in item['values']:
                total_sleep_mins.append(item['values']['total_sleep_mins'])
            if 'deep_sleep_mins' in item['values']:
                deep_sleep_mins.append(item['values']['deep_sleep_mins'])
            if 'sleep_quality_score' in item['values']:
                sleep_quality_scores.append(item['values']['sleep_quality_score'])
    
    stats = {}
    
    if total_sleep_mins:
        avg_total_sleep = sum(total_sleep_mins) / len(total_sleep_mins)
        stats["average_sleep_duration"] = {
            "minutes": avg_total_sleep,
            "hours": round(avg_total_sleep / 60, 1)
        }
    
    if deep_sleep_mins and total_sleep_mins:
        avg_deep_sleep = sum(deep_sleep_mins) / len(deep_sleep_mins)
        stats["average_deep_sleep"] = {
            "minutes": avg_deep_sleep,
            "hours": round(avg_deep_sleep / 60, 1),
            "percentage": round((avg_deep_sleep / (sum(total_sleep_mins) / len(total_sleep_mins))) * 100, 1) if avg_deep_sleep > 0 else 0
        }
    
    if sleep_quality_scores:
        stats["average_sleep_quality"] = sum(sleep_quality_scores) / len(sleep_quality_scores)
    
    return stats

def calculate_activity_stats(activity_data):
    """计算活动数据统计"""
    if not activity_data:
        return None
    
    steps_data = []
    calories_data = []
    active_minutes_data = []
    
    for item in activity_data:
        if isinstance(item.get('values'), dict):
            if 'steps' in item['values']:
                steps_data.append(item['values']['steps'])
            if 'calories' in item['values']:
                calories_data.append(item['values']['calories'])
            if 'active_minutes' in item['values']:
                active_minutes_data.append(item['values']['active_minutes'])
    
    stats = {}
    
    if steps_data:
        stats["total_steps"] = sum(steps_data)
        stats["average_daily_steps"] = sum(steps_data) / 7  # 假设是一周的数据
    
    if calories_data:
        stats["total_calories"] = sum(calories_data)
        stats["average_daily_calories"] = sum(calories_data) / 7
    
    if active_minutes_data:
        stats["total_active_minutes"] = sum(active_minutes_data)
        stats["average_daily_active_minutes"] = sum(active_minutes_data) / 7
    
    return stats
