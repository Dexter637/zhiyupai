from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from django.views.decorators.csrf import csrf_exempt

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from bson import ObjectId

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MongoDB模型
from models import WearableData, HeartRateAlert, ActivityRecommendation, UserProfile

# JSON序列化辅助函数
def serialize_mongo_data(data):
    """将MongoDB数据转换为可JSON序列化的格式"""
    if isinstance(data, list):
        return [serialize_mongo_data(item) for item in data]
    elif isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, (list, dict)):
                result[key] = serialize_mongo_data(value)
            else:
                result[key] = value
        return result
    else:
        return data

# 导入健康监测和推荐模块
from health_monitoring.heart_rate_monitor import HeartRateMonitor
from recommendation.sleep_activity_model import SleepActivityRecommender

# 心率数据API
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def heart_rate_list(request):
    """获取所有用户的心率数据列表"""
    try:
        # 获取查询参数
        limit = int(request.query_params.get('limit', 100))
        
        # 检索数据
        results = WearableData.find(
            query={"data_type": "heart_rate"},
            sort=[("timestamp", -1)],
            limit=limit
        )
        
        return Response({
            "status": "success",
            "count": len(results),
            "data": serialize_mongo_data(results)
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取心率数据时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def heart_rate_detail(request, user_id):
    """获取特定用户的心率数据"""
    try:
        # 确保user_id是字符串格式
        user_id = str(user_id)
        
        # 检查权限（只能查看自己的数据，除非是管理员）
        # 确保比较的是相同类型的用户ID（都转换为字符串）
        if str(request.user.id) != str(user_id) and not request.user.is_staff:
            return Response(
                {"error": "您没有权限查看此用户数据"},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # 获取查询参数
        start_time = request.query_params.get('start_time')
        end_time = request.query_params.get('end_time')
        limit = int(request.query_params.get('limit', 100))
        
        # 检索数据
        results = WearableData.find_by_user(
            user_id=user_id,
            data_type="heart_rate",
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return Response({
            "status": "success",
            "count": len(results),
            "data": serialize_mongo_data(results)
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取用户心率数据时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def heart_rate_alerts(request, user_id):
    """获取用户的心率异常警报"""
    try:
        # 确保user_id是字符串格式
        user_id = str(user_id)
        
        # 获取查询参数
        unread_only = request.query_params.get('unread_only', 'false').lower() == 'true'
        limit = int(request.query_params.get('limit', 20))
        
        # 检索警报
        alerts = HeartRateAlert.get_user_alerts(user_id, unread_only, limit)
        
        return Response({
            "status": "success",
            "count": len(alerts),
            "alerts": alerts
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取心率警报时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# 睡眠数据API
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def sleep_data_list(request):
    """获取所有用户的睡眠数据列表"""
    try:
        # 获取查询参数
        limit = int(request.query_params.get('limit', 100))
        
        # 检索数据
        results = WearableData.find(
            query={"data_type": "sleep"},
            sort=[("timestamp", -1)],
            limit=limit
        )
        
        return Response({
            "status": "success",
            "count": len(results),
            "data": serialize_mongo_data(results)
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取睡眠数据时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def sleep_data_detail(request, user_id):
    """获取特定用户的睡眠数据"""
    try:
        # 确保user_id是字符串格式
        user_id = str(user_id)
        
        # 检查权限（只能查看自己的数据，除非是管理员）
        # 确保比较的是相同类型的用户ID（都转换为字符串）
        if str(request.user.id) != str(user_id) and not request.user.is_staff:
            return Response(
                {"error": "您没有权限查看此用户数据"},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # 获取查询参数
        start_time = request.query_params.get('start_time')
        end_time = request.query_params.get('end_time')
        limit = int(request.query_params.get('limit', 100))
        
        # 检索数据
        results = WearableData.find_by_user(
            user_id=user_id,
            data_type="sleep",
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return Response({
            "status": "success",
            "count": len(results),
            "data": serialize_mongo_data(results)
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取用户睡眠数据时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# 活动推荐API
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def activity_recommendations(request, user_id):
    """获取用户的活动推荐"""
    try:
        # 确保user_id是字符串格式
        user_id = str(user_id)
        
        # 获取查询参数
        recent_only = request.query_params.get('recent_only', 'true').lower() == 'true'
        limit = int(request.query_params.get('limit', 10))
        
        # 检索推荐
        recommendations = ActivityRecommendation.get_user_recommendations(user_id, recent_only, limit)
        
        return Response({
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取活动推荐时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def complete_recommendation(request, recommendation_id):
    """标记活动推荐为已完成"""
    try:
        # 更新推荐状态
        result = ActivityRecommendation.update(
            recommendation_id,
            {"is_completed": True, "completed_at": datetime.now()}
        )
        
        if result.modified_count > 0:
            return Response({
                "status": "success",
                "message": "活动推荐已标记为完成"
            })
        else:
            return Response(
                {"error": "未找到指定的活动推荐或已经标记为完成"},
                status=status.HTTP_404_NOT_FOUND
            )
        
    except Exception as e:
        return Response(
            {"error": f"更新活动推荐状态时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# 可穿戴设备数据API
@api_view(['POST'])
@csrf_exempt
def wearable_data_upload(request):
    """接收来自可穿戴设备的数据上传"""
    try:
        data = JSONParser().parse(request)
        
        # 基本验证
        required_fields = ['user_id', 'device_id', 'data_type', 'timestamp', 'values']
        for field in required_fields:
            if field not in data:
                return Response(
                    {"error": f"缺少必填字段: {field}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # 处理时间戳
        if isinstance(data['timestamp'], str):
            try:
                data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                return Response(
                    {"error": "时间戳格式无效，请使用ISO 8601格式"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # 数据验证
        validation_result = validate_wearable_data(data)
        
        # 保存数据到MongoDB
        doc_id = WearableData.create(data)
        
        # 处理特定类型的数据
        if data['data_type'] == 'heart_rate':
            # 检测心率异常
            process_heart_rate_data(data)
        elif data['data_type'] == 'sleep':
            # 基于睡眠数据生成活动推荐
            process_sleep_data(data)
        
        if not validation_result['valid']:
            # 仍然保存数据，但返回验证警告
            return Response({
                "status": "success_with_warnings",
                "message": "数据已保存，但存在验证问题",
                "warnings": validation_result['issues'],
                "data_id": str(doc_id)
            }, status=status.HTTP_201_CREATED)
        
        return Response({
            "status": "success",
            "message": "数据上传成功",
            "data_id": str(doc_id)
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response(
            {"error": f"处理请求时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_wearable_data(request, user_id):
    """获取用户的可穿戴设备数据"""
    try:
        # 确保user_id是字符串格式
        user_id = str(user_id)
        
        # 检查权限（只能查看自己的数据，除非是管理员）
        # 确保比较的是相同类型的用户ID（都转换为字符串）
        if str(request.user.id) != str(user_id) and not request.user.is_staff:
            return Response(
                {"error": "您没有权限查看此用户数据"},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # 获取查询参数
        device_id = request.query_params.get('device_id')
        data_type = request.query_params.get('data_type')
        start_time = request.query_params.get('start_time')
        end_time = request.query_params.get('end_time')
        limit = int(request.query_params.get('limit', 100))
        
        # 检索数据
        results = WearableData.find_by_user(
            user_id=user_id,
            data_type=data_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return Response({
            "status": "success",
            "count": len(results),
            "data": serialize_mongo_data(results)
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取数据时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_data_summary(request, user_id, data_type):
    """获取用户特定类型数据的摘要统计"""
    try:
        # 确保user_id是字符串格式
        user_id = str(user_id)
        
        # 获取查询参数
        start_time = request.query_params.get('start_time')
        end_time = request.query_params.get('end_time')
        limit = int(request.query_params.get('limit', 1000))
        
        # 检索数据
        results = WearableData.find_by_user(
            user_id=user_id,
            data_type=data_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        if not results:
            return Response({
                "status": "success",
                "message": "未找到数据",
                "summary": None
            })
        
        # 提取数值数据
        values = []
        for item in results:
            if isinstance(item.get('values'), dict):
                # 处理复杂数据结构
                for k, v in item['values'].items():
                    if isinstance(v, (int, float)):
                        values.append(v)
            elif isinstance(item.get('values'), (int, float)):
                values.append(item['values'])
            elif isinstance(item.get('values'), list):
                values.extend([v for v in item['values'] if isinstance(v, (int, float))])
        
        if not values:
            return Response({
                "status": "success",
                "message": "未找到数值数据",
                "summary": None
            })
        
        # 计算统计摘要
        summary = {
            "count": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "min": np.min(values),
            "max": np.max(values),
            "std": np.std(values),
            "first_quartile": np.percentile(values, 25),
            "third_quartile": np.percentile(values, 75)
        }
        
        return Response({
            "status": "success",
            "data_type": data_type,
            "summary": summary
        })
        
    except Exception as e:
        return Response(
            {"error": f"计算摘要时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# 辅助函数
def validate_wearable_data(data):
    """验证可穿戴设备数据的有效性"""
    issues = []
    
    # 根据数据类型进行特定验证
    if data['data_type'] == 'heart_rate':
        # 心率数据验证
        if isinstance(data['values'], (int, float)):
            hr = data['values']
            if hr < 30 or hr > 220:
                issues.append(f"心率值 {hr} 超出正常范围 (30-220 BPM)")
        elif isinstance(data['values'], list):
            for hr in data['values']:
                if isinstance(hr, (int, float)) and (hr < 30 or hr > 220):
                    issues.append(f"心率值 {hr} 超出正常范围 (30-220 BPM)")
    
    elif data['data_type'] == 'sleep':
        # 睡眠数据验证
        if isinstance(data['values'], dict):
            if 'total_sleep_mins' in data['values']:
                total_sleep = data['values']['total_sleep_mins']
                if total_sleep < 0 or total_sleep > 1440:  # 24小时 = 1440分钟
                    issues.append(f"总睡眠时间 {total_sleep} 分钟超出合理范围 (0-1440)")
            
            if 'deep_sleep_mins' in data['values'] and 'total_sleep_mins' in data['values']:
                deep_sleep = data['values']['deep_sleep_mins']
                total_sleep = data['values']['total_sleep_mins']
                if deep_sleep > total_sleep:
                    issues.append(f"深度睡眠时间 ({deep_sleep} 分钟) 超过总睡眠时间 ({total_sleep} 分钟)")
    
    elif data['data_type'] == 'activity':
        # 活动数据验证
        if isinstance(data['values'], dict):
            if 'steps' in data['values']:
                steps = data['values']['steps']
                if steps < 0 or steps > 100000:  # 一天10万步是合理上限
                    issues.append(f"步数 {steps} 超出合理范围 (0-100000)")
            
            if 'calories' in data['values']:
                calories = data['values']['calories']
                if calories < 0 or calories > 10000:  # 一天10000卡路里是合理上限
                    issues.append(f"卡路里 {calories} 超出合理范围 (0-10000)")
    
    # 时间戳验证
    timestamp = data['timestamp']
    if isinstance(timestamp, datetime):
        now = datetime.now()
        if timestamp > now + timedelta(minutes=5):  # 允许5分钟的时钟偏差
            issues.append(f"时间戳 {timestamp.isoformat()} 在未来")
        
        if timestamp < now - timedelta(days=30):  # 数据不应该太旧
            issues.append(f"时间戳 {timestamp.isoformat()} 超过30天")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }

def process_heart_rate_data(data):
    """处理心率数据，检测异常并生成警报"""
    try:
        user_id = data['user_id']
        heart_rate = data['values']
        timestamp = data['timestamp']
        
        # 获取用户最近的心率数据
        recent_data = WearableData.find_by_user(
            user_id=user_id,
            data_type="heart_rate",
            start_time=timestamp - timedelta(days=7),
            end_time=timestamp,
            limit=1000
        )
        
        # 如果数据足够，进行异常检测
        if len(recent_data) > 10:
            # 准备数据
            df = pd.DataFrame(recent_data)
            df['heart_rate'] = df['values'].apply(lambda x: x if isinstance(x, (int, float)) else np.mean(x) if isinstance(x, list) else None)
            df = df.dropna(subset=['heart_rate'])
            
            # 初始化心率监测器
            monitor = HeartRateMonitor()
            
            # 检测异常
            monitor.fit(df)
            alerts = monitor.detect_anomalies(df, save_to_db=True)
            
            return alerts
        
        return None
    except Exception as e:
        print(f"处理心率数据时出错: {e}")
        return None

def process_sleep_data(data):
    """处理睡眠数据，生成活动推荐"""
    try:
        user_id = data['user_id']
        sleep_data = data['values']
        timestamp = data['timestamp']
        
        # 获取用户档案
        user_profile = UserProfile.find_by_user_id(user_id)
        
        if user_profile and isinstance(sleep_data, dict):
            # 初始化活动推荐器
            recommender = SleepActivityRecommender()
            
            # 生成推荐
            recommendation = recommender.recommend_activity(
                sleep_data=sleep_data,
                user_id=user_id,
                user_profile=user_profile,
                save_to_db=True
            )
            
            return recommendation
        
        return None
    except Exception as e:
        print(f"处理睡眠数据时出错: {e}")
        return None
