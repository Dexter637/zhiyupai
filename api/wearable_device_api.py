# 可穿戴设备数据接入API
from django.http import JsonResponse
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MongoDB模型
from models import WearableData as WearableDataModel

# 辅助函数
def save_wearable_data(data):
    """保存可穿戴设备数据到MongoDB"""
    # 确保timestamp是datetime对象
    if isinstance(data['timestamp'], str):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
    
    # 使用MongoDB模型保存数据
    doc_id = WearableDataModel.create(data)
    return doc_id

def get_wearable_data(user_id=None, device_id=None, data_type=None, start_time=None, end_time=None, limit=100):
    """从MongoDB检索可穿戴设备数据"""
    query = {}
    
    if user_id:
        query["user_id"] = user_id
    
    if device_id:
        query["device_id"] = device_id
    
    if data_type:
        query["data_type"] = data_type
    
    if start_time or end_time:
        query["timestamp"] = {}
        
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            query["timestamp"]["$gte"] = start_time
        
        if end_time:
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            query["timestamp"]["$lte"] = end_time
    
    # 使用MongoDB模型查询数据
    return WearableDataModel.find(
        query=query,
        sort=[("timestamp", -1)],
        limit=limit
    )

# API视图
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
        
        # 创建数据对象
        wearable_data = WearableData(
            user_id=data['user_id'],
            device_id=data['device_id'],
            data_type=data['data_type'],
            timestamp=data['timestamp'],
            values=data['values'],
            metadata=data.get('metadata', {})
        )
        
        # 数据预处理和验证
        validation_result = validate_wearable_data(data)
        
        # 保存数据到MongoDB
        doc_id = save_wearable_data(data)
        
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
def get_user_wearable_data(request, user_id):
    """获取用户的可穿戴设备数据"""
    try:
        # 获取查询参数
        device_id = request.query_params.get('device_id')
        data_type = request.query_params.get('data_type')
        start_time = request.query_params.get('start_time')
        end_time = request.query_params.get('end_time')
        limit = int(request.query_params.get('limit', 100))
        
        # 检索数据
        results = get_wearable_data(
            user_id=user_id,
            device_id=device_id,
            data_type=data_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # MongoDB结果已经是字典列表
        data_list = results
        
        return Response({
            "status": "success",
            "count": len(data_list),
            "data": data_list
        })
        
    except Exception as e:
        return Response(
            {"error": f"获取数据时出错: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_data_summary(request, user_id, data_type):
    """获取用户特定类型数据的摘要统计"""
    try:
        # 获取查询参数
        start_time = request.query_params.get('start_time')
        end_time = request.query_params.get('end_time')
        limit = int(request.query_params.get('limit', 1000))
        
        # 检索数据
        results = get_wearable_data(
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

# 数据验证函数
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

# 示例路由配置 (在实际Django项目中应放在urls.py)
"""
from django.urls import path
from . import views

urlpatterns = [
    path('api/wearable/upload/', views.wearable_data_upload, name='wearable_data_upload'),
    path('api/wearable/user/<str:user_id>/', views.get_user_wearable_data, name='get_user_wearable_data'),
    path('api/wearable/summary/<str:user_id>/<str:data_type>/', views.get_data_summary, name='get_data_summary'),
]
"""

# 示例使用
def generate_sample_data():
    """生成示例可穿戴设备数据"""
    np.random.seed(42)
    
    user_ids = ["user001", "user002"]
    device_ids = ["fitbit_123", "apple_watch_456", "mi_band_789"]
    data_types = ["heart_rate", "sleep", "activity"]
    
    now = datetime.now()
    
    sample_data = []
    
    # 生成心率数据
    for user_id in user_ids:
        for i in range(50):
            timestamp = now - timedelta(hours=i)
            heart_rate = np.random.normal(75, 10)  # 均值75，标准差10的正态分布
            
            data = {
                "user_id": user_id,
                "device_id": np.random.choice(device_ids),
                "data_type": "heart_rate",
                "timestamp": timestamp,
                "values": heart_rate,
                "metadata": {"measurement_method": "PPG", "confidence": "high"}
            }
            sample_data.append(data)
    
    # 生成睡眠数据
    for user_id in user_ids:
        for i in range(7):  # 一周的睡眠数据
            timestamp = now - timedelta(days=i, hours=8)  # 假设早上8点记录昨晚的睡眠
            
            total_sleep = np.random.normal(420, 60)  # 平均7小时 (420分钟)
            deep_sleep = total_sleep * np.random.uniform(0.15, 0.3)  # 深睡眠占15-30%
            rem_sleep = total_sleep * np.random.uniform(0.2, 0.25)  # REM睡眠占20-25%
            light_sleep = total_sleep - deep_sleep - rem_sleep
            
            data = {
                "user_id": user_id,
                "device_id": np.random.choice(device_ids),
                "data_type": "sleep",
                "timestamp": timestamp,
                "values": {
                    "total_sleep_mins": total_sleep,
                    "deep_sleep_mins": deep_sleep,
                    "rem_sleep_mins": rem_sleep,
                    "light_sleep_mins": light_sleep,
                    "sleep_efficiency": np.random.uniform(0.75, 0.95),
                    "interruptions": np.random.randint(0, 5)
                },
                "metadata": {"sleep_tracking_mode": "auto"}
            }
            sample_data.append(data)
    
    # 生成活动数据
    for user_id in user_ids:
        for i in range(14):  # 两周的活动数据
            timestamp = now - timedelta(days=i)
            
            steps = np.random.randint(3000, 15000)
            distance = steps * np.random.uniform(0.0006, 0.0008)  # 每步约0.6-0.8米
            calories = steps * np.random.uniform(0.03, 0.05)  # 每步约0.03-0.05卡路里
            
            data = {
                "user_id": user_id,
                "device_id": np.random.choice(device_ids),
                "data_type": "activity",
                "timestamp": timestamp,
                "values": {
                    "steps": steps,
                    "distance": distance,
                    "calories": calories,
                    "active_mins": np.random.randint(30, 180),
                    "floors": np.random.randint(0, 20)
                },
                "metadata": {"activity_tracking_mode": "auto"}
            }
            sample_data.append(data)
    
    return sample_data


if __name__ == "__main__":
    # 生成并保存示例数据
    sample_data = generate_sample_data()
    for data in sample_data:
        save_wearable_data(data)
    
    print(f"已生成 {len(sample_data)} 条示例数据")
    
    # 测试数据检索
    user_id = "user001"
    heart_rate_data = get_wearable_data(user_id=user_id, data_type="heart_rate")
    sleep_data = get_wearable_data(user_id=user_id, data_type="sleep")
    activity_data = get_wearable_data(user_id=user_id, data_type="activity")
    
    print(f"用户 {user_id} 的数据统计:")
    print(f"心率数据: {len(heart_rate_data)} 条记录")
    print(f"睡眠数据: {len(sleep_data)} 条记录")
    print(f"活动数据: {len(activity_data)} 条记录")
    
    # 打印第一条心率数据作为示例
    if heart_rate_data:
        print("\n示例心率数据:")
        print(heart_rate_data[0])