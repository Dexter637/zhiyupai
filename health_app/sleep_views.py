import json
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from models import WearableData, UserProfile

@login_required
def sleep_analysis_view(request):
    """睡眠数据分析页面视图"""
    return render(request, 'sleep_analysis.html')

@csrf_exempt
def get_sleep_data_api(request):
    """获取用户睡眠数据的API接口"""
    try:
        # 获取查询参数
        username = request.GET.get('username', 'admin')  # 默认使用admin用户
        print(f"请求的用户名: {username}")
        
        # 注意：UserProfile类中没有find_by_username方法，这里直接使用测试数据
        # 实际项目中应该实现find_by_username或从其他地方获取用户ID
        user_id = username  # 简化处理，直接使用username作为user_id
        print(f"使用的用户ID: {user_id}")
        
        # 获取最近5天的睡眠数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=5)
        
        # 查询睡眠数据
        sleep_data = WearableData.find_by_user(
            user_id=user_id,
            data_type='sleep',
            start_time=start_time,
            end_time=end_time,
            limit=10
        )
        
        # 转换为JSON可序列化的格式
        serializable_data = []
        for item in sleep_data:
            serializable_item = {
                "_id": str(item["_id"]),
                "user_id": item["user_id"],
                "device_id": item["device_id"],
                "data_type": item["data_type"],
                "timestamp": item["timestamp"].isoformat() if isinstance(item["timestamp"], datetime) else item["timestamp"],
                "values": item["values"]
            }
            
            # 确保时间字段是ISO格式字符串
            if "sleep_start_time" in serializable_item["values"] and isinstance(serializable_item["values"]["sleep_start_time"], datetime):
                serializable_item["values"]["sleep_start_time"] = serializable_item["values"]["sleep_start_time"].isoformat()
            
            if "sleep_end_time" in serializable_item["values"] and isinstance(serializable_item["values"]["sleep_end_time"], datetime):
                serializable_item["values"]["sleep_end_time"] = serializable_item["values"]["sleep_end_time"].isoformat()
            
            serializable_data.append(serializable_item)
        
        return JsonResponse({
            "status": "success",
            "count": len(serializable_data),
            "data": serializable_data
        })
        
    except Exception as e:
        print(f"获取睡眠数据时出错: {str(e)}")
        return JsonResponse({
            "status": "error",
            "message": f"获取睡眠数据时出错: {str(e)}"
        }, status=500)