#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为admin用户添加5天的睡眠数据，其中包含两条熬夜数据
"""

from datetime import datetime, timedelta
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入MongoDB模型和配置
from models import WearableData
from config import MONGO_URI, DATABASE_NAME, COLLECTIONS

# 确保中文正常显示
import matplotlib
matplotlib.use('Agg')

# 生成admin用户的睡眠数据
def add_sleep_data_for_admin():
    """
    为admin用户添加5天的睡眠数据，其中包含两条熬夜数据
    """
    print("开始为admin用户添加睡眠数据...")
    
    user_id = "admin"
    device_id = "fitbit_123"
    now = datetime.now()
    
    # 第一条数据：正常睡眠（22:30睡，6:30起，8小时）
    record_time1 = now - timedelta(days=4)
    total_sleep1 = 480  # 8小时 = 480分钟
    deep_sleep1 = total_sleep1 * 0.25  # 深睡眠占25%
    rem_sleep1 = total_sleep1 * 0.22  # REM睡眠占22%
    light_sleep1 = total_sleep1 - deep_sleep1 - rem_sleep1
    
    # 计算睡眠开始和结束时间
    sleep_end_time1 = record_time1.replace(hour=6, minute=30, second=0, microsecond=0)
    sleep_start_time1 = sleep_end_time1 - timedelta(minutes=total_sleep1)
    
    data1 = {
        "user_id": user_id,
        "device_id": device_id,
        "data_type": "sleep",
        "timestamp": record_time1,
        "sleep_start_time": sleep_start_time1,
        "sleep_end_time": sleep_end_time1,
        "values": {
            "total_sleep_mins": total_sleep1,
            "deep_sleep_mins": deep_sleep1,
            "rem_sleep_mins": rem_sleep1,
            "light_sleep_mins": light_sleep1,
            "sleep_efficiency": 0.92,
            "interruptions": 1
        },
        "metadata": {"sleep_tracking_mode": "auto"}
    }
    
    # 第二条数据：熬夜1（凌晨1:30睡，6:30起，5小时）
    record_time2 = now - timedelta(days=3)
    total_sleep2 = 300  # 5小时 = 300分钟
    deep_sleep2 = total_sleep2 * 0.20  # 深睡眠占20%
    rem_sleep2 = total_sleep2 * 0.18  # REM睡眠占18%
    light_sleep2 = total_sleep2 - deep_sleep2 - rem_sleep2
    
    # 计算睡眠开始和结束时间
    sleep_end_time2 = record_time2.replace(hour=6, minute=30, second=0, microsecond=0)
    sleep_start_time2 = sleep_end_time2 - timedelta(minutes=total_sleep2)
    
    data2 = {
        "user_id": user_id,
        "device_id": device_id,
        "data_type": "sleep",
        "timestamp": record_time2,
        "sleep_start_time": sleep_start_time2,
        "sleep_end_time": sleep_end_time2,
        "values": {
            "total_sleep_mins": total_sleep2,
            "deep_sleep_mins": deep_sleep2,
            "rem_sleep_mins": rem_sleep2,
            "light_sleep_mins": light_sleep2,
            "sleep_efficiency": 0.78,  # 睡眠效率较低
            "interruptions": 3
        },
        "metadata": {"sleep_tracking_mode": "auto"}
    }
    
    # 第三条数据：正常睡眠（23:00睡，7:00起，8小时）
    record_time3 = now - timedelta(days=2)
    total_sleep3 = 480  # 8小时 = 480分钟
    deep_sleep3 = total_sleep3 * 0.28  # 深睡眠占28%
    rem_sleep3 = total_sleep3 * 0.24  # REM睡眠占24%
    light_sleep3 = total_sleep3 - deep_sleep3 - rem_sleep3
    
    # 计算睡眠开始和结束时间
    sleep_end_time3 = record_time3.replace(hour=7, minute=0, second=0, microsecond=0)
    sleep_start_time3 = sleep_end_time3 - timedelta(minutes=total_sleep3)
    
    data3 = {
        "user_id": user_id,
        "device_id": device_id,
        "data_type": "sleep",
        "timestamp": record_time3,
        "sleep_start_time": sleep_start_time3,
        "sleep_end_time": sleep_end_time3,
        "values": {
            "total_sleep_mins": total_sleep3,
            "deep_sleep_mins": deep_sleep3,
            "rem_sleep_mins": rem_sleep3,
            "light_sleep_mins": light_sleep3,
            "sleep_efficiency": 0.95,
            "interruptions": 0
        },
        "metadata": {"sleep_tracking_mode": "auto"}
    }
    
    # 第四条数据：熬夜2（凌晨2:00睡，7:30起，5.5小时）
    record_time4 = now - timedelta(days=1)
    total_sleep4 = 330  # 5.5小时 = 330分钟
    deep_sleep4 = total_sleep4 * 0.19  # 深睡眠占19%
    rem_sleep4 = total_sleep4 * 0.20  # REM睡眠占20%
    light_sleep4 = total_sleep4 - deep_sleep4 - rem_sleep4
    
    # 计算睡眠开始和结束时间
    sleep_end_time4 = record_time4.replace(hour=7, minute=30, second=0, microsecond=0)
    sleep_start_time4 = sleep_end_time4 - timedelta(minutes=total_sleep4)
    
    data4 = {
        "user_id": user_id,
        "device_id": device_id,
        "data_type": "sleep",
        "timestamp": record_time4,
        "sleep_start_time": sleep_start_time4,
        "sleep_end_time": sleep_end_time4,
        "values": {
            "total_sleep_mins": total_sleep4,
            "deep_sleep_mins": deep_sleep4,
            "rem_sleep_mins": rem_sleep4,
            "light_sleep_mins": light_sleep4,
            "sleep_efficiency": 0.75,  # 睡眠效率较低
            "interruptions": 4
        },
        "metadata": {"sleep_tracking_mode": "auto"}
    }
    
    # 第五条数据：正常睡眠（22:00睡，6:00起，8小时）
    record_time5 = now
    total_sleep5 = 480  # 8小时 = 480分钟
    deep_sleep5 = total_sleep5 * 0.26  # 深睡眠占26%
    rem_sleep5 = total_sleep5 * 0.23  # REM睡眠占23%
    light_sleep5 = total_sleep5 - deep_sleep5 - rem_sleep5
    
    # 计算睡眠开始和结束时间
    sleep_end_time5 = record_time5.replace(hour=6, minute=0, second=0, microsecond=0)
    sleep_start_time5 = sleep_end_time5 - timedelta(minutes=total_sleep5)
    
    data5 = {
        "user_id": user_id,
        "device_id": device_id,
        "data_type": "sleep",
        "timestamp": record_time5,
        "sleep_start_time": sleep_start_time5,
        "sleep_end_time": sleep_end_time5,
        "values": {
            "total_sleep_mins": total_sleep5,
            "deep_sleep_mins": deep_sleep5,
            "rem_sleep_mins": rem_sleep5,
            "light_sleep_mins": light_sleep5,
            "sleep_efficiency": 0.93,
            "interruptions": 2
        },
        "metadata": {"sleep_tracking_mode": "auto"}
    }
    
    # 保存所有数据到数据库
    all_data = [data1, data2, data3, data4, data5]
    
    # 先清除admin用户的现有睡眠数据
    WearableData.get_collection().delete_many({"user_id": user_id, "data_type": "sleep"})
    print(f"已清除admin用户的现有睡眠数据")
    
    # 插入新数据
    for i, data in enumerate(all_data, 1):
        try:
            result = WearableData.create(data)
            print(f"第{i}条睡眠数据已成功添加，ID: {result}")
        except Exception as e:
            print(f"添加第{i}条睡眠数据时出错: {str(e)}")
    
    # 验证数据是否正确添加
    count = WearableData.get_collection().count_documents({"user_id": user_id, "data_type": "sleep"})
    print(f"\n验证结果：admin用户当前有 {count} 条睡眠数据")
    
    # 显示添加的熬夜数据信息
    late_night_sleeps = WearableData.find({
        "user_id": user_id,
        "data_type": "sleep",
        "sleep_start_time": {"$gte": now - timedelta(days=4, hours=1)}
    })
    
    if late_night_sleeps:
        print(f"\n成功添加了 {len(late_night_sleeps)} 条熬夜数据：")
        for i, sleep in enumerate(late_night_sleeps, 1):
            start_time = sleep["sleep_start_time"].strftime("%Y-%m-%d %H:%M")
            end_time = sleep["sleep_end_time"].strftime("%Y-%m-%d %H:%M")
            duration = sleep["values"]["total_sleep_mins"] / 60
            print(f"  {i}. 开始时间: {start_time}, 结束时间: {end_time}, 时长: {duration:.1f}小时")
    else:
        print("\n未找到熬夜数据")

if __name__ == "__main__":
    add_sleep_data_for_admin()
    print("\n睡眠数据添加完成！")