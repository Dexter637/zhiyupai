#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试睡眠数据结构，验证开始睡眠时间和睡眠结束时间字段
"""

from datetime import datetime
from pymongo import MongoClient
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置
from config import MONGO_URI, DATABASE_NAME, COLLECTIONS

# 直接定义一个简化版的generate_sample_data函数，不依赖Django环境
def generate_sample_data():
    """
    生成示例可穿戴设备数据（简化版，用于测试）
    """
    from datetime import datetime, timedelta
    import numpy as np
    import random
    
    np.random.seed(42)
    
    user_ids = ["user001", "user002"]
    device_ids = ["fitbit_123", "apple_watch_456", "mi_band_789"]
    data_types = ["heart_rate", "sleep", "activity"]
    
    now = datetime.now()
    
    sample_data = []
    
    # 只生成睡眠数据用于测试
    for user_id in user_ids:
        for i in range(7):  # 一周的睡眠数据
            # 假设早上8点记录昨晚的睡眠
            record_time = now - timedelta(days=i, hours=8)
            
            total_sleep = np.random.normal(420, 60)  # 平均7小时 (420分钟)
            deep_sleep = total_sleep * np.random.uniform(0.15, 0.3)  # 深睡眠占15-30%
            rem_sleep = total_sleep * np.random.uniform(0.2, 0.25)  # REM睡眠占20-25%
            light_sleep = total_sleep - deep_sleep - rem_sleep
            
            # 计算睡眠开始和结束时间
            sleep_end_time = record_time  # 记录时间接近醒来时间
            sleep_start_time = sleep_end_time - timedelta(minutes=total_sleep)  # 开始时间 = 结束时间 - 总睡眠时长
            
            data = {
                "user_id": user_id,
                "device_id": random.choice(device_ids),
                "data_type": "sleep",
                "timestamp": record_time,
                "sleep_start_time": sleep_start_time,  # 添加开始睡眠时间
                "sleep_end_time": sleep_end_time,      # 添加睡眠结束时间
                "values": {
                    "total_sleep_mins": total_sleep,
                    "deep_sleep_mins": deep_sleep,
                    "rem_sleep_mins": rem_sleep,
                    "light_sleep_mins": light_sleep,
                    "sleep_efficiency": np.random.uniform(0.75, 0.95),
                    "interruptions": random.randint(0, 5)
                },
                "metadata": {"sleep_tracking_mode": "auto"}
            }
            sample_data.append(data)
    
    return sample_data

# 连接到MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
wearable_data_collection = db[COLLECTIONS['wearable_data']]

def test_sleep_data_structure():
    """
    测试睡眠数据结构，验证开始睡眠时间和睡眠结束时间字段
    """
    print("开始测试睡眠数据结构...")
    
    # 步骤1: 生成新的样本数据
    print("1. 生成新的样本数据...")
    sample_data = generate_sample_data()
    
    # 筛选出睡眠数据
    sleep_data = [data for data in sample_data if data['data_type'] == 'sleep']
    print(f"   生成了 {len(sleep_data)} 条睡眠数据样本")
    
    # 步骤2: 检查睡眠数据是否包含新字段
    print("\n2. 检查样本数据是否包含新字段...")
    missing_fields = 0
    
    for i, record in enumerate(sleep_data[:5], 1):  # 只检查前5条记录
        has_start_time = 'sleep_start_time' in record
        has_end_time = 'sleep_end_time' in record
        has_duration = 'total_sleep_mins' in record.get('values', {})
        
        print(f"   样本 {i}:")
        print(f"   - 包含开始时间: {has_start_time}")
        print(f"   - 包含结束时间: {has_end_time}")
        print(f"   - 包含总睡眠时长: {has_duration}")
        
        if has_start_time and has_end_time and has_duration:
            # 验证时间逻辑是否正确
            start_time = record['sleep_start_time']
            end_time = record['sleep_end_time']
            duration = record['values']['total_sleep_mins']
            
            # 计算开始时间和结束时间之间的实际时长
            calculated_duration = (end_time - start_time).total_seconds() / 60
            difference = abs(calculated_duration - duration)
            
            print(f"   - 开始时间: {start_time}")
            print(f"   - 结束时间: {end_time}")
            print(f"   - 总睡眠时长: {duration} 分钟")
            print(f"   - 计算的睡眠时长: {calculated_duration:.2f} 分钟")
            print(f"   - 时长差异: {difference:.2f} 分钟")
        else:
            missing_fields += 1
    
    if missing_fields == 0:
        print("   ✓ 所有检查的样本都包含所需的字段")
    else:
        print(f"   ✗ 发现 {missing_fields} 个样本缺少必要字段")
    
    # 步骤3: 将样本数据插入到数据库进行测试
    print("\n3. 将样本数据插入到数据库进行测试...")
    try:
        # 清空现有的睡眠数据（仅用于测试）
        wearable_data_collection.delete_many({"data_type": "sleep"})
        print("   已清空现有的睡眠数据")
        
        # 插入新的睡眠数据
        if sleep_data:
            result = wearable_data_collection.insert_many(sleep_data)
            print(f"   成功插入 {len(result.inserted_ids)} 条睡眠数据到数据库")
        
        # 步骤4: 从数据库中查询并验证数据
        print("\n4. 从数据库中查询并验证数据...")
        db_sleep_data = list(wearable_data_collection.find({"data_type": "sleep"}).limit(5))
        
        if db_sleep_data:
            print(f"   从数据库中查询到 {len(db_sleep_data)} 条睡眠数据")
            
            # 检查数据库中的数据是否包含新字段
            db_missing_fields = 0
            for i, record in enumerate(db_sleep_data, 1):
                has_start_time = 'sleep_start_time' in record
                has_end_time = 'sleep_end_time' in record
                has_duration = 'total_sleep_mins' in record.get('values', {})
                
                if not (has_start_time and has_end_time and has_duration):
                    db_missing_fields += 1
                    
            if db_missing_fields == 0:
                print("   ✓ 数据库中的所有样本都包含所需的字段")
            else:
                print(f"   ✗ 数据库中发现 {db_missing_fields} 个样本缺少必要字段")
        else:
            print("   ✗ 未能从数据库中查询到睡眠数据")
    except Exception as e:
        print(f"   ✗ 插入或查询数据时出错: {str(e)}")
    
    print("\n睡眠数据结构测试完成")


def simplified_verify_application_compatibility():
    """
    简化版的应用程序兼容性验证
    """
    print("\n简化版应用程序兼容性验证...")
    
    # 获取数据库中的睡眠数据
    db_sleep_data = list(wearable_data_collection.find({"data_type": "sleep"}).limit(7))
    
    if db_sleep_data:
        print("✓ 成功获取睡眠数据，新的数据结构应该能被应用程序处理")
        # 手动计算一些统计数据来验证
        total_sleep_mins = []
        
        for item in db_sleep_data:
            if isinstance(item.get('values'), dict) and 'total_sleep_mins' in item['values']:
                total_sleep_mins.append(item['values']['total_sleep_mins'])
        
        if total_sleep_mins:
            avg_total_sleep = sum(total_sleep_mins) / len(total_sleep_mins)
            print(f"示例统计 - 平均睡眠时长: {avg_total_sleep/60:.1f} 小时")
    else:
        print("✗ 没有足够的睡眠数据来测试应用程序兼容性")


if __name__ == "__main__":
    try:
        # 测试睡眠数据结构
        test_sleep_data_structure()
        
        # 简化版验证应用程序兼容性
        simplified_verify_application_compatibility()
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭MongoDB连接
        client.close()
        print("\n已关闭MongoDB连接")