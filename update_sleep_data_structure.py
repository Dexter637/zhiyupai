#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
更新睡眠数据结构，添加开始睡眠时间和睡眠结束时间字段
"""

from datetime import datetime, timedelta
from pymongo import MongoClient
import random

# 导入配置
from config import MONGO_URI, DATABASE_NAME, COLLECTIONS

# 连接到MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
wearable_data_collection = db[COLLECTIONS['wearable_data']]

def update_sleep_data_structure():
    """
    更新所有睡眠数据记录，添加开始睡眠时间和睡眠结束时间字段
    """
    print("开始更新睡眠数据结构...")
    
    # 统计记录数量
    total_count = wearable_data_collection.count_documents({"data_type": "sleep"})
    print(f"找到 {total_count} 条睡眠数据记录需要更新")
    
    # 分批处理数据，避免内存问题
    batch_size = 100
    updated_count = 0
    
    # 遍历所有睡眠数据记录
    cursor = wearable_data_collection.find({"data_type": "sleep"}).batch_size(batch_size)
    
    for record in cursor:
        try:
            # 检查记录是否已经包含开始和结束时间
            if "sleep_start_time" in record and "sleep_end_time" in record:
                print(f"记录 {record['_id']} 已经包含开始和结束时间，跳过更新")
                continue
            
            # 获取当前的timestamp和总睡眠时长
            record_time = record.get("timestamp")
            values = record.get("values", {})
            total_sleep_mins = values.get("total_sleep_mins", 420)  # 默认7小时
            
            # 计算睡眠开始和结束时间
            # 假设记录时间是早上8点左右，代表昨晚的睡眠
            if record_time:
                # 计算睡眠结束时间（假设记录时间接近醒来时间）
                sleep_end_time = record_time
                
                # 计算睡眠开始时间（结束时间减去总睡眠时长）
                sleep_start_time = sleep_end_time - timedelta(minutes=total_sleep_mins)
            else:
                # 如果没有记录时间，使用当前时间生成模拟数据
                now = datetime.now()
                # 假设是昨晚的睡眠，随机在20:00-23:00之间开始
                sleep_start_hour = random.randint(20, 23)
                sleep_start_time = now.replace(hour=sleep_start_hour, minute=random.randint(0, 59), second=0, microsecond=0)
                
                # 计算睡眠结束时间
                sleep_end_time = sleep_start_time + timedelta(minutes=total_sleep_mins)
            
            # 更新记录
            result = wearable_data_collection.update_one(
                {"_id": record["_id"]},  # 更新条件
                {"$set": {
                    "sleep_start_time": sleep_start_time,  # 添加开始睡眠时间
                    "sleep_end_time": sleep_end_time,      # 添加睡眠结束时间
                    "updated_at": datetime.now()           # 更新修改时间
                }}
            )
            
            if result.modified_count == 1:
                updated_count += 1
                if updated_count % 100 == 0:
                    print(f"已更新 {updated_count}/{total_count} 条记录")
            
        except Exception as e:
            print(f"更新记录 {record.get('_id', '未知ID')} 时出错: {str(e)}")
    
    print(f"睡眠数据结构更新完成。成功更新 {updated_count} 条记录，跳过 {total_count - updated_count} 条记录")


def update_data_generation_functions():
    """
    更新数据生成函数，确保新生成的数据包含开始和结束时间
    """
    # 这个函数可以根据实际需求实现，用于更新项目中的数据生成相关代码
    print("提醒：请确保项目中的数据生成函数也被更新，以包含开始和结束时间字段")


def verify_update():
    """
    验证更新是否成功
    """
    print("\n验证更新结果...")
    
    # 随机查询几条记录进行验证
    sample_records = wearable_data_collection.find({"data_type": "sleep"}).limit(5)
    
    for i, record in enumerate(sample_records, 1):
        print(f"\n样本 {i}:")
        print(f"ID: {record['_id']}")
        print(f"记录时间: {record.get('timestamp')}")
        print(f"开始睡眠时间: {record.get('sleep_start_time')}")
        print(f"睡眠结束时间: {record.get('sleep_end_time')}")
        print(f"总睡眠时长: {record.get('values', {}).get('total_sleep_mins')} 分钟")
        
        # 验证开始时间和结束时间的逻辑关系
        if record.get('sleep_start_time') and record.get('sleep_end_time'):
            calculated_duration = (record['sleep_end_time'] - record['sleep_start_time']).total_seconds() / 60
            actual_duration = record.get('values', {}).get('total_sleep_mins', 0)
            difference = abs(calculated_duration - actual_duration)
            print(f"计算的睡眠时长: {calculated_duration:.2f} 分钟")
            print(f"时长差异: {difference:.2f} 分钟")


if __name__ == "__main__":
    try:
        # 更新睡眠数据结构
        update_sleep_data_structure()
        
        # 验证更新
        verify_update()
        
        # 提醒更新数据生成函数
        update_data_generation_functions()
        
    except Exception as e:
        print(f"更新过程中发生错误: {str(e)}")
    finally:
        # 关闭MongoDB连接
        client.close()
        print("\n已关闭MongoDB连接")