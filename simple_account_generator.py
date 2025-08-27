#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版测试账号生成器
使用方法: python manage.py shell -c "import simple_account_generator"
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
import pymongo
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.core.exceptions import ObjectDoesNotExist

# 导入MongoDB配置（根据实际项目配置修改）
from config import MONGO_URI, DATABASE_NAME, COLLECTIONS

class SimpleTestAccountGenerator:
    """简化版测试账号生成器"""
    
    def __init__(self):
        # 生成测试账号的配置
        self.num_users = 5
        self.test_prefix = "testuser"
        
        # 创建MongoDB客户端和集合引用
        self.client = pymongo.MongoClient(MONGO_URI)
        self.db = self.client[DATABASE_NAME]
        self.wearable_data_collection = self.db[COLLECTIONS['wearable_data']]
        self.user_profiles_collection = self.db[COLLECTIONS['user_profiles']]
        
        # 健康数据范围配置
        self.health_data_config = {
            "heart_rate": {
                "min": 60, 
                "max": 100,
                "mean": 75,
                "std": 10,
                "count": 50
            },
            "sleep": {
                "min_total": 420,  # 7小时，分钟
                "max_total": 540,  # 9小时，分钟
                "deep_sleep_ratio": (0.15, 0.3),
                "rem_sleep_ratio": (0.2, 0.25),
                "count": 7  # 一周的数据
            },
            "blood_glucose": {
                "min": 3.9, 
                "max": 6.1,
                "mean": 5.0,
                "std": 0.5,
                "count": 10
            },
            "blood_pressure": {
                "systolic_min": 110, 
                "systolic_max": 130,
                "diastolic_min": 70, 
                "diastolic_max": 90,
                "count": 10
            }
        }
    
    def generate_phone_number(self):
        """生成随机手机号"""
        prefix = "138"
        suffix = ''.join(random.choices('0123456789', k=8))
        return prefix + suffix
    
    def generate_username(self, index):
        """生成用户名"""
        return f"{self.test_prefix}{index:03d}"
    
    def generate_password(self):
        """生成随机密码"""
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789."
        return ''.join(random.choices(chars, k=8))
    
    def register_user(self, username, password, phone):
        """注册新用户"""
        try:
            # 检查用户是否已存在
            User.objects.get(username=username)
            print(f"用户 {username} 已存在，跳过注册")
            return None
        except ObjectDoesNotExist:
            pass
        
        try:
            # 创建Django用户
            user = User.objects.create(
                username=username,
                password=make_password(password)
            )
            
            # 创建MongoDB用户档案
            profile_data = {
                "user_id": str(user.id),
                "username": username,
                "phone": phone,
                "email": None,
                "date_joined": datetime.now(),
                "is_active": True,
                "age": random.randint(20, 60),
                "gender": random.choice(["male", "female"]),
                "height": round(random.uniform(150, 190), 1),
                "weight": round(random.uniform(45, 95), 1),
                "fitness_level": random.choice(["low", "medium", "high"])
            }
            
            # 保存用户档案到MongoDB
            result = self.user_profiles_collection.insert_one(profile_data)
            
            print(f"用户 {username} 注册成功，用户ID: {user.id}")
            
            return {
                "user": user,
                "profile_id": str(result.inserted_id),
                "username": username,
                "password": password,
                "phone": phone,
                "user_id": str(user.id)
            }
            
        except Exception as e:
            print(f"注册用户 {username} 时出错: {str(e)}")
            # 如果发生异常，尝试删除已创建的用户
            try:
                User.objects.get(username=username).delete()
            except:
                pass
            return None
    
    def generate_heart_rate_data(self, user_id, count):
        """生成随机心率数据"""
        config = self.health_data_config["heart_rate"]
        data_points = []
        now = datetime.now()
        
        for i in range(count):
            timestamp = now - timedelta(hours=i)
            # 使用正态分布生成心率值
            value = max(config["min"], min(config["max"], round(np.random.normal(config["mean"], config["std"]))))
            
            data = {
                "user_id": user_id,
                "data_type": "heart_rate",
                "value": value,
                "unit": "bpm",
                "timestamp": timestamp,
                "device_id": f"device_{random.randint(1000, 9999)}",
                "metadata": {
                    "measurement_method": "PPG",
                    "confidence": random.choice(["high", "medium", "low"])
                }
            }
            data_points.append(data)
        
        return data_points
    
    def generate_sleep_data(self, user_id, count):
        """生成随机睡眠数据"""
        config = self.health_data_config["sleep"]
        data_points = []
        now = datetime.now()
        
        for i in range(count):
            # 假设早上8点记录昨晚的睡眠
            record_time = now - timedelta(days=i, hours=8)
            
            # 生成总睡眠时长
            total_sleep = random.randint(config["min_total"], config["max_total"])
            
            # 计算深睡眠和REM睡眠时长
            deep_sleep_ratio = random.uniform(*config["deep_sleep_ratio"])
            rem_sleep_ratio = random.uniform(*config["rem_sleep_ratio"])
            
            deep_sleep = round(total_sleep * deep_sleep_ratio)
            rem_sleep = round(total_sleep * rem_sleep_ratio)
            light_sleep = total_sleep - deep_sleep - rem_sleep
            
            # 计算睡眠开始和结束时间
            sleep_end_time = record_time
            sleep_start_time = sleep_end_time - timedelta(minutes=total_sleep)
            
            data = {
                "user_id": user_id,
                "data_type": "sleep",
                "timestamp": record_time,
                "device_id": f"device_{random.randint(1000, 9999)}",
                "start_time": sleep_start_time,
                "end_time": sleep_end_time,
                "duration": total_sleep,  # 分钟
                "values": {
                    "total_sleep_mins": total_sleep,
                    "deep_sleep_mins": deep_sleep,
                    "rem_sleep_mins": rem_sleep,
                    "light_sleep_mins": light_sleep,
                    "sleep_efficiency": round(random.uniform(0.75, 0.95), 2),
                    "awakenings": random.randint(0, 5)
                },
                "metadata": {
                    "sleep_quality": random.choice(["good", "fair", "poor"])
                }
            }
            data_points.append(data)
        
        return data_points
    
    def generate_blood_glucose_data(self, user_id, count):
        """生成随机血糖数据"""
        config = self.health_data_config["blood_glucose"]
        data_points = []
        now = datetime.now()
        
        for i in range(count):
            timestamp = now - timedelta(hours=random.randint(1, 24 * 7))
            # 使用正态分布生成血糖值
            value = max(config["min"], min(config["max"], round(np.random.normal(config["mean"], config["std"]), 1)))
            
            data = {
                "user_id": user_id,
                "data_type": "blood_glucose",
                "value": value,
                "unit": "mmol/L",
                "timestamp": timestamp,
                "device_id": f"device_{random.randint(1000, 9999)}",
                "metadata": {
                    "measurement_type": random.choice(["fasting", "postprandial", "random"])
                }
            }
            data_points.append(data)
        
        return data_points
    
    def generate_blood_pressure_data(self, user_id, count):
        """生成随机血压数据"""
        config = self.health_data_config["blood_pressure"]
        data_points = []
        now = datetime.now()
        
        for i in range(count):
            timestamp = now - timedelta(hours=random.randint(1, 24 * 7))
            
            # 生成收缩压和舒张压
            systolic = random.randint(config["systolic_min"], config["systolic_max"])
            diastolic = random.randint(config["diastolic_min"], config["diastolic_max"])
            
            data = {
                "user_id": user_id,
                "data_type": "blood_pressure",
                "systolic": systolic,
                "diastolic": diastolic,
                "unit": "mmHg",
                "timestamp": timestamp,
                "device_id": f"device_{random.randint(1000, 9999)}",
                "metadata": {
                    "measurement_position": random.choice(["left_arm", "right_arm"])
                }
            }
            data_points.append(data)
        
        return data_points
    
    def generate_health_data(self, user_id):
        """为用户生成所有类型的健康数据"""
        all_data = []
        
        # 生成心率数据
        heart_rate_data = self.generate_heart_rate_data(
            user_id, 
            self.health_data_config["heart_rate"]["count"]
        )
        all_data.extend(heart_rate_data)
        
        # 生成睡眠数据
        sleep_data = self.generate_sleep_data(
            user_id, 
            self.health_data_config["sleep"]["count"]
        )
        all_data.extend(sleep_data)
        
        # 生成血糖数据
        blood_glucose_data = self.generate_blood_glucose_data(
            user_id, 
            self.health_data_config["blood_glucose"]["count"]
        )
        all_data.extend(blood_glucose_data)
        
        # 生成血压数据
        blood_pressure_data = self.generate_blood_pressure_data(
            user_id, 
            self.health_data_config["blood_pressure"]["count"]
        )
        all_data.extend(blood_pressure_data)
        
        return all_data
    
    def save_health_data(self, health_data):
        """保存健康数据到MongoDB"""
        try:
            if health_data:
                result = self.wearable_data_collection.insert_many(health_data)
                print(f"已保存 {len(result.inserted_ids)} 条健康数据")
                return True
            return False
        except Exception as e:
            print(f"保存健康数据时出错: {str(e)}")
            return False
    
    def run(self):
        """运行生成器"""
        generated_accounts = []
        
        print(f"开始生成 {self.num_users} 个测试账号...")
        
        for i in range(1, self.num_users + 1):
            username = self.generate_username(i)
            password = self.generate_password()
            phone = self.generate_phone_number()
            
            # 注册用户
            user_info = self.register_user(username, password, phone)
            
            if user_info:
                # 为用户生成健康数据
                print(f"为用户 {username} 生成健康数据...")
                health_data = self.generate_health_data(user_info["user_id"])
                
                # 保存健康数据
                self.save_health_data(health_data)
                
                # 保存用户信息，不包含密码之外的敏感信息
                generated_accounts.append({
                    "username": username,
                    "password": password,
                    "phone": phone
                })
        
        print("\n===== 生成的测试账号 =====")
        for account in generated_accounts:
            print(f"用户名: {account['username']}")
            print(f"密码: {account['password']}")
            print(f"手机号: {account['phone']}")
            print("---")
        
        # 保存账号信息到文件，方便用户查看
        with open("generated_test_accounts.txt", "w", encoding="utf-8") as f:
            f.write("===== 生成的测试账号 =====\n")
            for account in generated_accounts:
                f.write(f"用户名: {account['username']}\n")
                f.write(f"密码: {account['password']}\n")
                f.write(f"手机号: {account['phone']}\n")
                f.write("---\n")
        
        print(f"测试账号信息已保存到 generated_test_accounts.txt 文件")
        print("\n完成! 现在您可以使用这些账号登录系统并测试数据隔离功能。")

# 当作为模块导入时运行
generator = SimpleTestAccountGenerator()
generator.run()